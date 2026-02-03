import gc
import math
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import numpy as np
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from tqdm import trange
# 新增
import copy
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings


from anndata import AnnData


from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
import scvi
import scanpy as sc

import numpy as np
# import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, eval_scib_metrics, load_pretrained

try:
    from flash_attn.flash_attention import FlashMHA

    flash_attn_available = True
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    flash_attn_available = False

from dsbn import DomainSpecificBatchNorm1d
from grad_reverse import grad_reverse

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss

class MyTransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        nlayers_cls: int = 3,
        n_cls: int = 1,
        vocab: Any = None,
        dropout: float = 0.5,
        pad_token: str = "<pad>",
        pad_value: int = 0,
        do_mvc: bool = False,
        do_dab: bool = False,
        use_batch_labels: bool = False,
        num_batch_labels: Optional[int] = None,
        domain_spec_batchnorm: Union[bool, str] = False,
        input_emb_style: str = "continuous",
        n_input_bins: Optional[int] = None,
        cell_emb_style: str = "cls",
        mvc_decoder_style: str = "inner product",
        ecs_threshold: float = 0.3,
        explicit_zero_prob: bool = False,
        use_fast_transformer: bool = False,
        fast_transformer_backend: str = "flash",
        pre_norm: bool = False,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        self.do_dab = do_dab
        self.ecs_threshold = ecs_threshold
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.input_emb_style = input_emb_style
        self.cell_emb_style = cell_emb_style
        self.explicit_zero_prob = explicit_zero_prob
        self.norm_scheme = "pre" if pre_norm else "post"
        if self.input_emb_style not in ["category", "continuous", "scaling"]:
            raise ValueError(
                f"input_emb_style should be one of category, continuous, scaling, "
                f"got {input_emb_style}"
            )
        if cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {cell_emb_style}")
        if use_fast_transformer:
            if not flash_attn_available:
                warnings.warn(
                    "flash-attn is not installed, using pytorch transformer instead. "
                    "Set use_fast_transformer=False to avoid this warning. "
                    "Installing flash-attn is highly recommended."
                )
                use_fast_transformer = False
        self.use_fast_transformer = use_fast_transformer

        # TODO: add dropout in the GeneEncoder
        self.encoder = GeneEncoder(ntoken, d_model, padding_idx=vocab[pad_token])

        # Value Encoder, NOTE: the scaling style is also handled in _encode method
        if input_emb_style == "continuous":
            self.value_encoder = ContinuousValueEncoder(d_model, dropout)
        elif input_emb_style == "category":
            assert n_input_bins > 0
            self.value_encoder = CategoryValueEncoder(
                n_input_bins, d_model, padding_idx=pad_value
            )
        else:
            self.value_encoder = nn.Identity()  # nn.Softmax(dim=1)
            # TODO: consider row-wise normalization or softmax
            # TODO: Correct handle the mask_value when using scaling

        # Batch Encoder
        if use_batch_labels:
            self.batch_encoder = BatchLabelEncoder(num_batch_labels, d_model)

        if domain_spec_batchnorm is True or domain_spec_batchnorm == "dsbn":
            use_affine = True if domain_spec_batchnorm == "do_affine" else False
            print(f"Use domain specific batchnorm with affine={use_affine}")
            self.dsbn = DomainSpecificBatchNorm1d(
                d_model, num_batch_labels, eps=6.1e-5, affine=use_affine
            )
        elif domain_spec_batchnorm == "batchnorm":
            print("Using simple batchnorm instead of domain specific batchnorm")
            self.bn = nn.BatchNorm1d(d_model, eps=6.1e-5)

        if use_fast_transformer:
            if fast_transformer_backend == "linear":
                self.transformer_encoder = FastTransformerEncoderWrapper(
                    d_model, nhead, d_hid, nlayers, dropout
                )
            elif fast_transformer_backend == "flash":
                encoder_layers = FlashTransformerEncoderLayer(
                    d_model,
                    nhead,
                    d_hid,
                    dropout,
                    batch_first=True,
                    norm_scheme=self.norm_scheme,
                )
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        else:
            encoder_layers = TransformerEncoderLayer(
                d_model, nhead, d_hid, dropout, batch_first=True
            )
            self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.decoder = ExprDecoder(
            d_model,
            explicit_zero_prob=explicit_zero_prob,
            use_batch_labels=use_batch_labels,
        )
        self.cls_decoder = ClsDecoder(d_model, n_cls, nlayers=nlayers_cls)
        if do_mvc:
            self.mvc_decoder = MVCDecoder(
                d_model,
                arch_style=mvc_decoder_style,
                explicit_zero_prob=explicit_zero_prob,
                use_batch_labels=use_batch_labels,
            )

        if do_dab:
            self.grad_reverse_discriminator = AdversarialDiscriminator(
                d_model,
                n_cls=num_batch_labels,
                reverse_grad=True,
            )

        self.sim = Similarity(temp=0.5)  # TODO: auto set temp
        self.creterion_cce = nn.CrossEntropyLoss()

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        self._check_batch_labels(batch_labels)

        src = self.encoder(src)  # (batch, seq_len, embsize)    
        self.cur_gene_token_embs = src           # gene token embeddings

        values = self.value_encoder(values)  # (batch, seq_len, embsize) expression values embeddings
        if self.input_emb_style == "scaling":
            values = values.unsqueeze(2)
            total_embs = src * values
        else:
            total_embs = src + values

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def _check_batch_labels(self, batch_labels: Tensor) -> None:
        if self.use_batch_labels or self.domain_spec_batchnorm:
            assert batch_labels is not None
        elif batch_labels is not None:
            raise ValueError(
                "batch_labels should only be provided when `self.use_batch_labels`"
                " or `self.domain_spec_batchnorm` is True"
            )

    def generate(
        self,
        cell_emb: Tensor,
        src: Tensor,
        values: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        gen_iters: int = 1,
        batch_labels: Optional[Tensor] = None,  # (batch,)
    ) -> Tensor:
        """
        Args:
            cell_emb(:obj:`Tensor`): shape (batch, embsize)
            src(:obj:`Tensor`): shape (batch, seq_len)
            values(:obj:`Tensor`): shape (batch, seq_len), optional
            src_key_padding_mask(:obj:`Tensor`): shape (batch, seq_len), optional
            gen_iters(:obj:`int`): number of generation iterations
            batch_labels(:obj:`Tensor`): shape (batch,), optional
        """
        # TODO: should have a tag indicate the generation mode
        # TODO: if gen_iters > 1, should have a tag indicate the current iteration
        try:
            self._check_batch_labels(batch_labels)
        except:
            import warnings

            warnings.warn(
                "batch_labels is required but not provided, using zeros instead"
            )
            batch_labels = torch.zeros(
                cell_emb.shape[0], dtype=torch.long, device=cell_emb.device
            )

        src = self.encoder(src)  # (batch, seq_len, embsize)

        if values is not None:
            values = self.value_encoder(values)  # (batch, seq_len, embsize)
            if self.input_emb_style == "scaling":
                values = values.unsqueeze(2)
                total_embs = src * values
            else:
                total_embs = src + values
        else:
            total_embs = src

        if getattr(self, "dsbn", None) is not None:
            batch_label = int(batch_labels[0].item())
            total_embs = self.dsbn(total_embs.permute(0, 2, 1), batch_label).permute(
                0, 2, 1
            )  # the batch norm always works on dim 1
        elif getattr(self, "bn", None) is not None:
            total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        total_embs[:, 0, :] = cell_emb

        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                total_embs.shape[:2], dtype=torch.bool, device=total_embs.device
            )
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        output = mlm_output["pred"]  # (batch, seq_len)

        return output  # (batch, seq_len)

    def forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_labels: Optional[Tensor] = None,
        CLS: bool = False,
        CCE: bool = False,
        MVC: bool = False,
        ECS: bool = False,
        do_sample: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            batch_labels (:obj:`Tensor`): batch labels, shape [batch_size]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            CCE (:obj:`bool`): if True, return the contrastive cell embedding objective
                (CCE) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output
            ECS (:obj:`bool`): if True, return the elastic cell similarity objective
                (ECS) output.

        Returns:
            dict of output Tensors.
        """
        transformer_output = self._encode(
            src, values, src_key_padding_mask, batch_labels
        )

        if self.use_batch_labels:
            batch_emb = self.batch_encoder(batch_labels)  # (batch, embsize)

        output = {}
        mlm_output = self.decoder(
            transformer_output
            if not self.use_batch_labels
            else torch.cat(
                [
                    transformer_output,
                    batch_emb.unsqueeze(1).repeat(1, transformer_output.shape[1], 1),
                ],
                dim=2,
            ),
            # else transformer_output + batch_emb.unsqueeze(1),
        )
        if self.explicit_zero_prob and do_sample:
            bernoulli = Bernoulli(probs=mlm_output["zero_probs"])
            output["mlm_output"] = bernoulli.sample() * mlm_output["pred"]
        else:
            output["mlm_output"] = mlm_output["pred"]              # (batch, seq_len)
        if self.explicit_zero_prob:
            output["mlm_zero_probs"] = mlm_output["zero_probs"]

        cell_emb = self._get_cell_emb_from_layer(transformer_output, values)
        output["cell_emb"] = cell_emb

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if CCE:
            cell1 = cell_emb
            transformer_output2 = self._encode(
                src, values, src_key_padding_mask, batch_labels
            )
            cell2 = self._get_cell_emb_from_layer(transformer_output2)

            # Gather embeddings from all devices if distributed training
            if dist.is_initialized() and self.training:
                cls1_list = [
                    torch.zeros_like(cell1) for _ in range(dist.get_world_size())
                ]
                cls2_list = [
                    torch.zeros_like(cell2) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(tensor_list=cls1_list, tensor=cell1.contiguous())
                dist.all_gather(tensor_list=cls2_list, tensor=cell2.contiguous())

                # NOTE: all_gather results have no gradients, so replace the item
                # of the current rank with the original tensor to keep gradients.
                # See https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L186
                cls1_list[dist.get_rank()] = cell1
                cls2_list[dist.get_rank()] = cell2

                cell1 = torch.cat(cls1_list, dim=0)
                cell2 = torch.cat(cls2_list, dim=0)
            # TODO: should detach the second run cls2? Can have a try
            cos_sim = self.sim(cell1.unsqueeze(1), cell2.unsqueeze(0))  # (batch, batch)
            labels = torch.arange(cos_sim.size(0)).long().to(cell1.device)
            output["loss_cce"] = self.creterion_cce(cos_sim, labels)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb
                if not self.use_batch_labels
                else torch.cat([cell_emb, batch_emb], dim=1),
                # else cell_emb + batch_emb,
                self.cur_gene_token_embs,
            )
            if self.explicit_zero_prob and do_sample:
                bernoulli = Bernoulli(probs=mvc_output["zero_probs"])
                output["mvc_output"] = bernoulli.sample() * mvc_output["pred"]
            else:
                output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
            if self.explicit_zero_prob:
                output["mvc_zero_probs"] = mvc_output["zero_probs"]
        if ECS:
            # Here using customized cosine similarity instead of F.cosine_similarity
            # to avoid the pytorch issue of similarity larger than 1.0, pytorch # 78064
            # normalize the embedding
            cell_emb_normed = F.normalize(cell_emb, p=2, dim=1)
            cos_sim = torch.mm(cell_emb_normed, cell_emb_normed.t())  # (batch, batch)

            # mask out diagnal elements
            mask = torch.eye(cos_sim.size(0)).bool().to(cos_sim.device)
            cos_sim = cos_sim.masked_fill(mask, 0.0)
            # only optimize positive similarities
            cos_sim = F.relu(cos_sim)

            output["loss_ecs"] = torch.mean(1 - (cos_sim - self.ecs_threshold) ** 2)

        if self.do_dab:
            output["dab_output"] = self.grad_reverse_discriminator(cell_emb)

        return output

    def encode_batch(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        batch_size: int,
        batch_labels: Optional[Tensor] = None,
        output_to_cpu: bool = True,
        time_step: Optional[int] = None,
        return_np: bool = False,
    ) -> Tensor:
        """
        Args:
            src (Tensor): shape [N, seq_len]
            values (Tensor): shape [N, seq_len]
            src_key_padding_mask (Tensor): shape [N, seq_len]
            batch_size (int): batch size for encoding
            batch_labels (Tensor): shape [N, n_batch_labels]
            output_to_cpu (bool): whether to move the output to cpu
            time_step (int): the time step index in the transformer output to return.
                The time step is along the second dimenstion. If None, return all.
            return_np (bool): whether to return numpy array

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        N = src.size(0)
        device = next(self.parameters()).device

        # initialize the output tensor
        array_func = np.zeros if return_np else torch.zeros
        float32_ = np.float32 if return_np else torch.float32
        shape = (
            (N, self.d_model)
            if time_step is not None
            else (N, src.size(1), self.d_model)
        )
        outputs = array_func(shape, dtype=float32_)

        for i in trange(0, N, batch_size):
            raw_output = self._encode(
                src[i : i + batch_size].to(device),
                values[i : i + batch_size].to(device),
                src_key_padding_mask[i : i + batch_size].to(device),
                batch_labels[i : i + batch_size].to(device)
                if batch_labels is not None
                else None,
            )
            output = raw_output.detach()
            if output_to_cpu:
                output = output.cpu()
            if return_np:
                output = output.numpy()
            if time_step is not None:
                output = output[:, time_step, :]
            outputs[i : i + batch_size] = output

        return outputs


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)


class FastTransformerEncoderWrapper(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.fast_transformer_encoder = self.build_fast_transformer_encoder(
            d_model, nhead, d_hid, nlayers, dropout
        )

    @staticmethod
    def build_fast_transformer_encoder(
        d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float
    ) -> nn.Module:
        from fast_transformers.builders import TransformerEncoderBuilder

        if d_model % nhead != 0:
            raise ValueError(
                f"d_model must be divisible by nhead, "
                f"got d_model={d_model} and nhead={nhead}"
            )
        builder = TransformerEncoderBuilder.from_kwargs(
            n_layers=nlayers,
            n_heads=nhead,
            query_dimensions=d_model // nhead,
            value_dimensions=d_model // nhead,
            feed_forward_dimensions=d_hid,
            attention_type="linear",
            attention_dropout=dropout,
            dropout=dropout,
            activation="gelu",
        )
        assert builder.attention_type == "linear"
        return builder.get()

    @staticmethod
    def build_length_mask(
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> "LengthMask":
        from fast_transformers.masking import LengthMask

        seq_len = src.shape[1]
        num_paddings = src_key_padding_mask.sum(dim=1)
        actual_seq_len = seq_len - num_paddings  # (N,)
        length_mask = LengthMask(actual_seq_len, max_len=seq_len, device=src.device)

        if src_key_padding_mask[length_mask.bool_matrix].sum() != 0:
            raise ValueError(
                "Found padding tokens in the middle of the sequence. "
                "src_key_padding_mask and length_mask are not compatible."
            )
        return length_mask

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: torch.BoolTensor,
    ) -> Tensor:
        """
        Args:
            src: Tensor, shape [N, seq_len, embsize]
            src_key_padding_mask: Tensor, shape [N, seq_len]

        Returns:
            output Tensor of shape [N, seq_len, embsize]
        """
        if src_key_padding_mask.shape != src.shape[:2]:
            raise ValueError(
                f"src_key_padding_mask shape {src_key_padding_mask.shape} "
                f"does not match first two dims of src shape {src.shape[:2]}"
            )

        if src_key_padding_mask.dtype != torch.bool:
            raise ValueError(
                f"src_key_padding_mask needs to be of type torch.bool, "
                f"got {src_key_padding_mask.dtype}"
            )

        length_mask = self.build_length_mask(src, src_key_padding_mask)
        output = self.fast_transformer_encoder(src, length_mask=length_mask)
        return output


class FlashTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        # Version compatibility workaround
        if not hasattr(self.self_attn, "batch_first"):
            self.self_attn.batch_first = batch_first
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        if not src_key_padding_mask.any().item():
            # no padding tokens in src
            src_key_padding_mask_ = None
        else:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            src_key_padding_mask_ = ~src_key_padding_mask

        if self.norm_scheme == "pre":
            src = self.norm1(src)
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_value: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)
        return x


class BatchLabelEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, embsize)
        x = self.enc_norm(x)
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ):
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        self.fc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )
        self.explicit_zero_prob = explicit_zero_prob
        if explicit_zero_prob:
            self.zero_logit = nn.Sequential(
                nn.Linear(d_in, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, d_model),
                nn.LeakyReLU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        pred_value = self.fc(x).squeeze(-1)  # (batch, seq_len)

        if not self.explicit_zero_prob:
            return dict(pred=pred_value)
        zero_logits = self.zero_logit(x).squeeze(-1)  # (batch, seq_len)
        zero_probs = torch.sigmoid(zero_logits)
        return dict(pred=pred_value, zero_probs=zero_probs)
        # TODO: note that the return currently is only for training. Since decoder
        # is not used in the test setting for the integration task, the eval/inference
        # logic is not implemented yet. However, remember to implement it when
        # the decoder is used in any test setting. The inference logic will need
        # to sample from the bernoulli distribution with the zero_probs.


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: nn.Module = nn.Sigmoid,
        hidden_activation: nn.Module = nn.PReLU,
        explicit_zero_prob: bool = False,
        use_batch_labels: bool = False,
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model * 2 if use_batch_labels else d_model
        if arch_style in ["inner product", "inner product, detach"]:
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.W = nn.Linear(d_model, d_in, bias=False)
            if explicit_zero_prob:  # by default, gene-wise prob rate
                self.W_zero_logit = nn.Linear(d_model, d_in)
        elif arch_style == "concat query":
            self.gene2query = nn.Linear(d_model, 64)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model + 64, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        elif arch_style == "sum query":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = query_activation()
            self.fc1 = nn.Linear(d_model, 64)
            self.hidden_activation = hidden_activation()
            self.fc2 = nn.Linear(64, 1)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style
        self.do_detach = arch_style.endswith("detach")
        self.explicit_zero_prob = explicit_zero_prob

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)           torch.Size([64, 1024])    
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model) torch.Size([64, 1201, 512])
        """
        gene_embs = gene_embs.detach() if self.do_detach else gene_embs
        if self.arch_style in ["inner product", "inner product, detach"]:  # 'inner product'
            query_vecs = self.query_activation(self.gene2query(gene_embs)) # 经过线性映射 + softmax激活 torch.Size([64, 1201, 512])
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1) # 64,1201,1
            # the pred gene expr values, # (batch, seq_len)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(2) # （64,1201,1024）*（64,1024,1）-》64, 1201 基因的embeddings和cell_embeddings相乘，每个基因对于每个细胞的重要性
            if not self.explicit_zero_prob:
                return dict(pred=pred_value)
            # zero logits need to based on the cell_emb, because of input exprs
            zero_logits = torch.bmm(self.W_zero_logit(query_vecs), cell_emb).squeeze(2)
            zero_probs = torch.sigmoid(zero_logits)
            return dict(pred=pred_value, zero_probs=zero_probs)
        elif self.arch_style == "concat query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            # expand cell_emb to (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(1).expand(-1, gene_embs.shape[1], -1)

            h = self.hidden_activation(
                self.fc1(torch.cat([cell_emb, query_vecs], dim=2))
            )
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)
        elif self.arch_style == "sum query":
            query_vecs = self.query_activation(self.gene2query(gene_embs))
            cell_emb = cell_emb.unsqueeze(1)

            h = self.hidden_activation(self.fc1(cell_emb + query_vecs))
            if self.explicit_zero_prob:
                raise NotImplementedError
            return self.fc2(h).squeeze(2)  # (batch, seq_len)


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)

########### prepare_data ###########
def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    # print('train_batch_labels', train_batch_labels)
    # print('valid_batch_labels', valid_batch_labels)

    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)     
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

def impt_norm(impt):
        tanh = torch.nn.Tanh()
        for layer in range(impt.size(0)):
            impt[layer] = (impt[layer] - impt[layer].mean()) / impt[
                layer].std()  # 2D, we need to deal with this for each layer
        impt = tanh(impt).abs()

        return impt

def contrastive_loss(ofull, ofull_pos, oprev, tau=0.1):
    """
    Compute contrastive loss given full knowledge (ofull), its positive variants (ofull_pos), and previously learned knowledge (oprev).
    
    Args:
        ofull (Tensor): Full knowledge representations of shape (B, N_SEQ, DIM)
        ofull_pos (Tensor): Positive samples (dropout variants of ofull) of shape (B, N_SEQ, DIM)
        oprev (Tensor): Previously learned knowledge of shape (B, N_SEQ, DIM)
        tau (float): Temperature parameter for contrastive learning

    Returns:
        Tensor: Contrastive loss scalar
    """
    B, N_SEQ, DIM = ofull.shape
    
    # Flatten the batch and sequence dimensions to compute pairwise similarity
    ofull_flat = ofull.view(B * N_SEQ, DIM)  # Shape (B * N_SEQ, DIM)
    ofull_pos_flat = ofull_pos.view(B * N_SEQ, DIM)  # Shape (B * N_SEQ, DIM)
    oprev_flat = oprev.view(B * N_SEQ, DIM)  # Shape (B * N_SEQ, DIM)

        # Initialize the total loss
    total_loss = 0
    batch_size = 8
    # Iterate over the data in batches to compute similarity
    for i in range(0, B * N_SEQ, batch_size):
        # Slice the batch data
        ofull_batch = ofull_flat[i:i+batch_size]
        ofull_pos_batch = ofull_pos_flat[i:i+batch_size]
        oprev_batch = oprev_flat[i:i+batch_size]

        # Compute similarities
        sim_ofull_pos = torch.exp(F.cosine_similarity(ofull_batch, ofull_pos_batch) / tau)
        sim_ofull_all_pos = torch.exp(F.cosine_similarity(ofull_batch.unsqueeze(1), ofull_pos_batch.unsqueeze(0), dim=-1) / tau)
        sim_ofull_oprev = torch.exp(F.cosine_similarity(ofull_batch.unsqueeze(1), oprev_batch.unsqueeze(0), dim=-1) / tau)

        # Compute denominator
        denominator = sim_ofull_all_pos.sum(dim=1) + sim_ofull_oprev.sum(dim=1)

        # Compute batch loss
        loss = -torch.log(sim_ofull_pos / denominator).mean()
        total_loss += loss
    
    return loss

def emax(input_tensor1: torch.Tensor, input_tensor2: torch.Tensor) -> torch.Tensor:
    """
    Compute the EMAX function, which returns the element-wise maximum between two tensors.
    
    Args:
        input_tensor1 (torch.Tensor): The first input tensor.
        input_tensor2 (torch.Tensor): The second input tensor.
    
    Returns:
        torch.Tensor: A tensor with the element-wise maximum values.
    """
    return torch.maximum(input_tensor1, input_tensor2)

def load_pretrained(
    model:torch.nn.Module,
    # old_model: torch.nn.Module,
    pretrained_params: Mapping[str, torch.Tensor],
    strict: bool = False,
    prefix: Optional[List[str]] = None,
    verbose: bool = True,
):
    """
    Load pretrained weights to the model.

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (Mapping[str, torch.Tensor]): The pretrained parameters.
        strict (bool): Whether to strictly enforce that the keys in :attr:`pretrained_params`
            match the keys returned by this module's :meth:`Module.state_dict`. Default to False.
        prefix (List[str]): The list of prefix strings to match with the keys in
            :attr:`pretrained_params`. The matched keys will be loaded. Default to None.

    Returns:
        torch.nn.Module: The model with pretrained weights.
    """
    use_flash_attn = getattr(model, "use_fast_transformer", True)
    if not use_flash_attn:
        pretrained_params = {
            k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()
        }

    if prefix is not None and len(prefix) > 0:
        if isinstance(prefix, str):
            prefix = [prefix]
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if any(k.startswith(p) for p in prefix)
        }

    model_dict = model.state_dict()
    if strict:
        if verbose:
            for k, v in pretrained_params.items():
                logger.info(f"Loading parameter {k} with shape {v.shape}")
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
    else:
        if verbose:
            for k, v in pretrained_params.items():
                if k in model_dict and v.shape == model_dict[k].shape:
                    logger.info(f"Loading parameter {k} with shape {v.shape}")
        pretrained_params = {
            k: v
            for k, v in pretrained_params.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_params)
        model.load_state_dict(model_dict)
        # state_dict = pretrained_params
        # mapping_keys = {
        #     'decoder.fc1.weight': 'decoder.fc.0.weight',
        #     'decoder.fc1.bias': 'decoder.fc.0.bias',
        #     'decoder.fc2.weight': 'decoder.fc.2.weight',
        #     'decoder.fc2.bias': 'decoder.fc.2.bias',
        #     'decoder.fc3.weight': 'decoder.fc.4.weight',
        #     'decoder.fc3.bias': 'decoder.fc.4.bias',
        #     'decoder.fc4.weight': 'decoder.zero_logit.0.weight',
        #     'decoder.fc4.bias': 'decoder.zero_logit.0.bias',
        #     'decoder.fc5.weight': 'decoder.zero_logit.2.weight',
        #     'decoder.fc5.bias': 'decoder.zero_logit.2.bias',
        #     'decoder.fc6.weight': 'decoder.zero_logit.4.weight',
        #     'decoder.fc6.bias': 'decoder.zero_logit.4.bias'
        # }

        # # 创建新的 state_dict
        # new_state_dict = {}

        # # 按照 mapping_keys 映射新的键
        # for new_key, old_key in mapping_keys.items():
        #     if old_key in model_dict:
        #         new_state_dict[new_key] = model_dict[old_key]

        # # 将 model_dict 中未映射的键添加到 new_state_dict 中
        # for key, value in model_dict.items():
        #     if key not in mapping_keys.values():
        #         new_state_dict[key] = value

        # 加载新的 state_dict
        # model.load_state_dict(new_state_dict, strict=True)
    
    return model

def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = "<cls>",
    mod_type: np.ndarray = None,
    cls_id_mod_type: int = None,
) -> List[Tuple[Union[torch.Tensor, np.ndarray]]]:
    """
    Tokenize a batch of data. Returns a list of tuple (gene_id, count).

    Args:
        data (array-like): A batch of data, with shape (batch_size, n_features).
            n_features equals the number of all genes.
        gene_ids (array-like): A batch of gene ids, with shape (n_features,).
        return_pt (bool): Whether to return torch tensors of gene_ids and counts,
            default to True.

    Returns:
        list: A list of tuple (gene_id, count) of non zero gene expressions.
    """
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )
    if mod_type is not None and data.shape[1] != len(mod_type):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of mod_type ({len(mod_type)})."
        )

    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        mod_types = None
        if include_zero_gene:
            values = row
            genes = gene_ids
            if mod_type is not None:
                mod_types = mod_type
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
            if mod_type is not None:
                mod_types = mod_type[idx]
        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)            
            if mod_type is not None:
                mod_types = np.insert(mod_types, 0, cls_id_mod_type)
        if return_pt:
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            if mod_type is not None:
                mod_types = torch.from_numpy(mod_types).long()
        tokenized_data.append((genes, values, mod_types))
    return tokenized_data


def pad_batch(
    batch: List[Tuple],
    max_len: int,
    vocab: Vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Pad a batch of data. Returns a list of Dict[gene_id, count].

    Args:
        batch (list): A list of tuple (gene_id, count).
        max_len (int): The maximum length of the batch.
        vocab (Vocab): The vocabulary containing the pad token.
        pad_token (str): The token to pad with.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of gene_id and count.
    """
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
    gene_ids_list = []
    values_list = []
    mod_types_list = []

    for i in range(len(batch)):
        gene_ids, values, mod_types = batch[i]

        if len(gene_ids) > max_len:
            # sample max_len genes
            if not cls_appended:
                idx = np.random.choice(len(gene_ids), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            gene_ids = gene_ids[idx]
            values = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]
        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
            if mod_types is not None:
                mod_types = torch.cat(
                    [
                        mod_types,
                        torch.full(
                            (max_len - len(mod_types),),
                            mod_pad_id,
                            dtype=mod_types.dtype,
                        ),
                    ]
                )

        gene_ids_list.append(gene_ids)
        values_list.append(values)
        if mod_types is not None:
            mod_types_list.append(mod_types)

    batch_padded = {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }
    if mod_types is not None:
        batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
    return batch_padded


def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: Vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod: Vocab = None,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize and pad a batch of data. Returns a list of tuple (gene_id, count).
    """
    cls_id = vocab[cls_token]
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]
    tokenized_data = tokenize_batch(
        data,
        gene_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type if mod_type is not None else None,
    )

    batch_padded = pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        vocab_mod=vocab_mod,
    )
    return batch_padded

def initial_impt(n_encoder_layer, fc1_size, fc2_size):

    fc1_impt = torch.zeros(n_encoder_layer, fc1_size)   # 1*30
    fc1_mask = torch.ones(n_encoder_layer, fc1_size)
    fc1_mask = fc1_mask.to(device)
    fc1_impt = fc1_impt.to(device)
    fc1_mask.requires_grad_(requires_grad=True)

    fc2_impt = torch.zeros(n_encoder_layer, fc2_size)   # 1*10
    fc2_mask = torch.ones(n_encoder_layer, fc2_size)
    fc2_mask = fc2_mask.to(device)
    fc2_impt = fc2_impt.to(device)
    fc2_mask.requires_grad_(requires_grad=True)

    tot_tokens = 0.0

    return  fc1_impt, fc1_mask, fc2_impt, fc2_mask, tot_tokens

def train(model: nn.Module, loader: DataLoader, epoch: int, sample_size: int) -> None:
    """
    Train the model for one epoch.
    """
    # prepare the ewc regularizer
    
    end_losses = []
    model.train()
    total_loss, total_mse, total_gepc, total_ewc = 0.0, 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config['log_interval']
    start_time = time.time()
    num_batches = len(loader)
    iter = 0
    for batch, batch_data in enumerate(loader):
        # iter += 1
        # if iter >=5:
        #     break
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        # print('input_gene_ids',input_gene_ids)
        # print('input_values',input_values)
        # print('target_values',target_values)
        # print('batch_labels',batch_labels)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])   # 判断矩阵中是否有填充部分，全是False(没有样本的基因是缺失的)
        # print(src_key_padding_mask)
        with torch.cuda.amp.autocast(enabled=config['amp']):
            model.decoder.return_representation = False
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                MVC=config['GEPC'],
                ECS=config['ecs_thres'] > 0,
            )
           
            masked_positions = input_values.eq(mask_value)  # the postions to predict mask_value= -1
            output_dict["mlm_output"] = output_dict["mlm_output"].squeeze()
            output_dict["mlm_zero_probs"] = output_dict["mlm_zero_probs"].squeeze()
            output_dict["mvc_output"] = output_dict["mvc_output"].squeeze()
            output_dict["mvc_zero_probs"] = output_dict["mvc_zero_probs"].squeeze()
            
            loss = loss_mse = criterion(
                output_dict["mlm_output"], target_values, masked_positions
            )       # 只计算mask的部分  B, n_seq
            # masked_positions_ = masked_positions.float()
            # loss_ = F.mse_loss(output_dict["mlm_output"] * masked_positions_, target_values * masked_positions_, reduction="none")
            # loss_ = loss_.sum(dim=1) / masked_positions_.sum(dim=1) 

            metrics_to_log = {"train/mse": loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(            # 非负二项分布计算经过softmax之后的输出与target之间的值
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if config['GEPC']:  # # Gene expression modelling for cell objective
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            if config['GEPC'] and explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )
            if config['ecs_thres'] > 0:  # Elastic cell similarity objective
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            if output_dict["dab_output"].ndimension() != 2:
                    # 在前面加一维 (dim=0)
                output_dict["dab_output"] = output_dict["dab_output"].unsqueeze(0)
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)  # batch_loss
            loss = loss + config['dab_weight'] * loss_dab
            metrics_to_log.update({"train/dab": loss_dab.item()})

            # if epoch == config['epochs'] and len(end_losses) < sample_size // config["batch_size"]:
            #     end_losses.append(loss_)
            #     print('len(end_losses)', len(end_losses))
            ################## 增加EWC loss ###################
            ewc_loss = ewc_regularizer.regularize(model.named_parameters())[0].detach().cpu().to(device)
            loss += ewc_loss                     
        model.zero_grad()
        scaler.scale(loss).backward()

        ############## 梯度反缩放和梯度裁剪 ##############
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                print(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()


        # wandb.log(metrics_to_log)

        with torch.no_grad():
            if output_dict["mlm_output"].ndimension() != 2:
                output_dict["mlm_output"] = output_dict["mlm_output"].unsqueeze(0)
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )   # 计算相对误差：误差除以 target

        total_loss += loss.item()      
        total_mse += loss_mse.item()     # 仅MSE的loss
        total_gepc += loss_gepc.item() if config['GEPC'] else 0.0   # 仅MVC的loss
        total_error += mre.item()        # 计算相对误差：误差除以 target
        total_ewc += ewc_loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config['GEPC'] else 0.0
            cur_error = total_error / log_interval
            cur_ewc = total_ewc / log_interval
            # ppl = math.exp(cur_loss)
            print(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} ewc {cur_ewc:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config['GEPC'] else "")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            start_time = time.time()
    # if epoch == config['epochs']:
    #     return end_losses

def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            with torch.cuda.amp.autocast(enabled=config['amp']):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                )
                output_dict["mlm_output"] = output_dict["mlm_output"].squeeze()
                output_dict["dab_output"] = output_dict["dab_output"].squeeze()
                output_values = output_dict["mlm_output"]

                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                if output_dict["dab_output"].ndimension() != 2:
                    # 在前面加一维 (dim=0)
                    output_dict["dab_output"] = output_dict["dab_output"].unsqueeze(0)
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)

            total_loss += loss.item() * len(input_gene_ids)
            if output_values.ndimension() != 2:
                output_values = output_values.unsqueeze(0)
                total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)

    # wandb.log(
    #     {
    #         "valid/mse": total_loss / total_num,
    #         "valid/mre": total_error / total_num,
    #         "valid/dab": total_dab / total_num,
    #         "valid/sum_mse_dab": (total_loss + config['dab_weight'] * total_dab)
    #         / total_num,
    #         "epoch": epoch,
    #     },
    # )

    return total_loss / total_num, total_error / total_num

def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()

    # copy adata_t to avoid reuse previously computed results stored in adata_t
    adata_t = adata_t.copy()

    all_counts = (
        adata_t.layers[input_layer_key].A
        if issparse(adata_t.layers[input_layer_key])
        else adata_t.layers[input_layer_key]
    )

    celltypes_labels = adata_t.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_t.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    # Evaluate cls cell embeddings
    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=max_seq_len,
            vocab=vocab,
            pad_token=pad_token,
            pad_value=pad_value,
            append_cls=True,             # append <cls> token at the beginning
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config['amp']):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config['batch_size'],
                batch_labels=torch.from_numpy(batch_ids).long() if DSBN else None,
                time_step=0,
                return_np=True,
            )
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scGPT"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            logger.error(e)

        # sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        # sc.tl.umap(adata_t, min_dist=0.3)
        # fig = sc.pl.umap(
        #     adata_t,
        #     color=["str_batch"],
        #     title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
        #     frameon=False,
        #     return_fig=True,
        #     show=False,
        # )

        # results["batch_umap"] = fig

        # sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        # sc.tl.umap(adata_t, min_dist=0.3)
        # fig = sc.pl.umap(
        #     adata_t,
        #     color=["celltype"],
        #     title=[
        #         f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
        #     ],
        #     frameon=False,
        #     return_fig=True,
        #     show=False,
        # )

        # results["celltype_umap"] = fig

    if len(include_types) == 1:
        return results, adata_t

def load_data_and_model(hyperparameter_defaults,):
    dataset_name = hyperparameter_defaults['dataset_name']
    
    if dataset_name==   "PBMC_10K":
        adata = sc.read_h5ad('/workspace/geshuang/code/scGPT/data/pbmc/pbmc_10k.h5ad')
    elif dataset_name == "pancreas":
        adata = sc.read_h5ad('/workspace/geshuang/code/scGPT/data/PANCREAS/pancreas_data.h5ad')
    else:
        assert False, f"Unknown dataset {dataset_name}"
    if hyperparameter_defaults['load_model'] is not None:  # load_model="/workspace/geshuang/code/scGPT/save/scGPT_human"
        model_dir = Path(hyperparameter_defaults['load_model'])
        weight_dir = Path(hyperparameter_defaults['weight_dir'])
        print('model_dir', model_dir)
        model_config_file = model_dir / "args.json"
        model_file = weight_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        print(
            f"Resume model from {model_file}, the model args will be overriden by the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]   # 512
        nhead = model_configs["nheads"]      # 8
        d_hid = model_configs["d_hid"]        # 512
        nlayers = model_configs["nlayers"]    # 12
        n_layers_cls = model_configs["n_layers_cls"]  # 3
    else:
        embsize = hyperparameter_defaults['layer_size'] 
        nhead = hyperparameter_defaults['nhead']
        nlayers = hyperparameter_defaults['nlayers']  
        d_hid = hyperparameter_defaults['layer_size']

    if per_seq_batch_sample:
    # sort the adata by batch_id in advance
        adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()       # batch_id = sample_id

    input_layer_key = "X_binned"
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )              # 11990 × 1200
    genes = adata.var["gene_name"].tolist()
    print(len(genes))     # 1200 HVG genes
    celltypes_labels = adata.obs["celltype"].tolist()  # make sure count from 0
    num_types = len(set(celltypes_labels))
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()         # 0,1两个sample
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

    ############## split the data ##############
    (
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)
    sample_size = celltypes_labels.shape[0]
    ############## set up the vocab ##############
    if hyperparameter_defaults['load_model'] is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
    # print(vocab)
    # print(len(gene_ids))  # max=36627, min=88   总长=1200
    
    ############## tokenize and pad the data ##############
    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )                                               # genes & values
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=True,
    )
    return adata, adata_sorted, embsize, nhead, d_hid, nlayers, vocab, num_batch_types, model_file, logger, save_dir, \
        tokenized_train, tokenized_valid, train_batch_labels, valid_batch_labels, input_layer_key,gene_ids, sample_size

if __name__=='__main__':
    from config_all import param
    from ewc import EWC
    sc.set_figure_params(figsize=(4, 4))
    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings('ignore')

    # dataset_name = "pancreas"
    dataset_name = "PBMC_10K"
    experiment_name = "fine_tune_onpbmc"
    load_model = "/workspace/geshuang/code/scGPT/save/scGPT_human"
    # weight_dir = '/workspace/geshuang/code/scGPT/save/dev_PBMC_10K-Feb26-19-34'     # 修改模型加载路径
    # weight_dir = '/workspace/geshuang/code/scGPT/save/dev_PBMC_10K-Feb28-09-27'
    # weight_dir = '/workspace/geshuang/code/scGPT/tutorials/save/dev_PBMC_10K-Feb19-19-34'
    weight_dir = '/workspace/geshuang/code/scGPT/tutorials/save/dev_pancreas-Feb25-17-11'
    # weight_dir = "/workspace/geshuang/code/scGPT/save/scGPT_human"
    # fisher_path = './fisher/fisher_optpar_(reverse)pancreas.pth'      
    fisher_path = None          
    batch_size = 1       # 在计算fisher information的时候，batch_size=1
    epochs = 1           # 在计算fisher information的时候，epochs=1
    hyperparameter_defaults, config = param(dataset_name, load_model, weight_dir, batch_size, epochs)
    
    set_seed(hyperparameter_defaults['seed'])
    # settings for input and preprocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = 0.4
    mask_value = -1
    pad_value = -2
    n_input_bins = 51
    
    n_hvg = 1200  # number of highly variable genes
    max_seq_len = n_hvg + 1
    per_seq_batch_sample = True
    DSBN = True  # Domain-spec batchnorm
    explicit_zero_prob = True  # whether explicit bernoulli for zeros
    
    ############### datasets loading ###############
    save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(hyperparameter_defaults)
    adata, adata_sorted, embsize, nhead, d_hid, nlayers, vocab, num_batch_types, model_file, \
    logger, save_dir, tokenized_train, tokenized_valid, train_batch_labels, valid_batch_labels, \
    input_layer_key,gene_ids, sample_size = load_data_and_model(hyperparameter_defaults)
    
    # 16382
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    ntokens = len(vocab)  # size of vocabulary
    ewc_regularizer = EWC(device, fisher_path=fisher_path)
    model = MyTransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        dropout=hyperparameter_defaults['dropout'],
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=True,
        do_dab=True,
        use_batch_labels=True,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=True,
        n_input_bins=n_input_bins,
        ecs_threshold=hyperparameter_defaults['ecs_thres'],
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=True,
        pre_norm=False,
    )

    if hyperparameter_defaults['load_model'] is not None:
        load_pretrained(model, torch.load(model_file), verbose=False)

    model.to(device)
    
    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, Shape: {param.shape}")
    
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])
    
    # n_encoder_layer, fc1_size, fc2_size, fc3_size = 1, 512, 512
    # fc1_impt, fc1_mask, fc2_impt, fc2_mask, tot_tokens = initial_impt(n_encoder_layer, fc1_size, fc2_size)
    task = 1
    end_losses = []
    for epoch in range(1, config['epochs'] + 1):

        epoch_start_time = time.time()

        best_val_loss = float("inf")
        best_avg_bio = 0.0
        best_model = None
        # define_wandb_metrcis()
        train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
        train_loader = prepare_dataloader(
            train_data_pt,
            batch_size=config['batch_size'],
            shuffle=True,
            intra_domain_shuffle=True,
            drop_last=False,
        )
        valid_loader = prepare_dataloader(
            valid_data_pt,
            batch_size=config['batch_size'],
            shuffle=False,
            intra_domain_shuffle=False,
            drop_last=False,
        )

        if task == 1 and epoch == config['epochs']:
                # estimate the fisher information of the parameters and consolidate
                # them in the network.
            print(
                '=> Estimating diagonals of the fisher information matrix...',
                flush=True, end='',
            )
            ewc_regularizer.update_fisher_optpar(model, task, train_loader, 
                            sample_size, device, config, vocab, DSBN, pad_token, mask_value, criterion, scaler, optimizer,dataset_name, batch_size = config['batch_size'], consolidate=True)
            print(' Done!')
            break

        # if epoch == config['epochs']:
        #     end_losses = train(model, loader=train_loader, epoch = epoch, sample_size = sample_size)
        # else:
        
        train(model, loader = train_loader, epoch = epoch, sample_size = sample_size)
        
        val_loss, val_mre = evaluate(
            model,
            loader=valid_loader,
        )
        elapsed = time.time() - epoch_start_time
        logger.info("-" * 89)
        logger.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
        )
        logger.info("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            best_model_epoch = epoch
            logger.info(f"Best model with score {best_val_loss:5.4f}")

        if epoch % config['save_eval_interval'] == 0 or epoch == config['epochs']:
            logger.info(f"Saving model to {save_dir}")
            torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")

            # eval on testdata
            results, adata_sorted = eval_testdata(
                best_model,
                adata_t=adata_sorted if per_seq_batch_sample else adata,
                include_types=["cls"],
            )
      
            sc.pp.neighbors(adata_sorted, use_rep="X_scGPT")
            sc.tl.umap(adata_sorted, min_dist=0.3)
            sc.pl.umap(
                    adata_sorted,
                    color=["str_batch"],
                    title=[f"batch, avg_batch = {results.get('avg_batch', 0.0):.4f}"],
                    frameon=False,
                    # return_fig=True,
                    show=False,
                    # save = f"{dataset_name}_embeddings_batch_umap[cls]_e{best_model_epoch}.png"
            )
            plt.savefig(str(save_dir) + "/" + dataset_name + "_" + experiment_name + "_" + f"embeddings_batch_umap[cls]_e{best_model_epoch}.png",bbox_inches='tight')
            
            sc.pp.neighbors(adata_sorted, use_rep="X_scGPT")
            sc.tl.umap(adata_sorted, min_dist=0.3)
            sc.pl.umap(
                    adata_sorted,
                    color=["celltype"],
                    title=[
                    f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
                    ],
                    frameon=False,
                    # return_fig=True,
                    show=False,
                    # save=f"{dataset_name}_embeddings_celltype_umap[cls]_e{best_model_epoch}.png"
            )
            plt.savefig(str(save_dir) + "/" + dataset_name + "_" + experiment_name + "_" + f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", bbox_inches='tight')

        scheduler.step()

        if task == 1 and epoch == config['epochs']:
                # estimate the fisher information of the parameters and consolidate
                # them in the network.
            print(
                '=> Estimating diagonals of the fisher information matrix...',
                flush=True, end='',
            )
            ewc_regularizer.update_fisher_optpar(model, task, train_loader, 
                            sample_size, device, config, vocab, DSBN, pad_token, mask_value, criterion, scaler, optimizer, dataset_name, batch_size = config['batch_size'], consolidate=True)
            print(' Done!')

    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    
