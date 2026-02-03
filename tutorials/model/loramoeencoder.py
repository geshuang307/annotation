# Simplified LoRAMoE implementation with localized balancing constraint
import copy
from typing import Optional, Any, Union, Callable

import torch
import warnings
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import torch
import torch.nn as nn

# cummulate lora version2
# class LoRAMoE(nn.Module):
#     def __init__(self, d_model, adapter_dim=64, num_experts=4):
#         super(LoRAMoE, self).__init__()
#         self.num_experts = num_experts
#         self.d_model = d_model
#         self.adapter_dim = adapter_dim

#         # LoRA的低秩矩阵
#         self.lora_down = nn.ModuleList([Linear(d_model, adapter_dim, bias=False) for _ in range(num_experts)])
#         self.lora_up = nn.ModuleList([Linear(adapter_dim, d_model, bias=False) for _ in range(num_experts)])

#         # gating网络，输入是d_model维度，输出num_experts维度，表示每个token对专家的权重
#         # self.gate = Linear(d_model, num_experts)
#         # self.last_gating_probs = None

#     def get_gating_probs(self):
#         return self.last_gating_probs
    
#     def set_active_expert(self, active_id):
#         for i in range(self.num_experts):
#             trainable = bool(i == int(active_id.numpy()))
#             for p in self.lora_down[i].parameters():
#                 p.requires_grad = trainable
#             for p in self.lora_up[i].parameters():
#                 p.requires_grad = trainable

#     def forward(self, x, batch_id=None):
#         """
#         x: [..., d_model]
#         batch_id: int 或 None
#         - 训练时：指定当前要更新的专家编号 i ∈ [0, num_experts)
#         - 测试/评估：可传 None；或即使传入，也不会影响梯度（eval 模式）
#         """
#         self.set_active_expert(batch_id)
#         expert_outputs = []
#         if batch_id is not None:
#             for i in range(batch_id+1):
#                 # 训练阶段：只让 batch_id 对应专家参与反向传播，其它专家 no_grad
#                 if self.training and (batch_id is not None) and bool(i != int(batch_id.numpy())):
#                     with torch.no_grad():
#                         down = self.lora_down[i](x)     # [..., adapter_dim]
#                         up   = self.lora_up[i](down)    # [..., d_model]
#                 else:
#                     down = self.lora_down[i](x)
#                     up   = self.lora_up[i](down)

#                 expert_outputs.append(up)
#         else:
#             # 测试/评估阶段：所有专家都参与前向传播
#             for i in range(self.num_experts):
#                 down = self.lora_down[i](x)     # [..., adapter_dim]
#                 up   = self.lora_up[i](down)    # [..., d_model]
#                 expert_outputs.append(up)

#         # 堆叠并在专家维求和：[..., num_experts, d_model] -> [..., d_model]
#         expert_outputs = torch.stack(expert_outputs, dim=-2)
#         output = expert_outputs.sum(dim=-2)

#         return output

# single/two lora for each task
# class LoRAMoE(nn.Module):
#     def __init__(self, d_model, adapter_dim=64, num_experts=4):
#         super(LoRAMoE, self).__init__()
#         self.num_experts = num_experts
#         self.d_model = d_model
#         self.adapter_dim = adapter_dim

#         # LoRA的低秩矩阵
#         self.lora_down = nn.ModuleList([nn.Linear(d_model, adapter_dim, bias=False) for _ in range(num_experts)])
#         self.lora_up = nn.ModuleList([nn.Linear(adapter_dim, d_model, bias=False) for _ in range(num_experts)])

#     def compute_orthogonal_loss(self, current_expert_idx):
#         """
#         计算当前专家和之前专家之间的正交性损失。
        
#         Args:
#             current_expert_idx (int): 当前任务的专家索引
#         Returns:
#             torch.Tensor: 正交性损失
#         """
#         current_lora_down = self.lora_down[current_expert_idx].weight  # 当前专家的低秩矩阵
#         orth_loss = 0.0

#         # 计算当前专家与之前专家之间的正交性损失
#         for i in range(current_expert_idx):  # 遍历之前的专家
#             past_lora_down = self.lora_down[i].weight  # 过去任务的低秩矩阵
#             # 计算当前专家和过去专家之间的正交性损失
#             orth_loss += torch.sum(torch.matmul(current_lora_down.T, past_lora_down) ** 2)
#         return orth_loss
    
#     def forward(self, x, batch_id=None):
#         """
#         x: 输入张量，形状为 [..., d_model]
#         batch_id: int 或 None
#         - 训练时：指定当前要更新的专家编号 i ∈ [0, num_experts)
#         - 测试/评估：可传 None；或即使传入，也不会影响梯度（eval 模式）
#         """
#         # 如果传入了batch_id，设置当前专家为可训练
#         # if batch_id is not None:
#         #     self.set_active_expert(batch_id)
        
#         # 只使用当前专家的计算
#         if batch_id is not None:
#             # 使用 batch_id 对应的专家进行计算
#             down = self.lora_down[0](x)
#             up = self.lora_up[0](down)
#             down = self.lora_down[batch_id+1](x)
#             up = self.lora_up[batch_id+1](down)
#         else:
#             down = self.lora_down[0](x)
#             up = self.lora_up[0](down)

#         return up  # 只返回当前batch_id对应专家的输出

# origional version 1
class LoRAMoE(nn.Module):
    def __init__(self, d_model, adapter_dim=64, num_experts=4):
        super(LoRAMoE, self).__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.adapter_dim = adapter_dim

        # LoRA的低秩矩阵
        self.lora_down = nn.ModuleList([Linear(d_model, adapter_dim, bias=False) for _ in range(num_experts)])
        self.lora_up = nn.ModuleList([Linear(adapter_dim, d_model, bias=False) for _ in range(num_experts)])

        # gating网络，输入是d_model维度，输出num_experts维度，表示每个token对专家的权重
        self.gate = Linear(d_model, num_experts)
        self.last_gating_probs = None

    def get_gating_probs(self):
        return self.last_gating_probs
    def forward(self, x):
        # x: [seq_len, batch_size, d_model] 或者 [batch_size, seq_len, d_model]

        gate_scores = self.gate(x)  # [*, num_experts]
        gate_weights = torch.softmax(gate_scores, dim=-1)  # softmax归一化专家权重
        self.last_gating_probs = gate_weights.detach() 
        # 对每个专家做LoRA低秩变换，再加权求和
        expert_outputs = []
        for i in range(self.num_experts):
            down = self.lora_down[i](x)    # [*, adapter_dim]
            up = self.lora_up[i](down)     # [*, d_model]
            expert_outputs.append(up)

        # 堆叠专家输出：[*, num_experts, d_model]
        expert_outputs = torch.stack(expert_outputs, dim=-2)
        # gate_weights: [*, num_experts] -> unsqueeze最后一维方便广播
        gate_weights = gate_weights.unsqueeze(-1)

        # 加权求和专家输出
        output = (expert_outputs * gate_weights).sum(dim=-2)  # [*, d_model]

        return output

__all__ = ['Transformer', 'TransformerEncoder', 'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer']
def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

class TransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers.

    Users can build the BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first :
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
            self.use_nested_tensor = False


    def forward(
            self,
            src: Tensor,
            batch_id: Optional[Tensor] = None,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        for mod in self.layers:
            output = mod(output, batch_id=batch_id, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, adapter_dim=64, moe_experts=4) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        # 替换 Adapter 为 LoRAMoE
        self.lora_moe = LoRAMoE(d_model, adapter_dim=adapter_dim, num_experts=moe_experts)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu

    def _transformer_encoder_layer_fwd(self, src, batch_id, embed_dim, num_heads, in_proj_weight, in_proj_bias,
                                       out_proj_weight, out_proj_bias, is_gelu, norm_first, norm1_eps,
                                       norm1_weight, norm1_bias, norm2_weight, norm2_bias,
                                       linear1_weight, linear1_bias, linear2_weight, linear2_bias,
                                       merged_mask, mask_type):
        # 自注意力部分
        q, k, v = self.self_attn._in_proj_qkv(src, in_proj_weight, in_proj_bias)
        q = self.self_attn._in_proj_q(src, in_proj_weight, in_proj_bias)
        k = self.self_attn._in_proj_k(src, in_proj_weight, in_proj_bias)
        v = self.self_attn._in_proj_v(src, in_proj_weight, in_proj_bias)

        head_dim = embed_dim // num_heads
        q = q.contiguous().view(*q.shape[:-1], num_heads, head_dim).transpose(-3, -2)
        k = k.contiguous().view(*k.shape[:-1], num_heads, head_dim).transpose(-3, -2)
        v = v.contiguous().view(*v.shape[:-1], num_heads, head_dim).transpose(-3, -2)

        attn_output_weights = torch.matmul(q, k.transpose(-1, -2))
        attn_output_weights = attn_output_weights / (head_dim ** 0.5)

        if merged_mask is not None:
            attn_output_weights += merged_mask

        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.self_attn.dropout(attn_output_weights)

        attn_output = torch.matmul(attn_output_weights, v)
        attn_output = attn_output.transpose(-3, -2).contiguous().view(*attn_output.shape[:-3], -1, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

        # 残差连接
        x = src + attn_output     # B, 1201, d

        if norm_first:
            # 第一层归一化
            x_norm = F.layer_norm(x, (embed_dim,), norm1_weight, norm1_bias, norm1_eps)
            # MLP 部分
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
            # LoRAMoE模块
            if batch_id is not None:
                moe_output = self.lora_moe(x_norm, batch_id)
            else:
                moe_output = self.lora_moe(x_norm)
            # 合并 MLP 和 LoRAMoE 输出
            x = x + ff_output + moe_output
            # 第二层归一化
            x = F.layer_norm(x, (embed_dim,), norm2_weight, norm2_bias, norm1_eps)
        else:
            x = F.layer_norm(x, (embed_dim,), norm1_weight, norm1_bias, norm1_eps)
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))
            if batch_id is not None:
                moe_output = self.lora_moe(x, batch_id)
            else:
                moe_output = self.lora_moe(x)
            x = x + ff_output + moe_output
            x = F.layer_norm(x, (embed_dim,), norm2_weight, norm2_bias, norm1_eps)

        return x

    def forward(
            self,
            src: Tensor,
            batch_id : Optional[Tensor] = None,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
            
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first :
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim :
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in tensor_args):
                why_not_sparsity_fast_path = ("some Tensor argument's device is neither one of "
                                              f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask, src_key_padding_mask, src)
                x = self._transformer_encoder_layer_fwd(
                    src,
                    batch_id,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )
                return x

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
            # 添加 Adapter 模块输出
            if batch_id is None:
                x = x + self.lora_moe(self.norm2(x))
            else:
                x = x + self.lora_moe(self.norm2(x), batch_id)
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
            # 添加 Adapter 模块输出
            if batch_id is None:
                x = x + self.lora_moe(x)
            else:
                x = x + self.lora_moe(x, batch_id)
        return x
    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

            

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal
def localized_balancing_loss(gate_probs, task_types, expert_types, delta=0.5):
    # gate_probs: [batch, seq_len, num_experts]
    batch, seq_len, num_experts = gate_probs.shape
    Q = gate_probs.sum(dim=1)  # sum over tokens => [batch, num_experts]

    # I: importance matrix [batch, num_experts]
    I = torch.zeros_like(Q)
    for b in range(batch):
        for n in range(num_experts):
            if expert_types[n] == task_types[b]:
                I[b, n] = 1 + delta
            else:
                I[b, n] = 1 - delta

    Z = I * Q
    var = Z.var(dim=1).mean()
    mean = Z.mean()
    lbc_loss = var / (mean + 1e-8)
    return lbc_loss

# Example usage
if __name__ == '__main__':
    batch_size = 4
    seq_len = 16
    hidden_dim = 512
    num_experts = 6

    model = LoRAMoE(hidden_dim, num_experts)
    dummy_input = torch.randn(batch_size, seq_len, hidden_dim)
    output, gate_probs = model(dummy_input)

    # Task types: 0 = world knowledge, 1 = downstream
    task_types = [0, 1, 0, 1]  # per sample
    expert_types = [0, 0, 1, 1, 0, 1]  # per expert

    lbc_loss = localized_balancing_loss(gate_probs, task_types, expert_types)
    print("Localized Balancing Constraint Loss:", lbc_loss.item())
