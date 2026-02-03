import os
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import trange
from sklearn.cluster import KMeans
from scipy.sparse import issparse

import matplotlib.pyplot as plt
import anndata
import scvi
import scanpy as sc
import json
import itertools
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import scgpt as scg
# if config["adapter"]:
# from adaptermodel import TransformerModel                        # adapter
# from scgpt.model import TransformerModel, AdversarialDiscriminator
# from loramoemodel import TransformerModel, AdversarialDiscriminator   # loramoe
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, load_pretrained
from anndata import AnnData
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from scipy import sparse
import gc
import math
# 设置环境变量和警告
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score
from collections import defaultdict
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import f1_score
# from pytorch_lightning.callbacks import EarlyStopping

def init_wandb():
    hyperparameter_defaults = dict(
        seed=0,
        # dataset_name="ms",
        # dataset_name="pancreas",
        # dataset_name="PBMC_10K",
        # dataset_name = "pancreas_PBMC_10K",
        # dataset_name = "spleen",
        dataset_name = "myeloid",
        do_train=True,
        load_model="../save/scEvolver_human",
        weight_dir="../save/scEvolver_human",
        # experiment_name="fine_tune_on_pancreas_minibatch_baseline(uniformsplit_no_schedular_nlayers_cls0_pastmse[origion])",
        # experiment_name="fine_tune_on_pancreas_minibatch_baseline(uniformsplit_no_schedular_nlayers_cls0)",
        # experiment_name="fine_tune_on_pancreas_minibatch_baseline(randomsplit_no_schedular_nlayers_cls3_init_class8)",
        mask_ratio=0.0,
        epochs=1000,
        n_bins=51,
        MVC=False,
        ecs_thres=0.0,
        dab_weight=0.0,
        lr=1e-4,
        batch_size=32,
        layer_size=128,
        nlayers=4,
        nhead=4,
        dropout=0.2,
        schedule_ratio=0.9,
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,
        include_zero_gene=False,
        DSBN=False,
        k_samples=50,
        pastmse=False,
        replay=False,
        init_class = 9,             # few-shot场景下是7个初始类别, 原始是14
        filter_sample = True,     
        randomsplit = False,        # few-shot场景下需要False
        fewshot = 5,                # None
        nlayers_cls=3,              # 3
        init_weight = False,
        use_best_initnextbatch = False,
        adapter = False,
        freeze_all = True,
        loramoe = True,                      # 报错就是导入的Transformer模型没选对
        lora = False,
        num_of_expert = 4,
        num_of_gate = 4,
        batch_expert = False,
        repultion_loss = False,
        mi_loss = False,
        schedule = "cosine_schedule_with_warmup",                           # "cosine_schedule_with_warmup",stepLR
        # data_seed = [42, 2025, 2015, 2005, 1995]  
    )
    config = hyperparameter_defaults
    config["experiment_name"] = "fine_tune_on_" + str(config["dataset_name"]) +"_minibatch_baseline(" + "randomsplit_" + str(config["randomsplit"]) +\
         "nlayers_cls" + str(config["nlayers_cls"]) + "_init_class" + str(config["init_class"]) + "_adapter_" + str(config["adapter"])\
              + "_freeze_all_" + str(config["freeze_all"]) +\
               "_fewshot_" + str(config["fewshot"]) +\
                "_loramoe_" + str(config["loramoe"]) +\
                "_schedule_" + str(config["schedule"]) +\
                "_batch_expert_" + str(config["batch_expert"])
                #  + "_repultion_loss_" + str(config["repultion_loss"]) 
                 
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    
    if config["dataset_name"] == "pancreas":
        config["init_class"] = 7
        config["lr"] = 1e-4
    elif config["dataset_name"] == "myeloid":
        config["init_class"] = 9
        config["lr"] = 1e-5
    return config

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_metric, model):
        score = -val_metric if self.mode == 'min' else val_metric

        if self.best_score is None:
            self.best_score = score
            self.best_state = model.state_dict()
        elif (score < self.best_score + self.min_delta if self.mode == 'min'
              else score < self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_state = model.state_dict()
            self.counter = 0

def load_data(config):
    dataset_name = config["dataset_name"]
    if dataset_name == "PBMC_10K":
        adata = sc.read_h5ad("../data/pbmc/pbmc_10k.h5ad")
    elif dataset_name == "pancreas":
        adata = sc.read_h5ad("../data/PANCREAS/pancreas_data.h5ad")
    return adata


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
    per_seq_batch_sample: bool = False,
) -> DataLoader:
    # if num_workers == 0:
    #     num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

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

def prepare_testdata(sort_seq_batch, tokenized_test, test_batch_labels,test_celltype_labels, mask_ratio,\
                 mask_value, pad_value):
    masked_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    input_gene_ids_test = (tokenized_test["genes"])
    input_values_test = masked_values_test
    target_values_test = (
        tokenized_test["values"])
    tensor_batch_labels_test = torch.from_numpy(test_batch_labels).long()
    tensor_celltype_labels_test = torch.from_numpy(test_celltype_labels).long()
    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        test_sort_ids = np.argsort(test_batch_labels)
        input_gene_ids_test = input_gene_ids_test[test_sort_ids]
        input_values_test = input_values_test[test_sort_ids]
        target_values_test = target_values_test[test_sort_ids]
        tensor_batch_labels_test = tensor_batch_labels_test[test_sort_ids]
        tensor_celltype_labels_test = tensor_celltype_labels_test[test_sort_ids]
    test_data_pt = {
        "gene_ids": input_gene_ids_test,
        "values": input_values_test,
        "target_values": target_values_test,
        "batch_labels": tensor_batch_labels_test,
        "celltype_labels": tensor_celltype_labels_test,
    }
    return test_data_pt

def prepare_data(sort_seq_batch, tokenized_train, tokenized_valid, train_batch_labels, valid_batch_labels, train_celltype_labels, \
                 valid_celltype_labels, mask_ratio,\
                 mask_value, pad_value):
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
    # print(
    #     f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
    #     f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    # )

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

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    if input_gene_ids_train.shape[0]>=4000:
        perm = torch.randperm(input_gene_ids_train.shape[0])                # shuffle 数据取前2000，否则内存不够
        input_gene_ids_train = input_gene_ids_train[perm][:4000]
        input_values_train = input_values_train[perm][:4000]
        target_values_train = target_values_train[perm][:4000]
        tensor_batch_labels_train = tensor_batch_labels_train[perm][:4000]
        tensor_celltype_labels_train = tensor_celltype_labels_train[perm][:4000]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt

def eval_scib_metrics(
    adata: AnnData,
    batch_key: str = "str_batch",
    label_key: str = "celltype",
    notes: Optional[str] = None,
    
) -> Dict:
    import scib
   
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_scEvolver",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )
    if notes is not None:
        print(f"{notes}")

    print(f"{results}")

    result_dict = results[0].to_dict()
    result_dict = {k: (0 if isinstance(v, float) and np.isnan(v) else v) for k, v in result_dict.items()}

    print(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )
    result_dict["avg_batch"] = np.mean(
        [
            result_dict["graph_conn"],
            result_dict["ASW_label/batch"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict

def plot_entropy_accuracy(entropy_list, correct_mask, save_dir, test_batch_idx, labellist):
    # 转换为 numpy
    import pandas as pd
    entropy = torch.stack(entropy_list).cpu().numpy()
    correct_mask = torch.stack(correct_mask).cpu().numpy()

    # 按照正确 / 错误分组
    entropy_correct = entropy[correct_mask]
    entropy_wrong = entropy[~correct_mask]
    plt.figure()
    plt.boxplot([entropy_correct, entropy_wrong], labels=["Correct", "Wrong"])
    plt.ylabel("Entropy")
    plt.title("Entropy Distribution by Classification Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(save_dir) + "/" + f"batch_{test_batch_idx}_Accuracy_Entropy.png")

    df = pd.DataFrame({
        "Entropy": entropy,
        "Correct": ["Correct" if x else "Incorrect" for x in correct_mask],
        "CellType": labellist
    })

    # 设置图形风格
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # 箱型图：横轴为细胞类型，纵轴为熵，颜色为分类是否正确
    ax = sns.boxplot(data=df, x="CellType", y="Entropy", hue="Correct", palette="Set2")

    plt.title(f"Entropy Distribution per Cell Type (Batch {test_batch_idx})", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(save_dir, f"entropy_boxplot_batch{test_batch_idx}.png"))


def get_expert_token_importance(gating_scores, top_k=1):
    """
    参数:
    - gating_scores: [B, T, N] 每个 token 的 softmax 后的 expert 权重
    - top_k: 若指定，则只保留 top-k 权重

    返回:
    - importance_per_token: [B, T, N] 每个 token 分配给每个 expert 的权重（可稀疏化）
    """
    if top_k is not None:
        # 只保留 top-k 分配
        topk_values, topk_indices = torch.topk(gating_scores, top_k, dim=-1)
        mask = torch.zeros_like(gating_scores)
        mask.scatter_(-1, topk_indices, topk_values)
        return mask
    else:
        # 保留全部 soft 分配
        return gating_scores

from torch import Tensor
class CosineScheduleWithWarmup(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1  # PyTorch scheduler从-1开始
        if step < self.num_warmup_steps:
            # Linear warmup
            return [
                base_lr * step / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            progress = (step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return [
                base_lr * cosine_decay
                for base_lr in self.base_lrs
            ]
        
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
        # self.out_layer = nn.Linear(d_model, n_cls, bias=False)
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        out = self.out_layer(x)
        return out   
    
class ContinualClassify():
    def __init__(self, config, vocab, num_batch_types, max_batch_idx=5, modeldict_name = "best_model.pt"):
        self.config = config
        self.vocab = vocab
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_max_classes = self.config["init_class"]
        # self.classifier = self.config["classifier"]
        self.model, self.model_file = self.prepare_model(num_batch_types, self.num_max_classes, modeldict_name)
        self.model.to(self.device)
        self.model.cls_decoder = ClsDecoder(512, self.num_max_classes).to(self.device)
        if self.config["load_model"] is not None:
            load_pretrained(self.model, torch.load(self.model_file, map_location=self.device), verbose=False)

        if self.config["freeze_all"]:
            # state_dict = torch.load(self.model_file)
            # for name, param in state_dict.items():
            #     print(f"{name}: {param.shape}")
            # for name, param in self.model.named_parameters():
            #     print(f"{name}: {param.shape}")
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                # if 'norm' in name:
                #     param.requires_grad = True
                if 'decoder' in name:
                    param.requires_grad = True
                if 'out_layer' in name:
                    param.requires_grad = True
                # if "layers.11" or "layers.10" in name:
                #     param.requires_grad = True
        if self.config["adapter"]:
            # state_dict = torch.load(self.model_file)
            # for name, param in state_dict.items():
            #     print(f"{name}: {param.shape}")
            # for name, param in self.model.named_parameters():
            #     print(f"{name}: {param.shape}")
            for name, param in self.model.named_parameters():
                if "adapter" in name:
                    param.requires_grad = True 
        if self.config["loramoe"] or self.config["lora"]:
            for name, param in self.model.named_parameters():
                if "lora_moe" in name:
                    param.requires_grad = True 


        self.criterion = masked_mse_loss
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_dab = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["lr"], eps=1e-4 if config["amp"] else 1e-8
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=config["schedule_ratio"]
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
        self.past_data = None
        # self.past_model = copy.deepcopy(self.model)
        self.past_model = None
        self.old_model_annotation = None
        self.max_test_id = max_batch_idx
        self.repultion_loss_list = []
        self.past_valid_loaders = {}
        # self.add_prompt = L2PPromptPoolTopN()
        self.cell_emnbeddings = None

    def prepare_model(self, num_batch_types, num_max_classes=2, modeldict_name="best_model.pt"):
        if self.config["loramoe"]:
            from loramoemodel import TransformerModel, AdversarialDiscriminator   # loramoe
        elif self.config["lora"]:
            from loramodel import TransformerModel, AdversarialDiscriminator          # lora
        else:
            from scgpt.model import TransformerModel, AdversarialDiscriminator
        if self.config["load_model"] is not None:
            model_dir = Path(self.config["load_model"])
            weight_dir = Path(self.config["weight_dir"])
            model_config_file = model_dir / "args.json"
            model_file = weight_dir / modeldict_name
            vocab_file = model_dir / "vocab.json"

            self.vocab = GeneVocab.from_file(vocab_file)
            special_tokens = ["<pad>", "<cls>", "<eoc>"]
            for s in special_tokens:
                if s not in self.vocab:
                    self.vocab.append_token(s)

            with open(model_config_file, "r") as f:
                model_configs = json.load(f)
            embsize = model_configs["embsize"]
            nhead = model_configs["nheads"]
            d_hid = model_configs["d_hid"]
            nlayers = model_configs["nlayers"]
            n_layers_cls = model_configs["n_layers_cls"]
        else:
            embsize = self.config["layer_size"]
            nhead = self.config["nhead"]
            nlayers = self.config["nlayers"]
            d_hid = self.config["layer_size"]

        pad_token = "<pad>"
        pad_value = -2
        explicit_zero_prob = False
        if self.config["loramoe"]:
            model = TransformerModel(
                len(self.vocab),
                embsize,
                nhead,
                d_hid,
                nlayers,
                nlayers_cls=self.config["nlayers_cls"],
                n_cls=num_max_classes,                  # 初始化8个分类器
                vocab=self.vocab,
                dropout=self.config["dropout"],
                pad_token=pad_token,
                pad_value=pad_value,
                do_mvc=self.config["MVC"],
                do_dab=False,
                use_batch_labels=False,
                num_batch_labels=num_batch_types,
                domain_spec_batchnorm=self.config["DSBN"],
                input_emb_style="continuous",
                n_input_bins=self.config["n_bins"],
                cell_emb_style="cls",
                mvc_decoder_style="inner product",
                ecs_threshold=self.config["ecs_thres"],
                explicit_zero_prob=explicit_zero_prob,
                use_fast_transformer=self.config["fast_transformer"],
                fast_transformer_backend="flash",
                pre_norm=self.config["pre_norm"],
                moe_experts=self.config["num_of_expert"],
            )
        else:
                model = TransformerModel(
                len(self.vocab),
                embsize,
                nhead,
                d_hid,
                nlayers,
                nlayers_cls=self.config["nlayers_cls"],
                n_cls=num_max_classes,                  # 初始化8个分类器
                vocab=self.vocab,
                dropout=self.config["dropout"],
                pad_token=pad_token,
                pad_value=pad_value,
                do_mvc=self.config["MVC"],
                do_dab=False,
                use_batch_labels=False,
                num_batch_labels=num_batch_types,
                domain_spec_batchnorm=self.config["DSBN"],
                input_emb_style="continuous",
                n_input_bins=self.config["n_bins"],
                cell_emb_style="cls",
                mvc_decoder_style="inner product",
                ecs_threshold=self.config["ecs_thres"],
                explicit_zero_prob=explicit_zero_prob,
                use_fast_transformer=self.config["fast_transformer"],
                fast_transformer_backend="flash",
                pre_norm=self.config["pre_norm"],
            )
        # if self.config["load_model"] is not None:
        #     load_pretrained(model, torch.load(model_file), verbose=False)
        
        # state_dict = torch.load(model_file, map_location='cpu')

        # for name, param in state_dict.items():
        #     print(f"{name}: {param.shape}")
        # for name, param in model.named_parameters():
        #     print(f"{name:50} | shape: {str(tuple(param.shape)):20} | requires_grad: {param.requires_grad}")
        return model, model_file
    
    def extended_mutual_information_loss(prob_logits, eps=1e-8):
        """
        Extended Mutual Information Regularization Loss

        Args:
            prob_logits: Tensor of shape [batch_size, num_clusters]
                        The soft assignments (e.g., after a softmax layer)
        Returns:
            Scalar loss value (torch.Tensor)
        """
        # softmax over logits to get probabilities
        p = F.softmax(prob_logits, dim=1)     # shape: [B, K]
        p = p + eps  # numerical stability
        log_p = torch.log(p)

        # 1. Conditional entropy H(C|X) = -E_x sum_j p(c_j|x) log p(c_j|x)
        cond_entropy = -torch.mean(torch.sum(p * log_p, dim=1))  # scalar

        # 2. Marginal distribution p̂(C) ≈ mean over batch
        p_mean = torch.mean(p, dim=0)  # shape: [K]
        log_p_mean = torch.log(p_mean + eps)

        # 3. Entropy H(C) = -sum_j p̂(c_j) log p̂(c_j)
        marginal_entropy = -torch.sum(p_mean * log_p_mean)  # scalar

        # Mutual Information = H(C) - H(C|X), so we *minimize* -MI
        mi_loss = cond_entropy - marginal_entropy

        return mi_loss
    
    def repultion_loss(self, new_weight_matrix):
        repulsion_loss = 0.0
        for i in range(new_weight_matrix.size(0)):
            for j in range(new_weight_matrix.size(0)):
                if i != j:
                    sim = F.cosine_similarity(new_weight_matrix[i].unsqueeze(0),
                                            new_weight_matrix[j].unsqueeze(0))
                    repulsion_loss += torch.exp(sim)  # 越相似，惩罚越大
        repulsion_loss /= (new_weight_matrix.size(0) * (new_weight_matrix.size(0) - 1))
        return repulsion_loss
    
    def train(self, loader, logger, epoch, test_batch_idx, save_dir):
        self.model.train()
        if self.past_model:
            self.past_model.eval()
        total_loss, total_cls, total_num = 0.0, 0.0, 0.0
        log_interval = 100
        start_time = time.time()
        num_batches = len(loader)
        cell_emb_list = []
        celltype_labels_list = []
        train_iter_loss_list = []
        cumulative_importance = torch.zeros(1201, self.config["num_of_gate"]).to(self.device)
        num_iter = 0
        average_importance = torch.zeros(1201, self.config["num_of_gate"]).to(self.device)
        for batch, batch_data in enumerate(itertools.islice(loader, num_batches)):
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            target_values = batch_data["target_values"].to(self.device)
            batch_labels = batch_data["batch_labels"].to(self.device)
            celltype_labels = batch_data["celltype_labels"].to(self.device)

            src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

            with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                    output_dict, _ = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if False or self.config["DSBN"] else None,
                    # batch_id = torch.tensor(test_batch_idx),
                    batch_id = None,
                    CLS=True,
                    CCE=False,
                    MVC=self.config["MVC"],
                    ECS=self.config["ecs_thres"] > 0,
                    do_sample=False,
                )
                else:
                    output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if False or self.config["DSBN"] else None,
                    CLS=True,
                    CCE=False,
                    MVC=self.config["MVC"],
                    ECS=self.config["ecs_thres"] > 0,
                    do_sample=False,
                )
                cell_emb = output_dict["cell_emb"].squeeze(1)
                # cell_emb = torch.cat([cell_emb_past, cell_emb], dim=-1)
                cell_emb_list.append(cell_emb)
                celltype_labels_list.append(celltype_labels)
                masked_positions = input_values.eq(-1)
                metrics_to_log = {}
                total_num += len(input_gene_ids)

                loss_cls = self.criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                        (output_dict["cls_output"].argmax(1) == celltype_labels)
                        .sum()
                        .item()
                ) / celltype_labels.size(0)
                train_iter_loss_list.append(loss_cls.item())

                # found = any(
                #             self.model.cls_decoder.out_layer.weight.data_ptr() == p.data.data_ptr()
                #             for group in self.optimizer.param_groups
                #             for p in group['params']
                #         )
                # print("Decoder权重是否在optimizer中：", found)
                if self.config["repultion_loss"]:
                    repulsion_loss = self.repultion_loss(self.model.cls_decoder.out_layer.weight.data)
                    loss += repulsion_loss.item()
                    self.repultion_loss_list.append(repulsion_loss.item())
                # for name, param in self.model.named_parameters():
                #     print(f"{name}: {param.shape}")
                if self.config["loramoe"] and epoch == self.config["epochs"]:
                    gating_scores = get_expert_token_importance(gating_scores = self.model.transformer_encoder.layers[11].lora_moe.get_gating_probs())
                    gating_scores = gating_scores.view(celltype_labels.size(0), -1, self.config["num_of_gate"])  # reshape

                    cumulative_importance += gating_scores.mean(dim=0)  # [T, N]
                    num_iter += 1
                    average_importance = cumulative_importance / num_iter  # [T, N]
                    torch.save(average_importance, save_dir / f"batch{test_batch_idx}_expert_importance.pt")
            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0,
                    error_if_nonfinite=False if self.scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    logger.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {self.scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.config["schedule"] == "cosine_schedule_with_warmup":
                self.scheduler.step()
            print(metrics_to_log)

            total_loss += loss.item()
            total_cls += loss_cls.item() if True else 0.0
            lr = self.scheduler.get_last_lr()[0]
            if batch % log_interval == 0 and batch > 0:
                # lr = self.scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_cls = total_cls / log_interval if True else 0.0

                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                    f"cls loss {cur_cls:5.2f} | "
                    f"loss {cur_loss:5.2f} |"
                )
                total_loss = 0
                total_cls = 0
                start_time = time.time()
            torch.cuda.empty_cache()
        cur_loss = total_loss / (total_num+1e-6)
        # if self.config["loramoe"] and epoch == self.config["epochs"]:
        #     sns.heatmap(average_importance.cpu().numpy(), cmap="viridis")
        #     plt.xlabel("Expert Index")
        #     plt.ylabel("Token Position")
        #     plt.title("Token-wise Expert Importance")
        #     plt.savefig(save_dir / f"average_importance_{test_batch_idx}.png")
        return cell_emb_list, celltype_labels_list, cur_loss, train_iter_loss_list
    
    def evaluate(self, loader, epoch, save_dir, test_batch_idx):
        self.model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        accuracy = 0
        predictions = []
        labellist = []
        entropy_list, accuracy_list = [], []
        num_batches = len(loader)
        
        with torch.no_grad():
            for batch, batch_data in enumerate(itertools.islice(loader, num_batches)):
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                batch_labels = batch_data["batch_labels"].to(self.device)
                celltype_labels = batch_data["celltype_labels"].to(self.device)
                src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

                with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                    if self.config["adapter"] or self.config["loramoe"]:
                        output_dict, _ = self.model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=batch_labels if False or self.config["DSBN"] else None,
                            # batch_id = torch.tensor(test_batch_idx),
                            batch_id = None,
                            CLS=True,
                            CCE=False,
                            MVC=False,
                            ECS=False,
                            do_sample=False,
                        )
                    else:
                        output_dict = self.model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=batch_labels if False or self.config["DSBN"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=False,
                            ECS=False,
                            do_sample=False,
                        )
                    output_values = output_dict["cls_output"]
                    probs = F.softmax(output_dict["cls_output"], dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                    loss = self.criterion_cls(output_values, celltype_labels)

                entropy_list.extend(entropy)
                accuracy_list.extend(output_values.argmax(1) == celltype_labels)
                total_loss += loss.item() * len(input_gene_ids)
                correct = (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += len(input_gene_ids) - correct
                accuracy += correct
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).detach().cpu().numpy()
                predictions.extend(preds)
                labellist.extend(celltype_labels.detach().cpu().numpy())

        # F1 计算
        f1_macro = f1_score(labellist, predictions, average='macro')
        f1_micro = f1_score(labellist, predictions, average='micro')
        f1_weighted = f1_score(labellist, predictions, average='weighted')

        eval_metrics = {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "valid/f1_macro": f1_macro,
            "valid/f1_micro": f1_micro,
            "valid/f1_weighted": f1_weighted,
            "epoch": epoch,
        }

        print(eval_metrics)

        result_dict = {
            "accuracy": accuracy / total_num,
            "preds": [int(p) for p in predictions],
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
        }

        if epoch == self.config['epochs']:

            with open(str(save_dir) + "/" + f"batch_{test_batch_idx}" + "final_results" +".json", "w") as f:
                json.dump(result_dict, f, indent=4)
            #########################  confusion_matrix #################################
            class_names = np.unique(np.concatenate((labellist, predictions)))  # 获取所有类别
            cm = confusion_matrix(labellist, predictions, labels=class_names)

            # 绘制热力图

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(str(save_dir) + "/" + f"batch_{test_batch_idx}_Confusion_Matrix.png")
            plot_entropy_accuracy(entropy_list, accuracy_list, save_dir, test_batch_idx, labellist)    #  画熵-准确性图 
        # ######################### compute previous sample performance #########################
        
        #     if len(self.past_valid_loaders) > 1:
        #         past_items = list(self.past_valid_loaders.items())[:-1]  # 取前 N-1 个 (key, loader) 对
        #         for i, (task_id, loader) in enumerate(past_items):
        #             num_batches = len(loader)
        #             with torch.no_grad():
        #                 for batch, batch_data in enumerate(itertools.islice(loader, num_batches)):
        #                     input_gene_ids = batch_data["gene_ids"].to(self.device)
        #                     input_values = batch_data["values"].to(self.device)
        #                     target_values = batch_data["target_values"].to(self.device)
        #                     batch_labels = batch_data["batch_labels"].to(self.device)
        #                     celltype_labels = batch_data["celltype_labels"].to(self.device)
        #                     src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

        #                     with torch.cuda.amp.autocast(enabled=self.config["amp"]):
        #                         if self.config["adapter"] or self.config["loramoe"]:
        #                             output_dict, _ = self.model(
        #                                 input_gene_ids,
        #                                 input_values,
        #                                 src_key_padding_mask=src_key_padding_mask,
        #                                 batch_labels=batch_labels if False or self.config["DSBN"] else None,
        #                                 CLS=True,
        #                                 CCE=False,
        #                                 MVC=False,
        #                                 ECS=False,
        #                                 do_sample=False,
        #                             )
        #                         else:
        #                             output_dict = self.model(
        #                                 input_gene_ids,
        #                                 input_values,
        #                                 src_key_padding_mask=src_key_padding_mask,
        #                                 batch_labels=batch_labels if False or self.config["DSBN"] else None,
        #                                 CLS=True,
        #                                 CCE=False,
        #                                 MVC=False,
        #                                 ECS=False,
        #                                 do_sample=False,
        #                             )
        #                         output_values = output_dict["cls_output"]
        #                     accuracy += (output_values.argmax(1) == celltype_labels).sum().item()
        #                     preds = output_values.argmax(1).detach().cpu().numpy()
        #                     predictions.extend(preds)
        #                     total_num += len(input_gene_ids)
        #             past_result_dict[i] = {
        #             "accuracy": accuracy / total_num,
        #             "preds": [int(p) for p in predictions],  # 转成 list
        #             "total_num": total_num
        #         }
        #     with open(str(save_dir) + "/" + f"test_after_batch_{test_batch_idx}" + ".json", "w") as f:
        #         json.dump(past_result_dict, f, indent=4)
                
        return total_loss / total_num, total_error / total_num, result_dict

    def eval_testdata(self, adata_t, gene_ids, input_layer_key):
        self.model.eval()
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

        # logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=3001,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):
            cell_embeddings = self.model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=self.config["batch_size"],
                batch_labels=torch.from_numpy(batch_ids).long() if self.config["DSBN"] else None,
                time_step=0,
                return_np=True,
            )

        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scEvolver"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            # logger.error(e)

        return results, adata_t
    
    def best_model_evaluate(self, best_model, adata_t, gene_ids, input_layer_key, test_batch_idx):
        best_model.eval()
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

        # logger.info("Evaluating cls cell embeddings")
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=3001,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):
            if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                cell_embeddings = best_model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels=torch.from_numpy(batch_ids).long() if self.config["DSBN"] else None,
                    # batch_id = torch.tensor(test_batch_idx),
                    batch_id=None,
                    time_step=0,
                    return_np=True,
                )
            else:
                cell_embeddings = best_model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels=torch.from_numpy(batch_ids).long() if self.config["DSBN"] else None,
                    time_step=0,
                    return_np=True,
                )

        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

        adata_t.obsm["X_scEvolver"] = cell_embeddings

        results = {}
        try:
            results = eval_scib_metrics(adata_t)
        except Exception as e:
            traceback.print_exc()
            # logger.error(e)

        return results, adata_t
    def update_classifier_weights(self, embedding_list, label_list):
        """
        根据类的嵌入平均更新分类网络的权重
        :param embedding_list: 所有样本的嵌入列表
        :param label_list: 所有样本的标签列表
        """
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = torch.unique(label_list).cpu().numpy()
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self.model.cls_decoder.out_layer.weight.data[class_index] = proto.to(self.device)

    # def split_data(self, adata_paths):

    def build_fewshot_dataset(self, all_counts, celltypes_labels, batch_ids, shots_per_class=5, seed=42):
        # np.random.seed(seed)

        label_to_indices = defaultdict(list)

        # 收集每个类别的索引
        for idx, label in enumerate(celltypes_labels):
            label_to_indices[label].append(idx)         

        train_indices = []
        val_indices = []

        for label, indices in label_to_indices.items():
            indices = np.array(indices)
            np.random.shuffle(indices)

            if len(indices) <= shots_per_class:
                raise ValueError(f"Class {label} has only {len(indices)} samples, fewer than {shots_per_class} shots.")

            train_indices.extend(indices[:shots_per_class])
            val_indices.extend(indices[shots_per_class:])

        # 转为 numpy array
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        print("number of train samples", train_indices.shape)
        print("number of val samples", val_indices.shape)
        # 划分各个部分
        train_data = all_counts[train_indices]
        valid_data = all_counts[val_indices]

        train_celltype_labels = celltypes_labels[train_indices]
        valid_celltype_labels = celltypes_labels[val_indices]

        train_batch_labels = batch_ids[train_indices]
        valid_batch_labels = batch_ids[val_indices]

        return train_data, valid_data, train_celltype_labels, valid_celltype_labels, train_batch_labels, valid_batch_labels, train_indices, val_indices

    def process_batch(self, adata_train, logger, save_dir, dataset_name, experiment_name, all_adata_test, all_batch_results, test_batch_idx):
        '''
        adata_train, config, logger, save_dir, config["dataset_name"], config["experiment_name"], all_adata_test, all_batch_results,
            test_batch_idx
        '''
        ############################# 先搞定数据 #################################
        # 统计每种细胞类型的数量
        if self.config["init_class"] == 8 or self.config["fewshot"] is not None:
        # if self.config["filter_sample"]:
            celltype_counts = adata_train.obs["celltype"].value_counts()
            # ######### rare celltype <10 ################# 稀有细胞类型重复利用
            # rarecell_bank_current = sc.AnnData()
            # rarecell_type = celltype_counts[celltype_counts < 10].index            # 找出每个batch的稀有细胞
            # adata_rare = adata_train[adata_train.obs["celltype"].isin(rarecell_type)].copy()  
            # rarecell_bank_current = adata_rare if len(rarecell_bank_current) == 0 else rarecell_bank_current.concatenate(adata_rare) 
            # if test_batch_idx > 0:
            #     dict_cell = torch.load(save_dir / "rarecell.pth")
            #     rarecell_bank_previous = dict_cell['rarecell_bank'] 
            #     rarecell_bank_current = rarecell_bank_previous.concatenate(rarecell_bank_current)
            #     adata_train = adata_train.concatenate(rarecell_bank_current)
            #     torch.save({'rarecell_bank': rarecell_bank_current}, save_dir / "rarecell.pth")
            # else:
            #     # rarecell_bank_previous = sc.AnnData()
            #     # rarecell_bank_current = rarecell_bank_previous.concatenate(rarecell_bank_current)
            #     torch.save({'rarecell_bank': rarecell_bank_current}, save_dir / "rarecell.pth")
            # ##################################################################
            # 找出数量大于等于 4 的细胞类型
            valid_celltypes = celltype_counts[celltype_counts >= self.config["fewshot"]+1].index

            # 过滤掉数量小于 4 的细胞类型对应的样本
            adata_train = adata_train[adata_train.obs["celltype"].isin(valid_celltypes)].copy()

        le = LabelEncoder()
        adata_train.obs["batch_id"] = le.fit_transform(adata_train.obs["batch_id"])
        num_batch_types = adata_train.obs["batch_id"].nunique()

        input_layer_key = {
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }["binned"]
        all_counts = (
            adata_train.layers[input_layer_key].A
            if issparse(adata_train.layers[input_layer_key])
            else adata_train.layers[input_layer_key]
        )
        genes = adata_train.var["gene_name"].tolist()

        # 计算当前批次 (test_batch_idx) 的细胞类型和标签
        current_label_dict, current_celltype_labels = np.unique(
            np.array(adata_train.obs["celltype"].tolist()), return_inverse=True
        )

        adata_train.obs["celltype_labels"] = current_celltype_labels  # 先临时保存一下

        if test_batch_idx == 0:
            # 初始化：直接用当前批次的信息
            new_model_annotation = current_label_dict
            self.old_model_annotation = current_label_dict

            # 映射表：celltype -> label
            celltype_to_label = {ct: i for i, ct in enumerate(self.old_model_annotation)}
            adata_train.obs["celltype_labels"] = adata_train.obs["celltype"].map(celltype_to_label)
            celltypes_labels = adata_train.obs["celltype_labels"].to_numpy()
            with open(save_dir / f"celltype_to_label.json", 'w') as file:
                json.dump(celltype_to_label, file)
            # 定义初始分类头数
            # self.model.cls_decoder.out_layer.out_features = len(self.old_model_annotation)
            if self.config["replay"]:
                example_bank_previous = sc.AnnData()
        else:

            
            # 增量批次
            new_model_annotation = current_label_dict

            # 找出新增细胞类型
            new_to_add = [ct for ct in new_model_annotation if ct not in self.old_model_annotation]

            # 生成完整细胞类型列表
            combined = np.concatenate([self.old_model_annotation, new_to_add])
            self.old_model_annotation = combined

            # 更新映射表：celltype -> label (保持一致编号)
            celltype_to_label = {ct: i for i, ct in enumerate(self.old_model_annotation)}
            with open(save_dir / f"celltype_to_label.json", 'w') as file:
                json.dump(celltype_to_label, file)
            # 重新赋值 label，确保和旧数据一致
            adata_train.obs["celltype_labels"] = adata_train.obs["celltype"].map(celltype_to_label)
            celltypes_labels = adata_train.obs["celltype_labels"].to_numpy()
            # 更新分类头数
            # self.model.cls_decoder.out_layer.out_features = len(self.old_model_annotation)
            if self.config["replay"]:
                example_bank_previous = self.example_bank['example_bank']                              # 之前储存的样本
                adata_train = anndata.concat([adata_train, example_bank_previous], axis=0, join='outer', label=None, keys=None)
                example_bank_previous_label = [list(new_model_annotation).index(i) for i in np.array(example_bank_previous.obs['celltype'])]
                celltypes_labels = np.concatenate([celltypes_labels, np.array(example_bank_previous_label)], 0)
            

       ############################# 再搞定模型 ###############################
        old_out_layer = self.model.cls_decoder.out_layer
        old_out_features = old_out_layer.out_features
        in_features = old_out_layer.in_features
        new_out_features = len(self.old_model_annotation)
        if new_out_features > old_out_features:
            # 新建更大的 out_layer
            new_out_layer = nn.Linear(in_features, new_out_features).to(self.device)
            
            # 迁移已有权重
            with torch.no_grad():
                new_out_layer.weight[:old_out_features] = old_out_layer.weight
                new_out_layer.bias[:old_out_features] = old_out_layer.bias
            
            # 替换分类头
            self.model.cls_decoder.out_layer = new_out_layer

        if self.config["init_weight"]:
            new_out_layer = nn.Linear(in_features, self.config["init_class"]).to(self.device)   # 每个batch重新初始化
            self.model.cls_decoder.out_layer = new_out_layer

        # 收集所有已在 optimizer 中的参数的 data_ptr
        optimizer_param_ids = set(p.data_ptr() for group in self.optimizer.param_groups for p in group['params'])

        # 收集 cls_decoder.out_layer 的参数
        missing_params = []
        for name, param in self.model.cls_decoder.out_layer.named_parameters():
            if param.requires_grad and param.data_ptr() not in optimizer_param_ids:
                print(f"Param '{name}' not in optimizer. Will add.")
                missing_params.append(param)

        # 如果有缺失的参数，就添加进 optimizer
        if missing_params:
            self.optimizer.add_param_group({'params': missing_params})
            print(f"✅ Added {len(missing_params)} missing parameters to optimizer.")
        ############################ past_model ################################
        # if self.past_model is not None:
        #     old_out_layer = self.past_model.cls_decoder.out_layer
        #     old_out_features = old_out_layer.out_features
        #     in_features = old_out_layer.in_features
        #     new_out_features = len(self.old_model_annotation)
        #     if new_out_features > old_out_features:
        #         # 新建更大的 out_layer
        #         new_out_layer = nn.Linear(in_features, new_out_features).to(self.device)
                
        #         # 迁移已有权重
        #         with torch.no_grad():
        #             new_out_layer.weight[:old_out_features] = old_out_layer.weight
        #             new_out_layer.bias[:old_out_features] = old_out_layer.bias
                
        #         # 替换分类头
        #         self.past_model.cls_decoder.out_layer = new_out_layer
        #############################################
        batch_ids = adata_train.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        if self.config["randomsplit"]:
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
            # 首先划分索引
            adata_indices = np.arange(len(adata_train))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=0.1,
                shuffle=True,
                stratify=celltypes_labels if not self.config["randomsplit"] else None,
            )

            # 根据索引子集化 adata_train
            adata_train_split = adata_train[train_idx].copy()
            adata_valid_split = adata_train[valid_idx].copy()

        elif self.config["fewshot"] is not None:
            (
                train_data, 
                valid_data, 
                train_celltype_labels, 
                valid_celltype_labels, 
                train_batch_labels, 
                valid_batch_labels,
                train_indices, 
                val_indices
             ) = self.build_fewshot_dataset(all_counts, celltypes_labels, batch_ids, shots_per_class=self.config["fewshot"])
            adata_train_split = adata_train[train_indices].copy()
            adata_valid_split = adata_train[val_indices].copy()
            
        else:
            (
                train_data,
                valid_data,
                train_celltype_labels,
                valid_celltype_labels,
                train_batch_labels,
                valid_batch_labels,
            ) = train_test_split(
                all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True, stratify=celltypes_labels
            )

        # 从过去的 batch 数据中采样
        # if self.past_data is not None:
        #     kmeans = KMeans(n_clusters=self.config["k_samples"])
        #     kmeans.fit(self.past_data)
        #     sampled_indices = kmeans.cluster_centers_indices_
        #     sampled_past_data = self.past_data[sampled_indices]
        #     train_data = np.concatenate((train_data, sampled_past_data), axis=0)
        #     train_batch_labels = np.concatenate((train_batch_labels, np.full(len(sampled_past_data), -1)), axis=0)

        gene_ids = np.array(self.vocab(genes), dtype=int)

        tokenized_train = tokenize_and_pad_batch(
            train_data,
            gene_ids,
            max_len=3001,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        tokenized_valid = tokenize_and_pad_batch(
            valid_data,
            gene_ids,
            max_len=3001,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )

        logger.info(
            f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
        )
        logger.info(
            f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
        )

        best_val_loss = float("inf")
        best_model = None
        embedding_list = []
        label_list = []
        train_epoch_loss = []
        eval_epoch_loss = []
        
        ######################### 设置学习率 ##########################
        
        if self.config["schedule"] == "cosine_schedule_with_warmup":
            if self.config["fewshot"] is not None:
                warmup_steps = 10
                
            else:
                warmup_steps = 2
            early_stopper = EarlyStopping(patience=2, min_delta=1e-4, mode='min')
            total_steps = self.config["epochs"] * (train_data.shape[0] // self.config["batch_size"])
            del self.scheduler
            torch.cuda.empty_cache()
            gc.collect()
            self.scheduler = CosineScheduleWithWarmup(self.optimizer,
                        num_warmup_steps=warmup_steps,          # warmup 步数
                        num_training_steps=total_steps        # 总训练步数
                    )
        elif self.config["schedule"] == "stepLR":
            del self.scheduler
            torch.cuda.empty_cache()
            gc.collect()
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 1, gamma=0.9
            )
        else:
            pass
        

        for epoch in range(1, self.config["epochs"] + 1):
            epoch_start_time = time.time()
            train_data_pt, valid_data_pt = prepare_data(
                sort_seq_batch=False,
                tokenized_train=tokenized_train,
                tokenized_valid=tokenized_valid,
                train_batch_labels=train_batch_labels,
                valid_batch_labels=valid_batch_labels,
                train_celltype_labels=train_celltype_labels,
                valid_celltype_labels=valid_celltype_labels,
                mask_ratio=self.config["mask_ratio"],
                mask_value=-1,
                pad_value=-2,
            )

            train_loader = prepare_dataloader(
                train_data_pt,
                batch_size=self.config["batch_size"],
                shuffle=False,
                intra_domain_shuffle=True,
                drop_last=False,
                per_seq_batch_sample=False
            )
            valid_loader = prepare_dataloader(
                valid_data_pt,
                batch_size=self.config["batch_size"],
                shuffle=False,
                intra_domain_shuffle=False,
                drop_last=False,
                per_seq_batch_sample=False
            )
            ########### 保存过去的验证集加载器 ############
            self.past_valid_loaders[test_batch_idx] = valid_loader

            if self.config["do_train"] and epoch < self.config["epochs"]:
                cell_emb_list, celltype_labels, train_loss, _  = self.train(
                    loader=train_loader,
                    logger=logger,
                    epoch=epoch,
                    test_batch_idx=test_batch_idx,
                    save_dir = save_dir
                )
            elif self.config["do_train"] and epoch == self.config["epochs"]:
                cell_emb_list, celltype_labels, train_loss, _ = self.train(
                    loader=train_loader,
                    logger=logger,
                    epoch=epoch,
                    test_batch_idx=test_batch_idx,
                    save_dir = save_dir
                )
                # self.update_classifier_weights(cell_emb_list, celltype_labels)
            train_epoch_loss.append(train_loss)

            if self.config["fewshot"] and epoch % 20 == 0: 
                print(f'epoch={epoch}', '############## start_eval ##############')        
                val_loss, val_err, result_dict = self.evaluate(
                    loader=valid_loader,
                    epoch=epoch,
                    save_dir = save_dir,
                    test_batch_idx=test_batch_idx,
                
                )
                # early_stopper(val_loss, self.model)
                # if early_stopper.early_stop:
                #     print(f"Early stopping at epoch {epoch} on task {test_batch_idx}")
                #     break

                eval_epoch_loss.append(val_loss)
                elapsed = time.time() - epoch_start_time
                logger.info("-" * 89)
                logger.info(
                    f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
                    f"valid loss/mse {val_loss:5.4f} | err {val_err:5.4f}"
                )
                logger.info("-" * 89)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(self.model)
                    logger.info(f"Best model with score {best_val_loss:5.4f}")
                    if test_batch_idx == self.max_test_id:
                        torch.save(best_model.state_dict(), save_dir / f"best_model_batch_{test_batch_idx}.pt")
                        with open(str(save_dir) + "/" + f"batch_{test_batch_idx}_besteval_results.json", "w") as f:
                            json.dump(result_dict, f, indent=4)

                    # ######################### compute previous sample performance #########################
                    if len(self.past_valid_loaders) >= 1:
                        past_items = list(self.past_valid_loaders.items())  # 取前 N-1 个 (key, loader) 对
                        past_result_dict = {}
                        for i, (task_id, loader) in enumerate(past_items):
                            num_batches = len(loader)
                            accuracy, total_num = 0, 0
                            predictions, true_labels = [], []
                            with torch.no_grad():
                                for batch, batch_data in enumerate(itertools.islice(loader, num_batches)):
                                    input_gene_ids = batch_data["gene_ids"].to(self.device)
                                    input_values = batch_data["values"].to(self.device)
                                    target_values = batch_data["target_values"].to(self.device)
                                    batch_labels = batch_data["batch_labels"].to(self.device)
                                    celltype_labels = batch_data["celltype_labels"].to(self.device)
                                    src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

                                    with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                                        if self.config["adapter"] or self.config["loramoe"]:
                                            output_dict, _ = best_model(
                                                input_gene_ids,
                                                input_values,
                                                src_key_padding_mask=src_key_padding_mask,
                                                batch_labels=batch_labels if False or self.config["DSBN"] else None,
                                                # batch_id = torch.tensor(test_batch_idx),
                                                batch_id = None,
                                                CLS=True,
                                                CCE=False,
                                                MVC=False,
                                                ECS=False,
                                                do_sample=False,
                                            )
                                        else:
                                            output_dict = best_model(
                                                input_gene_ids,
                                                input_values,
                                                src_key_padding_mask=src_key_padding_mask,
                                                batch_labels=batch_labels if False or self.config["DSBN"] else None,
                                                CLS=True,
                                                CCE=False,
                                                MVC=False,
                                                ECS=False,
                                                do_sample=False,
                                            )
                                        output_values = output_dict["cls_output"]
                                    accuracy += (output_values.argmax(1) == celltype_labels).sum().item()
                                    preds = output_values.argmax(1).detach().cpu().numpy()
                                    predictions.extend(preds)
                                    true_labels.extend(celltype_labels.detach().cpu().numpy())
                                    total_num += len(input_gene_ids)

                            f1_macro = f1_score(true_labels, predictions, average="macro")
                            f1_micro = f1_score(true_labels, predictions, average="micro")
                            f1_weighted = f1_score(true_labels, predictions, average="weighted")

                            # 保存
                            past_result_dict[i] = {
                                "accuracy": accuracy / total_num,
                                "f1_macro": f1_macro,
                                "f1_micro": f1_micro,
                                "f1_weighted": f1_weighted,
                                "preds": [int(p) for p in predictions],
                                "total_num": total_num
                            }

                    with open(str(save_dir) + "/" + f"test_after_batch_{test_batch_idx}" + ".json", "w") as f:
                        json.dump(past_result_dict, f, indent=4)
                    early_stopper(val_loss, self.model)
                    if early_stopper.early_stop:
                        print(f"Early stopping at epoch {epoch} on task {test_batch_idx}")
                        break
            else:
                pass
            if self.config["schedule"] == "stepLR":
                self.scheduler.step()
            gc.collect()
            torch.cuda.empty_cache()
            del train_data_pt, valid_data_pt, train_loader, valid_loader
        if test_batch_idx == self.max_test_id:
            torch.save(self.model.state_dict(), save_dir / f"last_model_batch_{test_batch_idx}.pt")

        # all_adata_test.append(adata_train)
        all_adata_test.append(adata_valid_split)

        combined_adata_test = anndata.concat(
            all_adata_test,
            join="outer",
            merge="unique",
            label="batch",
            index_unique=None
        )
        combined_adata_test.obs["batch_id"] = le.fit_transform(combined_adata_test.obs["batch_id"])

        # for batch_i in range(len(all_adata_test)):
        #     this_batch_adata = all_adata_test[batch_i]
        #     this_batch_adata.obs["batch_id"] = le.fit_transform(this_batch_adata.obs["batch_id"])

        #     results_batch_i, _ = self.eval_testdata(
        #         adata_t=this_batch_adata,
        #         gene_ids=gene_ids,
        #         input_layer_key=input_layer_key,
        #         logger=logger
        #     )

        #     all_batch_results[f"batch_{batch_i}_after_batch_{test_batch_idx}"] = results_batch_i

        # with open(save_dir / f"{dataset_name}_{experiment_name}_after_batch_{test_batch_idx}_per_batch_results.json",
        #           "w") as f:
        #     json.dump(all_batch_results, f, indent=4)
        ##################### 用eval最好的模型用于聚类效果的评估和下一个batch模型的初始化 ######################
        if self.config["use_best_initnextbatch"]:
            load_pretrained(self.model, torch.load(save_dir / f"best_model_batch_{test_batch_idx}.pt"), verbose=False)

        # results, adata_sorted = self.eval_testdata(
        #     adata_t=combined_adata_test,
        #     gene_ids=gene_ids,
        #     input_layer_key=input_layer_key,
        #     # logger=logger
        # )

        # sc.pp.neighbors(adata_sorted, use_rep="X_scEvolver")
        # sc.tl.umap(adata_sorted, min_dist=0.3)
        # sc.pl.umap(
        #     adata_sorted,
        #     color=["str_batch"],
        #     title=[f"batch, avg_batch = {results.get('avg_batch', 0.0):.4f}"],
        #     frameon=False,
        #     show=False,
        # )
        # plt.savefig(
        #     str(save_dir) + "/" + dataset_name + "_" + experiment_name + "_" + f"embeddings_batch_umap[cls]_batch_{test_batch_idx}.png",
        #     bbox_inches='tight')

        # sc.pp.neighbors(adata_sorted, use_rep="X_scEvolver")
        # sc.tl.umap(adata_sorted, min_dist=0.3)
        # sc.pl.umap(
        #     adata_sorted,
        #     color=["celltype"],
        #     title=[f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
        #     frameon=False,
        #     show=False,
        # )
        # plt.savefig(
        #     str(save_dir) + "/" + dataset_name + "_" + experiment_name + "_" + f"embeddings_celltype_umap[cls]_batch_{test_batch_idx}.png",
        #     bbox_inches='tight')

        # with open(str(save_dir) + "/" + dataset_name + "_" + experiment_name + f"_batch_{test_batch_idx}_final_results.json",
        #           "w") as f:
        #     json.dump(results, f, indent=4)

        # 更新过去的数据和模型
        self.past_data = all_counts

        # if self.config["pastmse"]:
        #     self.past_model = copy.deepcopy(best_model)
        #     for p in self.past_model.parameters():
        #         p.requires_grad = False

        # if test_batch_idx == self.max_test_id:
            # self.plot_clusters_prototypes(self, combined_adata_test, self.model.cls_decoder.out_layer.weight.data, input_layer_key, gene_ids, save_dir)
        self.plot_train_val_loss(train_epoch_loss, eval_epoch_loss, save_dir, test_batch_idx)
        if self.config["repultion_loss"]:
            self.plot_proto_repultion_loss(self.contrastive_proto_loss_list, self.repultion_loss_list, save_dir, test_batch_idx)
        return best_model, combined_adata_test, gene_ids
    
    def plot_clusters_prototypes(self, adata, prototype, input_layer_key, gene_ids, save_dir):
    # prototypes = F.normalize(prototype.prototypes, dim=1)
    # # prototypes = prototype.prototypes
    # # Step 1: 获取细胞和原型嵌入
    # X_cells = adata_sorted.obsm["X_scEvolver"]  # 细胞嵌入向量，shape = [n_cells, 1200]
    # n_cells = X_cells.shape[0]
        import umap
        self.model.eval()
        if self.past_model is not None:
            self.past_model.eval()
        all_counts = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=3001,                           # 应该是1201
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        batch_ids = adata.obs["batch_id"].to_numpy()
        batch_ids = np.zeros_like(batch_ids)                    # 在实际训练中只训练了一个BN层，因此batch_ids都要设置为0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):
            if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                cell_embeddings = self.model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels=torch.from_numpy(batch_ids).long() if self.config["DSBN"] else None,
                    # batch_id = torch.tensor(test_batch_idx),
                    # batch_id = torch.tensor(5),
                    batch_id=None,
                    time_step=0,
                    return_np=True,
                )
            else:
                cell_embeddings = self.model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels=torch.from_numpy(batch_ids).long() if self.config["DSBN"] else None,
                    time_step=0,
                    return_np=True,
                )
            # if self.past_model is not None:
            #     cell_embeddings_past = self.past_model.encode_batch(
            #         all_gene_ids,
            #         all_values.float(),
            #         src_key_padding_mask=src_key_padding_mask,
            #         batch_size=self.config["batch_size"],
            #         # batch_labels=torch.from_numpy(adata.obs["batch_id"].to_numpy()).long() if DSBN else None,
            #         batch_labels = torch.from_numpy(batch_ids).long() if self.config["DSBN"] else None,
            #         time_step=0,
            #         return_np=True,
            #     )
            #     cell_embeddings = np.concatenate([cell_embeddings_past, cell_embeddings], axis=-1)
        n_cells = cell_embeddings.shape[0]
        cell_embeddings = torch.tensor(cell_embeddings, device=self.device)
        cell_embeddings = F.normalize(cell_embeddings, dim=1)
        X_cells = cell_embeddings
        prototypes = F.normalize(prototype, dim=1)
        # 假设你已有 prototypes（形状为 [n_prototypes, 1200]）
        # 若是 tensor，转为 numpy
        prototypes_np = prototypes.detach().cpu().numpy() if torch.is_tensor(prototypes) else prototypes
        n_prototypes = prototypes_np.shape[0]

        # Step 2: 合并所有向量
        X_all = np.concatenate([X_cells.detach().cpu().numpy(), prototypes_np], axis=0)

        # Step 3: 降维
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]

        # Step 4: 用真实标签绘图
        plt.figure(figsize=(9, 7))

        # 获取真实标签（必须存在于 .obs 中）
        labels = adata.obs["celltype"]

        # 用 seaborn 画 UMAP + 标签着色
        sns.scatterplot(
            x=X_umap[:n_cells, 0],
            y=X_umap[:n_cells, 1],
            hue=labels,
            palette="tab20",
            s=5,
            linewidth=0
        )

        # 添加原型
        plt.scatter(
            X_umap[n_cells:, 0],
            X_umap[n_cells:, 1],
            edgecolors='black',     # 边框颜色为黑色
            facecolors='none',      # 中空
            s=80,
            marker='X',
            label='Prototypes'
        )
        plt.title("UMAP of Cells Colored by True Labels with Prototypes")
        plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.tight_layout()
        plt.savefig(str(save_dir) + "/" + "umap_cluster_prototype.png", bbox_inches='tight')
    
    def predict(self, adata_test, save_dir, test_batch_idx, cell_type_map):
        self.config["weight_dir"] = save_dir
        le = LabelEncoder()
        adata_test.obs["batch_id"] = le.fit_transform(adata_test.obs["batch_id"])
        num_batch_types = adata_test.obs["batch_id"].nunique()
        input_layer_key = "X_binned"
        all_counts = (
            adata_test.layers[input_layer_key].A
            if issparse(adata_test.layers[input_layer_key])
            else adata_test.layers[input_layer_key]
        )
        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        # 计算当前批次 (test_batch_idx) 的细胞类型和标签
        # current_label_dict, current_celltype_labels = np.unique(
        #     np.array(adata_test.obs["celltype"].tolist()), return_inverse=True
        # )
        celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels                   # 先临时保存一下
        batch_ids = adata_test.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        # (
        #     train_data,
        #     valid_data,
        #     train_celltype_labels,
        #     valid_celltype_labels,
        #     train_batch_labels,
        #     valid_batch_labels,
        # ) = train_test_split(
        #     all_counts, current_celltype_labels, batch_ids, test_size=0.0, shuffle=True
        # )
        tokenized_test = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=3001,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        test_data_pt = prepare_testdata(
            sort_seq_batch=False,
            tokenized_test=tokenized_test,
            test_batch_labels=batch_ids,
            test_celltype_labels=current_celltype_labels,
            mask_ratio=self.config["mask_ratio"],
            mask_value=-1,
            pad_value=-2,
        )

        test_loader = prepare_dataloader(
            test_data_pt,
            batch_size=self.config["batch_size"],
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
            per_seq_batch_sample=False
        )
        self.model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        accuracy = 0
        predictions, labellist = [], []
        num_batches = len(test_loader)
        with torch.no_grad():
            for batch, batch_data in enumerate(itertools.islice(test_loader, num_batches)):
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                batch_labels = batch_data["batch_labels"].to(self.device)
                celltype_labels = batch_data["celltype_labels"].to(self.device)
                src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

                with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                    if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                        output_dict, _ = self.model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=batch_labels if False or self.config["DSBN"] else None,
                            # batch_id = torch.tensor(test_batch_idx),
                            batch_id = None,
                            CLS=True,
                            CCE=False,
                            MVC=False,
                            ECS=False,
                            do_sample=False,
                        )
                    else:
                        output_dict = self.model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=batch_labels if False or self.config["DSBN"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=False,
                            ECS=False,
                            do_sample=False,
                        )

                    output_values = output_dict["cls_output"]
                    loss = self.criterion_cls(output_values, celltype_labels)

                # total_loss += loss.item() * len(input_gene_ids)
                accuracy += (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).detach().cpu().numpy()
                predictions.extend(preds)
                labellist.extend(celltype_labels.detach().cpu().numpy())
            f1_macro = f1_score(labellist, predictions, average="macro")
            f1_micro = f1_score(labellist, predictions, average="micro")
            f1_weighted = f1_score(labellist, predictions, average="weighted")

            # 保存
            result_dict = {
                "accuracy": accuracy / total_num,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,
                "preds": [int(p) for p in predictions],
                "labels": [int(p) for p in labellist],
                "total_num": total_num
            }

            with open(str(save_dir) + "/" + f"predict_test_batch_{test_batch_idx}.json", "w") as f:
                json.dump(result_dict, f)
                
            ############### 画混淆矩阵 ###################
            class_names = np.unique(np.concatenate((labellist, predictions)))  # 获取所有类别
            cm = confusion_matrix(labellist, predictions, labels=class_names)

            # 绘制热力图

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(str(save_dir) + "/" + f"predict_test_batch_{test_batch_idx}_Confusion_Matrix.png")
            
            self.plot_clusters_prototypes(adata_test, self.model.cls_decoder.out_layer.weight.data, input_layer_key, gene_ids, save_dir)
        return gene_ids
    
    def plot_train_val_loss(self, train_loss_list_all, val_loss_list_all, save_dir, test_batch_idx):
        epochs_train = range(1, len(train_loss_list_all) + 1)
        epochs_val = range(1, len(val_loss_list_all) + 1)

        plt.figure(figsize=(10, 8))

        # 子图1：训练损失
        plt.subplot(2, 1, 1)
        plt.plot(epochs_train, train_loss_list_all, label='Train Loss', marker='o', color='blue')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.tight_layout()

        # 子图2：验证损失
        plt.subplot(2, 1, 2)
        plt.plot(epochs_val, val_loss_list_all, label='Validation Loss', marker='s', color='orange')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.grid(True)
        plt.tight_layout()

        # 保存图像
        plt.savefig(save_dir / f"train_val_loss_batch{test_batch_idx}.png")
    def plot_proto_repultion_loss(self, contrastive_proto_loss_list, repultion_loss_list, save_dir, test_batch_idx):

        proto_loss = range(1, len(contrastive_proto_loss_list) + 1)
        repultion_loss = range(1, len(repultion_loss_list) + 1)

        plt.figure(figsize=(10, 8))

        # 子图1：训练损失
        plt.subplot(2, 1, 1)
        plt.plot(proto_loss, contrastive_proto_loss_list, label='Proto Loss', marker='o', color='blue')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Proto Loss")
        plt.grid(True)
        plt.tight_layout()

        # 子图2：验证损失
        plt.subplot(2, 1, 2)
        plt.plot(repultion_loss, repultion_loss_list, label='Repultion Loss', marker='s', color='orange')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Repultion Loss")
        plt.grid(True)
        plt.tight_layout()

        # 保存图像
        plt.savefig(save_dir / f"Proto_Repultion_loss_batch{test_batch_idx}.png")


    def evaluate_all(self, adata_train, save_dir, test_batch_idx, cell_type_map):
        if self.config["init_class"] == 7 or self.config["filter_sample"]: 
            celltype_counts = adata_train.obs["celltype"].value_counts()
            valid_celltypes = celltype_counts[celltype_counts >= self.config["fewshot"] + 1].index
            adata_train = adata_train[adata_train.obs["celltype"].isin(valid_celltypes)].copy()

        input_layer_key = "X_binned"
        all_counts = (
            adata_train.layers[input_layer_key].A
            if issparse(adata_train.layers[input_layer_key])
            else adata_train.layers[input_layer_key]
        )
        genes = adata_train.var["gene_name"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        celltype_str_list = np.array(adata_train.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_train.obs["celltype_labels"] = current_celltype_labels                   # 用最终的label修改现在的label编号
        batch_ids = adata_train.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        if self.config["randomsplit"]:
            (
                train_data,
                valid_data,
                train_celltype_labels,
                valid_celltype_labels,
                train_batch_labels,
                valid_batch_labels,
            ) = train_test_split(
                all_counts, current_celltype_labels, batch_ids, test_size=0.1, shuffle=True
            )
            # 首先划分索引
            adata_indices = np.arange(len(adata_train))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=0.1,
                shuffle=True,
                stratify=current_celltype_labels if not self.config["randomsplit"] else None,
            )

            # 根据索引子集化 adata_train
            adata_train_split = adata_train[train_idx].copy()
            adata_valid_split = adata_train[valid_idx].copy()
        elif self.config["fewshot"] is not None:
            (
                train_data, 
                valid_data, 
                train_celltype_labels, 
                valid_celltype_labels, 
                train_batch_labels, 
                valid_batch_labels,
                train_indices, 
                val_indices
             ) = self.build_fewshot_dataset(all_counts, current_celltype_labels, batch_ids, shots_per_class=self.config["fewshot"], seed = 42)
            adata_train_split = adata_train[train_indices].copy()
            adata_valid_split = adata_train[val_indices].copy()     

        else:
            (
                train_data,
                valid_data,
                train_celltype_labels,
                valid_celltype_labels,
                train_batch_labels,
                valid_batch_labels,
            ) = train_test_split(
                all_counts, current_celltype_labels, batch_ids, test_size=0.1, shuffle=True, stratify=current_celltype_labels
            )
        
        return adata_valid_split, gene_ids

    def evaluate_predict(self, adata_test, save_dir, test_batch_idx, cell_type_map):
        self.config["weight_dir"] = save_dir
                # 统计每种细胞类型的数量
        # celltype_counts = adata_test.obs["celltype"].value_counts()
        # # 找出数量大于等于 4 的细胞类型
        # valid_celltypes = celltype_counts[celltype_counts >= 2].index

        # # 过滤掉数量小于 4 的细胞类型对应的样本
        # adata_test = adata_test[adata_test.obs["celltype"].isin(valid_celltypes)].copy()
        le = LabelEncoder()
        adata_test.obs["batch_id"] = le.fit_transform(adata_test.obs["batch_id"])
        num_batch_types = adata_test.obs["batch_id"].nunique()
        input_layer_key = "X_binned"
        all_counts = (
            adata_test.layers[input_layer_key].A
            if issparse(adata_test.layers[input_layer_key])
            else adata_test.layers[input_layer_key]
        )
        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        # 计算当前批次 (test_batch_idx) 的细胞类型和标签
        # current_label_dict, current_celltype_labels = np.unique(
        #     np.array(adata_test.obs["celltype"].tolist()), return_inverse=True
        # )
        celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels                   # 先临时保存一下
        batch_ids = adata_test.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        if self.config["randomsplit"]:
            (
                train_data,
                valid_data,
                train_celltype_labels,
                valid_celltype_labels,
                train_batch_labels,
                valid_batch_labels,
            ) = train_test_split(
                all_counts, current_celltype_labels, batch_ids, test_size=0.1, shuffle=True
            )
            # 首先划分索引
            adata_indices = np.arange(len(adata_test))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=0.1,
                shuffle=True,
                stratify=current_celltype_labels if not self.config["randomsplit"] else None,
            )

            # 根据索引子集化 adata_train
            adata_train_split = adata_test[train_idx].copy()
            adata_valid_split = adata_test[valid_idx].copy()
        else:
            (
                train_data,
                valid_data,
                train_celltype_labels,
                valid_celltype_labels,
                train_batch_labels,
                valid_batch_labels,
            ) = train_test_split(
                all_counts, current_celltype_labels, batch_ids, test_size=0.1, shuffle=True, stratify=current_celltype_labels
            )
        tokenized_test = tokenize_and_pad_batch(
            valid_data,
            gene_ids,
            max_len=3001,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        test_data_pt = prepare_testdata(
            sort_seq_batch=False,
            tokenized_test=tokenized_test,
            test_batch_labels=valid_batch_labels,
            test_celltype_labels=valid_celltype_labels,
            mask_ratio=self.config["mask_ratio"],
            mask_value=-1,
            pad_value=-2,
        )

        test_loader = prepare_dataloader(
            test_data_pt,
            batch_size=self.config["batch_size"],
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
            per_seq_batch_sample=False
        )
        self.model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        accuracy = 0
        predictions, labellist = [], []
        num_batches = len(test_loader)
        with torch.no_grad():
            for batch, batch_data in enumerate(itertools.islice(test_loader, num_batches)):
                input_gene_ids = batch_data["gene_ids"].to(self.device)
                input_values = batch_data["values"].to(self.device)
                target_values = batch_data["target_values"].to(self.device)
                batch_labels = batch_data["batch_labels"].to(self.device)
                celltype_labels = batch_data["celltype_labels"].to(self.device)
                src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

                with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                    output_dict = self.model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if False or self.config["DSBN"] else None,
                        CLS=True,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=False,
                    )
                    output_values = output_dict["cls_output"]
                    loss = self.criterion_cls(output_values, celltype_labels)

                # total_loss += loss.item() * len(input_gene_ids)
                accuracy += (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).detach().cpu().numpy()
                predictions.extend(preds)
                labellist.extend(celltype_labels.detach().cpu().numpy())
            f1 = f1_score(np.array(labellist), np.array(predictions), average='macro')       # 宏平均
            f1_micro = f1_score(np.array(labellist), np.array(predictions), average='micro') # 微平均
            f1_weighted = f1_score(np.array(labellist), np.array(predictions), average='weighted') # 加权平均
            result_dict = {
                "num_samples":total_num,
                "accuracy": accuracy / total_num,
                "f1":f1,
                "f1_micro":f1_micro,
                "f1_weighted":f1_weighted,
                "preds":  [int(p) for p in predictions],  # 转成 list
                "labels": [int(p) for p in labellist],
                }
            with open(str(save_dir) + "/" + f"evaluate_batch_{test_batch_idx}_results.json", "w") as f:
                json.dump(result_dict, f)
                
            ############### 画混淆矩阵 ###################
            class_names = np.unique(np.concatenate((labellist, predictions)))  # 获取所有类别
            cm = confusion_matrix(labellist, predictions, labels=class_names)

            # 绘制热力图

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.savefig(str(save_dir) + "/" + f"evaluate_batch_{test_batch_idx}_Confusion_Matrix.png")
            # self.plot_clusters_prototypes(adata_test, self.model.cls_decoder.out_layer.weight.data, input_layer_key, gene_ids, save_dir)
        return adata_valid_split, gene_ids
    
def evaluate_predict(save_dir, cell_type_map):
    config = init_wandb()
    set_seed(config["seed"])
    if config["dataset_name"] == "pancreas":
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6)]  # 剩下三个batch作为测试集

    model_dir = Path(config["load_model"])
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    num_batch_types = 1
    config["weight_dir"] = save_dir
    config["init_class"] = 13
    continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_5.pt")

    all_adata_list = []
    all_batch_results = {}

    for test_batch_idx in range(len(data_paths)):
        adata_test = sc.read_h5ad(data_paths[test_batch_idx])
        if config["init_class"] == 7 or config["filter_sample"]: 
            celltype_counts = adata_test.obs["celltype"].value_counts()
            valid_celltypes = celltype_counts[celltype_counts >= config["fewshot"] + 1].index
            adata_test = adata_test[adata_test.obs["celltype"].isin(valid_celltypes)].copy()
        all_adata_test, gene_ids = continual_classify.evaluate_predict(
            adata_test, save_dir, test_batch_idx, cell_type_map
        )
        # all_adata_test, gene_ids = continual_classify.evaluate_all(adata_test, save_dir, test_batch_idx, cell_type_map)
        all_adata_list.append(all_adata_test)

    combined_adata_test = anndata.concat(
        all_adata_list,
        join="outer",
        merge="unique",
        label="batch",
        index_unique=None
    )
    print(combined_adata_test.X.shape)
    continual_classify.predict(combined_adata_test, save_dir, 12345, cell_type_map)
    # final_results, final_adata = continual_classify.eval_testdata(
    #     adata_t=combined_adata_test,
    #     gene_ids=gene_ids,
    #     input_layer_key={
    #         "normed_raw": "X_normed",
    #         "log1p": "X_normed",
    #         "binned": "X_binned",
    #     }["binned"],
    # )

    # with open(str(save_dir) + "/final_evaluate_results.json", "w") as f:
    #     json.dump(final_results, f, indent=4)

    # sc.pp.neighbors(final_adata, use_rep="X_scEvolver")
    # sc.tl.umap(final_adata, min_dist=0.3)
    # sc.pl.umap(
    #     final_adata,
    #     color=["str_batch"],
    #     title=[f"batch, avg_batch = {final_results.get('avg_batch', 0.0):.4f}"],
    #     frameon=False,
    #     show=False,
    # )
    # plt.savefig(
    #     str(save_dir) + "/" + "embeddings_batch_umap[cls]_batch_eval_all.png",
    #     bbox_inches='tight')

    # sc.pp.neighbors(final_adata, use_rep="X_scEvolver")
    # sc.tl.umap(final_adata, min_dist=0.3)
    # sc.pl.umap(
    #     final_adata,
    #     color=["celltype"],
    #     title=[f"celltype, avg_bio = {final_results.get('avg_bio', 0.0):.4f}"],
    #     frameon=False,
    #     show=False,
    # )
    # plt.savefig(
    #     str(save_dir) + "/" + "embeddings_celltype_umap[cls]_batch_eval_all.png",
    #     bbox_inches='tight')
    
def main():
    config = init_wandb()
    set_seed(config["seed"])

    save_dir = Path(f"./save/dev_{config['dataset_name']}-{time.strftime('%b%d-%H-%M-%S')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")

    # adata = load_data(config)
    with open(save_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    if config["dataset_name"] == "PBMC_10K":
        data_paths = ["../data/pbmc/" + f"pbmc_batch{i}.h5ad" for i in range(2)]
    elif config["dataset_name"] == "pancreas":
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6)]  # 剩下三个batch作为测试集
        
    elif config["dataset_name"] == "myeloid":
        data_paths = [f"./data/myeloid/myeloid_batch{i}.h5ad" for i in [1, 2, 5, 6, 7]]
    elif config["dataset_name"] == "spleen":
        data_paths = ["../data/spleen/spleen_processed.h5ad"]

    model_dir = Path(config["load_model"])
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    num_batch_types = 1
    max_batch_idx = len(data_paths)-1
    continual_classify = ContinualClassify(config, vocab, num_batch_types, max_batch_idx)

    all_adata_test = []
    all_batch_results = {}
    # for index in config["data_seed"]:
    for test_batch_idx in range(len(data_paths)):
        adata_train = sc.read_h5ad(data_paths[test_batch_idx])
        best_model, combined_adata_test, gene_ids = continual_classify.process_batch(
            adata_train, logger, save_dir, config["dataset_name"], config["experiment_name"], all_adata_test, all_batch_results,
            test_batch_idx
        )
        besteval_results, besteval_adata = continual_classify.best_model_evaluate(best_model, adata_t=combined_adata_test, gene_ids=gene_ids,
                                                input_layer_key={
                                                    "normed_raw": "X_normed",
                                                    "log1p": "X_normed",
                                                    "binned": "X_binned",
                                                }["binned"],test_batch_idx=test_batch_idx)
        with open(str(save_dir) + "/" + f"batch_{test_batch_idx}_bestval_results.json",
                "w") as f:
            json.dump(besteval_results, f, indent=4)

        sc.pp.neighbors(besteval_adata, use_rep="X_scEvolver")
        sc.tl.umap(besteval_adata, min_dist=0.3)
        sc.pl.umap(
            besteval_adata,
            color=["str_batch"],
            title=[f"batch, avg_batch = {besteval_results.get('avg_batch', 0.0):.4f}"],
            frameon=False,
            show=False,
        )
        plt.savefig(
            str(save_dir) + "/" + f"embeddings_batch_umap[cls]_batch_{test_batch_idx}.png",
            bbox_inches='tight')

        sc.pp.neighbors(besteval_adata, use_rep="X_scEvolver")
        sc.tl.umap(besteval_adata, min_dist=0.3)
        sc.pl.umap(
            besteval_adata,
            color=["celltype"],
            title=[f"celltype, avg_bio = {besteval_results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            show=False,
        )
        plt.savefig(
            str(save_dir) + "/" + f"embeddings_celltype_umap[cls]_batch_{test_batch_idx}.png",
            bbox_inches='tight')


    # final_results, final_adata = continual_classify.eval_testdata(
    #     adata_t=combined_adata_test,
    #     gene_ids=gene_ids,
    #     input_layer_key={
    #         "normed_raw": "X_normed",
    #         "log1p": "X_normed",
    #         "binned": "X_binned",
    #     }["binned"],
    # )

    # with open(str(save_dir) + "/" + config["dataset_name"] + "_" + config["experiment_name"] + "_final_results.json",
    #           "w") as f:
    #     json.dump(final_results, f, indent=4)
    
def predict(save_dir, cell_type_map, test_batch_list):
    config = init_wandb()
    set_seed(config["seed"])
    if config["dataset_name"] == "pancreas":
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6, 9)]  # 剩下三个batch作为测试集
    elif config["dataset_name"] == "myeloid":
        data_paths = [f"../data/myeloid/myeloid_batch{i}.h5ad" for i in test_batch_list]

    model_dir = Path(config["load_model"])
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    num_batch_types = 1
    config["weight_dir"] = save_dir
    config["init_class"] = len(cell_type_map)   # fewshot
    if config["dataset_name"] == "pancreas":
        continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_5.pt")
    elif config["dataset_name"] == "myeloid":
        continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_4.pt")     

    all_adata_test = []
    all_batch_results = {}

    for test_batch_idx in range(len(data_paths)):
        adata_test = sc.read_h5ad(data_paths[test_batch_idx])
        if config["init_class"] == 7 or config["filter_sample"]: 
            celltype_counts = adata_test.obs["celltype"].value_counts()
            valid_celltypes = celltype_counts[celltype_counts >= config["fewshot"] + 1].index
            adata_test = adata_test[adata_test.obs["celltype"].isin(valid_celltypes)].copy()
        # gene_ids = continual_classify.predict(
        #     adata_test, save_dir, test_batch_idx, cell_type_map
        # )
        all_adata_test.append(adata_test)

    combined_adata_test = anndata.concat(
        all_adata_test,
        join="outer",
        merge="unique",
        label="batch",
        index_unique=None
    )
    gene_ids = continual_classify.predict(
            combined_adata_test, save_dir, test_batch_idx + 1, cell_type_map
        )
    final_results, final_adata = continual_classify.eval_testdata(
        adata_t=combined_adata_test,
        gene_ids=gene_ids,
        input_layer_key={
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }["binned"],
    )

    with open(str(save_dir) + "/" + "predict_all_finaltest_results.json",
              "w") as f:
        json.dump(final_results, f, indent=4)
    
    sc.pp.neighbors(final_adata, use_rep="X_scEvolver")
    sc.tl.umap(final_adata, min_dist=0.3)
    sc.pl.umap(
        final_adata,
        color=["str_batch"],
        title=[f"batch, avg_batch = {final_results.get('avg_batch', 0.0):.4f}"],
        frameon=False,
        show=False,
    )
    plt.savefig(
        str(save_dir) + "/" + f"embeddings_batch_umap[cls]_batch_6_9.png",
        bbox_inches='tight')

    sc.pp.neighbors(final_adata, use_rep="X_scEvolver")
    sc.tl.umap(final_adata, min_dist=0.3)
    sc.pl.umap(
        final_adata,
        color=["celltype"],
        title=[f"celltype, avg_bio = {final_results.get('avg_bio', 0.0):.4f}"],
        frameon=False,
        show=False,
    )
    plt.savefig(
        str(save_dir) + "/" + f"embeddings_celltype_umap[cls]_batch_6_9.png",
        bbox_inches='tight')

if __name__ == "__main__":
    # main()

    # save_dir = "../save/dev_pancreas-May14-14-40-30" # baseline(no_schedular)
    # save_dir = "../save/dev_pancreas-May20-14-45-44"  # baseline(no_schedular) nlayers_cls=0
    # save_dir = "../save/dev_pancreas-May25-20-45-57"    # baseline (init_class14_nlayers_cls3)
    # save_dir = "../save/dev_pancreas-May26-21-56-55"    # init_class14_nlayers_cls3_freeze_all(except_layers1011)_adapter
    # save_dir = "../save/dev_pancreas-Jun27-22-20-58"    # baseline + schduler
    # save_dir = "../save/dev_pancreas-Jun27-22-20-22"      # baseline-online + freeze
    # save_dir = "../save/dev_pancreas-Jun29-10-25-48"        # baseline-online + not freeze
    # save_dir = "../save/dev_pancreas-Jul01-16-49-34"        # baseline-online+warmcosine + freeze
    # save_dir = "../save/dev_pancreas-Jul01-16-41-24"         # loramoe + warmcosine（expert=4)
    # save_dir = "../save/dev_pancreas-Jul01-11-35-54"    # prototype + loramoe + warmcosine（expert=4)
    # save_dir = "../save/dev_pancreas-Jul07-10-28-02"    # baseline
    # save_dir = "../save/dev_pancreas-Jul07-10-28-55"    # baseline + lora
    save_dir = "../save/dev_myeloid-Dec19-16-29-32"       # myeloid 
    with open(save_dir + "/celltype_to_label.json", "r") as f:
        cell_type_map = json.load(f)
        f.close()
    predict(save_dir, cell_type_map, test_batch_list=[0, 3, 4])
    # evaluate_predict(save_dir, cell_type_map)
