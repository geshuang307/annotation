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
from torch import Tensor
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
import pandas as pd
import matplotlib.pyplot as plt
import anndata
import scvi
import scanpy as sc
import json
import itertools
# from torch.nn import TransformerEncoder, TransformerEncoderLayer
import scgpt as scg
# from scgpt.model import TransformerModel, AdversarialDiscriminator
# from adaptermodel import TransformerModel                        # adapter
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
def init_wandb():
    hyperparameter_defaults = dict(
        seed=0,
        # dataset_name="ms",
        # dataset_name="pancreas",
        dataset_name = "myeloid",
        do_train=True,
        load_model="/workspace/geshuang/code/scGPT/save/scGPT_human",
        weight_dir="/workspace/geshuang/code/scGPT/save/scGPT_human",
        # experiment_name="fine_tune_on_pancreas_minibatch_baseline(uniformsplit_no_schedular_nlayers_cls0_pastmse[origion])",
        # experiment_name="fine_tune_on_pancreas_minibatch_baseline(uniformsplit_no_schedular_nlayers_cls0)",
        # experiment_name="fine_tune_on_pancreas_minibatch_baseline(randomsplit_no_schedular_nlayers_cls3_init_class13)",
        mask_ratio=0.0,
        epochs=15,                 # few-shot下为1000
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
        init_class = 14,
        filter_sample = False,
        randomsplit = True,
        fewshot = None,
        nlayers_cls=3,
        use_best_initnextbatch = False,
        adapter=False,
        freeze_except_layer1011 = False,
        freeze_all = True,
        loramoe = True,  
        proto_loss = True,
        repultion_loss = False,
        entropy = True,
        classifier = "Linear",
        anchorloss = False,
        schedule = None,               # "cosine_schedule_with_warmup",     # stepLR
        proto_weight = 1,
        cope_loss = False,
        weight_miloss = False
    )
    config = hyperparameter_defaults
    config["experiment_name"] = "fine_tune_on_pancreas_minibatch_prototype(" +\
          "_init_class" + str(config["init_class"]) +\
                  "_proto_loss_" + str(config["proto_loss"]) + \
                  "_proto_weight_" +str(config["proto_weight"]) +\
            "_classifier_" + str(config["classifier"]) +\
            "_loramoe_" + str(config["loramoe"]) +\
            "_freeze_all_" + str(config["freeze_all"]) +\
            "_entropy_" + str(config["entropy"]) +\
            "_schedule_" + str(config["schedule"]) 
            # "_miloss_" + str(config["weight_miloss"])
            # "_anchorloss_" + str(config["anchorloss"]) 
                # "_freeze_except_layer1011_" + str(config["freeze_except_layer1011"]) +\
    # "_entropy_" + str(config["entropy"])     # "_relpution_loss_" + str(config["repultion_loss"])  

    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    if config["dataset_name"] == "pancreas":
        config["init_class"] = 14
        config["lr"] = 1e-4
    elif config["dataset_name"] == "myeloid":
        config["init_class"] = 12
        config["lr"] = 1e-5
        config["epochs"] = 15
    return config

def check_out_layer_in_optimizer(model, optimizer):
    out_layer = model.cls_decoder.out_layer
    # 获取 optimizer 中所有参数的 data_ptr
    optimizer_param_ptrs = {p.data_ptr() for g in optimizer.param_groups for p in g['params']}

    # 检查哪些参数不在 optimizer 中
    missing = []
    for name, param in out_layer.named_parameters():
        if param.requires_grad and param.data_ptr() not in optimizer_param_ptrs:
            missing.append(name)

    if missing:
        print(f"❌ The following parameters in out_layer are NOT in the optimizer: {missing}")
    else:
        print("✅ All parameters in out_layer are included in the optimizer.")
    
    return missing
class AdvancedAnchorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, labels):
        """
        Args:
            features: Tensor [N, D] - feature embeddings
            labels:   Tensor [N]    - class labels
        Returns:
            Anchor loss (scalar)
        """
        device = features.device
        features = F.normalize(features, p=2, dim=1)  # L2 normalize all features
        unique_classes = torch.unique(labels)

        loss = 0.0
        total_samples = 0

        for cls in unique_classes:
            mask = labels == cls
            class_feats = features[mask]  # [Nc, D]

            if class_feats.size(0) < 2:
                continue  # skip single-sample class

            # (1) compute class center
            class_center = class_feats.mean(dim=0)
            class_center = F.normalize(class_center.unsqueeze(0), dim=1)  # [1, D]

            # (5) choose anchor (most similar to class center)
            sims = torch.matmul(class_feats, class_center.T).squeeze(1)  # cosine similarity
            anchor_idx = torch.argmax(sims)
            anchor = class_feats[anchor_idx]  # [D]

            # (4) compute anchor loss for this class
            sim_with_anchor = torch.matmul(class_feats, anchor.unsqueeze(0).T).squeeze(1)  # [Nc]
            class_loss = torch.mean(1.0 - sim_with_anchor)

            loss += class_loss * class_feats.size(0)
            total_samples += class_feats.size(0)

        if total_samples == 0:
            return torch.tensor(0.0, device=device)

        return loss / total_samples
    
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
        adata = sc.read_h5ad("/workspace/geshuang/code/scGPT/data/pbmc/pbmc_10k.h5ad")
    elif dataset_name == "pancreas":
        adata = sc.read_h5ad("/workspace/geshuang/code/scGPT/data/PANCREAS/pancreas_data.h5ad")
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
    # import scib_metrics
    # scib.metrics.metrics
    # from scib_metrics.benchmark import Benchmarker
    # results = scib_metrics.metrics(
    #     adata,
    #     adata_int=adata,
    #     batch_key=batch_key,
    #     label_key=label_key,
    #     embed="X_scGPT",
    #     isolated_labels_asw_=False,
    #     silhouette_=True,
    #     hvg_score_=False,
    #     graph_conn_=True,
    #     pcr_=True,
    #     isolated_labels_f1_=False,
    #     trajectory_=False,
    #     nmi_=True,  # use the clustering, bias to the best matching
    #     ari_=True,  # use the clustering, bias to the best matching
    #     cell_cycle_=False,
    #     kBET_=False,  # kBET return nan sometimes, need to examine
    #     ilisi_=False,
    #     clisi_=False,
    # )
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="X_scGPT",
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

from torch import Tensor

def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)

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
        
class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # Xavier 初始化（推荐）
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(10)

    # def forward(self, input):
    #     out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

    #     if self.to_reduce:
    #         # Reduce_proxy
    #         out = reduce_proxies(out, self.nb_proxy)

    #     if self.sigma is not None:
    #         out = self.sigma * out

    #     return out
    def forward(self, input, label=None, margin=0.35):
        # L2-normalize input and weight
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))  # (B, C)

        if label is not None:
            # Apply CosFace margin
            one_hot = F.one_hot(label, num_classes=cosine.size(1)).float().to(input.device)
            cosine = cosine - one_hot * margin

        if self.to_reduce:
            cosine = reduce_proxies(cosine, self.nb_proxy)

        if self.sigma is not None:
            cosine = self.sigma * cosine

        return cosine
    
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
        classifier: str = "Linear"     # "Linear" or "Cosine"
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        # self.out_layer = nn.Linear(d_model, n_cls, bias=False)
        if classifier == "Linear":
            self.out_layer = nn.Linear(d_model, n_cls)
        elif classifier == "Cosine":
            self.out_layer = CosineLinear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        out = self.out_layer(x)
        return out, x   
    
class ContinualClassify():
    def __init__(self, config, vocab, num_batch_types, modeldict_name = "best_model.pt"):
        self.config = config
        self.vocab = vocab
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.num_max_classes = self.config["init_class"]
        self.classifier = self.config["classifier"]
        self.model, self.model_file = self.prepare_model(num_batch_types, self.num_max_classes, modeldict_name)
        self.model.to(self.device)
        self.model.cls_decoder = ClsDecoder(512, self.num_max_classes, classifier=self.classifier).to(self.device)
        if self.config["load_model"] is not None:
            load_pretrained(self.model, torch.load(self.model_file), verbose=False)
        
        if self.config["freeze_except_layer1011"]:
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
                if "layers.11" or "layers.10" in name:
                    param.requires_grad = True

        if self.config["freeze_all"]:
            for name, param in self.model.named_parameters():
                param.requires_grad = False
                if 'decoder' in name:
                    param.requires_grad = True
                if 'out_layer' in name:
                    param.requires_grad = True    

        if self.config["loramoe"]:
            for name, param in self.model.named_parameters():
                if "lora_moe" in name:
                    param.requires_grad = True 

        if self.config["adapter"]:
            for name, param in self.model.named_parameters():
                if "adapter" in name:
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
        self.max_test_id = 5
        self.old_proto = defaultdict()
        self.past_valid_loaders = {}
        self.contrastive_proto_loss_list, self.repultion_loss_list = [], []

    def prepare_model(self, num_batch_types, num_max_classes, modeldict_name="best_model.pt"):
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
            from loramoemodel import TransformerModel, AdversarialDiscriminator   # loramoe
        else:
            from scgpt.model import TransformerModel, AdversarialDiscriminator
            
        model = TransformerModel(
            len(self.vocab),
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=3,
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

    def train(self, loader, logger, epoch, test_batch_idx):
        self.model.train()
        if self.past_model:
            self.past_model.eval()
        total_loss, total_cls, total_num = 0.0, 0.0, 0.0
        proto_loss = None
        cope_loss = None
        log_interval = 100
        start_time = time.time()
        num_batches = len(loader)
        cell_emb_list = []
        celltype_labels_list = []
        train_iter_loss_list = []
        entropy_list, accuracy_list = [], []
        proto_loss_list = []
        repultion_loss_list = []
        
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
                        batch_id = torch.tensor(test_batch_idx),
                        # batch_id = None,
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
                # cell_emb = output_dict["cls_output"][1]
                cell_emb_list.append(cell_emb)
                celltype_labels_list.append(celltype_labels)
                masked_positions = input_values.eq(-1)
                # metrics_to_log = {}
                total_num += len(input_gene_ids)

                
                output_values = output_dict["cls_output"][0]
                loss_cls = self.criterion_cls(output_dict["cls_output"][0], celltype_labels)
                loss = loss_cls
                # print(f"loss requires_grad? {loss.requires_grad}")
                probs = F.softmax(output_values, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)                 # 
                entropy_list.extend(entropy)
                accuracy_list.extend(output_values.argmax(1) == celltype_labels)
                # train_loss_list.append(loss.item())
                # metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                        (output_dict["cls_output"][0].argmax(1) == celltype_labels)
                        .sum()
                        .item()
                ) / celltype_labels.size(0)
                train_iter_loss_list.append(loss_cls.item())
                # 计算过去模型和当前模型的 MSE 损失
                if self.past_model and test_batch_idx !=0 and self.config["pastmse"]:
                    with torch.no_grad():
                        past_output_dict = self.past_model(
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
                        cell_emb_past = past_output_dict["cell_emb"].squeeze(1)
                        # loss_past = self.criterion_cls(past_output_dict["cls_output"], celltype_labels)     # 这个 past model 的cls_decoder是随机初始化的，结果是无效的
                        ################### 加一个concat之后做分类的loss ########################
                 
                    # mse_past_current = F.mse_loss(output_dict["cls_output"], past_output_dict["cls_output"])
                    mse_past_current = F.mse_loss(cell_emb, cell_emb_past)
                    loss += mse_past_current
                    # loss += loss_past
                    # metrics_to_log.update({"train/mse_past_current": mse_past_current.item()})
                if test_batch_idx != 0 and self.config["proto_loss"]:
                    # print('length of old_proto', len(self.old_proto))
                    proto_loss = self.contrastive_proto_loss(self.old_proto, self.model.cls_decoder.out_layer.weight)
                  
                    loss = loss + proto_loss * self.config["proto_weight"]  
                    # print(f"loss requires_grad???? {loss.requires_grad}")
                    proto_loss_list.append((proto_loss * self.config["proto_weight"]).item())
                    # self.contrastive_proto_loss_list.append(proto_loss.detach().cpu().numpy()) 
                    # check_out_layer_in_optimizer(self.model, self.optimizer)
                    # found = any(
                    #         self.model.cls_decoder.out_layer.weight.data_ptr() == p.data.data_ptr()
                    #         for group in self.optimizer.param_groups
                    #         for p in group['params']
                    #     )
                    # print("Decoder权重是否在optimizer中：", found)
                    # for name, param in self.model.named_parameters():
                    #     if "cls_decoder" in name:
                    #         print(name, param.requires_grad)
                    # print('#######################', all(p.requires_grad for p in self.model.cls_decoder.out_layer.parameters()))
                    # print('contrastive_proto_loss:', proto_loss)
                    # print('Current decoder weight:', self.model.cls_decoder.out_layer.weight.data)
                    # first_key = list(self.old_proto.keys())[0]
                    # first_value = self.old_proto[first_key][:5]
                    # print('Old proto:', first_value)
                if self.config["repultion_loss"]:
                    repulsion_loss = self.repultion_loss(self.model.cls_decoder.out_layer.weight.data)
                    loss += repulsion_loss
                    repultion_loss_list.append(repulsion_loss.item())

                    # metrics_to_log.update({"train/repulsion_loss": repulsion_loss.item()})
                if self.config["weight_miloss"]:
                    weight_miloss = self.extended_mutual_information_loss_weighted(output_values, class_weights=None, lambda_weight=1.0, eps=1e-8)
                    loss = loss + weight_miloss

                if test_batch_idx != 0 and self.config["cope_loss"]:
                    cope_loss = self.cope_ppp_loss(cell_emb, celltype_labels, self.old_proto)
                    loss = loss + cope_loss
                    
                    
                if self.config["anchorloss"]:
                    anchor_loss = AdvancedAnchorLoss()(cell_emb, celltype_labels)
                    loss += 10 * anchor_loss
                    print('anchor_loss:', 10*anchor_loss)
                    # self.
                
                print(f"train/cls: {loss_cls.item()}, cope_loss: {cope_loss.item() if isinstance(cope_loss, torch.Tensor) else None}, train_loss: {loss.item()}")
            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            # for name, param in self.model.named_parameters():
            #     if param.grad is None:
            #         print(f"{name} → No gradient")
            #     else:
            #         print(f"{name} → Grad mean: {param.grad.abs().mean().item():.6f}")
            # metrics_to_log.update({
            #         "train/cls": loss_cls.item(), 
            #         "proto_loss": contrastive_proto_loss.item() if isinstance(contrastive_proto_loss, torch.Tensor) else contrastive_proto_loss,
            #         "train_loss": loss.item()
            #     })
            
            # print("Type of proto_loss:", type(proto_loss))
            # print("proto_loss requires_grad:", proto_loss.requires_grad if isinstance(proto_loss, torch.Tensor) else "Not tensor")

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

            # print(metrics_to_log)

            total_loss += loss.item()   
            total_cls += loss_cls.item() if True else 0.0
            
            if batch % log_interval == 0 and batch > 0:
                lr = self.scheduler.get_last_lr()[0]
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
        cur_loss = total_loss / total_num    # 当前epoch的loss
        proto_loss_last = sum(proto_loss_list) / total_num
        repultion_loss = sum(repultion_loss_list) / total_num
        return cell_emb_list, celltype_labels_list, cur_loss, train_iter_loss_list, proto_loss_last, repultion_loss, entropy_list, accuracy_list
    
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
        eval_iter_loss_list = []
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
                            batch_id = torch.tensor(test_batch_idx),
                            # batch_id = None,
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
                            MVC=False,
                            ECS=False,
                            do_sample=False,
                        )
                    output_values = output_dict["cls_output"][0]
                    probs = F.softmax(output_dict["cls_output"][0], dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                    loss = self.criterion_cls(output_values, celltype_labels)
                eval_iter_loss_list.append(loss)
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
            
            with open(str(save_dir) + "/" + f"_batch_{test_batch_idx}_final_results.json", "w") as f:
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
            plt.savefig(str(save_dir) + "/" + f"batch_{test_batch_idx}_Confusion_Matrix.png")
            plot_entropy_accuracy(entropy_list, accuracy_list, save_dir, test_batch_idx, labellist)    #  画熵-准确性图 

        return total_loss / total_num, total_error / total_num, result_dict, eval_iter_loss_list

    def eval_testdata(self, adata_t, gene_ids, input_layer_key):
        # evalmodel = copy.deepcopy
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

        adata_t.obsm["X_scGPT"] = cell_embeddings

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
                    batch_id = torch.tensor(test_batch_idx),
                    # batch_id=None,
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

        adata_t.obsm["X_scGPT"] = cell_embeddings

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
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0] 
            embedding = embedding_list[data_index]
            proto = embedding.mean(0).float()
            self.model.cls_decoder.out_layer.weight.data[int(class_index)] = proto.to(self.device)
            if self.config["proto_loss"]:
                self.old_proto[int(class_index)] = proto.clone()

    def update_prototype(self, embedding_list, label_list):
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = torch.unique(label_list).cpu().numpy()
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0] 
            embedding = embedding_list[data_index]
            proto = embedding.mean(0).float()
            # self.model.cls_decoder.out_layer.weight.data[int(class_index)] = proto.to(self.device)
            if self.config["cope_loss"]:
                self.old_proto[int(class_index)] = proto.clone()

    def update_classifier_weights_entropy(self, embedding_list, label_list, entropy_list, accuracy_list):
        embedding_list = torch.cat(embedding_list, dim=0)    # shape: [N, D]
        label_list = torch.cat(label_list, dim=0)            # shape: [N]
        entropy_list = torch.stack(entropy_list)
        # entropy_list = torch.cat(entropy_list, dim=0)        # shape: [N]
        accuracy_list = torch.tensor(accuracy_list, dtype=torch.bool, device=embedding_list.device)    # shape: [N], bool or int (0/1)

        class_list = torch.unique(label_list).cpu().numpy()

        for class_index in class_list:
            # 找到该类所有样本索引
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0]
            
            # 提取该类对应的熵、准确性、embedding
            class_entropy = entropy_list[data_index]
            class_accuracy = accuracy_list[data_index]
            class_embedding = embedding_list[data_index]

            # 找出分类正确的索引
            correct_mask = class_accuracy.bool()
            arange_indices = torch.arange(len(data_index), device=correct_mask.device)
            correct_indices = arange_indices[correct_mask]


            if correct_indices.numel() >= 1:
                # 仅保留分类正确的熵
                filtered_entropy = class_entropy[correct_indices]
                filtered_embedding = class_embedding[correct_indices]

                # 找到熵最小的前10个索引
                topk = min(10, filtered_entropy.size(0))
                _, top_indices = torch.topk(-filtered_entropy, topk)  # -entropy: 小到大排序

                selected_embedding = filtered_embedding[top_indices]
                print('entropy filter samples of class:{}'.format(class_index), selected_embedding.shape[0])
                # 计算原型
                proto = selected_embedding.mean(0).float().detach()
                assert proto.shape[0] == self.model.cls_decoder.out_layer.weight.shape[1]

                # 更新该类的 prototype（out_layer 权重）
                with torch.no_grad():
                    self.model.cls_decoder.out_layer.weight[int(class_index)] = proto.to(self.device)
                if self.config["proto_loss"]:
                    self.old_proto[int(class_index)] = proto.clone()
                    
            else:
                # 没有分类正确的样本则跳过更新
                continue
        print('#########################', len(self.old_proto))

    def build_fewshot_dataset(self, all_counts, celltypes_labels, batch_ids, shots_per_class=5, seed=42):
        np.random.seed(seed)

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
    
    def extended_mutual_information_loss_weighted(self, prob_logits, class_weights=None, lambda_weight=1.0, eps=1e-8):
        """
        Extended Mutual Information Loss with class weighting.

        Args:
            prob_logits (Tensor): [B, K] logits per sample.
            class_weights (Tensor or None): [K] weights per class, e.g., inverse class frequency.
            lambda_weight (float): scaling factor for marginal entropy.
            eps (float): small value for numerical stability.

        Returns:
            Scalar loss (Tensor)
        """
        # Softmax to get predicted probabilities
        p = F.softmax(prob_logits, dim=1) + eps  # [B, K]
        log_p = torch.log(p)

        # Conditional entropy: H(C|X) = -E_x sum_j p(c_j|x) log p(c_j|x)
        cond_entropy = -torch.mean(torch.sum(p * log_p, dim=1))  # scalar

        # Marginal distribution: p̂(C) ≈ mean over batch
        p_mean = torch.mean(p, dim=0)  # [K]
        log_p_mean = torch.log(p_mean + eps)

        # Apply class weights if provided
        if class_weights is not None:
            weights = class_weights / class_weights.sum()  # normalize
            weighted_entropy = -torch.sum(weights * p_mean * log_p_mean)
        else:
            weighted_entropy = -torch.sum(p_mean * log_p_mean)

        # Final loss: H(C|X) - λ * H(C) or H_w(C)
        loss = cond_entropy - lambda_weight * weighted_entropy
        return loss
    
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
        p = F.softmax(prob_logits, dim=1)     # shape: [B, K]   聚类概率分布
        p = p + eps  # numerical stability
        log_p = torch.log(p)

        # 1. Conditional entropy H(C|X) = -E_x sum_j p(c_j|x) log p(c_j|x)
        cond_entropy = -torch.mean(torch.sum(p * log_p, dim=1))  # scalar 条件熵

        # 2. Marginal distribution p̂(C) ≈ mean over batch
        p_mean = torch.mean(p, dim=0)  # shape: [K]
        log_p_mean = torch.log(p_mean + eps)

        # 3. Entropy H(C) = -sum_j p̂(c_j) log p̂(c_j)
        marginal_entropy = -torch.sum(p_mean * log_p_mean)  # scalar   每个类别的边际分布

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


    def cope_ppp_loss(self, features, labels, prototypes, temperature=0.1):
        """
        CoPE PPP Loss (only for classes that exist in prototypes).

        Args:
            features: Tensor [B, D] — 当前 batch 的嵌入特征（应已归一化）
            labels: Tensor [B] — 样本标签
            prototypes: defaultdict(int -> Tensor[D]) — 类别原型向量
            temperature: float — 温度参数 τ

        Returns:
            Scalar loss tensor
        """
        device = features.device
        B, D = features.shape
        all_class_ids = list(prototypes.keys())

        if len(all_class_ids) == 0:
            return torch.tensor(0.0, device=device)

        # 所有原型向量归一化后组成矩阵 [C, D]
        all_protos = torch.stack([F.normalize(prototypes[c].to(device), dim=0) for c in all_class_ids]).detach()

        total_loss = 0.0
        count = 0

        for cls in all_class_ids:
            cls = int(cls)
            mask_pos = (labels == cls)
            mask_neg = (labels != cls)

            pos_feats = features[mask_pos]  # [N_pos, D]
            neg_feats = features[mask_neg]  # [N_neg, D]

            if pos_feats.size(0) < 1:
                continue

            pc = F.normalize(prototypes[cls].to(device), dim=0).detach()  # [D]

            for i in range(pos_feats.size(0)):
                xi = pos_feats[i]  # 当前正样本

                # -------- Attractor Set --------
                pseudo_protos = [pos_feats[j].detach() for j in range(pos_feats.size(0)) if j != i]
                attractor_set = [pc] + pseudo_protos
                attractor_set = torch.stack(attractor_set)  # [K, D]
                sim_pos = torch.matmul(attractor_set, xi) / temperature  # [K]
                sim_soft = F.softmax(sim_pos, dim=0)
                pos_prob = sim_soft[0]  # pc is第一个

                # -------- Repellor Set --------
                if neg_feats.size(0) > 0:
                    sim_pc = torch.matmul(neg_feats, pc.unsqueeze(1)).squeeze(1) / temperature  # [N_neg]
                    sim_all = torch.matmul(neg_feats, all_protos.T) / temperature  # [N_neg, C]
                    denom = torch.logsumexp(sim_all, dim=1)
                    log_p_c_given_neg = sim_pc - denom
                    neg_term = -torch.sum(torch.log(1.0 - torch.exp(log_p_c_given_neg) + 1e-8)) / neg_feats.size(0)
                else:
                    neg_term = 0.0

                loss_i = -torch.log(pos_prob + 1e-8) + neg_term
                total_loss += loss_i
                count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / count

    
    def contrastive_proto_loss(self, old_protos, new_weight_matrix, temperature=0.1):
        """
        Args:
            old_protos: Dict[int, Tensor] — 历史 prototype
            new_weight_matrix: Tensor — 当前模型分类器权重 (num_classes x dim)
        Returns:
            torch.Tensor — 支持梯度反向传播的 contrastive loss
        """
        device = new_weight_matrix.device
        losses = []

        for i, proto_old in old_protos.items():
            proto_old = proto_old.to(device)
            proto_new = new_weight_matrix[i]

            # 正样本（同类）
            pos_sim = F.cosine_similarity(proto_old.unsqueeze(0), proto_new.unsqueeze(0), dim=-1) / temperature

            # 负样本（不同类）
            neg_sims = []
            for j in range(new_weight_matrix.size(0)):
                if j != i:
                    neg_proto = new_weight_matrix[j]
                    neg_sim = F.cosine_similarity(proto_old.unsqueeze(0), neg_proto.unsqueeze(0), dim=-1) / temperature
                    neg_sims.append(neg_sim)

            neg_sims = torch.cat(neg_sims, dim=0)  # shape: (num_classes - 1,)
            logits = torch.cat([pos_sim, neg_sims], dim=0)  # shape: (num_classes,)
            labels = torch.zeros(1, dtype=torch.long, device=device)  # 正样本是第一个位置

            # 使用 InfoNCE 对比损失（即 CrossEntropy）
            loss = F.cross_entropy(logits.unsqueeze(0), labels)
            losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return torch.stack(losses).mean()

    def process_batch(self, adata_train, logger, save_dir, dataset_name, experiment_name, all_adata_test, all_batch_results, test_batch_idx):
        '''
        adata_train, config, logger, save_dir, config["dataset_name"], config["experiment_name"], all_adata_test, all_batch_results,
            test_batch_idx
        '''
        ############################# 先搞定数据 #################################
        # 统计每种细胞类型的数量        
        if self.config["init_class"] == 8 or self.config['fewshot'] is not None:
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
            valid_celltypes = celltype_counts[celltype_counts >= self.config["fewshot"] + 1].index

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
            with open(save_dir / "celltype_to_label.json", 'w') as file:
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
            with open(save_dir / "celltype_to_label.json", 'w') as file:
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
            if self.config["classifier"] == "linear":
                new_out_layer = nn.Linear(in_features, new_out_features).to(self.device)
                
                # 迁移已有权重
                with torch.no_grad():
                    new_out_layer.weight[:old_out_features] = old_out_layer.weight
                    new_out_layer.bias[:old_out_features] = old_out_layer.bias
            else:
                new_out_layer = CosineLinear(in_features, new_out_features).to(self.device)
                with torch.no_grad():
                    new_out_layer.weight[:old_out_features] = old_out_layer.weight
                    new_out_layer.sigma.data = copy.deepcopy(old_out_layer.sigma.data)
                    # new_out_layer.bias[:old_out_features] = old_out_layer.bias
            
            # 替换分类头
            self.model.cls_decoder.out_layer = new_out_layer
        
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
             ) = self.build_fewshot_dataset(all_counts, celltypes_labels, batch_ids, shots_per_class=self.config["fewshot"], seed = 42)
            adata_train_split = adata_train[train_indices].copy()
            adata_valid_split = adata_train[val_indices].copy()     
            print('train_sample:', adata_train_split)   
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
        # np.save(save_dir / "train_celltype_labels.npy", train_celltype_labels)
        # np.save(save_dir / "valid_celltype_labels.npy", valid_batch_labels)
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
            max_len=3001,    # 这个不影响最终维度tokenized_train['genes'].shape：torch.Size([903, 1201])
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
        train_epoch_loss, proto_epoch_loss, repultion_epoch_loss = [], [], []
        eval_epoch_loss = []
        ######################### 设置学习率 ##########################
        self.early_stopper = EarlyStopping(patience=5, min_delta=1e-4, mode='min')
        if self.config["schedule"] == "cosine_schedule_with_warmup":
            if self.config["fewshot"] is not None:
                warmup_steps = 10
            else:
                warmup_steps = 2
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
                cell_emb_list, train_labels, train_loss, _, proto_loss, repultion_loss, entropy_list, accuracy_list = self.train(
                    loader=train_loader,
                    logger=logger,
                    epoch=epoch,
                    test_batch_idx=test_batch_idx
                )
            elif self.config["do_train"] and epoch == self.config["epochs"]:
                cell_emb_list, train_labels, train_loss, _, proto_loss, repultion_loss, entropy_list, accuracy_list = self.train(
                    loader=train_loader,
                    logger=logger,
                    epoch=epoch,
                    test_batch_idx=test_batch_idx
                )
            

            # if self.config["do_train"]:
            #     # if self.early_stopper.early_stop or epoch == self.config["epochs"]:
            #     if epoch == self.config["epochs"]:
            #         if self.config["entropy"]:
            #             self.update_classifier_weights_entropy(cell_emb_list, celltype_labels, entropy_list, accuracy_list)
            #         else:
            #             self.update_classifier_weights(cell_emb_list, celltype_labels)
            train_epoch_loss.append(train_loss)
            proto_epoch_loss.append(proto_loss)
            repultion_epoch_loss.append(repultion_loss)

            # if self.config["fewshot"] and epoch % 10 == 0:    # 在fewshot下发生变化        
            val_loss, val_err, result_dict, eval_iter_loss_list = self.evaluate(
                loader=valid_loader,
                epoch=epoch,
                save_dir = save_dir,
                test_batch_idx=test_batch_idx
            )

            # elif epoch == self.config["epochs"]:
            #     if self.config["entropy"]:
            #         self.update_classifier_weights_entropy(cell_emb_list, celltype_labels, entropy_list, accuracy_list)
            #     else:
            #         self.update_classifier_weights(cell_emb_list, celltype_labels)
            # else:
            #     pass

            # print(f"Epoch: {epoch}, early_stop: {self.early_stopper.early_stop}, total_epochs: {self.config['epochs']}")
            
        
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
                torch.save(best_model.state_dict(), save_dir / f"best_model_batch_{test_batch_idx}.pt")
                with open(str(save_dir) + "/" + f"_batch_{test_batch_idx}_besteval_results.json", "w") as f:
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
                                    output_values = output_dict["cls_output"][0]
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

            self.early_stopper(val_loss, self.model)
            if self.early_stopper.early_stop or epoch == self.config["epochs"]:
                print(f"Early stopping at epoch {epoch} on task {test_batch_idx}")
                if self.config["entropy"]:
                    self.update_classifier_weights_entropy(cell_emb_list, train_labels, entropy_list, accuracy_list)
                elif self.config["cope_loss"]:
                    self.update_prototype(cell_emb_list, train_labels)
                else:
                    self.update_classifier_weights(cell_emb_list, train_labels)
                break

            if self.config["schedule"] == "stepLR":
                self.scheduler.step()
                
        gc.collect()
        torch.cuda.empty_cache()
        del train_data_pt, valid_data_pt, train_loader, valid_loader
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

        # sc.pp.neighbors(adata_sorted, use_rep="X_scGPT")
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

        # sc.pp.neighbors(adata_sorted, use_rep="X_scGPT")
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

        if test_batch_idx == self.max_test_id:
            self.plot_clusters_prototypes(combined_adata_test, self.model.cls_decoder.out_layer.weight.data, input_layer_key, gene_ids, save_dir)
        self.plot_train_val_loss(train_epoch_loss, eval_epoch_loss, save_dir, test_batch_idx)
        if self.config["repultion_loss"] or self.config["proto_loss"]:
            self.plot_proto_repultion_loss(proto_epoch_loss, repultion_epoch_loss, save_dir, test_batch_idx)
        return best_model, combined_adata_test, gene_ids
    
    def plot_clusters_prototypes(self, adata, prototype, input_layer_key, gene_ids, save_dir):
    # prototypes = F.normalize(prototype.prototypes, dim=1)
    # # prototypes = prototype.prototypes
    # # Step 1: 获取细胞和原型嵌入
    # X_cells = adata_sorted.obsm["X_scGPT"]  # 细胞嵌入向量，shape = [n_cells, 1200]
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
        all_gene_ids, all_values = tokenized_all["genes"].to(self.device), tokenized_all["values"].to(self.device)
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        batch_ids = torch.tensor(adata.obs["batch_id"].values).to(self.device)
        batch_ids = torch.zeros_like(batch_ids)                    # 在实际训练中只训练了一个BN层，因此batch_ids都要设置为0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):
            cell_embeddings = self.model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=self.config["batch_size"],
                # batch_labels=torch.from_numpy(adata.obs["batch_id"].to_numpy()).long() if DSBN else None,
                batch_labels = torch.from_numpy(batch_ids).long() if self.config["DSBN"] else None,
                time_step=0,
                return_np=True,
            )

            if self.past_model is not None:
                cell_embeddings_past = self.past_model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    # batch_labels=torch.from_numpy(adata.obs["batch_id"].to_numpy()).long() if DSBN else None,
                    batch_labels = torch.from_numpy(batch_ids).long() if self.config["DSBN"] else None,
                    time_step=0,
                    return_np=True,
                )
                cell_embeddings = np.concatenate([cell_embeddings_past, cell_embeddings], axis=-1)
        # cell_embeddings = output_dict["cls_output"][1]
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
                    if self.config["adapter"] or self.config["loramoe"]:
                        output_dict, _ = self.model(
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

            with open(str(save_dir) + "/" + f"predict_batch_{test_batch_idx}_finaltest_results.json", "w") as f:
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
            plt.savefig(str(save_dir) + "/" + f"predict_batch_{test_batch_idx}_finaltest_Confusion_Matrix.png")
            
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

    def evaluate_all(self, adata_train, save_dir, test_batch_idx, cell_type_map, all_adata_test):
                # 统计每种细胞类型的数量        
        if self.config["init_class"] == 8 or self.config["filter_sample"]: 
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
        all_adata_test.append(adata_valid_split)
        return all_adata_test, gene_ids

    def evaluate_predict(self, adata_test, save_dir, test_batch_idx, cell_type_map, mod_type):
        self.config["weight_dir"] = save_dir
                # 统计每种细胞类型的数量
        # celltype_counts = adata_test.obs["celltype"].value_counts()
        # # 找出数量大于等于 4 的细胞类型
        # valid_celltypes = celltype_counts[celltype_counts >= 2].index

        # # 过滤掉数量小于 4 的细胞类型对应的样本
        # adata_test = adata_test[adata_test.obs["celltype"].isin(valid_celltypes)].copy()
        le = LabelEncoder()
        adata_train.obs["batch_id"] = le.fit_transform(adata_train.obs["batch_id"])
        num_batch_types = adata_train.obs["batch_id"].nunique()
        input_layer_key = "X_binned"
        all_counts = (
            adata_train.layers[input_layer_key].A
            if issparse(adata_train.layers[input_layer_key])
            else adata_train.layers[input_layer_key]
        )
        # genes = adata_test.var["gene_name"].tolist()
        # gene_ids = np.array(self.vocab(genes), dtype=int)
        
        # current_label_dict, current_celltype_labels = np.unique(
        #     np.array(adata_test.obs["celltype"].tolist()), return_inverse=True
        # )
        celltype_str_list = np.array(adata_train.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_train.obs["celltype_labels"] = current_celltype_labels                   
        batch_ids = adata_train.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        if self.config["use_multimod"]:
            modality_to_id = {mod: i for i, mod in enumerate(adata_train.obs["modality"].unique())}
            adata_train.obs["modality_id"] = adata_train.obs["modality"].map(modality_to_id).astype(int)
            multimods_ids = adata_train.obs["modality_id"].tolist()
            num_multimods = len(set(multimods_ids))
            multimods_ids = np.array(multimods_ids)

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
            from functions.balanced_sampler import stratified_split_with_small_class_reserve
            if self.config["use_multimod"]:
                (
                    train_data,
                    valid_data,
                    train_celltype_labels,
                    valid_celltype_labels,
                    train_batch_labels,
                    valid_batch_labels,
                    train_multimod_labels,
                    valid_multimod_labels,
                    train_indices,
                    val_indices
                ) = stratified_split_with_small_class_reserve(
                    all_counts,
                    current_celltype_labels,
                    batch_ids,
                    multimods_ids,
                    test_size=0.1,
                    min_class_size=20
                )
            else:
                (
                    train_data,
                    valid_data,
                    train_celltype_labels,
                    valid_celltype_labels,
                    train_batch_labels,
                    valid_batch_labels,
                    train_indices,
                    val_indices
                ) = stratified_split_with_small_class_reserve(
                    all_counts,
                    current_celltype_labels,
                    batch_ids,
                    test_size=0.1,
                    min_class_size=20
                )
            adata_train_split = adata_train[train_indices].copy()
            adata_valid_split = adata_train[val_indices].copy() 
        tokenized_test = tokenize_and_pad_batch(
            valid_data,
            self.gene_ids,
            max_len=3001,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
            mod_type=mod_type if self.config["use_mod"] else None,
            vocab_mod=self.vocab_mod if self.config["use_mod"] else None,
        )
        test_data_pt = prepare_testdata(
            sort_seq_batch=False,
            tokenized_test=tokenized_test,
            test_batch_labels=valid_batch_labels,
            test_celltype_labels=valid_celltype_labels,
            test_multimod_labels=valid_multimod_labels,
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
                mod_types = batch_data["mod_types"].to(self.device)
                if self.config["use_multimod"]:
                    multimod_labels = batch_data["multimod_types"].to(self.device)
                with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                    output_dict = self.model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=batch_labels if False or self.config["DSBN"] else None,
                        multimod_labels=multimod_labels if self.config["use_multimod"] else None,
                        CLS=True,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        # do_sample=False,
                        mod_types=mod_types if self.config["use_mod"] else None,
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
            f1 = f1_score(np.array(labellist), np.array(predictions), average='macro')       
            f1_micro = f1_score(np.array(labellist), np.array(predictions), average='micro') 
            f1_weighted = f1_score(np.array(labellist), np.array(predictions), average='weighted') 
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
        return self.gene_ids
    
def evaluate_predict(save_dir, cell_type_map):
    config = init_wandb()
    set_seed(config["seed"])
    if config["dataset_name"] == "pancreas":
        data_paths = ["/workspace/geshuang/code/scGPT/data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6)]  # 剩下三个batch作为测试集

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
    continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_5.pt")

    all_adata_test = []
    all_batch_results = {}

    for test_batch_idx in range(len(data_paths)):
        adata_test = sc.read_h5ad(data_paths[test_batch_idx])
        # gene_ids = continual_classify.evaluate_predict(
        #     adata_test, save_dir, test_batch_idx, cell_type_map
        # )
        all_adata_test, gene_ids = continual_classify.evaluate_all(adata_test, save_dir, test_batch_idx, cell_type_map, all_adata_test)
        all_adata_test.append(adata_test)

        combined_adata_test = anndata.concat(
            all_adata_test,
            join="outer",
            merge="unique",
            label="batch",
            index_unique=None
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

    with open(str(save_dir) + "final_evaluate_results.json", "w") as f:
        json.dump(final_results, f, indent=4)

    sc.pp.neighbors(final_adata, use_rep="X_scGPT")
    sc.tl.umap(final_adata, min_dist=0.3)
    sc.pl.umap(
        final_adata,
        color=["str_batch"],
        title=[f"batch, avg_batch = {final_results.get('avg_batch', 0.0):.4f}"],
        frameon=False,
        show=False,
    )
    plt.savefig(
        str(save_dir) + "/" + "embeddings_batch_umap[cls]_batch_eval_all.png",
        bbox_inches='tight')

    sc.pp.neighbors(final_adata, use_rep="X_scGPT")
    sc.tl.umap(final_adata, min_dist=0.3)
    sc.pl.umap(
        final_adata,
        color=["celltype"],
        title=[f"celltype, avg_bio = {final_results.get('avg_bio', 0.0):.4f}"],
        frameon=False,
        show=False,
    )
    plt.savefig(
        str(save_dir) + "/" + "embeddings_celltype_umap[cls]_batch_eval_all.png",
        bbox_inches='tight')
    
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
        data_paths = ["/pbmc/" + f"pbmc_batch{i}.h5ad" for i in range(2)]
    elif config["dataset_name"] == "pancreas":
        data_paths = ["/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6)]  # 剩下三个batch作为测试集
        # data_paths = ["/PANCREAS/" + f"pancreas_batch{2}.h5ad", 
        #               "/PANCREAS/" + f"pancreas_batch{3}.h5ad"]

    model_dir = Path(config["load_model"])
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    num_batch_types = 1
    continual_classify = ContinualClassify(config, vocab, num_batch_types)

    all_adata_test = []
    all_batch_results = {}

    for test_batch_idx in range(len(data_paths)):
        adata_train = sc.read_h5ad(data_paths[test_batch_idx])
        best_model, combined_adata_test, gene_ids = continual_classify.process_batch(
            adata_train, logger, save_dir, config["dataset_name"], config["experiment_name"], all_adata_test, all_batch_results,
            test_batch_idx
        )
        ########################## Debug stage close ##################################
        besteval_results, besteval_adata = continual_classify.best_model_evaluate(best_model, adata_t=combined_adata_test, gene_ids=gene_ids,
                                                input_layer_key={
                                                    "normed_raw": "X_normed",
                                                    "log1p": "X_normed",
                                                    "binned": "X_binned",
                                                }["binned"],test_batch_idx=test_batch_idx)
        with open(str(save_dir) + "/" + f"batch_{test_batch_idx}_bestval_results.json",
                  "w") as f:
            json.dump(besteval_results, f, indent=4)

        sc.pp.neighbors(besteval_adata, use_rep="X_scGPT")
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

        sc.pp.neighbors(besteval_adata, use_rep="X_scGPT")
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


    final_results, final_adata = continual_classify.eval_testdata(
        adata_t=combined_adata_test,
        gene_ids=gene_ids,
        input_layer_key={
            "normed_raw": "X_normed",
            "log1p": "X_normed",
            "binned": "X_binned",
        }["binned"],
    )

    with open(str(save_dir) + "/" + config["dataset_name"] + "_" + config["experiment_name"] + "_final_results.json",
              "w") as f:
        json.dump(final_results, f, indent=4)
    
def predict(save_dir, cell_type_map):
    config = init_wandb()
    set_seed(config["seed"])
    if config["dataset_name"] == "pancreas":
        data_paths = ["/workspace/geshuang/code/scGPT/data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6, 9)]  # 剩下三个batch作为测试集

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
    continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_5.pt")

    all_adata_test = []
    all_batch_results = {}

    for test_batch_idx in range(len(data_paths)):
        adata_test = sc.read_h5ad(data_paths[test_batch_idx])
        gene_ids = continual_classify.predict(
            adata_test, save_dir, test_batch_idx, cell_type_map
        )
        all_adata_test.append(adata_test)

        combined_adata_test = anndata.concat(
            all_adata_test,
            join="outer",
            merge="unique",
            label="batch",
            index_unique=None
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

    with open(str(save_dir) + "/" + config["dataset_name"] + "_" + config["experiment_name"] + "_finaltest_results.json",
              "w") as f:
        json.dump(final_results, f, indent=4)
    
    sc.pp.neighbors(final_adata, use_rep="X_scGPT")
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

    sc.pp.neighbors(final_adata, use_rep="X_scGPT")
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
    main()

    # save_dir = "/" # baseline(no_schedular)

    # with open(save_dir + "/celltype_to_label.json", "r") as f:
    #     cell_type_map = json.load(f)
    #     f.close()
    # # predict(save_dir, cell_type_map)                         # 测试集
    # evaluate_predict(save_dir, cell_type_map)              # 验证集


