# CUDA_VISIBLE_DEVICES=1
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,6,7" 
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
import pickle
import umap
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
from tutorials.utils import (
    EarlyStopping,
    SeqDataset,
    prepare_dataloader,
    prepare_testdata as utils_prepare_testdata,
    prepare_data,
    eval_scib_metrics,
    plot_entropy_accuracy,
    reduce_proxies,
    CosineLinear,
    ClsDecoder,
)
# Set environment variables and suppress warnings
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score
from collections import defaultdict
from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import f1_score
from loss.ppp_loss import PPPloss
from captum.attr import IntegratedGradients
import scipy.sparse as sp
import glob
# from fastai.losses import FocalLoss
class FocalLoss(nn.Module):
    # y_int=True # y interpolation
    def __init__(self, 
        gamma:float=2.0, # Focusing parameter. Higher values down-weight easy examples' contribution to loss
        weight:Tensor=None, # Manual rescaling weight given to each class
        reduction:str='mean' # PyTorch reduction to apply to the output
    ): 
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        "Applies Focal Loss: https://arxiv.org/pdf/1708.02002.pdf"
    
    def forward(self, inp:Tensor, targ:Tensor) -> Tensor:
        "Applies focal loss based on https://arxiv.org/pdf/1708.02002.pdf"
        ce_loss = F.cross_entropy(inp, targ, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t)**self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
    
def init_wandb():
    hyperparameter_defaults = dict(
        seed=0,
        # dataset_name="ms",
        # dataset_name="pancreas",
        # dataset_name = "myeloid",        
        # dataset_name = "BMMC",
        dataset_name = "BMMC_filter3",
        do_train=True,
        load_model="../save/scEvolver_human",

        weight_dir = "../tutorials/save/dev_BMMC-Jan09-11-29",         # Latest results (pretrained weights)
        use_mod = True,
        use_multimod = True, 
        mask_ratio=0.0,
        epochs=15,                 # For few-shot set to 1000
        n_bins=51,
        MVC=False,
        ecs_thres=0.0,
        # dab_weight=0.5,
        lr=1e-3,
        batch_size=32,
        layer_size=512,
        nlayers=4,
        nhead=8,
        dropout=0.2,
        schedule_ratio=0.9,
        save_eval_interval=5,
        fast_transformer=False,
        pre_norm=False,
        amp=True,
        include_zero_gene=False,
        DSBN=False,                            
        k_samples=50,
        pastmse=False,
        replay=True,
        init_class = 17,
        filter_sample = False,
        # randomsplit = False,              # Set to False when using balanced_sampler
        randomsplit = False,
        fewshot = None,
        nlayers_cls=3,
        use_best_initnextbatch = False,
        adapter=False,
        freeze_except_layer1011 = False,
        freeze_all = True,
        loramoe = True,  
        proto_loss = True,
        num_of_expert = 4,         # When loramoe=False this is None
        num_of_gate = 4,
        repultion_loss = False,
        entropy =  False,
        classifier = "Linear",
        anchorloss = False,
        schedule = "cosine_schedule_with_warmup",               # "cosine_schedule_with_warmup",     # stepLR
        # schedule = "plateau", 
        proto_weight = 1.0,
        cope_loss = False,
        weight_miloss = False,
        update_classifier = False,
        ema = True,
        correct_sample = True,
        patience = 5,
        weighted_loss = False,
        contrastive_proto_loss = False,
        post_pretrained = True,
        optimizer_lr = False,
        do_dab = False,
        weight_decay = False,
        log1p = False,
        focal_loss = False,
        freeze_after_batch1 = False,         # Not effective
        freeze_lora = False,
        # schedule = "plateau",
        init_decoder = False,
        add_decoder = False,
        adapter_dim = 64,
        decrease_lr_all = True,
        valid_ratio = 0.1,
        save_weight = True,
        # loss_func = "proto_dist",    # "cross_entropy"  "proto_dist"
        loss_func = "cross_entropy"
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
            "_updata_per5_batch_" + "_blanced_sampler" + "_memory_proto_loss_1stbatch"  +"_early_stopper" + "_decrease_lr_all(*3)" + "Euclidean distance" +\
            "_correct_sample_replay"
    
            #  "_updata_bestval_epoch"
            # "_contrastive_proto_loss(all_past_proto)"
            #    "weighted_loss" + str(config["weighted_loss"]) 
            # 
            # 
            # "update_last_epoch(ema=0.5)" +\
                
    # "_update_every_epochs(ema)" 
            # "_schedule_" + str(config["schedule"]) 
            # "_correct_sample_" + str(config["correct_sample"]) 
            # "_ema_" + str(config["ema"]) 
            # "_update_classifier_" + str(config["update_classifier"])
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
        config["lr"] = 1e-4
    return config

def check_out_layer_in_optimizer(model, optimizer):
    out_layer = model.cls_decoder.out_layer
    # Get data_ptr for all parameters in the optimizer
    optimizer_param_ptrs = {p.data_ptr() for g in optimizer.param_groups for p in g['params']}

    # Check which parameters are not included in the optimizer
    missing = []
    for name, param in out_layer.named_parameters():
        if param.requires_grad and param.data_ptr() not in optimizer_param_ptrs:
            missing.append(name)

    if missing:
        print(f"❌ The following parameters in out_layer are NOT in the optimizer: {missing}")
    else:
        print(" All parameters in out_layer are included in the optimizer.")
    
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

# Reuse `EarlyStopping`, `SeqDataset`, and `prepare_dataloader` from `tutorials.utils`

def prepare_testdata(sort_seq_batch, tokenized_test, test_batch_labels, test_celltype_labels, test_multimod_labels, mask_ratio, \
                     mask_value, pad_value):
    """Wrapper around `tutorials.utils.prepare_testdata` that also attaches `mod_types` and `multimod_types` when present."""
    test_data_pt = utils_prepare_testdata(sort_seq_batch, tokenized_test, test_batch_labels, test_celltype_labels, mask_ratio, mask_value, pad_value)

    # attach mod types and multimod types if available in tokenized input
    if "mod_types" in tokenized_test:
        tensor_mod_types_test = tokenized_test["mod_types"].long()
        test_data_pt["mod_types"] = tensor_mod_types_test
    if test_multimod_labels is not None:
        test_data_pt["multimod_types"] = torch.from_numpy(test_multimod_labels).long()
    return test_data_pt

def prepare_testdata(sort_seq_batch, tokenized_test, test_batch_labels,test_celltype_labels, test_multimod_labels, mask_ratio,\
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
    if test_multimod_labels is not None:
        tensor_multimod_labels_test= torch.from_numpy(test_multimod_labels).long()
    tensor_mod_types_test =  tokenized_test["mod_types"].long()
    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        test_sort_ids = np.argsort(test_batch_labels)
        input_gene_ids_test = input_gene_ids_test[test_sort_ids]
        input_values_test = input_values_test[test_sort_ids]
        target_values_test = target_values_test[test_sort_ids]
        tensor_batch_labels_test = tensor_batch_labels_test[test_sort_ids]
        tensor_celltype_labels_test = tensor_celltype_labels_test[test_sort_ids]
        tensor_mod_types_test = tensor_mod_types_test[test_sort_ids]
        if test_multimod_labels is not None:
                tensor_multimod_labels_test = tensor_multimod_labels_test[test_sort_ids]

    test_data_pt = {
        "gene_ids": input_gene_ids_test,
        "values": input_values_test,
        "target_values": target_values_test,
        "batch_labels": tensor_batch_labels_test,
        "celltype_labels": tensor_celltype_labels_test,
    }
    test_data_pt["mod_types"] = tensor_mod_types_test
    if test_multimod_labels is not None:
        test_data_pt["multimod_types"] = tensor_multimod_labels_test
    return test_data_pt

def prepare_data(sort_seq_batch, tokenized_train, tokenized_valid, train_batch_labels, valid_batch_labels, train_celltype_labels, \
                 valid_celltype_labels, mask_ratio,\
                 mask_value, pad_value, train_multimod_labels=None, valid_multimod_labels=None):
    from functions.balanced_sampler import stratified_sample
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
    if train_multimod_labels is not None:
        tensor_multimod_labels_train = torch.from_numpy(train_multimod_labels).long()
        tensor_multimod_labels_valid = torch.from_numpy(valid_multimod_labels).long()
        
    tensor_mod_types_train, tensor_mod_types_valid = (
            tokenized_train["mod_types"].long(),
            tokenized_valid["mod_types"].long(),
        )
    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]
        tensor_mod_types_train = tensor_mod_types_train[train_sort_ids]
        if train_multimod_labels is not None:
            tensor_multimod_labels_train = tensor_multimod_labels_train[train_sort_ids]
        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]
        tensor_mod_types_valid = tensor_mod_types_valid[valid_sort_ids]
        if train_multimod_labels is not None:
            tensor_multimod_labels_valid = tensor_multimod_labels_valid[train_sort_ids]
    if input_gene_ids_train.shape[0]>=2000 and train_multimod_labels is None:
        (input_gene_ids_train,
            input_values_train,
            target_values_train,
            tensor_batch_labels_train,
            tensor_celltype_labels_train,
                tensor_mod_types_train, indices_to_keep) = stratified_sample(
                input_gene_ids_train,
                input_values_train,
                target_values_train,
                tensor_batch_labels_train,
                tensor_celltype_labels_train,
                tensor_mod_types_train,
                total_samples=2000,
                min_per_class=1,
                seed=42
        )
    elif input_gene_ids_train.shape[0]>=2000 and train_multimod_labels is not None:
        (input_gene_ids_train,
            input_values_train,
            target_values_train,
            tensor_batch_labels_train,
            tensor_celltype_labels_train,
            tensor_multimod_labels_train,
                tensor_mod_types_train,
                indices_to_keep) = stratified_sample(
                input_gene_ids_train,
                input_values_train,
                target_values_train,
                tensor_batch_labels_train,
                tensor_celltype_labels_train,
                tensor_multimod_labels_train,
                tensor_mod_types_train,
                total_samples=2000,
                min_per_class=1,
                seed=42
        )
    else:
        indices_to_keep = list(range(input_gene_ids_train.shape[0]))
    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
        "indices_to_keep": indices_to_keep
    }

    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,

    }
    train_data_pt["mod_types"] = tensor_mod_types_train
    valid_data_pt["mod_types"] = tensor_mod_types_valid
    if train_multimod_labels is not None:
        train_data_pt["multimod_types"] = tensor_multimod_labels_train
        valid_data_pt["multimod_types"] = tensor_multimod_labels_valid
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
    # convert to numpy
    import pandas as pd
    entropy = torch.stack(entropy_list).cpu().numpy()
    correct_mask = torch.stack(correct_mask).cpu().numpy()

    # group by correct / incorrect
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

    # set plot style
    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # Boxplot: x=cell type, y=entropy, color indicates classification correctness
    ax = sns.boxplot(data=df, x="CellType", y="Entropy", hue="Correct", palette="Set2")

    plt.title(f"Entropy Distribution per Cell Type (Batch {test_batch_idx})", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # save figure
    plt.savefig(os.path.join(save_dir, f"entropy_boxplot_batch{test_batch_idx}.png"))

from torch import Tensor

# Reuse `plot_entropy_accuracy`, `reduce_proxies`, `CosineLinear`, and `ClsDecoder` from `tutorials.utils`

class CosineScheduleWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, min_lr=1e-6, last_epoch=-1):
        self.num_warmup_steps = int(num_warmup_steps)
        self.num_training_steps = int(num_training_steps)
        self.min_lr = float(min_lr)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.num_warmup_steps:
            return [base_lr * step / max(1, self.num_warmup_steps) for base_lr in self.base_lrs]
        else:
            progress = (step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            lrs = [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]
            return [max(self.min_lr, lr) for lr in lrs]
        

class ProtodistClassifier(nn.Module):
    def __init__(self):
        super(ProtodistClassifier, self).__init__()

    def forward(self, prototypes, cell_emb, celltype_labels):
        """
        prototypes: A tensor of shape [num_classes, feature_dim], representing the class prototypes.
        cell_emb: A tensor of shape [batch_size, feature_dim], representing the embeddings of the current samples.
        celltype_labels: A tensor of shape [batch_size], representing the true labels for the samples.

        Returns:
        loss: The negative log likelihood loss based on the predicted probabilities.
        log_p_y: A tensor of shape [batch_size, num_classes], representing the log-probability of each sample belonging to each class.
        """
        # compute Euclidean distance between current samples and class prototypes
        prototypes_tensor = torch.stack([v.squeeze(0) if v.dim() == 2 else v for v in prototypes.values()])
        dists = torch.cdist(cell_emb, prototypes_tensor)  # [batch_size, num_classes]
        
        # Compute per-class probabilities using negative distances (softmax on -dists)
        log_p_y = F.log_softmax(-dists, dim=1)
        
        # compute negative log-likelihood loss
        loss = F.nll_loss(log_p_y, celltype_labels)
        
        return loss, log_p_y
    
class ContinualClassify():
    def __init__(self, config, vocab, num_batch_types, genes, model_file, max_batch_idx=5, modeldict_name = "best_model.pt"):
        self.config = config
        self.vocab = vocab
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.num_max_classes = self.config["init_class"]
        self.vocab_mod = {'ADT': 0, 'ATAC': 1, 'RNA': 2, '<pad>': 3, '<cls>': 4, '<eoc>': 5}
        self.classifier = self.config["classifier"]
        self.model_file = model_file
        self.model, self.gene_ids = self.prepare_model(num_batch_types, self.num_max_classes, genes, modeldict_name)
        self.model.to(self.device)
        # self.model.cls_decoder = ClsDecoder(512, self.num_max_classes, classifier=self.classifier).to(self.device)
        # if self.config["load_model"] is not None:
        #     load_pretrained(self.model, torch.load(self.model_file), verbose=False)
        
        if self.config["freeze_except_layer1011"]:

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

        if self.config["freeze_lora"]:
            for name, param in self.model.named_parameters():
                if "lora_moe" in name:
                    param.requires_grad = False

        if self.config["adapter"]:
            for name, param in self.model.named_parameters():
                if "adapter" in name:
                    param.requires_grad = True
        self.criterion = masked_mse_loss

        # self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_cls_proto = ProtodistClassifier()

        self.criterion_dab = nn.CrossEntropyLoss()
        self.ppp_loss = PPPloss(net=self.model, mode="joint", T=0.8, tracker={'log_it': [], 'loss': [], 'lnL_pos': [], 'lnL_neg': []})
        
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=config["lr"], eps=1e-4 if config["amp"] else 1e-8
        # )
        if self.config["optimizer_lr"]:
            self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.cls_decoder.parameters(), "lr": self.config["lr"] * 10},
                {"params": [p for n, p in self.model.named_parameters() if "cls_decoder" not in n], "lr": self.config["lr"]},
            ], eps = 1e-8
        )
        # elif self.config["weight_decay"]:
        #     self.optimizer = torch.optim.Adam(
        #     self.model.parameters(), lr=config["lr"], eps=1e-4 if config["amp"] else 1e-8, weight_decay=1e-4
        # )
            # self.optimizer = torch.optim.Adam(
            #     filter(lambda p: p.requires_grad, self.model.parameters()), 
            #     lr=config["lr"], eps=1e-4 if config["amp"] else 1e-8, weight_decay=1e-4)
        else:
            self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config["lr"], eps=1e-4 if config["amp"] else 1e-8
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=config["schedule_ratio"]
        )
        if self.config["decrease_lr_all"]:
            warmup_steps = 2
            total_steps = self.config["epochs"] * (2000 // self.config["batch_size"]) *3
            self.scheduler = CosineScheduleWithWarmup(self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps, min_lr=1e-6)

        self.scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])
        self.past_data = None
        # self.past_model = copy.deepcopy(self.model)
        self.past_model = None
        self.old_model_annotation = None
        self.max_test_id = max_batch_idx
        self.old_proto = defaultdict()
        for class_idx in range(self.config["init_class"]):
            self.old_proto[class_idx] = torch.zeros(512).to(self.device)
        self.memory_proto = defaultdict(list)
        self.past_valid_loaders = {}
        # self.contrastive_proto_loss_list, self.repultion_loss_list = [], []

    def prepare_model(self, num_batch_types, num_max_classes, genes=None, modeldict_name="best_model.pt"):
        if self.config["loramoe"]:
            # from loramoemodel import TransformerModel, AdversarialDiscriminator   # loramoe
            from model.loramoemultiomics import MultiOmicTransformerModel, AdversarialDiscriminator   # loramoe
        else:
            # from scgpt.model import MultiOmicTransformerModel, AdversarialDiscriminator
            from multiomicmodel import MultiOmicTransformerModel, AdversarialDiscriminator
        # if self.config["load_model"] is not None:
        #     model_dir = Path(self.config["load_model"])
        #     weight_dir = Path(self.config["weight_dir"])
        #     model_config_file = model_dir / "args.json"
        #     model_file = weight_dir / modeldict_name
        #     vocab_file = model_dir / "vocab.json"

        #     self.vocab = GeneVocab.from_file(vocab_file)
        #     special_tokens = ["<pad>", "<cls>", "<eoc>"]
        #     for s in special_tokens:
        #         if s not in self.vocab:
        #             self.vocab.append_token(s)

        #     with open(model_config_file, "r") as f:
        #         model_configs = json.load(f)
        #     embsize = model_configs["embsize"]
        #     nhead = model_configs["nheads"]
        #     d_hid = model_configs["d_hid"]
        #     nlayers = model_configs["nlayers"]
        #     n_layers_cls = model_configs["n_layers_cls"]
        # else:
        embsize = self.config["layer_size"]
        nhead = self.config["nhead"]
        nlayers = self.config["nlayers"]
        d_hid = self.config["layer_size"]
        special_tokens = ["<pad>", "<cls>", "<eoc>"]
        pretrained_genes = [g for g in genes + special_tokens if g in self.vocab]        # 1052
        new_genes = [g for g in genes + special_tokens if g not in self.vocab]           # 1335
        gene_ids_pretrained = np.array(self.vocab(pretrained_genes), dtype=int)          # vocab.get_stoi() dictionary {'gene_name':id}
        # https://discuss.pytorch.org/t/expand-an-existing-embedding-and-linear-layer-nan-loss-value/55670/2
        # Retrieve pretrained weights
        combined_genes = pretrained_genes + new_genes            # append new genes to the existing token sequence
        unique_genes = list(dict.fromkeys(combined_genes))        # 1337
        # vocab = Vocab(VocabPybind(pretrained_genes + new_genes, None))
        self.vocab = Vocab(VocabPybind(unique_genes, None))            # new vocab containing many new tokens
        self.vocab.set_default_index(self.vocab["<pad>"])
        gene_ids = np.array(self.vocab(genes), dtype=int)    # 0-1336

        # gene_rna_df = pd.DataFrame(index = adata.var.index.tolist())  # create
        # gene_rna_df['mod'] = 'RNA'
        # if self.config["use_mod"]:
        #     mod_type = np.array([gene_loc_df.loc[g, 'mod'] for g in genes])    # 1368 RNA, Protein
        #     vocab_mod = Vocab(VocabPybind(np.unique(gene_loc_df['mod']).tolist() + special_tokens, None))  # [config["pad_token"], "<cls>", "<eoc>"]
        #     vocab_mod.set_default_index(vocab_mod["<pad>"])
        #     mod_type = np.array(vocab_mod(list(mod_type)), dtype=int)
        #     ntokens_mod = len(vocab_mod)

        # vocab_mod = {'<eoc>': 4, '<cls>': 3, 'RNA': 1, '<pad>': 2, 'Protein': 0, }
        # ntokens_mod = len(vocab_mod)
        
        pad_token = "<pad>"
        pad_value = -2
        explicit_zero_prob = False
            
        model = MultiOmicTransformerModel(
            len(self.vocab),
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=3,
            n_cls=num_max_classes,                  # initialize classifiers (num_classes)
            vocab=self.vocab,
            dropout=self.config["dropout"],
            pad_token=pad_token,
            pad_value=pad_value,
            do_mvc=self.config["MVC"],
            do_dab=self.config["do_dab"],
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
            use_mod=self.config["use_mod"],
            use_multimod=self.config["use_multimod"],
            ntokens_mod=len(self.vocab_mod) if self.config["use_mod"] else None,
            vocab_mod=self.vocab_mod if self.config["use_mod"] else None,
            num_multimod_labels= 2 if self.config["use_multimod"] else None,
            num_experts = self.config["num_of_expert"] if self.config["loramoe"] else None,
            device = self.device,
            adapter_dim = self.config["adapter_dim"],
        )
        # model.cls_decoder = ClsDecoder(512, self.config["init_class"]).to(self.device)
        if self.config["load_model"] is not None:
            model_dict = torch.load(self.model_file)
            # print(model_dict)
        if self.config["post_pretrained"]:
            load_pretrained(model, model_dict, verbose=False, strict = False)
        else:
            with torch.no_grad():
                pretrained_emb_weights = model_dict['encoder.embedding.weight'][gene_ids_pretrained, :]         # select pretrained weights for the gene ids, e.g. 1092 x 512
                print(model.encoder.embedding.weight.shape)  # 1337, 512
                print(len(pretrained_genes))           # 1109
                model.encoder.embedding.weight.data[:len(pretrained_genes), :] = pretrained_emb_weights     # 1337 x 512 encoder is the filtered embedding result
                model.encoder.enc_norm.weight.data = model_dict['encoder.enc_norm.weight']
                print('model.encoder.enc_norm.weight.data', model.encoder.enc_norm.weight.data.shape)

        for name, param in model.named_parameters():
            print(f"Parameter name: {name}, shape: {param.shape}")
        # print(model)
        return model, gene_ids

    def train(self, loader, logger, epoch, test_batch_idx):
        self.model.train()
        if self.past_model:
            self.past_model.eval()
        total_loss, total_cls, total_num = 0.0, 0.0, 0.0
        proto_loss = None
        cope_loss = None
        ortho_loss = torch.tensor(0.0, device=self.device)
        proto_contrastive_loss = None
        log_interval = 10
        start_time = time.time()
        num_batches = len(loader)
        cell_emb_list = []
        celltype_labels_list = []
        adata_train_indices_list = []
        train_iter_loss_list = []
        entropy_list, accuracy_list = [], []
        proto_loss_list = []
        repultion_loss_list = []
        proto_contrastive_loss_list = []
        
        for batch, batch_data in enumerate(itertools.islice(loader, num_batches)):
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            target_values = batch_data["target_values"].to(self.device)
            batch_labels = batch_data["batch_labels"].to(self.device)
            celltype_labels = batch_data["celltype_labels"].to(self.device)
            mod_types = batch_data["mod_types"].to(self.device)
            if "indices_to_keep" in batch_data:
                adata_train_indices = batch_data["indices_to_keep"].to(self.device)
            if self.config["use_multimod"]:
                multimod_labels = batch_data["multimod_types"].to(self.device)
            src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])
            # print('test_batch_idx', test_batch_idx)
            # print('*************************', type(test_batch_idx), '**********************')
            with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                if self.config["adapter"] or self.config["loramoe"]:
                    output_dict, _ = self.model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                        # batch_id = torch.tensor(test_batch_idx),
                        batch_id = None,
                        multimod_labels=multimod_labels if self.config["use_multimod"] else None,
                        CLS=True,
                        CCE=False,
                        MVC=self.config["MVC"],
                        ECS=self.config["ecs_thres"] > 0,
                        mod_types=mod_types if self.config["use_mod"] else None,
                        do_sample=False,
                    )
                else:
                    output_dict = self.model(
                        input_gene_ids,
                        input_values,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                        # batch_labels=test_batch_idx,
                        multimod_labels=multimod_labels if self.config["use_multimod"] else None,
                        CLS=True,
                        CCE=False,
                        MVC=self.config["MVC"],
                        ECS=self.config["ecs_thres"] > 0,
                        mod_types=mod_types if self.config["use_mod"] else None,
                        do_sample=False,
                    )
                cell_emb = output_dict["cell_emb"].squeeze(1)
                cell_emb = F.normalize(cell_emb, p=2, dim=1)
                # cell_emb = torch.cat([cell_emb_past, cell_emb], dim=-1)
                # cell_emb = output_dict["cls_output"][1]
                # cell_emb_list.append(cell_emb.detach().cpu())
                cell_emb_list.append(cell_emb)
                celltype_labels_list.append(celltype_labels.detach().cpu())
                adata_train_indices_list.append(adata_train_indices.detach().cpu())
                masked_positions = input_values.eq(-1)
                # metrics_to_log = {}
                total_num += len(input_gene_ids)

                if self.config["loss_func"] == "cross_entropy":
                    output_values = output_dict["cls_output"].detach()
                    loss_cls = self.criterion_cls(output_dict["cls_output"], celltype_labels)
                    loss = loss_cls
                    # print(f"loss requires_grad? {loss.requires_grad}")
                    probs = F.softmax(output_values, dim=-1)
                # elif self.loss_func == "proto_dist":
                else:
                    if epoch == 1 and test_batch_idx == 0:
                        output_values = output_dict["cls_output"].detach()
                        loss_cls = self.criterion_cls(output_dict["cls_output"], celltype_labels)
                        loss = loss_cls
                        # print(f"loss requires_grad? {loss.requires_grad}")
                        probs = F.softmax(output_values, dim=-1)
                    else:
                        loss_cls, log_p_y = self.criterion_cls_proto(self.old_proto, cell_emb, celltype_labels)
                        loss = loss_cls
                        probs = torch.exp(log_p_y)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)                 # 
                entropy_list.extend(entropy.detach().cpu())
                preds = probs.argmax(1)
                accuracy_list.extend((preds == celltype_labels.detach()).cpu())
                # train_loss_list.append(loss.item())
                # metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (preds == celltype_labels.detach())
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
                train_iter_loss_list.append(loss_cls.item())
                # compute MSE between past model and current model
                if self.past_model and test_batch_idx !=0 and self.config["pastmse"]:
                    with torch.no_grad():
                        past_output_dict = self.past_model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=self.config["MVC"],
                            ECS=self.config["ecs_thres"] > 0,
                            do_sample=False,
                        )
                        cell_emb_past = past_output_dict["cell_emb"].squeeze(1)
                        # loss_past = self.criterion_cls(past_output_dict["cls_output"], celltype_labels)     # this past model's cls_decoder is randomly initialized, result is invalid
                        ################### add classification loss after concatenation ########################
                 
                    # mse_past_current = F.mse_loss(output_dict["cls_output"], past_output_dict["cls_output"])
                    mse_past_current = F.mse_loss(cell_emb, cell_emb_past)
                    loss += mse_past_current
                    # loss += loss_past
                    # metrics_to_log.update({"train/mse_past_current": mse_past_current.item()})
                # if test_batch_idx != 0 and self.config["proto_loss"]:
                if self.config["proto_loss"] and len(self.old_proto) != 0:
                    print('length of old_proto', len(self.old_proto))
                    # proto_loss = self.contrastive_proto_loss(self.old_proto, self.model.cls_decoder.out_layer.weight)
                    # proto_loss = self.cope_ppp_loss(cell_emb, celltype_labels, self.old_proto)
                    
                    ################################
                    proto_loss = self.ppp_loss(cell_emb, celltype_labels, self.old_proto, self.memory_proto, self.device, eps=1e-8)
                    loss = loss + proto_loss * self.config["proto_weight"]

                    # if self.config["lora_ortho"]:
                    #     for i in range(12):
                    #         ortho_loss += self.model.transformer_encoder.layers[i].lora_moe.compute_orthogonal_loss(test_batch_idx)
                    #     loss += ortho_loss
                    # grads = torch.autograd.grad(proto_loss, self.model.transformer_encoder.layers[0].self_attn.in_proj_weight, retain_graph=True)
                    # print(grads)
                    ################################
                    # proto_loss = torch.zeros(1, device=self.device)
                    # print(f"loss requires_grad???? {loss.requires_grad}")
                    proto_loss_list.append((proto_loss * self.config["proto_weight"]).item())
                    if  len(self.memory_proto) != 0 and self.config["contrastive_proto_loss"]:
                        proto_contrastive_loss = self.contrastive_proto_loss(cell_emb, celltype_labels, self.memory_proto, self.old_proto)
                        loss = loss + proto_contrastive_loss * 10
                        proto_contrastive_loss_list.append(proto_contrastive_loss.item())
                        # grads = torch.autograd.grad(proto_contrastive_loss, self.model.transformer_encoder.layers[0].self_attn.in_proj_weight, retain_graph=True)

                if self.config["repultion_loss"]:
                    repulsion_loss = self.repultion_loss(self.model.cls_decoder.out_layer.weight.data)
                    loss += repulsion_loss
                    repultion_loss_list.append(repulsion_loss.item())
                    # print(f"repulsion_loss: {repulsion_loss.item()}")

                    # metrics_to_log.update({"train/repulsion_loss": repulsion_loss.item()})
                if self.config["weight_miloss"]:
                    weight_miloss = self.extended_mutual_information_loss_weighted(output_values, class_weights=None, lambda_weight=1.0, eps=1e-8)
                    loss = loss + weight_miloss

                if self.config["anchorloss"]:
                    anchor_loss = AdvancedAnchorLoss()(cell_emb, celltype_labels)
                    loss += 10 * anchor_loss
                    print('anchor_loss:', 10*anchor_loss)
                    # self.
                
                print(f"train/cls: {loss_cls.item()},"
                      f"proto_loss: {proto_loss.item() if isinstance(proto_loss, torch.Tensor) else None},"
                      f"contrastive_loss:{10*proto_contrastive_loss.item()if isinstance(proto_contrastive_loss, torch.Tensor) else None},"
                       f"total_loss: {loss.item()}")
            self.model.zero_grad()
            self.optimizer.zero_grad()
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
                # lr = self.scheduler.get_last_lr()[0]
                lr = self.optimizer.param_groups[0]['lr']
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                # cur_loss = total_loss / log_interval
                # cur_cls = total_cls / log_interval if True else 0.0
                cur_loss = total_loss / total_num
                cur_cls = total_cls / total_num 
                logger.info(
                    f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                    f"lr {lr:0.8f} | ms/batch {ms_per_batch:5.2f} | "
                    f"cls loss {cur_cls:5.2f} | "
                    f"loss {cur_loss:5.2f} |"
                )
                # total_loss = 0
                # total_cls = 0
                start_time = time.time()
            torch.cuda.empty_cache()
        cur_loss = total_loss / total_num    # current epoch loss
        proto_loss_last = sum(proto_loss_list) / total_num
        repultion_loss = sum(repultion_loss_list) / total_num
        contrastive_proto_loss = sum(proto_contrastive_loss_list) / total_num
        del proto_loss, total_loss
        gc.collect()     # collect Python garbage
        torch.cuda.empty_cache()  # release unused PyTorch GPU cache
        return cell_emb_list, celltype_labels_list, adata_train_indices_list, cur_loss, train_iter_loss_list, proto_loss_last, contrastive_proto_loss, entropy_list, accuracy_list
    
    def evaluate(self, loader, epoch, save_dir, test_batch_idx):
        self.model.eval()
        total_loss = 0.0
        total_proto_loss = 0.0
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
                mod_types = batch_data["mod_types"].to(self.device)
                if self.config["use_multimod"]:
                    multimod_labels = batch_data["multimod_types"].to(self.device)
                src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

                with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                    if self.config["adapter"] or self.config["loramoe"]:
                        output_dict, _ = self.model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                            # batch_labels= test_batch_idx if False or self.config["DSBN"] else None,
                            # batch_id = torch.tensor(test_batch_idx),
                            batch_id = None,
                            multimod_labels=multimod_labels if self.config["use_multimod"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=self.config["MVC"],
                            ECS=self.config["ecs_thres"] > 0,
                            mod_types=mod_types if self.config["use_mod"] else None,
                            do_sample=False,
                        )
                    else:
                        output_dict = self.model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                            # batch_labels= test_batch_idx if False or self.config["DSBN"] else None,
                            multimod_labels=multimod_labels if self.config["use_multimod"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=False,
                            ECS=False,
                            mod_types=mod_types if self.config["use_mod"] else None,
                            do_sample=False,
                        )
                    cell_emb = output_dict["cell_emb"].squeeze(1)
                    cell_emb = F.normalize(cell_emb, p=2, dim=1)
                    
                    if self.config["loss_func"] == "cross_entropy":
                        output_values = output_dict["cls_output"]
                        probs = F.softmax(output_dict["cls_output"], dim=-1)
                        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                        loss = self.criterion_cls(output_values, celltype_labels)
                    else:
                        if epoch == 1 and test_batch_idx == 0:
                            output_values = output_dict["cls_output"]
                            probs = F.softmax(output_dict["cls_output"], dim=-1)
                            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                            loss = self.criterion_cls(output_values, celltype_labels)
                        else:
                            loss, log_p_y = self.criterion_cls_proto(self.old_proto, cell_emb, celltype_labels)
                            probs = torch.exp(log_p_y)
                            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                    if len(self.old_proto) != 0 and self.config["proto_loss"]:
                        proto_loss = self.ppp_loss(cell_emb, celltype_labels, self.old_proto, self.memory_proto, self.device, eps=1e-8)
                        total_proto_loss += proto_loss.item() 
                   
                eval_iter_loss_list.append(loss)
                entropy_list.extend(entropy)
                accuracy_list.extend(probs.argmax(1) == celltype_labels)
                total_loss += loss.item() * len(input_gene_ids)
                
                correct = (probs.argmax(1) == celltype_labels).sum().item()
                total_error += len(input_gene_ids) - correct
                accuracy += correct
                total_num += len(input_gene_ids)
                preds = probs.argmax(1).detach().cpu().numpy()
                predictions.extend(preds)
                labellist.extend(celltype_labels.detach().cpu().numpy())

        # compute F1 scores
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
            class_names = np.unique(np.concatenate((labellist, predictions)))  # get all class names
            cm = confusion_matrix(labellist, predictions, labels=class_names)

            # plot heatmap

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.savefig(str(save_dir) + "/" + f"batch_{test_batch_idx}_Confusion_Matrix.png")
            plot_entropy_accuracy(entropy_list, accuracy_list, save_dir, test_batch_idx, labellist)    #  plot entropy vs. accuracy

        return total_loss / total_num, total_error / total_num, result_dict, eval_iter_loss_list, total_proto_loss / total_num

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
            max_len=3061,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        print('test_batch_idx', test_batch_idx)
        
        # print('*************************', type(test_batch_idx), '**********************')
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):

            if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                cell_embeddings = best_model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels = torch.from_numpy(np.full_like(batch_ids, test_batch_idx)).long() if self.config["DSBN"] else None,
                    # batch_labels=torch.from_numpy(np.array([test_batch_idx, test_batch_idx])),
                    # batch_labels = None,
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
                    # batch_labels = None,
                    batch_labels=torch.from_numpy(np.full_like(batch_ids, test_batch_idx)).long() if self.config["DSBN"] else None,
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
    
    def eval_testdata(self, model, adata_t, gene_ids, input_layer_key, test_batch_idx=None):
        # evalmodel = copy.deepcopy
        model.eval()
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
            max_len=3061,
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
                cell_embeddings = model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                    # batch_labels=torch.from_numpy(test_batch_idx).long() if self.config["DSBN"] else None,
                    # batch_id = torch.tensor(test_batch_idx),
                    batch_id=None,
                    time_step=0,
                    return_np=True,
                )
            else:
                cell_embeddings = model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                    # batch_labels=torch.from_numpy(test_batch_idx).long() if self.config["DSBN"] else None,
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
        Update classifier weights by class embedding averages
        :param embedding_list: list of embeddings for all samples
        :param label_list: list of labels for all samples
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
                # print()

    # def update_prototype(self, embedding_list, label_list, epoch = None):
    #     embedding_list = torch.cat(embedding_list, dim=0)
    #     label_list = torch.cat(label_list, dim=0)

    #     class_list = torch.unique(label_list).cpu().numpy()
    #     for class_index in class_list:
    #         data_index = (label_list == class_index).nonzero(as_tuple=True)[0] 
    #         embedding = embedding_list[data_index]
    #         new_proto = embedding.mean(0).float().detach()
    #         # self.model.cls_decoder.out_layer.weight.data[int(class_index)] = proto.to(self.device)
    #         # if self.config["cope_loss"]:
    #         # self.old_proto[int(class_index)] = proto.clone()
    #         proto = self.old_proto.get(int(class_index), torch.zeros(512).to(self.device))

    #         if self.config["proto_loss"] and self.config["ema"]:
    #             # if prototype not initialized, assign directly
    #             if proto.sum() == 0:
    #                 self.old_proto[int(class_index)] = new_proto.clone()
    #             else:
    #                 # EMA update prototype
    #                 ema_momentum = 0.5
    #                 self.old_proto[int(class_index)] = (
    #                     ema_momentum  * self.old_proto[int(class_index)] +
    #                     (1 - ema_momentum)* new_proto
    #                 )
    #         else:
    #             self.old_proto[int(class_index)] = new_proto.clone()
    #         self.old_proto[int(class_index)] = F.normalize(self.old_proto[int(class_index)], p=2, dim=0)
    #         if epoch != None:
    #             self.memory_proto[int(class_index)].append(F.normalize(self.old_proto[int(class_index)], p=2, dim=0))
    def update_prototype(self, embedding_list, label_list, epoch=None):
        # for idx, emb in enumerate(embedding_list):
        #     print(f"--- emb[{idx}] ---")
        #     print("requires_grad:", emb.requires_grad)
        #     print("grad_fn:", emb.grad_fn)
        # concatenate embeddings and labels within the batch
        embedding_list = torch.cat(embedding_list, dim=0).to(self.device)   # [N, D]
        label_list = torch.cat(label_list, dim=0).to(self.device)           # [N]

        class_list = torch.unique(label_list).cpu().numpy()
        for class_index in class_list:
            # find indices for the current class
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0]  
            embedding = embedding_list[data_index]  # [num_samples, D]

            # compute new prototype directly, keep computation graph
            new_proto = embedding.mean(0).float()  # no longer detach
            # print(embedding.requires_grad)  # check if True
            # print(new_proto.requires_grad)  # check if True
            # if prototype doesn't exist, initialize directly
            proto = self.old_proto.get(int(class_index), torch.zeros_like(new_proto).to(self.device))

            if self.config["proto_loss"] and self.config["ema"]:
                # EMA update, keep computation graph
                if proto.sum() == 0:
                    self.old_proto[int(class_index)] = new_proto.clone()
                else:
                    ema_momentum = 0.95
                    self.old_proto[int(class_index)] = (
                        ema_momentum * proto + (1 - ema_momentum) * new_proto
                    )
            else:
                self.old_proto[int(class_index)] = new_proto.clone()

            # normalize, keep computation graph
            self.old_proto[int(class_index)] = F.normalize(self.old_proto[int(class_index)], p=2, dim=0)

            # memory_proto can directly use old_proto, keep graph
            if epoch is not None:
                if int(class_index) not in self.memory_proto:
                    self.memory_proto[int(class_index)] = []
                self.memory_proto[int(class_index)].append(self.old_proto[int(class_index)])

    def update_prototype_with_entropy(self, embedding_list, label_list, entropy_list, accuracy_list):
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        entropy_list = torch.stack(entropy_list)
        accuracy_list = torch.tensor(accuracy_list, dtype=torch.bool, device=embedding_list.device)

        class_list = torch.unique(label_list).cpu().numpy()
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0] 

            # extract entropy, accuracy, and embedding for this class
            class_entropy = entropy_list[data_index]
            class_accuracy = accuracy_list[data_index]
            class_embedding = embedding_list[data_index]
            # find indices of correctly classified samples
            correct_mask = class_accuracy.bool()
            arange_indices = torch.arange(len(data_index), device=correct_mask.device)
            correct_indices = arange_indices[correct_mask]

            if correct_indices.numel() >= 1:
                # keep only entropies for correctly classified samples
                filtered_entropy = class_entropy[correct_indices]
                filtered_embedding = class_embedding[correct_indices]

                # find top-k indices with smallest entropy
                if self.config["correct_sample"]:
                    selected_embedding = filtered_embedding
                else:
                    topk = min(10, filtered_entropy.size(0))
                    _, top_indices = torch.topk(-filtered_entropy, topk)  # -entropy: sort ascending

                    selected_embedding = filtered_embedding[top_indices]
                
                print('entropy filter samples of class:{}'.format(class_index), selected_embedding.shape[0])
                # compute prototype
                # proto = selected_embedding.mean(0).float()
                # assert proto.shape[0] == self.model.cls_decoder.out_layer.weight.shape[1]

                # update this class's prototype (out_layer weights)
                # with torch.no_grad():
                #     self.model.cls_decoder.out_layer.weight[int(class_index)] = proto.to(self.device)
                # if self.config["proto_loss"]:
                #     self.old_proto[int(class_index)] = proto.clone()
                new_proto = selected_embedding.mean(0).float()
                proto = self.old_proto.get(int(class_index), torch.zeros(512).to(self.device))

                if self.config["proto_loss"] and self.config["ema"]:
                    # if prototype not initialized, assign directly
                    if proto.sum() == 0:
                        self.old_proto[int(class_index)] = new_proto.clone()
                    else:
                        # EMA update prototype
                        ema_momentum = 0.5
                        self.old_proto[int(class_index)] = (
                            (1 - ema_momentum) * self.old_proto[int(class_index)] +
                            ema_momentum * new_proto
                        )
                else:
                    self.old_proto[int(class_index)] = new_proto.clone()
            else:
                # skip update if no correctly classified samples
                continue

    def update_classifier_weights_entropy(self, embedding_list, label_list, entropy_list, accuracy_list):
        embedding_list = torch.cat(embedding_list, dim=0)    # shape: [N, D]
        label_list = torch.cat(label_list, dim=0)            # shape: [N]
        entropy_list = torch.stack(entropy_list)
        # entropy_list = torch.cat(entropy_list, dim=0)        # shape: [N]
        accuracy_list = torch.tensor(accuracy_list, dtype=torch.bool, device=embedding_list.device)    # shape: [N], bool or int (0/1)

        class_list = torch.unique(label_list).cpu().numpy()

        for class_index in class_list:
            # find all sample indices for this class
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0]
            
            # extract entropy, accuracy, and embedding for this class
            class_entropy = entropy_list[data_index]
            class_accuracy = accuracy_list[data_index]
            class_embedding = embedding_list[data_index]

            # find indices of correctly classified samples
            correct_mask = class_accuracy.bool()
            arange_indices = torch.arange(len(data_index), device=correct_mask.device)
            correct_indices = arange_indices[correct_mask]


            if correct_indices.numel() >= 1:
                # keep only entropies from correctly classified samples
                filtered_entropy = class_entropy[correct_indices]
                filtered_embedding = class_embedding[correct_indices]

                # find top-k indices with smallest entropy
                topk = min(10, filtered_entropy.size(0))
                _, top_indices = torch.topk(-filtered_entropy, topk)  # -entropy: sort ascending (smallest to largest)

                selected_embedding = filtered_embedding[top_indices]
                print('entropy filter samples of class:{}'.format(class_index), selected_embedding.shape[0])
                # compute prototype
                proto = selected_embedding.mean(0).float().detach()
                assert proto.shape[0] == self.model.cls_decoder.out_layer.weight.shape[1]

                # update prototype for this class (out_layer weights)
                with torch.no_grad():
                    self.model.cls_decoder.out_layer.weight[int(class_index)] = proto.to(self.device)
                if self.config["proto_loss"]:
                    self.old_proto[int(class_index)] = proto.clone()
                    
            else:
                # skip update if there are no correctly classified samples
                continue
        print('#########################', len(self.old_proto))

    def build_fewshot_dataset(self, all_counts, celltypes_labels, batch_ids, shots_per_class=5, seed=42):
        np.random.seed(seed)

        label_to_indices = defaultdict(list)

        # collect indices for each class
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

        # convert to numpy array
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        print("number of train samples", train_indices.shape)
        print("number of val samples", val_indices.shape)
        # split into parts
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
        p = F.softmax(prob_logits, dim=1)     # shape: [B, K]   cluster probability distribution
        p = p + eps  # numerical stability
        log_p = torch.log(p)

        cond_entropy = -torch.mean(torch.sum(p * log_p, dim=1))  # scalar conditional entropy

        p_mean = torch.mean(p, dim=0)  # shape: [K]
        log_p_mean = torch.log(p_mean + eps)

        marginal_entropy = -torch.sum(p_mean * log_p_mean)  # scalar marginal distribution over classes

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
                    repulsion_loss += torch.exp(sim)  # the more similar, the larger the penalty
        repulsion_loss /= (new_weight_matrix.size(0) * (new_weight_matrix.size(0) - 1))
        return repulsion_loss
    def cope_ppp_loss(self, features, labels, prototypes, temperature=0.1):
        """
        CoPE PPP Loss (only for classes that exist in prototypes).

        Args:
            features: Tensor [B, D] — embeddings for current batch (should be normalized)
            labels: Tensor [B] — sample labels
            prototypes: defaultdict(int -> Tensor[D]) — prototypes for each class
            temperature: float — temperature parameter τ

        Returns:
            Scalar loss tensor
        """

        device = features.device
        B, D = features.shape
        all_class_ids = list(prototypes.keys())

        if len(all_class_ids) == 0:
            return torch.tensor(0.0, device=device)

        # all prototype vectors normalized into matrix [C, D]
        all_protos = torch.stack([F.normalize(prototypes[c].to(device), dim=0) for c in all_class_ids]).detach()

        total_loss = 0.0
        count = 0

        for cls in all_class_ids:
            cls = int(cls)
            # mask for current class
            mask_pos = (labels == cls)
            mask_neg = (labels != cls)

            # samples of the current class and their embeddings
            pos_feats = features[mask_pos]  # [N_pos, D]
            neg_feats = features[mask_neg]  # [N_neg, D]

            if pos_feats.size(0) < 1:
                continue # need at least two positive-class samples to form an attractor

            pc = F.normalize(prototypes[cls].to(device), dim=0).detach()  # [D]

            for i in range(pos_feats.size(0)):
                xi = pos_feats[i]  # current positive sample

                # -------- Attractor Set --------
                pseudo_protos = [pos_feats[j].detach() for j in range(pos_feats.size(0)) if j != i]
                attractor_set = [pc] + pseudo_protos
                attractor_set = torch.stack(attractor_set)  # [K, D]
                sim_pos = torch.matmul(attractor_set, xi) / temperature  # [K]
                sim_soft = F.softmax(sim_pos, dim=0)
                pos_prob = sim_soft[0]  # positive class prob is first

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
    
    def contrastive_proto_loss(self, features, labels, old_protos, new_protos, temperature=0.1):
        """
        Args:
            old_protos: Dict[int, Tensor] — historical prototypes
            new_weight_matrix: Tensor — current model classifier weights (num_classes x dim)
        Returns:
            torch.Tensor — contrastive loss supporting gradient backpropagation
        """
        # device = new_protos.device
        losses = []

        for i, proto_old_list in old_protos.items():
            proto_new = new_protos[i].to(self.device)  # new protypes 

            for proto_old in proto_old_list:  # iterate over all historical prototypes for this class
                proto_old = proto_old.to(self.device)

                # similarity among positive samples
                pos_sim = F.cosine_similarity(proto_old.unsqueeze(0), 
                                            proto_new.unsqueeze(0), dim=-1) / temperature

                # similarity to negative samples (new prototypes of other classes)
                neg_sims = []
                for j, neg_proto in new_protos.items():
                    if j != i:  # only take different classes
                        neg_proto = neg_proto.to(self.device)
                        neg_sim = F.cosine_similarity(proto_old.unsqueeze(0), 
                                                    neg_proto.unsqueeze(0), dim=-1) / temperature
                        neg_sims.append(neg_sim)

                if len(neg_sims) == 0:
                    continue  # skip when there's only one class

                neg_sims = torch.cat(neg_sims, dim=0)  # (num_classes - 1,)
                logits = torch.cat([pos_sim, neg_sims], dim=0)  # (num_classes,)
                labels = torch.zeros(1, dtype=torch.long, device=self.device)  # positive sample is at index 0

                # InfoNCE loss
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(losses).mean()

    def process_batch(self, adata_train, logger, save_dir, dataset_name, experiment_name, all_adata_test, all_batch_results, test_batch_idx, mod_type):
        '''
        adata_train, config, logger, save_dir, config["dataset_name"], config["experiment_name"], all_adata_test, all_batch_results,
            test_batch_idx
        '''
        # if self.config["freeze_after_batch1"] and test_batch_idx > 0:
        #     for name, param in self.model.named_parameters():
        #         param.requires_grad = False
        #         if 'decoder' in name:
        #             param.requires_grad = True
        #         if 'out_layer' in name:
        #             param.requires_grad = True  
        learning_rate = []
        if self.config["freeze_lora"]:
            for name, param in self.model.named_parameters():
                if f'lora_moe.lora_down.{test_batch_idx+1}' in name or f'lora_moe.lora_up.{test_batch_idx+1}' in name or 'lora_moe.gate' in name:
                    param.requires_grad = True  
                elif f'lora_moe.lora_down.0' in name or f'lora_moe.lora_up.0' in name:
                    param.requires_grad = True
                elif f'lora_moe.lora_down.0' not in name or f'lora_moe.lora_up.0' not in name and f'lora_moe.lora_down' in name and f'lora_moe.lora_up' in name:
                    param.requires_grad = False
                    pass
                # if 'decoder' in name:
                #     param.requires_grad = True 
                # if 'out_layer' in name:
                #     param.requires_grad = True 
            
        # elif self.config["freeze_lora"] and test_batch_idx > 0:
        #     for name, param in self.model.named_parameters():
        #         if f'lora_moe.lora_down.{test_batch_idx}' in name or f'lora_moe.lora_up.{test_batch_idx}' in name or 'lora_moe.gate' in name:
        #             param.requires_grad = True  
        #         else:
        #             param.requires_grad = False
        #         # if 'decoder' in name:
        #         #     param.requires_grad = True     # decoder also frozen
        #         if 'out_layer' in name:
        #             param.requires_grad = True 
        # else:
        #     pass
        
        if self.config["init_decoder"]:
            self.model.cls_decoder = ClsDecoder(512, self.config["init_class"]).to(self.device)

        if self.config["add_decoder"]:
            self.model.cls_decoder.add_new_decoder()
            for name, param in self.model.named_parameters():
                for decoder_idx in range(test_batch_idx + 1):
                    if decoder_idx == test_batch_idx+1 and f'cls_decoder.decoders.{test_batch_idx+1}' in name:
                        param.requires_grad = True
                    elif decoder_idx != test_batch_idx+1 and f'cls_decoder.decoders.{test_batch_idx+1}' not in name and \
                        f'cls_decoder.decoders' in name and f'cls_decoder.decoders.0' not in name:
                        param.requires_grad = False
                    elif f'cls_decoder.decoders.0' in name:
                        param.requires_grad = True
                    else:
                        pass

        for name, param in self.model.named_parameters():
            print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        from functions.memory_bank import example_bank_update
        ############################# Prepare data #############################
        # Count number of each cell type
        if self.config["init_class"] == 8 or self.config['fewshot'] is not None:
        # if self.config["filter_sample"]:
            celltype_counts = adata_train.obs["celltype"].value_counts()

            # Find cell types with count >= fewshot + 1
            valid_celltypes = celltype_counts[celltype_counts >= self.config["fewshot"] + 1].index

            # Filter out samples for cell types with fewer than the required samples
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
        # genes = adata_train.var["gene_name"].tolist()

        # Compute cell types and labels for the current batch (test_batch_idx)
        current_label_dict, current_celltype_labels = np.unique(
            np.array(adata_train.obs["celltype"].tolist()), return_inverse=True
        )

        adata_train.obs["celltype_labels"] = current_celltype_labels  # temporarily store labels

        if test_batch_idx == 0:
            # Initialization: use current batch information
            new_model_annotation = current_label_dict
            self.old_model_annotation = current_label_dict

            # Mapping table: celltype -> label
            celltype_to_label = {ct: i for i, ct in enumerate(self.old_model_annotation)}
            adata_train.obs["celltype_labels"] = adata_train.obs["celltype"].map(celltype_to_label)
            celltypes_labels = adata_train.obs["celltype_labels"].to_numpy()
            with open(save_dir / "celltype_to_label.json", 'w') as file:
                json.dump(celltype_to_label, file)
            # Define initial number of classifier heads
            # self.model.cls_decoder.out_layer.out_features = len(self.old_model_annotation)
            if self.config["replay"]:
                example_bank_previous = sc.AnnData()
        else:
            # Incremental batch
            new_model_annotation = current_label_dict

            # Find newly added cell types
            new_to_add = [ct for ct in new_model_annotation if ct not in self.old_model_annotation]

            # Generate full list of cell types
            combined = np.concatenate([self.old_model_annotation, new_to_add])
            self.old_model_annotation = combined

            # Update mapping table: celltype -> label (keep consistent indices)
            celltype_to_label = {ct: i for i, ct in enumerate(self.old_model_annotation)}
            with open(save_dir / "celltype_to_label.json", 'w') as file:
                json.dump(celltype_to_label, file)
            # Reassign labels to ensure consistency with previous data
            adata_train.obs["celltype_labels"] = adata_train.obs["celltype"].map(celltype_to_label)
            celltypes_labels = adata_train.obs["celltype_labels"].to_numpy()
            # Update number of classifier heads
            # self.model.cls_decoder.out_layer.out_features = len(self.old_model_annotation)
            if self.config["replay"]:
                example_bank_previous = torch.load(save_dir / "example_bank.pth")['example_bank']                         # previously stored samples
                # memory bank labels and batch IDs
                example_bank_previous_label = [
                    list(self.old_model_annotation).index(i)
                    for i in np.array(example_bank_previous.obs['celltype'])
                ]
                example_bank_previous_batch = np.array(example_bank_previous.obs['batch_id'])
                if self.config["use_multimod"]:
                    example_bank_previous_multimod = np.array(example_bank_previous.obs['modality_id'])
                # adata_train = anndata.concat([adata_train, example_bank_previous], axis=0, merge='same', label=None, keys=None)
                # example_bank_previous_label = [list(self.old_model_annotation).index(i) for i in np.array(example_bank_previous.obs['celltype'])]
                # celltypes_labels = np.concatenate([celltypes_labels, np.array(example_bank_previous_label)], 0)
                # all_counts = (
                #     adata_train.layers[input_layer_key].A
                #     if issparse(adata_train.layers[input_layer_key])
                #     else adata_train.layers[input_layer_key]
                # )
    ############################# Then handle model ###############################
        if not self.config["add_decoder"]:
            old_out_layer = self.model.cls_decoder.out_layer
            old_out_features = old_out_layer.out_features
            in_features = old_out_layer.in_features
            new_out_features = len(self.old_model_annotation)
            if new_out_features > old_out_features:
                # Create a larger out_layer
                if self.config["classifier"] == "Linear":
                    new_out_layer = nn.Linear(in_features, new_out_features).to(self.device)
                    
                    # Transfer existing weights
                    with torch.no_grad():
                        new_out_layer.weight[:old_out_features] = old_out_layer.weight
                        new_out_layer.bias[:old_out_features] = old_out_layer.bias
                else:
                    new_out_layer = CosineLinear(in_features, new_out_features).to(self.device)
                    with torch.no_grad():
                        new_out_layer.weight[:old_out_features] = old_out_layer.weight
                        new_out_layer.sigma.data = copy.deepcopy(old_out_layer.sigma.data)
                        # new_out_layer.bias[:old_out_features] = old_out_layer.bias
                
                # Replace classifier head
                self.model.cls_decoder.out_layer = new_out_layer
        ##########################################################################
            optimizer_param_ids = set(p.data_ptr() for group in self.optimizer.param_groups for p in group['params'])

            # Collect parameters of cls_decoder.out_layer
            missing_params = []
            for name, param in self.model.cls_decoder.out_layer.named_parameters():
                if param.requires_grad and param.data_ptr() not in optimizer_param_ids:
                    print(f"Param '{name}' not in optimizer. Will add.")
                    missing_params.append(param)

            # If there are missing params, add them to the optimizer
            if missing_params:
                self.optimizer.add_param_group({'params': missing_params})
        else:
            pass
        ############################ past_model ################################
        # if self.past_model is not None:
        #     old_out_layer = self.past_model.cls_decoder.out_layer
        #     old_out_features = old_out_layer.out_features
        #     in_features = old_out_layer.in_features
        #     new_out_features = len(self.old_model_annotation)
        #     if new_out_features > old_out_features:
        #         # Create a larger out_layer
        #         new_out_layer = nn.Linear(in_features, new_out_features).to(self.device)
                
        #         # Transfer existing weights
        #         with torch.no_grad():
        #             new_out_layer.weight[:old_out_features] = old_out_layer.weight
        #             new_out_layer.bias[:old_out_features] = old_out_layer.bias
                
        #         # Replace classifier head
        #         self.past_model.cls_decoder.out_layer = new_out_layer
        #############################################
        batch_ids = adata_train.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        if self.config["use_multimod"]:
            modality_to_id = {mod: i for i, mod in enumerate(adata_train.obs["modality"].unique())}
            adata_train.obs["modality_id"] = adata_train.obs["modality"].map(modality_to_id).astype(int)
            multimods_ids = adata_train.obs["modality_id"].tolist()
            num_multimods = len(set(multimods_ids))
            multimods_ids = np.array(multimods_ids)

        if self.config["randomsplit"]:
            if self.config["use_multimod"]:
                (
                    train_data,
                    valid_data,
                    train_celltype_labels,
                    valid_celltype_labels,
                    train_batch_labels,
                    valid_batch_labels,
                    train_multimod_labels,
                    valid_multimod_labels
                ) = train_test_split(
                    all_counts, celltypes_labels, batch_ids, multimods_ids, test_size=self.config["valid_ratio"], shuffle=True
                )
            else:
                (
                    train_data,
                    valid_data,
                    train_celltype_labels,
                    valid_celltype_labels,
                    train_batch_labels,
                    valid_batch_labels,
                ) = train_test_split(
                    all_counts, celltypes_labels, batch_ids, test_size=self.config["valid_ratio"], shuffle=True
                )
            # First, split indices
            adata_indices = np.arange(len(adata_train))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=self.config["valid_ratio"],
                shuffle=True,
                stratify=celltypes_labels if not self.config["randomsplit"] else None,
            )

            # Subset adata_train using the indices
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
                    celltypes_labels,
                    batch_ids,
                    multimods_ids,
                    test_size=self.config["valid_ratio"],
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
                    celltypes_labels,
                    batch_ids,
                    test_size=self.config["valid_ratio"],
                    min_class_size=20
                )
            adata_train_split = adata_train[train_indices].copy()
            adata_valid_split = adata_train[val_indices].copy() 
            if self.config['replay'] and test_batch_idx != 0:
                # Append memory bank samples to the training set
                train_data = np.concatenate(
                    [train_data,
                    example_bank_previous.layers[input_layer_key].A
                    if issparse(example_bank_previous.layers[input_layer_key])
                    else example_bank_previous.layers[input_layer_key]],
                    axis=0
                )
                train_celltype_labels = np.concatenate(
                    [train_celltype_labels, np.array(example_bank_previous_label)],
                    axis=0
                )
                train_batch_labels = np.concatenate(
                    [train_batch_labels, example_bank_previous_batch],
                    axis=0
                )
                adata_train_split = anndata.concat(
                    [adata_train_split, example_bank_previous],
                    axis=0,
                    merge='same'
                )
                if self.config["use_multimod"]:
                    train_multimod_labels = np.concatenate(
                        [train_multimod_labels, np.array(example_bank_previous_multimod)],
                        axis=0
                    )
           
        if self.config["weighted_loss"]:
            labels = torch.tensor(adata_train_split.obs["celltype_labels"].values, dtype=torch.long).to(self.device)
            class_counts = torch.bincount(labels)  # shape: [num_classes]

            # Simple approach: class weight = 1 / frequency
            class_weights = 1.0 / (class_counts.float() + 1e-8)

            # Optionally normalize so the weights sum to num_classes
            class_weights = class_weights * (len(class_counts) / class_weights.sum())

            class_weights = class_weights.to(self.device)

            # Define weighted cross-entropy loss
            self.criterion_cls = nn.CrossEntropyLoss(weight=class_weights)  # Redefine cross-entropy loss
        elif self.config["focal_loss"]:
            labels = torch.tensor(adata_train_split.obs["celltype_labels"].values, dtype=torch.long).to(self.device)
            num_classes = 17
            class_counts = torch.bincount(labels, minlength=num_classes)  # shape: [num_classes]
            total_samples = labels.size(0)
            alpha = total_samples / class_counts.float()
            # alpha = alpha / alpha.sum()
            alpha[class_counts == 0] = 1.0
            self.criterion_cls = FocalLoss(gamma=2, weight=alpha)
        else:
            self.criterion_cls = nn.CrossEntropyLoss()

        tokenized_train = tokenize_and_pad_batch(
            train_data,
            self.gene_ids,
            max_len=3061,    # This does not affect the final dimension tokenized_train['genes'].shape: torch.Size([903, 1201])
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
            mod_type = mod_type if self.config["use_mod"] else None,
            vocab_mod = self.vocab_mod if self.config["use_mod"] else None,
        )
        tokenized_valid = tokenize_and_pad_batch(
            valid_data,
            self.gene_ids,
            max_len=3061,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
            mod_type=mod_type if self.config["use_mod"] else None,
            vocab_mod=self.vocab_mod if self.config["use_mod"] else None,
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
        train_epoch_loss, train_proto_list, val_proto_list = [], [], []
        eval_epoch_loss = []
        contrastive_proto_loss_list = []
        ######################### early stopping ##########################
        self.early_stopper = EarlyStopping(patience=self.config["patience"], min_delta=1e-4, mode='min')
        if self.config["schedule"] == "cosine_schedule_with_warmup":
            if self.config["fewshot"] is not None:
                warmup_steps = 10
            else:
                warmup_steps = 2
            
            if self.config["decrease_lr_all"]:
                pass
            else:
                total_steps = self.config["epochs"] * (2000 // self.config["batch_size"])
                del self.scheduler

                torch.cuda.empty_cache()
                gc.collect()
                if self.config["add_decoder"]:                     # reinitialize optimizer after each training iter
                    self.optimizer = torch.optim.Adam(
                        self.model.parameters(), lr=self.config["lr"], eps=1e-4 if self.config["amp"] else 1e-8)
                self.scheduler = CosineScheduleWithWarmup(self.optimizer,
                            num_warmup_steps=warmup_steps,          # warmup steps
                            num_training_steps=total_steps        # total steps
                        )
        elif self.config["schedule"] == "stepLR":
            del self.scheduler
            torch.cuda.empty_cache()
            gc.collect()
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 1, gamma=0.9
            )
        elif self.config["schedule"] == "plateau":
            del self.scheduler
            torch.cuda.empty_cache()
            gc.collect()
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
            )

        for epoch in range(1, self.config["epochs"] + 1):
            epoch_start_time = time.time()
            if self.config["use_multimod"]:
                train_data_pt, valid_data_pt = prepare_data(
                    sort_seq_batch=False,
                    tokenized_train=tokenized_train, 
                    tokenized_valid=tokenized_valid, 
                    train_batch_labels=train_batch_labels,
                    valid_batch_labels=valid_batch_labels,
                    train_celltype_labels=train_celltype_labels,
                    valid_celltype_labels=valid_celltype_labels,
                    train_multimod_labels=train_multimod_labels,
                    valid_multimod_labels=valid_multimod_labels,
                    mask_ratio=self.config["mask_ratio"],
                    mask_value=-1,
                    pad_value=-2,
                )
            else:
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

            self.past_valid_loaders[test_batch_idx] = valid_loader

            if self.config["do_train"] and epoch < self.config["epochs"]:
                cell_emb_list, train_labels, adata_train_indices_list, train_loss, _, proto_loss, contrastive_proto_loss, entropy_list, accuracy_list = self.train(
                    loader=train_loader,
                    logger=logger,
                    epoch=epoch,
                    test_batch_idx=test_batch_idx
                )
            elif self.config["do_train"] and epoch == self.config["epochs"]:
                cell_emb_list, train_labels, adata_train_indices_list, train_loss, _, proto_loss, contrastive_proto_loss, entropy_list, accuracy_list = self.train(
                    loader=train_loader,
                    logger=logger,
                    epoch=epoch,
                    test_batch_idx=test_batch_idx
                )

            # if self.config["fewshot"] and epoch % 10 == 0:        
            val_loss, val_err, result_dict, eval_iter_loss_list, val_proto_loss = self.evaluate(
                loader=valid_loader,
                epoch=epoch,
                save_dir = save_dir,
                test_batch_idx=test_batch_idx
            )
            train_epoch_loss.append(train_loss)
            train_proto_list.append(proto_loss)
            val_proto_list.append(val_proto_loss)
            contrastive_proto_loss_list.append(contrastive_proto_loss)
            learning_rate.append(self.optimizer.param_groups[0]['lr'])
            if self.config["schedule"] == "plateau":
                self.scheduler.step(val_loss)
        
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
                with open(str(save_dir) + "/" + f"_batch_{test_batch_idx}_besteval_results.json", "w") as f:
                    json.dump(result_dict, f, indent=4)

                # ######################### compute previous sample performance #########################
                if len(self.past_valid_loaders) >= 1:
                    past_items = list(self.past_valid_loaders.items())  
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
                                mod_types = batch_data["mod_types"].to(self.device)
                                if self.config["use_multimod"]:
                                    multimod_labels = batch_data["multimod_types"].to(self.device)
                                src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

                                with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                                    if self.config["adapter"] or self.config["loramoe"]:
                                        output_dict, _ = best_model(
                                            input_gene_ids,
                                            input_values,
                                            src_key_padding_mask=src_key_padding_mask,
                                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                                            # batch_id = torch.tensor(task_id),
                                            batch_id = None,
                                            multimod_labels=multimod_labels if self.config["use_multimod"] else None,
                                            CLS=True,
                                            CCE=False,
                                            MVC=False,
                                            ECS=False,
                                            mod_types=mod_types if self.config["use_mod"] else None,
                                            do_sample=False,
                                        )
                                    else:
                                        output_dict = best_model(
                                            input_gene_ids,
                                            input_values,
                                            src_key_padding_mask=src_key_padding_mask,
                                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                                            multimod_labels=multimod_labels if self.config["use_multimod"] else None,
                                            CLS=True,
                                            CCE=False,
                                            MVC=False,
                                            ECS=False,
                                            mod_types=mod_types if self.config["use_mod"] else None,
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

                        # Save
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

            # if self.early_stopper.early_stop or epoch == self.config["epochs"]:
            # if (epoch-1) % 5 == 0 or self.early_stopper.early_stop or epoch == self.config["epochs"]:
            if (epoch-1) % 5 == 0:
                with torch.no_grad():   
                    # print(f"Early stopping at epoch {epoch} on task {test_batch_idx}")
                    if self.config["entropy"] and self.config["update_classifier"]:
                        self.update_classifier_weights_entropy(cell_emb_list, train_labels, entropy_list, accuracy_list) 
                        print("update_classifier_weights_with_entropy")
                    elif self.config["proto_loss"] and not self.config["update_classifier"] and not self.config["entropy"]:
                        self.update_prototype(cell_emb_list, train_labels)
                        print("Update prototype weights")
                        # elif self.config["cope_loss"] or not self.config["update_classifier"]
                    # elif self.config["epochs_val"] and not self.config["update_classifier"] and self.config["entropy"]:
                    #     self.update_prototype_with_entropy(cell_emb_list, train_labels, entropy_list, accuracy_list)
                    #     print("Update prototype weights with entropy")
                    elif not self.config["update_classifier"] or not self.config["proto_loss"]:
                        pass
                        print("no update on prototype and classifier")
                    else:
                        self.update_classifier_weights(cell_emb_list, train_labels)
                        print("update_classifier_weights")
            self.early_stopper(val_loss, self.model)
            if (self.early_stopper.early_stop or epoch == self.config["epochs"]) and self.config["proto_loss"]:
                self.update_prototype(cell_emb_list, train_labels, epoch='final')
                print("***************************Final prototype update***************************")
                break
            elif (self.early_stopper.early_stop or epoch == self.config["epochs"]) and not self.config["proto_loss"]:
                print("***************************No prototype update, training finished***************************")
                break
            else:
                pass

            if self.config["schedule"] == "stepLR":
                self.scheduler.step()
            
        gc.collect()
        torch.cuda.empty_cache()
        del train_data_pt, valid_data_pt, train_loader, valid_loader, train_loss, proto_loss, val_proto_loss,
        if self.config["save_weight"]:
            torch.save(self.model.state_dict(), save_dir / f"last_model_batch_{test_batch_idx}.pt")

        # all_adata_test.append(adata_train)
        if self.config["replay"]:
            example_bank_update(adata_train_split, adata_train_indices_list, train_labels, entropy_list, accuracy_list, save_dir, test_batch_idx, self.old_proto, cell_emb_list)
            
        all_adata_test.append(adata_valid_split)

        del cell_emb_list, train_labels, entropy_list, accuracy_list

        combined_adata_test = anndata.concat(
            all_adata_test,
            join="outer",
            merge="unique",
            label="batch",
            index_unique=None
        )
        combined_adata_test.obs["batch_id"] = le.fit_transform(combined_adata_test.obs["batch_id"])

        if self.config["use_best_initnextbatch"]:
            load_pretrained(self.model, torch.load(save_dir / f"best_model_batch_{test_batch_idx}.pt"), verbose=False)

        # if test_batch_idx == self.max_test_id:
        if self.config["proto_loss"]:
            self.plot_clusters_prototypes(combined_adata_test, self.old_proto, input_layer_key, self.gene_ids, test_batch_idx, 
                                          save_dir, best_model)
        self.plot_train_val_loss(train_epoch_loss, eval_epoch_loss, save_dir, test_batch_idx)
        del train_epoch_loss, eval_epoch_loss
        if self.config["repultion_loss"] or self.config["proto_loss"]:
            self.plot_proto_repultion_loss(train_proto_list, val_proto_list, save_dir, test_batch_idx)
        if self.config["contrastive_proto_loss"]:
            self.plot_contrastive_loss(contrastive_proto_loss_list, save_dir, test_batch_idx)
        del train_proto_list, val_proto_list
        return best_model, combined_adata_test, learning_rate
    
    def plot_feature_umap(self, cell_gene_emb, combined_adata_test, gene_ids, test_batch_idx, save_dir, 
                                         cell_type_map, legend_on = False, use_modalities=("RNA", "ADT", "ATAC")):
        if isinstance(cell_gene_emb, np.ndarray):
            cell_gene_emb = torch.from_numpy(cell_gene_emb)
        cell_gene_emb = cell_gene_emb.detach() if torch.is_tensor(cell_gene_emb) else cell_gene_emb
        n_cells, L, D = cell_gene_emb.shape
        n_features = combined_adata_test.var.shape[0]
        assert L == n_features + 1, f"Expected L=n_features+1, got {L} vs {n_features}+1"
        H = cell_gene_emb[:, 1:, :]  # [samples, n_features, 512]
        ftype = combined_adata_test.var["feature_type"].astype(str).values.astype(int)
        mod2id = {"RNA": 0, "ADT": 1, "ATAC": 2}
        # id2mod = {0: "RNA", 1: "ADT", 2: "ATAC"}
        Z_list, names = [], []
        for mod in use_modalities:
            mod_id = mod2id[mod] if isinstance(mod, str) else int(mod)
            m = (ftype == mod_id)                #  Directly compare to modality name to get feature mask
            if m.sum() == 0:
                raise ValueError(f"No features found for modality='{mod}'. Unique types: {np.unique(ftype)}")
            Z = H[:, m, :].mean(dim=1).cpu().numpy()  # (samples, 512)
            Z_list.append(Z)
            names.append(mod)
        Z_joint = np.vstack(Z_list)  # (len(mods)*samples, 512)

        U_joint = umap.UMAP(
                        n_neighbors=15,
                        min_dist=0.3,
                        metric="cosine",
                        random_state=42,
                    ).fit_transform(Z_joint)
    
        # if cell_type_map is not None:
        #     index2cell = {v: k for k, v in cell_type_map.items()}

        #     celltype_str_list = np.array(combined_adata_test.obs["celltype"]).tolist()
        #     current_celltype_labels = [index2cell[cell_type_map[cell]] for cell in celltype_str_list]
        #     current_celltype_labels = np.array(current_celltype_labels)
        #     combined_adata_test.obs["celltype_labels"] = current_celltype_labels
        #     labels = combined_adata_test.obs["celltype_labels"]
        # else:
        #     labels = combined_adata_test.obs["celltype"]
        # # Plot by str_batch
        # if self.config["dataset_name"] == "myeloid":
        #     batches = combined_adata_test.obs["str_batch"].astype(str).str.split("_").str[1]    
        # elif self.config["dataset_name"] == "BMMC":
        #     # batches = adata.obs["DonorID_new"].astype(str)
        #     # batches = adata.obs["DonorID"].astype(str) + adata.obs["modality"].astype(str)
        #     batches = combined_adata_test.obs["DonorID"].astype(str) + "_" + combined_adata_test.obs["batch"].astype(str) + "_" + combined_adata_test.obs["modality"].astype(str)
        # else:
        #     batches = combined_adata_test.obs["str_batch"]

        # highlight_batch_list = batches.unique()

        #  Get all celltypes and build a global color mapping
        # if self.config["dataset_name"] == "BMMC":
        #     reference_batch = ['15078_s1d1_multiome', '15078_s1d1_citeseq', '10886_s1d2_multiome', '10886_s1d2_citeseq', '18303_s1d3_multiome', '18303_s1d3_citeseq']
        #     highlight_batch_list = ['28045_s3d6_multiome', '28045_s3d6_citeseq', '11466_s3d7_multiome', '11466_s3d7_citeseq', '15078_s4d1_multiome', '15078_s4d1_citeseq']
        
        #     U_joint = umap.UMAP(
        #                 n_neighbors=15,
        #                 min_dist=0.3,
        #                 metric="cosine",
        #                 random_state=42,
        #             ).fit_transform(Z_joint)  # [n_cells, 2]
        #     ################################################################# 
            
        #     # # Get all celltypes and build a global color mapping
        #     # all_celltypes = sorted(np.unique(labels))
        #     all_celltypes = np.unique(labels)
        #     palette = sns.color_palette("tab20", n_colors=len(all_celltypes))
        #     color_map = dict(zip(all_celltypes, palette))   
        #     n_cells = U_joint.shape[0]
        #     ref_mask = batches.isin(reference_batch)
        #     query_mask = batches.isin(highlight_batch_list)
            
            # for mod in names:              # iterate over each mod; each mod has shape (samples, 2)
            #     U = U_joint[start:start + n_cells]
                ######################## Plot reference figure ###################### 
                # plt.figure(figsize=(5, 5))
                
                # sns.scatterplot(
                #     x=U[:n_cells, 0][ref_mask],
                #     y=U[:n_cells, 1][ref_mask],
                #     hue=labels[ref_mask],
                #     palette=color_map,       # fixed palette
                #     s=30000 / n_cells,       # 5
                #     linewidth=0,
                #     legend=legend_on
                # )
                # plt.title(f"t={test_batch_idx}", fontsize=24)
                # plt.xticks([])
                # plt.yticks([])
                # plt.tight_layout()
                # ax = plt.gca()
                # for spine in ax.spines.values():
                #     spine.set_edgecolor("black")
                #     spine.set_linewidth(1.5)
                # if legend_on:
                #     plt.legend(markerscale=15, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
                #     out_path = os.path.join(
                #         str(save_dir),
                #         f"gray_{mod}_incremental_all_batch_legend.png"
                #     )
                # else:
                #     out_path = os.path.join(
                #         str(save_dir),
                #         f"gray_{mod}_incremental_all_batch.png"
                #     )
                # plt.savefig(out_path, bbox_inches="tight", edgecolor="black", dpi=300)
                # plt.close()    
                ######################## Plot query incremental additions ######################
                # Loop to plot
                # start = 0
                # from plot_prototype.plot_umap import plot_umap_gray_incremental
                # for test_batch_idx in range(len(highlight_batch_list)):
                #     selected_batches = highlight_batch_list[:test_batch_idx+1]
                #     mask = batches.isin(selected_batches)
                #     plot_umap_gray_incremental(
                #         U, n_cells, batches, labels, color_map,
                #         mask, test_batch_idx,
                #         save_dir, save_name = f"modal_alignment_{mod}", legend_on = False
                #     )
        start = 0
        for mod in names:
            U = U_joint[start:start + n_cells]  # (samples, 2)
            start += n_cells
            # labels: used for hue
            if cell_type_map is not None:
                celltype_str_list = np.array(combined_adata_test.obs["celltype"]).astype(str).tolist()
                current_celltype_labels = np.array([cell_type_map[cell] for cell in celltype_str_list])
                # Optional: temporarily save
                combined_adata_test.obs["celltype_labels"] = current_celltype_labels
                labels = combined_adata_test.obs["celltype_labels"]
            else:
                labels = combined_adata_test.obs["celltype"].astype(str)

            plt.figure(figsize=(5, 5))
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)
            sns.scatterplot(
                x=U[:, 0],
                y=U[:, 1],
                hue=labels,
                palette="tab20",
                s=5,
                linewidth=0,
                legend=False
            )
            plt.title(f"{mod} UMAP (batch {test_batch_idx})")
            # plt.xlabel("UMAP-1")
            # plt.ylabel("UMAP-2")
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            out = os.path.join(save_dir, f"umap_{mod}_batch{test_batch_idx}.png")
            plt.savefig(out, dpi=300)
            plt.close()

        return U_joint
            
    def plot_gray_batch(self, adata, prototype, input_layer_key, gene_ids, test_batch_idx, save_dir, best_model = None, 
                        save_name="val", cell_type_map=None, legend_on=False, experiment = "query_mapping"):
        from plot_prototype.plot_umap import plot_outlier_detection_umap
        if prototype is not None:
            prototype_list = [prototype[c] for c in sorted(prototype.keys())]
            prototype_tensor = torch.stack(prototype_list, dim=0)  # shape: [num_classes, D]
            prototype = F.normalize(prototype_tensor, dim=1) 
        import umap
        if best_model is None:
            self.model.eval()
            model = self.model
        else:
            model = best_model
        if self.past_model is not None:
            self.past_model.eval()
        all_counts = (
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )                             # (43667, 2384)
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            self.gene_ids,                 # (2384,)
            max_len=3061,                           # should be 1201
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"].to(self.device), tokenized_all["values"].to(self.device)
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        # le = LabelEncoder()
        # adata.obs["batch_id"] = le.fit_transform(adata.obs["batch_id"])
        batch_ids = torch.tensor(adata.obs["batch_id"].values).to(self.device)
        batch_ids = torch.zeros_like(batch_ids)                    # Only one BN layer is trained in practice; set batch_ids to 0
        if torch.isnan(all_values).any():
            print("x contains NaN values!")
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):
            if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                cell_embeddings = model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels= torch.from_numpy(np.full_like(batch_ids)).long() if self.config["DSBN"] else None,
                    # batch_id = torch.tensor(test_batch_idx),
                    batch_id=None,
                    time_step=0,
                    return_np=True,
                )
            else:
                cell_embeddings = model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels= torch.from_numpy(np.full_like(batch_ids)).long() if self.config["DSBN"] else None,
                    time_step=0,
                    return_np=True,
                )

            if self.past_model is not None:
                cell_embeddings_past = self.past_model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    # batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                    # batch_labels = torch.from_numpy(test_batch_idx).long() if self.config["DSBN"] else None,
                    time_step=0,
                    return_np=True,
                )
                cell_embeddings = np.concatenate([cell_embeddings_past, cell_embeddings], axis=-1)
        # cell_embeddings = output_dict["cls_output"][1]
        n_cells = cell_embeddings.shape[0]
        cell_embeddings = torch.tensor(cell_embeddings, device=self.device)
        cell_embeddings = F.normalize(cell_embeddings, dim=1)
        X_cells = cell_embeddings
        if prototype is not None:
            # prototypes = F.normalize(prototype, dim=1)
            # Assume you already have prototypes (shape [n_prototypes, 1200])
            # If it's a tensor, convert to numpy
            prototypes_np = prototype.detach().cpu().numpy() if torch.is_tensor(prototype) else prototype
            n_prototypes = prototypes_np.shape[0]
        #     ################### Compute Euclidean distances as prediction ############### 
            X_cells = torch.tensor(cell_embeddings, device=self.device, dtype=torch.float32)
            prototypes = torch.tensor(prototype, device=self.device, dtype=torch.float32)
            distances = torch.cdist(X_cells, prototypes, p=2)
            temperature = 0.5
            softmax_dist = torch.softmax(-distances / temperature, dim=1)
            prediction_labels = torch.argmax(softmax_dist, dim=1).detach().cpu().numpy().tolist()
            index2cell = {v: k for k, v in cell_type_map.items()}
            current_predict_labels = [index2cell[cell] for cell in prediction_labels]
        #     ##############################################################
        #     # Step 2: Concatenate all vectors
        #     X_all = np.concatenate([X_cells.detach().cpu().numpy(), prototypes_np], axis=0)
        #     # Step 3: Dimensionality reduction
        #     umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        #     X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]
        # else:
        #     umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        #     X_umap = umap_model.fit_transform(X_cells.detach().cpu().numpy())  # [n_cells, 2]
        ######################### After saving X_all ##############################
        # np.save(str(save_dir) + "/" + "X_all_query_mapping.npy", X_all)
        # Step 2: Concatenate all vectors
        X_all = np.concatenate([X_cells.detach().cpu().numpy(), prototypes_np], axis=0)
        #######################################################################
        # Get true labels (must exist in .obs)
        if cell_type_map is not None:
            index2cell = {v: k for k, v in cell_type_map.items()}

            celltype_str_list = np.array(adata.obs["celltype"]).tolist()
            current_celltype_labels = [index2cell[cell_type_map[cell]] for cell in celltype_str_list]
            current_celltype_labels = np.array(current_celltype_labels)
            adata.obs["celltype_labels"] = current_celltype_labels
            labels = adata.obs["celltype_labels"]
        else:
            labels = adata.obs["celltype"]
        # Plot by str_batch
        if self.config["dataset_name"] == "myeloid":
            batches = adata.obs["str_batch"].astype(str).str.split("_").str[1]    
        elif self.config["dataset_name"] == "BMMC" or self.config["dataset_name"] == "BMMC_filter3":
            # batches = adata.obs["DonorID_new"].astype(str)
            # batches = adata.obs["DonorID"].astype(str) + adata.obs["modality"].astype(str)
            batches = adata.obs["DonorID"].astype(str) + "_" + adata.obs["batch"].astype(str) + "_" + adata.obs["modality"].astype(str)
        else:
            batches = adata.obs["str_batch"]

        highlight_batch_list = batches.unique()

        #  1. Get all celltypes and build a global color mapping
        if self.config["dataset_name"] == "BMMC" or self.config["dataset_name"] == "BMMC_filter3":
            reference_batch = ['15078_s1d1_multiome', '15078_s1d1_citeseq', '10886_s1d2_multiome', '10886_s1d2_citeseq', '18303_s1d3_multiome', '18303_s1d3_citeseq']
            highlight_batch_list = ['28045_s3d6_multiome', '28045_s3d6_citeseq', '11466_s3d7_multiome', '11466_s3d7_citeseq', '15078_s4d1_multiome', '15078_s4d1_citeseq']
        ref_mask = batches.isin(reference_batch)
        query_mask = batches.isin(highlight_batch_list)
        n_cells = X_all.shape[0] - n_prototypes
        if experiment == "outlier_detection":
            plot_outlier_detection_umap(self.config, n_cells, X_all, labels, prototypes_np, ref_mask, query_mask, save_dir, test_batch_idx, legend_on = True)
        # umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        # X_ref = X_all[:n_cells][ref_mask]
        # X_query = X_all[:n_cells][query_mask]
        # umap_model.fit(X_ref)
        # U_ref = umap_model.transform(X_ref)
        # U_query = umap_model.transform(X_query)
        # prototypes_np = umap_model.transform(X_all[n_cells:])
        # X_umap = np.vstack([U_ref, U_query, prototypes_np])
        # labels_ref = labels[ref_mask]
        # labels_query = labels[query_mask]
        elif experiment == "query_mapping":
            ######################## Non-query mapping case ########################
            umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
            X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]
            ################################################################# 
            
            # # #  1. Get all celltypes and build a global color mapping
            # all_celltypes = sorted(np.unique(labels))
            all_celltypes = np.unique(labels)
            palette = sns.color_palette("tab20", n_colors=len(all_celltypes))
            color_map = dict(zip(all_celltypes, palette))   
            n_cells = all_counts.shape[0]
            
            ######################## Plot reference figure ######################
            plt.figure(figsize=(5, 5))
            
            sns.scatterplot(
                x=X_umap[:n_cells, 0][ref_mask],
                y=X_umap[:n_cells, 1][ref_mask],
                hue=labels[ref_mask],
                palette=color_map,       # fixed palette
                s=30000 / n_cells,       # 5
                linewidth=0,
                legend=legend_on
            )
            plt.title(f"t={test_batch_idx}", fontsize=24)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.5)
            if legend_on:
                plt.legend(markerscale=15, bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
                out_path = os.path.join(
                    str(save_dir),
                    f"gray_prototype_incremental_reference_all_batch_legend.png"
                )
            else:
                out_path = os.path.join(
                    str(save_dir),
                    f"gray_prototype_incremental_reference_all_batch.png"
                )
            plt.savefig(out_path, bbox_inches="tight", edgecolor="black", dpi=300)
            plt.close()    
            
            
            #  2. Loop to plot
            from plot_prototype.plot_umap import plot_umap_gray_incremental
            for test_batch_idx in range(len(highlight_batch_list)):
                selected_batches = highlight_batch_list[:test_batch_idx+1]
                mask = batches.isin(selected_batches)
                plot_umap_gray_incremental(
                    X_umap, n_cells, batches, labels, color_map,
                    mask, test_batch_idx,
                    save_dir, save_name = "label_eval&test_new", legend_on = False
                )
                ################# Change: add prediction visualization ########################
                plot_umap_gray_incremental(
                    X_umap, n_cells, batches, current_predict_labels, color_map,
                    mask, test_batch_idx,
                    save_dir, save_name = "prediction_label_eval&test_new", legend_on = False
                )
            ###################################################################
            # if legend_on:
            #     plt.figure(figsize=(7, 5))
            # else:
            #     plt.figure(figsize=(5, 5))

            # # Background: all cells grey
            # sns.scatterplot(
            #     x=X_umap[:n_cells, 0],
            #     y=X_umap[:n_cells, 1],
            #     color="lightgray",
            #     s=5,
            #     linewidth=0,
            #     legend=False
            # )

            # #  Foreground: show first test_batch_idx+1 batches
            # selected_batches = highlight_batch_list[:test_batch_idx+1]
            # mask = batches.isin(selected_batches)

            # sns.scatterplot(
            #     x=X_umap[:n_cells, 0][mask],
            #     y=X_umap[:n_cells, 1][mask],
            #     hue=labels[mask],
            #     palette=color_map,   # fixed palette
            #     s=5,
            #     linewidth=0,
            #     legend=legend_on
            # )

            # Add prototypes
            # if prototype is not None:
            #     plt.scatter(
            #         X_umap[n_cells:, 0],
            #         X_umap[n_cells:, 1],
            #         edgecolors='black',
            #         facecolors='none',
            #         s=60,
            #         marker='X',
            #         label='Prototypes',
            #     )

            #     plt.title(f"t={test_batch_idx}", fontsize=24)
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.tight_layout()
            # else:
            #     plt.title(f"t={test_batch_idx}", fontsize=24)
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.tight_layout()

            # if legend_on:
            #     plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
            #     plt.savefig(
            #         str(save_dir) + "/" + save_name + f"gray_prototype_incremental_batch{test_batch_idx}_legend_label_evaltest.png",
            #         bbox_inches='tight',
            #         edgecolor='black',
            #         dpi=300
            #     )
            # else:
            #     plt.savefig(
            #         str(save_dir) + "/" + save_name + f"gray_prototype_incremental_batch{test_batch_idx}_label_evaltest.png",
            #         bbox_inches='tight',
            #         edgecolor='black',
            #         dpi=300
            #     )
            # plt.close()
        # # plot batch corectionif legend_on:
            # if legend_on:
            #     plt.figure(figsize=(7, 5))
            # else:
            #     plt.figure(figsize=(5, 5))
            # sns.scatterplot(
            #     x=X_umap[:n_cells, 0][mask],
            #     y=X_umap[:n_cells, 1][mask],
            #     hue=batches[mask],
            #     palette="tab10",
            #     s=5,
            #     linewidth=0,
            #     legend=legend_on
            # )
            # plt.xticks([])  # remove x-axis ticks
            # plt.yticks([])  # remove y-axis ticks
            # plt.tight_layout()
            # ax = plt.gca()
            # for spine in ax.spines.values():
            #     spine.set_edgecolor("black")
            #     spine.set_linewidth(1.5)
            # if legend_on:
            #     plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
            #     plt.savefig(str(save_dir) + "/" + save_name + f"umap_batch_correction_batch{test_batch_idx}_legend.png", bbox_inches='tight', edgecolor='black',dpi=300)
            # else:
            #     plt.savefig(str(save_dir) + "/" + save_name + f"umap_batch_correction_batch{test_batch_idx}.png", bbox_inches='tight', edgecolor='black',dpi=300)
            # ############# Plot batches by modality ##############
            # plt.figure(figsize=(5, 5))
            # sns.scatterplot(
            #     x=X_umap[:n_cells, 0][mask],
            #     y=X_umap[:n_cells, 1][mask],
            #     hue=modalities[mask],
            #     palette="tab10",
            #     s=5,
            #     linewidth=0,
            #     legend=legend_on
            # )
            # plt.xticks([])  # remove x-axis ticks
            # plt.yticks([])  # remove y-axis ticks
            # plt.tight_layout()
            # ax = plt.gca()
            # for spine in ax.spines.values():
            #     spine.set_edgecolor("black")
            #     spine.set_linewidth(1.5)
            # plt.savefig(str(save_dir) + "/" + save_name + f"umap_modalities_batch{test_batch_idx}.png", bbox_inches='tight', edgecolor='black',dpi=300)

    def plot_clusters_prototypes(self, adata, prototype, input_layer_key, gene_ids, test_batch_idx, save_dir, best_model = None, save_name="val", cell_type_map = None):
    # prototypes = F.normalize(prototype.prototypes, dim=1)
    # # prototypes = prototype.prototypes
    # # Step 1: Get cell and prototype embeddings
    # X_cells = adata_sorted.obsm["X_scEvolver"]  # cell embedding vectors, shape = [n_cells, 1200]
    # n_cells = X_cells.shape[0]
        # plt.rcParams.update({'font.size': 16})
        # plt.rcParams['font.family'] = 'ARIAL'
        if save_name == 'val' and self.config["save_weight"]:
            with open(str(save_dir) + "/" + f"prototype_{test_batch_idx}.pkl", "wb") as f:
                pickle.dump(prototype, f)
        prototype_list = [prototype[c] for c in sorted(prototype.keys())]
        prototype_tensor = torch.stack(prototype_list, dim=0)  # shape: [num_classes, D]

        prototype = F.normalize(prototype_tensor, dim=1) 
        
        if best_model is None:
            self.model.eval()
            model = self.model
        else:
            model = best_model
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
            max_len=3061,                           # should be 1201
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"].to(self.device), tokenized_all["values"].to(self.device)
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        batch_ids = torch.tensor(adata.obs["batch_id"].values).to(self.device)
        batch_ids = torch.zeros_like(batch_ids)                    # Only one BN layer was trained in practice; set batch_ids to 0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):
            if self.config["adapter"] or self.config["loramoe"]:
                cell_embeddings = model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels= torch.from_numpy(np.full_like(batch_ids)).long() if self.config["DSBN"] else None,
                    # batch_id = torch.tensor(test_batch_idx),
                    batch_id=None,
                    time_step=0,
                    return_np=True,
                )
            else:
                cell_embeddings = model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels= torch.from_numpy(np.full_like(batch_ids)).long() if self.config["DSBN"] else None,
                    time_step=0,
                    return_np=True,
                )

            if self.past_model is not None:
                cell_embeddings_past = self.past_model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    # batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                    # batch_labels = torch.from_numpy(test_batch_idx).long() if self.config["DSBN"] else None,
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
        # Assume you already have prototypes (shape [n_prototypes, 1200])
        # If it's a tensor, convert to numpy
        prototypes_np = prototypes.detach().cpu().numpy() if torch.is_tensor(prototypes) else prototypes
        n_prototypes = prototypes_np.shape[0]

        # Step 2: Concatenate all vectors
        X_all = np.concatenate([X_cells.detach().cpu().numpy(), prototypes_np], axis=0)

        # Step 3: Dimensionality reduction
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]

        # Step 4: Plot using true labels
        plt.figure(figsize=(5, 5))

        # Get true labels (must exist in .obs)
        if cell_type_map !=None:
            celltype_str_list = np.array(adata.obs["celltype"]).tolist()
            current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
            current_celltype_labels = np.array(current_celltype_labels)
            adata.obs["celltype_labels"] = current_celltype_labels                   # temporarily store labels
            labels = adata.obs["celltype_labels"]
        else:
            labels = adata.obs["celltype"]

        # Use seaborn to plot UMAP + label coloring
        sns.scatterplot(
            x=X_umap[:n_cells, 0],
            y=X_umap[:n_cells, 1],
            hue=labels,
            palette="tab20",
            s=5,
            linewidth=0,
            legend=True
        )
 
        # Add prototypes
        plt.scatter(
            X_umap[n_cells:, 0],
            X_umap[n_cells:, 1],
            edgecolors='black',     # edge color black
            facecolors='none',      # hollow
            s=60,
            marker='X',
            label='Prototypes',
        )
        
        plt.title("t={}".format(test_batch_idx), fontsize=24)
        plt.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        # plt.xlabel("UMAP1")
        # plt.ylabel("UMAP2")
        plt.xticks([])  # remove x-axis ticks
        plt.yticks([])  # remove y-axis ticks
        plt.margins(0)        #  Key: remove default margin
        plt.axis("equal")
        plt.tight_layout()
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        plt.savefig(str(save_dir) + "/" + save_name + f"umap_cluster_prototype_batch{test_batch_idx}.png", bbox_inches='tight', edgecolor='black',dpi=300)
        # plot batch corection
        plt.close()
        plt.figure(figsize=(5, 5))
        if self.config["dataset_name"] == "myeloid":
            batches = adata.obs["str_batch"].astype(str).str.split("_").str[1]
        else:
            batches = adata.obs["str_batch"]
        sns.scatterplot(
            x=X_umap[:n_cells, 0],
            y=X_umap[:n_cells, 1],
            hue=batches,
            palette="tab10",
            s=5,
            linewidth=0,
            legend=True
        )
        plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        plt.xticks([])  # remove x-axis ticks
        plt.yticks([])  # remove y-axis ticks
        plt.tight_layout()
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        plt.savefig(str(save_dir) + "/" + save_name + f"umap_batch_correction_batch{test_batch_idx}.png", bbox_inches='tight', edgecolor='black',dpi=300)

    def subplot_clusters_prototypes(self, adata, prototype, input_layer_key, gene_ids, test_batch_idx, save_dir, max_batch_idx, save_name="val", cell_type_map=None):

        if save_name == 'val':
            with open(str(save_dir) + "/" + f"prototype_{test_batch_idx}.pkl", "wb") as f:
                pickle.dump(prototype, f)
        prototype_list = [prototype[c] for c in sorted(prototype.keys())]
        prototype_tensor = torch.stack(prototype_list, dim=0)  # shape: [num_classes, D]

        prototype = F.normalize(prototype_tensor, dim=1) 
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
            self.gene_ids,
            max_len=3061,                           # should be 1201
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"].to(self.device), tokenized_all["values"].to(self.device)
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        batch_ids = torch.tensor(adata.obs["batch_id"].values).to(self.device)
        batch_ids = torch.zeros_like(batch_ids)                    # Only one BN layer was trained in practice; set batch_ids to 0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):
            if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                cell_embeddings = self.model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels= torch.from_numpy(np.full_like(batch_ids)).long() if self.config["DSBN"] else None,
                    # batch_id = torch.tensor(test_batch_idx),
                    # batch_id=None,
                    time_step=0,
                    return_np=True,
                )
            else:
                cell_embeddings = self.model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels= torch.from_numpy(np.full_like(batch_ids)).long() if self.config["DSBN"] else None,
                    time_step=0,
                    return_np=True,
                )

            if self.past_model is not None:
                cell_embeddings_past = self.past_model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    # batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                    # batch_labels = torch.from_numpy(test_batch_idx).long() if self.config["DSBN"] else None,
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
        # Assume you already have prototypes (shape: [n_prototypes, 1200])
        # If it's a tensor, convert to numpy
        prototypes_np = prototypes.detach().cpu().numpy() if torch.is_tensor(prototypes) else prototypes
        n_prototypes = prototypes_np.shape[0]

        # Step 2: Concatenate all vectors
        X_all = np.concatenate([X_cells.detach().cpu().numpy(), prototypes_np], axis=0)

        # Step 3: Dimensionality reduction
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]

        # Step 4: Plot using true labels
        # plt.figure(figsize=(7, 5))
        # plt.subplot(1, 6, test_batch_idx+1)
        # Get true labels (must exist in .obs)
        # labels = adata.obs["celltype"]
        if cell_type_map !=None:
            celltype_str_list = np.array(adata.obs["celltype"]).tolist()
            current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
            current_celltype_labels = np.array(current_celltype_labels)
            adata.obs["celltype_labels"] = current_celltype_labels                   # temporarily save
            labels = adata.obs["celltype_labels"]
        else:
            labels = adata.obs["celltype"]

        if self.config["dataset_name"] == "myeloid":
            labels_batch = adata.obs["str_batch"].astype(str).str.split("_").str[1]
        elif self.config["dataset_name"] == "BMMC":
            labels_batch = adata.obs["DonorID_new"].astype(str) 
            modality_batch = adata.obs["modality"].astype(str) 
        else:
            labels_batch = adata.obs["str_batch"]

        # Use seaborn to plot UMAP + label coloring
        ax1 = plt.subplot(3, max_batch_idx, test_batch_idx+1)
        sc1 = sns.scatterplot(
            x=X_umap[:n_cells, 0],
            y=X_umap[:n_cells, 1],
            hue=labels,
            palette="tab20",
            s=5,
            linewidth=0,
            legend=False,
            ax=ax1
        )
 
        # Add prototypes
        plt.scatter(
            X_umap[n_cells:, 0],
            X_umap[n_cells:, 1],
            edgecolors='black',     # edge color black
            facecolors='none',      # hollow
            s=60,
            marker='X',
            label='Prototypes'
        )
        plt.title("t={}".format(test_batch_idx), fontsize=24)
        
        # plt.xlabel("UMAP1")
        # plt.ylabel("UMAP2")
        plt.xticks([])  # remove x-axis ticks
        plt.yticks([])  # remove y-axis ticks

        ax2 = plt.subplot(3, max_batch_idx, max_batch_idx + test_batch_idx + 1)   # second row of subplot grid
        sc2 = sns.scatterplot(
            x=X_umap[:n_cells, 0],
            y=X_umap[:n_cells, 1],
            hue=labels_batch,
            palette="tab10",
            s=8,
            linewidth=0,
            legend=False,     # If set to 'full' for saving, add manual legend code below
            ax=ax2
        )
        # Manually generate legend
        # handles2, labels2 = ax2.get_legend_handles_labels()
        # n_items2 = len(labels2)

        # # Dynamically set number of columns
        # ncol2 = 1 if n_items2 <= 3 else 2

        # # Place below subplots
        # ax2.legend(
        #     handles2, labels2,
        #     loc="upper center",
        #     bbox_to_anchor=(0.5, -0.15),  # below plot
        #     ncol=ncol2,
        #     fontsize=20,
        #     frameon=False,
        #     markerscale=3 
        # )
        ax3 = plt.subplot(3, max_batch_idx, max_batch_idx + test_batch_idx + 1 + 1)   # next row of subplot grid
        sc3 = sns.scatterplot(
            x=X_umap[:n_cells, 0],
            y=X_umap[:n_cells, 1],
            hue=modality_batch,
            palette="Set1", 
            s=5,
            linewidth=0,
            legend=False,     # If set to 'full' for saving, add manual legend code below
            ax=ax2
        )
        plt.xticks([]); plt.yticks([])

        return sc1, sc2, labels, labels_batch

    def forward_for_ig(self, input_gene_ids, input_values, src_key_padding_mask):
        """
        Forward function specifically for Integrated Gradients.
        Returns the norm of each cell's embedding or a scalar output.
        """
        input_gene_ids = input_gene_ids.long()
        input_values = input_values.float()
        output_dict, _ = self.model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,  # or pass as needed
            CLS=True,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False,
        )
        # Returns a scalar, usable for IG
        cell_emb = output_dict["cell_emb"]  # [batch, dim]
        scalar_output = cell_emb.norm(dim=1)  # compute norm across embedding dims for each cell, scalar per cell
        return scalar_output
    
    def forward_latent_with_ig(self, adata_test, save_dir, test_batch_idx, cell_type_map):
        from plot_prototype.plot_clusters import plot_eval_cell_emb
        """
        Extends original forward_latent with Integrated Gradients attribution computation.
        """
        # ---- Keep original data preparation flow unchanged ----
        self.config["weight_dir"] = save_dir
        le = LabelEncoder()
        adata_test.obs["batch_id"] = le.fit_transform(adata_test.obs["batch_id"])
        input_layer_key = "X_binned"
        all_counts = (
            adata_test.layers[input_layer_key].A
            if issparse(adata_test.layers[input_layer_key])
            else adata_test.layers[input_layer_key]
        )
        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)
        celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels
        batch_ids = np.array(adata_test.obs["batch_id"].tolist())
        
        tokenized_test = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=3061,
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
            batch_size=1,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
            per_seq_batch_sample=False
        )
        
        self.model.eval()
        all_attr = []
        all_label = []
        # ---- Iterate over test batches ----
        num_batches = len(test_loader)
        # with torch.no_grad():
        for batch, batch_data in enumerate(itertools.islice(test_loader, num_batches)):
            input_values = batch_data["values"].to(self.device).float()
            input_values.requires_grad_(True)
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            celltype_labels = batch_data["celltype_labels"]
            src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

            # baseline can be zeros
            baseline_values = torch.zeros_like(input_values).to(self.device)

            n_steps = 10  # number of IG steps
            alphas = torch.linspace(0, 1, n_steps).to(self.device)

            # used to accumulate gradients
            grads = torch.zeros_like(input_values)
            with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                for alpha in alphas:
                    # construct interpolation
                    interpolated = baseline_values + alpha * (input_values - baseline_values)
                    interpolated.requires_grad_(True)
                    
                    # forward pass
                    
                    output_dict, _ = self.model(
                        input_gene_ids,
                        interpolated,
                        src_key_padding_mask=src_key_padding_mask,
                        batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                        batch_id=None,
                        CLS=True,
                        CCE=False,
                        MVC=False,
                        ECS=False,
                        do_sample=False,
                    )
                    
                    # scalar output (here using norm of each cell embedding)
                    cell_emb = output_dict["cell_emb"]  # [B, dim]
                    scalar_output = cell_emb.norm(dim=1)  # scalar per cell

                    grad = torch.autograd.grad(
                        outputs=scalar_output,
                        inputs=interpolated,
                        grad_outputs=torch.ones_like(scalar_output),
                        create_graph=False,
                        retain_graph=True,
                        only_inputs=True
                    )[0]

                    grads += grad

            # approximate integral
            attr = (input_values - baseline_values) * grads / n_steps   # (1, 1201)
            all_attr.append(attr.cpu().detach())
            all_label.append(celltype_labels)
        all_attr = torch.cat(all_attr)
        plot_eval_cell_emb(save_dir, all_attr, all_label, cell_type_map, save_name = f'clustermap_Cell_Embeddings_attr_genes_testbatch{test_batch_idx}.png')
           
        return all_attr  # list of attribution scores for each batch

    def predict_confidence(self, adata_test, save_dir, test_batch_idx, cell_type_map, mod_type):
        from tutorials.prototype_analysis import compute_confidence, prototype_dist_correlation
        Times = 5
        ########################## compute confidence and save them ##########################
        # compute_confidence(Times, self.model, self.config, self.device, self.gene_ids, self.vocab, self.vocab_mod, adata_test, save_dir, test_batch_idx, cell_type_map, mod_type)
        prototype_dist_correlation(save_dir, adata_test, cell_type_map)
        
    def predict(self, adata_test, save_dir, test_batch_idx, cell_type_map, mod_type, plot_gene_umap=None):
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
        # Compute cell types and labels for the current batch (test_batch_idx)
        # current_label_dict, current_celltype_labels = np.unique(
        #     np.array(adata_test.obs["celltype"].tolist()), return_inverse=True
        # )
        celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels                   # temporarily save
        batch_ids = adata_test.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)
        if self.config["use_multimod"]:
            modality_to_id = {mod: i for i, mod in enumerate(adata_test.obs["modality"].unique())}
            adata_test.obs["modality_id"] = adata_test.obs["modality"].map(modality_to_id).astype(int)
            multimods_ids = adata_test.obs["modality_id"].tolist()
            num_multimods = len(set(multimods_ids))
            test_multimod_labels = np.array(multimods_ids)
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
            self.gene_ids,
            max_len=3001,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
            mod_type = mod_type if self.config["use_mod"] else None,
            vocab_mod = self.vocab_mod if self.config["use_mod"] else None,
        )
        if self.config["use_multimod"]:
            test_data_pt = prepare_testdata(
                sort_seq_batch=False,
                tokenized_test=tokenized_test,
                test_batch_labels=batch_ids,
                test_celltype_labels=current_celltype_labels,
                test_multimod_labels=test_multimod_labels,
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
        predictions, labellist, problist = [], [], []
        num_batches = len(test_loader)
        feature_emb = []
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
                    if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                        output_dict, _ = self.model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                            # batch_labels= test_batch_idx if False or self.config["DSBN"] else None,
                            # batch_id = torch.tensor(test_batch_idx),
                            # batch_id = None,
                            multimod_labels=multimod_labels if self.config["use_multimod"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=self.config["MVC"],
                            ECS=self.config["ecs_thres"] > 0,
                            mod_types=mod_types if self.config["use_mod"] else None,
                            do_sample=False,
                        )
                    else:
                        output_dict = self.model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=False,
                            ECS=False,
                            do_sample=False,
                        )
                    
                    output_values = output_dict["cls_output"]
                    # loss = self.criterion_cls(output_values, celltype_labels)
                    cell_emb = output_dict["cell_emb"]
                    cell_emb = F.normalize(cell_emb, p=2, dim=1)
                if plot_gene_umap is not None:
                    feature_emb.append(_.detach().cpu().numpy())
                    del _
                # total_loss += loss.item() * len(input_gene_ids)
                accuracy += (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).detach().cpu().numpy()
                probs = F.softmax(output_values, dim=-1)
                # entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).detach().cpu().numpy()
                # problist.extend(probs)
                predictions.extend(preds)
                labellist.extend(celltype_labels.detach().cpu().numpy())
            if plot_gene_umap is not None:    
                feature_emb = np.concatenate(feature_emb, axis=0)
                return feature_emb
            # Save
            f1_macro = f1_score(labellist, predictions, average="macro")
            f1_micro = f1_score(labellist, predictions, average="micro")
            f1_weighted = f1_score(labellist, predictions, average="weighted")
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

            ############## Plot confusion matrix ###################
            class_names = np.unique(np.concatenate((labellist, predictions)))  # get all classes
            pred_counts = pd.Series(labellist).value_counts()
            pred_counts = pred_counts.reindex(class_names, fill_value=0)   # keep order consistent
            
            palette = sns.color_palette("tab20", len(class_names))
            celltype_colors = dict(zip(class_names, palette))
            
            x_positions = np.arange(len(class_names))
            bar_colors = [celltype_colors[ct] for ct in class_names]

            fig, ax = plt.subplots(figsize=(10, 2))

            # ----------- Plot bar chart ----------
            ax.bar(
                x_positions,
                pred_counts.values,
                width=1.0,
                color=bar_colors,
                alpha=0.9
            )

            # y-axis label
            ax.set_ylabel("Count", fontsize=12)

            # x-axis ticks show class names
            ax.set_xticks(x_positions)
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)

            # ----------- Add numeric annotations ----------
            for i, count in enumerate(pred_counts.values):
                ax.text(
                    i,
                    count + pred_counts.max() * 0.03,
                    str(int(count)),
                    ha='center',
                    va='bottom',
                    fontsize=12
                )

            # Styling tweaks
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{save_dir}/predict_test_batch_{test_batch_idx}_Confusion_Matrix_with_counts.png",
                dpi=300)
            ############################## heatmap confusion matrix ##############################
            
            cm = confusion_matrix(labellist, predictions, labels=class_names)
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm = np.round(cm, 1)  # round each element to 1 decimal place
            cm = pd.DataFrame(cm, index=class_names, columns=class_names)
            # Draw heatmap

            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt=".1f", cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})  # smaller font size

            # plt.xlabel('Predicted Labels')
            # plt.ylabel('True Labels')
            # plt.title('Confusion Matrix')
            # plt.tight_layout()
            # plt.savefig(str(save_dir) + "/" + f"predict_test_batch_{test_batch_idx}_Confusion_Matrix.png", dpi=300)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # ====== Draw colorbars on X/Y axis ======
            # x-axis colorbar
            for i, ct in enumerate(class_names):
                ax.add_patch(plt.Rectangle((i, cm.shape[0]), 1, 0.2, 
                                        color=celltype_colors[ct], clip_on=False, transform=ax.transData))

            # y-axis colorbar
            for i, ct in enumerate(class_names):
                ax.add_patch(plt.Rectangle((-0.2, i), 0.2, 1, 
                                        color=celltype_colors[ct], clip_on=False, transform=ax.transData))

            # ====== Add legend ======
            # handles = [plt.Line2D([0], [0], marker='s', color=color, linestyle='', markersize=10) 
            #         for color in celltype_colors.values()]
            # ax.legend(handles, class_names, title="Cell Types", bbox_to_anchor=(1.05, 1), loc='upper left')
            # Create a list of cell names based on the class names (indices)
            class_name_labels = [list(cell_type_map.keys())[list(cell_type_map.values()).index(i)] for i in range(len(class_names))]

            # Create handles for the legend using the colors associated with each cell type
            handles = [plt.Line2D([0], [0], marker='s', color=color, linestyle='', markersize=10) 
                    for color in celltype_colors.values()]

            # Use the mapped cell names in the legend
            ax.legend(handles, class_name_labels, bbox_to_anchor=(1.16, 1.0), loc='upper left', frameon=False) 
            
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.tight_layout(pad=2.0)  # Adjust padding to avoid overlap
            plt.savefig(str(save_dir) + "/" + f"predict_test_batch_{test_batch_idx}_Confusion_Matrix.png",dpi=300)
            print(f'print current batch confusion matrix {test_batch_idx} done!')

            ################# Plot entropy distribution ####################
            
        return self.gene_ids
    
    def plot_train_val_loss(self, train_loss_list_all, val_loss_list_all, save_dir, test_batch_idx):
        epochs_train = range(1, len(train_loss_list_all) + 1)
        epochs_val = range(1, len(val_loss_list_all) + 1)

        plt.figure(figsize=(10, 8))

        # Subplot 1: Training loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs_train, train_loss_list_all, label='Train Loss', marker='o', color='blue')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.tight_layout()

        # Subplot 2: Validation loss
        plt.subplot(2, 1, 2)
        plt.plot(epochs_val, val_loss_list_all, label='Validation Loss', marker='s', color='orange')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.grid(True)
        plt.tight_layout()

        # Save figure(s)
        plt.savefig(save_dir / f"train_val_loss_batch{test_batch_idx}.png")

    def plot_contrastive_loss(self, contrastive_proto_loss_list,save_dir, test_batch_idx):
        contrastive_loss = range(1, len(contrastive_proto_loss_list) + 1)
        plt.figure(figsize=(5, 8))
        plt.plot(contrastive_loss, contrastive_proto_loss_list, label='Proto Loss', marker='o', color='blue')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Contrastive Proto Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / f"Contrastive_Proto_Repultion_loss_batch{test_batch_idx}.png")

    def plot_proto_repultion_loss(self, contrastive_proto_loss_list, repultion_loss_list, save_dir, test_batch_idx):

        proto_loss = range(1, len(contrastive_proto_loss_list) + 1)
        repultion_loss = range(1, len(repultion_loss_list) + 1)

        plt.figure(figsize=(10, 8))

        # Subplot 1: Training loss
        plt.subplot(2, 1, 1)
        plt.plot(proto_loss, contrastive_proto_loss_list, label='Proto Loss', marker='o', color='blue')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Proto Loss")
        plt.grid(True)
        plt.tight_layout()

        # Subplot 2: Validation loss
        plt.subplot(2, 1, 2)
        plt.plot(repultion_loss, repultion_loss_list, label='Repultion Loss', marker='s', color='orange')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Repultion Loss")
        plt.grid(True)
        plt.tight_layout()

        # Save figure
        plt.savefig(save_dir / f"Proto_Repultion_loss_batch{test_batch_idx}.png")

    def evaluate_all(self, adata_train, save_dir, test_batch_idx, cell_type_map, all_adata_test):
                # Count number of each cell type
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
        adata_train.obs["celltype_labels"] = current_celltype_labels                   # replace current labels with final label indices
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
                all_counts, current_celltype_labels, batch_ids, test_size=self.config["valid_ratio"], shuffle=True
            )
            # First split indices
            adata_indices = np.arange(len(adata_train))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=self.config["valid_ratio"],
                shuffle=True,
                # stratify=current_celltype_labels if not self.config["randomsplit"] else None,
            )

            # Subset adata_train by indices
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
                test_size=self.config["valid_ratio"],
                min_class_size=20
            )
            adata_train_split = adata_train[train_indices].copy()
            adata_valid_split = adata_train[val_indices].copy() 
        # all_adata_test.append(adata_valid_split)
        return adata_valid_split, gene_ids

    def evaluate_predict(self, adata_train, save_dir, test_batch_idx, cell_type_map, mod_type):
        self.config["weight_dir"] = save_dir
                # Count number of each cell type
        # celltype_counts = adata_test.obs["celltype"].value_counts()
        # # Find cell types with counts >= 4
        # valid_celltypes = celltype_counts[celltype_counts >= 2].index

        # # Filter out samples whose cell types have fewer than 4 counts
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
        # genes = adata_train.var["gene_name"].tolist()
        # gene_ids = np.array(self.vocab(genes), dtype=int)
        # Compute cell types and labels for the current batch (test_batch_idx)
        # current_label_dict, current_celltype_labels = np.unique(
        #     np.array(adata_test.obs["celltype"].tolist()), return_inverse=True
        # )
        celltype_str_list = np.array(adata_train.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_train.obs["celltype_labels"] = current_celltype_labels                   # temporarily save
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
                all_counts, current_celltype_labels, batch_ids, test_size=self.config["valid_ratio"], shuffle=True
            )
            # First split indices
            adata_indices = np.arange(len(adata_train))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=self.config["valid_ratio"],
                shuffle=True,
                stratify=current_celltype_labels if not self.config["randomsplit"] else None,
            )

            # Subset adata_train by indices
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
                    test_size=self.config["valid_ratio"],
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
                    test_size=self.config["valid_ratio"],
                    min_class_size=20
                )
            adata_train_split = adata_train[train_indices].copy()
            adata_valid_split = adata_train[val_indices].copy() 
      
        tokenized_test = tokenize_and_pad_batch(
            valid_data,
            self.gene_ids,
            max_len=3061,
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
        predictions, labellist, cell_emb_list = [], [], []
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
                    output_dict, _ = self.model(
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
                    cell_emb = output_dict["cell_emb"]
                    loss = self.criterion_cls(output_values, celltype_labels)
                cell_emb_list.append(cell_emb.detach().cpu().numpy())
                # total_loss += loss.item() * len(input_gene_ids)
                accuracy += (output_values.argmax(1) == celltype_labels).sum().item()
                total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                total_num += len(input_gene_ids)
                preds = output_values.argmax(1).detach().cpu().numpy()
                predictions.extend(preds)
                labellist.extend(celltype_labels.detach().cpu().numpy())
            f1 = f1_score(np.array(labellist), np.array(predictions), average='macro')       # macro average
            f1_micro = f1_score(np.array(labellist), np.array(predictions), average='micro') # micro average
            f1_weighted = f1_score(np.array(labellist), np.array(predictions), average='weighted') # weighted average
            result_dict = {
                "num_samples":total_num,
                "accuracy": accuracy / total_num,
                "f1":f1,
                "f1_micro":f1_micro,
                "f1_weighted":f1_weighted,
                "preds":  [int(p) for p in predictions],  # convert to list
                "labels": [int(p) for p in labellist],
                }
            with open(str(save_dir) + "/" + f"evaluate_batch_{test_batch_idx}_results.json", "w") as f:
                json.dump(result_dict, f)
                
            ############### Plot confusion matrix ###################
            class_names = np.unique(np.concatenate((labellist, predictions)))  # get all classes
            cm = confusion_matrix(labellist, predictions, labels=class_names)

            # Draw heatmap

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.savefig(str(save_dir) + "/" + f"evaluate_batch_{test_batch_idx}_Confusion_Matrix.png")
            # self.plot_clusters_prototypes(adata_test, self.model.cls_decoder.out_layer.weight.data, input_layer_key, gene_ids, save_dir)
        return adata_valid_split, self.gene_ids, cell_emb_list, labellist

    def plot_bank_prototype(self, adata, prototype, input_layer_key, config, save_dir, cell_type_map):

        prototype_list = [prototype[c] for c in sorted(prototype.keys())]
        prototype_tensor = torch.stack(prototype_list, dim=0)  # shape: [num_classes, D]

        prototype = F.normalize(prototype_tensor, dim=1) 
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
            self.gene_ids,
            max_len=3061,                           # should be 1201
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"].to(self.device), tokenized_all["values"].to(self.device)
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        batch_ids = torch.tensor(adata.obs["batch_id"].values).to(self.device)
        batch_ids = torch.zeros_like(batch_ids)                    # Only one BN layer was trained in practice; set batch_ids to 0

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):
            if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                cell_embeddings = self.model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels= torch.from_numpy(np.full_like(batch_ids)).long() if self.config["DSBN"] else None,
                    # batch_id = torch.tensor(test_batch_idx),
                    # batch_id=None,
                    time_step=0,
                    return_np=True,
                )
            else:
                cell_embeddings = self.model.encode_batch(
                    all_gene_ids,
                    all_values.float(),
                    src_key_padding_mask=src_key_padding_mask,
                    batch_size=self.config["batch_size"],
                    batch_labels= torch.from_numpy(np.full_like(batch_ids)).long() if self.config["DSBN"] else None,
                    time_step=0,
                    return_np=True,
                )

        # cell_embeddings = output_dict["cls_output"][1]
        n_cells = cell_embeddings.shape[0]
        cell_embeddings = torch.tensor(cell_embeddings, device=self.device)
        cell_embeddings = F.normalize(cell_embeddings, dim=1)
        X_cells = cell_embeddings
        prototypes = F.normalize(prototype, dim=1)
        # Assume you already have prototypes (shape: [n_prototypes, 1200])
        # If it's a tensor, convert to numpy
        prototypes_np = prototypes.detach().cpu().numpy() if torch.is_tensor(prototypes) else prototypes
        n_prototypes = prototypes_np.shape[0]

        # Step 2: Concatenate all vectors
        X_all = np.concatenate([X_cells.detach().cpu().numpy(), prototypes_np], axis=0)

        # Step 3: Dimensionality reduction
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]

        # Step 4: Plot using true labels
        # plt.figure(figsize=(7, 5))
        # plt.subplot(1, 6, test_batch_idx+1)
        # Get true labels (must exist in .obs)
        # labels = adata.obs["celltype"]
        if cell_type_map !=None:
            celltype_str_list = np.array(adata.obs["celltype"]).tolist()
            current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
            current_celltype_labels = np.array(current_celltype_labels)
            adata.obs["celltype_labels"] = current_celltype_labels                   # temporarily save
            labels = adata.obs["celltype_labels"]
            labels_name = adata.obs["cell_type"]

        else:
            labels = adata.obs["celltype"]

        if self.config["dataset_name"] == "myeloid":
            labels_batch = adata.obs["str_batch"].astype(str).str.split("_").str[1]
        elif self.config["dataset_name"] == "BMMC":
            labels_batch = adata.obs["DonorID_new"].astype(str) 
        else:
            labels_batch = adata.obs["str_batch"]
        
        
        # Use seaborn to plot UMAP + label coloring
        # ax1 = plt.subplot(2, max_batch_idx, test_batch_idx+1)
        num_classes = len(np.unique(labels))
        palette = sns.color_palette("tab20", n_colors=num_classes)
        label_to_color = {label: palette[i] for i, label in enumerate(np.unique(labels))}

        # Plot original cells + prototypes
        plt.figure(figsize=(4, 4))
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')  # Set the color of the border (spines)
            spine.set_linewidth(0.5)      # Set the width of the border (spines)
        if "state" in adata.obs.columns:
            origion_mask = adata.obs["state"] == "origion"
            sns.scatterplot(
                x=X_umap[:n_cells, 0][origion_mask],
                y=X_umap[:n_cells, 1][origion_mask],
                hue=labels[origion_mask],
                palette=label_to_color,
                s=15,
                linewidth=0,
                legend=False,
                alpha=1.0
            )
        plt.scatter(
            X_umap[n_cells:, 0],
            X_umap[n_cells:, 1],
            # facecolors="#FFFEFE",
            edgecolors='black',     # edge color black
            facecolors='none',      # hollow
            s=60,
            marker='X',
            label='Prototypes',
        )
        plt.xticks([])  # remove x-axis ticks
        plt.yticks([])  # remove y-axis ticks
        plt.tight_layout()
        plt.savefig(str(save_dir) + "/" + config["dataset_name"] + f"_origion_" + f"_prototype_nolengend.png", bbox_inches='tight', edgecolor='black',dpi=300)
        
        # Plot cells (background) + memory-replay samples
        plt.figure(figsize=(4, 4))
        for spine in plt.gca().spines.values():
            spine.set_edgecolor('black')  # Set the color of the border (spines)
            spine.set_linewidth(0.5)      # Set the width of the border (spines)
        sns.scatterplot(
            x=X_umap[:n_cells, 0][origion_mask],
            y=X_umap[:n_cells, 1][origion_mask],
            # hue=labels[origion_mask],  # Keep hue to maintain structure, but we will override palette
            color = "lightgray",
            # palette=["gray"],  # Set all points to gray color
            s=15,
            linewidth=0,
            legend=False,
            # alpha=0.3
        )

        if "state" in adata.obs.columns:
            bank_mask = adata.obs["state"] == "bank"
            if bank_mask.any():
                sns.scatterplot(
                    x=X_umap[:n_cells, 0][bank_mask],  # Corrected to use keyword argument 'x'
                    y=X_umap[:n_cells, 1][bank_mask],  # Corrected to use keyword argument 'y'
                    # edgecolors='black',
                    hue=labels[bank_mask],
                    palette=label_to_color,
                    s=15,
                    # linewidth=1.0,
                    alpha=1.0,  # Slight transparency to avoid covering colorful points
                    label="bank",
                    legend=False
                )

        plt.xticks([])  # remove x-axis ticks
        plt.yticks([])  # remove y-axis ticks
        plt.tight_layout()
        plt.savefig(str(save_dir) + "/" + config["dataset_name"] + f"_color_bank_" + f"_nolengend.png", bbox_inches='tight', edgecolor='black',dpi=300)
        return labels, labels_batch

def plot_bank(save_dir, cell_type_map):
    with open(save_dir + "/config.json", "r", encoding="utf-8") as f:
        config = json.load(f) 
    set_seed(config["seed"])
    if config["dataset_name"] == "pancreas":
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6)]  # remaining three batches used as test set
    elif config["dataset_name"] == "myeloid":
        data_paths = ["../data/myeloid/" + f"myeloid_batch{i}.h5ad" for i in [1,2,5,6,7]]
    elif config["dataset_name"] == "BMMC":
        batchID = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        batchID = batchID[:6]
        all_paths = glob.glob("../data/BMMC/BMMC_batch*_*.h5ad")
        data_paths = []
        
        for b in batchID:
            matched = [p for p in all_paths if f"batch{b}_" in p]
            if matched:
                data_paths.append(matched[0])  # assume one file per combination
    adata_current_path = data_paths[5]
    genes = np.load('../data/BMMC/gene_list_atac_adt.npy', allow_pickle=True).tolist()
    model_dir = Path(config["load_model"])
    modeldict_name = "best_model_batch_5.pt"
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    num_batch_types = 1
    config["weight_dir"] = save_dir
    weight_dir = Path(config["weight_dir"])
    model_file = weight_dir / modeldict_name
    if config["dataset_name"] == "pancreas": 
        continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_5.pt")
    elif config["dataset_name"] == "myeloid":
        continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_4.pt")
    elif config["dataset_name"] == "BMMC":
        continual_classify = ContinualClassify(config, vocab, num_batch_types, genes, model_file, modeldict_name = modeldict_name)
    example_bank = torch.load(save_dir + "/example_bank.pth")
    example_bank_data = example_bank["example_bank"]

    adata_current = sc.read_h5ad(adata_current_path)
    if not sp.issparse(adata_current.X):
        adata_current.X = sp.csr_matrix(adata_current.X, dtype=np.float32)
    adata_current.var["gene_name"] = adata_current.var.index.tolist()
    adata_current.var["feature_type"] = adata_current.var["feature_type"].map({"RNA": 0, "ADT": 1, "ATAC":2}).astype(int)
    adata_current.obs["state"] = "origion"
    example_bank_data.obs["state"] = "bank"
    
    # Merge (align by same genes/features)
    adata_merged = anndata.concat(
        [adata_current, example_bank_data],
        join="outer",      # Use "outer" to align if var differ
        label="source",    # Optional: record source label in .uns["concat"]
        index_unique=None  # keep original obs_names
    )
    with open(str(save_dir) + "/" + f"prototype_{5}.pkl", "rb") as f:
        proto = pickle.load(f)

    continual_classify.plot_bank_prototype(adata_merged, proto, "X_binned", config, save_dir, cell_type_map)

    

def evaluate_predict(save_dir, cell_type_map):
    from plot_prototype.plot_clusters import plot_eval_cell_emb
    # config = init_wandb()
    with open(save_dir + "/config.json", "r", encoding="utf-8") as f:
        config = json.load(f) 
    set_seed(config["seed"])
    if config["dataset_name"] == "pancreas":
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6)]  # remaining three batches used as test set
    elif config["dataset_name"] == "myeloid":
        data_paths = ["../data/myeloid/" + f"myeloid_batch{i}.h5ad" for i in [1,2,5,6,7]]
    elif config["dataset_name"] == "BMMC":
        batchID = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        batchID = batchID[:6]
        all_paths = glob.glob("../data/BMMC/BMMC_batch*_*.h5ad")
        data_paths = []
        
        for b in batchID:
            matched = [p for p in all_paths if f"batch{b}_" in p]
            if matched:
                data_paths.append(matched[0])  # assume one file per combination
        genes = np.load('../data/BMMC/gene_list_atac_adt.npy', allow_pickle=True).tolist()

    model_dir = Path(config["load_model"])
    modeldict_name = "best_model_batch_5.pt"
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    num_batch_types = 1
    max_batch_idx = len(data_paths)
    config["weight_dir"] = save_dir
    weight_dir = Path(config["weight_dir"])
    model_file = weight_dir / modeldict_name
    if config["dataset_name"] == "pancreas": 
        continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_5.pt")
    elif config["dataset_name"] == "myeloid":
        continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_4.pt")
    elif config["dataset_name"] == "BMMC":
        continual_classify = ContinualClassify(config, vocab, num_batch_types, genes, model_file, modeldict_name = modeldict_name)

    all_adata_list = []
    all_batch_results = {}
    all_cell_emb = []
    all_label = []
    fig, axs = plt.subplots(3, max_batch_idx, figsize=(5*max_batch_idx+2, 10), constrained_layout=True)      # when legend=True, figsize should be larger
    # fig.subplots_adjust(bottom=0.25)
    for test_batch_idx in range(len(data_paths)):
        adata_test = sc.read_h5ad(data_paths[test_batch_idx])
        # genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        # all_adata_test, gene_ids, cell_emb_batch, label_batch = continual_classify.evaluate_predict(
        #     adata_test, save_dir, test_batch_idx, cell_type_map)
        # all_adata_test, gene_ids = continual_classify.evaluate_all(adata_test, save_dir, test_batch_idx, cell_type_map)
        # all_adata_list.append(all_adata_test)
        # all_cell_emb.extend(cell_emb_batch)
        # all_label.extend(label_batch)
        if not sp.issparse(adata_test.X):
            adata_test.X = sp.csr_matrix(adata_test.X, dtype=np.float32)
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        adata_test.var["feature_type"] = adata_test.var["feature_type"].map({"RNA": 0, "ADT": 1, "ATAC":2}).astype(int)
        # adata_train.obs["mod_types"] = adata_train.obs["mod"].map({'RNA':0, 'Protein':1}).astype(int)
        # mod_type = np.array(adata_test.var["feature_type"]) 
        all_adata_list.append(adata_test)   # plotting stage draws all train + val data
        ################### plot the process & save figures ###########################
        with open(str(save_dir) + "/" + f"prototype_{test_batch_idx}.pkl", "rb") as f:
            proto = pickle.load(f)
        combined_adata_test = anndata.concat(
                    all_adata_list,
                    join="outer",
                    merge="unique",
                    label="batch_new",
                    index_unique=None
                )
        continual_classify.subplot_clusters_prototypes(combined_adata_test, proto, 
                            "X_binned", gene_ids, test_batch_idx, save_dir, max_batch_idx, save_name=f"trainval_batch_{test_batch_idx}_", cell_type_map=cell_type_map)
#####################################################
    # plt.tight_layout()
    # plt.savefig(str(save_dir) + "/" + config["dataset_name"] + f"_trainval_batch_all_" + f"umap_cluster_prototype_lengend.png", bbox_inches='tight', edgecolor='black',dpi=300)
############################################################################
    # plot_eval_cell_emb(save_dir, all_cell_emb, all_label, cell_type_map)
    
    combined_adata_test = anndata.concat(
        all_adata_list,
        join="outer",
        merge="unique",
        label="batch",
        index_unique=None
    )
    # mod_type = np.array(combined_adata_test.var["feature_type"])
    all_adata_test, gene_ids, cell_emb_batch, label_batch = continual_classify.evaluate_predict(
            combined_adata_test, save_dir, test_batch_idx+100, cell_type_map)
    # le = LabelEncoder()
    # print("1*******************", combined_adata_test.obs["batch_id"])
    # combined_adata_test.obs["batch_id"] = le.fit_transform(combined_adata_test.obs["batch_id"])
    # print("2*******************", combined_adata_test.obs["batch_id"])
    # final_results, final_adata = continual_classify.eval_testdata(
    #     model = continual_classify.model,
    #     adata_t=combined_adata_test,
    #     gene_ids=gene_ids,
    #     input_layer_key={
    #         "normed_raw": "X_normed",
    #         "log1p": "X_normed",
    #         "binned": "X_binned",
    #     }["binned"],test_batch_idx=test_batch_idx
    # )

    # with open(str(save_dir) +  "/final_evaluate_results.json", "w") as f:
    #     json.dump(final_results, f, indent=4)
    # print("**********************save json************************")

    # sc.pp.neighbors(final_adata, use_rep="X_scGPT")
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

    # sc.pp.neighbors(final_adata, use_rep="X_scGPT")
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
def plot_learning_rate(save_dir, all_learning_rate_list):
    plt.figure(figsize=(8, 4))
    plt.plot(all_learning_rate_list, label="Learning Rate", linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_dir) + "/" + "learning_rate.png")

def main(config):
    # config = init_wandb()
    # set_seed(config["seed"])

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
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6)]  # remaining three batches used as test set
        # data_paths = ["../data/PANCREAS/" + f"pancreas_batch{2}.h5ad", 
        #               "../data/PANCREAS/" + f"pancreas_batch{3}.h5ad"]
    elif config["dataset_name"] == "myeloid":
        data_paths = [f"../data/myeloid/myeloid_batch{i}.h5ad" for i in [1, 2, 5, 6, 7]]
    elif config["dataset_name"] == "BMMC":
    
        # donorID = [15078, 10886, 18303, 12710, 16710, 28045, 11466, 19593, 13272]
        # donorID = donorID[:3]
        # all_paths = glob.glob("../data/BMMC/BMMC_batch*_*.h5ad")
        # data_paths = []
        # for d in donorID:
        #     for modality in ["multiome", "citeseq"]:
        #         # Find matching files
        #         matched = [p for p in all_paths if f"batch{d}" in p and modality in p]
        #         if matched:
        #             data_paths.append(matched[0])  # assume one file per combination
        batchID = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        batchID = batchID[:6]
        all_paths = glob.glob("../data/BMMC/BMMC_batch*_*.h5ad")
        data_paths = []
        
        for b in batchID:
            matched = [p for p in all_paths if f"batch{b}_" in p]
            if matched:
                data_paths.append(matched[0])  # assume one file per combination
        genes = np.load('../data/BMMC/gene_list_atac_adt.npy', allow_pickle=True).tolist()
        # genes = adata_train_org.var["gene_name"].tolist()
        print(len(genes))
        
    elif config["dataset_name"] == "BMMC_filter3":
        batchID = [0, 1, 2, 3, 4, 5]
        all_paths = glob.glob("../data/BMMC/filter3/BMMC_batch*_delete3_filtered.h5ad")
        data_paths = []
        
        for b in batchID:
            matched = [p for p in all_paths if f"batch{b}_" in p]
            if matched:
                data_paths.append(matched[0])  # assume one file per combination
        genes = np.load('../data/BMMC/gene_list_atac_adt.npy', allow_pickle=True).tolist()
        # genes = adata_train_org.var["gene_name"].tolist()
        print(len(genes))        
             
    model_dir = Path(config["load_model"])
    weight_dir = Path(config["weight_dir"])
    modeldict_name = "best_model.pt"
    # modeldict_name = "model_e25.pt"
    model_file = weight_dir / modeldict_name
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])
    if config["DSBN"]:
        num_batch_types = len(data_paths)    # originally 1
    else:
        num_batch_types = 1

    max_batch_idx = len(data_paths)-1 
    continual_classify = ContinualClassify(config, vocab, num_batch_types, genes, model_file, max_batch_idx)

    all_adata_test = []
    all_batch_results = {}
    all_learning_rate = []
    for test_batch_idx in range(len(data_paths)):
        adata_train = sc.read_h5ad(data_paths[test_batch_idx])
        if config["log1p"]:
            sc.pp.log1p(adata_train)                            # apply log1p to data
        if not sp.issparse(adata_train.X):
            adata_train.X = sp.csr_matrix(adata_train.X, dtype=np.float32)
        adata_train.var["gene_name"] = adata_train.var.index.tolist()
        adata_train.var["feature_type"] = adata_train.var["feature_type"].map({"RNA": 0, "ADT": 1, "ATAC":2}).astype(int)
        # adata_train.obs["mod_types"] = adata_train.obs["mod"].map({'RNA':0, 'Protein':1}).astype(int)
        mod_type = np.array(adata_train.var["feature_type"]) 

        best_model, combined_adata_test, learning_rate = continual_classify.process_batch(
            adata_train, logger, save_dir, config["dataset_name"], config["experiment_name"], all_adata_test, all_batch_results,
            test_batch_idx, mod_type
        )
        print('combined_adata_test.shape', combined_adata_test.shape)
        all_learning_rate.extend(learning_rate)
        ########################## Debug stage off ##################################
        # besteval_results, besteval_adata = continual_classify.best_model_evaluate(best_model, adata_t=combined_adata_test, gene_ids=gene_ids,
        #                                         input_layer_key={
        #                                             "normed_raw": "X_normed",
        #                                             "log1p": "X_normed",
        #                                             "binned": "X_binned",
        #                                         }["binned"],test_batch_idx=test_batch_idx)
    # plot_learning_rate(save_dir, all_learning_rate) 
    torch.cuda.empty_cache()
    del combined_adata_test, best_model, learning_rate
    import gc
    gc.collect()
    return save_dir
        
    #     with open(str(save_dir) + "/" + f"batch_{test_batch_idx}_bestval_results.json",
    #               "w") as f:
    #         json.dump(besteval_results, f, indent=4)

    #     sc.pp.neighbors(besteval_adata, use_rep="X_scGPT")
    #     sc.tl.umap(besteval_adata, min_dist=0.3)
    #     sc.pl.umap(
    #         besteval_adata,
    #         color=["str_batch"],
    #         title=[f"batch, avg_batch = {besteval_results.get('avg_batch', 0.0):.4f}"],
    #         frameon=False,
    #         show=False,
    #     )
    #     plt.savefig(
    #         str(save_dir) + "/" + f"embeddings_batch_umap[cls]_batch_{test_batch_idx}.png",
    #         bbox_inches='tight')

    #     sc.pp.neighbors(besteval_adata, use_rep="X_scGPT")
    #     sc.tl.umap(besteval_adata, min_dist=0.3)
    #     sc.pl.umap(
    #         besteval_adata,
    #         color=["celltype"],
    #         title=[f"celltype, avg_bio = {besteval_results.get('avg_bio', 0.0):.4f}"],
    #         frameon=False,
    #         show=False,
    #     )
    #     plt.savefig(
    #         str(save_dir) + "/" + f"embeddings_celltype_umap[cls]_batch_{test_batch_idx}.png",
    #         bbox_inches='tight')


    # final_results, final_adata = continual_classify.eval_testdata(
    #     model = best_model,
    #     adata_t=combined_adata_test,
    #     gene_ids=gene_ids,
    #     input_layer_key={
    #         "normed_raw": "X_normed",
    #         "log1p": "X_normed",
    #         "binned": "X_binned",
    #     }["binned"],test_batch_idx=test_batch_idx
    # )
    
    # with open(str(save_dir) + "/" + config["dataset_name"] + "_" + config["experiment_name"] + "_final_results.json",
    #           "w") as f:
    #     json.dump(final_results, f, indent=4)
    # print("**********************save json************************")
    # print('all_test_data:done', combined_adata_test.shape)
    
def predict(save_dir, cell_type_map, test_batch_list=None, experiment = None):
    with open(str(save_dir) + "/config.json", "r", encoding="utf-8") as f:
        config = json.load(f) 
    if 'classifier' not in config:
        config['classifier'] = 'Linear'
    if 'do_dab' not in config:
        config["do_dab"] = False
    if 'adapter_dim' not in config:
        config["adapter_dim"] = 64
    # config = init_wandb()
    set_seed(config["seed"])
    if config["dataset_name"] == "pancreas":
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6, 9)]  # remaining three batches used as test set
    elif config["dataset_name"] == "myeloid":
        data_paths = [f"../data/myeloid/myeloid_batch{i}.h5ad" for i in test_batch_list]
        
    elif config["dataset_name"] == "BMMC" and experiment != "query_mapping" and experiment != "outlier_detection":
        # batchID = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]                 # plot latent-space visualization
        batchID = [0, 1, 2, 5, 6, 7]                                                 # baselines
        all_paths = glob.glob("../data/BMMC/test/BMMC_batch*_*.h5ad")           # change path to test data
        data_paths = []
        
        for b in batchID:
            matched = [p for p in all_paths if f"batch{b}_" in p]
            if matched:
                data_paths.append(matched[0])  # assume one file per combination
        genes = np.load('../data/BMMC/gene_list_atac_adt.npy', allow_pickle=True).tolist()
        # genes = adata_train_org.var["gene_name"].tolist()
        print(len(genes))
    elif config["dataset_name"] == 'BMMC' and experiment == 'query_mapping':
        batchID = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]
        batchID = batchID[:6]
        all_paths = glob.glob("../data/BMMC/BMMC_batch*_*.h5ad")
        data_paths = []
        
        for b in batchID:
            matched = [p for p in all_paths if f"batch{b}_" in p]
            if matched:
                data_paths.append(matched[0])  # assume one file per combination
        
        batchID = [0, 1, 2, 5, 6, 7]                                                 # baselines
        all_paths = glob.glob("../data/BMMC/test/BMMC_batch*_*.h5ad")           # change path to test data
        
        for b in batchID:
            matched = [p for p in all_paths if f"batch{b}_" in p]
            if matched:
                data_paths.append(matched[0])  # assume one file per combination
        genes = np.load('../data/BMMC/gene_list_atac_adt.npy', allow_pickle=True).tolist()
        print(len(genes))
        
    elif config["dataset_name"] == "BMMC_filter3":
        batchID = [0, 1, 2, 3, 4, 5]                                                 # baselines
        all_paths = glob.glob("../data/BMMC/filter3/BMMC_batch*_*.h5ad")           # change path to test data
        data_paths = []
        
        for b in batchID:
            matched = [p for p in all_paths if f"batch{b}_" in p]
            if matched:
                data_paths.append(matched[0])  # assume one file per combination
        batchID_test = [0, 1, 2, 5, 6, 7]                                                 # baselines
        all_paths = glob.glob("../data/BMMC/test/BMMC_batch*_*.h5ad")           # change path to test data
        
        for b in batchID_test:
            matched = [p for p in all_paths if f"batch{b}_" in p]
            if matched:
                data_paths.append(matched[0])  # assume one file per combination
        genes = np.load('../data/BMMC/gene_list_atac_adt.npy', allow_pickle=True).tolist()
        print(len(genes))

    model_dir = Path(config["load_model"])
    modeldict_name = "best_model_batch_5.pt"
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab["<pad>"])

    num_batch_types = 1
    config["weight_dir"] = save_dir
    weight_dir = Path(config["weight_dir"])
    model_file = weight_dir / modeldict_name
    continual_classify = ContinualClassify(config, vocab, num_batch_types, genes, model_file, modeldict_name = modeldict_name)

    all_adata_test = []
    all_batch_results = {}

    if os.path.exists(str(save_dir) + "/" + "prototype_4.pkl"):
        if config["dataset_name"] == "myeloid":
            with open(str(save_dir) + "/" + "prototype_4.pkl", "rb") as f:
                old_proto = pickle.load(f)
        elif config["dataset_name"] == "pancreas":
            with open(str(save_dir) + "/" + "prototype_5.pkl", "rb") as f:
                old_proto = pickle.load(f)
        elif config["dataset_name"] == "BMMC" or config["dataset_name"] == "BMMC_filter3":
            with open(str(save_dir) + "/" + "prototype_5.pkl", "rb") as f:
                old_proto = pickle.load(f)
        else:
            pass
    else:
        old_proto = None
        
    if config["dataset_name"] == "BMMC_filter3":
        cell_type_map["pDC"] = 14                
        cell_type_map["ILC"] = 15
        cell_type_map["Lymph prog"] = 16
        
    for test_batch_idx in range(len(data_paths)):
        adata_test = sc.read_h5ad(data_paths[test_batch_idx])
        if not sp.issparse(adata_test.X):
            adata_test.X = sp.csr_matrix(adata_test.X, dtype=np.float32)
        adata_test.var["gene_name"] = adata_test.var.index.tolist()
        adata_test.var["feature_type"] = adata_test.var["feature_type"].map({"RNA": 0, "ADT": 1, "ATAC":2}).astype(int)
        mod_type = np.array(adata_test.var["feature_type"])
        # gene_ids = np.array(vocab(genes), dtype=int)
        gene_ids = continual_classify.gene_ids
        # gene_ids = continual_classify.predict(
        #     adata_test, save_dir, test_batch_idx, cell_type_map, mod_type
        # )
        ############################# Plot feature UMAP #############################
        if experiment == "plot_gene_umap":
            cell_gene_emb = continual_classify.predict(
            adata_test, save_dir, test_batch_idx, cell_type_map, mod_type, plot_gene_umap=True
        )   
            continual_classify.plot_feature_umap(cell_gene_emb, adata_test, gene_ids, test_batch_idx, save_dir, 
                                         cell_type_map=cell_type_map)
        #######################################################################
        # genes = adata_test.var["gene_name"].tolist()
        # gene_ids = np.array(vocab(genes), dtype=int)
        ###################### Compute gene importance #####################
        # gene_ids = continual_classify.forward_latent_with_ig(adata_test, save_dir, test_batch_idx, cell_type_map)
        ##########################################
        all_adata_test.append(adata_test)

        combined_adata_test = anndata.concat(
            all_adata_test,
            join="outer",
            merge="unique",
            label="batch_new",
            index_unique=None
        )

    # combined_adata_test.write_h5ad(f"{save_dir}/combined_adata_test.h5ad")

    
    # continual_classify.plot_gray_batch(combined_adata_test, old_proto, 
    #                          "X_binned", gene_ids, test_batch_idx, 
    #                          save_dir, save_name=f"batch_test_{test_batch_idx}", 
    #                          cell_type_map=cell_type_map, legend_on=False,  experiment = "query_mapping")
    # continual_classify.plot_gray_batch(combined_adata_test, old_proto, 
    #                          "X_binned", gene_ids, test_batch_idx, 
    #                          save_dir, save_name=f"batch_test_{test_batch_idx}", 
    #                          cell_type_map=cell_type_map, legend_on=False,  experiment = "outlier_detection")
    # continual_classify.plot_gray_batch(combined_adata_test, old_proto, 
    #                          "X_binned", gene_ids, test_batch_idx, 
    #                          save_dir, save_name=f"batch_test_{test_batch_idx}", cell_type_map=cell_type_map, legend_on=True)
    # gene_ids = continual_classify.predict(
    #         combined_adata_test, save_dir, test_batch_idx = test_batch_idx+1, cell_type_map = cell_type_map, mod_type = mod_type
    #     )
    ############## Compute confidence code ######################
    gene_ids = continual_classify.predict_confidence(
                combined_adata_test, save_dir, \
                test_batch_idx = test_batch_idx+1, cell_type_map = cell_type_map, mod_type = mod_type
    )

    # continual_classify.plot_clusters_prototypes(combined_adata_test, old_proto, 
    #                          "X_binned", gene_ids, test_batch_idx+100, save_dir, save_name="batch_testall_")
    
    # final_results, final_adata = continual_classify.eval_testdata(
    #     model = continual_classify.model,
    #     adata_t=combined_adata_test,
    #     gene_ids=gene_ids,
    #     input_layer_key={
    #         "normed_raw": "X_normed",
    #         "log1p": "X_normed",
    #         "binned": "X_binned",
    #     }["binned"]
    # )

    # with open(str(save_dir) + "/" + "predict_all_finaltest_results.json",
    #           "w") as f:
    #     json.dump(final_results, f, indent=4)
    
    # sc.pp.neighbors(final_adata, use_rep="X_scGPT")
    # sc.tl.umap(final_adata, min_dist=0.3)
    # sc.pl.umap(
    #     final_adata,
    #     color=["str_batch"],
    #     title=[f"batch, avg_batch = {final_results.get('avg_batch', 0.0):.4f}"],
    #     frameon=False,
    #     show=False,
    # )
    # plt.savefig(
    #     str(save_dir) + "/" + f"embeddings_batch_umap[cls]_batch_testall.png",
    #     bbox_inches='tight')

    # sc.pp.neighbors(final_adata, use_rep="X_scGPT")
    # sc.tl.umap(final_adata, min_dist=0.3)
    # sc.pl.umap(
    #     final_adata,
    #     color=["celltype"],
    #     title=[f"celltype, avg_bio = {final_results.get('avg_bio', 0.0):.4f}"],
    #     frameon=False,
    #     show=False,
    # )
    # plt.savefig(
    #     str(save_dir) + "/" + f"embeddings_celltype_umap[cls]_batch_testall.png",
    #     bbox_inches='tight')
    
if __name__ == "__main__":

    save_dir = "../save/dev_BMMC-Oct23-10-16-37"    # Latest BMMC plotting
    with open(save_dir + "/celltype_to_label.json", "r") as f:
        cell_type_map = json.load(f)
        f.close()
    # evaluate_predict(save_dir, cell_type_map)                                         # plotting for validation set
    # predict(save_dir, cell_type_map, test_batch_list=None, experiment = "query_mapping")                              # test set
    # predict(save_dir, cell_type_map, test_batch_list=None, experiment = "plot_gene_umap")
    # print("cell_type_map", cell_type_map)
    predict(save_dir, cell_type_map, test_batch_list=None, experiment = None) 
    # predict(save_dir, cell_type_map, test_batch_list=None, experiment = "outlier_detection")
    # plot_bank(save_dir, cell_type_map)

