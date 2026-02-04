import os
import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from anndata import AnnData

import scib
from scgpt.tokenizer import random_mask_value
from scgpt import SubsetsBatchSampler

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

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
    per_seq_batch_sample: bool = False,
) -> DataLoader:
    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
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

def prepare_testdata(sort_seq_batch, tokenized_test, test_batch_labels, test_celltype_labels, mask_ratio, mask_value, pad_value):
    masked_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    input_gene_ids_test = (tokenized_test["genes"])
    input_values_test = masked_values_test
    target_values_test = (tokenized_test["values"])
    tensor_batch_labels_test = torch.from_numpy(test_batch_labels).long()
    tensor_celltype_labels_test = torch.from_numpy(test_celltype_labels).long()
    if sort_seq_batch:
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

def prepare_data(sort_seq_batch, tokenized_train, tokenized_valid, train_batch_labels, valid_batch_labels, train_celltype_labels, valid_celltype_labels, mask_ratio, mask_value, pad_value):
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

    if sort_seq_batch:
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

    if input_gene_ids_train.shape[0] >= 2000:
        (input_gene_ids_train,
                input_values_train,
                target_values_train,
                tensor_batch_labels_train,
                tensor_celltype_labels_train, indices_to_keep) = stratified_sample(
                    input_gene_ids_train,
                    input_values_train,
                    target_values_train,
                    tensor_batch_labels_train,
                    tensor_celltype_labels_train,
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

    return train_data_pt, valid_data_pt

def eval_scib_metrics(adata: AnnData, batch_key: str = "str_batch", label_key: str = "celltype", notes: Optional[str] = None) -> Dict:
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
        nmi_=True,
        ari_=True,
        cell_cycle_=False,
        kBET_=False,
        ilisi_=False,
        clisi_=False,
    )
    if notes is not None:
        print(f"{notes}")

    result_dict = results[0].to_dict()
    result_dict = {k: (0 if isinstance(v, float) and np.isnan(v) else v) for k, v in result_dict.items()}

    result_dict["avg_bio"] = np.mean([
        result_dict.get("NMI_cluster/label", 0),
        result_dict.get("ARI_cluster/label", 0),
        result_dict.get("ASW_label", 0),
    ])
    result_dict["avg_batch"] = np.mean([
        result_dict.get("graph_conn", 0),
        result_dict.get("ASW_label/batch", 0),
    ])

    result_dict = {k: v for k, v in result_dict.items() if not (isinstance(v, float) and np.isnan(v))}

    return result_dict


def plot_entropy_accuracy(entropy_list, correct_mask, save_dir, test_batch_idx, labellist):
    entropy = torch.stack(entropy_list).cpu().numpy()
    correct_mask = torch.stack(correct_mask).cpu().numpy()

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

    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(data=df, x="CellType", y="Entropy", hue="Correct", palette="Set2")
    plt.title(f"Entropy Distribution per Cell Type (Batch {test_batch_idx})", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"entropy_boxplot_batch{test_batch_idx}.png"))


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


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, nb_proxy=1, to_reduce=False, sigma=True):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
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

    def forward(self, input, label=None, margin=0.35):
        cosine = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))

        if label is not None:
            one_hot = F.one_hot(label, num_classes=cosine.size(1)).float().to(input.device)
            cosine = cosine - one_hot * margin

        if self.to_reduce:
            cosine = reduce_proxies(cosine, self.nb_proxy)

        if self.sigma is not None:
            cosine = self.sigma * cosine

        return cosine


class ClsDecoder(nn.Module):
    def __init__(self, d_model: int, n_cls: int, nlayers: int = 3, activation: callable = nn.ReLU, classifier: str = "Linear"):
        super().__init__()
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        if classifier == "Linear":
            self.out_layer = nn.Linear(d_model, n_cls)
        elif classifier == "Cosine":
            self.out_layer = CosineLinear(d_model, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self._decoder:
            x = layer(x)
        out = self.out_layer(x)
        return out
