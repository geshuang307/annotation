from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
import torch
from pathlib import Path
import scanpy as sc
from scgpt.tokenizer.gene_tokenizer import GeneVocab
import json
from scipy.sparse import issparse
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
from torchtext.vocab import Vocab
from torch.utils.data import Dataset, DataLoader
from scgpt import SubsetsBatchSampler
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
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

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

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
        # if hyperparameter_defaults['load_model'] is None:
        #     vocab = Vocab(
        #         VocabPybind(genes + special_tokens, None)
        #     )  # bidirectional lookup [gene <-> int]
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
        return adata, adata_sorted, embsize, nhead, d_hid, nlayers, vocab, num_batch_types, model_file, \
            tokenized_train, tokenized_valid, train_batch_labels, valid_batch_labels, input_layer_key,gene_ids, sample_size

    # def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    def prepare_data(
        tokenized_train: Dict[str, torch.Tensor], train_batch_labels: torch.Tensor,
        tokenized_valid: Dict[str, torch.Tensor], valid_batch_labels: torch.Tensor,
        sort_seq_batch: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        
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

if __name__=='__main':
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
