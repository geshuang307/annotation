import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7" 
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
from multimodal_prototype import prepare_testdata, prepare_dataloader
from sklearn.preprocessing import normalize, MinMaxScaler
from scipy.stats import mannwhitneyu

def enable_dropout_only(model: nn.Module):
    """
    Set model to eval globally, then switch Dropout submodules back to train().
    This keeps Dropout stochastic while BatchNorm/LayerNorm and other modules remain in eval mode.
    """
    model.eval()
    for m in model.modules():
        # Enable all Dropout modules (including in attention/MLP)
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.train()
        # Explicitly keep BatchNorm modules in eval (safer)
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    return model

def check_modes(model):
    dps, bns = [], []
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            dps.append(m.training)   # expect True
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bns.append(m.training)   # expect False
    print("Any Dropout in train():", any(dps), "All BN eval():", all(not b for b in bns))


def plot_single(X_umap, values, save_dir, save_name):
    fig, ax = plt.subplots(figsize=(6, 5))

    sc_plot = ax.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=values,
        s=30000 / values.shape[0],
        cmap="viridis",            # use viridis
        linewidth=0,
        edgecolors='none'
    )

    # Unified appearance
    # ax.set_title(gene, fontsize=12)   # keep title only once
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar
    cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label("Expression", fontsize=9)

    # Remove spines and ticks
    # for spine in ax.spines.values():
    #     spine.set_visible(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_title("UMAP coloured by predicted cell type", fontsize=12)
    plt.savefig(str(save_dir) + "/" + "predict_" + save_name + ".png", dpi=300)
    
def compute_confidence_myeloid(Times, model, config, device, gene_ids, vocab, adata_test, save_dir, test_batch_idx, 
                               cell_type_map, prototype_file = f"prototype_{4}.pkl"):
    # from testtime_dropout import enable_dropout_only, check_modes
    from utils import prepare_testdata, prepare_dataloader
    config["weight_dir"] = save_dir
    le = LabelEncoder()
    adata_test.obs["batch_id"] = le.fit_transform(adata_test.obs["batch_id"])
    num_batch_types = adata_test.obs["batch_id"].nunique()
    input_layer_key = "X_binned"
    all_counts = (
        adata_test.layers[input_layer_key].A
        if issparse(adata_test.layers[input_layer_key])
        else adata_test.layers[input_layer_key]
    )                          # (95550, 1893)
    # compute cell types and labels for current batch (test_batch_idx)
    # current_label_dict, current_celltype_labels = np.unique(
    #     np.array(adata_test.obs["celltype"].tolist()), return_inverse=True
    # )
    all_counts = np.nan_to_num(all_counts, nan=0.0)
    celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()                 # string
    current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]     # numeric
    current_celltype_labels = np.array(current_celltype_labels)
    adata_test.obs["celltype_labels"] = current_celltype_labels                   # temporarily save
    batch_ids = adata_test.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)
   
    tokenized_test = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=3001,
        vocab=vocab,
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
        mask_ratio=config["mask_ratio"],
        mask_value=-1,
        pad_value=-2,
    )

    test_loader = prepare_dataloader(
        test_data_pt,
        batch_size=config["batch_size"],
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
        per_seq_batch_sample=False
    )
    # self.model.eval()
    model = enable_dropout_only(model)
    check_modes(model)
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    accuracy = 0
    num_batches = len(test_loader)
    all_runs_conf = []      # per-run per-sample true-label probability vectors
    all_runs_entropy = []   # per-run per-sample entropy vectors
    all_indices = []        # (optional) record batch concatenation order
    cell_emb_list_all = []
    predictions_list_all = []
    with torch.no_grad():
        for t in range(Times):
            run_conf_list = []      # per-batch results for this run (true-label probabilities)
            run_entropy_list = []   # per-batch entropy results for this run (entropy)
            cell_emb_list = []
            predictions_list = []
            labellist = []
            for batch, batch_data in enumerate(itertools.islice(test_loader, num_batches)):
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)
                src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
                
                with torch.cuda.amp.autocast(enabled=config["amp"]):
                    if config["adapter"] or config["loramoe"] or config["lora"]:
                        output_dict, _ = model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if config["DSBN"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=config["MVC"],
                            ECS=config["ecs_thres"] > 0,
                            do_sample=False,
                        )
                    else:
                        output_dict = model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if config["DSBN"] else None,
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
                    cell_emb_list.append(cell_emb.cpu().detach().numpy())
                    probs = F.softmax(output_values, dim=-1)
                # total_loss += loss.item() * len(input_gene_ids)
                
                p_y = probs.gather(1, celltype_labels.unsqueeze(1)).squeeze(1)
                run_conf_list.append(p_y.detach().cpu())

                accuracy += (output_values.argmax(1) == celltype_labels).sum().item()
                print(accuracy)
                # total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                # total_num += len(input_gene_ids)
                preds = output_values.argmax(1).detach().cpu()
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                run_entropy_list.append(entropy.detach().cpu())
                predictions_list.extend(preds)
                labellist.extend(celltype_labels.detach().cpu())
            cell_emb_matrix = np.concatenate(cell_emb_list, axis=0)
            cell_emb_list_all.append(cell_emb_matrix)
            predictions_matrix = torch.stack(predictions_list)
            run_conf = torch.cat(run_conf_list, dim=0)         # [num_samples]
            run_entropy = torch.cat(run_entropy_list, dim=0)   # [num_samples]
            all_runs_conf.append(run_conf)                     # append one run
            all_runs_entropy.append(run_entropy)
            predictions_list_all.append(predictions_matrix)
        conf_T = torch.stack(all_runs_conf, dim=0)             # [T, N]
        ent_T  = torch.stack(all_runs_entropy, dim=0)          # [T, N]
        predictions_all = torch.stack(predictions_list_all, dim=0)  # # [T, N]
        final_predictions = torch.mode(predictions_all, dim=0)[0].numpy()
        labels = torch.stack(labellist, dim=0).numpy()
        is_correct_array = final_predictions == labels
  
        
        mean_conf = conf_T.mean(dim=0).numpy()                 # (N,)
        var_conf  = conf_T.var(dim=0, unbiased=False).numpy()  # (N,)
        entropies_mat = ent_T.mean(dim=0).numpy() 
        cell_emb_list_all = [normalize(Z, axis=1) for Z in cell_emb_list_all]
        Z_mean = np.mean(cell_emb_list_all, axis=0)
        
        with open(str(save_dir) + "/" + prototype_file, "rb") as f:
            prototype = pickle.load(f)
        prototype_list = [prototype[c] for c in sorted(prototype.keys())]
        prototype_tensor = torch.stack(prototype_list, dim=0)  # shape: [num_classes, D]

        prototypes = F.normalize(prototype_tensor, dim=1) 
        prototypes_np = prototypes.detach().cpu().numpy() if torch.is_tensor(prototypes) else prototypes
        n_prototypes = prototypes_np.shape[0]
        print("num_proto", n_prototypes)

        # Step 2: Concatenate all vectors
        X_all = np.concatenate([Z_mean, prototypes_np], axis=0)
        np.save(os.path.join(save_dir, "X_all.npy"), X_all)
        
        #####################
        # X_all = np.load(os.path.join(save_dir, "X_all.npy"))
        # if cell_type_map is not None:
        #     # index2cell = {v: k for k, v in cell_type_map.items()}

        #     # celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()   
        #     celltype_str_list = np.array(adata_test.obs["celltype_origion"]).tolist()            
        #     current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        #     current_celltype_labels = np.array(current_celltype_labels)
        #     adata_test.obs["celltype_labels"] = current_celltype_labels
        #     labels = adata_test.obs["celltype_labels"]
        # else:
        #     labels = adata_test.obs["celltype"]\
        if cell_type_map is not None:
            index2cell = {v: k for k, v in cell_type_map.items()}
            celltype_str_list = np.array(adata_test.obs["celltype_origion"]).tolist()     
            # celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
            # current_celltype_labels = [index2cell[cell_type_map[cell]] for cell in celltype_str_list]
            # current_celltype_labels = np.array(current_celltype_labels)
            current_celltype_labels = []
            for cell in celltype_str_list:
                    # if a cell isn't in the mapping, keep original name or mark as 'unknown'
                mapped_index = cell_type_map.get(cell, None)
                if mapped_index is not None:
                    current_celltype_labels.append(index2cell.get(mapped_index, cell))
                else:
                    current_celltype_labels.append(cell)   # or 'unknown'
                    
            adata_test.obs["celltype_labels"] = current_celltype_labels
            labels = adata_test.obs["celltype_labels"]
        else:
            labels = adata_test.obs["celltype"]
        # Step 3: Dimensionality reduction
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="euclidean", random_state=42)         # adjusted plotting configuration
        X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]
        # -------------------- only use colors for labels that actually appear --------------------
        all_celltypes = np.unique(labels)  # labels that actually appear (e.g., 12)
        palette = sns.color_palette("tab20", n_colors=len(all_celltypes))
        color_map = dict(zip(all_celltypes, palette))

        # ######################## Plot UMAP (colored by label) ######################
        plt.figure(figsize=(7, 6))

        sns.scatterplot(
            x=X_umap[:labels.shape[0], 0],
            y=X_umap[:labels.shape[0], 1],
            hue=labels,
            palette=color_map,            # map only labels that appear
            s=100000 / labels.shape[0],
            linewidth=0,
            alpha=1.0,
            legend=True            # True/False or "brief"/"full"
        )

        plt.title("UMAP Visualization", fontsize=24)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)

        # unify legend style (show only labels that appear)

        plt.legend(
            title="Labels",
            markerscale=15,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=False
        )

        out_path = os.path.join(str(save_dir), f"predict_umap_labels_Times{Times}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    
        # np.save(os.path.join(save_dir, "mean_conf.npy"),    mean_conf)
        # np.save(os.path.join(save_dir, "var_conf.npy"),     var_conf)
        # np.save(os.path.join(save_dir, "entropy_mean.npy"), entropies_mat)
        # np.save(os.path.join(save_dir, "X_umap_new.npy"),       X_umap)         
        # np.save(os.path.join(save_dir, "is_correct_array.npy"), is_correct_array)
        # np.save(os.path.join(save_dir, "final_predictions.npy"), final_predictions)


        # plot_single(X_umap[:Z_mean.shape[0], :], mean_conf, save_dir, save_name = "confidence")
        # plot_single(X_umap[:Z_mean.shape[0], :], var_conf, save_dir, save_name = "variance")
        # plot_single(X_umap[:Z_mean.shape[0], :], entropies_mat, save_dir, save_name = "entropy")
        
        # adata_test.obsm["cell_emb"] = Z_mean
        # adata_test.write_h5ad(f"{save_dir}/adata_test_embedding.h5ad")
        
def compute_confidence(Times, model, config, device, gene_ids, vocab, vocab_mod, adata_test, save_dir, test_batch_idx, cell_type_map, mod_type):
    # from testtime_dropout import enable_dropout_only, check_modes
    config["weight_dir"] = save_dir
    le = LabelEncoder()
    adata_test.obs["batch_id"] = le.fit_transform(adata_test.obs["batch_id"])
    num_batch_types = adata_test.obs["batch_id"].nunique()
    input_layer_key = "X_binned"
    all_counts = (
        adata_test.layers[input_layer_key].A
        if issparse(adata_test.layers[input_layer_key])
        else adata_test.layers[input_layer_key]
    )
    # 计算当前批次 (test_batch_idx) 的细胞类型和标签
    # current_label_dict, current_celltype_labels = np.unique(
    #     np.array(adata_test.obs["celltype"].tolist()), return_inverse=True
    # )
    celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()                 # 字符
    current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]     # 数字
    current_celltype_labels = np.array(current_celltype_labels)
    adata_test.obs["celltype_labels"] = current_celltype_labels                   # 先临时保存一下
    batch_ids = adata_test.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)
    if config["use_multimod"]:
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
        gene_ids,
        max_len=3001,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=True,
        mod_type = mod_type if config["use_mod"] else None,
        vocab_mod = vocab_mod if config["use_mod"] else None,
    )
    if config["use_multimod"]:
        test_data_pt = prepare_testdata(
            sort_seq_batch=False,
            tokenized_test=tokenized_test,
            test_batch_labels=batch_ids,
            test_celltype_labels=current_celltype_labels,
            test_multimod_labels=test_multimod_labels,
            mask_ratio=config["mask_ratio"],
            mask_value=-1,
            pad_value=-2,
        )

    test_loader = prepare_dataloader(
        test_data_pt,
        batch_size=config["batch_size"],
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
        per_seq_batch_sample=False
    )
    # self.model.eval()
    model = enable_dropout_only(model)
    check_modes(model)
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    accuracy = 0
    num_batches = len(test_loader)
    all_runs_conf = []      # 每次运行，逐样本“真实标签概率”向量
    all_runs_entropy = []   # 每次运行，逐样本熵向量
    all_indices = []        # （可选）记录批次拼接顺序
    cell_emb_list_all = []
    predictions_list_all = []
    with torch.no_grad():
        for t in range(Times):
            run_conf_list = []      # 当前轮的逐 batch 结果（真实标签概率）
            run_entropy_list = []   # 当前轮的逐 batch 结果（熵）
            cell_emb_list = []
            predictions_list = []
            labellist = []
            for batch, batch_data in enumerate(itertools.islice(test_loader, num_batches)):
                input_gene_ids = batch_data["gene_ids"].to(device)
                input_values = batch_data["values"].to(device)
                target_values = batch_data["target_values"].to(device)
                batch_labels = batch_data["batch_labels"].to(device)
                celltype_labels = batch_data["celltype_labels"].to(device)
                src_key_padding_mask = input_gene_ids.eq(vocab["<pad>"])
                mod_types = batch_data["mod_types"].to(device)
                if config["use_multimod"]:
                    multimod_labels = batch_data["multimod_types"].to(device)
                with torch.cuda.amp.autocast(enabled=config["amp"]):
                    if config["adapter"] or config["loramoe"] or config["lora"]:
                        output_dict, _ = model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if config["DSBN"] else None,
                            # batch_labels= test_batch_idx if False or self.config["DSBN"] else None,
                            # batch_id = torch.tensor(test_batch_idx),
                            # batch_id = None,
                            multimod_labels=multimod_labels if config["use_multimod"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=config["MVC"],
                            ECS=config["ecs_thres"] > 0,
                            mod_types=mod_types if config["use_mod"] else None,
                            do_sample=False,
                        )
                    else:
                        output_dict = model(
                            input_gene_ids,
                            input_values,
                            src_key_padding_mask=src_key_padding_mask,
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if config["DSBN"] else None,
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
                    cell_emb_list.append(cell_emb.cpu().detach().numpy())
                    probs = F.softmax(output_values, dim=-1)
                # total_loss += loss.item() * len(input_gene_ids)
                
                p_y = probs.gather(1, celltype_labels.unsqueeze(1)).squeeze(1)
                run_conf_list.append(p_y.detach().cpu())

                accuracy += (output_values.argmax(1) == celltype_labels).sum().item()
                print(accuracy)
                # total_error += (1 - accuracy / len(input_gene_ids)) * len(input_gene_ids)
                # total_num += len(input_gene_ids)
                preds = output_values.argmax(1).detach().cpu()
                
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                run_entropy_list.append(entropy.detach().cpu())
                predictions_list.extend(preds)
                labellist.extend(celltype_labels.detach().cpu())
            cell_emb_matrix = np.concatenate(cell_emb_list, axis=0)
            cell_emb_list_all.append(cell_emb_matrix)
            predictions_matrix = torch.stack(predictions_list)
            run_conf = torch.cat(run_conf_list, dim=0)         # [num_samples]
            run_entropy = torch.cat(run_entropy_list, dim=0)   # [num_samples]
            all_runs_conf.append(run_conf)                     # 追加一轮
            all_runs_entropy.append(run_entropy)
            predictions_list_all.append(predictions_matrix)
        conf_T = torch.stack(all_runs_conf, dim=0)             # [T, N]
        ent_T  = torch.stack(all_runs_entropy, dim=0)          # [T, N]
        predictions_all = torch.stack(predictions_list_all, dim=0)  # # [T, N]
        final_predictions = torch.mode(predictions_all, dim=0)[0].numpy()
        labels = torch.stack(labellist, dim=0).numpy()
        is_correct_array = final_predictions == labels
  
        
        mean_conf = conf_T.mean(dim=0).numpy()                 # (N,)
        var_conf  = conf_T.var(dim=0, unbiased=False).numpy()  # (N,)
        entropies_mat = ent_T.mean(dim=0).numpy() 
        cell_emb_list_all = [normalize(Z, axis=1) for Z in cell_emb_list_all]
        Z_mean = np.mean(cell_emb_list_all, axis=0)
        
        with open(str(save_dir) + "/" + f"prototype_{5}.pkl", "rb") as f:
            prototype = pickle.load(f)
        prototype_list = [prototype[c] for c in sorted(prototype.keys())]
        prototype_tensor = torch.stack(prototype_list, dim=0)  # shape: [num_classes, D]

        prototypes = F.normalize(prototype_tensor, dim=1) 
        prototypes_np = prototypes.detach().cpu().numpy() if torch.is_tensor(prototypes) else prototypes
        n_prototypes = prototypes_np.shape[0]
        print("num_proto", n_prototypes)

        # # Step 2: 合并所有向量
        X_all = np.concatenate([Z_mean, prototypes_np], axis=0)

        # Step 3: 降维
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="euclidean", random_state=42)
        X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]
        
        np.save(os.path.join(save_dir, "mean_conf.npy"),    mean_conf)
        np.save(os.path.join(save_dir, "var_conf.npy"),     var_conf)
        np.save(os.path.join(save_dir, "entropy_mean.npy"), entropies_mat)
        np.save(os.path.join(save_dir, "X_umap.npy"),       X_umap)
        np.save(os.path.join(save_dir, "is_correct_array.npy"), is_correct_array)
        np.save(os.path.join(save_dir, "final_predictions.npy"), final_predictions)


        # plot_single(X_umap[:Z_mean.shape[0], :], mean_conf, save_dir, save_name = "confidence")
        # plot_single(X_umap[:Z_mean.shape[0], :], var_conf, save_dir, save_name = "variance")
        # plot_single(X_umap[:Z_mean.shape[0], :], entropies_mat, save_dir, save_name = "entropy")
        
        adata_test.obsm["cell_emb"] = Z_mean
        adata_test.write_h5ad(f"{save_dir}/adata_test_embedding.h5ad")

def plot_celltype_prototype(cell_type_map, adata_test, X_umap, X_cell, proto, save_dir):
    from matplotlib.lines import Line2D
    if cell_type_map is not None:
        index2cell = {v: k for k, v in cell_type_map.items()}

        celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
        current_celltype_labels = [index2cell[cell_type_map[cell]] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels
        labels = adata_test.obs["celltype_labels"]
    else:
        labels = adata_test.obs["celltype"]
    
    all_celltypes = sorted(np.unique(labels))
    palette = sns.color_palette("tab20", n_colors=len(all_celltypes))
    color_map = dict(zip(all_celltypes, palette))   # {celltype: color}
    
    # plt.figure(figsize=(5, 5))
    plt.figure(figsize=(10, 6))
    scatter_cells = sns.scatterplot(
        x=X_umap[:X_cell.shape[0], 0],
        y=X_umap[:X_cell.shape[0], 1],
        hue=labels,
        palette=color_map,   # 固定颜色
        s=30000 / X_cell.shape[0],
        linewidth=0,
        # legend=False
        legend=True
        )
    if proto is not None:
        scatter_prototypes = plt.scatter(
            X_umap[X_cell.shape[0]:, 0],
            X_umap[X_cell.shape[0]:, 1],
            edgecolors='black',
            facecolors='none',
            s=60,
            marker='X',
            label='Prototypes',
        )
    plt.xticks([])  # remove x-axis ticks
    plt.yticks([])  # remove y-axis ticks
    # Create custom legend handles for cell sizes and prototypes
    # cell_legend = Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=20, label="Cells")
    # proto_legend = Line2D([0], [0], marker='X', color='w', markerfacecolor='none', markeredgewidth=2, markersize=20, label="Prototypes")
    # cell_legend = scatter_cells.legend_elements()[0][0]  # Get the first element for cells
    # proto_legend = Line2D([0], [0], marker='X', color='w', markerfacecolor='none', markeredgewidth=2, markersize=20, label="Prototypes")

    # Add the custom legend to the plot
    # plt.legend(handles=[cell_legend, proto_legend], markerscale=1, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, handleheight=2.5) 
    plt.tight_layout()
    plt.savefig(
            str(save_dir) + "/" + f"predict_celltype_prototype.png",
            bbox_inches='tight',
            edgecolor='black',
            dpi=300
            )
    plt.close()

def plot_celltype_prototype_gut(cell_type_map, adata_test, X_umap, X_cell, proto, save_dir):
    from matplotlib.lines import Line2D
    if cell_type_map is not None:
        index2cell = {v: k for k, v in cell_type_map.items()}

        celltype_str_list = np.array(adata_test.obs["celltype_origion"]).tolist()
        current_celltype_labels = [index2cell[cell_type_map[cell]] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels
        labels = adata_test.obs["celltype_labels"]
    else:
        labels = adata_test.obs["celltype_origion"]
    
    all_celltypes = sorted(np.unique(labels))
    palette = sns.color_palette("tab20", n_colors=len(all_celltypes))
    color_map = dict(zip(all_celltypes, palette))   # {celltype: color}
    
    # plt.figure(figsize=(5, 5))
    plt.figure(figsize=(10, 6))
    scatter_cells = sns.scatterplot(
        x=X_umap[:X_cell.shape[0], 0],
        y=X_umap[:X_cell.shape[0], 1],
        hue=labels,
        palette=color_map,   # 固定颜色
        s=30000 / X_cell.shape[0],
        linewidth=0,
        # legend=False
        legend=True
        )
    if proto is not None:
        scatter_prototypes = plt.scatter(
            X_umap[X_cell.shape[0]:, 0],
            X_umap[X_cell.shape[0]:, 1],
            edgecolors='black',
            facecolors='none',
            s=60,
            marker='X',
            label='Prototypes',
        )
    plt.xticks([])  # 去掉x轴刻度
    plt.yticks([])  # 去掉y轴刻度

    ################# legend #################
    cell_legend = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[cell], markersize=10, label=cell) for cell in color_map]

    # 2. 为原型生成自定义图例
    proto_legend = Line2D([0], [0], marker='X', linestyle='None', markerfacecolor='none', markeredgecolor='black', markeredgewidth=2, markersize=15, label="Prototypes")
    
    # plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, handleheight=2.5) 
    # plt.legend(handles=[cell_legend, proto_legend], bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.legend(handles=cell_legend + [proto_legend], bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    plt.savefig(
            str(save_dir) + "/" + f"predict_celltype_prototype.png",
            bbox_inches='tight',
            edgecolor='black',
            dpi=300
            )
    plt.close()
    
def plot_eachclasses_dist_conf_variance(labels, sample_proto_distances, mean_conf, var_conf, save_dir):
    
    num_classes = len(np.unique(labels))  # 类别数
    
    # 使用 MinMaxScaler 来进行归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 计算子图的行列数（最多 18 个子图）
    num_rows = 4
    num_cols = 5

    # 创建图形，设置子图布局为三行六列
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 20))
    
    # 如果类别数少于 18，删除多余的子图
    for i in range(num_classes, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    for class_id in range(num_classes):
        # 获取当前类别的样本
        class_mask = (labels == class_id)
        if np.sum(np.array(class_mask)) > 0:
            # 当前类别的方差、置信度和距离
            class_dists = sample_proto_distances[class_mask]
            class_conf = mean_conf[class_mask]  # 置信度
            class_var = var_conf[class_mask]    # 方差

            # 对当前类别的距离进行归一化
            class_dists_normalized = scaler.fit_transform(class_dists.reshape(-1, 1)).flatten()  # 归一化到 [0, 1] 范围

            # 确定当前类别绘制在哪个子图上
            ax = axes.flatten()[class_id]

            # 根据归一化的距离绘制散点图，点的颜色根据归一化的距离来设置
            sc = ax.scatter(class_var, class_conf, c=class_dists_normalized, cmap='plasma', alpha=0.7, s=10)

            # 设置标题和坐标轴标签
            ax.set_title(f"Class {class_id} - Var vs Confidence", fontsize=12)
            ax.set_xlabel('Variance')
            ax.set_ylabel('Confidence')

            # 添加颜色条
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Normalized Distance', fontsize=9)
        else:
            pass

    # 调整布局，确保子图不重叠
    plt.tight_layout()

    # 保存图像
    plt.savefig(f"{save_dir}/normalized_distance_var_vs_confidence_all_classes.png", dpi=300)
    
def plot_dist_conf_variance(labels, sample_proto_distances, mean_conf, var_conf, save_dir):

    num_classes = len(np.unique(labels))  # 类别数
    
    # 存储合并后的数据
    all_dists = []
    all_conf = []
    all_var = []
    all_labels = []

    # 使用 MinMaxScaler 来进行归一化
    scaler = MinMaxScaler(feature_range=(0, 1))

    for class_id in range(num_classes):
        # 获取当前类别的样本
        class_mask = (labels == class_id)
        if np.sum(np.array(class_mask)) > 0:
            # 当前类别的方差、置信度和距离
            class_dists = sample_proto_distances[class_mask]
            class_conf = mean_conf[class_mask]  # 置信度
            class_var = var_conf[class_mask]    # 方差

            # 对当前类别的距离进行归一化
            class_dists_normalized = scaler.fit_transform(class_dists.reshape(-1, 1)).flatten()  # 归一化到 [0, 1] 范围

            # 将当前类别的数据添加到合并的数据列表中
            all_dists.extend(class_dists_normalized)
            all_conf.extend(class_conf)
            all_var.extend(class_var)
            all_labels.extend([class_id] * len(class_dists))  # 保存每个样本的类别标签
        else:
            pass

    # 将合并后的数据转化为 NumPy 数组
    all_dists = np.array(all_dists)
    all_conf = np.array(all_conf)
    all_var = np.array(all_var)
    all_labels = np.array(all_labels)

    # 绘制合并后的散点图
    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(all_var, all_conf, c=all_dists, cmap='plasma', alpha=0.7, s = 30000/all_dists.shape[0])

    # 设置标题和坐标轴标签
    ax.set_title("Variance vs Confidence (All Classes)", fontsize=14)
    ax.set_xlabel('Variance')
    ax.set_ylabel('Confidence')

    # 添加颜色条
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Normalized Distance to Prototype', fontsize=9)

    # 保存图像
    plt.tight_layout()
    plt.savefig(str(save_dir) + "/" + "combined_distance_confidence_variance.png", dpi=300)
    
# 创建一个新的图形，单独绘制三个直方图
    fig, axes = plt.subplots(1, 3, figsize=(18, 3))

    # 绘制 Confidence 分布曲线（左侧）
    axes[0].hist(all_conf, bins=50, density=False, color='gray', alpha=0.7)
    axes[0].set_title('Confidence Distribution', fontsize=12)
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Density')
    
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['left'].set_visible(True)
    axes[0].spines['bottom'].set_visible(True)

    # 绘制 Variance 分布曲线（中间）
    axes[1].hist(all_var, bins=50, density=False, color='gray', alpha=0.7)
    axes[1].set_title('Variance Distribution', fontsize=12)
    axes[1].set_xlabel('Variance')
    axes[1].set_ylabel('Density')

    # 去除顶部和右侧的边框
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['left'].set_visible(True)
    axes[1].spines['bottom'].set_visible(True)
    
    # 绘制 Distance 分布曲线（右侧）
    axes[2].hist(all_dists, bins=50, density=False, color='purple', alpha=0.7)
    axes[2].set_title('Distance Distribution', fontsize=12)
    axes[2].set_xlabel('Distance')
    axes[2].set_ylabel('Density')
    # 去除顶部和右侧的边框
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['left'].set_visible(True)
    axes[2].spines['bottom'].set_visible(True)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(str(save_dir) + "/" + "distributions_confidence_variance_distance.png", dpi=300)
    plt.close(fig)

def plot_normalized_distance_vs_confidence(is_correct_array, mean_conf, distances, labels, save_dir):
    # 使用 MinMaxScaler 进行归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    all_dists = []
    all_conf = []
    all_labels = []
    all_colors = []  # 用于存储每个样本的颜色（绿色表示正确，橙色表示错误）
    
    num_classes = len(np.unique(labels))  # 获取类别数
    
    for class_id in range(num_classes):
        # 获取当前类别的样本
        class_mask = (labels == class_id)
        if np.sum(np.array(class_mask)) > 0:
            # 当前类别的置信度和距离
            class_conf = mean_conf[class_mask]  # 置信度
            class_dists = distances[class_mask]  # 距离
            class_correct = is_correct_array[class_mask]  # 是否正确分类

            # 对当前类别的距离进行归一化
            # class_dists_normalized = scaler.fit_transform(class_dists.reshape(-1, 1)).flatten()  # 归一化到 [0, 1] 范围

            # 将当前类别的数据添加到合并的数据列表中
            all_dists.extend(class_dists)
            all_conf.extend(class_conf)
            all_labels.extend(class_correct)

            # 根据分类结果设置颜色：绿色表示正确，橙色表示错误
            all_colors.extend(['green' if correct else 'orange' for correct in class_correct])
        else:
            pass

    # 转换为 NumPy 数组
    all_dists = np.array(all_dists)
    all_conf = np.array(all_conf)
    all_labels = np.array(all_labels)
    all_colors = np.array(all_colors)

    # 绘制散点图
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # 根据颜色绘制散点图：绿色表示正确，橙色表示错误
    scatter = ax.scatter(all_dists, all_conf, c=all_colors, alpha=0.7, s=30000/all_dists.shape[0])

    # 设置标题和坐标轴标签
    ax.set_title("Normalized Distance vs Confidence (Correct vs Incorrect)", fontsize=14)
    ax.set_xlabel('Normalized Distance')
    ax.set_ylabel('Confidence')

    # 设置图例
    ax.scatter([], [], color='green', label='Correct')  # 添加绿色标签
    ax.scatter([], [], color='orange', label='Incorrect')  # 添加橙色标签
    ax.legend(loc='best')

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_dir}/normalized_distance_vs_confidence.png", dpi=300)

def plot_accuracy_conf_variance(is_correct_array, mean_conf, var_conf, labels, sample_proto_distances, save_dir):
    ################ 对于softmax距离 #####################
    all_conf = mean_conf
    all_var = var_conf
    all_dists = sample_proto_distances
    all_labels =  is_correct_array
    # 使用 MinMaxScaler 来进行归一化
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # all_dists = []
    # all_conf = []
    # all_var = []
    # all_labels = []
    # num_classes = len(np.unique(labels))  # 类别数
    # for class_id in range(num_classes):
    #     # 获取当前类别的样本
    #     class_mask = (labels == class_id)

    #     # 当前类别的方差、置信度和距离
    #     class_dists = sample_proto_distances[class_mask]
    #     class_conf = mean_conf[class_mask]  # 置信度
    #     class_var = var_conf[class_mask]    # 方差
    #     class_correct = is_correct_array[class_mask]
    #     # 对当前类别的距离进行归一化
    #     class_dists_normalized = scaler.fit_transform(class_dists.reshape(-1, 1)).flatten()  # 归一化到 [0, 1] 范围

    #     # 将当前类别的数据添加到合并的数据列表中
    #     all_dists.extend(class_dists_normalized)
    #     all_conf.extend(class_conf)
    #     all_var.extend(class_var)
    #     all_labels.extend(class_correct)
    
    # all_dists = np.array(all_dists)
    # all_conf = np.array(all_conf)
    # all_var = np.array(all_var)
    # all_labels = np.array(all_labels)
    all_colors = ['green' if correct else 'orange' for correct in all_labels]  # 分类正确为绿色，错误为橙色
    # 绘制散点图
    fig, ax = plt.subplots(figsize=(6.5, 6))

    # 根据颜色绘制散点图：绿色表示正确，橙色表示错误
    scatter = ax.scatter(all_var, all_conf, c=all_colors, alpha=0.7, s = 30000/all_labels.shape[0])

    # 设置标题和坐标轴标签
    ax.set_title("Variance vs Confidence (Correct vs Incorrect)", fontsize=14)
    ax.set_xlabel('Variance')
    ax.set_ylabel('Confidence')

    # 设置图例
    ax.scatter([], [], color='green', label='Correct')  # 添加绿色标签
    ax.scatter([], [], color='orange', label='Incorrect')  # 添加橙色标签
    ax.legend(loc='best')

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_confidence_variance.png", dpi=300)
        # 创建一个新的图形，绘制正确样本和错误样本的分布图
    fig, axes = plt.subplots(1, 3, figsize=(18, 3))

    # 筛选出正确和错误样本
    correct_mask = (all_labels == True)
    incorrect_mask = (all_labels == False)
    
    # 绘制 Confidence 分布（正确样本 - 绿色，错误样本 - 橙色）
    axes[0].hist(all_conf[correct_mask], bins=50, density=False, color='green', alpha=0.7, label='Correct')
    axes[0].hist(all_conf[incorrect_mask], bins=50, density=False, color='orange', alpha=0.7, label='Incorrect')
    axes[0].set_title('Confidence Distribution', fontsize=12)
    axes[0].set_xlabel('Confidence')
    axes[0].set_ylabel('Density')
    axes[0].legend(loc='best')
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['top'].set_visible(False)

    # 绘制 Variance 分布（正确样本 - 绿色，错误样本 - 橙色）
    axes[1].hist(all_var[correct_mask], bins=50, density=False, color='green', alpha=0.7, label='Correct')
    axes[1].hist(all_var[incorrect_mask], bins=50, density=False, color='orange', alpha=0.7, label='Incorrect')
    axes[1].set_title('Variance Distribution', fontsize=12)
    axes[1].set_xlabel('Variance')
    axes[1].set_ylabel('Density')
    axes[1].legend(loc='best')
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['top'].set_visible(False)

    # 绘制 Distance 分布（正确样本 - 绿色，错误样本 - 橙色）
    axes[2].hist(all_dists[correct_mask], bins=50, density=False, color='green', alpha=0.7, label='Correct')
    axes[2].hist(all_dists[incorrect_mask], bins=50, density=False, color='orange', alpha=0.7, label='Incorrect')
    axes[2].set_title('Distance Distribution', fontsize=12)
    axes[2].set_xlabel('Distance')
    axes[2].set_ylabel('Density')
    axes[2].legend(loc='best')
    axes[2].spines['right'].set_visible(False)
    axes[2].spines['top'].set_visible(False)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correction_confidence_variance_distance.png", dpi=300)

def plot_correct_incorrect_marker_gene_bubble(adata_test, mean_conf, save_dir):
    celltype_marker_genes = {
        "CD14+ Mono": ['FCGR3A', 'MPO'],
        "CD16+ Mono": ['FCGR3A', 'MPO'],
        "CD4+ T activated": ['IL2RA', 'IFNG', 'CD28', 'CD44', 'CD69'],
        "CD4+ T naive": ['IL7R', 'LEF1', 'TCF7', 'CCR7', 'CD38'],
        "Erythroblast": ['HBA1', 'HBB', 'EPB42', 'SLC4A1'],
        "G/M prog": ['CSF3R', 'MPO'],
        "HSC": ['CD34', 'KIT', 'GATA2'],
        "ILC": ['NCR1', 'IL7R', 'GATA3', 'T-bet', 'KLRG1'],
        "Lymph prog": ['PAX5', 'CD7', 'IL7R', 'TCF7', 'IGLL1'],
        "MK/E prog": ['TBXAS1', 'ERG'],
        "NK": ['NKG7', 'GZMB', 'PRF1', 'CD8A', 'KLRG1', 'NCR1'],
        "Normoblast": ['HBA1', 'HBB', 'EPB42', 'SLC4A1'],
        "Proerythroblast": ['GATA1', 'HBA1', 'HBB', 'SLC4A1'],
        "Transitional B": ['PAX5', 'IGLL1'],
        "cDC2": ['ITGAX', 'CX3CR1', 'IRF4'],
        "pDC": ['IRF7'],
        "CD8+ T naive": ['CD8A', 'IL7R', 'PRF1', 'CD28', 'NCR1']
    }

    # 2. 提取字典中的所有基因并按顺序排列
    all_marker_genes = [gene for genes in celltype_marker_genes.values() for gene in genes]

    # 3. 确保提取的基因在adata_test.var["gene_name"]中
    valid_genes = list(dict.fromkeys([gene for gene in all_marker_genes if gene in adata_test.var["gene_name"].values]))
    
    # 如果没有任何基因匹配，可以中止代码或进行适当的处理
    if not valid_genes:
        raise ValueError("No marker genes found in the dataset!")
    
    # 4. 获取每个基因的索引位置
    gene_indices = [np.where(adata_test.var['gene_name'] == gene)[0][0] for gene in valid_genes]

    # 5. 使用'X_binned'获取表达数据
    if 'X_binned' in adata_test.layers:
        expression_data = adata_test.layers['X_binned'][:, gene_indices]
        print("Using 'X_binned' from layers.")
    else:
        expression_data = adata_test.X[:, gene_indices]
        print("Using 'X' from adata_test.")

    # 6. 将表达数据转换为 DataFrame，使用细胞类型作为行索引
    expression_df = pd.DataFrame(expression_data.toarray() if hasattr(expression_data, 'toarray') else expression_data,
                            columns=valid_genes,
                            index=adata_test.obs['celltype'])
    non_zero_proportion_df = (expression_df > 0).groupby('celltype').mean()
    # Step 1: Create masks for correct and misclassify cells based on mean_conf
    incorrect_cells = mean_conf < 0.2                          # 认为是标签模糊的样本
    correct_cells = mean_conf > 0.8
    ambigeous_cells = (mean_conf > 0.2) & (mean_conf < 0.8)

    expression_incorrect = expression_df[incorrect_cells]
    expression_correct = expression_df[correct_cells]
    expression_ambigeous = expression_df[ambigeous_cells]

    # 7. 计算每个细胞在每个标记基因上的平均表达量
    expression_avg_incorrect = expression_incorrect.groupby('celltype').mean()
    expression_avg_correct = expression_correct.groupby('celltype').mean()
    expression_avg_ambigeous = expression_ambigeous.groupby('celltype').mean()
    # 8. 计算每个细胞类型中每个基因的表达细胞比例
    non_zero_proportion_incorrect = (expression_incorrect > 0).groupby('celltype').mean()
    non_zero_proportion_correct = (expression_correct > 0).groupby('celltype').mean()
    non_zero_proportion_ambigeous = (expression_ambigeous > 0).groupby('celltype').mean()
    
        # Define cell types (17 in total)
    genes = expression_avg_ambigeous.columns
    celltypes = expression_avg_ambigeous.index

    # Create a grid for the bubble plot
    x = np.tile(genes, len(celltypes))  # gene names on x-axis
    y = np.repeat(celltypes, len(genes))  # cell types on y-axis
    size = (non_zero_proportion_ambigeous.values.flatten() * 1000)  # Bubble size, scaled by non-zero proportion
    color = expression_avg_ambigeous.values.flatten()  # Bubble color, based on expression values

    # Create the figure and axis
    plt.figure(figsize=(18, 8))

    # Plot bubbles
    scatter = plt.scatter(x, y, s=size, c=color, cmap='viridis', alpha=0.8)

    # Add color bar to represent expression levels
    plt.colorbar(scatter, label='Expression Level')
    plt.xticks(rotation=90, ha='right')
    # 6. Adjust layout and display
    plt.tight_layout()

    # 7. Save the figure
    plt.savefig(f"{save_dir}/Marker_Gene_Expression_ambigeous_Bubble.png", dpi=300)
    
    
    # Define cell types (17 in total)
    genes = expression_avg_incorrect.columns
    celltypes = expression_avg_incorrect.index

    # Create a grid for the bubble plot
    x = np.tile(genes, len(celltypes))  # gene names on x-axis
    y = np.repeat(celltypes, len(genes))  # cell types on y-axis
    size = (non_zero_proportion_incorrect.values.flatten() * 1000)  # Bubble size, scaled by non-zero proportion
    color = expression_avg_incorrect.values.flatten()  # Bubble color, based on expression values

    # Create the figure and axis
    plt.figure(figsize=(18, 8))

    # Plot bubbles
    scatter = plt.scatter(x, y, s=size, c=color, cmap='viridis', alpha=0.8)

    # Add color bar to represent expression levels
    plt.colorbar(scatter, label='Expression Level')
    plt.xticks(rotation=90, ha='right')
    # 6. Adjust layout and display
    plt.tight_layout()

    # 7. Save the figure
    plt.savefig(f"{save_dir}/Marker_Gene_Expression_incorrect_Bubble.png", dpi=300)
    
    genes = expression_avg_correct.columns
    celltypes = expression_avg_correct.index

    # Create a grid for the bubble plot
    x = np.tile(genes, len(celltypes))  # gene names on x-axis
    y = np.repeat(celltypes, len(genes))  # cell types on y-axis
    size = (non_zero_proportion_correct.values.flatten() * 1000)  # Bubble size, scaled by non-zero proportion
    color = expression_avg_correct.values.flatten()  # Bubble color, based on expression values

    # Create the figure and axis
    plt.figure(figsize=(18, 8))

    # Plot bubbles
    scatter = plt.scatter(x, y, s=size, c=color, cmap='viridis', alpha=0.8)

    # Add color bar to represent expression levels
    plt.colorbar(scatter, label='Expression Level')
    plt.xticks(rotation=90, ha='right')
    # 6. Adjust layout and display
    plt.tight_layout()

    # 7. Save the figure
    plt.savefig(f"{save_dir}/Marker_Gene_Expression_correct_Bubble.png", dpi=300)
    
    #################### CORRECT-INCORRECT ###########################
    x = np.tile(genes, len(celltypes))  # gene names on x-axis
    y = np.repeat(celltypes, len(genes))  # cell types on y-axis
    non_zero_proportion_diff = non_zero_proportion_correct - non_zero_proportion_incorrect
    expression_avg_diff = expression_avg_correct - expression_avg_incorrect
    size = (non_zero_proportion_diff.values.flatten() * 1000)  # Bubble size, scaled by non-zero proportion
    color = expression_avg_diff.values.flatten()  # Bubble color, based on expression values

    # Create the figure and axis
    plt.figure(figsize=(18, 8))

    # Plot bubbles
    scatter = plt.scatter(x, y, s=size, c=color, cmap='coolwarm')

    # Add color bar to represent expression levels
    plt.colorbar(scatter, label='Expression Level')
    plt.xticks(rotation=90, ha='right')
    # 6. Adjust layout and display
    plt.tight_layout()

    # 7. Save the figure
    plt.savefig(f"{save_dir}/Marker_Gene_Expression_diff_Bubble.png", dpi=300)
    
def convert_csv_to_dict(csv_file):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 创建一个字典来存储每个细胞类型及其标志基因
    celltype_marker_genes = {}

    # 遍历每一行数据
    for _, row in df.iterrows():
        cell_type = row['label']  # 获取细胞类型
        gene = row['genes']  # 获取基因名称
        
        # 如果该细胞类型在字典中不存在，则初始化为空列表
        if cell_type not in celltype_marker_genes:
            celltype_marker_genes[cell_type] = []
        
        # 将基因添加到对应细胞类型的标志基因列表中
        celltype_marker_genes[cell_type].append(gene)

    return celltype_marker_genes

# def count_total_marker_genes(celltype_marker_genes):
#     total_genes = 0
    
#     # 遍历字典中的所有基因列表，将每个列表的基因数加总
#     for genes in celltype_marker_genes.values():
#         total_genes += len(genes)  # 将每个细胞类型的标志基因数加到总数中
    
#     return total_genes
def compute_mann_whitney_for_all_celltypes(marker_gene, expression_correct, expression_incorrect, celltypes):
    from scipy.stats import mannwhitneyu
    p_values = []
    for cell_type in celltypes:
        # Extract expression values for this cell type and marker gene
        correct_values = expression_correct.loc[cell_type, marker_gene].values.flatten()
        incorrect_values = expression_incorrect.loc[cell_type, marker_gene].values.flatten()
        
        # Perform Mann-Whitney U test between correct and incorrect expression
        stat, p_value = mannwhitneyu(correct_values, incorrect_values, alternative='two-sided')
        p_values.append(p_value)
    
    return p_values

def count_marker_genes_per_celltype(celltype_marker_genes):
    """
    统计每个细胞类型的标志基因数量。

    Parameters:
    - celltype_marker_genes: 字典，包含细胞类型及其标志基因列表

    Returns:
    - gene_counts: 字典，包含每个细胞类型的标志基因数量
    """
    gene_counts = {}
    
    # 遍历每个细胞类型及其标志基因
    for cell_type, genes in celltype_marker_genes.items():
        gene_counts[cell_type] = len(genes)  # 计算每个细胞类型的标志基因数
    
    return gene_counts

def plot_volin(adata, save_dir, cell_type, gene_names, save_name = "critical_genes_10_violint"):
    # import matplotlib as mpl
    # mpl.rcParams['lines.linewidth'] = 1
    # mpl.rcParams['patch.linewidth'] = 1
    # mpl.rcParams['axes.linewidth'] = 1
    cell_type_dir = os.path.join(save_dir, cell_type)

    adata.var_names = adata.var_names.to_series().str.replace(r'^(a_a_)', 'a_', regex=True)
    gene_names = [g.replace('a_a_', 'a_') for g in gene_names]
    # 3. 用 scaled 版本绘图
    # plt.figure(figsize=(60, 10))
    sc.pl.stacked_violin(adata, var_names=gene_names, \
                         groupby='correction',
                        #  figsize=(8, 2),
                         show=False
                        )
    for ax in plt.gcf().axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    # plt.tight_layout()
    plt.savefig(f"{cell_type_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
    
def plot_dist_correlation_marker_gene_eachcelltype_violin(mean_conf, cell_type_map, sample_proto_distances, save_dir):
    # from gsea.compute_gsea import plot_volin 
    new_adata = sc.read_h5ad(f"{save_dir}/correction_adata.h5ad")
    csv_file = f"{save_dir}/dist_gene_correlation_top10_norm.csv"  # 替换成你的文件路径
    # celltype_marker_genes = convert_csv_to_dict(csv_file)
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='correlation', ascending=False)
    # 3. 构建字典：每个 label 对应一个排序后的基因列表
    celltype_marker_genes = (
        df_sorted.groupby('label')['genes']
        .apply(list)
        .to_dict()
    )
    # 调用函数并打印结果
    # gene_counts = count_marker_genes_per_celltype(celltype_marker_genes)
    # fig_size_factor = 18 / 33  # 你可以调整这个比例来获得合适的图形大小
    # fig_sizes = {cell_type: (gene_counts[cell_type] * fig_size_factor + 3) for cell_type in gene_counts}
    
    for cell_type in list(celltype_marker_genes.keys()):
        gene_names = celltype_marker_genes[cell_type]
        cell_type_adata = new_adata[new_adata.obs["celltype"] == cell_type, gene_names]
        plot_volin(cell_type_adata, save_dir, cell_type, gene_names, save_name = "dist_correlation_genes_violin")
        
def plot_dist_correlation_marker_gene_eachcelltype_bubble(adata_test, mean_conf, sample_proto_distances, save_dir):
        # 调用函数并打印结果
    # csv_file = f"{save_dir}/dist_gene_correlation_top10_norm.csv"  # 替换成你的文件路径
    # celltype_marker_genes = convert_csv_to_dict(csv_file)
    # print(celltype_marker_genes)

    csv_file = f"{save_dir}/dist_gene_correlation_top10_norm.csv"  # 替换成你的文件路径
    # celltype_marker_genes = convert_csv_to_dict(csv_file)
    df = pd.read_csv(csv_file)
    df_sorted = df.sort_values(by='correlation', ascending=False)
    # 3. 构建字典：每个 label 对应一个排序后的基因列表
    celltype_marker_genes = (
        df_sorted.groupby('label')['genes']
        .apply(list)
        .to_dict()
    )
    # 调用函数并打印结果
    gene_counts = count_marker_genes_per_celltype(celltype_marker_genes)
    fig_size_factor = 18 / 33  # 你可以调整这个比例来获得合适的图形大小
    fig_sizes = {cell_type: (gene_counts[cell_type] * fig_size_factor + 3) for cell_type in gene_counts}
    # 2. 提取字典中的所有基因并按顺序排列
    all_marker_genes = [gene for genes in celltype_marker_genes.values() for gene in genes]

    # 3. 确保提取的基因在adata_test.var["gene_name"]中
    valid_genes = list(dict.fromkeys([gene for gene in all_marker_genes if gene in adata_test.var["gene_name"].values]))
    
    # 如果没有任何基因匹配，可以中止代码或进行适当的处理
    if not valid_genes:
        raise ValueError("No marker genes found in the dataset!")
    
    # 4. 获取每个基因的索引位置
    gene_indices = [np.where(adata_test.var['gene_name'] == gene)[0][0] for gene in valid_genes]

    # 5. 使用'X_binned'获取表达数据
    if 'X_binned' in adata_test.layers:
        expression_data = adata_test.layers['X_binned'][:, gene_indices]
        print("Using 'X_binned' from layers.")
    else:
        expression_data = adata_test.X[:, gene_indices]
        print("Using 'X' from adata_test.")

    # 6. 将表达数据转换为 DataFrame，使用细胞类型作为行索引
    expression_df = pd.DataFrame(expression_data.toarray() if hasattr(expression_data, 'toarray') else expression_data,
                            columns=valid_genes,
                            index=adata_test.obs['celltype'])
       # 计算每个细胞在每个标记基因上的平均表达量
    expression_avg = expression_df.groupby('celltype').mean()            # (17, 133)
    # Step 1: Create masks for correct and misclassify cells based on mean_conf
    if sample_proto_distances is not None:
        incorrect_cells = (mean_conf < 0.2) & (sample_proto_distances < 0.4)
        correct_cells = (mean_conf > 0.8) & (sample_proto_distances > 0.8)
        ambigeous_cells = (mean_conf > 0.2) & (mean_conf < 0.8) & (sample_proto_distances > 0.4) & (sample_proto_distances < 0.8)
    else:
        incorrect_cells = mean_conf < 0.2                          # 认为是标签模糊的样本
        correct_cells = mean_conf > 0.8
        ambigeous_cells = (mean_conf > 0.2) & (mean_conf < 0.8)

    expression_incorrect = expression_df[incorrect_cells]
    expression_correct = expression_df[correct_cells]
    expression_ambigeous = expression_df[ambigeous_cells]

    # 7. 计算每个细胞在每个标记基因上的平均表达量
    expression_avg_incorrect = expression_incorrect.groupby('celltype').mean()         # 17, 133
    expression_avg_correct = expression_correct.groupby('celltype').mean()
    expression_avg_ambigeous = expression_ambigeous.groupby('celltype').mean()
    
    # 8. 计算每个细胞类型中每个基因的表达细胞比例
    non_zero_proportion_incorrect = (expression_incorrect > 0).groupby('celltype').mean()  # 17, 133
    non_zero_proportion_correct = (expression_correct > 0).groupby('celltype').mean()
    non_zero_proportion_ambigeous = (expression_ambigeous > 0).groupby('celltype').mean()
    
    for cell_type in expression_avg.index:
        # 为当前细胞类型创建文件夹
        cell_type_dir = os.path.join(save_dir, cell_type)
        if not os.path.exists(cell_type_dir):
            os.makedirs(cell_type_dir)

        # 获取当前细胞类型的标志基因（marker genes）
        marker_genes = celltype_marker_genes.get(cell_type, [])
        
        if not marker_genes:
            print(f"No marker genes found for cell type: {cell_type}")
            continue
        
        # 过滤表达数据，只选择当前细胞类型的标志基因
        # valid_marker_gene_indices = [np.where(adata_test.var['gene_name'] == gene)[0][0] for gene in marker_genes if gene in adata_test.var['gene_name'].values]
        valid_marker_genes = [gene for gene in marker_genes if gene in expression_df.columns]
    
        #################### Unified Color for Both Correct and Incorrect ###########################
        # Bubble size based on the non-zero proportion of gene expression (using the existing matrix)
        size_correct = non_zero_proportion_correct.loc[cell_type, valid_marker_genes] * 1000  # Size based on non-zero proportion (scaled)
        size_incorrect = non_zero_proportion_incorrect.loc[cell_type, valid_marker_genes] * 1000  # Same for incorrect cells

        # Combine expression data for both correct and incorrect cells for unified color scale
        color_combined = np.concatenate([expression_avg_incorrect.loc[cell_type, valid_marker_genes].values.flatten(),
                                         expression_avg_correct.loc[cell_type, valid_marker_genes].values.flatten()])

        # Normalize the color values (using Min-Max scaling)
        scaler = MinMaxScaler()
        color_combined_normalized = scaler.fit_transform(color_combined.reshape(-1, 1)).flatten()
        size_combined = np.concatenate([size_incorrect.values.flatten(), size_correct.values.flatten()])

        # 对大小进行最大最小归一化
        size_combined_normalized = scaler.fit_transform(size_combined.reshape(-1, 1)).flatten()

        # Create the figure for overlaying both correct and incorrect results
        valid_marker_genes = [gene.replace("a_a_", "a_") for gene in valid_marker_genes]
        x = np.tile(valid_marker_genes, 2)  # Gene names on x-axis
        y = np.repeat(['Incorrect', 'Correct'], len(valid_marker_genes))  # Cell types on y-axis

        x_correct = x[len(valid_marker_genes):]
        x_incorrect = x[:len(valid_marker_genes)]
        # y_correct = np.ones_like(len(valid_marker_genes)*[0]) * 1  # 将Correct类别设置在y=1处
        # y_incorrect = np.ones_like(len(valid_marker_genes)*[0]) * 2 
        y_correct = y[len(valid_marker_genes):] 
        y_incorrect = y[:len(valid_marker_genes)]
        # Create a figure to overlay correct and incorrect bubbles
        plt.figure(figsize=(fig_sizes[cell_type], 3))

        # Plot correct cells with unified color scheme (scaled to the same color scale)
        # scatter_correct = plt.scatter(x_correct, y_correct, s=size_correct.values.flatten(), c=color_combined_normalized[:len(size_correct)], cmap='coolwarm', alpha=0.8, edgecolors='w', label="Correct")
        
        # Plot incorrect cells with unified color scheme (scaled to the same color scale)
        scatter_incorrect = plt.scatter(x_incorrect, y_incorrect, s=size_combined_normalized[:len(size_correct)] * 1000, c=color_combined_normalized[:len(size_correct)], cmap='coolwarm', alpha=0.8, edgecolors='w', label="Incorrect")
        scatter_correct = plt.scatter(x_correct, y_correct, s=size_combined_normalized[len(size_correct):] * 1000, c=color_combined_normalized[len(size_correct):], cmap='coolwarm', alpha=0.8, edgecolors='w', label="Correct")
        # Add a single colorbar (use the same color scale for both)
        plt.colorbar(scatter_correct, label='Expression Level')

        # Add legend
        # plt.legend()
        # 调整图像的显示范围，确保气泡不超出图像范围
        # plt.xlim(min(x) - 1, max(x) + 1)  # 根据实际数据调整范围
        plt.ylim(-1, 2)  # y轴限制为0到3，确保"Correct"和"Incorrect"气泡不会重叠
        # Set plot labels
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()

        # Save the figure for the current cell type
        plt.savefig(f"{cell_type_dir}/Marker_Gene_Expression_Correct_vs_Incorrect_conf&dist.png", dpi=300)
    
def plot_dist_correlation_marker_gene_bubble(adata_test, mean_conf, save_dir):
    
    # 调用函数并打印结果
    csv_file = f"{save_dir}/dist_gene_correlation_top10_norm.csv"  # 替换成你的文件路径
    celltype_marker_genes = convert_csv_to_dict(csv_file)
    print(celltype_marker_genes)


    # 调用函数并打印结果
    gene_counts = count_marker_genes_per_celltype(celltype_marker_genes)
    fig_size_factor = 18 / 33  # 你可以调整这个比例来获得合适的图形大小
    fig_sizes = {cell_type: (gene_counts[cell_type] * fig_size_factor + 5) for cell_type in gene_counts}
    # 2. 提取字典中的所有基因并按顺序排列
    all_marker_genes = [gene for genes in celltype_marker_genes.values() for gene in genes]

    # 3. 确保提取的基因在adata_test.var["gene_name"]中
    valid_genes = list(dict.fromkeys([gene for gene in all_marker_genes if gene in adata_test.var["gene_name"].values]))
    
    # 如果没有任何基因匹配，可以中止代码或进行适当的处理
    if not valid_genes:
        raise ValueError("No marker genes found in the dataset!")
    
    # 4. 获取每个基因的索引位置
    gene_indices = [np.where(adata_test.var['gene_name'] == gene)[0][0] for gene in valid_genes]

    # 5. 使用'X_binned'获取表达数据
    if 'X_binned' in adata_test.layers:
        expression_data = adata_test.layers['X_binned'][:, gene_indices]
        print("Using 'X_binned' from layers.")
    else:
        expression_data = adata_test.X[:, gene_indices]
        print("Using 'X' from adata_test.")

    # 6. 将表达数据转换为 DataFrame，使用细胞类型作为行索引
    expression_df = pd.DataFrame(expression_data.toarray() if hasattr(expression_data, 'toarray') else expression_data,
                            columns=valid_genes,
                            index=adata_test.obs['celltype'])
       # 计算每个细胞在每个标记基因上的平均表达量
    expression_avg = expression_df.groupby('celltype').mean()            # (17, 133)
    # Step 1: Create masks for correct and misclassify cells based on mean_conf
    incorrect_cells = mean_conf < 0.2                          # 认为是标签模糊的样本
    correct_cells = mean_conf > 0.8
    ambigeous_cells = (mean_conf > 0.2) & (mean_conf < 0.8)

    expression_incorrect = expression_df[incorrect_cells]
    expression_correct = expression_df[correct_cells]
    expression_ambigeous = expression_df[ambigeous_cells]

    # 7. 计算每个细胞在每个标记基因上的平均表达量
    expression_avg_incorrect = expression_incorrect.groupby('celltype').mean()
    expression_avg_correct = expression_correct.groupby('celltype').mean()
    expression_avg_ambigeous = expression_ambigeous.groupby('celltype').mean()
    
    # num_expression_correct = expression_correct.groupby('celltype')
    # num_expression_incorrect = expression_incorrect.groupby('celltype')
    # print('num_expression_correct.shape', num_expression_correct.shape)
    # print('num_expression_incorrect.shape', num_expression_incorrect.shape)
    # 8. 计算每个细胞类型中每个基因的表达细胞比例
    non_zero_proportion_incorrect = (expression_incorrect > 0).groupby('celltype').mean()
    non_zero_proportion_correct = (expression_correct > 0).groupby('celltype').mean()
    non_zero_proportion_ambigeous = (expression_ambigeous > 0).groupby('celltype').mean()
    
    # 绘制显著性差异图
    # The code `plot_MWU_test(cell_type, celltype_of_markers)` appears to be a function call with two
    # arguments `cell_type` and `celltype_of_markers`. It is likely a function that performs a
    # Mann-Whitney U test and plots the results for the specified cell types and markers.
    # plot_MWU_test(cell_type, celltype_of_markers)
    for cell_type in expression_avg.index:
        # 为当前细胞类型创建文件夹
        cell_type_dir = os.path.join(save_dir, cell_type)
        if not os.path.exists(cell_type_dir):
            os.makedirs(cell_type_dir)

        # 获取当前细胞类型的标志基因（marker genes）
        marker_genes = celltype_marker_genes.get(cell_type, [])
        
        if not marker_genes:
            print(f"No marker genes found for cell type: {cell_type}")
            continue
        
        # 过滤表达数据，只选择当前细胞类型的标志基因
        # valid_marker_gene_indices = [np.where(adata_test.var['gene_name'] == gene)[0][0] for gene in marker_genes if gene in adata_test.var['gene_name'].values]
        valid_marker_genes = [gene for gene in marker_genes if gene in expression_df.columns]
        
        #################### Correct ###########################
        # # Plot for the correct classification
        size_correct = (non_zero_proportion_correct.loc[:, valid_marker_genes].values.flatten() * 1000)  # Bubble size, scaled by non-zero proportion
        color_correct = expression_avg_correct.loc[:, valid_marker_genes].values.flatten()  # Bubble color, based on expression values

        # Create the figure and axis for the correct plot
        celltypes = expression_avg.index

        # Compute Mann-Whitney U test p-values for the correct and incorrect expression values
        
        #################### Mann-Whitney U Test for all celltypes ######################
        p_values_for_all_celltypes = []  # Store p-values for all celltypes and marker genes
        
        for marker_gene in valid_marker_genes:
            # Perform the Mann-Whitney U test for all cell types for the current marker gene
            p_values = compute_mann_whitney_for_all_celltypes(marker_gene, expression_correct, expression_incorrect, celltypes)
            p_values_for_all_celltypes.append(p_values)

        # Flatten the list of p-values for plotting
        p_values_for_all_celltypes = np.array(p_values_for_all_celltypes)           # 10, 17
        
        ##################################
        x_labels = valid_marker_genes  # marker genes
        y_labels = celltypes  # cell types

        # Create a figure for the heatmap
        # plt.figure(figsize=(fig_sizes[cell_type], 8))

        # # Plot the heatmap using p_values_for_all_celltypes
        # # We use `cmap='coolwarm'` for color mapping, but you can choose other colormaps (e.g., 'viridis', 'plasma', etc.)
        # heatmap = plt.imshow(p_values_for_all_celltypes.T, aspect='auto', cmap='coolwarm', interpolation='nearest')

        # # Add color bar to represent p-value scale
        # plt.colorbar(heatmap, label='P-value (Correct vs Incorrect)')

        # # Set x and y ticks to show marker genes and cell types
        # plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90, ha='right')
        # plt.yticks(np.arange(len(y_labels)), y_labels)

        # # Add axis labels
        # plt.xlabel('Marker Genes')
        # plt.ylabel('Cell Types')

        # # Adjust the layout for better readability
        # plt.tight_layout()

        # # Save the heatmap to a file
        # plt.savefig(f"{cell_type_dir}/Marker_Gene_Expression_heatmap_pvalues.png", dpi=300)
        ############### 气泡图 ###################

        # x = np.tile(valid_marker_genes, len(celltypes))  # gene names on x-axis
        # y = np.repeat(celltypes, len(valid_marker_genes))  # cell types on y-axis
        # plt.figure(figsize=(fig_sizes[cell_type], 8))
        # scatter_correct = plt.scatter(x, y, s=size_correct, c=color_correct, cmap='viridis', alpha=0.8)
        # plt.colorbar(scatter_correct, label='Expression Level (Correct)')
        # plt.xticks(rotation=90, ha='right')
        # plt.tight_layout()

        # # Save the figure for this cell type (correct)
        # plt.savefig(f"{cell_type_dir}/Marker_Gene_Expression_correct_Bubble_dist_corr.png", dpi=300)

        # #################### Incorrect ###########################
        # Plot for the incorrect classification
        size_incorrect = (non_zero_proportion_incorrect.loc[:, valid_marker_genes].values.flatten() * 1000)  # Bubble size, scaled by non-zero proportion
        color_incorrect = expression_avg_incorrect.loc[:, valid_marker_genes].values.flatten()  # Bubble color, based on expression values

        # Create the figure and axis for the incorrect plot
        plt.figure(figsize=(fig_sizes[cell_type], 8))
        scatter_incorrect = plt.scatter(x, y, s=size_incorrect, c=color_incorrect, cmap='viridis', alpha=0.8)
        plt.colorbar(scatter_incorrect, label='Expression Level (Incorrect)')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()

        # Save the figure for this cell type (incorrect)
        plt.savefig(f"{cell_type_dir}/Marker_Gene_Expression_incorrect_Bubble_dist_corr.png", dpi=300)
        
def plot_marker_gene_bubble(adata_test, save_dir):
    # marker_genes = [
    #     "CD14", "CCR2", "CSF1R", "FCGR3A", "CD16", "CD68", "CD11b", "MPO", "TLR2",  # CD14+ Mono
    #     "CD16", "FCGR3A", "CCR2", "CSF1R", "MPO", "CD14", "CD11b", "CX3CR1",  # CD16+ Mono
    #     "CD4", "IL2RA", "TNF", "IFNG", "CD69", "CD25", "CD44", "CD28", "CXCR3",  # CD4+ T activated
    #     "CD4", "IL7R", "LEF1", "TCF7", "CD45RA", "CD62L", "CCR7", "CD28",  # CD4+ T naive
    #     "HBB", "HBA1", "GATA1", "SLC4A1", "EPO", "KLF1", "SCL", "AHSP",  # Erythroblast
    #     "CSF3R", "CD33", "CD14", "MPO", "CD11b", "CD16", "CXCR1", "ITGB2", "CD115",  # G/M prog
    #     "CD34", "KIT", "THY1", "SCF", "CD90", "CD45RA", "CD38", "CD133", "GATA2",  # HSC
    #     "NCR1", "IL7R", "GATA3", "T-bet", "RORC", "KLRG1", "CD56", "CD161",  # ILC
    #     "CD7", "CD19", "IL7R", "PAX5", "NOTCH1", "CD10", "CD22", "BTK",  # Lymph prog
    #     "CD41", "CD61", "GATA1", "TBXAS1", "CD34", "ITGA2B", "GFI1", "MPL", "ERG",  # MK/E prog
    #     "NKG7", "KLRG1", "FCGR3A", "CD56", "NCR1", "GZMB", "CD16", "PRF1", "IL2RB",  # NK
    #     "HBA1", "HBB", "SLC4A1", "GATA1", "KLF1", "EPB42", "AHSP", "PAX5",  # Normoblast
    #     "GATA1", "HBA1", "HBB", "SLC4A1", "KLF1", "PAX5", "SP1", "EPO",  # Proerythroblast
    #     "CD19", "CD20", "CD22", "BAFFR", "PAX5", "BTK", "CD10", "IGLL1", "CD24",  # Transitional B
    #     "CD11c", "CLEC10A", "CX3CR1", "FSCN1", "ITGAX", "TLR7", "TLR9", "IRF4", "MHCII",  # cDC2
    #     "CD123", "BDCA2", "IRF7", "SIGLEC6", "TLR7", "MHCII", "IFNAR1", "CD4",  # pDC
    #     "CD8A", "CD45RA", "CCR7", "IL7R", "CD62L", "CD28", "SELL", "ITGA4", "CD3E"  # CD8+ T naive
    # ]

    # 1. 查找数据集中的基因名是否包含标记基因
    # existing_genes = list(set(marker_genes) & set(adata_test.var["gene_name"]))
    celltype_marker_genes = {
        "CD14+ Mono": ['FCGR3A', 'MPO'],
        "CD16+ Mono": ['FCGR3A', 'MPO'],
        "CD4+ T activated": ['IL2RA', 'IFNG', 'CD28', 'CD44', 'CD69'],
        "CD4+ T naive": ['IL7R', 'LEF1', 'TCF7', 'CCR7', 'CD38'],
        "Erythroblast": ['HBA1', 'HBB', 'EPB42', 'SLC4A1'],
        "G/M prog": ['CSF3R', 'MPO'],
        "HSC": ['CD34', 'KIT', 'GATA2'],
        "ILC": ['NCR1', 'IL7R', 'GATA3', 'T-bet', 'KLRG1'],
        "Lymph prog": ['PAX5', 'CD7', 'IL7R', 'TCF7', 'IGLL1'],
        "MK/E prog": ['TBXAS1', 'ERG'],
        "NK": ['NKG7', 'GZMB', 'PRF1', 'CD8A', 'KLRG1', 'NCR1'],
        "Normoblast": ['HBA1', 'HBB', 'EPB42', 'SLC4A1'],
        "Proerythroblast": ['GATA1', 'HBA1', 'HBB', 'SLC4A1'],
        "Transitional B": ['PAX5', 'IGLL1'],
        "cDC2": ['ITGAX', 'CX3CR1', 'IRF4'],
        "pDC": ['IRF7'],
        "CD8+ T naive": ['CD8A', 'IL7R', 'PRF1', 'CD28', 'NCR1']
    }

    # 2. 提取字典中的所有基因并按顺序排列
    all_marker_genes = [gene for genes in celltype_marker_genes.values() for gene in genes]

    # 3. 确保提取的基因在adata_test.var["gene_name"]中
    valid_genes = list(dict.fromkeys([gene for gene in all_marker_genes if gene in adata_test.var["gene_name"].values]))
    
    # 如果没有任何基因匹配，可以中止代码或进行适当的处理
    if not valid_genes:
        raise ValueError("No marker genes found in the dataset!")
    
    # 4. 获取每个基因的索引位置
    gene_indices = [np.where(adata_test.var['gene_name'] == gene)[0][0] for gene in valid_genes]

    # 5. 使用'X_binned'获取表达数据
    if 'X_binned' in adata_test.layers:
        expression_data = adata_test.layers['X_binned'][:, gene_indices]
        print("Using 'X_binned' from layers.")
    else:
        expression_data = adata_test.X[:, gene_indices]
        print("Using 'X' from adata_test.")

    # 6. 将表达数据转换为 DataFrame，使用细胞类型作为行索引
    expression_df = pd.DataFrame(expression_data.toarray() if hasattr(expression_data, 'toarray') else expression_data,
                             columns=valid_genes,
                             index=adata_test.obs['celltype'])

    # 7. 计算每个细胞在每个标记基因上的平均表达量
    expression_avg_df = expression_df.groupby('celltype').mean()


    # 8. 计算每个细胞类型中每个基因的表达细胞比例
    non_zero_proportion_df = (expression_df > 0).groupby('celltype').mean()

    genes = expression_avg_df.columns
    celltypes = expression_avg_df.index

    # Create a grid for the bubble plot
    x = np.tile(genes, len(celltypes))  # gene names on x-axis
    y = np.repeat(celltypes, len(genes))  # cell types on y-axis
    size = (non_zero_proportion_df.values.flatten() * 1000)  # Bubble size, scaled by non-zero proportion
    color = expression_avg_df.values.flatten()  # Bubble color, based on expression values

    # Create the figure and axis
    plt.figure(figsize=(18, 8))

    # Plot bubbles
    scatter = plt.scatter(x, y, s=size, c=color, cmap='viridis', alpha=0.8)

    # Add color bar to represent expression levels
    plt.colorbar(scatter, label='Expression Level')

    # Set labels and title
    plt.xlabel('Genes')
    plt.ylabel('Cell Types')
    plt.title('Bubble Chart of Gene Expression by Cell Type')

    # Adjust the labels for better visibility
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/Marker_Gene_Expression_Bubble.png", dpi=300)


def compute_rank_genes(adata_test, save_dir, mean_conf, cell_type_map, sample_proto_distances=None):

    # Step 1: Create masks for correct and misclassify cells based on mean_conf
    if sample_proto_distances is not None:
        incorrect_cells = (mean_conf < 0.2) & (sample_proto_distances < 0.4)
        correct_cells = (mean_conf > 0.8) & (sample_proto_distances > 0.8)
        ambigeous_cells = (mean_conf > 0.2) & (mean_conf < 0.8) & (sample_proto_distances > 0.4) & (sample_proto_distances < 0.8)
    else:
        incorrect_cells = mean_conf < 0.2                          # 认为是标签模糊的样本
        correct_cells = mean_conf > 0.8
        ambigeous_cells = (mean_conf > 0.2) & (mean_conf < 0.8)
    
    adata_test.obs.loc[incorrect_cells, 'correction'] = 'incorrect'
    adata_test.obs.loc[correct_cells, 'correction'] = 'correct'
    sc.pp.log1p(adata_test)
    # adata_test.X = adata_test.layers["X_binned"]
    # Step 3: Filter the data to include only incorrect and correct cells
    new_adata = adata_test[adata_test.obs['correction'].isin(['incorrect', 'correct']), :]
    new_adata.obs['correction_label'] = new_adata.obs['correction'].astype('category')
    new_adata.write_h5ad(f"{save_dir}/correction_adata.h5ad")
    
    for cell_type in cell_type_map.keys():
        cell_type_dir = os.path.join(save_dir, cell_type)
        if not os.path.exists(cell_type_dir):
            os.makedirs(cell_type_dir)
        cell_type_adata = new_adata[new_adata.obs['celltype'] == cell_type, :]
        if cell_type_adata.shape[0] != 0:
            sc.tl.rank_genes_groups(cell_type_adata, 'correction_label', method='wilcoxon', pts=True)   
            # Step 3: Extract the results from `uns['rank_genes_groups']`
            df_name = pd.DataFrame(cell_type_adata.uns['rank_genes_groups']['names'])
            df_lgfc = pd.DataFrame(cell_type_adata.uns['rank_genes_groups']['logfoldchanges'])
            df_pval = pd.DataFrame(cell_type_adata.uns['rank_genes_groups']['pvals_adj'])
            df_pts = pd.DataFrame(cell_type_adata.uns['rank_genes_groups']['pts'])

            # Step 4: Loop through the comparisons for each cell type
            for c in df_name.columns:
                # Create a DataFrame with gene names, log fold change, p-values, and PTS
                tmp_df = pd.DataFrame({
                    "gene_name": df_name[c],
                    "log_fc": df_lgfc[c],
                    'pvals_adj': df_pval[c],
                    'pts': df_pts.loc[df_name[c].values, c].values
                })
                
                # Save the full result as a CSV file
                tmp_df.to_csv(f"{cell_type_dir}/critical_genes_{c}_vs_others.csv")
                
                # Filter the genes based on your thresholds
                
                gene = tmp_df
                gene = gene[gene['log_fc'].abs() > 0.1]  # Absolute log fold change > 0.1
                gene = gene[gene['pvals_adj'] <= 0.05]   # Adjusted p-value <= 0.05
                gene = gene[gene['pts'] > 0.1]           # PTS > 0.2

                # Sort genes by log fold change (descending order)
                gene = gene.sort_values("log_fc", ascending=False)
                
                # Save upregulated genes (log_fc > 0)
                up = list(gene[gene['log_fc'] > 0]['gene_name'])
                up_score = list(gene[gene['log_fc'] > 0]['log_fc'])
                
                df_up = pd.DataFrame(columns=['gene', 'score'])
                df_up['gene'] = up
                df_up['score'] = up_score
                df_up.to_csv(f"{cell_type_dir}/critical_genes_{c}_up_sorted_gene.csv")

                # Save downregulated genes (log_fc < 0)
                down = list(gene[gene['log_fc'] < 0]['gene_name'])
                down_score = list(gene[gene['log_fc'] < 0]['log_fc'])
                
                df_down = pd.DataFrame(columns=['gene', 'score'])
                df_down['gene'] = down
                df_down['score'] = down_score
                df_down.to_csv(f"{cell_type_dir}/critical_genes_{c}_down_sorted_gene.csv")
    
def compute_dist_genes_correlation(sample_proto_distances, adata_test, labels, save_dir, cell_type_map, predictions=None):
    from scipy.stats import pearsonr
    """
    计算每个细胞类别的基因表达与距离之间的相关性，并可视化，结果保存为CSV文件。
    
    Parameters:
    - sample_proto_distances: 样本与每个原型的距离矩阵，形状为 (n_samples, n_prototypes)
    - adata_test: AnnData 对象，包含基因表达数据
    - labels: 每个细胞的真实类别标签
    - cell_type_map: 字典，映射每个真实标签到细胞类型的名称
    - predictions: （可选）模型预测的细胞类型标签
    - output_csv: （可选）保存相关性结果的CSV文件路径
    """
    
    # 1. 获取基因表达数据
    X = adata_test.layers["X_binned"]  # 假设基因表达存储在 AnnData 对象的 X 属性中
    gene_names = adata_test.var["gene_name"].values 
    # 2. 创建一个字典保存每个类别的基因表达与距离的相关性
    correlation_results = {}
    p_values_results = {}
    # 3. 遍历每个类别，计算基因表达量与距离的相关性
    for label in np.unique(labels):
        # 找出属于当前类别的细胞
        cell_indices = np.where(labels == label)[0]
        
        # 获取这些细胞的基因表达数据
        gene_expression = X[cell_indices]  # 属于该类别的所有细胞的基因表达
        distance_to_proto = sample_proto_distances[cell_indices]  # 属于该类别的细胞到原型的距离
        
        # 计算基因表达与距离之间的相关性（皮尔逊相关系数）
        correlations = []
        p_values = []
        for gene_index in range(gene_expression.shape[1]):
            valid_indices = ~np.isnan(gene_expression[:, gene_index]) & ~np.isnan(distance_to_proto)
            # print('NaN 值数目为', gene_expression[:, gene_index].shape[0] - np.sum(valid_indices))
            valid_gene_expression = gene_expression[valid_indices, gene_index]
            valid_distance_to_proto = distance_to_proto[valid_indices]
            # corr, p_value = pearsonr(valid_gene_expression.flatten(), valid_distance_to_proto.flatten())
            x = valid_gene_expression.flatten()
            y = valid_distance_to_proto.flatten()

            if x.size < 2 or y.size < 2:
                corr, p_value = np.nan, np.nan
            else:
                corr, p_value = pearsonr(x, y)
            # corr, p_value = pearsonr(valid_gene_expression[:, gene_index].flatten(), valid_distance_to_proto.flatten())
            correlations.append(corr)
            p_values.append(p_value)
        
                # 使用 cell_type_map 来获取实际的细胞类型名称
        cell_type_name = [key for key, value in cell_type_map.items() if value == label][0]
        # 保存每个类别的相关性结果
        correlation_results[cell_type_name] = correlations
        p_values_results[cell_type_name] = p_values
       
    # 4. 将结果转换为 DataFrame，方便查看和进一步分析
    correlation_df = pd.DataFrame(correlation_results, index=gene_names)
    p_values_df = pd.DataFrame(p_values_results, index=gene_names)
    # 5. 将相关性和p值结果合并到同一个 DataFrame
    merged_df = pd.concat([correlation_df, p_values_df], axis=1)
    merged_df.columns = [f"{col}_correlation" for col in correlation_df.columns] + [f"{col}_p_value" for col in p_values_df.columns]

    # 6. 保存结果为同一个 CSV 文件
    # if save_dir:
    #     merged_df.to_csv(f"{save_dir}/dist_gene_correlation_with_p_values_norm.csv", index=True)
    
    # 5. 为每个细胞类别筛选排名前10的基因，并格式化结果
    # top_genes_list = []
    
    # for label in correlation_results:
    #     # 获取每个类别的相关性并按相关性值排序（从大到小）
    #     sorted_correlations = sorted(zip(correlation_results[label], p_values_results[label], correlation_df.index), reverse=True)
        
    #     # 取前10个基因
    #     top_10_genes = sorted_correlations[:10]
        
    #     # 格式化每个类别的前10个基因
    #     for corr, p_value, gene in top_10_genes:
    #         if not np.isnan(corr):
    #             top_genes_list.append([label, gene, corr, p_value])
    
    # # 6. 将格式化后的结果保存为 DataFrame
    # top_genes_df = pd.DataFrame(top_genes_list, columns=['label', 'genes', 'correlation', 'p_value'])
    
    # # 7. 保存结果为CSV文件
    # top_genes_df.to_csv(f"{save_dir}/dist_gene_correlation_top10_norm.csv", index=False)
    
    all_genes_list = []  # 初始化一个空列表用于保存结果
    top_10_genes_list = []
    all_corr_gene_list = []
    # 遍历每个类别
    for label in correlation_results:
        cell_type_dir = f"{save_dir}/{label}"
        if not os.path.exists(cell_type_dir):
            os.makedirs(cell_type_dir) 
        sorted_genes = [] 
        # 获取每个类别的相关性和 p-value，并进行遍历
        for corr, p_value, gene in zip(correlation_results[label], p_values_results[label], correlation_df.index):
            # 只保存 p_value < 0.05 且相关性值不为 NaN 的结果
            if not np.isnan(corr) and p_value < 0.05:
                all_genes_list.append([label, gene, corr, p_value])
                sorted_genes.append([corr, p_value, gene, abs(corr)]) 
                if abs(corr) >= 0.2:
                    all_corr_gene_list.append([label, gene, corr, p_value])
        # sorted_genes = sorted(sorted_genes, reverse=True, key=lambda x: x[3])  # 根据相关性降序排序
        # top_10_genes = sorted_genes[:10]  # 获取前10个相关性最大的基因
        # 分离正相关性和负相关性基因
        positive_genes = [gene for gene in sorted_genes if gene[0] > 0]
        negative_genes = [gene for gene in sorted_genes if gene[0] < 0]
        
        # 根据相关性排序，分别取前5个正相关性和前5个负相关性最大的基因
        top_5_positive_genes = sorted(positive_genes, reverse=True, key=lambda x: x[0])[:5]
        top_5_negative_genes = sorted(negative_genes, reverse=False, key=lambda x: x[0])[:5]
        
        # 组合正负相关性最强的前10个基因
        top_10_genes = top_5_positive_genes + top_5_negative_genes
        for corr, p_value, gene, abs_corr in top_10_genes:
            top_10_genes_list.append([label, gene, corr, p_value])
    # 将格式化后的结果保存为 DataFrame
    significant_corr_gene_df = pd.DataFrame(all_corr_gene_list, columns=['label', 'genes', 'correlation', 'p_value'])
    top_genes_df = pd.DataFrame(all_genes_list, columns=['label', 'genes', 'correlation', 'p_value'])
    top_10_genes_df = pd.DataFrame(top_10_genes_list, columns=['label', 'genes', 'correlation', 'p_value'])

    significant_corr_gene_df.to_csv(f"{save_dir}/dist_gene_correlation_pvalue_0.05_corr_0.2.csv", index=False)
    top_genes_df.to_csv(f"{save_dir}/dist_gene_correlation_pvalue_0.05.csv", index=False)
        # 保存结果为CSV文件
    # top_genes_df.to_csv(f"{save_dir}/dist_gene_correlation_pvalue_0.05.csv", index=False)
    top_10_genes_df.to_csv(f"{save_dir}/dist_gene_correlation_top10_norm.csv", index=False)
    
def plot_celltype_subtypes(adata_test, save_dir, cell_type_map, distance_based = False):
    if distance_based:
        csv_file = f"{save_dir}/dist_gene_correlation_top10_norm.csv"  # 替换成你的文件路径
        # celltype_marker_genes = convert_csv_to_dict(csv_file)
        df = pd.read_csv(csv_file)
        df_sorted = df.sort_values(by='correlation', ascending=False)
        # 3. 构建字典：每个 label 对应一个排序后的基因列表
        celltype_marker_genes = (
            df_sorted.groupby('label')['genes']
            .apply(list)
            .to_dict()
        )
        
        # 2. 提取字典中的所有基因并按顺序排列
        all_marker_genes = [gene for genes in celltype_marker_genes.values() for gene in genes]

        # 3. 确保提取的基因在adata_test.var["gene_name"]中
        # valid_genes = list(dict.fromkeys([gene for gene in all_marker_genes if gene in adata_test.var["gene_name"].values]))
        
    X_umap = np.load(os.path.join(save_dir, "X_umap.npy"))
    X_cell = X_umap[:adata_test.X.shape[0]]
    for cell_type in cell_type_map.keys():
        if cell_type != "SF_like":
            cell_type_dir = os.path.join(save_dir, cell_type) 

            if distance_based:
                # gene_names = celltype_marker_genes[cell_type]
                gene_names = ["S100A8", "FCN1", "CTSS", "p_CD11b", "p_CD36", "p_CD44", "p_CD45", "FCGR3A", "a_a_LYN"]
                # gene_names = ["p_CD35", "S100A8", "FCN1", "p_CD36", "p_CD44", "FCGR3A", "a_a_SPIDR", "a_a_ETV6", "a_a_DGKG", "a_a_LYN"]
            
            else:
                correct_gsea_results = pd.read_csv(cell_type_dir + "/critical_genes_correct_up_sorted_gene.csv", sep = ',', index_col=0)
                incorrect_gsea_results = pd.read_csv(cell_type_dir + "/critical_genes_incorrect_up_sorted_gene.csv", sep = ',', index_col=0)
                gene_names = list(correct_gsea_results['gene'].head(5))+ list(incorrect_gsea_results['gene'].head(5))
            cell_type_adata = adata_test[adata_test.obs["celltype"] == cell_type, gene_names]
            
            print("adata_test obs_names unique:",
            adata_test.obs_names.is_unique)

            print("cell_type_adata obs_names unique:",
            cell_type_adata.obs_names.is_unique)
            indices = adata_test.obs_names.get_indexer(cell_type_adata.obs_names)

            
            
            X_cell_type = X_cell[indices]
                # # 6️⃣ 绘图（2行 × 3列）
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            axes = axes.flatten()

            for i, gene in enumerate(gene_names):
                ax = axes[i]
                expr = cell_type_adata[:, gene].X
                expr = expr.toarray().flatten() if issparse(expr) else np.ravel(expr)
                print('X_cell_type[:, 0].shape[0]', X_cell_type[:, 0].shape[0])
                print('expr.shape', expr.shape[0])
                sc_plot = ax.scatter(
                    X_cell_type[:, 0],
                    X_cell_type[:, 1],
                    c=expr,
                    s=30000 / expr.shape[0],
                    cmap="Reds",
                    linewidth=0
                )

                ax.set_title(gene.replace("a_a_", "a_"), fontsize=24)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                # 加上颜色条
                cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Expression", fontsize=9)

            # 去掉多余子图（若少于6个基因）
            for j in range(len(gene_names), len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            if distance_based:
                # plt.savefig(f"{cell_type_dir}/X_cell_umap_genes_distance_based.png", bbox_inches="tight", dpi=300)
                plt.savefig(f"{cell_type_dir}/X_cell_umap_genes_distance_based_mannual_v2.png", bbox_inches="tight", dpi=300)
            else:
                plt.savefig(f"{cell_type_dir}/X_cell_umap_genes_rank_based.png", bbox_inches="tight", dpi=300)

def plot_pseudotime(is_correct_array,
                    adata,
                    root_cell_index=None,
                    root_obs_key='HSC',
                    use_rep='cell_emb',            # e.g. 'X_pca' or None to compute PCA
                    n_pcs=50,
                    n_neighbors=15,
                    compute_umap=True,
                    umap_key='X_umap',
                    save_path=None,
                    cmap='viridis',
                    figsize=(6,5)):
    """
    计算 DPT pseudotime 并在 UMAP（或已有低维）上绘图。
    参数：
      adata: AnnData
      root_cell_index: int, 可选，指定作为 root 的细胞（adata.obs_names 的行索引）
      root_obs_key: str, 可选，adata.obs 中的列名，若提供则随机选择该列中对应的一个细胞作为 root （或可以设定为某个类别）
      use_rep: str or None，若为 'X_pca' 表示使用已有 adata.obsm['X_pca']；若 None 则会计算 PCA
      n_pcs: int，用于邻域构建的 PCA 维数
      n_neighbors: int，邻居数
      compute_umap: bool，是否计算 UMAP（若已有 umap 并且为 False 则不覆盖）
      umap_key: str，存放 UMAP 的 key
      save_path: str 或 None，若提供则保存图像
      cmap: matplotlib colormap 名称
      figsize: 图像尺寸
    返回：
      修改后的 adata（包含 .uns['dpt_pseudotime'] 以及 adata.obs['dpt_pseudotime']），并显示图像
    """
    import random
    np.random.seed(42)  # 设置 numpy 的随机种子
    random.seed(42) 
    # 1. 基本检查与预处理
    if 'cell_emb' not in adata.obsm:
        raise ValueError("adata 中没有表达矩阵。请确认 adata 已正确读取。")

    # 2. 计算/确保 PCA
    if use_rep == 'cell_emb' and 'cell_emb' in adata.obsm:
        pass
    else:
        # 只有在没有 PCA 时才计算 PCA（避免重复覆盖）
        if 'cell_emb' not in adata.obsm:
            sc.pp.pca(adata, n_comps=n_pcs, svd_solver='arpack', copy=False)
        use_rep = 'X_pca'

    # 3. 计算邻域图（knn）用于 dpt
    # sc.pp.neighbors(adata, )
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep=use_rep)

    # 4. 选择 root
    if root_cell_index is not None:
        # 支持传入行索引或 obs 名称
        try:
            root_cell = adata.obs_names[root_cell_index]
        except Exception:
            # 如果传入的是 obs 名称
            root_cell = root_cell_index
    # elif root_obs_key is not None:
    #     # 若给了一个 obs 列名，尝试从该列选取一个代表作为 root（如某个 cluster）
    #     if root_obs_key in adata.obs["celltype"].unique():
    #         vals = adata.obs[root_obs_key].unique()
    #         # 选择第一个类别的第一个细胞作为 root（用户可根据需要改）
    #         sel_cat = vals[0]
    #         candidates = adata.obs_names[adata.obs[root_obs_key] == sel_cat].tolist()
    #         root_cell = candidates[0]
    #     else:
    #         raise ValueError(f"adata.obs 中没有列名 {root_obs_key}")
    elif root_obs_key == "HSC":
        # root_obs_key 用于特别指定某一类细胞作为 root
        if "cell_type" not in adata.obs.columns:
            raise ValueError("adata.obs 中没有 'cell_type' 列，无法选择 HSC 作为 root。")

        # 找出所有 cell_type 为 HSC 的细胞
        hsc_cells = adata.obs_names[adata.obs["cell_type"] == "HSC"].tolist()

        if len(hsc_cells) == 0:
            raise ValueError("在 adata.obs['cell_type'] 中没有找到任何 HSC 细胞！")

        # 取第一个 HSC 作为 root cell
        root_cell = hsc_cells[0]
        print('root cell is:', root_cell)
    # else:
    #     # 默认选择度中心（degree 最大）作为 root 的近似：neighbors graph deg 最大的点
    #     G = adata.obsp.get('connectivities', None)
    #     if G is None:
    #         raise ValueError("无法找到邻接/connectivities 矩阵。请先运行 sc.pp.neighbors 或提供 root。")
    #     deg = np.array(G.sum(axis=1)).ravel()
    #     root_idx = int(np.argmax(deg))
    #     root_cell = adata.obs_names[root_idx]
    #     print('root cell is:', root_cell)
    # 5. 计算 DPT pseudotime
    # sc.tl.dpt 会在 adata.obs['dpt_pseudotime'] 里写入结果
    root_index = adata.obs_names.get_loc(root_cell)
    adata.uns['iroot'] = root_index
    sc.tl.dpt(adata, n_dcs=10, min_group_size=0.01)

    # 将 pseudotime 拷贝到 obs 方便调用（scanpy 会放在 adata.obs['dpt_pseudotime']）
    if 'dpt_pseudotime' not in adata.obs.columns:
        # scanpy 可能把结果存在 adata.uns['dpt_pseudotime'] 的结构里
        try:
            adata.obs['dpt_pseudotime'] = adata.uns['dpt']['pseudotime']
        except Exception:
            raise RuntimeError("未能在 adata 中找到 dpt pseudotime 结果。")
    # sc.pl.diffmap(adata, color='dpt_pseudotime')
    # 6. 低维（UMAP）计算或使用已有
    if compute_umap:
        if umap_key in adata.obsm:
            # 使用已有的 umap，但我们通常希望覆盖以保证和当前 neighbors 对应
            sc.tl.umap(adata, copy=False)
        else:
            sc.tl.umap(adata)

    # 7. 绘图：UMAP 点图 + pseudotime 颜色
    umap = adata.obsm.get('X_umap', None)
    if umap is None:
        raise RuntimeError("未找到 UMAP 嵌入（X_umap）。请确保 compute_umap=True 或者 adata.obsm['X_umap'] 存在。")

    pt = adata.obs['dpt_pseudotime'].values.astype(float)

    plt.figure(figsize=figsize)
    sc.pl.embedding(adata, basis='diffmap', color='dpt_pseudotime', cmap=cmap, show=False)
    plt.title('DPT pseudotime on UMAP')
    # 获取细胞的编号
    cell_ids = adata.obs_names

    # 添加编号标签
    # for i, txt in enumerate(cell_ids):
    #     # 只在前几个点上添加标签，以避免重叠
    #     if i % 500 == 0:  # 每50个细胞加一个编号，你可以调整这个值
    #         plt.text(umap[i, 0], umap[i, 1], str(i), fontsize=8, color='red', ha='right')
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    # 1. 绘制细胞类型颜色标记的 UMAP 图
    if 'cell_type' in adata.obs.columns:
        plt.figure(figsize=figsize)
        sc.pl.embedding(adata, basis='umap', color='cell_type', cmap='tab20', show=False)  # 使用tab20颜色映射（可根据需要更改）
        plt.title('Cell Types on UMAP')
        if save_path is not None:
            plt.savefig(save_path.replace('.png', '_cell_types.png'), dpi=300, bbox_inches='tight')
    else:
        print("警告: adata.obs 中未找到 'cell_type' 列，无法绘制细胞类型 UMAP 图。")

    # 2. 绘制分类是否正确的 UMAP 图
    adata.obs['is_correct_array'] = is_correct_array
    if 'is_correct_array' in locals():  # 确保 is_correct_array 已定义
        plt.figure(figsize=figsize)
        sc.pl.embedding(adata, basis='umap', color='is_correct_array', cmap='coolwarm', show=False)  # coolwarm 映射表示正确/错误分类
        plt.title('Classification Accuracy on UMAP')
        if save_path is not None:
            plt.savefig(save_path.replace('.png', '_classification_accuracy.png'), dpi=300, bbox_inches='tight')
    else:
        print("警告: 'is_correct_array' 未定义，无法绘制分类正确性 UMAP 图。")
    return adata

    
def run_trajectory_analysis(adata, root_cell_type="HSC", n_dcs=10, figsize=(8, 6), save_path=None):
    """
    基于 adata 执行完整的轨迹分析，并以 HSC 作为根节点进行轨迹推断。
    
    参数:
        adata (AnnData): 输入的单细胞数据对象。
        root_cell_type (str): 根节点细胞类型，默认是 "HSC"。
        n_dcs (int): 计算的 diffusion components 数量，默认为 10。
        figsize (tuple): 绘图尺寸，默认为 (8, 6)。
        save_path (str, optional): 保存图像的路径。如果不为 None，则保存图像。
    """
    
    # 1. 计算邻接图
    # adata.X = adata.obsm['cell_emb'].copy()
   
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, use_rep='cell_emb')

    # 3. 选择 HSC 细胞作为根节点
    hsc_cells = adata.obs_names[adata.obs["cell_type"] == "HSC"].tolist()
    root_index = adata.obs_names.get_loc(hsc_cells[0])
    adata.uns['iroot'] = root_index
    # 2. 计算 DPT（Diffusion Pseudotime）
    sc.tl.dpt(adata, n_dcs=10, min_group_size=0.01)
    # 4. 计算 PAGA（推断细胞的轨迹）
    adata.obs['celltype_labels'] = adata.obs['celltype_labels'].astype('category')
    sc.tl.paga(adata, groups='celltype_labels')
    
    # 5. 绘制 PAGA 图（轨迹图）
    plt.figure(figsize=figsize)
    sc.pl.paga(adata, layout='fr', fontsize=6, edge_width_scale=0.3,
    threshold=0.2, show=False)
    plt.title(f"PAGA Graph with {root_cell_type} as root cell")
    
    # 保存 PAGA 图（如果指定了 save_path）
    if save_path is not None:
        plt.savefig(save_path.replace(".png", "_paga.png"), dpi=300, bbox_inches='tight')
        print(f"PAGA 图已保存：{save_path.replace('.png', '_paga.png')}")
    
    # 6. 绘制伪时间轨迹图
    # 计算 UMAP（如果没有计算过）
    sc.tl.umap(adata, min_dist=0.3, spread=1.0)
    
    # 伪时间图绘制
    plt.figure(figsize=figsize)
    if 'dpt_pseudotime' not in adata.obs.columns:
        # scanpy 可能把结果存在 adata.uns['dpt_pseudotime'] 的结构里
        try:
            adata.obs['dpt_pseudotime'] = adata.uns['dpt']['pseudotime']
        except Exception:
            raise RuntimeError("未能在 adata 中找到 dpt pseudotime 结果。")
        
    sc.pl.embedding(adata, basis='umap', color='dpt_pseudotime', cmap='viridis', show=False)
    plt.title(f"DPT Pseudotime on UMAP (Root: {root_cell_type})")

    # 保存伪时间 UMAP 图（如果指定了 save_path）
    if 'cell_type' in adata.obs.columns:
        plt.figure(figsize=figsize)
        sc.pl.embedding(adata, basis='umap', color='cell_type', cmap='tab20', show=False)  # 使用tab20颜色映射（可根据需要更改）
        plt.title('Cell Types on UMAP')
        if save_path is not None:
            plt.savefig(save_path.replace('.png', '_cell_types.png'), dpi=300, bbox_inches='tight')
    plt.close('all')
    
    
    
    
def run_trajectory_analysis_v2(adata, root_cell_type="HSC", n_dcs=10, figsize=(8, 6), save_path=None):
    # 1. Neighbors 基于 cell_emb
    required_celltypes = [
    'HSC', 
    'G/M prog', 
    'CD14+ Mono', 
    'CD16+ Mono', 
    'Lymph prog', 
    'CD4+ T naive', 
    'CD4+ T activated'
]
    adata = adata[adata.obs['cell_type'].isin(required_celltypes), :]
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='cell_emb')

    # 2. Diffusion map —— 轨迹增强核心步骤
    sc.tl.diffmap(adata)  
    # 现在 adata.obsm['X_diffmap'] 是 cell_emb 的平滑轨迹化版本

    ###########################
    # 🔥 关键：DPT 使用 diffusion map，而不是 cell_emb 邻域
    ###########################

    # 3. 设置 HSC 为根节点
    hsc_cells = adata.obs_names[adata.obs["cell_type"] == "HSC"].tolist()
    root_index = adata.obs_names.get_loc(hsc_cells[0])
    adata.uns['iroot'] = root_index

    # 4. DPT 使用 diffusion components (来自 X_diffmap)
    sc.tl.dpt(adata)    # 不要传 n_dcs，这会自动用 diffmap dim

    ###########################
    # 🔥 PAGA 使用 celltype_labels
    ###########################

    adata.obs['celltype_labels'] = adata.obs['celltype_labels'].astype('category')
    sc.tl.paga(adata, groups='celltype_labels')

    # —— PAGA 图 ——
    plt.figure(figsize=figsize)
    sc.pl.paga(
        adata,
        layout='fr',
        fontsize=6,
        edge_width_scale=0.3,
        threshold=0.2,
        show=False
    )
    plt.title(f"PAGA Graph with {root_cell_type} as root cell")
    if save_path is not None:
        plt.savefig(save_path.replace(".png", "_paga.png"), dpi=300, bbox_inches='tight')

    ###########################
    # 🔥 UMAP 不用 diffmap，也不用 cell_emb，随你选
    # 但推荐用 diffusion map（更轨迹友好）
    ###########################

    sc.tl.umap(adata, min_dist=0.3, spread=1.0, init_pos='X_diffmap')
    # sc.tl.umap(adata, min_dist=0.3, spread=1.0)
    # 这样能保证 UMAP 的轨迹方向和 diffmap 一致

    ###########################
    # —— UMAP pseudotime 图 ——
    ###########################
    # 获取 UMAP 坐标
    umap_coords = adata.obsm['X_umap']

    # 获取 DPT 伪时间信息
    pseudotime = adata.obs['dpt_pseudotime']

    # 按伪时间排序细胞
    sorted_idx = np.argsort(pseudotime)

    # 按伪时间排序后的坐标
    sorted_coords = umap_coords[sorted_idx]
    
    plt.figure(figsize=figsize)
    sc.pl.embedding(
        adata,
        basis='umap',
        color='dpt_pseudotime',
        cmap='viridis',
        show=False
    )
    # 绘制伪时间轨迹线
    # num_cells_to_connect = 10  # 每个细胞类型选择10个细胞作为代表

    # representative_cells = []  # 用于存储代表性细胞的坐标
    # for cell_type in required_celltypes:
    #     # 筛选出当前细胞类型的细胞
    #     cell_type_idx = adata.obs['cell_type'] == cell_type
        
    #     # 获取该细胞类型的 UMAP 坐标和伪时间
    #     cell_type_umap = umap_coords[cell_type_idx]
    #     cell_type_pseudotime = pseudotime[cell_type_idx]
        
    #     # 按伪时间排序
    #     sorted_idx = np.argsort(cell_type_pseudotime)
    #     sorted_coords = cell_type_umap[sorted_idx]
        
    #     # 选择前 num_cells_to_connect 个细胞（根据伪时间排序）
    #     selected_coords = sorted_coords[:num_cells_to_connect]
    #     representative_cells.append(selected_coords)

    # # 将不同细胞类型的代表性细胞连接起来
    # representative_cells = np.vstack(representative_cells)  # 合并所有细胞类型的代表性细胞
    # plt.plot(representative_cells[:, 0], representative_cells[:, 1], color='black', lw=2)
    
    plt.title(f"DPT Pseudotime on UMAP (Root: {root_cell_type})")
    if save_path is not None:
        plt.savefig(save_path.replace(".png", "_dpt_pseudotime.png"), dpi=300, bbox_inches='tight')

    ###########################
    # —— 细胞类型 UMAP 图 ——
    ###########################

    if 'cell_type' in adata.obs.columns:
        plt.figure(figsize=figsize)
        sc.pl.embedding(
            adata,
            basis='umap',
            color='cell_type',
            cmap='tab20',
            show=False
        )
        plt.title('Cell Types on UMAP')
        if save_path is not None:
            plt.savefig(save_path.replace('.png', '_cell_types.png'), dpi=300, bbox_inches='tight')

    plt.close('all')

# def trajectory_mono3(adata, root_cell_type="HSC", n_dcs=10, figsize=(8, 6), save_path=None):
#     import monocle3 as m3
#     required_celltypes = [
#     'HSC', 
#     'G/M prog', 
#     'CD14+ Mono', 
#     'CD16+ Mono', 
#     'Lymph prog', 
#     'CD4+ T naive', 
#     'CD4+ T activated'
# ]
#     adata = adata[adata.obs['cell_type'].isin(required_celltypes), :]
#     sc.pp.neighbors(adata, n_neighbors=15, use_rep='cell_emb')
#     sc.tl.pca(adata)

#     # 使用Monocle3生成伪时间
#     adata_monocle = m3.AnnData(adata.X)
#     adata_monocle.obs = adata.obs
#     adata_monocle.var = adata.var
#     m3.tl.dpt(adata_monocle)

#     # 绘制轨迹
#     sc.pl.umap(adata, color=['dpt_pseudotime'], title="Pseudotime Trajectory")
    
def plot_paga(is_correct_array, save_dir):
    adata_test = sc.read_h5ad(f"{save_dir}/adata_test_embedding.h5ad")
    # adata = plot_pseudotime(is_correct_array, adata_test, save_path=f"{save_dir}/pseudotime_umap.png")
    # run_trajectory_analysis(adata_test, root_cell_type="HSC", n_dcs=10, figsize=(8, 6), save_path=f"{save_dir}/trajectory_analysis.png")
    run_trajectory_analysis_v2(adata_test, root_cell_type="HSC", n_dcs=10, figsize=(8, 6), save_path=f"{save_dir}/trajectory_analysis.png")
    # from sklearn.neighbors import NearestNeighbors
    # new_adata = sc.AnnData(X=adata_test.X) 
    # new_adata.obs = adata_test.obs.copy() 
    # new_adata.obs["cell_type"] = adata_test.obs["cell_type"].copy()

    # # Step 3: 计算细胞之间的邻接图
    # knn = NearestNeighbors(n_neighbors=15, metric='euclidean')  # 使用欧氏距离计算邻接图
    # knn.fit(X_cell)
    # distances, indices = knn.kneighbors(X_cell)
    
    # # Step 4: 创建完整的距离矩阵（方阵）
    # n_cells = X_cell.shape[0]
    # distance_matrix = np.full((n_cells, n_cells), np.inf)  # 初始化为无穷大

    # # 填充最近邻的距离
    # for i in range(n_cells):
    #     distance_matrix[i, indices[i]] = distances[i]
    #     distance_matrix[indices[i], i] = distances[i]  # 因为距离是对称的
    # np.fill_diagonal(distance_matrix, 0)
    # # Step 4: 创建邻接矩阵
    # adj_matrix = np.zeros((X_cell.shape[0], X_cell.shape[0]))
    # for i in range(X_cell.shape[0]):
    #     adj_matrix[i, indices[i]] = 1  # 标记邻接关系（距离最近的邻居设为1）

    # # Step 5: 将邻接矩阵添加到 new_adata 的 obsp 中
    # new_adata.obsp["connectivities"] = adj_matrix
    # new_adata.obsp["distances"] = distance_matrix
    # # Step 7: 将 X_cell 存储到 obsm 中，这样可以在后续计算中使用
    # # new_adata.obsm["X_umap"] = X_cell  # 存储到 obsm 中，供后续分析使用
    # # Step 8: 运行 `sc.pp.neighbors` 计算邻接图
    # sc.pp.neighbors(new_adata, n_neighbors=15, use_rep='X')  
    # sc.tl.paga(new_adata, groups="cell_type")  # 使用 "cell_type" 作为分群信息
    # # Step 8: 计算伪时间轨迹
    # sc.tl.dpt(new_adata)
    # # Step 7: 可视化 PAGA 结果
    # sc.pl.paga(new_adata, color="cell_type", layout="fr")
    # plt.savefig(f"{save_dir}/paga_allcell.png", dpi = 300)
    # # Step 9: 绘制伪时间轨迹图
    # sc.pl.dpt_groups_pseudotime(new_adata)
    # plt.savefig(f"{save_dir}/dpt_allcell.png", dpi = 300)
    
from scipy.stats import mannwhitneyu, ttest_ind
import pandas as pd
import os

def gene_distribution_significance(save_dir, df_long, genes, target_cts, top3_genes_per_ct):
    """
    只针对 top3_genes_per_ct 中给出的 (celltype, gene) 做检验：
    检验：该基因在当前细胞类型中的表达分布 vs 其它细胞类型中的表达分布
    结果保存为 CSV。
    """
    results = []

    # # 遍历字典中的细胞类型和对应的 top 基因
    # for ct, genes_ct in top3_genes_per_ct.items():
    #     for g in genes_ct:
    #         # 只看这个基因的数据
    #         df_g = df_long[df_long["gene"] == g]

    #         # 当前细胞类型中的表达
    #         expr_ct = df_g.loc[df_g["celltype"] == ct, "expression"].values
    #         # 其它细胞类型中的表达（这里是 df_long 里所有其它 celltype）
    #         expr_others = df_g.loc[df_g["celltype"] != ct, "expression"].values

    #         # 如果当前细胞类型或其他细胞太少，就跳过
    #         if len(expr_ct) < 3 or len(expr_others) < 3:
    #             print(
    #                 f"[Warning] 基因 {g}, 细胞类型 {ct} 样本数太少，跳过 "
    #                 f"(n_ct={len(expr_ct)}, n_others={len(expr_others)})"
    #             )
    #             continue

    #         stat, p_value = mannwhitneyu(
    #             expr_ct,
    #             expr_others,
    #             alternative="two-sided"
    #         )

    #         results.append({
    #             "celltype": ct,
    #             "gene": g,
    #             "n_cells_ct": len(expr_ct),
    #             "n_cells_others": len(expr_others),
    #             "mannwhitneyu_stat": stat,
    #             "p_value": p_value,
    #             "mean_ct": expr_ct.mean(),
    #             "mean_others": expr_others.mean(),
    #         })
    # 遍历每个细胞类型及其对应的基因
    for ct, genes_ct in top3_genes_per_ct.items():
        for g in genes_ct:
            # 只看当前基因的数据
            df_g = df_long[df_long["gene"] == g]

            # 当前细胞类型中的表达
            expr_ct = df_g.loc[df_g["celltype"] == ct, "expression"].values

            # 如果当前细胞类型中的样本太少，跳过
            if len(expr_ct) < 3:
                print(f"[Warning] 基因 {g}, 细胞类型 {ct} 样本数太少，跳过 (n_ct={len(expr_ct)})")
                continue

            # 遍历所有其他细胞类型，并与当前细胞类型进行检验
            for other_ct in df_g["celltype"].unique():
                if other_ct == ct:
                    continue  # 跳过当前细胞类型本身

                # 其他细胞类型中的表达
                expr_others = df_g.loc[df_g["celltype"] == other_ct, "expression"].values

                # 如果其他细胞类型中的样本太少，跳过
                if len(expr_others) < 3:
                    print(f"[Warning] 基因 {g}, 细胞类型 {ct} 与 {other_ct} 样本数太少，跳过 "
                        f"(n_ct={len(expr_ct)}, n_others={len(expr_others)})")
                    continue

                # # 进行 Mann-Whitney U 检验
                # stat, p_value = mannwhitneyu(
                #     expr_ct,
                #     expr_others,
                #     alternative="two-sided"
                # )

                # # 记录结果
                # results.append({
                #     "celltype": ct,
                #     "other_celltype": other_ct,
                #     "gene": g,
                #     "n_cells_ct": len(expr_ct),
                #     "n_cells_others": len(expr_others),
                #     "mannwhitneyu_stat": stat,
                #     "p_value": p_value,
                #     "mean_ct": expr_ct.mean(),
                #     "mean_others": expr_others.mean(),
                # })
                # 进行 t-test
                t_stat, p_value = ttest_ind(expr_ct, expr_others, equal_var=False)  # 使用Welch's t-test（不假设方差相等）

                # 计算 log2 Fold Change
                mean_ct = expr_ct.mean()
                mean_others = expr_others.mean()

                # 计算 log2 fold change，避免除以0
                if mean_others > 0:
                    log2_fc = np.log2(mean_ct / mean_others)
                else:
                    log2_fc = np.nan  # 如果其他细胞类型的平均值为0，则将log2_fold_change设为NaN

                # 判断基因是否为显著性高表达或显著性低表达
                if p_value < 0.05:
                    if log2_fc > 0:
                        expression_status = "Significantly higher expression in " + ct
                    elif log2_fc < 0:
                        expression_status = "Significantly lower expression in " + ct
                    else:
                        expression_status = "No significant difference"
                else:
                    expression_status = "No significant difference"

                # 记录结果
                results.append({
                    "celltype": ct,
                    "other_celltype": other_ct,
                    "gene": g,
                    "n_cells_ct": len(expr_ct),
                    "n_cells_others": len(expr_others),
                    "ttest_stat": t_stat,
                    "ttest_p_value": p_value,
                    "log2_fold_change": log2_fc,
                    "mean_ct": mean_ct,
                    "mean_others": mean_others,
                    "expression_status": expression_status
                })
    if not results:
        print("[Error] 没有得到任何有效的检验结果，未生成 CSV。")
        return

    # pval_df = pd.DataFrame(results)

    # # 可以按 p 值从小到大排序，或者按 celltype/gene 再按 p_value 排
    # pval_df = pval_df.sort_values(["celltype", "p_value"])

    # # 保存到文件
    # pval_path = os.path.join(save_dir, "top3_genes_per_ct_mannwhitney_pvalues.csv")
    # pval_df.to_csv(pval_path, index=False)
    # print(f"top3_genes_per_ct 中每个 (celltype, gene) 的 p-value 已保存到: {pval_path}")
    # 将结果保存为 DataFrame
    results_df = pd.DataFrame(results)

    # 查看结果
    print(results_df.head())

    # 可选择将结果保存为 CSV 文件
    results_df.to_csv(save_dir + "/gene_expression_significance_with_status.csv", index=False)


def plot_highly_variable_genes_distribution(adata_test, save_dir):
    top10_gene_csv = pd.read_csv(save_dir + "/dist_gene_correlation_top10_norm.csv")
    celltype_col = "label"   # 按你的 csv 修改
    gene_col     = "genes"
    corr_col     = "correlation"       # 如是 pearson_r 等，改一下

    target_cts = ["HSC", "G/M prog", "CD14+ Mono", "CD16+ Mono", "cDC2", "pDC"]
    top3_genes_per_ct = {}
    rows_for_csv = []
    
    for ct in target_cts:
        df_ct = top10_gene_csv[top10_gene_csv[celltype_col] == ct]
        if df_ct.empty:
            print(f"[Warning] CSV 中没有找到细胞类型 {ct} 的记录，跳过")
            continue
        df_ct = df_ct[df_ct[corr_col].abs() > 0.3]
        if df_ct.empty:
            print(f"[Warning] {ct} 中 |{corr_col}| > 0.3 的基因为空，跳过")
            continue

        df_ct = df_ct.sort_values(by=corr_col, ascending=False)
        df_top3 = df_ct.head(3)
        genes_ct = df_ct[gene_col].head(3).tolist()
        top3_genes_per_ct[ct] = genes_ct
        
        for _, row in df_top3.iterrows():
            rows_for_csv.append({
                "celltype": ct,
                "gene": row[gene_col],
                "correlation": row[corr_col],
            })
        df_out = pd.DataFrame(rows_for_csv, columns=["celltype", "gene", "correlation"])
        df_out.to_csv(os.path.join(save_dir, "top3_genes_per_ct.csv"), index=False)
        # 只保留在 adata.var_names 中存在的基因
        # genes_ct = [g for g in genes_ct if g in adata_test.var_names]
        # if len(genes_ct) == 0:
        #     print(f"[Warning] {ct} 的 top3 基因都不在 adata.var_names 中，跳过")
        #     continue
        if not top3_genes_per_ct:
            print("[Error] 没有任何细胞类型得到有效的 top3 基因，函数结束。")
            return

        # 汇总所有细胞类型的 top3 基因，去重、排序
        # genes = sorted({g for gs in top3_genes_per_ct.values() for g in gs})
        # print("用于绘图的基因列表：", genes)
        # 按 target_cts 的顺序展开字典，同时保留每个 ct 内部的 top3 顺序
        genes = []
        for ct in target_cts:
            if ct not in top3_genes_per_ct:
                continue
            for g in top3_genes_per_ct[ct]:
                if g not in genes:      # 去重但不打乱顺序
                    genes.append(g)
            
        json_path = os.path.join(save_dir, "top3_genes_per_ct.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(top3_genes_per_ct, f, ensure_ascii=False, indent=2)
            
        # ==== 3. 只保留这 6 种细胞类型的细胞 ====
        mask = adata_test.obs["celltype"].isin(target_cts)
        adata_sub = adata_test[mask].copy()

        if adata_sub.n_obs == 0:
            print("[Error] 在 adata_test.obs['celltype'] 中找不到目标细胞类型，函数结束。")
            return

        # ==== 4. 提取表达矩阵（从 .X，如果你用的是某个 layer，可以改这里） ====
        X = adata_sub[:, genes].X
        if issparse(X):
            X = X.A                                     # 稀疏矩阵转 dense

        # 构造成 DataFrame：每行一个细胞，列是基因
        df_expr = pd.DataFrame(X, columns=genes)
        df_expr["celltype"] = adata_sub.obs["celltype"].values

        # 转成长表：每行 = (celltype, gene, expression)
        df_long = df_expr.melt(
            id_vars="celltype",
            var_name="gene",
            value_name="expression"
        )
        
        gene_distribution_significance(save_dir, df_long, genes, target_cts, top3_genes_per_ct)
        # ==== 5. 画一张图：x=gene, y=expression, hue=celltype 的箱型图 ====
        # plt.figure(figsize=(max(8, 0.8 * len(genes)), 6))
        plt.figure(figsize=(14, 6))
        # plt.figure()
        sns.boxplot(
            data=df_long,
            x="gene",
            y="expression",
            hue="celltype",
            showfliers=False,  # 不画离群点，避免太乱；需要可以改成 True
            medianprops={"linewidth": 2.5}
        )

        plt.xlabel("Gene")
        plt.ylabel("Expression")
        plt.xticks(rotation=90)

        plt.legend(
            title="Cell type",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0.
        )

        plt.tight_layout()
        out_path = os.path.join(save_dir, "top3_genes_boxplot_HSC_GMprog_Mono_cDC_pDC.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"箱型图已保存到: {out_path}")

def evaluate_subclusters(save_dir, adata_test, X_cell):
    adata_sub = adata_test[adata_test.obs["celltype"] == "CD14+ Mono"].copy()
    cd14_mono_indices = adata_test.obs[adata_test.obs["celltype"] == "CD14+ Mono"].index
    # 2. 将索引转换为整数数组
    # cd14_mono_indices_int = adata_test.obs.loc[cd14_mono_indices].index.astype(int)
    embeddings_cd14_mono = X_cell[adata_test.obs_names.isin(cd14_mono_indices), :]
    adata_sub.obsm["X_latent"] = embeddings_cd14_mono  # 你模型的embedding

    sc.pp.neighbors(adata_sub, use_rep="X_latent", n_neighbors=15)
    sc.tl.leiden(adata_sub, resolution=0.012)  # 或 Louvain

    # 3. 可视化 UMAP 图
    sc.tl.umap(adata_sub)       # 计算 UMAP
    plt.figure(figsize=(8, 6))  # 设置图像大小

    # 4. 绘制 UMAP 图，着色按 Leiden 聚类结果
    sc.pl.umap(adata_sub, color=["leiden"], title="Leiden Clustering for CD14+ Mono", show=False)

    # 5. 保存 UMAP 图像
    umap_save_path = f"{save_dir}/umap_cd14_mono_leiden.png"
    plt.savefig(umap_save_path, dpi=300, bbox_inches="tight")

def evaluate_mislabel_detection(adata_test, labels, cell_type_map, softmax_dist, save_dir, temperature, src_ct = "CD16+ Mono", 
                                ax=None, save_single=True, show_xticklabels=True):
    celltype_arr = np.asarray(adata_test.obs["celltype"])
    idx_src = np.where(celltype_arr == src_ct)[0]
    sim_mat = softmax_dist[idx_src, :]  # (n_src, n_proto)

    # 横坐标类别名称：按 cell_type_map 的 id 排序（0..n_proto-1）
    id2ct = {int(v): k for k, v in cell_type_map.items()}
    n_proto = sim_mat.shape[1]
    x_labels = [id2ct.get(i, f"class_{i}") for i in range(n_proto)]

    # 每个 prototype 一列数据作为一个 violin
    data_list = [sim_mat[:, i] for i in range(n_proto)]

    if ax is None:
        plt.figure(figsize=(max(10, 0.6 * n_proto), 4))
        ax = plt.gca()
        
    if sim_mat.shape[0] != 0:
        ax.violinplot(
            data_list,
            showmeans=False,
            showmedians=True,
            showextrema=False
        )

        ax.set_xticks(np.arange(1, n_proto + 1))
        ax.set_ylabel("Similarity", fontsize=16)
        ax.set_title(f"{src_ct}", fontsize=18)
        
        if show_xticklabels:
            ax.set_xticklabels(x_labels, rotation=90, ha="center", fontsize=20)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        # ✅只有 save_single=True（单图模式）才保存/关闭
        if save_single:
            plt.tight_layout()
            out_path = f"{save_dir}/violin_{src_ct.replace(' ','_')}_all_prototypes_T{temperature:.1f}.png"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close()
        
def plot_dist_distribution(adata_test, labels, cell_type_map, softmax_dist, save_dir, temperature):
    os.makedirs(save_dir, exist_ok=True)

    all_cts = list(cell_type_map.keys())
    n_panels = len(all_cts)

    ncols = 1
    nrows = n_panels

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(12, max(2.2 * nrows, 6)),
        sharex=True,   # ✅共享横轴
        sharey=True
    )
    axes = np.array(axes).reshape(-1)

    for i, k in enumerate(all_cts):
        evaluate_mislabel_detection(
            adata_test, labels, cell_type_map, softmax_dist,
            save_dir, temperature,
            src_ct=k,
            ax=axes[i],
            save_single=False,
            show_xticklabels=(i == n_panels - 1)  # ✅只在最后一行显示x标签
        )

    # 全局标题/轴标题
    fig.suptitle(
        f"Distribution of Prototype similarity (T={temperature:.1f})",
        fontsize=14
    )
    fig.supxlabel("Prototype (cell type)", fontsize=12)
    # y轴每行都有，这里不额外设置全局 ylabel（也可以加：fig.supylabel(...)）

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 输出路径：如果你想放 ckpt_dir 就用它，否则用 save_dir
    out_path = os.path.join(save_dir, f"violin_ALL_celltypes_all_prototypes_T{temperature:.1f}.png")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# 计算熵
def compute_entropy(softmax_dist):
    # 计算每个样本的熵: H(p) = -sum(p * log(p))
    p = softmax_dist
    p[p == 0] = 1e-10  # 防止 log(0) 的错误
    entropy = -np.sum(p * np.log(p), axis=1)  # 按行计算熵
    return entropy

def plot_outlier_distribution(adata_test, labels, cell_type_map, softmax_dist, save_dir):
    # 计算每个样本与原型的最大相似度值
    max_similarity = np.max(softmax_dist, axis=1)                       # 每行的最大值，表示与最相似原型的相似度
    cell_type_str_list = np.array(adata_test.obs["celltype"]).tolist()  # 实际类别标签
    current_celltype_labels = [cell_type_map[cell] for cell in cell_type_str_list]
    current_celltype_labels = np.array(current_celltype_labels)

    # 目标类别（数值型类别）
    # target_classes = [11, 12, 13]    # pancreas
    # target_classes = [9, 10, 11]       # myeloid
    target_classes = [14, 15, 16]       # BMMC

    # 找到属于目标类别的样本
    target_indices = np.isin(current_celltype_labels, target_classes)
    # 提取目标类别样本的熵和非目标类别样本的熵
    target_similarity = max_similarity[target_indices]
    non_target_similarity = max_similarity[~target_indices]
    # 创建绘图
    plt.figure(figsize=(6, 6))
    # sns.boxplot(data=[target_similarity, non_target_similarity], 
    #             palette=["skyblue", "orange"], 
    #             showmeans=False, 
    #             widths=0.5,
    #             boxprops=dict(linewidth=2),  # 调整箱体边缘线粗细
    #             medianprops=dict(linewidth=3),  # 调整中间线粗细
    #             capprops=dict(linewidth=1.5),  # 调整上下边缘线粗细
    #             flierprops=dict(marker='o', markersize=5, linestyle='none'))
    sns.violinplot(
    data=[target_similarity, non_target_similarity],
    palette=["skyblue", "orange"],
    cut=0,              # 不外推分布（很重要）
    inner="quartile",   # 显示中位数和四分位
    linewidth=2,
    scale="width"       # 避免样本量差异导致小提琴宽度误导
)
    
    # 设置标题和标签
    plt.title("Boxplot of Max Similarity Values", fontsize=20)
    # plt.xlabel("Classes", fontsize=14)
    plt.ylabel("Max Similarity", fontsize=18)
    plt.xticks([0, 1], ['Unseen Samples', 'Other Samples'])  # 自定义横坐标标签
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_dir}/max_similarity_boxplot.png", dpi=300, bbox_inches='tight')

def plot_entropy(adata_test, labels, cell_type_map, softmax_dist, save_dir):
    entropy = compute_entropy(softmax_dist)
    np.save(os.path.join(save_dir, "unknown_entropy.npy"), entropy)
    # 假设 cell_type_map 已经定义
    cell_type_str_list = np.array(adata_test.obs["celltype"]).tolist()  # 实际类别标签
    current_celltype_labels = [cell_type_map[cell] for cell in cell_type_str_list]
    current_celltype_labels = np.array(current_celltype_labels)

    # 目标类别（数值型类别）
    target_classes = [11, 12, 13]

    # 找到属于目标类别的样本
    target_indices = np.isin(current_celltype_labels, target_classes)
    # 提取目标类别样本的熵和非目标类别样本的熵
    target_entropy = entropy[target_indices]
    non_target_entropy = entropy[~target_indices]

    # 画出箱型图
    max_len = max(len(target_entropy), len(non_target_entropy))
    target_entropy = np.pad(target_entropy, (0, max_len - len(target_entropy)), mode='constant', constant_values=np.nan)
    non_target_entropy = np.pad(non_target_entropy, (0, max_len - len(non_target_entropy)), mode='constant', constant_values=np.nan)

    # 处理填充后的数据，将 NaN 替换为 0（或者其他值，如果合适的话）
    target_entropy = np.nan_to_num(target_entropy, nan=0)
    non_target_entropy = np.nan_to_num(non_target_entropy, nan=0)
    # 将数据放入列表中，供 sns.boxplot 使用
    data = [target_entropy, non_target_entropy]

    # 设置图像大小
    plt.figure(figsize=(5, 4))

    # 绘制箱型图
    sns.boxplot(data=data, 
                widths=0.5,        # 箱体宽度
                patch_artist=True, # 填充颜色
                boxprops=dict(facecolor='skyblue', color='black'),  # 箱体颜色
                flierprops=dict(markerfacecolor='red', marker='o', markersize=5),  # 异常值标记
                medianprops=dict(color='black'),  # 中位数线颜色
                showmeans=False)   # 不显示均值点

    # 设置箱型图的标签和标题
    plt.xticks([0, 1], ['Unseen Classes', 'Other Classes'])
    plt.ylabel('Entropy')
    plt.title('Entropy Distribution of Known vs Unseen Samples')
    plt.savefig(os.path.join(save_dir, "unseen_similarity_entropy.png"), dpi = 300)
    plt.close() 

def compute_dist_correction(is_correct_array, sample_proto_distances):
    from scipy.stats import pearsonr, spearmanr
    # Compute Pearson correlation coefficient and p-value
    pearson_corr, pearson_p_value = pearsonr(is_correct_array, sample_proto_distances)

    # Compute Spearman correlation coefficient and p-value
    spearman_corr, spearman_p_value = spearmanr(is_correct_array, sample_proto_distances)

    # Print results
    print(f"Pearson Correlation: {pearson_corr:.4f}, p-value: {pearson_p_value:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}, p-value: {spearman_p_value:.4f}")
    
def prototype_dist_correlation(save_dir, adata_test, cell_type_map):
    mean_conf = np.load(os.path.join(save_dir, "mean_conf.npy"))
    var_conf  = np.load(os.path.join(save_dir, "var_conf.npy"))
    is_correct_array = np.load(os.path.join(save_dir, "is_correct_array.npy"))
    # # entropy_mean = np.load(os.path.join(save_dir, "entropy_mean.npy"))
    X_umap = np.load(os.path.join(save_dir, "X_umap.npy"))
    X_cell = X_umap[:mean_conf.shape[0], :]
    proto = X_umap[mean_conf.shape[0]:, :]
    
    if cell_type_map is not None:
        # index2cell = {v: k for k, v in cell_type_map.items()}

        celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels
        labels = adata_test.obs["celltype_labels"]
    else:
        labels = adata_test.obs["celltype"]
    
        
    distances = np.linalg.norm(X_cell[:, None, :] - proto[None, :, :], axis=2)    # (30401, 17)

    # temps = np.round(np.arange(0.1, 1.01, 0.1), 2)

    # for temperature in temps:
    temperature = 0.5  # 或者其他小的值
    softmax_dist = np.exp(-distances / temperature) / np.sum(np.exp(-distances / temperature), axis=1, keepdims=True)
    sample_proto_distances = softmax_dist[np.arange(X_cell.shape[0]), labels]
    ######################## mislabel detection ################
    # plot_dist_distribution(adata_test, labels, cell_type_map, softmax_dist, save_dir, temperature)
    # plot_entropy(adata_test, labels, cell_type_map, softmax_dist, save_dir)
    # plot_outlier_distribution(adata_test, labels, cell_type_map, softmax_dist, save_dir)
    ######################## 画图部分 ########################
    
    # plot_celltype_prototype(cell_type_map, adata_test, X_umap, X_cell, proto, save_dir)
    # plot_dist_conf_variance(labels, sample_proto_distances, mean_conf, var_conf, save_dir)
    
    # plot_eachclasses_dist_conf_variance(labels, sample_proto_distances, mean_conf, var_conf, save_dir)
    
    # plot_accuracy_conf_variance(is_correct_array, mean_conf, var_conf, labels, sample_proto_distances, save_dir)
    # plot_normalized_distance_vs_confidence(is_correct_array, mean_conf, sample_proto_distances, labels, save_dir)
    # compute_dist_correction(is_correct_array, sample_proto_distances)
    # plot_marker_gene_bubble(adata_test, save_dir)
    # plot_correct_incorrect_marker_gene_bubble(adata_test, mean_conf, save_dir)
    
    
    # compute_dist_genes_correlation(sample_proto_distances, adata_test, labels, save_dir, cell_type_map, predictions = None)
    
    # plot_dist_correlation_marker_gene_bubble(adata_test, mean_conf, save_dir)
    # plot_dist_correlation_marker_gene_eachcelltype_bubble(adata_test, mean_conf, sample_proto_distances, save_dir)   # 画出每种细胞类型的 correction 与 GE 的关系
    # compute_rank_genes(adata_test, save_dir, mean_conf, cell_type_map, sample_proto_distances=None)
    
    # plot_dist_correlation_marker_gene_eachcelltype_violin(mean_conf, cell_type_map, sample_proto_distances, save_dir)
    
    # plot_celltype_subtypes(adata_test, save_dir, cell_type_map, distance_based=True)
    
    
    
    ################# 画出高变基因的分布图 ########################
    # plot_highly_variable_genes_distribution(adata_test, save_dir)
    ################# 画出发育轨迹图 #############################
    # plot_paga(is_correct_array, save_dir)
    ################ 进一步聚类找出 Monocyte CD14亚群 ########################
    # evaluate_subclusters(save_dir, adata_test, X_cell)
    ################# 通过差异基因找出显著性富集的通路 ########################

def prototype_dist_correlation_gut_sf(save_dir, adata_test, cell_type_map):
    mean_conf = np.load(os.path.join(save_dir, "mean_conf.npy"))
    var_conf  = np.load(os.path.join(save_dir, "var_conf.npy"))
    is_correct_array = np.load(os.path.join(save_dir, "is_correct_array.npy"))
    # # entropy_mean = np.load(os.path.join(save_dir, "entropy_mean.npy"))
    X_umap = np.load(os.path.join(save_dir, "X_umap.npy"))
    X_cell = X_umap[:mean_conf.shape[0], :]
    proto = X_umap[mean_conf.shape[0]:, :]
    
    adata_test_copy = adata_test.copy()
    # adata_test_copy.obs["celltype_labels"] = adata_test_copy.obs["celltype_labels"].replace("SF_like", "Surface_foveolar")
    
    if cell_type_map is not None:
        # index2cell = {v: k for k, v in cell_type_map.items()}

        celltype_str_list = np.array(adata_test_copy.obs["celltype"]).tolist()               
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test_copy.obs["celltype_labels"] = current_celltype_labels
        labels = adata_test_copy.obs["celltype_labels"]
    else:
        labels = adata_test_copy.obs["celltype"]
        
    distances = np.linalg.norm(X_cell[:, None, :] - proto[None, :, :], axis=2)    # (30401, 17)

    temperature = 0.5  # 或者其他小的值
    softmax_dist = np.exp(-distances / temperature) / np.sum(np.exp(-distances / temperature), axis=1, keepdims=True)
    sample_proto_distances = softmax_dist[np.arange(X_cell.shape[0]), labels]
     
    ######################## 画图部分 ########################
    
    # plot_celltype_prototype_gut(cell_type_map, adata_test, X_umap, X_cell, proto, save_dir)           # 画出 gut_sf 的 prototype 分布图
    # plot_dist_conf_variance(labels, sample_proto_distances, mean_conf, var_conf, save_dir)            # 画出整体的 dist vs conf vs var 图
    # plot_eachclasses_dist_conf_variance(labels, sample_proto_distances, mean_conf, var_conf, save_dir)
    # plot_accuracy_conf_variance(is_correct_array, mean_conf, var_conf, labels, sample_proto_distances, save_dir) # 画出正确率 vs conf vs var 图
    # plot_normalized_distance_vs_confidence(is_correct_array, mean_conf, sample_proto_distances, labels, save_dir) # 画出 normalized dist vs conf 图
    # plot_marker_gene_bubble(adata_test, save_dir)
    # plot_correct_incorrect_marker_gene_bubble(adata_test, mean_conf, save_dir)
    
    
    compute_dist_genes_correlation(sample_proto_distances, adata_test, labels, save_dir, cell_type_map, predictions = None) # 计算 dist 与基因表达的相关性
    # plot_dist_correlation_marker_gene_bubble(adata_test, mean_conf, save_dir)
    # plot_dist_correlation_marker_gene_eachcelltype_bubble(adata_test, mean_conf, sample_proto_distances, save_dir)   # 画出每种细胞类型的 correction 与 GE 的关系
    # compute_rank_genes(adata_test, save_dir, mean_conf, cell_type_map, sample_proto_distances=None)
    plot_dist_correlation_marker_gene_eachcelltype_violin(mean_conf, cell_type_map, sample_proto_distances, save_dir)
    
    plot_celltype_subtypes(adata_test, save_dir, cell_type_map, distance_based=True)

    



        
