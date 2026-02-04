import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# import gym
from torch.distributions import Normal
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.distributions import Bernoulli
from tqdm import trange

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
import copy
import json
from pathlib import Path
import sys
import time
import traceback
from typing import List, Tuple, Dict, Union, Optional
import warnings

from anndata import AnnData
# from torchtext.vocab import Vocab
# from torchtext._torchtext import (
#     Vocab as VocabPybind,
# )
# import scvi
import scanpy as sc

import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# sys.path.insert(0, "../scgpt/model")
# sys.path.insert(0, "../")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import scgpt as scg
# from scgpt.model import TransformerModel, AdversarialDiscriminator
# from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt.preprocess import Preprocessor
# from scgpt import SubsetsBatchSampler
# from scgpt.utils import set_seed, eval_scib_metrics, load_pretrained

# sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

# dataset_name = "PBMC_10K"
dataset_name = "pancreas"
load_model = "../save/scEvolver_human"
model_dir = Path(load_model)
vocab_file = model_dir / "vocab.json"

if dataset_name == "PBMC_10K":
    adata = sc.read_h5ad(save_path='../data/pbmc/')  # 11990 × 3346
    print(adata)
    ori_batch_col = "batch"
    print('adata.obs["str_labels"]', adata.obs["str_labels"])
    adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
    adata.var = adata.var.set_index("gene_symbols")
    print(adata.var)
    data_is_raw = True
    

elif dataset_name == "pancreas":
    adata = sc.read_h5ad("../data/PANCREAS/human_pancreas_norm_complexBatch.h5ad")
    ori_batch_col = "tech"
    adata.var['gene_symbols'] = adata.var.index        # add gene_symbols column
    adata.var = adata.var.set_index("gene_symbols")    # set gene_symbols as index
    adata.obs["celltype"] = adata.obs["celltype"].astype("category")
    data_is_raw = True
# make the batch category column
adata.obs["str_batch"] = adata.obs[ori_batch_col].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()


vocab = GeneVocab.from_file(vocab_file)

preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=51,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key="str_batch" if dataset_name != "heart_cell" else None)

adata_sorted = adata[adata.obs["batch_id"].argsort()].copy() 
# adata_sorted.write('../data/pbmc/pbmc_10k_new.h5ad')
adata_sorted.write('../data/PANCREAS/pancreas_data_new.h5ad')  # 16382, 18262

# adata = sc.read_h5ad('../data/pbmc/pbmc_10k_new.h5ad')
# for batch_id in adata.obs["batch_id"].unique():
#     adata_batch = adata[adata.obs["batch_id"] == batch_id].copy()
#     filename = "../data/pbmc/" + f"pbmc_new_batch{batch_id}.h5ad"
#     adata_batch.write(filename)
#     print(f"Saved: {filename}")
for batch_id in adata.obs["batch_id"].unique():
    adata_batch = adata[adata.obs["batch_id"] == batch_id].copy()
    filename = "../data/pbmc/" + f"pancreas_batch{batch_id}.h5ad"
    adata_batch.write(filename)
    print(f"Saved: {filename}")