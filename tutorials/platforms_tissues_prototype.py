# CUDA_VISIBLE_DEVICES=1
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
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

def init_wandb():
    hyperparameter_defaults = dict(
        seed=0,
        # dataset_name="ms",
        # dataset_name="pancreas",
        # dataset_name = "myeloid", 
        # dataset_name = "pancreas_filter3",       
        dataset_name = "myeloid_filter3",
        do_train=True,
        load_model="../save/scEvolver_human",
        weight_dir="../save/scEvolver_human",
        # experiment_name="fine_tune_on_pancreas_minibatch_baseline(uniformsplit_no_schedular_nlayers_cls0_pastmse[origion])",
        # experiment_name="fine_tune_on_pancreas_minibatch_baseline(uniformsplit_no_schedular_nlayers_cls0)",
        # experiment_name="fine_tune_on_pancreas_minibatch_baseline(randomsplit_no_schedular_nlayers_cls3_init_class13)",
        mask_ratio=0.0,
        epochs=15,                 # set to 1000 for few-shot
        n_bins=51,
        MVC=False,
        ecs_thres=0.0,
        dab_weight=0.0,
        lr=1e-4,
        batch_size=16,
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
        DSBN=False,                            # 
        k_samples=50,
        pastmse=False,
        replay=True,
        init_class = 14,
        filter_sample = False,
        randomsplit = False,              # set to False when using balanced_sampler
        fewshot = None,
        nlayers_cls=3,
        use_best_initnextbatch = False,
        adapter=False,
        freeze_except_layer1011 = False,
        freeze_all = True,
        loramoe = True,  
        proto_loss = True,
        num_of_expert = 4,         # when loramoe=False this should be None
        num_of_gate = 4,
        repultion_loss = False,
        entropy =  False,
        classifier = "Linear",
        anchorloss = False,
        schedule = "cosine_schedule_with_warmup",               # "cosine_schedule_with_warmup",     # stepLR
        proto_weight = 1,
        cope_loss = False,
        weight_miloss = False,
        update_classifier = False,
        ema = False,
        correct_sample = True,
        patience = 5,
        weighted_loss = False,
        contrastive_proto_loss = False,
        adapter_dim = 64,
        decrease_lr_all = False,
        valid_ratio = 0.1,
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
            "_updata_per5batch_" + "_blanced_sampler" + "_proto_loss_1stbatch" + "_contrastive_loss" + "_decrease_lr_all(*3)"
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
    if config["dataset_name"] == "pancreas" or config["dataset_name"] == "pancreas_filter3":
        config["init_class"] = 14
        config["lr"] = 1e-4
    elif config["dataset_name"] == "myeloid" or config["dataset_name"] == "myeloid_filter3":
        config["init_class"] = 12
        config["lr"] = 1e-5
    return config

def check_out_layer_in_optimizer(model, optimizer):
    out_layer = model.cls_decoder.out_layer
    # Get data_ptrs for all parameters in the optimizer
    optimizer_param_ptrs = {p.data_ptr() for g in optimizer.param_groups for p in g['params']}

    # Check which parameters are not included in the optimizer
    missing = []
    for name, param in out_layer.named_parameters():
        if param.requires_grad and param.data_ptr() not in optimizer_param_ptrs:
            missing.append(name)

    if missing:
        print(f"❌ The following parameters in out_layer are NOT in the optimizer: {missing}")
    else:
        print("✅ All parameters in out_layer are included in the optimizer.")
    
from tutorials.utils import (
    EarlyStopping,
    prepare_dataloader,
    prepare_testdata,
    prepare_data,
    eval_scib_metrics,
    plot_entropy_accuracy,
    CosineLinear,
    ClsDecoder,
)

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
        
class ContinualClassify():
    def __init__(self, config, vocab, num_batch_types, max_batch_idx=5, modeldict_name = "best_model.pt"):
        self.config = config
        self.vocab = vocab
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_max_classes = self.config["init_class"]
        self.classifier = self.config["classifier"]
        self.model, self.model_file = self.prepare_model(num_batch_types, self.num_max_classes, modeldict_name)
        self.model.to(self.device)
        self.model.cls_decoder = ClsDecoder(512, self.num_max_classes, classifier=self.classifier).to(self.device)
        if self.config["load_model"] is not None:
            load_pretrained(self.model, torch.load(self.model_file, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')), verbose=False)
        
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
        self.ppp_loss = PPPloss(net=self.model, mode="joint", T=0.8, tracker={'log_it': [], 'loss': [], 'lnL_pos': [], 'lnL_neg': []})
        
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
        self.memory_proto = defaultdict(list)
        self.past_valid_loaders = {}
        # self.contrastive_proto_loss_list, self.repultion_loss_list = [], []

    def prepare_model(self, num_batch_types, num_max_classes, modeldict_name="best_model.pt"):
        if self.config["loramoe"]:
            from model.loramoemodel import TransformerModel, AdversarialDiscriminator   # loramoe
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
                n_cls=num_max_classes,                  # initialize classifier heads (n_cls=num_max_classes)
                vocab=self.vocab,
                dropout=self.config["dropout"],
                pad_token=pad_token,
                pad_value=pad_value,
                do_mvc=self.config["MVC"],
                do_dab=False,                           # Set to True when using DSBN
                use_batch_labels=False,                 # Set to True when using DSBN
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
                adapter_dim = self.config["adapter_dim"],
            )
        else:
            model = TransformerModel(
            len(self.vocab),
            embsize,
            nhead,
            d_hid,
            nlayers,
            nlayers_cls=self.config["nlayers_cls"],
            n_cls=num_max_classes,                  # initialize classifier heads (n_cls=num_max_classes)
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
            if "indices_to_keep" in batch_data:
                adata_train_indices = batch_data["indices_to_keep"].to(self.device)
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
                        batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                        # batch_labels=test_batch_idx,
                        CLS=True,
                        CCE=False,
                        MVC=self.config["MVC"],
                        ECS=self.config["ecs_thres"] > 0,
                        do_sample=False,
                    )
                cell_emb = output_dict["cell_emb"].squeeze(1)
                cell_emb = F.normalize(cell_emb, p=2, dim=1)
                # cell_emb = torch.cat([cell_emb_past, cell_emb], dim=-1)
                # cell_emb = output_dict["cls_output"][1]
                cell_emb_list.append(cell_emb.detach().cpu())
                celltype_labels_list.append(celltype_labels.detach().cpu())
                adata_train_indices_list.append(adata_train_indices.detach().cpu())
                masked_positions = input_values.eq(-1)
                # metrics_to_log = {}
                total_num += len(input_gene_ids)

                
                output_values = output_dict["cls_output"].detach()
                loss_cls = self.criterion_cls(output_dict["cls_output"], celltype_labels)
                loss = loss_cls
                # print(f"loss requires_grad? {loss.requires_grad}")
                probs = F.softmax(output_values, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)                 # 
                entropy_list.extend(entropy.detach().cpu())
                preds = output_values.argmax(1)
                accuracy_list.extend((preds == celltype_labels.detach()).cpu())
                # train_loss_list.append(loss.item())
                # metrics_to_log.update({"train/cls": loss_cls.item()})

                error_rate = 1 - (
                    (preds == celltype_labels.detach())
                    .sum()
                    .item()
                ) / celltype_labels.size(0)
                train_iter_loss_list.append(loss_cls.item())
                # compute MSE between past and current model
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
                        # loss_past = self.criterion_cls(past_output_dict["cls_output"], celltype_labels)     # the past model's cls_decoder is randomly initialized, result is invalid
                        ################### Add classification loss after concatenation ########################
                 
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
                    ################################
                    # proto_loss = torch.zeros(1, device=self.device)
                    # print(f"loss requires_grad???? {loss.requires_grad}")
                    proto_loss_list.append((proto_loss * self.config["proto_weight"]).item())
                    if  len(self.memory_proto) != 0 and self.config["contrastive_proto_loss"]:
                        proto_contrastive_loss = self.contrastive_proto_loss(self.memory_proto, self.old_proto)
                        loss = loss + proto_contrastive_loss * 10
                        proto_contrastive_loss_list.append(proto_contrastive_loss.item())

                print(f"train/cls: {loss_cls.item()}, proto_loss: {proto_loss.item() if isinstance(proto_loss, torch.Tensor) else None}, train_loss: {loss.item()}")
            self.model.zero_grad()
            self.optimizer.zero_grad()
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
        cur_loss = total_loss / total_num    # current epoch loss
        proto_loss_last = sum(proto_loss_list) / total_num
        repultion_loss = sum(repultion_loss_list) / total_num
        contrastive_proto_loss = sum(proto_contrastive_loss_list) / total_num
        del proto_loss, total_loss
        gc.collect()     # free Python garbage
        torch.cuda.empty_cache()  # free unused PyTorch CUDA memory cache
        return cell_emb_list, celltype_labels_list,adata_train_indices_list, cur_loss, train_iter_loss_list, proto_loss_last, contrastive_proto_loss, entropy_list, accuracy_list
    
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
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                            # batch_labels= test_batch_idx if False or self.config["DSBN"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=False,
                            ECS=False,
                            do_sample=False,
                        )
                    cell_emb = output_dict["cell_emb"].squeeze(1)
                    cell_emb = F.normalize(cell_emb, p=2, dim=1)
                    output_values = output_dict["cls_output"]
                    probs = F.softmax(output_dict["cls_output"], dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                    loss = self.criterion_cls(output_values, celltype_labels)
                    if test_batch_idx != 0 and self.config["proto_loss"]:
                        proto_loss = self.ppp_loss(cell_emb, celltype_labels, self.old_proto, self.memory_proto, self.device, eps=1e-8)
                        total_proto_loss += proto_loss.item() 
                   
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

        # compute F1
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
            class_names = np.unique(np.concatenate((labellist, predictions)))  # get all classes
            cm = confusion_matrix(labellist, predictions, labels=class_names)

            # plot heatmap

            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.savefig(str(save_dir) + "/" + f"batch_{test_batch_idx}_Confusion_Matrix.png")
            plot_entropy_accuracy(entropy_list, accuracy_list, save_dir, test_batch_idx, labellist)    #  plot entropy-accuracy figure

        return total_loss / total_num, total_error / total_num, result_dict, eval_iter_loss_list, total_proto_loss / total_num

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
        Update classifier weights using class embedding means.
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

    def update_prototype(self, embedding_list, label_list, epoch = None):
        embedding_list = torch.cat(embedding_list, dim=0).to(self.device)
        label_list = torch.cat(label_list, dim=0).to(self.device)

        class_list = torch.unique(label_list).cpu().numpy()
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0] 
            embedding = embedding_list[data_index]
            new_proto = embedding.mean(0).float() 
            # self.model.cls_decoder.out_layer.weight.data[int(class_index)] = proto.to(self.device)
            # if self.config["cope_loss"]:
            # self.old_proto[int(class_index)] = proto.clone()
            # proto = self.old_proto.get(int(class_index), torch.zeros(512).to(self.device))
            proto = self.old_proto.get(int(class_index), torch.zeros_like(new_proto).to(self.device))

            if self.config["proto_loss"] and self.config["ema"]:
                # If prototype not initialized, assign directly
                if proto.sum() == 0:
                    self.old_proto[int(class_index)] = new_proto.clone()
                else:
                    # Update prototype via EMA
                    ema_momentum = 0.5
                    self.old_proto[int(class_index)] = (
                        ema_momentum  * self.old_proto[int(class_index)] +
                        (1 - ema_momentum)* new_proto
                    )
            else:
                self.old_proto[int(class_index)] = new_proto.clone()
            self.old_proto[int(class_index)] = F.normalize(self.old_proto[int(class_index)], p=2, dim=0)
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

                # extract entropy, accuracy, and embeddings for the class
            class_entropy = entropy_list[data_index]
            class_accuracy = accuracy_list[data_index]
            class_embedding = embedding_list[data_index]
            # find indices that were classified correctly
            correct_mask = class_accuracy.bool()
            arange_indices = torch.arange(len(data_index), device=correct_mask.device)
            correct_indices = arange_indices[correct_mask]

            if correct_indices.numel() >= 1:
                # keep only entropies of correctly classified samples
                filtered_entropy = class_entropy[correct_indices]
                filtered_embedding = class_embedding[correct_indices]

                # select embeddings: either all correct samples or top-k lowest entropy
                if self.config["correct_sample"]:
                    selected_embedding = filtered_embedding
                else:
                    topk = min(10, filtered_entropy.size(0))
                    _, top_indices = torch.topk(-filtered_entropy, topk)  # -entropy: sort by increasing entropy

                    selected_embedding = filtered_embedding[top_indices]
                
                print('entropy filter samples of class:{}'.format(class_index), selected_embedding.shape[0])
                # compute prototype
                # proto = selected_embedding.mean(0).float()
                # assert proto.shape[0] == self.model.cls_decoder.out_layer.weight.shape[1]

                # update class prototype (out_layer weights)
                # with torch.no_grad():
                #     self.model.cls_decoder.out_layer.weight[int(class_index)] = proto.to(self.device)
                # if self.config["proto_loss"]:
                #     self.old_proto[int(class_index)] = proto.clone()
                new_proto = selected_embedding.mean(0).float()
                proto = self.old_proto.get(int(class_index), torch.zeros(512).to(self.device))

                if self.config["proto_loss"] and self.config["ema"]:
                    # If prototype not initialized, assign directly
                    if proto.sum() == 0:
                        self.old_proto[int(class_index)] = new_proto.clone()
                    else:
                        # Update prototype via EMA
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
            # find all indices for this class
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0]
            
            # extract entropy, accuracy, and embeddings for the class
            class_entropy = entropy_list[data_index]
            class_accuracy = accuracy_list[data_index]
            class_embedding = embedding_list[data_index]

            # find indices that were classified correctly
            correct_mask = class_accuracy.bool()
            arange_indices = torch.arange(len(data_index), device=correct_mask.device)
            correct_indices = arange_indices[correct_mask]


            if correct_indices.numel() >= 1:
                # keep only entropies of correctly classified samples
                filtered_entropy = class_entropy[correct_indices]
                filtered_embedding = class_embedding[correct_indices]

                # select top-k samples with smallest entropy
                topk = min(10, filtered_entropy.size(0))
                _, top_indices = torch.topk(-filtered_entropy, topk)  # -entropy: sort by increasing entropy

                selected_embedding = filtered_embedding[top_indices]
                print('entropy filter samples of class:{}'.format(class_index), selected_embedding.shape[0])
                # compute prototype
                proto = selected_embedding.mean(0).float().detach()
                assert proto.shape[0] == self.model.cls_decoder.out_layer.weight.shape[1]

                # update class prototype（out_layer weights）
                with torch.no_grad():
                    self.model.cls_decoder.out_layer.weight[int(class_index)] = proto.to(self.device)
                if self.config["proto_loss"]:
                    self.old_proto[int(class_index)] = proto.clone()
                    
            else:
                # if classify wrong, no correctly classified samples then skip update
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

        # convert to numpy arrays
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
        p = F.softmax(prob_logits, dim=1)     # shape: [B, K]   clustering probability distribution
        p = p + eps  # numerical stability
        log_p = torch.log(p)

        cond_entropy = -torch.mean(torch.sum(p * log_p, dim=1))  # scalar conditional entropy

        p_mean = torch.mean(p, dim=0)  # shape: [K]
        log_p_mean = torch.log(p_mean + eps)

        marginal_entropy = -torch.sum(p_mean * log_p_mean)  # scalar marginal entropy over classes

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
                    repulsion_loss += torch.exp(sim)  # the more similar, the greater the penalty
        repulsion_loss /= (new_weight_matrix.size(0) * (new_weight_matrix.size(0) - 1))
        return repulsion_loss
    def cope_ppp_loss(self, features, labels, prototypes, temperature=0.1):
        """
        CoPE PPP Loss (only for classes that exist in prototypes).

        Args:
            features: Tensor [B, D] — current batch embeddings (should be normalized)
            labels: Tensor [B] — sample labels
            prototypes: defaultdict(int -> Tensor[D]) — class prototype vectors
            temperature: float — temperature parameter τ

        Returns:
            Scalar loss tensor
        """

        device = features.device
        B, D = features.shape
        all_class_ids = list(prototypes.keys())

        if len(all_class_ids) == 0:
            return torch.tensor(0.0, device=device)

        # stack all normalized prototype vectors into a matrix [C, D]
        all_protos = torch.stack([F.normalize(prototypes[c].to(device), dim=0) for c in all_class_ids]).detach()

        total_loss = 0.0
        count = 0

        for cls in all_class_ids:
            cls = int(cls)
            # mask for current class
            mask_pos = (labels == cls)
            mask_neg = (labels != cls)

            # samples and their embeddings for the current class
            pos_feats = features[mask_pos]  # [N_pos, D]
            neg_feats = features[mask_neg]  # [N_neg, D]

            if pos_feats.size(0) < 1:
                continue  # skip if no positive samples

            pc = F.normalize(prototypes[cls].to(device), dim=0).detach()  # [D]

            for i in range(pos_feats.size(0)):
                xi = pos_feats[i]  # current positive sample

                # -------- Attractor Set --------
                pseudo_protos = [pos_feats[j].detach() for j in range(pos_feats.size(0)) if j != i]
                attractor_set = [pc] + pseudo_protos
                attractor_set = torch.stack(attractor_set)  # [K, D]
                sim_pos = torch.matmul(attractor_set, xi) / temperature  # [K]
                sim_soft = F.softmax(sim_pos, dim=0)
                pos_prob = sim_soft[0]  # pc is the first element

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
    
    def contrastive_proto_loss(self, old_protos, new_protos, temperature=0.1):
        """
        Args:
            old_protos: Dict[int, Tensor] — historical prototypes
            new_weight_matrix: Tensor — current model classifier weights (num_classes x dim)
        Returns:
            torch.Tensor — contrastive loss that supports gradient backpropagation
        """
        # device = new_protos.device
        losses = []

        for i, proto_old_list in old_protos.items():
            proto_new = new_protos[i].to(self.device)  # new prototype for current class

            for proto_old in proto_old_list:  # iterate through historical prototypes for this class
                proto_old = proto_old.to(self.device)

                # positive sample similarity
                pos_sim = F.cosine_similarity(proto_old.unsqueeze(0), 
                                            proto_new.unsqueeze(0), dim=-1) / temperature

                # negative sample similarities (new protos of other classes)
                neg_sims = []
                for j, neg_proto in new_protos.items():
                    if j != i:  # only take other classes
                        neg_proto = neg_proto.to(self.device)
                        neg_sim = F.cosine_similarity(proto_old.unsqueeze(0), 
                                                    neg_proto.unsqueeze(0), dim=-1) / temperature
                        neg_sims.append(neg_sim)

                if len(neg_sims) == 0:
                    continue  # skip when only one class

                neg_sims = torch.cat(neg_sims, dim=0)  # (num_classes - 1,)
                logits = torch.cat([pos_sim, neg_sims], dim=0)  # (num_classes,)
                labels = torch.zeros(1, dtype=torch.long, device=self.device)  # positive sample is at position 0

                # InfoNCE loss
                loss = F.cross_entropy(logits.unsqueeze(0), labels)
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(losses).mean()

    def process_batch(self, adata_train, logger, save_dir, dataset_name, experiment_name, all_adata_test, all_batch_results, test_batch_idx):
        '''
        adata_train, config, logger, save_dir, config["dataset_name"], config["experiment_name"], all_adata_test, all_batch_results,
            test_batch_idx
        '''
        learning_rate = []
        from functions.memory_bank import example_bank_update
        ############################# Prepare data #################################
        # count each cell type
        if self.config["init_class"] == 8 or self.config['fewshot'] is not None:
        # if self.config["filter_sample"]:
            celltype_counts = adata_train.obs["celltype"].value_counts()
            # find cell types with sufficient counts
            valid_celltypes = celltype_counts[celltype_counts >= self.config["fewshot"] + 1].index

            # filter out samples from classes with insufficient counts
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

        # compute cell types and labels for current batch (test_batch_idx)
        current_label_dict, current_celltype_labels = np.unique(
            np.array(adata_train.obs["celltype"].tolist()), return_inverse=True
        )

        adata_train.obs["celltype_labels"] = current_celltype_labels  # temporarily save

        if test_batch_idx == 0:
            # initialization: use current batch info
            new_model_annotation = current_label_dict
            self.old_model_annotation = current_label_dict

            # mapping table: celltype -> label
            celltype_to_label = {ct: i for i, ct in enumerate(self.old_model_annotation)}
            adata_train.obs["celltype_labels"] = adata_train.obs["celltype"].map(celltype_to_label)
            celltypes_labels = adata_train.obs["celltype_labels"].to_numpy()
            with open(save_dir / "celltype_to_label.json", 'w') as file:
                json.dump(celltype_to_label, file)
            # define initial number of classifier heads
            # self.model.cls_decoder.out_layer.out_features = len(self.old_model_annotation)
            if self.config["replay"]:
                example_bank_previous = sc.AnnData()
        else:

            
            # incremental batch
            new_model_annotation = current_label_dict

            # find newly added cell types
            new_to_add = [ct for ct in new_model_annotation if ct not in self.old_model_annotation]

            # create full list of cell types
            combined = np.concatenate([self.old_model_annotation, new_to_add])
            self.old_model_annotation = combined

            # update mapping table: celltype -> label (keep consistent numbering)
            celltype_to_label = {ct: i for i, ct in enumerate(self.old_model_annotation)}
            with open(save_dir / "celltype_to_label.json", 'w') as file:
                json.dump(celltype_to_label, file)
            # reassign labels to ensure consistency with previous data
            adata_train.obs["celltype_labels"] = adata_train.obs["celltype"].map(celltype_to_label)
            celltypes_labels = adata_train.obs["celltype_labels"].to_numpy()
            # update number of classifier heads
            # self.model.cls_decoder.out_layer.out_features = len(self.old_model_annotation)
            if self.config["replay"]:
                example_bank_previous = torch.load(save_dir / "example_bank.pth")['example_bank']                         # previously stored examples
                # memory bank labels and batches
                example_bank_previous_label = [
                    list(self.old_model_annotation).index(i)
                    for i in np.array(example_bank_previous.obs['celltype'])
                ]
                example_bank_previous_batch = np.array(example_bank_previous.obs['batch_id'])
                # adata_train = anndata.concat([adata_train, example_bank_previous], axis=0, merge='same', label=None, keys=None)
                # example_bank_previous_label = [list(self.old_model_annotation).index(i) for i in np.array(example_bank_previous.obs['celltype'])]
                # celltypes_labels = np.concatenate([celltypes_labels, np.array(example_bank_previous_label)], 0)
                # all_counts = (
                #     adata_train.layers[input_layer_key].A
                #     if issparse(adata_train.layers[input_layer_key])
                #     else adata_train.layers[input_layer_key]
                # )
            
    ############################# Prepare model ###############################
        old_out_layer = self.model.cls_decoder.out_layer
        old_out_features = old_out_layer.out_features
        in_features = old_out_layer.in_features
        new_out_features = len(self.old_model_annotation)
        if new_out_features > old_out_features:
            # create a larger out_layer
            if self.config["classifier"] == "Linear":
                new_out_layer = nn.Linear(in_features, new_out_features).to(self.device)
                
                # migrate existing weights
                with torch.no_grad():
                    new_out_layer.weight[:old_out_features] = old_out_layer.weight
                    new_out_layer.bias[:old_out_features] = old_out_layer.bias
            else:
                new_out_layer = CosineLinear(in_features, new_out_features).to(self.device)
                with torch.no_grad():
                    new_out_layer.weight[:old_out_features] = old_out_layer.weight
                    new_out_layer.sigma.data = copy.deepcopy(old_out_layer.sigma.data)
                    # new_out_layer.bias[:old_out_features] = old_out_layer.bias
            
            # replace classifier head
            self.model.cls_decoder.out_layer = new_out_layer
        ##########################################################################
        optimizer_param_ids = set(p.data_ptr() for group in self.optimizer.param_groups for p in group['params'])

        # collect cls_decoder.out_layer parameters
        missing_params = []
        for name, param in self.model.cls_decoder.out_layer.named_parameters():
            if param.requires_grad and param.data_ptr() not in optimizer_param_ids:
                print(f"Param '{name}' not in optimizer. Will add.")
                missing_params.append(param)

        # add missing parameters to optimizer if any
        if missing_params:
            self.optimizer.add_param_group({'params': missing_params})

        ############################ past_model ################################
        # if self.past_model is not None:
        #     old_out_layer = self.past_model.cls_decoder.out_layer
        #     old_out_features = old_out_layer.out_features
        #     in_features = old_out_layer.in_features
        #     new_out_features = len(self.old_model_annotation)
        #     if new_out_features > old_out_features:
        #         # create a larger out_layer
        #         new_out_layer = nn.Linear(in_features, new_out_features).to(self.device)
                
        #         # migrate existing weights
        #         with torch.no_grad():
        #             new_out_layer.weight[:old_out_features] = old_out_layer.weight
        #             new_out_layer.bias[:old_out_features] = old_out_layer.bias
                
        #         # replace classifier head
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
                all_counts, celltypes_labels, batch_ids, test_size=self.config["valid_ratio"], shuffle=True
            )
            # first split indices
            adata_indices = np.arange(len(adata_train))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=self.config["valid_ratio"],
                shuffle=True,
                stratify=celltypes_labels if not self.config["randomsplit"] else None,
            )

            # subset adata_train by indices
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
                # append memory bank examples to the training set
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
          
        if self.config["weighted_loss"]:
            labels = torch.tensor(adata_train_split.obs["celltype_labels"].values, dtype=torch.long).to(self.device)
            class_counts = torch.bincount(labels)  # shape: [num_classes]

            # Simple approach: class_weights = 1 / frequency
            class_weights = 1.0 / (class_counts.float() + 1e-8)

            # Optionally normalize weights so they sum to num_classes
            class_weights = class_weights * (len(class_counts) / class_weights.sum())

            class_weights = class_weights.to(self.device)

            # define weighted cross-entropy
            self.criterion_cls = nn.CrossEntropyLoss(weight=class_weights)  # redefine cross-entropy loss with weights

        gene_ids = np.array(self.vocab(genes), dtype=int)

        tokenized_train = tokenize_and_pad_batch(
            train_data,
            gene_ids,
            max_len=3001,    # this does not affect the final tokenized shape: tokenized_train['genes'].shape: torch.Size([903, 1201])
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
        train_epoch_loss, train_proto_list, val_proto_list = [], [], []
        eval_epoch_loss = []
        contrastive_proto_loss_list = []
        ######################### Set learning rate ##########################
        self.early_stopper = EarlyStopping(patience=self.config["patience"], min_delta=1e-4, mode='min')
        if self.config["schedule"] == "cosine_schedule_with_warmup":
            if self.config["fewshot"] is not None:
                warmup_steps = 10
            else:
                warmup_steps = 2
            if self.config["decrease_lr_all"]:
                pass
            else:
                total_steps = self.config["epochs"] * (train_data.shape[0] // self.config["batch_size"]) * 30
                del self.scheduler
                torch.cuda.empty_cache()
                gc.collect()
                self.scheduler = CosineScheduleWithWarmup(self.optimizer,
                            num_warmup_steps=warmup_steps,          # warmup steps
                            num_training_steps=total_steps        # total training steps
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
            ########### Save past validation loaders ############
            self.past_valid_loaders[test_batch_idx] = valid_loader

            if self.config["do_train"] and epoch < self.config["epochs"]:
                cell_emb_list, train_labels,adata_train_indices_list, train_loss, _, proto_loss, contrastive_proto_loss, entropy_list, accuracy_list = self.train(
                    loader=train_loader,
                    logger=logger,
                    epoch=epoch,
                    test_batch_idx=test_batch_idx
                )
            elif self.config["do_train"] and epoch == self.config["epochs"]:
                cell_emb_list, train_labels,adata_train_indices_list, train_loss, _, proto_loss, contrastive_proto_loss, entropy_list, accuracy_list = self.train(
                    loader=train_loader,
                    logger=logger,
                    epoch=epoch,
                    test_batch_idx=test_batch_idx
                )
            

            # if self.config["fewshot"] and epoch % 10 == 0:    # changes under fewshot        
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
                    past_items = list(self.past_valid_loaders.items())  # take first N-1 (key, loader) pairs
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
                                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                                            # batch_id = torch.tensor(task_id),
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
                                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
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

                        # save
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
                    elif self.config["proto_loss"] and not self.config["update_classifier"] and self.config["entropy"]:
                        self.update_prototype_with_entropy(cell_emb_list, train_labels, entropy_list, accuracy_list)
                        print("Update prototype weights with entropy")
                    else:
                        self.update_classifier_weights(cell_emb_list, train_labels)
                        print("update_classifier_weights")
            self.early_stopper(val_loss, self.model)
            if self.early_stopper.early_stop or epoch == self.config["epochs"]:
                self.update_prototype(cell_emb_list, train_labels, epoch='final')
                break

            if self.config["schedule"] == "stepLR":
                self.scheduler.step()
            
        gc.collect()
        torch.cuda.empty_cache()
        del train_data_pt, valid_data_pt, train_loader, valid_loader, train_loss, proto_loss, val_proto_loss,
        torch.save(self.model.state_dict(), save_dir / f"last_model_batch_{test_batch_idx}.pt")

        # all_adata_test.append(adata_train)
        if self.config["replay"]:
            example_bank_update(adata_train_split,adata_train_indices_list, train_labels, entropy_list, accuracy_list, save_dir, test_batch_idx, self.old_proto, cell_emb_list)
            
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

        ##################### Use best eval model for clustering evaluation and next-batch initialization ######################
        if self.config["use_best_initnextbatch"]:
            load_pretrained(self.model, torch.load(save_dir / f"best_model_batch_{test_batch_idx}.pt"), verbose=False)

        if test_batch_idx == self.max_test_id:
            self.plot_clusters_prototypes(combined_adata_test, self.old_proto, input_layer_key, gene_ids, test_batch_idx, save_dir, best_model)
        self.plot_train_val_loss(train_epoch_loss, eval_epoch_loss, save_dir, test_batch_idx)
        del train_epoch_loss, eval_epoch_loss
        if self.config["repultion_loss"] or self.config["proto_loss"]:
            self.plot_proto_repultion_loss(train_proto_list, val_proto_list, save_dir, test_batch_idx)
        if self.config["contrastive_proto_loss"]:
            self.plot_contrastive_loss(contrastive_proto_loss_list, save_dir, test_batch_idx)
        del train_proto_list, val_proto_list
        return best_model, combined_adata_test, gene_ids, learning_rate
        
    def plot_gray_batch(self, adata, prototype, input_layer_key, gene_ids, test_batch_idx, save_dir, best_model = None,
                         save_name="val", cell_type_map=None, legend_on=False, experiment = "query_mapping"):
        from plot_prototype.plot_umap import plot_outlier_detection_umap
        if save_name == 'val':
            with open(str(save_dir) + "/" + f"prototype_{test_batch_idx}.pkl", "wb") as f:
                pickle.dump(prototype, f)
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
        )
        tokenized_all = tokenize_and_pad_batch(
            all_counts,
            gene_ids,
            max_len=3001,                           # should be 1201
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"].to(self.device), tokenized_all["values"].to(self.device)
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        batch_ids = torch.tensor(adata.obs["batch_id"].values).to(self.device)
        batch_ids = torch.zeros_like(batch_ids)                    # During training only one BN layer was trained; set batch_ids to 0

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
        prototypes = F.normalize(prototype, dim=1)
        # 若是 tensor，转为 numpy
        prototypes_np = prototypes.detach().cpu().numpy() if torch.is_tensor(prototypes) else prototypes
        n_prototypes = prototypes_np.shape[0]
        ################### compute the Euclidean distance as the prediction results ############### 
        X_cells = torch.tensor(cell_embeddings, device=self.device, dtype=torch.float32)
        prototypes = torch.tensor(prototype, device=self.device, dtype=torch.float32)
        distances = torch.cdist(X_cells, prototypes, p=2)
        temperature = 0.5
        softmax_dist = torch.softmax(-distances / temperature, dim=1)
        prediction_labels = torch.argmax(softmax_dist, dim=1).detach().cpu().numpy().tolist()
        index2cell = {v: k for k, v in cell_type_map.items()}
        current_predict_labels = [index2cell[cell] for cell in prediction_labels]
        ##############################################################
        # Step 2: concatenate all vectors (cell & prototypes)
        X_all = np.concatenate([X_cells.detach().cpu().numpy(), prototypes_np], axis=0)
        #######################################################
        # get real cell-type labels（must contain in .obs）
        if cell_type_map is not None:
            index2cell = {v: k for k, v in cell_type_map.items()}

            celltype_str_list = np.array(adata.obs["celltype"]).tolist()
            current_celltype_labels = [index2cell[cell_type_map[cell]] for cell in celltype_str_list]
            current_celltype_labels = np.array(current_celltype_labels)
            adata.obs["celltype_labels"] = current_celltype_labels
            labels = adata.obs["celltype_labels"]
        else:
            labels = adata.obs["celltype"]

        if self.config["dataset_name"] == "myeloid":
            batches = adata.obs["str_batch"].astype(str).str.split("_").str[1]
        else:
            batches = adata.obs["str_batch"]

        highlight_batch_list = batches.unique()
        print("highlight_batch_list", highlight_batch_list)
        ################## query mapping #################################
        if self.config["dataset_name"] == "pancreas" or self.config["dataset_name"] == "pancreas_filter3":
            reference_batch = ["celseq", "celseq2", "fluidigmc1", "inDrop1", "inDrop2", "inDrop3"]
            highlight_batch_list = ["inDrop4", "smarter", "smartseq2"]
        elif self.config["dataset_name"] == "myeloid" or self.config["dataset_name"] == "myeloid_filter3":
            reference_batch = ["GSE154763_KIDNEY", "GSE154763_LYM", "GSE154763_PAAD", "GSE154763_THCA", "GSE154763_UCEC"]
            highlight_batch_list = ["GSE154763_ESCA", "GSE154763_MYE", "GSE154763_OV-FTC"]
        ref_mask = batches.isin(reference_batch)
        query_mask = batches.isin(highlight_batch_list)
        np.save(str(save_dir) + "/" + "X_all.npy", X_all)
        X_all = np.load(str(save_dir) + "/" + "X_all.npy")
        n_cells = X_all.shape[0] - n_prototypes
        if experiment == "outlier_detection":
            plot_outlier_detection_umap(self.config, n_cells, X_all, labels, prototypes_np, ref_mask, query_mask, save_dir, test_batch_idx, legend_on = False)

        elif experiment == "query_mapping":
            ######################## not query mapping ########################
            umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
            X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]
            ################################################################# 
            
            # # # get all celltype and construct global mapping
            # all_celltypes = sorted(np.unique(labels))
            all_celltypes = np.unique(labels)
            palette = sns.color_palette("tab20", n_colors=len(all_celltypes))
            color_map = dict(zip(all_celltypes, palette))   

            # ######################## plot reference figures ######################
            plt.figure(figsize=(5, 5))
            
            sns.scatterplot(
                x=X_umap[:n_cells, 0][ref_mask],
                y=X_umap[:n_cells, 1][ref_mask],
                hue=labels[ref_mask],
                palette=color_map,      
                s=30000 / n_cells,       
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
            
            # # 2. plot figure in a loop: query mapping
            from plot_prototype.plot_umap import plot_umap_gray_incremental
            for test_batch_idx in range(len(highlight_batch_list)):
                selected_batches = highlight_batch_list[:test_batch_idx+1]
                mask = batches.isin(selected_batches)
                plot_umap_gray_incremental(
                    X_umap, n_cells, batches, labels, color_map,
                    mask, test_batch_idx,
                    save_dir, save_name = "label_eval&test_new", legend_on = False
                )
                ################# add prediction visualization ########################
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
                
                # # background：plot gray of all cells initially
                # sns.scatterplot(
                #     x=X_umap[:n_cells, 0],
                #     y=X_umap[:n_cells, 1],
                #     color="lightgray",
                #     s=5,
                #     linewidth=0,
                #     legend=False
                # )

                # # ：first test_batch_idx+1 batches
                # # selected_batches = highlight_batch_list[:test_batch_idx+1]
                # # mask = batches.isin(selected_batches)
                # # 
                # sns.scatterplot(
                #     x=X_umap[:n_cells, 0][mask],
                #     y=X_umap[:n_cells, 1][mask],
                #     hue=labels[mask],
                #     palette=color_map,   
                #     s=5,
                #     linewidth=0,
                #     legend=legend_on
                # )

                # add prototypes
                # plt.scatter(
                #     X_umap[n_cells:, 0],
                #     X_umap[n_cells:, 1],
                #     edgecolors='black',
                #     facecolors='none',
                #     s=60,
                #     marker='X',
                #     label='Prototypes',
                # )

                # plt.title(f"t={test_batch_idx}", fontsize=24)
                # plt.xticks([])
                # plt.yticks([])
                # plt.tight_layout()
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
                
        ##################### batch plot ############################   
        # plot batch corectionif legend_on:
            # if legend_on:
            #     plt.figure(figsize=(7, 5))
            # else:
            #     plt.figure(figsize=(5, 5))
            # sns.scatterplot(
            #     x=X_umap[:n_cells, 0][mask],
            #     y=X_umap[:n_cells, 1][mask],
            #     hue=batches[mask],
            #     palette="tab10",
            #     s=30000 / n_cells,
            #     linewidth=0,
            #     legend=legend_on
            # )
            # plt.xticks([])  
            # plt.yticks([])  
            # plt.tight_layout()
            # ax = plt.gca()
            # for spine in ax.spines.values():
            #     spine.set_edgecolor("black")
            #     spine.set_linewidth(1.5)
            # if legend_on:
            #     plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
            #     plt.savefig(str(save_dir) + "/" + save_name + f"umap_batch_correction_batch{test_batch_idx}_legend.png", bbox_inches='tight', edgecolor='black',dpi=300)
            # else:
            #     plt.savefig(str(save_dir) + "/" + save_name + f"umap_batch_correction_batch{test_batch_idx}_evaltest.png", bbox_inches='tight', edgecolor='black',dpi=300)
        
        ################### filter rare cell type index #################################
        # if self.config["dataset_name"] == "pancreas_filter3":
        #     target_cts = {"macrophage", "mast", "quiescent_stellate"}
        # elif self.config["dataset_name"] == "myeloid_filter3":
        #     target_cts = {"pDC_LILRA4", "cDC3_LAMP3", "Macro_SPP1"}            # 'pDC_LILRA4', 'cDC3_LAMP3', 'Macro_SPP1'
        # else:
        #     pass
        # mask_filter3 = adata.obs["celltype"].astype(str).isin(target_cts)   
        
        # target_indices = adata.obs_names[mask_filter3]          # Index 
        # plt.figure(figsize=(5, 5))

   
        # sns.scatterplot(
        #     x=X_umap[:n_cells, 0],
        #     y=X_umap[:n_cells, 1],
        #     color="lightgray",
        #     s=30000/n_cells,
        #     linewidth=0,
        #     legend=False
        # )
        # sns.scatterplot(
        #     x=X_umap[:n_cells, 0][mask_filter3],
        #     y=X_umap[:n_cells, 1][mask_filter3],
        #     hue=labels[mask_filter3],
        #     palette=color_map,  
        #     s=30000/n_cells,
        #     linewidth=0,
        #     legend=legend_on
        # )
        # plt.title(f"t={test_batch_idx}", fontsize=24)
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        # if legend_on:
        #     plt.legend(markerscale=15, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        #     plt.savefig(
        #         str(save_dir) + "/" + save_name + f"gray_rare_cell_incremental_batch_all_legend_evaltest.png",
        #         bbox_inches='tight',
        #         edgecolor='black',
        #         dpi=300
        #     )
        # else:
        #     plt.savefig(
        #         str(save_dir) + "/" + save_name + f"gray_rare_cell_incremental_batch_all_evaltest.png",
        #         bbox_inches='tight',
        #         edgecolor='black',
        #         dpi=300
        #     )

    def plot_clusters_prototypes(self, adata, prototype, input_layer_key, gene_ids, test_batch_idx, save_dir, best_model = None, save_name="val", cell_type_map = None):
    # prototypes = F.normalize(prototype.prototypes, dim=1)
    # # prototypes = prototype.prototypes
    # X_cells = adata_sorted.obsm["X_scGPT"]  # cell embedding vectors，shape = [n_cells, 1200]
    # n_cells = X_cells.shape[0]
        # plt.rcParams.update({'font.size': 16})
        # plt.rcParams['font.family'] = 'ARIAL'
        if save_name == 'val':
            with open(str(save_dir) + "/" + f"prototype_{test_batch_idx}.pkl", "wb") as f:
                pickle.dump(prototype, f)
        prototype_list = [prototype[c] for c in sorted(prototype.keys())]
        prototype_tensor = torch.stack(prototype_list, dim=0)  # shape [num_classes, D]

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
        )
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
        all_gene_ids, all_values = tokenized_all["genes"].to(self.device), tokenized_all["values"].to(self.device)
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        batch_ids = torch.tensor(adata.obs["batch_id"].values).to(self.device)
        batch_ids = torch.zeros_like(batch_ids)                    # only one BN layer is trained, so the batch_ids need to be set to 0.

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
        prototypes = F.normalize(prototype, dim=1)
        # prototypes（shape [n_prototypes, 1200]）
        prototypes_np = prototypes.detach().cpu().numpy() if torch.is_tensor(prototypes) else prototypes
        n_prototypes = prototypes_np.shape[0]

        # concatenate all vectors (cell & prototypes)
        X_all = np.concatenate([X_cells.detach().cpu().numpy(), prototypes_np], axis=0)

        # reduce dimension with UMAP
        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]

        # use true labels to plot
        plt.figure(figsize=(5, 5))

        # get real labels（must exsist in .obs）
        if cell_type_map !=None:
            celltype_str_list = np.array(adata.obs["celltype"]).tolist()
            current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
            current_celltype_labels = np.array(current_celltype_labels)
            adata.obs["celltype_labels"] = current_celltype_labels                
            labels = adata.obs["celltype_labels"]
        else:
            labels = adata.obs["celltype"]

        sns.scatterplot(
            x=X_umap[:n_cells, 0],
            y=X_umap[:n_cells, 1],
            hue=labels,
            palette="tab20",
            s=5,
            linewidth=0,
            legend=True
        )
 
        # prototypes
        plt.scatter(
            X_umap[n_cells:, 0],
            X_umap[n_cells:, 1],
            edgecolors='black',    
            facecolors='none',    
            s=60,
            marker='X',
            label='Prototypes',
        )
        
        plt.title("t={}".format(test_batch_idx), fontsize=24)
        plt.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
        # plt.xlabel("UMAP1")
        # plt.ylabel("UMAP2")
        plt.xticks([]) 
        plt.yticks([])  
        plt.margins(0)       
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
        plt.xticks([])  
        plt.yticks([])  
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
            gene_ids,
            max_len=3001,                 
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=-2,
            append_cls=True,
            include_zero_gene=True,
        )
        all_gene_ids, all_values = tokenized_all["genes"].to(self.device), tokenized_all["values"].to(self.device)
        src_key_padding_mask = all_gene_ids.eq(self.vocab["<pad>"])
        batch_ids = torch.tensor(adata.obs["batch_id"].values).to(self.device)
        batch_ids = torch.zeros_like(batch_ids)                    # only one BN layer is trained, so the batch_ids need to be set to 0.

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config["amp"]):
            if self.config["adapter"] or self.config["loramoe"] or self.config["lora"]:
                cell_embeddings = self.model.encode_batch(
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
        # prototypes（shape [n_prototypes, 1200]）
        prototypes_np = prototypes.detach().cpu().numpy() if torch.is_tensor(prototypes) else prototypes
        n_prototypes = prototypes_np.shape[0]

        X_all = np.concatenate([X_cells.detach().cpu().numpy(), prototypes_np], axis=0)

        umap_model = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=42)
        X_umap = umap_model.fit_transform(X_all)  # [n_cells + n_prototypes, 2]

        # labels = adata.obs["celltype"]
        if cell_type_map !=None:
            celltype_str_list = np.array(adata.obs["celltype"]).tolist()
            current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
            current_celltype_labels = np.array(current_celltype_labels)
            adata.obs["celltype_labels"] = current_celltype_labels             
            labels = adata.obs["celltype_labels"]
        else:
            labels = adata.obs["celltype"]

        if self.config["dataset_name"] == "myeloid":
            labels_batch = adata.obs["str_batch"].astype(str).str.split("_").str[1]
        else:
            labels_batch = adata.obs["str_batch"]
        
        ax1 = plt.subplot(2, max_batch_idx, test_batch_idx+1)
        sc1 = sns.scatterplot(
            x=X_umap[:n_cells, 0],
            y=X_umap[:n_cells, 1],
            hue=labels,
            palette="tab20",
            s=30000/n_cells,
            linewidth=0,
            legend=False,
            ax=ax1
        )

        # prototypes
        # plt.scatter(
        #     X_umap[n_cells:, 0],
        #     X_umap[n_cells:, 1],
        #     edgecolors='black',     
        #     facecolors='none',     
        #     s=60,
        #     marker='X',
        #     label='Prototypes'
        # )
        plt.title("t={}".format(test_batch_idx), fontsize=24)
        
        # plt.xlabel("UMAP1")
        # plt.ylabel("UMAP2")
        plt.xticks([])  
        plt.yticks([])  

        ax2 = plt.subplot(2, max_batch_idx, max_batch_idx + test_batch_idx + 1)  
        sc2 = sns.scatterplot(
            x=X_umap[:n_cells, 0],
            y=X_umap[:n_cells, 1],
            hue=labels_batch,
            palette="tab10",
            s=30000/n_cells,
            linewidth=0,
            legend=False,    
            ax=ax2
        )
        plt.xticks([]); plt.yticks([])

        return sc1, sc2, labels, labels_batch

    def forward_for_ig(self, input_gene_ids, input_values, src_key_padding_mask):

        input_gene_ids = input_gene_ids.long()
        input_values = input_values.float()
        output_dict, _ = self.model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=src_key_padding_mask,
            batch_labels=None,  
            CLS=True,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False,
        )

        cell_emb = output_dict["cell_emb"] 
        scalar_output = cell_emb.norm(dim=1)  
        return scalar_output
    
    def forward_latent_with_ig(self, adata_test, save_dir, test_batch_idx, cell_type_map):
        from plot_prototype.plot_clusters import plot_eval_cell_emb
        """
        Add Integrated Gradients (IG) attribution computation on top of forward_latent
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
            batch_size=1,
            shuffle=False,
            intra_domain_shuffle=True,
            drop_last=False,
            per_seq_batch_sample=False
        )
        
        self.model.eval()
        all_attr = []
        all_label = []

        num_batches = len(test_loader)
        # with torch.no_grad():
        for batch, batch_data in enumerate(itertools.islice(test_loader, num_batches)):
            input_values = batch_data["values"].to(self.device).float()
            input_values.requires_grad_(True)
            input_gene_ids = batch_data["gene_ids"].to(self.device)
            celltype_labels = batch_data["celltype_labels"]
            src_key_padding_mask = input_gene_ids.eq(self.vocab["<pad>"])

            # baseline can be all zeros
            baseline_values = torch.zeros_like(input_values).to(self.device)

            n_steps = 10  # IG steps
            alphas = torch.linspace(0, 1, n_steps).to(self.device)

            grads = torch.zeros_like(input_values)
            with torch.cuda.amp.autocast(enabled=self.config["amp"]):
                for alpha in alphas:
                  
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
                    
                    # scalar output (here use the norm of the cell embedding)
                    cell_emb = output_dict["cell_emb"]  # [B, dim]
                    scalar_output = cell_emb.norm(dim=1)  # scalar

                    grad = torch.autograd.grad(
                        outputs=scalar_output,
                        inputs=interpolated,
                        grad_outputs=torch.ones_like(scalar_output),
                        create_graph=False,
                        retain_graph=True,
                        only_inputs=True
                    )[0]

                    grads += grad

            # approximate integration
            attr = (input_values - baseline_values) * grads / n_steps   # (1, 1201)
            all_attr.append(attr.cpu().detach())
            all_label.append(celltype_labels)
        all_attr = torch.cat(all_attr)
        plot_eval_cell_emb(save_dir, all_attr, all_label, cell_type_map, save_name = f'clustermap_Cell_Embeddings_attr_genes_testbatch{test_batch_idx}.png')
           
        return all_attr  # list of attribution scores for each batch

    def predict_confidence(self, adata_test, save_dir, gene_ids, test_batch_idx, cell_type_map):
        from tutorials.prototype_analysis import compute_confidence_myeloid, prototype_dist_correlation
        Times = 5
        ########################## compute confidence and save them ##########################
        # if self.config["dataset_name"] == "pancreas" or self.config["dataset_name"] == "pancreas_filter3":
        #     compute_confidence_myeloid(Times, self.model, self.config, self.device, gene_ids, self.vocab, adata_test, save_dir, 
        #                         test_batch_idx, cell_type_map, prototype_file = f"prototype_{5}.pkl")
        # if self.config["dataset_name"] == "myeloid" or self.config["dataset_name"] == "myeloid_filter3":
        #     compute_confidence_myeloid(Times, self.model, self.config, self.device, gene_ids, self.vocab, adata_test, save_dir, 
        #                         test_batch_idx, cell_type_map, prototype_file = f"prototype_{4}.pkl")
        prototype_dist_correlation(save_dir, adata_test, cell_type_map)

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
        # compute cell types and labels for current batch (test_batch_idx)
        # current_label_dict, current_celltype_labels = np.unique(
        #     np.array(adata_test.obs["celltype"].tolist()), return_inverse=True
        # )
        
        celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels                   # temporarily save
        batch_ids = adata_test.obs["batch_id"].tolist()
        batch_ids = np.array(batch_ids)

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
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
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
                            batch_labels=torch.from_numpy(np.array([test_batch_idx])).long() if self.config["DSBN"] else None,
                            CLS=True,
                            CCE=False,
                            MVC=False,
                            ECS=False,
                            do_sample=False,
                        )
                    output_values = output_dict["cls_output"]
                    loss = self.criterion_cls(output_values, celltype_labels)
                    # cell_emb = output_dict["cell_emb"]
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
                
            ############## Plot confusion matrix ###################
            
            class_names = np.unique(np.concatenate((labellist, predictions)))  # get all classes
            pred_counts = pd.Series(labellist).value_counts()
            pred_counts = pred_counts.reindex(class_names, fill_value=0)   # keep order consistent
            
            palette = sns.color_palette("tab20", len(class_names))
            celltype_colors = dict(zip(class_names, palette))
            
            x_positions = np.arange(len(class_names))
            bar_colors = [celltype_colors[ct] for ct in class_names]

            fig, ax = plt.subplots(figsize=(10, 2))

            # ----------- draw bar chart ----------
            ax.bar(
                x_positions,
                pred_counts.values,
                width=1.0,
                color=bar_colors,
                alpha=0.9
            )

            # y-axis label
            ax.set_ylabel("Count", fontsize=12)

            # x-axis labels show class names
            ax.set_xticks(x_positions)
            ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=10)

            # ----------- add numeric annotations ----------
            for i, count in enumerate(pred_counts.values):
                ax.text(
                    i,
                    count + pred_counts.max() * 0.03,
                    str(int(count)),
                    ha='center',
                    va='bottom',
                    fontsize=12
                )

            # beautify
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(f"{save_dir}/predict_test_batch_{test_batch_idx}_Confusion_Matrix_with_counts.png",
                dpi=300)
            ############################## heatmap confusion matrix ##############################
            
            cm = confusion_matrix(labellist, predictions, labels=class_names)
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm = np.round(cm, 1)  # round each element to one decimal place
            cm = pd.DataFrame(cm, index=class_names, columns=class_names)
            # 绘制热力图

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(cm, annot=True, fmt=".1f", cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 12})  # 调小字体)

            # plt.xlabel('Predicted Labels')
            # plt.ylabel('True Labels')
            # plt.title('Confusion Matrix')
            # plt.tight_layout()
            # plt.savefig(str(save_dir) + "/" + f"predict_test_batch_{test_batch_idx}_Confusion_Matrix.png", dpi=300)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # ====== draw color bars on X/Y axes ======
            # x-axis color bar
            for i, ct in enumerate(class_names):
                ax.add_patch(plt.Rectangle((i, cm.shape[0]), 1, 0.2, 
                                        color=celltype_colors[ct], clip_on=False, transform=ax.transData))

            # y-axis color bar
            for i, ct in enumerate(class_names):
                ax.add_patch(plt.Rectangle((-0.2, i), 0.2, 1, 
                                        color=celltype_colors[ct], clip_on=False, transform=ax.transData))

            # ====== add legend ======
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


        return gene_ids
    
    def plot_train_val_loss(self, train_loss_list_all, val_loss_list_all, save_dir, test_batch_idx):
        epochs_train = range(1, len(train_loss_list_all) + 1)
        epochs_val = range(1, len(val_loss_list_all) + 1)

        plt.figure(figsize=(10, 8))

        # Subplot 1: Training Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs_train, train_loss_list_all, label='Train Loss', marker='o', color='blue')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.grid(True)
        plt.tight_layout()

        # Subplot 2: Validation Loss
        plt.subplot(2, 1, 2)
        plt.plot(epochs_val, val_loss_list_all, label='Validation Loss', marker='s', color='orange')
        plt.xlabel("Iter")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.grid(True)
        plt.tight_layout()

            # save figure
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

        # save figure
        plt.savefig(save_dir / f"Proto_Repultion_loss_batch{test_batch_idx}.png")

    def evaluate_all(self, adata_train, save_dir, test_batch_idx, cell_type_map, all_adata_test):
                # count number of each cell type        
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
        adata_train.obs["celltype_labels"] = current_celltype_labels                   # update current label indices to final labels
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
            # first split indices
            adata_indices = np.arange(len(adata_train))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=self.config["valid_ratio"],
                shuffle=True,
                # stratify=current_celltype_labels if not self.config["randomsplit"] else None,
            )

            # subset adata_train by indices
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

    def evaluate_predict(self, adata_test, save_dir, test_batch_idx, cell_type_map):
        self.config["weight_dir"] = save_dir
            # count number of each cell type
        # celltype_counts = adata_test.obs["celltype"].value_counts()
        # filter cell types with counts less than 4
        # valid_celltypes = celltype_counts[celltype_counts >= 2].index

        # # filter adata_test
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
        # compute cell types and labels for current batch (test_batch_idx)
        # current_label_dict, current_celltype_labels = np.unique(
        #     np.array(adata_test.obs["celltype"].tolist()), return_inverse=True
        # )
        celltype_str_list = np.array(adata_test.obs["celltype"]).tolist()
        current_celltype_labels = [cell_type_map[cell] for cell in celltype_str_list]
        current_celltype_labels = np.array(current_celltype_labels)
        adata_test.obs["celltype_labels"] = current_celltype_labels                  
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
                all_counts, current_celltype_labels, batch_ids, test_size=self.config["valid_ratio"], shuffle=True
            )
    
            adata_indices = np.arange(len(adata_test))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=self.config["valid_ratio"],
                shuffle=True,
                # stratify=current_celltype_labels if not self.config["randomsplit"] else None,
            )

            # use index to split adata_train
            adata_train_split = adata_test[train_idx].copy()
            adata_valid_split = adata_test[valid_idx].copy()
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
            adata_train_split = adata_test[train_indices].copy()
            adata_valid_split = adata_test[val_indices].copy()     

        else:
            (
                train_data,
                valid_data,
                train_celltype_labels,
                valid_celltype_labels,
                train_batch_labels,
                valid_batch_labels,
            ) = train_test_split(
                all_counts, current_celltype_labels, batch_ids, test_size=self.config["valid_ratio"], shuffle=True, stratify=current_celltype_labels
            )
        
            adata_indices = np.arange(len(adata_test))
            train_idx, valid_idx = train_test_split(
                adata_indices,
                test_size=self.config["valid_ratio"],
                shuffle=True,
                stratify=current_celltype_labels if not self.config["randomsplit"] else None,
            )

            adata_train_split = adata_test[train_idx].copy()
            adata_valid_split = adata_test[valid_idx].copy()
      
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

                with torch.cuda.amp.autocast(enabled=self.config["amp"]):
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


            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)

            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.savefig(str(save_dir) + "/" + f"evaluate_batch_{test_batch_idx}_Confusion_Matrix.png")
            # self.plot_clusters_prototypes(adata_test, self.model.cls_decoder.out_layer.weight.data, input_layer_key, gene_ids, save_dir)
        return adata_valid_split, gene_ids, cell_emb_list, labellist

def evaluate_predict(save_dir, cell_type_map):
    from plot_prototype.plot_clusters import plot_eval_cell_emb
    # config = init_wandb()
    with open(save_dir + "/config.json", "r", encoding="utf-8") as f:
        config = json.load(f) 
    if 'classifier' not in config:
        config['classifier'] = 'Linear'
    if "decrease_lr_all" not in config:
        config["decrease_lr_all"] = False
    if "adapter_dim" not in config:
        config["adapter_dim"] = 64
    if "num_of_expert" not in config:
        config["num_of_expert"] = 4
    if "valid_ratio" not in config:
        config["valid_ratio"] = 0.1
    set_seed(config["seed"])
    if config["dataset_name"] == "pancreas" or config["dataset_name"] == "pancreas_":
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6)]  # the remaining three batches used as test set
    elif config["dataset_name"] == "myeloid":
        data_paths = ["../data/myeloid/" + f"myeloid_batch{i}.h5ad" for i in [1,2,5,6,7]]

    model_dir = Path(config["load_model"])
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
    if config["dataset_name"] == "pancreas": 
        continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_5.pt")
    elif config["dataset_name"] == "myeloid":
        continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = "best_model_batch_4.pt")

    all_adata_list = []
    all_batch_results = {}
    all_cell_emb = []
    all_label = []
    fig, axs = plt.subplots(2, max_batch_idx, figsize=(5*max_batch_idx+2, 10), constrained_layout=True)    
    # fig.subplots_adjust(bottom=0.25)
    for test_batch_idx in range(len(data_paths)):
        adata_test = sc.read_h5ad(data_paths[test_batch_idx])
        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        # all_adata_test, gene_ids, cell_emb_batch, label_batch = continual_classify.evaluate_predict(
        #     adata_test, save_dir, test_batch_idx, cell_type_map
        # )
        # all_adata_test, gene_ids = continual_classify.evaluate_all(adata_test, save_dir, test_batch_idx, cell_type_map)
        # all_adata_list.append(all_adata_test)
        # all_cell_emb.extend(cell_emb_batch)
        # all_label.extend(label_batch)

        all_adata_list.append(adata_test)   # in plotting stage, plot all train + val data
        ################### plot the process ###########################
        with open(str(save_dir) + "/" + f"prototype_{test_batch_idx}.pkl", "rb") as f:
            proto = pickle.load(f)
        combined_adata_test = anndata.concat(
                    all_adata_list,
                    join="outer",
                    merge="unique",
                    label="batch",
                    index_unique=None
                )
    # combined_adata_test, gene_ids, cell_emb_batch, label_batch = continual_classify.evaluate_predict(
    #         combined_adata_test, save_dir, test_batch_idx + 1, cell_type_map
    #     )
        continual_classify.subplot_clusters_prototypes(combined_adata_test, proto, 
                            "X_binned", gene_ids, test_batch_idx, save_dir, max_batch_idx, save_name=f"trainval_batch_{test_batch_idx}_", cell_type_map=cell_type_map)
        
    # plt.legend(markerscale=3, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(save_dir) + "/" + config["dataset_name"] + f"_trainval_batch_all_" + f"umap_cluster_nolengend_new.png", bbox_inches='tight', edgecolor='black',dpi=300)
############################################################################
    # plot_eval_cell_emb(save_dir, all_cell_emb, all_label, cell_type_map)
    
    # combined_adata_test = anndata.concat(
    #     all_adata_list,
    #     join="outer",
    #     merge="unique",
    #     label="batch",
    #     index_unique=None
    # )
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
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in range(6)]  # 剩下三个batch作为测试集
        # data_paths = ["../data/PANCREAS/" + f"pancreas_batch{2}.h5ad", 
        #               "../data/PANCREAS/" + f"pancreas_batch{3}.h5ad"]
    elif config["dataset_name"] == "myeloid":
        data_paths = [f"../data/myeloid/myeloid_batch{i}.h5ad" for i in [1, 2, 5, 6, 7]]
    elif config["dataset_name"] == "pancreas_filter3":
        data_paths = ["../data/PANCREAS/filter3" + f"/pancreas_batch{i}_delete3_filtered.h5ad" for i in range(6)]
    elif config["dataset_name"] == "myeloid_filter3":
        data_paths = ["../data/myeloid//filter3" + f"/myeloid_batch{i}_delete3_filtered.h5ad" for i in [1, 2, 5, 6, 7]]
    model_dir = Path(config["load_model"])
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
    continual_classify = ContinualClassify(config, vocab, num_batch_types, max_batch_idx)

    all_adata_test = []
    all_batch_results = {}
    all_learning_rate = []
    for test_batch_idx in range(len(data_paths)):
        adata_train = sc.read_h5ad(data_paths[test_batch_idx])
        best_model, combined_adata_test, gene_ids, learning_rate = continual_classify.process_batch(
            adata_train, logger, save_dir, config["dataset_name"], config["experiment_name"], all_adata_test, all_batch_results,
            test_batch_idx
        )
        print('combined_adata_test.shape', combined_adata_test.shape)
        all_learning_rate.extend(learning_rate)
        
    plot_learning_rate(save_dir, all_learning_rate) 
    # torch.cuda.empty_cache()
    del combined_adata_test, best_model, learning_rate
    import gc
    gc.collect()
        ########################## Debug stage turned off ##################################
        # besteval_results, besteval_adata = continual_classify.best_model_evaluate(best_model, adata_t=combined_adata_test, gene_ids=gene_ids,
        #                                         input_layer_key={
        #                                             "normed_raw": "X_normed",
        #                                             "log1p": "X_normed",
        #                                             "binned": "X_binned",
        #                                         }["binned"],test_batch_idx=test_batch_idx)
        
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
    
def predict(save_dir, cell_type_map, test_batch_list=None, modeldict_name = "best_model_batch_5.pt"):
    with open(save_dir + "/config.json", "r", encoding="utf-8") as f:
        config = json.load(f) 
    if 'classifier' not in config:
        config['classifier'] = 'Linear'
    if "decrease_lr_all" not in config:
        config["decrease_lr_all"] = False
    if "adapter_dim" not in config:
        config["adapter_dim"] = 64
    if "num_of_expert" not in config:
        config["num_of_expert"] = 4
    # config = init_wandb()
    set_seed(config["seed"])
    
    # The filtered data contains all test batches (including outlier cells and known types) + all reference data (filter3 dataset)
    if config["dataset_name"] == "pancreas":
         # the remaining three batches used as test set; for `pancreas_filter3` experiments this is used only for outlier detection distance plotting
        data_paths = ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in test_batch_list] 
    elif config["dataset_name"] == "pancreas_filter3":
        ################################## For `pancreas_filter3`, this code is used only for outlier query mapping #####################
        data_paths = ["../data/PANCREAS/filter3" + f"/pancreas_batch{i}_delete3_filtered.h5ad" for i in range(6)] + \
            ["../data/PANCREAS/" + f"pancreas_batch{i}.h5ad" for i in test_batch_list]
    elif config["dataset_name"] == "myeloid":
        data_paths = [f"../data/myeloid/" + f"myeloid_batch{i}.h5ad" for i in test_batch_list]
    elif config["dataset_name"] == "myeloid_filter3":
        
        data_paths = ["../data/myeloid/filter3/" + f"myeloid_batch{i}_delete3_filtered.h5ad" for i in [1, 2, 5, 6, 7]] +\
            [f"../data/myeloid/" + f"myeloid_batch{i}.h5ad" for i in test_batch_list]
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
    continual_classify = ContinualClassify(config, vocab, num_batch_types, modeldict_name = modeldict_name)

    all_adata_test = []
    all_batch_results = {}
    if config["dataset_name"] == "myeloid" or config["dataset_name"] == "myeloid_filter3":
        with open(str(save_dir) + "/" + "prototype_4.pkl", "rb") as f:
            old_proto = pickle.load(f)
    elif config["dataset_name"] == "pancreas" or config["dataset_name"] == "pancreas_filter3":
        with open(str(save_dir) + "/" + "prototype_5.pkl", "rb") as f:
            old_proto = pickle.load(f)
            
    if config["dataset_name"] == "pancreas_filter3":
        cell_type_map["macrophage"] = 11
        cell_type_map["mast"] = 12
        cell_type_map["quiescent_stellate"] = 13
    elif config["dataset_name"] == "myeloid_filter3":
        cell_type_map["pDC_LILRA4"] = 9
        cell_type_map["cDC3_LAMP3"] = 10
        cell_type_map["Macro_SPP1"] = 11
        
    for test_batch_idx in range(len(data_paths)):
        adata_test = sc.read_h5ad(data_paths[test_batch_idx])
        # gene_ids = continual_classify.predict(
        #     adata_test, save_dir, test_batch_idx, cell_type_map
        # )
        genes = adata_test.var["gene_name"].tolist()
        gene_ids = np.array(vocab(genes), dtype=int)
        ###################### compute gene importance #####################
        # gene_ids = continual_classify.forward_latent_with_ig(adata_test, save_dir, test_batch_idx, cell_type_map)
        ##########################################
        all_adata_test.append(adata_test)

        combined_adata_test = anndata.concat(
            all_adata_test,
            join="outer",
            merge="unique",
            label="batch",
            index_unique=None
        )
    # continual_classify.plot_gray_batch(combined_adata_test, old_proto, 
    #                          "X_binned", gene_ids, test_batch_idx, 
    #                          save_dir, save_name=f"batch_test_{test_batch_idx}", 
    #                          cell_type_map=cell_type_map, legend_on=False, experiment = "query_mapping")
    continual_classify.plot_gray_batch(combined_adata_test, old_proto, 
                             "X_binned", gene_ids, test_batch_idx, 
                             save_dir, save_name=f"batch_test_{test_batch_idx}", 
                             cell_type_map=cell_type_map, legend_on=False, experiment = "outlier_detection")
    # continual_classify.plot_gray_batch(combined_adata_test, old_proto, 
    #                          "X_binned", gene_ids, test_batch_idx, 
    #                          save_dir, save_name=f"batch_test_{test_batch_idx}", cell_type_map=cell_type_map, legend_on=True)
    
    # continual_classify.plot_feature_umap(combined_adata_test, gene_ids, test_batch_idx, save_dir, cell_type_map)
    
    # gene_ids = continual_classify.predict_confidence(
    #             combined_adata_test, save_dir, gene_ids,
    #             test_batch_idx = test_batch_idx + 1, cell_type_map = cell_type_map
    # )
    # gene_ids = continual_classify.predict(
    #         combined_adata_test, save_dir, test_batch_idx = test_batch_idx+1, cell_type_map = cell_type_map
    #     )

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
    # from captum.attr import IntegratedGradients
    # main()
    
    save_dir = "../save/dev_myeloid_filter3-Dec19-17-03-05"    # removed three cell types

    with open(save_dir + "/celltype_to_label.json", "r") as f:
        cell_type_map = json.load(f)
        f.close()
    # evaluate_predict(save_dir, cell_type_map)                                         # validation set
    # predict(save_dir, cell_type_map, test_batch_list=[6, 7, 8])                         # test set
    predict(save_dir, cell_type_map, test_batch_list = [0, 3, 4], modeldict_name = "best_model_batch_4.pt")                          # test set
    # predict(save_dir, cell_type_map, test_batch_list=[0, 1, 2, 3, 4, 5, 6, 7, 8])
