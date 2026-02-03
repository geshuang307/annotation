import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from torch.distributions import Normal
from typing import Dict, Mapping, Optional, Tuple, Any, Union

import torch
import numpy as np
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
import torch.autograd as autograd
from torch.autograd import Variable

try:
    from flash_attn.flash_attention import FlashMHA

    flash_attn_available = True
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    flash_attn_available = False

from dsbn import DomainSpecificBatchNorm1d
from grad_reverse import grad_reverse
import random
import itertools

seed = 2025
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)

        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss
    
class EWC(nn.Module):
    def __init__(self,device, fisher_path=None):
        super().__init__()
        # self.args = args
        self.ewc_lambda = 4
        self.tasks_encountered = [1]
        self.device = device
        self.fisher = {}
        self.optpar = {}
        if fisher_path is not None:
            self.checkpoint = torch.load(fisher_path, map_location='cpu')  # 加载到 CPU
            self.fisher = self.checkpoint['fisher']
            self.optpar = self.checkpoint['optpar']

    def forward(self, named_params):
        net_loss = Variable(torch.Tensor([0])).to(self.device)
        if not self.ewc_lambda:
            return net_loss
        for task_id in self.tasks_encountered:
            ############# 根据action确定哪些层的params要进行ewc正则化 #############
            for name, param in named_params:
                if "batch_encoder" not in name and "grad_reverse_discriminator.out_layer" not in name:
                    fisher = Variable(self.fisher[task_id][name]).to(self.device) # 400,1024   ; 400
                    optpar = Variable(self.optpar[task_id][name]).to(self.device) # 维度同上
                    # nan_mask = torch.isnan(fisher)  # 生成 NaN 掩码
                    # fill_value = torch.tensor(0.0, device=self.device)  # 让 0 也在正确的 device 上
                    # fisher[nan_mask] = fill_value
                    # optpar[nan_mask] = fill_value
                    net_loss += (fisher * (optpar - param).pow(2)).sum() 
        return net_loss * self.ewc_lambda/2

    def regularize(self, named_params):
        """Calculate the EWC regularization component in the overall loss.
        For all the tasks encountered in past, L2-norm loss is calculated
        between current model parameters and optimal parameters of previous
        tasks, weighted by terms from fisher matrix.

        Arguments
        =========
        named_params : generator
            Named parameters of model to be regularized.
        """
        return self.forward(named_params)
    
    def find_nan_inf(self, tensor):
        outliers = torch.isnan(tensor) | torch.isinf(tensor)  # 查找NaN或Inf
        return outliers
    
    # Update the Fisher Information
    # def init_fisher_optpar(self, model):
    def update_fisher_optpar(self, model, current_itr, data_loader, sample_size, device, config, vocab, DSBN, pad_token, mask_value, criterion,scaler, \
                             optimizer, dataset_name,\
                             batch_size=32, consolidate=True):
        if consolidate:
            if current_itr == 1:       # task 永远只有1个
                current_itr = 1
                self.tasks_encountered = [1]
            else:
                self.tasks_encountered.append(current_itr)
        
        # estimate the fisher information of the parameters.

        model.train()
        sample_grads = []

        iter_max = 2000
        iter = 0
        for batch, batch_data in enumerate(data_loader):
            model.zero_grad()
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
 
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
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
                )       # 只计算mask的部分

                grads = autograd.grad(scaler.scale(loss), model.parameters(), allow_unused=True, retain_graph=False)    # 193个网络层的参数
                iter += 1
                if iter >= iter_max:
                    break
                print(batch)
                sample_grads.append([g.detach().cpu() if g is not None else torch.zeros_like(p).detach().cpu() for g, p in zip(grads, model.parameters())])
        ############## 梯度反缩放和梯度裁剪 ##############
            # scaler.unscale_(optimizer)
            # with warnings.catch_warnings(record=True) as w:
            #     warnings.filterwarnings("always")
            #     torch.nn.utils.clip_grad_norm_(
            #         model.parameters(),
            #         1.0,
            #         error_if_nonfinite=False if scaler.is_enabled() else True,
            #     )
            #     if len(w) > 0:
            #         print(
            #             f"Found infinite gradient. This may be caused by the gradient "
            #             f"scaler. The current scale is {scaler.get_scale()}. This warning "
            #             "can be ignored if no longer occurs after autoscaling of the scaler."
            #         )
            # scaler.step(optimizer)
            # scaler.update()
        
        sample_grads = list(zip(*sample_grads))  

        # 计算每一层的 Fisher 对角矩阵
        fisher_diagonals = [(torch.stack(gs) ** 2).mean(0) for gs in sample_grads]

        for idx, tensor in enumerate(fisher_diagonals):
            # 找到异常值的位置
            outliers = self.find_nan_inf(tensor)
            print(f"Tensor {idx+1} - NaN or Inf values before replacement: {tensor[outliers]}")  # 输出异常值
            # 将异常值替换为0
            tensor = torch.where(outliers, torch.zeros_like(tensor), tensor)
            # 更新fisher_diagonals中的tensor
            fisher_diagonals[idx] = tensor
            print(f"Tensor {idx+1} after replacing NaN or Inf: {tensor}")

        fisher_diagonals_log = [torch.log(1 + F) for F in fisher_diagonals]
        for idx, tensor in enumerate(fisher_diagonals_log):
            outliers = self.find_nan_inf(tensor)
            print(f"Tensor {idx+1} - NaN or Inf values: {tensor[outliers]}")

        self.fisher[current_itr] = {}
        self.optpar[current_itr] = {}

        for (name, param), fisher in zip(model.named_parameters(), fisher_diagonals_log):
            self.optpar[current_itr][name] = param.data.detach().cpu()  # task1 学完之后网络的参数
            self.fisher[current_itr][name] = fisher.detach().cpu()     # 梯度计算的fisher信息
        torch.save({'fisher': self.fisher, 'optpar': self.optpar}, '/workspace/geshuang/code/scGPT/grads/grads_previous_' + dataset_name +'.pth')
        print('Fisher information and optimal parameters saved.')

# ========== 1. 定义 scGPT 持续学习环境 ========== #
class scGPTIntegrationEnv(gym.Env):
    def __init__(self, scgpt_model, dataset_loader_init, dataset_loader):
        super(scGPTIntegrationEnv, self).__init__()
        self.model = scgpt_model
        self.init_model = copy.deepcopy(scgpt_model)
        self.dataset_loader_init = dataset_loader_init
        self.dataset_loader = dataset_loader
        self.num_layers = len(list(self.model.parameters()))
        # self.state_dim = sum(p.numel() for p in self.model.parameters()) * 2  # 每层参数
        self.state_dim = 169 * 11
        # self.action_dim = self.num_layers
        self.action_dim = 169
        self.hidden_dim = 512
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32)
        self.current_itr = 1
        self.fisher = {}
        self.optpar = {}
        self.state = {}
        self.ewc_lambda = 40
    def find_nan_inf(self, tensor):
        outliers = torch.isnan(tensor) | torch.isinf(tensor)  # 查找NaN或Inf
        return outliers
    
    def compute_gradient_features(self, gradients):
        """
        Compute statistical features for a given layer's gradients across a batch of samples.

        Args:
            gradients (list of torch.Tensor): A list of gradient tensors from multiple samples.

        Returns:
            np.array: Feature vector containing statistical features of the gradient batch.
        """
        if gradients is None or gradients.numel() == 0 or torch.all(gradients == 0): # torch.Size([2, 60697, 512]) 
            return np.zeros(11)  # 若该层所有样本的梯度都为空，则返回零向量

        # 将所有样本的梯度转换为 NumPy 数组
        # grad_list = [g.detach().cpu().numpy().flatten() for g in gradients if g is not None]
        
        # if len(grad_list) == 0:
        #     return np.zeros(11)

        # grad_batch = np.stack(grad_list, axis=0)  # 形状：(batch_size, num_params)
        grad_batch = gradients.detach().cpu().numpy().flatten()  
        # 统计特征（跨 batch 计算）
        mean = np.mean(grad_batch)
        variance = np.var(grad_batch)
        median = np.median(grad_batch)
        max_val = np.max(grad_batch)
        min_val = np.min(grad_batch)
        range_val = max_val - min_val
        skewness = np.mean((grad_batch - mean) ** 3) / (np.std(grad_batch) ** 3 + 1e-8)  # 避免除零 偏度（Skewness）和峰度（Kurtosis）
        kurtosis = np.mean((grad_batch - mean) ** 4) / (np.std(grad_batch) ** 4 + 1e-8)

        # 稀疏性特征
        l1_norm = np.sum(np.abs(grad_batch)) / grad_batch.shape[0]  # 取 batch 均值
        l2_norm = np.sqrt(np.sum(grad_batch ** 2)) / grad_batch.shape[0]
        sparsity_rate = np.sum(grad_batch == 0) / grad_batch.size  # 计算梯度为 0 的比例

        return np.array([
            mean, variance, median, max_val, min_val, range_val,
            skewness, kurtosis, l1_norm, l2_norm, sparsity_rate
        ])

    def get_state(self):
        self.gradient_features = []
        # state[1] = {}
        for name, param in self.model.named_parameters():
            # state.extend(param.data.view(-1).cpu().numpy())               # 当前层的参数
            # state.extend(param.data.cpu().numpy()) 
            if param.grad is not None and param.shape[-1] == 512 and "batch_encoder" not in name and "grad_reverse_discriminator.out_layer" not in name:
                # state.extend(param.grad.view(-1).cpu().numpy())
                self.gradient_features.append(self.compute_gradient_features(param.grad))
                # state[1][name] = param.grad.cpu().numpy()                   # 当前层的梯度（求完loss才有梯度）
            elif param.grad is None and param.shape[-1] == 512 and "batch_encoder" not in name and "grad_reverse_discriminator.out_layer" not in name:
                self.gradient_features.append(np.zeros(11))                       # 用于计算张量（Tensor）中的总元素个数
            else:
                pass
        state = np.stack(self.gradient_features).reshape(-1)
        return state

    def step(self, actions, episode):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])  
        """ 逐层根据 actions 决策是否应用 Fisher """
        net_loss = Variable(torch.Tensor([0])).to(device)
        loss = Variable(torch.Tensor([0])).to(device)
        iter = 0
        if not self.ewc_lambda:
            return net_loss
            ############# 根据action确定哪些层的params要进行ewc正则化 #############
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if "batch_encoder" not in name and "grad_reverse_discriminator.out_layer" not in name:
                    fisher = Variable(self.fisher[1][name]).to(device) # 400,1024   ; 400
                    optpar = Variable(self.optpar[1][name]).to(device) # 维度同上
                    nan_mask = torch.isnan(fisher)  # 生成 NaN 掩码
                    fill_value = torch.tensor(0.0, device=device)  # 让 0 也在正确的 device 上
                    fisher[nan_mask] = fill_value
                    optpar[nan_mask] = fill_value
                    if param.shape[-1] == 512 and actions[iter] > 0.5 and "batch_encoder" not in name and "grad_reverse_discriminator.out_layer" not in name:
                        net_loss += (fisher * (optpar - param).pow(2)).sum() 
                        iter += 1
            net_loss = net_loss * self.ewc_lambda/2

        self.model.train()
        print('episode:', episode)
        # data = next(iter(self.dataset_loader))
        num_batches = 2                         # 这里面选择的batch数目
        dataset_loader = SeqDataset.prepare_dataloader(
        train_data_pt,
        batch_size=64,
        shuffle=True,
        intra_domain_shuffle=True,
        drop_last=False,
    )
        # 获取 dataset_loader 的总批次数（假设 batch_size 已知）
        total_batches = len(dataset_loader)  

        # 随机选择 num_batches 个批次索引
        random_indices = sorted(random.sample(range(total_batches), num_batches))
        selected_batches = [batch for i, batch in 
                        enumerate(itertools.islice(dataset_loader, max(random_indices) + 1)) 
                        if i in random_indices]
        # 初始化存储变量
        # selected_batches = []
        # index = 0
        # 遍历数据加载器并提取指定索引的 batch
        # for i, data in enumerate(self.dataset_loader):
        #     if i in random_indices:
        #         index += 1
        for data in selected_batches:
            input_gene_ids = data["gene_ids"].to(device)
            input_values = data["values"].to(device)
            target_values = data["target_values"].to(device)
            batch_labels = data["batch_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            self.model.zero_grad()
            with torch.cuda.amp.autocast(enabled=config['amp']):
                output_dict = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                    MVC=config['GEPC'],
                    ECS=config['ecs_thres'] > 0,
                )
                masked_positions = input_values.eq(mask_value)
                output_dict["mlm_output"] = output_dict["mlm_output"].squeeze()
                loss = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss += net_loss.item()
                loss.backward()
                # scaler.scale(loss).backward()
                # print("Scaled Loss:", scaler.scale(loss).item())
                print('loss:', loss)
                # if index >= 10:
                #     break
                    ############## 梯度反缩放和梯度裁剪 ##############
                    # scaler.unscale_(optimizer)
                    # with warnings.catch_warnings(record=True) as w:
                    #     warnings.filterwarnings("always")
                    #     torch.nn.utils.clip_grad_norm_(
                    #         self.model.parameters(),
                    #         1.0,
                    #         error_if_nonfinite=False if scaler.is_enabled() else True,
                    #     )
                    #     if len(w) > 0:
                    #         print(
                    #             f"Found infinite gradient. This may be caused by the gradient "
                    #             f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    #             "can be ignored if no longer occurs after autoscaling of the scaler."
                    #         )
                    # scaler.step(optimizer)
                    # scaler.update()
        ################## 测试代码 ##################
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None:
        #         a = param.grad.cpu().numpy()
        reward = self.compute_reward()
        next_state = self.get_state()
        done = False
        if episode % 50 ==0 and episode != 0:
            torch.save(self.model.state_dict(), save_dir / f"model_e{episode}.pt")
        return next_state, reward, done, {}

    def reset(self):
        # 初始化参数和梯度
        # for param in self.model.parameters():
        #     param.grad = None
        # self.model.train()
        # # 计算初始梯度
        # data = next(iter(self.dataset_loader))
        # input_gene_ids = data["gene_ids"].to(device)
        # input_values = data["values"].to(device)
        # target_values = data["target_values"].to(device)
        # batch_labels = data["batch_labels"].to(device)
        # src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
        # model.zero_grad()
        # with torch.cuda.amp.autocast(enabled=config['amp']):
        #     output_dict = model(
        #         input_gene_ids,
        #         input_values,
        #         src_key_padding_mask=src_key_padding_mask,
        #         batch_labels=batch_labels if DSBN else None,
        #         MVC=config['GEPC'],
        #         ECS=config['ecs_thres'] > 0,
        #     )
        #     masked_positions = input_values.eq(mask_value)
        #     output_dict["mlm_output"] = output_dict["mlm_output"].squeeze()
        #     loss = criterion(
        #         output_dict["mlm_output"], target_values, masked_positions
        #     )
        #     loss.backward()
        # state = self.get_state()
        fisher_grad_path = '/workspace/geshuang/code/scGPT/scgpt/memory_v2/rl_fisher_optpar_grad_init_' + dataset_name +'v2.pth'
        self.checkpoint = torch.load(fisher_grad_path, map_location='cpu')  # 加载到 CPU
        self.fisher = self.checkpoint['fisher']
        self.optpar = self.checkpoint['optpar']
        self.state = self.checkpoint['grad']
        self.state = self.state.reshape(-1)  # 将输入数据展平

        return self.state
    
    def init(self, loader):
        # 初始化参数和梯度
        for name, param in self.init_model.named_parameters():
            if param.requires_grad:  # 只处理需要梯度的参数
                param.grad = None
        self.init_model.train()
        
        fisher_grad_path = '/workspace/geshuang/code/scGPT/scgpt/memory/rl_fisher_optpar_grad_init' + dataset_name +'.pth'
        if os.path.exists(fisher_grad_path):
            self.checkpoint = torch.load(fisher_grad_path, map_location='cpu')  # 加载到 CPU
            self.fisher = self.checkpoint['fisher']
            self.optpar = self.checkpoint['optpar']
            self.state = self.checkpoint['grad']
            self.gradient_features = []
            for (name, param) in model.named_parameters():
                if param.shape[-1] == 512 and "batch_encoder" not in name and "grad_reverse_discriminator.out_layer" not in name:
                    self.gradient_features.append(self.compute_gradient_features(self.state[1][name]))
            self.gradient_features = np.stack(self.gradient_features)     # 169 * 11 的特征向量, 169层，每层11个特征点
            torch.save({'fisher': self.fisher, 'optpar': self.optpar, 'grad': self.gradient_features}, '/workspace/geshuang/code/scGPT/scgpt/memory_v2/rl_fisher_optpar_grad_init_' + dataset_name +'v2.pth')
            print('Fisher information and optimal parameters saved.')
            # 计算初始梯度
            self.state = self.gradient_features
            return self.state
        
        kd_loss = DistillKL(1)
        num_batches = len(loader)
        sample_grads = []
        iter_max = 1000
        iter = 0
        for batch, batch_data in enumerate(loader):
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            # print('input_gene_ids',input_gene_ids)
            # print('input_values',input_values)
            # print('target_values',target_values)
            # print('batch_labels',batch_labels)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            # print(src_key_padding_mask)
            with torch.cuda.amp.autocast(enabled=config['amp']):
                # model.decoder.return_representation = True
                output_dict, transformer_output = self.init_model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                    MVC=config['GEPC'],
                    ECS=config['ecs_thres'] > 0,
                    return_interval_representations = True
                )
                
                output_dict1, transformer_output1 = self.init_model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                    MVC=config['GEPC'],
                    ECS=config['ecs_thres'] > 0,
                    return_interval_representations = True
                )  
                # loss_kd.backward()
                loss_kd = kd_loss(F.log_softmax(transformer_output + 1e-9, dim=-1), F.softmax(transformer_output1, dim=-1))
                # print("Scaled Loss:", scaler.scale(loss_kd).item())
                grads = autograd.grad(loss_kd, self.init_model.parameters(), allow_unused=True, retain_graph=False)    # 193个网络层的参数
                iter += 1
                if iter >= iter_max:
                    break
                print(batch)
                sample_grads.append([g.detach().cpu() if g is not None else torch.zeros_like(p).detach().cpu() for g, p in zip(grads, model.parameters())])

        sample_grads = list(zip(*sample_grads))                                           # 全部样本的 grads
        grad_state = [(torch.stack(gs).mean(0)) for gs in sample_grads]                 # 这里对全部样本求平均          
        # 计算每一层的 Fisher 对角矩阵
        fisher_diagonals = [(torch.stack(gs) ** 2).mean(0) for gs in sample_grads]

        for idx, tensor in enumerate(fisher_diagonals):
            # 找到异常值的位置
            outliers = self.find_nan_inf(tensor)
            print(f"Tensor {idx+1} - NaN or Inf values before replacement: {tensor[outliers]}")  # 输出异常值
            # 将异常值替换为0
            tensor = torch.where(outliers, torch.zeros_like(tensor), tensor)
            # 更新fisher_diagonals中的tensor
            fisher_diagonals[idx] = tensor
            print(f"Tensor {idx+1} after replacing NaN or Inf: {tensor}")
        
        for idx, tensor in enumerate(grad_state):
            # 找到异常值的位置
            outliers = self.find_nan_inf(tensor)
            # print(f"Tensor {idx+1} - NaN or Inf values before replacement: {tensor[outliers]}")  # 输出异常值
            # 将异常值替换为0
            tensor = torch.where(outliers, torch.zeros_like(tensor), tensor)
            # 更新grad_state中的tensor
            grad_state[idx] = tensor
            # print(f"Tensor {idx+1} after replacing NaN or Inf: {tensor}")


        # 取对数
        fisher_diagonals_log = [torch.log(1 + F) for F in fisher_diagonals]
        for idx, tensor in enumerate(fisher_diagonals_log):
            outliers = self.find_nan_inf(tensor)
            # print(f"Tensor {idx+1} - NaN or Inf values: {tensor[outliers]}")

        current_itr = 1
        self.fisher[current_itr] = {}
        self.optpar[current_itr] = {}
        # self.state[current_itr] = {}
        self.gradient_features = []
        for (name, param), fisher, grad in zip(model.named_parameters(), fisher_diagonals_log, grad_state):

            self.optpar[current_itr][name] = param.data.detach().cpu()  # task1 学完之后网络的参数
            self.fisher[current_itr][name] = fisher.detach().cpu()     # 梯度计算的fisher信息
            # self.state[current_itr][name] = grad.detach().cpu()             # 梯度计算的fisher信息
            if param.shape[-1] == 512 and "batch_encoder" not in name and "grad_reverse_discriminator.out_layer" not in name:
                self.gradient_features.append(self.compute_gradient_features(grad))
        self.gradient_features = np.stack(self.gradient_features)     # 169 * 11 的特征向量, 169层，每层11个特征点
        torch.save({'fisher': self.fisher, 'optpar': self.optpar, 'grad': self.gradient_features}, '/workspace/geshuang/code/scGPT/scgpt/memory_v2/rl_fisher_optpar_grad_init_' + dataset_name +'v2.pth')
        print('Fisher information and optimal parameters saved.')
        # torch.save('/workspace/geshuang/code/scGPT/scgpt/memory/t_1_cell_embeddings.pth')
        # state = self.get_state()                  # 
        
        # for param in self.model.parameters():
            # state.extend(param.data.view(-1).cpu().numpy())               # 当前层的参数
            # state.extend(param.data.cpu().numpy()) 
            # if param.grad is not None:
                # state.extend(param.grad.view(-1).cpu().numpy())
                # state.extend(param.grad.cpu().numpy())                      # 当前层的梯度（求完loss才有梯度）
            # else:
            #     state.extend(np.zeros(param.numel()))                       # 用于计算张量（Tensor）中的总元素个数
        self.state = self.gradient_features
        return self.state
    
    def get_init_embeddings(self):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config['amp']):
            cell_embeddings = self.init_model.encode_batch(
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
        
        torch.save(cell_embeddings, '/workspace/geshuang/code/scGPT/scgpt/memory_v2/t0_cell_embeddings.pth')
        return cell_embeddings
    
    def eval_testdata(self,
        model: nn.Module,
        adata_t: AnnData,
        include_types: List[str] = ["cls"],
    ) -> Optional[Dict]:
        """evaluate the model on test dataset of adata_t"""
        model.eval()

        # copy adata_t to avoid reuse previously computed results stored in adata_t
        adata_t = adata_t.copy()

        # all_counts = (
        #     adata_t.layers[input_layer_key].A
        #     if issparse(adata_t.layers[input_layer_key])
        #     else adata_t.layers[input_layer_key]
        # )

        # celltypes_labels = adata_t.obs["celltype"].tolist()
        # celltypes_labels = np.array(celltypes_labels)

        # batch_ids = adata_t.obs["batch_id"].tolist()
        # batch_ids = np.array(batch_ids)

        # # Evaluate cls cell embeddings
        if "cls" in include_types:
        #     logger.info("Evaluating cls cell embeddings")
        #     tokenized_all = tokenize_and_pad_batch(
        #         all_counts,
        #         gene_ids,
        #         max_len=max_seq_len,
        #         vocab=vocab,
        #         pad_token=pad_token,
        #         pad_value=pad_value,
        #         append_cls=True,  # append <cls> token at the beginning
        #         include_zero_gene=True,
        #     )
        #     all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
        #     src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])
        
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

        if len(include_types) == 1:
            return results, adata_t, cell_embeddings
        
    def compute_reward(self):
        '''
        reward: [torch.tensor(0.3444, device='cuda:1')]
        '''
        kl = DistillKL(1)
        # 这里需要实现全部数据集聚类NMI值的计算
        results, adata_t, cell_embeddings = self.eval_testdata(
                self.model,                                 # 基础模型
                adata_t=adata_sorted if per_seq_batch_sample else adata,
                include_types=["cls"],
            )
        avg_bio = results.get('avg_bio', 0.0)     # NMI,ARI,ASW
        avg_batch = results.get('avg_batch', 0.0) # graph_conn, ASW_label/batch
        avg_cluster = 0.5*avg_bio + 0.5*avg_batch
        self.t_cell_embeddings = torch.tensor(cell_embeddings).to(device)
        self.t0_cell_embeddings = self.get_init_embeddings()
        self.t0_cell_embeddings = torch.tensor(self.t0_cell_embeddings).to(device)
        if os.path.exists('/workspace/geshuang/code/scGPT/scgpt/memory_v2/t_1_cell_embeddings.pth'):
            self.t_1_cell_embeddings = torch.load('/workspace/geshuang/code/scGPT/scgpt/memory_v2/t_1_cell_embeddings.pth')
            kl_embeddings_t = kl(self.t_cell_embeddings, self.t_1_cell_embeddings.to(device))
            kl_embeddings_t0 = kl(self.t_cell_embeddings, self.t0_cell_embeddings)
            reward = avg_cluster - 0.1 * kl_embeddings_t - 0.1 * kl_embeddings_t0
        else:
            kl_embeddings_t0 = kl(self.t_cell_embeddings, self.t0_cell_embeddings)
            reward = avg_cluster - 0.1 * kl_embeddings_t0 

        torch.save(self.t_cell_embeddings.detach().cpu(), '/workspace/geshuang/code/scGPT/scgpt/memory_v2/t_1_cell_embeddings.pth')   # 11990*512
        # reward = torch.tensor(0.3444).to(device)
        return reward

# ========== 2. 定义 CPPO 代理 ========== #
class CPPOAgent(nn.Module):
    def __init__(self, model, state_dim, action_dim, hidden_dim, discrete_action=True):
        super(CPPOAgent, self).__init__()
        self.device = device  # 确保设备正确设置
        self.shared_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        ).to(self.device)  # 将网络移动到指定设备

        self.discrete_action = discrete_action

        # Actor 分支
        if self.discrete_action:
            self.actor_head = nn.Linear(hidden_dim, action_dim).to(self.device)  
        else:
            self.mean_layer = nn.Linear(hidden_dim, action_dim).to(self.device)
            self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))  # 确保 log_std 在同一设备上

        # Critic 分支
        self.critic = nn.Linear(hidden_dim, 1).to(self.device)
        self.base_model = model.to(self.device)  # 确保 base_model 也在同一设备上

    def forward(self, state):
        state = state.to(self.device).float()  # 确保输入数据在同一设备
        
        shared_features = self.shared_net(state)  # torch.Size([169, 11])
        value = self.critic(shared_features)

        if self.discrete_action:
            action_logits = self.actor_head(shared_features)
            action_probs = torch.sigmoid(action_logits)  
            return action_probs, value
        else:
            action_mean = self.mean_layer(shared_features)
            action_std = self.log_std.exp()
            return action_mean, action_std, value
        
    def get_action(self, state):
        """
        根据当前策略采样动作
        :param state: (batch_size, state_dim)
        :return: action, log_prob
        """
        state = torch.tensor(state).to(device)
        if self.discrete_action:
            action_probs, value = self.forward(state)
            dist = Bernoulli(action_probs)  # 每个动作是 0/1（例如选或不选）
        else:
            action_mean, action_std, value = self.forward(state)
            dist = Normal(action_mean, action_std)  # 生成高斯分布
        
        action = dist.sample()  # 采样动作
        log_prob = dist.log_prob(action).sum(dim=-1)  # 计算 log 概率
        return action, log_prob, value
    
    # def get_action(self, state):
    #     """ 逐层决策 Fisher 选择 """
    #     actions = {}
    #     log_probs = {}
    #     values = {}
    #     value_list = []
    #     log_probs_list = []
        
    #     # for name, param in self.base_model.named_parameters():
    #     #     state = states[1][name]
    #     #     if state.shape == (512,):
    #     #         print("The tensor shape is [512].")
    #     #         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, state_dim)
    #     #         action_prob, value = self.forward(state_tensor)
    #     #         dist = Bernoulli(action_prob)           # 基于概率分布的随机采样
    #     #         action = dist.sample()
    #     #         log_prob = dist.log_prob(action)        # 计算对数概率 logP(A=a)=alogp+(1−a)log(1−p)

    #     #         actions[name] = action.item()
    #     #         log_probs[name] = log_prob
    #     #         values[name] = value
    #     #         value_list.append(value)
    #     #         log_probs_list.append(log_prob)
    #     return actions, log_probs, value, value_list, log_probs_list  # 返回所有层的决策结果
class PPO:
    def __init__(
        self,
        model,
        state_dim,
        action_dim,
        hidden_dim,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        c1=1.0,
        c2=0.01,
        batch_size=64,
        n_epochs=1          
    ):
        self.actor_critic = CPPOAgent(model, state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1  # Value loss coefficient
        self.c2 = c2  # Entropy coefficient
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * (1 - dones[t]) - values[t]
            print(delta)
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            print(gae)
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages)
        returns = advantages + torch.tensor(values)                # 预测的状态价值 = 优势 + critic网络预测的state value
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, rewards, dones, next_state, episode):    
        # Convert to tensor
        states = torch.FloatTensor(states)            # maxstep, 169*11 
        actions = torch.FloatTensor(actions)          # maxstep, 169
        old_log_probs = torch.FloatTensor(old_log_probs)     # ?
        
        # Get values for all states
        with torch.no_grad():
            _, values = self.actor_critic(states)                                   # 当前状态输入到actor model和critic(预训练好的打分)model 得到当前输出（actor）和（value)
            _, next_value = self.actor_critic(torch.FloatTensor([next_state]))      
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(np.array([t.detach().cpu().numpy() for t in rewards]), values.detach().cpu().numpy(), 
                                            next_value.item(), dones)                  # 计算action的优势， returns = advantages + torch.tensor(values)
        if advantages.shape[0] <= 2:
            advantages = advantages.mean()
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for iter in range(self.n_epochs):               # 基于同一批采样的状态信息，获取多个action的分布，提高采样利用率
            # Get current policy distribution and values
            # action_mean, action_std, values = self.actor_critic(states)       # 当前状态输入到actor model和critic(预训练好的打分)model 得到当前输出（actor）和（value)
            action_probs, values = self.actor_critic(states)                    
            # dist = Normal(action_mean, action_std)                            # 输出分布
            # curr_log_probs = dist.log_prob(actions).sum(dim=-1)               # 分布中采样,这里可以是多个action的概率
            # entropy = dist.entropy().mean()
            dist = Bernoulli(action_probs)  # 每个动作是 0/1（例如选或不选）
        
            action = dist.sample()  # 采样动作
            curr_log_probs = dist.log_prob(action).sum(dim=-1).detach().cpu()  # 计算 log 概率
            entropy = dist.entropy().mean()
            
            # Compute ratio and surrogate loss
            ratio = (curr_log_probs - old_log_probs).exp()      # 采样很多个当下，跟过去的p比                        
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantages    
            
            # Compute actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()                         # Policy-gradient 的期望 
            values = values.squeeze().detach().cpu() 
            critic_loss = 0.5 * ((returns - values) ** 2).mean()       # 打分损失
            
            # Total loss
            loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy        
            
            if iter == self.n_epochs-1:
                print(f"Episode {episode}, Loss: {loss.item()}, Actor_loss:{actor_loss.item()}, critic_loss:{critic_loss.item()}, entropy:{entropy.item()}")
            # Update networks
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return actor_loss, critic_loss, entropy, returns
            
    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
        
    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))

# ========== 3. 训练 CPPO 代理 ========== #
def train_ppo(scgpt_model, dataset_loader_init, dataset_loader, num_episodes=1):      # 500
    env = scGPTIntegrationEnv(scgpt_model, dataset_loader_init, dataset_loader)
    agent = CPPOAgent(env.model, state_dim=env.state_dim, action_dim=env.action_dim, hidden_dim = env.hidden_dim)
    ppo = PPO(env.model, state_dim=env.state_dim, action_dim=env.action_dim, hidden_dim = env.hidden_dim, batch_size=config['batch_size'], n_epochs = 10)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    gamma = 0.99
    clip_ratio = 0.2
    value_loss_coef = 0.5
    entropy_coef = 0.01
    maxsteps = 1         # 50
    
    state = env.init(dataset_loader_init)
    all_rewards, all_values, all_returns, all_actions = [], [], [], []
    critic_loss_list, actor_loss_list, entropy_loss_list = [], [], []
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [],[]
        episode_reward = 0
        for t in range(maxsteps):
            # state_tensor = torch.tensor(state, dtype=torch.float32)
            action, log_prob, value  = agent.get_action(state)   # 获取全部层动作和对数概率的list
            next_state, reward, done, _ = env.step(action, episode)

            log_probs.append(log_prob.detach().item())
            values.append(value)
            rewards.append(reward)
            states.append(state)                              # 1859
            actions.append(action.detach().cpu().numpy())
            dones.append(done)
            episode_reward += reward

            state = next_state        # 这边跟state格式不一致，需要修改，有没有“1”这个维度
        
        # Update PPO agent
        actor_loss, critic_loss, entropy, returns = ppo.update(states, actions, log_probs, rewards, dones, next_state, episode)     # 再循环10次                
        
        episode_rewards.append(episode_reward)
        
        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
        all_rewards.extend(rewards)
        all_values.extend(values)
        all_returns.extend(returns)
        all_actions.extend(actions)
        # # 计算奖励折扣回报
        # returns = []
        # discounted_sum = 0
        # for r in reversed(rewards):
        #     discounted_sum = r + gamma * discounted_sum
        #     returns.insert(0, discounted_sum)
        # all_returns.extend(returns)
        # returns = torch.tensor(returns, dtype=torch.float32)
        # values = torch.stack(values).squeeze()
        # log_probs = torch.stack(log_probs)

        # # 计算 PPO 损失
        # advantages = returns - values.detach()
        # ratio = torch.exp(log_probs - log_probs.detach())
        # surr1 = ratio * advantages
        # surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        # policy_loss = - torch.min(surr1, surr2).mean()

        # # 计算价值损失
        # value_loss = value_loss_coef * (returns - values).pow(2).mean()

        # # 计算熵损失
        # entropy_loss = -entropy_coef * (torch.exp(log_probs) * log_probs).mean()

        # # 总损失
        # loss = policy_loss + value_loss + entropy_loss
        actor_loss_list.append(actor_loss)
        critic_loss_list.append(critic_loss)
        entropy_loss_list.append(entropy)

        # # 更新策略网络
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # if episode % 10 == 0:
        #     print(f"Episode {episode}, Loss: {loss.item()}, Policy loss:{policy_loss.item()}, value loss:{value_loss.item()}, entropy_loss:{entropy_loss.item()}, Reward: {torch.mean(torch.stack([t.detach() for t in rewards]))}")
    torch.save(agent.state_dict(), '/workspace/geshuang/code/scGPT/scgpt/memory_v2/ppo_agent.pth')
    return agent, actor_loss_list, critic_loss_list, entropy_loss_list, all_rewards, all_values, all_returns, all_actions

def plot_results(all_rewards, all_values, all_returns, all_actions, actor_loss_list, critic_loss_list, entropy_loss_list):
    rewards_ = [reward.detach().cpu().numpy() for reward in all_rewards]
    values_ = [value.detach().cpu().numpy() for value in all_values]
    returns_ = [returns.detach().cpu().numpy() for returns in all_returns]
    # actions_ = [actions.detach().cpu().numpy() for actions in all_actions]
    actor_loss_list = [actor_loss.detach().cpu().numpy() for actor_loss in actor_loss_list]
    critic_loss_list = [critic_loss.detach().cpu().numpy() for critic_loss in critic_loss_list]
    entropy_loss_list = [entropy_loss.detach().cpu().numpy() for entropy_loss in entropy_loss_list]

    plt.figure(figsize=(5, 5))
    plt.plot(rewards_, color='blue', label='Rewards')
    plt.plot(values_, color='red', label = 'Values')
    plt.plot(returns_, color='green', label = 'Returns')
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Reward_value_returns')
    plt.savefig('/workspace/geshuang/code/scGPT/scgpt/memory_v2/training_progress.png')
    plt.close()
    plt.figure(figsize=(5, 5))
    plt.plot(actor_loss_list, color='blue', label='Actor_loss')
    plt.plot(critic_loss_list, color='red', label = 'Critic_loss')
    plt.plot(entropy_loss_list, color='green', label = 'Entropy_loss')
    plt.title('Loss')
    plt.xlabel('Episode')
    plt.ylabel('Policy_loss_value_loss_entropy_loss')
    plt.savefig('/workspace/geshuang/code/scGPT/scgpt/memory_v2/loss.png')
    plt.close()
    results = {}
    results['rewards'] = rewards_
    results['values'] = values_
    results['returns'] = returns_
    results['actions'] = all_actions
    results['actor_loss_list'] = actor_loss_list
    results['critic_loss_list'] = critic_loss_list
    results['entropy_loss_list'] = entropy_loss_list

    # 保存字典
    with open("/workspace/geshuang/code/scGPT/scgpt/memory_v2/results.pkl", "wb") as f:
        pickle.dump(results, f)



# ========== 4. 运行训练 ========== #
if __name__ == "__main__":
    # from model import TransformerModel  # 假设 scGPT 作为模型库
    # from scgpt.model.data_loader import load_data_and_model, prepare_data, prepare_dataloader # 数据加载函数
    from loader import SeqDataset
    from rltransformer import TransformerModel
    from config_all import param
    import pickle
   

    sc.set_figure_params(figsize=(4, 4))
    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings('ignore')

    # dataset_name = "pancreas"
    dataset_name = "PBMC_10K"
    experiment_name = "fine_tune_on_scgpt"
    load_model = "/workspace/geshuang/code/scGPT/save/scGPT_human"
    # weight_dir = '/workspace/geshuang/code/scGPT/save/dev_PBMC_10K-Feb26-19-34'     # 修改模型加载路径
    # weight_dir = '/workspace/geshuang/code/scGPT/save/dev_PBMC_10K-Feb28-09-27'
    # weight_dir = '/workspace/geshuang/code/scGPT/tutorials/save/dev_PBMC_10K-Feb19-19-34'
    # weight_dir = '/workspace/geshuang/code/scGPT/tutorials/save/dev_pancreas-Feb25-17-11'
    weight_dir = "/workspace/geshuang/code/scGPT/save/scGPT_human"
    # fisher_path = './fisher/fisher_optpar_(reverse)pancreas.pth'      
    fisher_path = None          
    batch_size = 32       # 在计算fisher information的时候，batch_size=1
    epochs = 15           # 在计算fisher information的时候，epochs=1
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
    tokenized_train, tokenized_valid, train_batch_labels, valid_batch_labels, \
    input_layer_key,gene_ids, sample_size = SeqDataset.load_data_and_model(hyperparameter_defaults)
    # 全部的 counts value and gene_ids

    all_counts = (
            adata_sorted.layers[input_layer_key].A
            if issparse(adata_sorted.layers[input_layer_key])
            else adata_sorted.layers[input_layer_key]
        )

    celltypes_labels = adata_sorted.obs["celltype"].tolist()
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata_sorted.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)
    tokenized_all = tokenize_and_pad_batch(
        all_counts,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=True,
    )
    all_gene_ids, all_values = tokenized_all["genes"], tokenized_all["values"]
    src_key_padding_mask = all_gene_ids.eq(vocab[pad_token])

    # scgpt_model = scGPT.load_pretrained_model("scGPT_pretrained.pth")  # 加载预训练模型
    # dataset_loader = get_dataset_loader("new_scRNAseq_data.h5ad")  # 加载新数据
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
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
    print(model)
    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name} | Shape: {param.shape}")
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params}")         # 54.4 M
    train_data_pt, valid_data_pt = SeqDataset.prepare_data(tokenized_train, train_batch_labels, tokenized_valid, valid_batch_labels, sort_seq_batch=per_seq_batch_sample)
    train_loader_init = SeqDataset.prepare_dataloader(
        train_data_pt,
        batch_size=1,
        shuffle=True,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader_init = SeqDataset.prepare_dataloader(
        valid_data_pt,
        batch_size=1,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    train_loader = SeqDataset.prepare_dataloader(
        train_data_pt,
        batch_size=64,
        shuffle=True,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = SeqDataset.prepare_dataloader(
        valid_data_pt,
        batch_size=64,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    criterion = masked_mse_loss
    criterion_dab = nn.CrossEntropyLoss()
    
    scaler = torch.cuda.amp.GradScaler(enabled=config['amp'])  
    agent, actor_loss_list, critic_loss_list, entropy_loss_list, all_rewards, all_values, all_returns, all_actions = train_ppo(model, train_loader_init, train_loader)
    plot_results(all_rewards, all_values, all_returns, all_actions, actor_loss_list, critic_loss_list, entropy_loss_list)

