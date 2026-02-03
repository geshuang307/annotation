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

try:
    from flash_attn.flash_attention import FlashMHA

    flash_attn_available = True
except ImportError:
    import warnings

    warnings.warn("flash_attn is not installed")
    flash_attn_available = False

from dsbn import DomainSpecificBatchNorm1d
from grad_reverse import grad_reverse
# ========== 1. 定义 scGPT 持续学习环境 ========== #
class scGPTIntegrationEnv(gym.Env):
    def __init__(self, scgpt_model, dataset_loader, initial_state):
        super(scGPTIntegrationEnv, self).__init__()
        
        self.model = scgpt_model  # 预训练的 scGPT
        self.dataset_loader = dataset_loader  # 数据加载器
        self.state = initial_state  # 当前隐空间表示
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)                   # 4 个可调整参数
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(512,), dtype=np.float32)  # 512 维的隐空间

    def step(self, action):
        """执行 action 并返回新的 state、reward、done"""
        integration_weight, mask_ratio, learning_rate, regularization_strength = action
        
        # 更新 scGPT 训练参数
        self.model.set_training_params(mask_ratio, learning_rate, regularization_strength)
        
        # 获取新数据
        new_data = next(iter(self.dataset_loader))  
        new_state_embedding = self.model.get_latent_representation(new_data)  # 计算新数据的隐空间

        # 计算 KL 散度（衡量新数据与旧数据的差异）
        kl_div = self.compute_kl_divergence(self.state, new_state_embedding)

        # 计算奖励
        reward = -kl_div  # 目标是最小化 KL，使新数据更好地整合
        
        # 更新状态
        self.state = integration_weight * new_state_embedding + (1 - integration_weight) * self.state
        
        return self.state, reward, False, {}

    def reset(self):
        """重置环境"""
        self.state = self.model.get_initial_latent_representation()
        return self.state
    
    def compute_kl_divergence(self, old_state, new_state):
        """计算 KL 散度"""
        return torch.nn.functional.kl_div(old_state.log(), new_state, reduction="batchmean").item()

# ========== 2. 定义 CPPO 代理 ========== #
class CPPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CPPOAgent, self).__init__()
        
        # Actor 网络
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Critic 网络
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # 标准差参数

    def forward(self, state):
        """计算动作均值和状态值"""
        action_mean = self.actor(state)
        action_std = self.log_std.exp()
        value = self.critic(state)
        return action_mean, action_std, value

    def get_action(self, state):
        """采样动作"""
        action_mean, action_std, _ = self.forward(state)
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

# ========== 3. 训练 CPPO 代理 ========== #
def train_ppo(scgpt_model, dataset_loader, num_episodes=500):
    env = scGPTIntegrationEnv(scgpt_model, dataset_loader, scgpt_model.get_initial_latent_representation())
    agent = CPPOAgent(state_dim=512, action_dim=4)  # 4 维 action: (integration_weight, mask_ratio, learning_rate, regularization_strength)
    
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)
    gamma = 0.99  # 折扣因子
    clip_ratio = 0.2  # PPO 约束
    value_loss_coef = 0.5  # 价值网络损失权重
    entropy_coef = 0.01  # 熵损失权重

    for episode in range(num_episodes):
        state = env.reset()
        log_probs, values, rewards = [], [], []

        for _ in range(50):  # 每个 episode 采样 50 个 step
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, log_prob = agent.get_action(state_tensor)
            
            next_state, reward, done, _ = env.step(action.detach().numpy())
            
            log_probs.append(log_prob)
            values.append(agent.critic(state_tensor))
            rewards.append(reward)
            
            state = next_state
        
        # 计算奖励折扣回报
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        values = torch.stack(values).squeeze()
        log_probs = torch.stack(log_probs)

        # 计算 PPO 损失
        advantages = returns - values.detach()
        ratio = torch.exp(log_probs - log_probs.detach())  # 计算重要性采样比率
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()  # 目标是最大化收益，因此取负号

        # 计算价值损失
        value_loss = value_loss_coef * (returns - values).pow(2).mean()

        # 计算熵损失（鼓励探索）
        entropy_loss = -entropy_coef * (torch.exp(log_probs) * log_probs).mean()

        # 总损失
        loss = policy_loss + value_loss + entropy_loss
        
        # 更新策略网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Loss: {loss.item()}, Reward: {np.mean(rewards)}")


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)

# ========== 4. 运行训练 ========== #
if __name__ == "__main__":
    from scgpt.model import TransformerModel  # 假设 scGPT 作为模型库
    from scgpt.model.data_loader import load_data_and_model, prepare_data, prepare_dataloader # 数据加载函数
    from scgpt.model import TransformerModel
    from config_all import param


    sc.set_figure_params(figsize=(4, 4))
    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings('ignore')

    # dataset_name = "pancreas"
    dataset_name = "PBMC_10K"
    experiment_name = "fine_tune_onpancreas"
    load_model = "/workspace/geshuang/code/scGPT/save/scGPT_human"
    # weight_dir = '/workspace/geshuang/code/scGPT/save/dev_PBMC_10K-Feb26-19-34'     # 修改模型加载路径
    # weight_dir = '/workspace/geshuang/code/scGPT/save/dev_PBMC_10K-Feb28-09-27'
    # weight_dir = '/workspace/geshuang/code/scGPT/tutorials/save/dev_PBMC_10K-Feb19-19-34'
    weight_dir = '/workspace/geshuang/code/scGPT/tutorials/save/dev_pancreas-Feb25-17-11'
    # weight_dir = "/workspace/geshuang/code/scGPT/save/scGPT_human"
    fisher_path = './fisher/fisher_optpar_(reverse)pancreas.pth'      
    # fisher_path = None          
    batch_size = 32       # 在计算fisher information的时候，batch_size=1
    epochs = 15           # 在计算fisher information的时候，epochs=1
    hyperparameter_defaults, config = param(dataset_name, load_model, weight_dir, batch_size, epochs)
    
    set_seed(hyperparameter_defaults['seed'])
    # settings for input and preprocessing
    pad_token = "<pad>"
    # special_tokens = [pad_token, "<cls>", "<eoc>"]
    # mask_ratio = 0.4
    # mask_value = -1
    pad_value = -2
    n_input_bins = 51
    
    # n_hvg = 1200  # number of highly variable genes
    # max_seq_len = n_hvg + 1
    per_seq_batch_sample = True
    # DSBN = True  # Domain-spec batchnorm
    explicit_zero_prob = True  # whether explicit bernoulli for zeros
    
    ############### datasets loading ###############
    save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scg.logger
    scg.utils.add_file_handler(logger, save_dir / "run.log")
    logger.info(hyperparameter_defaults)
    
    adata, adata_sorted, embsize, nhead, d_hid, nlayers, vocab, num_batch_types, model_file, \
    logger, save_dir, tokenized_train, tokenized_valid, train_batch_labels, valid_batch_labels, \
    input_layer_key,gene_ids, sample_size = load_data_and_model(hyperparameter_defaults)

    # scgpt_model = scGPT.load_pretrained_model("scGPT_pretrained.pth")  # 加载预训练模型
    # dataset_loader = get_dataset_loader("new_scRNAseq_data.h5ad")  # 加载新数据
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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
    
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config['batch_size'],
        shuffle=True,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config['batch_size'],
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    train_ppo(model, train_loader)
