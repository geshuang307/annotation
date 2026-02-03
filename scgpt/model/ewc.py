import torch
from torch.autograd import Variable
from torch import nn
import numpy as np
import torch.autograd as autograd
import utils
from torch.nn import functional as F
import warnings
from backpack import backpack
from backpack.extensions import DiagHessian


USE_CUDA = torch.cuda.is_available()

def Variable(data, *args, **kwargs):
    var = autograd.Variable(data, *args, **kwargs)
    if USE_CUDA:
        var = var.cuda()
    return var


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
    def update_fisher_optpar(self, model, current_itr, data_loader, sample_size, device, config, vocab, DSBN, pad_token, mask_value, criterion,scaler, \
                             optimizer, dataset_name,\
                             batch_size=32, consolidate=True):
        if consolidate:
            if current_itr == 1:       # task 永远只有1个
                current_itr = 1
                self.tasks_encountered = [1]
            else:
                self.tasks_encountered.append(current_itr)
        
        # data_loader = utils.get_data_loader(dataset, batch_size)
        # losses = []
        # for x, y in data_loader:
        #     x = x.view(batch_size, -1)
        #     x = Variable(x).cuda() if USE_CUDA else Variable(x)
        #     y = Variable(y).cuda() if USE_CUDA else Variable(y)

        #     losses.append(
        #         F.log_softmax(model(x), dim=1)[range(batch_size), y.data]
        #     )                           # 每次计算batch_size个样本损失
        #     if len(losses) >= sample_size // batch_size:
        #         break
        # estimate the fisher information of the parameters.
        model.train()
        sample_grads = []
        # for param in model.parameters():
        #     param.requires_grad = False  # 默认不计算梯度

        # # 然后只为需要计算梯度的参数设置 requires_grad=True
        # params_to_compute = list(model.parameters())[:60]
        iter_max = 2000
        iter = 0
        state = []
        for batch, batch_data in enumerate(data_loader):
            model.zero_grad()
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
                
                    # scaler.scale(loss).backward()
                # with backpack(DiagHessian()):
                #     loss.backward()
                # for name, param in model.named_parameters():
                #     if hasattr(param, 'diag_fisher'):
                #         print(param.diag_fisher.detach())  # 获取 Fisher 信息
                #     else:
                #         print(f"Warning: Parameter {name} does not have diag_fisher attribute.") # 访问 Fisher 信息
                # for param in self.model.parameters():
                #     param.grad = None
                # for param in model.parameters():
                #     # state.extend(param.data.view(-1).cpu().numpy())               # 当前层的参数
                #     state.extend(param.data.cpu().numpy())             # 60697 参数总量
                #     if param.grad is not None:                         # loss反向求梯度才有值
                #         # state.extend(param.grad.view(-1).cpu().numpy())
                #         state.extend(param.grad.cpu().numpy())
                grads = autograd.grad(scaler.scale(loss), model.parameters(), allow_unused=True, retain_graph=False)    # 193个网络层的参数  每个样本
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
        torch.save({'fisher': self.fisher, 'optpar': self.optpar}, '/workspace/geshuang/code/scGPT/fisher/fisher_optpar_(reverse)' + dataset_name +'.pth')
        print('Fisher information and optimal parameters saved.')
        
        
        