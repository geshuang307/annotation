import torch
import traceback
import torch.nn.functional as F

class PPPloss(object):
    """ Pseudo-Prototypical Proxy loss
    """
    modes = ["joint", "pos", "neg"]  # Include attractor (pos) or repellor (neg) terms

    def __init__(self, net, mode="joint", T=1, tracker=None, ):
        """
        :param margin: margin on distance between pos vs neg samples (see TripletMarginLoss)
        :param dist: distance function 2 vectors (e.g. L2-norm, CosineSimilarity,...)
        """
        assert mode in self.modes
        self.net = net
        self.mode = mode
        self.T = T
        self.margin = 1

        # INIT tracker
        self.tracker = tracker
        self.tracker['log_it'] = []
        self.tracker['loss'] = []
        self.tracker['lnL_pos'] = []
        self.tracker['lnL_neg'] = []

    def __call__(self, x_metric, labels, prototypes, memory_prototypes, device, eps=1e-8):
        """
        Standard reduction is mean, as we use full batch information instead of per-sample.
        Symmetry in the distance function inhibits summing loss over all samples.

        :param x_metric: embedding output of batch
        :param labels: labels batch
        :param class_mem: Stored prototypes/exemplars per seen class
        """
        if self.mode == "joint":
            pos, neg = True, True
        elif self.mode == "pos":
            pos, neg = True, False
        elif self.mode == "neg":
            pos, neg = False, True
        else:
            raise NotImplementedError()
        return self.softmax_joint(x_metric, labels, prototypes, memory_prototypes, device, gpu=True, pos=pos, neg=neg)

    # def softmax_joint(self, x_metric, y, prototypes, device, gpu=True, pos=True, neg=True):
    #     """
    #     - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))

    #     Note:
    #     log(Exp(y)) makes y always positive, which is required for our loss.
    #     """
    #     x_metric = F.normalize(x_metric, p=2, dim=1)
    #     if torch.isnan(x_metric).any():
    #         print("skipping NaN batch")
    #         return torch.tensor(0)
    #     assert pos or neg, "At least one of the pos/neg terms must be activated in the Loss!"
    #     assert len(x_metric.shape) == 2, "Should only have batch and metric dimension."
    #     bs = x_metric.size(0)

    #     # All prototypes
    #     # p_x, p_y = self.net.get_all_prototypes()
    #     p_y = torch.tensor(list(prototypes.keys())).to(device)  # tensor([0, 1, 2])
    #     if p_y.numel() == 0:
    #         return torch.tensor(0, device=device)  # No prototypes available
    #     p_x = torch.stack([prototypes[c].squeeze(0).to(device) for c in p_y.tolist()]).to(device)  # shape: [3, 16]
    #     # Init
    #     loss = torch.tensor(0.0)
    #     y_unique = torch.unique(y).squeeze()
    #     neg = False if len(y_unique.size()) == 0 else neg  # If only from the same class, there is no neg term
    #     y_unique = y_unique.view(-1)

    #     # Log
    #     tmplate = str("{: >20} " * y_unique.size(0))
    #     # if self.net.log:
    #     # print("\n".join(["-" * 40, "LOSS", tmplate.format(*list(y_unique))]))
    #     self.tracker['lnL_pos'].append(0)
    #     self.tracker['lnL_neg'].append(0)

    #     # for label_idx in range(y_unique.size(0)):  # [summation over i]
    #     for c in p_y.tolist():
    #         # c = y_unique[label_idx]

    #         # Select from batch
    #         xc_idxs = (y == c).nonzero().squeeze(dim=1)
    #         if xc_idxs.numel() == 0:    # 增加
    #             continue
    #         xc = x_metric.index_select(0, xc_idxs)

    #         xk_idxs = (y != c).nonzero().squeeze(dim=1)
    #         xk = x_metric.index_select(0, xk_idxs)

    #         # p_idx = (p_y == c).nonzero().squeeze(dim=1)
    #         # pc = p_x[p_idx].detach()
    #         # pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1:]]).detach()  # Other class prototypes
    #         p_idx = (p_y == c).nonzero(as_tuple=False).squeeze()
    #         if p_idx.numel() != 1:
    #             raise ValueError(f"Expected exactly one prototype index for class {c}, but got {p_idx}")
    #         p_idx = p_idx.item()  # Convert to Python int

    #         # Now you can slice safely
    #         pc = p_x[p_idx].detach()
    #         pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1:]], dim=0).detach()  # Other class prototypes    
    #         # if self.net.log:
    #         # print("Class {}:".format(str(c.item())), end='')

    #         lnL_pos = self.attractor(pc, pk, xc, gpu, device, include_batch=True) if pos else torch.tensor(0.0)  # Pos
    #         lnL_neg = self.repellor(pc, pk, xc, xk, gpu, device, include_batch=True) if neg else torch.tensor(0.0)  # Neg

    #         # Pos + Neg
    #         Loss_c = -lnL_pos - lnL_neg  # - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))
    #         # if self.net.log:
    #         # print("{: >20}".format(
    #         #     "| TOTAL: {:.1f} + {:.1f} = {:.1f}".format(float(-lnL_pos), float(-lnL_neg), float(Loss_c))))
    #         self.tracker['lnL_pos'][-1] -= lnL_pos.item()
    #         self.tracker['lnL_neg'][-1] -= lnL_neg.item()

    #         # Update loss
    #         loss = Loss_c if loss is None else loss + Loss_c

    #         # Checks
    #         try:
    #             assert lnL_pos <= 0
    #             assert lnL_neg <= 0
    #             assert loss >= 0 and loss < 1e10
    #         except:
    #             traceback.print_exc()
    #             exit(1)
    #     # if self.net.log:
    #     self.tracker['loss'].append(loss.item())
    #     # print("-" * 40)
    #     return loss / bs  # Make independent batch size
    def softmax_joint(self, x_metric, y, prototypes, memory_prototypes, device, gpu=True, pos=True, neg=True):
        """
        - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))

        Note:
        log(Exp(y)) makes y always positive, which is required for our loss.
        """
        x_metric = F.normalize(x_metric, p=2, dim=1)
        if torch.isnan(x_metric).any():
            print("skipping NaN batch")
            return torch.tensor(0)
        assert pos or neg, "At least one of the pos/neg terms must be activated in the Loss!"
        assert len(x_metric.shape) == 2, "Should only have batch and metric dimension."
        bs = x_metric.size(0)

        # All prototypes
        # p_x, p_y = self.net.get_all_prototypes()
        prototypes = {k: v for k, v in prototypes.items() if not torch.all(v == 0)}            # 改：增加了一句代码
        p_y = torch.tensor(list(prototypes.keys())).to(device)  # tensor([0, 1, 2])
        if p_y.numel() == 0:
            return torch.tensor(0, device=device)  # No prototypes available
        p_x = torch.stack([prototypes[c].squeeze(0).to(device) for c in p_y.tolist()]).to(device)  # shape: [3, 16]
        # Init
        loss = torch.tensor(0.0)
        y_unique = torch.unique(y).squeeze()
        neg = False if len(y_unique.size()) == 0 else neg  # If only from the same class, there is no neg term
        y_unique = y_unique.view(-1)

        # Log
        tmplate = str("{: >20} " * y_unique.size(0))
        # if self.net.log:
        # print("\n".join(["-" * 40, "LOSS", tmplate.format(*list(y_unique))]))
        self.tracker['lnL_pos'].append(0)
        self.tracker['lnL_neg'].append(0)

        # for label_idx in range(y_unique.size(0)):  # [summation over i]
        for c in p_y.tolist():
            # c = y_unique[label_idx]

            # Select from batch
            xc_idxs = (y == c).nonzero().squeeze(dim=1)
            if xc_idxs.numel() == 0:    # 增加
                continue
            xc = x_metric.index_select(0, xc_idxs)

            xk_idxs = (y != c).nonzero().squeeze(dim=1)
            xk = x_metric.index_select(0, xk_idxs)

            # p_idx = (p_y == c).nonzero().squeeze(dim=1)
            # pc = p_x[p_idx].detach()
            # pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1:]]).detach()  # Other class prototypes
            p_idx = (p_y == c).nonzero(as_tuple=False).squeeze()     
                      
            if p_idx.numel() != 1:
                raise ValueError(f"Expected exactly one prototype index for class {c}, but got {p_idx}")
            p_idx = p_idx.item()  # Convert to Python int

            # Now you can slice safely
            # pc = p_x[p_idx].detach()
            # pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1:]], dim=0).detach()  # Other class prototypes   

######################################## 修改加入记忆原型 ############################################### 
            pc_current = p_x[p_idx].detach()  # 当前批次原型
            memory_pc_list = memory_prototypes.get(c, [])    # 历史原型列表
            if len(memory_pc_list) > 0:
                memory_pc_tensor = torch.stack([m for m in memory_pc_list], dim=0)  # shape: [M, D]
                pc = torch.cat([pc_current.unsqueeze(0), memory_pc_tensor], dim=0).detach()  # shape: [1+M, D]
            else:
                pc = pc_current.unsqueeze(0)
            
            other_classes = [k for k in p_y.tolist() if k != c]
            pk_list = []
            for k in other_classes:
                pk_current = prototypes[k].squeeze(0).cpu()
                memory_k_list = memory_prototypes.get(k, [])
                if len(memory_k_list) > 0:
                    memory_k_tensor = torch.stack([m.cpu() for m in memory_k_list], dim=0)
                    k_proto = torch.cat([pk_current.unsqueeze(0), memory_k_tensor], dim=0)
                else:
                    k_proto = pk_current.unsqueeze(0)
                pk_list.append(k_proto)              # 15

            if len(pk_list) > 0:
                pk = torch.cat(pk_list, dim=0).detach()  # CPU 上 detach
            else:
                pk = torch.empty((0, pc.size(1)), device='cpu')
            pc = pc.to(device)
            pk = pk.to(device)
#######################################################################################################

            lnL_pos = self.attractor(pc, pk, xc, gpu, device, include_batch=True) if pos else torch.tensor(0.0)  # Pos
            lnL_neg = self.repellor(pc, pk, xc, xk, gpu, device, include_batch=True) if neg else torch.tensor(0.0)  # Neg

            # Pos + Neg
            Loss_c = -lnL_pos - lnL_neg  # - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))
            # if self.net.log:
            # print("{: >20}".format(
            #     "| TOTAL: {:.1f} + {:.1f} = {:.1f}".format(float(-lnL_pos), float(-lnL_neg), float(Loss_c))))
            self.tracker['lnL_pos'][-1] -= lnL_pos.item()
            self.tracker['lnL_neg'][-1] -= lnL_neg.item()

            # Update loss
            loss = Loss_c if loss is None else loss + Loss_c

            # Checks
            try:
                assert lnL_pos <= 0
                assert lnL_neg <= 0
                assert loss >= 0 and loss < 1e10
            except:
                traceback.print_exc()
                exit(1)
        # if self.net.log:
        self.tracker['loss'].append(loss.item())
        # print("-" * 40)
        return loss / bs  # Make independent batch size
    def repellor(self, pc, pk, xc, xk, gpu, device, include_batch=True):
        # Gather per other-class samples
        if not include_batch:
            union_c = pc
        else:
            # union_c = torch.cat([xc, pc.unsqueeze(0)])
            union_c = torch.cat([xc, pc])
        union_ck = torch.cat([union_c, pk]) #.clone().detach()
        c_split = union_c.shape[0]
        if gpu:
            union_ck = union_ck.to(device)

        neg_Lterms = torch.mm(union_ck, xk.t()).div_(self.T).exp_()  # Last row is with own prototype
        pk_terms = neg_Lterms[c_split:].sum(dim=0).unsqueeze(0)  # For normalization
        pc_terms = neg_Lterms[:c_split]
        Pneg = pc_terms / (pc_terms + pk_terms)

        expPneg = (Pneg[:-1] + Pneg[-1].unsqueeze(0)) / 2  # Expectation pseudo/prototype
        lnPneg_k = expPneg.mul_(-1).add_(1).log_()  # log( (1 - Pk))
        lnPneg = lnPneg_k.sum()  # Sum over (pseudo-prototypes), and instances

        # if self.net.log:
        # print(" + (#k) {:.1f}/({:.1f} + {:.1f})".format(float(pc_terms.mean().item()),
        #                                                     float(pc_terms.mean().item()),
        #                                                     float(pk_terms.mean().item())), end='')
        try:
            assert -10e10 < lnPneg <= 0
        except:
            print("error")
            traceback.print_exc()
            exit(1)
        return lnPneg

    def attractor(self, pc, pk, xc, gpu, device, include_batch=True):
        # Union: Current class batch-instances, prototype, memory
        if include_batch:
            pos_union_l = [xc.clone()]
            pos_len = xc.shape[0]
        else:
            pos_union_l = []
            pos_len = 1
        # pos_union_l.append(pc.unsqueeze(0)) 
        pos_union_l.append(pc)                  # 增加 memory_prototypes的时候使用

        if gpu:
            pos_union_l = [x.to(device) for x in pos_union_l]
        pos_union = torch.cat(pos_union_l)
        all_pos_union = torch.cat([pos_union, pk]).clone().detach()  # Include all other-class prototypes p_k
        pk_offset = pos_union.shape[0]  # from when starts p_k

        # Resulting distance columns are per-instance loss terms (don't include self => diagonal)
        pos_Lterms = torch.mm(all_pos_union, xc.t()).div_(self.T).exp_()  # .fill_diagonal_(0)
        if include_batch:
            mask = torch.eye(*pos_Lterms.shape).bool().to(device) if gpu else torch.eye(*pos_Lterms.shape).bool()
            pos_Lterms = pos_Lterms.masked_fill(mask, 0)  # Fill with zeros

        Lc_pos = pos_Lterms[:pk_offset]
        Lk_pos = pos_Lterms[pk_offset:].sum(dim=0)  # sum column dist to pk's

        # Divide each of the terms by itself+ Lk term to get probability
        Pc_pos = Lc_pos / (Lc_pos + Lk_pos)
        expPc_pos = Pc_pos.sum(0) / (pos_len)  # Don't count self in
        lnL_pos = expPc_pos.log_().sum()

        # try:                                    # 改
        #     assert lnL_pos <= 0
        # except:
        #     traceback.print_exc()
        #     exit(1)
        return lnL_pos
    
if __name__ == "__main__":
    import torch
    import torch.nn.functional as F
    from collections import defaultdict
    # 假设特征维度为 D，batch size 为 B，类别数为 C
    torch.manual_seed(42)
    B, D, C = 12, 16, 3
    temperature = 0.7

    # 生成伪造的特征向量 [B, D]
    features = F.normalize(torch.randn(B, D), dim=1)  # 模拟嵌入后的特征，单位向量

    # 生成标签 [B]
    # 让每个类都有若干样本，例如 4 个样本/类
    labels = torch.tensor([0]*4 + [1]*4 + [2]*4)

    # 构造原型字典，使用 mean pooling 生成每类原型
    prototypes = defaultdict(lambda: torch.zeros(D))
    for c in range(C):
        cls_feats = features[labels == c]
        prototypes[c] = cls_feats.mean(dim=0)
        prototypes[c] = F.normalize(prototypes[c], dim=0)  # 归一化

    # 调用 ppp_loss
    # loss_value = ppp_loss(features, labels, prototypes, temperature=temperature)
    # print(f"PPP Loss = {loss_value.item():.6f}")


    loss_module = PPPloss(net=None, mode="joint", T=temperature, tracker={'log_it': [], 'loss': [], 'lnL_pos': [], 'lnL_neg': []})
    loss = loss_module(features, labels, prototypes, eps=1e-8)
    print(f"PPP Loss (module) = {loss.item():.6f}")