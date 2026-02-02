import torch
import torch.nn.functional as F
from collections import defaultdict

class PrototypeScheme(object):
    """
    Prototype update scheme using self.old_proto (defaultdict) to store class prototypes.
    EMA update with momentum and optional L2 normalization.
    """

    def __init__(self, device, p_momentum=0.9):
        """
        :param device: torch.device
        :param p_momentum: EMA momentum for prototype update
        """
        self.device = device
        self.p_momentum = p_momentum
        self.old_proto = defaultdict(lambda: None)

    def __call__(self, f, y, old_proto):
        """
        Update prototypes for a batch.
        :param f: Tensor, feature embeddings (batch, feat_dim)
        :param y: Tensor, labels (batch,)
        :param old_proto: defaultdict storing class prototypes {class_idx: prototype_tensor}
        """
        with torch.no_grad():
            f, y = f.to(self.device), y.to(self.device)
            self.update_prototypes(f, y, old_proto)

    def momentum_update(self, old_value, new_value):
        """EMA update of prototype."""
        return self.p_momentum * old_value + (1 - self.p_momentum) * new_value

    def summarize_p_update(self, c, new_p, old_p):
        """Print prototype update info."""
        delta = (new_p - old_p).pow(2).sum().sqrt()
        print(f"Class {c} prototype update: L2 delta={delta.item():.4f}")

    def update_prototypes(self, f, y, old_proto):
        """Update prototypes for each class in the batch."""
        unique_labels = torch.unique(y).squeeze()
        for c in unique_labels:
            c = c.item()
            idxs = (y == c).nonzero(as_tuple=True)[0]
            batch_mean = f[idxs].mean(dim=0)

            # If old prototype doesn't exist, initialize as zeros
            old_p = old_proto.get(c, torch.zeros_like(batch_mean).to(self.device))

            # EMA update
            new_p = self.momentum_update(old_p, batch_mean)

            # Log update
            self.summarize_p_update(c, new_p, old_p)

            # Store normalized prototype
            old_proto[c] = F.normalize(new_p, p=2, dim=0)

    def update_prototype_old(self, embedding_list, label_list, epoch = None):
        if isinstance(embedding_list, (list, tuple)):
            embedding_list = torch.cat(embedding_list, dim=0)
        if isinstance(label_list, (list, tuple)):
            label_list = torch.cat(label_list, dim=0)

        class_list = torch.unique(label_list).cpu().numpy()
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0] 
            embedding = embedding_list[data_index]
            new_proto = embedding.mean(0).float().detach()
            # self.model.cls_decoder.out_layer.weight.data[int(class_index)] = proto.to(self.device)
            # if self.config["cope_loss"]:
            # self.old_proto[int(class_index)] = proto.clone()
            proto = self.old_proto.get(int(class_index), torch.zeros(512).to(self.device))

            # if self.config["proto_loss"] and self.config["ema"]:
            if True:
           
                if proto.sum() == 0:
                    self.old_proto[int(class_index)] = new_proto.clone()
                else:
                    # EMA 更新原型
                    ema_momentum = 0.95
                    self.old_proto[int(class_index)] = (
                        ema_momentum  * self.old_proto[int(class_index)] +
                        (1 - ema_momentum)* new_proto
                    )
            # else:
            #     self.old_proto[int(class_index)] = new_proto.clone()
            self.old_proto[int(class_index)] = F.normalize(self.old_proto[int(class_index)], p=2, dim=-1)
if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_scheme = PrototypeScheme(device, p_momentum=0.95)
    old_proto = defaultdict(lambda: None)

    # Dummy data
    features = torch.randn(10, 64)  # 10 samples, 64-dim features
    labels = torch.tensor([0, 1, 0, 1, 2, 2, 0, 1, 2, 0])  # Class labels

    # Update prototypes
    p_scheme(features, labels, old_proto)

    # Print updated prototypes
    for cls, proto in old_proto.items():
        print(f"Class {cls} prototype: {proto}")

    p_scheme.update_prototype_old(features, labels)

    for cls, proto_old in p_scheme.old_proto.items():
        print(f"Class {cls} prototype: {proto_old}")

    

