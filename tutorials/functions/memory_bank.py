import torch
import anndata
import numpy as np

def example_bank_update(adata, adata_train_indices_list, label_list, entropy_list, accuracy_list, save_dir, test_batch_idx, current_prototypes, cell_emb):
        if test_batch_idx != 0:
            checkpoint = torch.load(save_dir / "example_bank.pth")
            example_bank_previous = checkpoint['example_bank']
            example_bank_previous_emb = checkpoint['example_bank_emb'] 
        else:
            example_bank_previous = None
            example_bank_previous_emb = None
        
        cell_emb = torch.cat(cell_emb, dim=0).detach().cpu()  # shape: [N, D]
        label_list = torch.cat(label_list, dim=0)           # shape: [N]
        entropy_list = torch.cat([t.unsqueeze(0) for t in entropy_list])
        accuracy_list = torch.cat([t.unsqueeze(0) for t in accuracy_list])     # shape: [N], bool or int (0/1)
        adata_train_indices_list = torch.cat(adata_train_indices_list, dim=0)
        adata = adata[adata_train_indices_list.numpy()]
        class_list = torch.unique(label_list).detach().cpu().numpy()
        max_total = 1000
        num_classes = len(class_list)
        max_per_class = max_total // num_classes

        selected_adata_list = []
        selected_emb_list = []
        selected_label_list = []

        for class_index in class_list:
            data_index = (label_list == class_index).nonzero(as_tuple=True)[0]

            class_entropy = entropy_list[data_index]
            class_accuracy = accuracy_list[data_index]

            correct_mask = class_accuracy.bool()
            if correct_mask.sum() == 0:
                continue

            arange_indices = torch.arange(len(data_index), device=correct_mask.device)
            correct_indices = arange_indices[correct_mask]

            # arange_indices = torch.arange(len(data_index), device=incorrect_mask.device)
            # incorrect_indices = arange_indices[incorrect_mask]

            filtered_entropy = class_entropy[correct_indices]    # classify the True 

            topk = min(max_per_class, filtered_entropy.size(0))
            # _, top_indices = torch.topk(-filtered_entropy, topk)             
            _, top_indices = torch.topk(filtered_entropy, topk)               
            selected_entropies = filtered_entropy[top_indices].detach().cpu().numpy()

            selected_indices = data_index[correct_indices[top_indices]]
            

            selected_adata = adata[selected_indices.detach().cpu().numpy()].copy()
            selected_adata.obs["selected_class"] = class_index  
            selected_adata.obs["entropy"] = selected_entropies
            selected_adata.obs["test_batch_idx"] = test_batch_idx
            selected_adata.var = adata.var.copy()

            selected_adata_list.append(selected_adata)
            selected_emb_list.append(cell_emb[selected_indices])
            selected_label_list.append(label_list[selected_indices])

 
        if len(selected_adata_list) == 0:
            print("⚠️ No samples selected for example bank")
            return None
        else:
            example_bank_current = anndata.concat(
            selected_adata_list,
            axis=0,
            merge="same"  
        )

        current_emb = torch.cat(selected_emb_list, dim=0)   
        current_labels = torch.cat(selected_label_list, dim=0)


        if example_bank_previous is not None and example_bank_previous_emb is not None:
            example_bank_all = anndata.concat([example_bank_previous, example_bank_current], axis=0, merge='same')
            all_embeddings = torch.cat([example_bank_previous_emb, current_emb], dim=0).detach().cpu().numpy()
            all_labels = np.concatenate([
                checkpoint['example_bank_labels'], 
                current_labels.detach().cpu().numpy()
            ])
        else:
            example_bank_all = example_bank_current
            all_embeddings = current_emb.detach().cpu().numpy()
            all_labels = current_labels.detach().cpu().numpy()
    
        if isinstance(cell_emb, torch.Tensor):
            cell_emb = cell_emb.detach().cpu().numpy()

        filtered_bank = []
        filtered_emb_list = []
        filtered_label_list = []
       
        for class_index in np.unique(all_labels):
            idx_in_bank = np.where(all_labels == class_index)[0]
            if len(idx_in_bank) > max_per_class:
                proto = current_prototypes[int(class_index)]
                if isinstance(proto, torch.Tensor):
                    proto = proto.detach().cpu().numpy()
                proto = proto / np.linalg.norm(proto)
                emb_class = all_embeddings[idx_in_bank]
                emb_class = emb_class / np.linalg.norm(emb_class, axis=1, keepdims=True)

                # distances = np.linalg.norm(emb_class - proto, axis=1)
                # sorted_idx = np.argsort(distances)
                # keep_idx = idx_in_bank[sorted_idx[:max_per_class]]
                # cosine_sim = np.sum(emb_class * proto, axis=1)  
                diff_proto = emb_class - proto        # (n, dim)
                dist_to_proto = np.linalg.norm(diff_proto, axis=1)  
                # sorted_idx = np.argsort(-cosine_sim)  
                # sorted_idx = np.argsort(cosine_sim) 
                sorted_idx = np.argsort(-dist_to_proto) 
                # num_keep = max_per_class
                # num_center = int(num_keep * 0.7)
                # num_boundary = num_keep - num_center

                # keep_center = sorted_idx[:num_center]       
                # keep_boundary = sorted_idx[-num_boundary:]  
                # keep_idx = idx_in_bank[np.concatenate([keep_center, keep_boundary])]    
                keep_idx = idx_in_bank[sorted_idx[:max_per_class]]
            else:
                keep_idx = idx_in_bank

            filtered_bank.append(example_bank_all[keep_idx])
            filtered_emb_list.append(torch.tensor(all_embeddings[keep_idx]))
            filtered_label_list.append(all_labels[keep_idx])

        example_bank_updated = anndata.concat(filtered_bank, axis=0, merge='same')
        updated_embeddings = torch.cat(filtered_emb_list, dim=0)
        updated_labels = np.concatenate(filtered_label_list)

        torch.save({
            'example_bank': example_bank_updated,
            'example_bank_emb': updated_embeddings,
            'example_bank_labels': updated_labels,
        }, save_dir / "example_bank.pth")
        
        return example_bank_updated