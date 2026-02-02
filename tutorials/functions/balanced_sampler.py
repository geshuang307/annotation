import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import numpy as np


def stratified_sample(input_gene_ids, 
                      input_values, 
                      target_values, 
                      batch_labels, 
                      celltype_labels = None, 
                      multimod_labels = None,
                      mod_types = None,
                      total_samples=2000, 
                      min_per_class=1, 
                      seed=None):
    """
    Stratified sampling by class proportions to avoid dropping small classes.

    Parameters:
        input_gene_ids: Tensor, gene IDs
        input_values: Tensor, input values
        target_values: Tensor, target values
        batch_labels: Tensor, batch labels
        celltype_labels: Tensor, cell type labels (used for stratified sampling)
        total_samples: int, total number of samples
        min_per_class: int, minimum samples per class
        seed: int or None, random seed (for reproducible sampling)

    Returns:
        Tuple: (input_gene_ids_sampled, input_values_sampled, target_values_sampled, batch_labels_sampled, celltype_labels_sampled)
    """
    if seed is not None:
        torch.manual_seed(seed)

    unique_labels = torch.unique(celltype_labels)
    indices_to_keep = []

    # Sample by class proportions
    for cls in unique_labels:
        cls_indices = (celltype_labels == cls).nonzero(as_tuple=True)[0]
        cls_count = cls_indices.size(0)
        
        # Allocate number of samples proportionally
        num_samples_cls = max(min_per_class, int(total_samples * cls_count / celltype_labels.size(0)))
        num_samples_cls = min(num_samples_cls, cls_count)  # Do not exceed the class count

        # Randomly sample indices from this class
        perm_cls = torch.randperm(cls_count)[:num_samples_cls]
        sampled_cls_indices = cls_indices[perm_cls]
        
        indices_to_keep.append(sampled_cls_indices)

    # Combine indices
    indices_to_keep = torch.cat(indices_to_keep)
    indices_to_keep = indices_to_keep[torch.randperm(indices_to_keep.size(0))]  # Shuffle again
    if mod_types is not None and multimod_labels is None:
        return (input_gene_ids[indices_to_keep],
            input_values[indices_to_keep],
            target_values[indices_to_keep],
            batch_labels[indices_to_keep],
            celltype_labels[indices_to_keep],
            mod_types[indices_to_keep], indices_to_keep)
    elif mod_types is not None and multimod_labels is not None:
        return (input_gene_ids[indices_to_keep],
            input_values[indices_to_keep],
            target_values[indices_to_keep],
            batch_labels[indices_to_keep],
            celltype_labels[indices_to_keep],
            multimod_labels[indices_to_keep],
            mod_types[indices_to_keep], indices_to_keep)
    else:
        return (input_gene_ids[indices_to_keep],
            input_values[indices_to_keep],
            target_values[indices_to_keep],
            batch_labels[indices_to_keep],
            celltype_labels[indices_to_keep], indices_to_keep)


def stratified_split_with_small_class_reserve(
    all_counts,
    celltypes_labels,
    batch_ids,
    multimods_ids=None,
    test_size=0.1,
    min_class_size=20,
    random_state=42
):
    """
    Stratified split with small-class reserve strategy
    - Classes smaller than min_class_size are all placed in the training set
    - Remaining classes are split using stratified sampling

    Parameters:
        all_counts: feature matrix (numpy array or torch tensor)
        celltypes_labels: class labels (numpy array)
        batch_ids: batch labels (numpy array)
        test_size: proportion for test set
        min_class_size: threshold for small classes
        random_state: random seed
    """

    # Convert inputs to numpy arrays if needed
    if not isinstance(all_counts, np.ndarray):
        all_counts = all_counts.cpu().numpy()
    if not isinstance(celltypes_labels, np.ndarray):
        celltypes_labels = np.array(celltypes_labels)
    if not isinstance(batch_ids, np.ndarray):
        batch_ids = np.array(batch_ids)

    # Count samples per class
    unique_classes, counts = np.unique(celltypes_labels, return_counts=True)
    small_classes = unique_classes[counts < min_class_size]

    # Find indices of small and big classes
    small_class_mask = np.isin(celltypes_labels, small_classes)
    big_class_mask = ~small_class_mask
    # All small-class samples go to the training set
    small_indices = np.where(small_class_mask)[0]

    # Indices of big classes
    big_indices = np.where(big_class_mask)[0]

    # Stratified sampling for big classes
    if len(big_indices) < 2:
        big_train_indices = big_indices
        big_val_indices = []
    else:
        big_train_indices, big_val_indices = train_test_split(
            big_indices,
            test_size=test_size,
            shuffle=True,
            stratify=celltypes_labels[big_indices],
            random_state=random_state
        )

    # Training indices = all small-class indices + big-class training indices
    train_indices = np.concatenate([small_indices, big_train_indices])
    val_indices = big_val_indices  # Validation set contains only big-class portion

    # Index into data arrays
    train_data = all_counts[train_indices]
    valid_data = all_counts[val_indices]
    train_celltype_labels = celltypes_labels[train_indices]
    valid_celltype_labels = celltypes_labels[val_indices]
    train_batch_labels = batch_ids[train_indices]
    valid_batch_labels = batch_ids[val_indices]
    if multimods_ids is not None:
        train_multimod_labels = multimods_ids[train_indices]
        valid_multimod_labels = multimods_ids[val_indices]
        return train_data, valid_data, train_celltype_labels, valid_celltype_labels, train_batch_labels, valid_batch_labels, train_multimod_labels, valid_multimod_labels, train_indices, val_indices

    return train_data, valid_data, train_celltype_labels, valid_celltype_labels, train_batch_labels, valid_batch_labels, train_indices, val_indices

