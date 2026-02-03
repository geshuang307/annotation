import wandb
def param(dataset_name, load_model, weight_dir, batch_size, epochs):   
    hyperparameter_defaults = dict(
    seed=42,
    # dataset_name="PBMC_10K", # Dataset name
    dataset_name= dataset_name,
    do_train=True, # Flag to indicate whether to do update model parameters during training
    load_model = load_model, # Path to pre-trained model
    # weight_dir = "/workspace/geshuang/code/scGPT/save/dev_PBMC_10K-Feb26-19-34",
    weight_dir = weight_dir, 
    GEPC=True,  # Gene expression modelling for cell objective
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0, # DAR objective weight for batch correction
    mask_ratio=0.4, # Default mask ratio
    epochs=epochs, # Default number of epochs for fine-tuning
    n_bins=51, # Default number of bins for value binning in data pre-processing
    lr=1e-4, # Default learning rate for fine-tuning
    batch_size=batch_size, # Default batch size for fine-tuning
    layer_size=128,
    nlayers=4,
    nhead=4, # if load model, batch_size, layer_size, nlayers, nhead will be ignored
    dropout=0.2, # Default dropout rate during model fine-tuning
    schedule_ratio=0.9,  # Default rate for learning rate decay
    save_eval_interval=5, # Default model evaluation interval
    log_interval=100, # Default log interval
    fast_transformer=True, # Default setting
    pre_norm=False, # Default setting
    amp=True,  # # Default setting: Automatic Mixed Precision
)
#     run = wandb.init(
#     config=hyperparameter_defaults,
#     project="scGPT",
#     reinit=True,
#     settings=wandb.Settings(start_method="fork"),
# )
    # config = wandb.config
    config = hyperparameter_defaults
    return hyperparameter_defaults,  config