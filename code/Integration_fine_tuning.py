#region loading packages
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

import torch
from anndata import AnnData
import scanpy as sc
# import scvi ### Not compatible with python 3.8
import numpy as np
import wandb
from scipy.sparse import issparse
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)


sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model import TransformerModel, AdversarialDiscriminator
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

sc.set_figure_params(figsize=(4, 4))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')
#endregion

#region Finetuning Primary
# Fine-tuning on Pre-trained Model with Batch Integration
# 1. Specify hyper-parameter setup for integration task
# 2. Load and pre-process data
# 3. Load the pre-trained scGPT model
# 4. Finetune scGPT with task-specific objectives
# 5. Evaluate fine-tuned scGPT

# adata = sc.read_h5ad("/diazlab/data3/.abhinav/projects/Brain_metastasis/prim_metastasis/saveh5ad/adata_combined.h5ad")
adata = sc.read_h5ad("/diazlab/data3/.abhinav/projects/Brain_metastasis/prim_metastasis/saveh5ad/adata_combined_processed.h5ad")

# Specifying the hyper-parameter setup for integration task
hyperparameter_defaults = dict(
    seed=42,
    dataset_name="Brain_Metastasis",
    do_train=True, # Flag to indicate whether to do update model parameters during training
    load_model="/diazlab/data3/.abhinav/projects/Brain_metastasis/scgpt/resources/scGPT_CP/", # Path to pre-trained model
    GEPC=True,  # Gene expression modelling for cell objective
    ecs_thres=0.8,  # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
    dab_weight=1.0, # DAR objective weight for batch correction
    mask_ratio=0.4, # Default mask ratio 40% of the genes for each cell will be masked out, and the model will try to predict their values from the rest.
    epochs=15, # Default number of epochs for fine-tuning
    n_bins=51, # Default number of bins for value binning in data pre-processing
    lr=1e-4, # Default learning rate for fine-tuning
    batch_size=64, # Default batch size for fine-tuning
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

# GEP & GEPC → capture gene-level biology.
# ECS → enforce meaningful cell embeddings.
# DAR → explicitly remove batch signal.
# DSBN → normalize per-batch while keeping global embedding space stable.
# GEPC: Masked gene prediction → learns gene dependencies.
# ECS: Contrastive similarity → clusters of similar cells.
# In the embedding space (like the 512-dim scGPT embeddings):
# Cosine similarity measures the angle between two vectors.
# Closer direction → similarity close to 1.
# Orthogonal → similarity near 0.
# Opposite → similarity near -1.
# Positive pairs → high cosine similarity.
# Negative pairs → low cosine similarity.
# Connection to KNN
# In Scanpy, when you call sc.pp.neighbors, it typically uses Euclidean distance (or PCA-Euclidean) to construct the KNN graph.
# In contrastive learning (ECS), scGPT focuses on cosine similarity to define closeness in the latent space.

# Both are about closeness in embedding space.
# Cosine similarity cares about direction (patterns of expression).
# Euclidean distance cares about absolute magnitude and scale.
# DAR / DAB: Gradient reversal → removes batch effect.
# Mask ratio: Forces model to infer missing expression, like BERT masking.
# AMP: Mixed precision training → faster and more memory efficient.

# Model architecture parameters
# layer_size=128
# This is the hidden dimension of the transformer model.
# Every embedding vector and every hidden state will be 128-dimensional.
# Larger layer_size = more model capacity, but slower + more memory.
# nlayers=4
# Number of transformer encoder layers (stacked blocks).
# Each layer has: multi-head attention + feed-forward network + normalization.
# More layers = deeper model, better representation capacity, but slower.
# nhead=4
# Number of attention heads in multi-head self-attention.
# Each head learns a different type of relationship (e.g., co-expression, anti-correlation).
# layer_size / nhead = dimension per head (here: 128/4 = 32).
# dropout=0.2
# Probability of dropping out neurons during training (20%).
# Helps avoid overfitting by preventing reliance on specific neurons.
# 🔹 Training schedule parameters
# schedule_ratio=0.9
# Controls learning rate decay.
# Means: over training, the learning rate is reduced to 90% of its original value (exponential or cosine schedule depending on implementation).
# Prevents overshooting and stabilizes convergence.
# save_eval_interval=5
# Evaluate & save the model every 5 epochs.
# Useful for monitoring progress (e.g., losses, metrics).
# log_interval=100
# Print training logs (loss, lr, etc.) every 100 batches.
# Doesn’t affect training itself, just reporting.


# layer_size = width of embeddings/hidden states.
# nlayers = depth (# of transformer blocks).
# nhead = # of parallel attention mechanisms.
# dropout = regularization to prevent overfitting.
# schedule_ratio = learning rate decay factor.
# save_eval_interval = how often to save & evaluate the model.
# log_interval = how often to print progress.

# 1. Learning rate (LR)
# The learning rate controls how big a step the optimizer takes when updating weights.
# High LR → fast learning but risk of overshooting minima.
# Low LR → stable but slow learning.
# Example:
# If LR = 0.1, and gradient = 5 → weight update = 0.5.
# If LR = 0.001, with the same gradient → update = 0.005.
# So LR decides the step size in gradient descent.
# 2. Decay rate (schedule ratio = 0.9 in your case)
# The LR is usually not kept constant → it’s decayed during training.
# Decay prevents overshooting later and helps fine-tune around minima.
# With a schedule ratio of 0.9, LR shrinks by 10% every scheduled step.
# Example:
#  Initial LR = 0.01
# After 1st step → 0.01×0.9=0.0090.01 \times 0.9 = 0.0090.01×0.9=0.009
# After 2nd step → 0.009×0.9=0.00810.009 \times 0.9 = 0.00810.009×0.9=0.0081
# After 3rd step → 0.0081×0.9=0.007290.0081 \times 0.9 = 0.007290.0081×0.9=0.00729
# This is exponential decay of LR.
# LR starts high → model learns quickly.
# Then LR decays → model fine-tunes carefully near optimal solution.
# The cell embedding is derived from the <cls> token output, which summarizes the whole cell by integrating information from all gene embeddings.
# So <cls> is not itself a gene embedding, but a “summary embedding” that becomes the cell representation.

# settings for input and preprocessing
os.environ["WANDB_MODE"] = "offline"
run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
wandb.config.update({"batch_size": 8}, allow_val_change=True)
wandb.config.update({"epochs": 1}, allow_val_change=True)

# How to classify a cell (via <cls>)
# Where the cell ends (via <eoc>)
# Ignore padding (via <pad>)

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = -1 ## This is the placeholder value used when a gene’s expression is masked during training (e.g. with mask_ratio = 0.4). The model sees:
pad_value = -2 ## 3. pad_value = -2 Used to pad sequences to a fixed length (since some cells may have more expressed genes than others). Important for batching.
n_input_bins = config.n_bins ##  4. n_input_bins = config.n_bins 
print("Passed Wandb")

n_hvg = 3000  # number of highly variable genes
max_seq_len = n_hvg + 1 ## per_seq_batch_sample = True This controls how batches (cells) are sampled during training.
per_seq_batch_sample = True
DSBN = True  # Domain-spec batchnorm
explicit_zero_prob = True  # whether explicit bernoulli for zeros
dataset_name = config.dataset_name
save_dir = Path(f"/diazlab/data3/.abhinav/projects/Brain_metastasis/prim_metastasis/fine_tuning/")
save_dir.mkdir(parents=True, exist_ok=True)
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

# Keep only rows that do NOT contain "Unknown"
adata_orig = adata.copy()
adata = adata[~adata.obs['Celltypes'].str.contains("Unknown", na=False)].copy()

# make the batch category column
adata.obs["str_batch"] = adata.obs["sample"].astype(str)
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
adata.var["gene_name"] = adata.var.index.tolist()

# Cross-check gene set with the pre-trained model
if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"] 
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info( 
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0] ## Keep only the genes that are in the vocabulary

    with open(model_config_file, "r") as f: # loading model configuration
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will be overriden by the "
        f"config {model_config_file}."
    )  
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]
else:
    embsize = config.layer_size
    nhead = config.nhead
    nlayers = config.nlayers
    d_hid = config.layer_size

### Preprocessing the dataset
# We follow the standardized pipline of depth normalization, log normalization, and highly vairable gene (HVG) selection for data pre-processing. We further 
# introduced value binning to obtain the relative expressions of each HVG.
# set up the preprocessor, use the args to config the workflow
# adata.layers['normalized'] = adata.X
data_is_raw = True
preprocessor = Preprocessor(
    use_key="X",  # the key in adata.layers to use as raw data
    filter_gene_by_counts=3,  # step 1
    filter_cell_by_counts=False,  # step 2
    normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
    result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
    log1p=data_is_raw,  # 4. whether to log1p the normalized data
    result_log1p_key="X_log1p",
    subset_hvg=n_hvg,  # 5. whether to subset the raw data to highly variable genes
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    # hvg_flavor="cell_ranger",
    binning=config.n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key="str_batch" if dataset_name != "Brain_Metastasis" else None)

if per_seq_batch_sample:
    # sort the adata by batch_id in advance
    adata_sorted = adata[adata.obs["batch_id"].argsort()].copy()

# Tokenize the input data for model fine-tuning
input_layer_key = "X_binned"
all_counts = (
    adata.layers[input_layer_key].A ## .A is an alias for .toarray() used in SciPy sparse matrices
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

adata.obs['Celltypes'] = adata.obs['Celltypes'].astype(str) ### to remove the categories
mask = adata.obs['Celltypes'].str.contains('Tumor', na=False)
adata.obs.loc[mask, 'Celltypes'] = adata.obs.loc[mask, 'type']

### Combining the cell types as T to T cells,  MDMs/Monocytes/Macrophage 1 to macrophage
adata.obs['Celltypes']=adata.obs['Celltypes'].replace("T","T_cell")
adata.obs['Celltypes']=adata.obs['Celltypes'].replace("MDMs/Monocytes/Macrophage 1 ","Macrophage")
adata.obs['Celltypes']=adata.obs['Celltypes'].replace("Plasma_cell","B/Plasma")
adata.obs['Celltypes']=adata.obs['Celltypes'].replace("Fibroblasts","Fibroblast")

adata.obs['batch']=adata.obs['batch'].replace("0","Primary")
adata.obs['batch']=adata.obs['batch'].replace("1","Metastasis")

celltypes_labels = adata.obs["Celltypes"].tolist()  # make sure count from 0
num_types = len(set(celltypes_labels))
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)

(
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
)

#### Checking the size
print(train_data.shape)
print(valid_data.shape)
print(len(train_celltype_labels))
print(len(valid_celltype_labels))
print(len(train_batch_labels))
print(len(valid_batch_labels))

if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)

tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=True,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
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

# To take pre-tokenized gene expression data (tokenized_train and tokenized_valid), randomly mask values, prepare input and target tensors, and return dictionaries 
# ready for model training.

# This function prepares training & validation batches where:
# gene_ids → tokens representing which genes are present.
# values → masked expression levels (input to model).
# target_values → true expression levels (used for loss).
# batch_labels → batch IDs (used for domain-adversarial training).

def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )
    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )
    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
    # If enabled, sorts cells by batch ID → useful for grouping during training.
    # Mostly for optimization efficiency.
    # Labels for which batch each cell belongs to.
    # Used for Domain Adversarial (DAR) objective → correct batch effects with gradient reversal.
    if sort_seq_batch:
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
    }
    return train_data_pt, valid_data_pt


# dataset
# Stores a dictionary of tensors (like from prepare_data())
# Implements PyTorch's Dataset interface
# At each index, returns a dictionary of tensors for that sample to be used for the training.
# dataset[5]  
# {"gene_ids": tensor([...]), "values": tensor([...]), "target_values": tensor([...]), "batch_labels": 2}
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
    def __len__(self):
        return self.data["gene_ids"].shape[0]
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

# data_loader
# adata → tensors → dataset → dataloader → model
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = SeqDataset(data_pt)
    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        # Groups samples (cells) by batch label.
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        # intra_subset_shuffle = shuffle within each batch domain.
        # inter_subset_shuffle = shuffle across batches.
        # This is useful for balanced sampling, so batches during training contain cells fairly from each dataset/batch.
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size, # number of cells loading to the model for training
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

'''
drop_last
Definition: Whether to drop the last incomplete batch if the dataset size isn’t divisible by batch_size.
Example:
Dataset = 10 cells, batch_size=4.
Batches: [cells 1–4], [cells 5–8], [cells 9–10].
Last batch only has 2 cells.
If drop_last=False → you keep [9, 10] as a smaller batch.
If drop_last=True → you throw away [9, 10], so all batches are exactly size 4.
Why would you drop the last batch?
Some training code assumes all batches are the same size (matrix operations fail if dimensions don’t match).
It can also stabilize training when using batch normalization, since very tiny batches give noisy statistics.
'''

## Step 3: Load the pre-trained scGPT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    dropout=config.dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=config.GEPC,
    do_dab=True,
    use_batch_labels=True,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=DSBN,
    n_input_bins=n_input_bins,
    ecs_threshold=config.ecs_thres,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=config.fast_transformer,
    pre_norm=config.pre_norm,
)
if config.load_model is not None:
    load_pretrained(model, torch.load(model_file), verbose=False)

model.to(device)
wandb.watch(model)

# Top-Level Structure
# This appears to be the SCGPTModel, and these are its key output "heads" (decoders):
# decoder: Main expression value prediction
# cls_decoder: Class prediction (e.g., cell type)
# mvc_decoder: Multi-Value Completion decoder (for masked value imputation)
# grad_reverse_discriminator: Adversarial batch/domain classifier
# sim: Similarity scorer (e.g., for contrastive or triplet learning)
# creterion_cce: Loss function (CrossEntropy)
'''
Gene IDs → GeneEncoder
Expression values → ValueEncoder
Batch IDs → BatchEncoder
↓
Combine embeddings → Transformer Encoder (12 layers)
↓
[CLS] embedding = Cell embedding
'''
TransformerModel(
  (encoder): GeneEncoder(
    (embedding): Embedding(60697, 512, padding_idx=60694) # The number 60697 is the size of the gene vocabulary in the pretrained model with 512 dimensions
    (enc_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True) # LayerNorm (short for Layer Normalization) normalizes the inputs across features (the 512 dimensions in your case) for each data point individually.
    #  It then applies a learnable scale and bias (elementwise_affine=True) to allow the model to shift/stretch the values as needed. 
    # eps is to prevent from normaliztion
  )
  (value_encoder): ContinuousValueEncoder(
    (dropout): Dropout(p=0.2, inplace=False) # Dropout randomly "drops" (sets to zero) a fraction of the neurons in a layer during each forward pass of the model. This forces the model to learn more robust features and prevents the model from becoming too dependent on specific neurons, which can lead to overfitting.
    (linear1): Linear(in_features=1, out_features=512, bias=True) ### here it takes 1 gene expression value from each gene for each cell and transforms to 512 dimension provide continous representation of gene expression.
    (activation): ReLU() ### This adds the non linearity in the model to learn it in a better way
    (linear2): Linear(in_features=512, out_features=512, bias=True)
    (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    '''
    The ValueEncoder transforms the continuous gene expression values into embeddings that align with the gene embeddings produced by the GeneEncoder. 
    These embeddings will help the Transformer model understand the relationships between the genes in terms of both their identity and their expression levels.
    '''
  )
  (batch_encoder): BatchLabelEncoder(
    (embedding): Embedding(156, 512) ### 156 batches on 512 dimensions
    (enc_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    '''
    Instead of applying a single batch normalization layer to the entire input, DomainSpecificBatchNorm1d applies a separate BatchNorm for each batch label (domain). 
    This could be useful in situations where you have data from different sources or conditions (i.e., different "domains") and you want to normalize the features
    separately for each domain.
    '''
  (dsbn): DomainSpecificBatchNorm1d(
    (bns): ModuleList(
      (0-155): 156 x BatchNorm1d(512, eps=6.1e-05, momentum=0.1, affine=False, track_running_stats=True)
    )
  )
#   After the GeneEncoder, ValueEncoder, and BatchEncoder produce their respective embeddings, the outputs are combined. These combined embeddings are fed into the Transformer Encoder, and the Multihead Attention mechanism begins.
  (transformer_encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-11): 12 x TransformerEncoderLayer( 
      # ModuleList: This is a container that holds the 12 Transformer Encoder layers. So, the model has 12 layers of Transformer Encoder stacked on top of each other. The layers are indexed from 0 to 11 (12 layers in total).
      # Earlier in the hyper parameter it is 4 encoder blocks. But 12 layer of encoder blocks represents that model was pretrained on 12 encoder block and in fine-tuning you use 4 encoder blocks
        (self_attn): MultiheadAttention(
          (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
        )
        (linear1): Linear(in_features=512, out_features=512, bias=True)
        (dropout): Dropout(p=0.2, inplace=False)
        (linear2): Linear(in_features=512, out_features=512, bias=True)
        (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (dropout1): Dropout(p=0.2, inplace=False)
        (dropout2): Dropout(p=0.2, inplace=False)
      )
    )
  )
'''
How the GeneEncoder, ValueEncoder, and BatchEncoder Interact in Multihead Attention:
When the embeddings from the GeneEncoder, ValueEncoder, and BatchEncoder are passed into the Transformer Encoder:
They are all concatenated or combined into a single input sequence of embeddings (of size 512 for each gene, value, or batch embedding).
This combined sequence is then processed by the Multihead Attention mechanism in the Transformer Encoder, where the model learns the dependencies between different genes, their expression values, and batch-related information.
The Multihead Attention computes how much attention each part of the input (gene, value, or batch) should give to the other parts. For example, it might learn that certain genes interact in a particular way depending on their expression values and the batch they belong to.
The attention mechanism will help the model learn complex relationships and patterns, such as how gene expression across different genes might influence each other or how batch effects might affect the overall gene expression pattern.

Summary of Decoder Components:
ExprDecoder: Predicts continuous gene expression values and possibly zero log probabilities for missing data.
ClsDecoder: Used for classification tasks, predicts a single value that represents a class or a regression output.
MVCDecoder: Focuses on generating predictions for multi-view consistency, useful for multi-condition or multi-task learning.
Adversarial Discriminator: Helps the model to learn domain-invariant representations, likely used for batch effect mitigation.
Similarity: Computes cosine similarity between vectors, used for measuring how similar predicted outputs are.
CrossEntropyLoss: Used to train the Adversarial Discriminator for batch prediction tasks.

 MVCDecoder (Multi-View Consistency Decoder):
    This decoder is responsible for generating multi-view consistency predictions. It seems like it might be used to learn a representation of gene expressions under different conditions or views.
    The discriminator helps the model learn to distinguish features that are invariant to domain shifts (e.g., batch-specific variations in gene expression). This is 
    helpful if your data has significant batch effects, and you want to learn representations that generalize well across different batches.

'''
  (decoder): ExprDecoder(
      # Purpose: Predict gene expression values for masked genes (Masked Value Prediction, like MLM in BERT).
      # fc branch: Predicts the actual continuous expression value of the gene.
      # zero_logit branch: Predicts whether a gene is expressed or not (zero vs non-zero expression).
      # Single-cell RNA-seq data is very sparse (lots of zeros due to dropout), so having this branch helps the model handle sparsity better.
    (fc): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): LeakyReLU(negative_slope=0.01) ## To introduce non-linearity
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): LeakyReLU(negative_slope=0.01)
      (4): Linear(in_features=512, out_features=1, bias=True)
    )
    (zero_logit): Sequential(
      (0): Linear(in_features=1024, out_features=512, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): Linear(in_features=512, out_features=512, bias=True)
      (3): LeakyReLU(negative_slope=0.01)
      (4): Linear(in_features=512, out_features=1, bias=True)
    )
  )
  (cls_decoder): ClsDecoder(
      # Purpose: Decode the [CLS] token embedding into higher-level predictions.
      # Examples:
      # Cell type classification
      # Predicting a biological label (like disease state, condition, etc.)
      # The out_layer reduces the 512-dim CLS embedding → 1 (binary classification) or more (if multi-class).
    (_decoder): ModuleList(
      (0): Linear(in_features=512, out_features=512, bias=True)
      (1): ReLU()
      (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (3): Linear(in_features=512, out_features=512, bias=True)
      (4): ReLU()
      (5): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (out_layer): Linear(in_features=512, out_features=1, bias=True)
  )
  (mvc_decoder): MVCDecoder(
    # Purpose: Supports contrastive learning by aligning representations from multiple “views” of the data.
    # e.g., a cell might be represented by its genes and by its expression values → the model learns consistent embeddings across these “views.”
    # gene2query: Projects embeddings into a query space.
    # W and W_zero_logit: Help reconstruct features under different masking strategies.
    # Here, the decoder doesn’t reduce to a single scalar.
    # Instead, it projects the embedding into a higher-dimensional "query space" (1024 dims).
    # This creates richer representations for contrastive similarity calculations (cosine similarity, InfoNCE loss, etc.).
    # The model compares embeddings in this 1024-D space to check if two views of the same cell are "close" or not.
    #  Think of it like this:
    # ExprDecoder → “What is the number for this gene?”
    # ClsDecoder → “What is the label for this cell?”
    # MVCDecoder → “Give me a rich vector representation so I can compare cells in different views.”
    (gene2query): Linear(in_features=512, out_features=512, bias=True)
    (query_activation): Sigmoid()
    (W): Linear(in_features=512, out_features=1024, bias=False)
    (W_zero_logit): Linear(in_features=512, out_features=1024, bias=True)
  )
  (grad_reverse_discriminator): AdversarialDiscriminator(
    (_decoder): ModuleList(
      (0): Linear(in_features=512, out_features=512, bias=True)
      (1): LeakyReLU(negative_slope=0.01)
      (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
      (3): Linear(in_features=512, out_features=512, bias=True)
      (4): LeakyReLU(negative_slope=0.01)
      (5): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
    (out_layer): Linear(in_features=512, out_features=156, bias=True)
  )
    #   After the model processes the data, the cosine similarity can be used to measure how similar the predicted output is to some reference (or between different parts of the model's output).
  (sim): Similarity(
    (cos): CosineSimilarity()
  )
  (creterion_cce): CrossEntropyLoss()
)

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)
#endregion

#region training the model
def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    This function trains the model for one epoch:
    Loops over mini-batches from the DataLoader.
    Moves data to GPU (device).
    Runs a forward pass through the model → gets predictions (output_dict).
    Computes different losses depending on tasks enabled (MLM, GEPC, ECS, DAB…).
    Sums them up → total loss.
    Runs backpropagation → updates weights.
    Logs metrics along the way (loss, MSE, etc.).
    """
    # Initialization
    model.train()
    total_loss, total_mse, total_gepc = 0.0, 0.0, 0.0
    total_error = 0.0
    log_interval = config.log_interval
    start_time = time.time()
    num_batches = len(loader)
    # You iterate through your batches from the DataLoader.
    # Each batch has: gene_ids (token IDs), values (masked gene expression), target_values (ground-truth expression), batch_labels (domain or dataset label)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token]) # This mask tells the model which positions are padded, so it can ignore them.
        # Performs forward pass with automatic mixed precision if enabled.
        """
        Returns a dictionary output_dict with:
            mlm_output: main prediction (masked value prediction)
            mlm_zero_probs: predicted zero probability
            mvc_output: GEPC output (if enabled)
            GEPC stands for Gene Expression Prediction from Context.
            It's a kind of auxiliary training objective used in models like scGPT to improve the model’s understanding of gene-gene relationships and help it generalize better.
            dab_output: classification (e.g., domain)
            loss_ecs: extra constraint loss (if enabled)
        """
        with torch.cuda.amp.autocast(enabled=config.amp):
            # forward passs
            '''
            mlm_output → predictions for masked values.
            mlm_zero_probs → predicts if a gene is truly zero (dropout) vs masked.
            mvc_output → GEPC auxiliary predictions.
            dab_output → domain classifier output (for adversarial batch correction).
            loss_ecs → extra loss if elastic cell similarity is enabled.
            '''
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if DSBN else None,
                MVC=config.GEPC,
                ECS=config.ecs_thres > 0,
            )
            masked_positions = input_values.eq(mask_value)  # the postions to predict
            loss = loss_mse = criterion( #Standard MSE loss → predict the masked expression values.
                # Even though the inputs are tokenized, scGPT’s main objective is to predict continuous expression values. 
                # That’s why MSE is the core loss. Cross-entropy is used only for side tasks (batch classification, zero inflation).
                output_dict["mlm_output"], target_values, masked_positions
            )
            '''
            Extra loss terms (conditional)
            Zero probability loss (Bernoulli log-likelihood):
            Encourages model to distinguish true zeros from masked ones.
            GEPC (Gene Expression Prediction from Context):
            Extra regression loss on mvc_output.
            GEPC zero prob loss: same as above but for zero logits.
            ECS (Elastic Cell Similarity):
            Contrastive loss → similar cells closer in embedding space.
            DAB (Domain Adversarial Batch correction):
            Cross-entropy loss from batch classifier head, scaled by dab_weight.
            ''''
            metrics_to_log = {"train/mse": loss_mse.item()}
            if explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})
            if config.GEPC:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/mvc": loss_gepc.item()})
            if config.GEPC and explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update(
                    {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
                )
            if config.ecs_thres > 0:
                loss_ecs = 10 * output_dict["loss_ecs"]
                loss = loss + loss_ecs
                metrics_to_log.update({"train/ecs": loss_ecs.item()})
            loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            loss = loss + config.dab_weight * loss_dab # combined loss
            metrics_to_log.update({"train/dab": loss_dab.item()})
        # Backpropagation
        # Uses AMP (Automatic Mixed Precision) for faster training on GPU.
        # Clip gradients to prevent exploding gradients.
        # Optimizer step updates model weights.
        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()
        wandb.log(metrics_to_log)
        with torch.no_grad():
            mre = masked_relative_error(
                output_dict["mlm_output"], target_values, masked_positions
            )
        total_loss += loss.item()
        total_mse += loss_mse.item()
        total_gepc += loss_gepc.item() if config.GEPC else 0.0
        total_error += mre.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_gepc = total_gepc / log_interval if config.GEPC else 0.0
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} | mre {cur_error:5.2f} |"
                + (f"gepc {cur_gepc:5.2f} |" if config.GEPC else "")
            )
            total_loss = 0
            total_mse = 0
            total_gepc = 0
            total_error = 0
            start_time = time.time()

# This training loop is multi-task:
# 🎯 Regression (MLM, GEPC) → predict expression.
# 🎯 Adversarial classification (DAB) → remove batch effect.
# 🎯 Contrastive (ECS) → keep biological neighbors close.
# 🎯 Zero vs masked discrimination → better dropout modeling.
# The final training signal is the weighted combination of all these objectives.

def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")

def evaluate(model: nn.Module, loader: DataLoader) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if DSBN else None,
                )
                output_values = output_dict["mlm_output"]
                masked_positions = input_values.eq(mask_value)
                loss = criterion(output_values, target_values, masked_positions)
                loss_dab = criterion_dab(output_dict["dab_output"], batch_labels)
            total_loss += loss.item() * len(input_gene_ids)
            total_error += masked_relative_error(
                output_values, target_values, masked_positions
            ).item() * len(input_gene_ids)
            total_dab += loss_dab.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)
    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/mre": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + config.dab_weight * total_dab)
            / total_num,
            "epoch": epoch,
        },
    )
    return total_loss / total_num, total_error / total_num


def eval_testdata(
    model: nn.Module,
    adata_t: AnnData,
    include_types: List[str] = ["cls"],
) -> Optional[Dict]:
    """evaluate the model on test dataset of adata_t"""
    model.eval()
    # copy adata_t to avoid reuse previously computed results stored in adata_t
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
    # Evaluate cls cell embeddings
    if "cls" in include_types:
        logger.info("Evaluating cls cell embeddings")
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
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=config.amp):
            cell_embeddings = model.encode_batch(
                all_gene_ids,
                all_values.float(),
                src_key_padding_mask=src_key_padding_mask,
                batch_size=config.batch_size,
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
        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["str_batch"],
            title=[f"batch, avg_bio = {results.get('avg_bio', 0.0):.4f}"],
            frameon=False,
            return_fig=True,
            show=False,
        )
        results["batch_umap"] = fig
        sc.pp.neighbors(adata_t, use_rep="X_scGPT")
        sc.tl.umap(adata_t, min_dist=0.3)
        fig = sc.pl.umap(
            adata_t,
            color=["celltype"],
            title=[
                f"celltype, avg_bio = {results.get('avg_bio', 0.0):.4f}",
            ],
            frameon=False,
            return_fig=True,
            show=False,
        )
        results["celltype_umap"] = fig
    if len(include_types) == 1:
        return results

### After initialing time to Finetune scGPT with task specific objective
best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()

criterion = masked_mse_loss
criterion_dab = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
)
# Reduces the learning rate every epoch (step_size=1)
# Multiplies LR by gamma=config.schedule_ratio (e.g., 0.9 = 10% decay per epoch)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# from torch.cuda.amp import autocast, GradScaler
# scaler = GradScaler()
# def train(model, loader):
#     model.train()
#     for batch in loader:
#         optimizer.zero_grad()
#         # Move all tensors to GPU
#         batch = {k: v.to(device) for k, v in batch.items()}
#         with autocast():  # <- Mixed precision context
#             output = model(batch)
#             loss = compute_loss(output, batch)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

for epoch in range(1, config.epochs + 1):
    epoch_start_time = time.time()
    train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
    print("loading train loader")
    train_loader = prepare_dataloader(
        train_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=True,
        drop_last=False,
    )
    print("finished train loader")
    print("loading valid loader")
    valid_loader = prepare_dataloader(
        valid_data_pt,
        batch_size=config.batch_size,
        shuffle=False,
        intra_domain_shuffle=False,
        drop_last=False,
    )
    print("finished valid loader")
    print("model training for train loader")
    if config.do_train:
        train(
            model,
            loader=train_loader,
        )
    print("model training for valid loader")
    val_loss, val_mre = evaluate(
        model,
        loader=valid_loader,
    )
    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info(
        f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
        f"valid loss/mse {val_loss:5.4f} | mre {val_mre:5.4f}"
    )
    logger.info("-" * 89)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)
        best_model_epoch = epoch
        logger.info(f"Best model with score {best_val_loss:5.4f}")
    if epoch % config.save_eval_interval == 0 or epoch == config.epochs:
        logger.info(f"Saving model to {save_dir}")
        torch.save(best_model.state_dict(), save_dir / f"model_e{best_model_epoch}.pt")
        # eval on testdata
        results = eval_testdata(
            best_model,
            adata_t=adata_sorted if per_seq_batch_sample else adata,
            include_types=["cls"],
        )
        results["batch_umap"].savefig(
            save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png", dpi=300
        )
        results["celltype_umap"].savefig(
            save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png", dpi=300
        )
        metrics_to_log = {"test/" + k: v for k, v in results.items()}
        metrics_to_log["test/batch_umap"] = wandb.Image(
            str(save_dir / f"embeddings_batch_umap[cls]_e{best_model_epoch}.png"),
            caption=f"celltype avg_bio epoch {best_model_epoch}",
        )
        metrics_to_log["test/celltype_umap"] = wandb.Image(
            str(save_dir / f"embeddings_celltype_umap[cls]_e{best_model_epoch}.png"),
            caption=f"celltype avg_bio epoch {best_model_epoch}",
        )
        metrics_to_log["test/best_model_epoch"] = best_model_epoch
        wandb.log(metrics_to_log)
        wandb.log({"avg_bio": results.get("avg_bio", 0.0)})
    scheduler.step()

## save the best model
torch.save(best_model.state_dict(), save_dir / "best_model.pt")

artifact = wandb.Artifact(f"best_model", type="model")
glob_str = os.path.join(save_dir, "best_model.pt")
artifact.add_file(glob_str)
run.log_artifact(artifact)

run.finish()
wandb.finish()
gc.collect()


#endregion
