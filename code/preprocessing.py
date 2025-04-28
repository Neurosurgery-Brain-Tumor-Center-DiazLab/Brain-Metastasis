# conda activate py_r_env
#region Loading packages
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

#region Metastasis
#endregion

#region preprocessing
import scanpy as sc
import pandas as pd
data_dir = '/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdata/' 
adata = sc.read_mtx(f"{data_dir}/brain_met_count.mtx")
adata = adata.transpose()

### Loading Genes
genes = pd.read_csv(f"{data_dir}/genes.csv", header=0)
adata.var_names = genes.iloc[:, 0].astype(str)

# Load cell barcodes
barcodes = pd.read_csv(f"{data_dir}/barcodes.csv", header=0)
adata.obs_names = barcodes.iloc[:, 0].astype(str)

metadata = pd.read_csv(f"{data_dir}/metadata.csv", index_col=0, header=0)
adata.obs = metadata.loc[adata.obs_names]
# adata.write("/diazlab/data3/.abhinav/projects/Brain_metastasis/data/brain_meta.h5ad")
adata = sc.read_h5ad(f"{data_dir}brain_meta.h5ad")

#### Filtering the dataset
adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
sc.pp.calculate_qc_metrics(
    adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)
save_dir = "/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/analysis/"
os.makedirs(f"{save_dir}/QC", exist_ok=True)
os.chdir(f"{save_dir}QC")
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"],
    jitter=0.4,
    size = 0,
    multi_panel=True,
    save="brain_met_violin.pdf" 
)
plt.close()

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))  
sample_counts = adata.obs['sample'].value_counts()
sample_counts.plot(kind='bar')
plt.xlabel('Sample')
plt.ylabel('Number of Cells')
plt.title('Cell Distribution per Sample')
plt.xticks(rotation=90, fontsize = 5)
plt.tight_layout()
plt.savefig("/diazlab/data3/.abhinav/projects/Brain_metastasis/analysis/QC/figures/meta_cell_distribution.pdf", dpi=300)
plt.close()

### Removing low quality cell, samples, and genes
samples_to_keep = sample_counts[sample_counts >= 500].index
adata = adata[adata.obs['sample'].isin(samples_to_keep)].copy()
adata_copy = adata.copy()
adata = adata_copy.copy()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.scrublet(adata, batch_key="sample")
adata_clean = adata[~adata.obs['predicted_doublet'], :]
adata_clean.write("/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/analysis/saveh5ad/adata_metastasis_clean.h5ad")
#endregion

#region Primary
#endregion

#region preprocessing
import scanpy as sc
import pandas as pd
data_dir = '/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/rawdata/' 
adata_pri = sc.read_mtx(f"{data_dir}/brain_met_count.mtx")
adata_pri = adata_pri.transpose()

### Loading Genes
genes = pd.read_csv(f"{data_dir}/genes.csv", header=0)
adata_pri.var_names = genes.iloc[:, 0].astype(str)

# Load cell barcodes
barcodes = pd.read_csv(f"{data_dir}/barcodes.csv", header=0)
adata_pri.obs_names = barcodes.iloc[:, 0].astype(str)

metadata = pd.read_csv(f"{data_dir}/metadata.csv", index_col=0, header=0)
adata_pri.obs = metadata.loc[adata_pri.obs_names]
# adata_pri.write("/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/rawdata/brain_primary_meta.h5ad")
adata_pri = sc.read_h5ad("/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/rawdata/brain_primary_meta.h5ad")

#### Filtering the dataset
adata_pri.var["mt"] = adata_pri.var_names.str.startswith("MT-")
adata_pri.var["ribo"] = adata_pri.var_names.str.startswith(("RPS", "RPL"))
adata_pri.var["hb"] = adata_pri.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    adata_pri, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)
save_dir = "/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/analysis/"
os.makedirs(f"{save_dir}/QC", exist_ok=True)
os.chdir(f"{save_dir}QC")
sc.pl.violin(
    adata_pri,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"],
    jitter=0.4,
    size = 0,
    multi_panel=True,
    save="brain_primary_violin.pdf" 
)
plt.close()

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))  
sample_counts = adata_pri.obs['sample'].value_counts()
sample_counts.plot(kind='bar')

plt.xlabel('Sample')
plt.ylabel('Number of Cells')
plt.title('Cell Distribution per Sample')
plt.xticks(rotation=90, fontsize = 5)
plt.tight_layout()
plt.savefig(f'{save_dir}QC/figures/meta_cell_distribution.pdf', dpi=300)
plt.close()

### Removing 
# samples with less than 500 cells
# cells with less than 100 genes
# genes which expressed in less than 4 cells
samples_to_keep = sample_counts[sample_counts >= 500].index
adata_pri = adata_pri[adata_pri.obs['sample'].isin(samples_to_keep)].copy()
sc.pp.filter_cells(adata_pri, min_genes=200)
sc.pp.filter_genes(adata_pri, min_cells=3)
sc.pp.scrublet(adata_pri, batch_key="sample")

adata_clean = adata_pri[~adata_pri.obs['predicted_doublet'], :]
adata_clean.var["mt"] = adata_clean.var_names.str.startswith("MT-")
adata_clean.var["ribo"] = adata_clean.var_names.str.startswith(("RPS", "RPL"))
adata_clean.var["hb"] = adata_clean.var_names.str.contains("^HB[^(P)]")

sc.pp.calculate_qc_metrics(
    adata_clean, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True
)

import os
save_dir = "/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/analysis/"
os.makedirs(f"{save_dir}/QC", exist_ok=True)
os.chdir(f"{save_dir}QC")
sc.pl.violin(
    adata_clean,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt", "pct_counts_ribo", "pct_counts_hb"],
    jitter=0.4,
    size = 0,
    multi_panel=True,
    save="brain_primary_violin_clean.pdf" 
)
plt.close()

adata_clean.write("/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/analysis/saveh5ad/adata_primary_clean.h5ad")

#endregion

#region combining
adata_pri = sc.read_h5ad("/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/analysis/saveh5ad/adata_primary_clean.h5ad")
adata_met = sc.read_h5ad("/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/analysis/saveh5ad/adata_metastasis_clean.h5ad")
pd.set_option('display.max_rows', None)  # Show all rows
print(genes_df)
pd.reset_option('display.max_rows')  # Reset back to default
adata_combined = adata_pri.concatenate(adata_met, join='inner', index_unique='-') ### would like to include only those genes present in both the primary and metastasis
ribosomal_genes = adata_combined.var.index.str.contains('^RPS|^RPL', regex=True)
adata_combined_no_ribo = adata_combined[:, ~ribosomal_genes]

# Normalization
# The next preprocessing step is normalization. A common approach is count depth scaling with subsequent log plus one (log1p) transformation. Count depth scaling normalizes 
# the data to a “size factor” such as the median count depth in the dataset, ten thousand (CP10k) or one million (CPM, counts per million). The size factor for count depth 
# scaling can be controlled via target_sum in pp.normalize_total. We are applying median count depth normalization with log1p transformation (AKA log1PF).

adata_combined_no_ribo.layers["counts"] = adata_combined_no_ribo.X.copy()
sc.pp.normalize_total(adata_combined_no_ribo)
sc.pp.log1p(adata_combined_no_ribo)
adata_combined_no_ribo.write_h5ad("/diazlab/data3/.abhinav/projects/Brain_metastasis/prim_metastasis/saveh5ad/adata_combined.h5ad")

#endregion