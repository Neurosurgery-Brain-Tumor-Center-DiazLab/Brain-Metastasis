#region Testing 
### Installing scGPT
# conda create -n py_r_env python=3.8 r-base=4.0
# conda activate py_r_env
## Since torch text is not working with cuda 12.4 toolkit
# pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 torchtext==0.18.0 --extra-index-url https://download.pytorch.org/whl/cu118

# Testing if CUDA is installed
import torch
print(torch.__version__)  # Should match the installed version
print(torch.cuda.is_available())  # Should return True if CUDA is set up correctly

print("CUDA available:", torch.cuda.is_available())

# If available, print some basic info
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("CUDA version:", torch.version.cuda)
else:
    print("CUDA not available. Using CPU.")

# Create a tensor and move it to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(3, 3).to(device)

print("Tensor on device:", x.device)
print(x)

# pip install ipython
# pip install scgpt "flash-attn<1.0.5" ## to install scGPT
# pip install wandb ## for logging and visualization
### Performing Zero Shot integration
# This tutorial covers the zero-shot integration with scGPT. This particular workflow works for scRNA-seq datasets without fine-tuning 
# (or any extensive training) of scGPT.
# We will use the Lung-Kim dataset as an example. The dataset comprises 14 primary human lung adenocarcinoma samples and 32,493 cells. 
# This dataset is publicly accessible via the Curated Cancer Cell Atlas, which can be downloaded from here. You may place the dataset 
# under data directory at the outer level.
#endregion

# Import scGPT and dependencies
### Loading the modules and running it
# https://github.com/bowang-lab/scGPT/blob/main/tutorials/zero-shot/Tutorial_ZeroShot_Integration.ipynb
from pathlib import Path
import warnings

import scanpy as sc
import scib
import numpy as np
import sys

sys.path.insert(0, "../")

import scgpt as scg
import matplotlib.pyplot as plt
import anndata

plt.style.context('default')
warnings.simplefilter("ignore", ResourceWarning)

#region zeroshot
adata = sc.read_h5ad("/diazlab/data3/.abhinav/projects/Brain_metastasis/prim_metastasis/saveh5ad/adata_combined.h5ad")
adata.var["features"] = adata.var.index
adata.obs['cellType'] = adata.obs['cellType'].replace('Erythroblast', 'Unknown')
adata.obs['Celltypes'] = adata.obs['cellType']
adata.obs['Celltypes'] = adata.obs['Celltypes'].replace("Unknown","Unknown_celltypes") ## so that unknown cancer type and cell type should not be same
adata.obs['type'] = adata.obs['type'].replace("Unknown","Unknown_cancertype") ## so that unknown cancer type and cell type should not be same

### Replacing the Tumor with the different types of the cancers
# Make sure both columns are accessible
adata.obs['Celltypes'] = adata.obs['Celltypes'].astype(str) ### to remove the categories
mask = adata.obs['Celltypes'].str.contains('Tumor', na=False)
adata.obs.loc[mask, 'Celltypes'] = adata.obs.loc[mask, 'type']
# Keep only rows that do NOT contain "Unknown"
adata = adata[~adata.obs['Celltypes'].str.contains("Unknown", na=False)].copy()


# https://github.com/bowang-lab/scGPT/blob/main/tutorials/zero-shot/Tutorial_ZeroShot_Integration_Continual_Pretraining.ipynb
# Further now performing the integration using Tabula Sapiens using model underwent continual pretraining (CP) scGPT_CP
model_dir_CP = Path("/diazlab/data3/.abhinav/projects/Brain_metastasis/scgpt/resources/scGPT_CP")
gene_col = "features"
cell_type_key = "Celltypes"
batch_key = "sample"
N_HVG = 3000

# Remove unannotated cells:
celltype_id_labels = adata.obs[cell_type_key].astype("category").cat.codes.values
adata = adata[celltype_id_labels >= 0] ### if there is a missing or NaN that will be -1 

org_adata = adata.copy() ## Making copy of the dataset

# highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG, flavor='seurat_v3')
adata = adata[:, adata.var['highly_variable']]

# Generate the cell embeddings
# Now we will generate the cell embeddings for the dataset using embed_data function. embed_data calculates the cell embedding for each cell with the given 
# scGPT model. The extracted embedding is stored in the X_scGPT field of obsm in AnnData.
embed_adata_CP = scg.tasks.embed_data(
    adata,
    model_dir_CP,
    gene_col=gene_col,
    batch_size=64,
)

# Visualize the integration performance
# UMAP of scGPT embedding colored by cell type:
sc.pp.neighbors(embed_adata_CP, use_rep="X_scGPT")
sc.tl.umap(embed_adata_CP)

### Combining the cell types as T to T cells,  MDMs/Monocytes/Macrophage 1 to macrophage
embed_adata_CP.obs['Celltypes']=embed_adata_CP.obs['Celltypes'].replace("T","T_cell")
embed_adata_CP.obs['Celltypes']=embed_adata_CP.obs['Celltypes'].replace("MDMs/Monocytes/Macrophage 1 ","Macrophage")
embed_adata_CP.obs['Celltypes']=embed_adata_CP.obs['Celltypes'].replace("Plasma_cell","B/Plasma")
embed_adata_CP.obs['Celltypes']=embed_adata_CP.obs['Celltypes'].replace("Fibroblasts","Fibroblast")

import os
os.chdir("/diazlab/data3/.abhinav/projects/Brain_metastasis/prim_metastasis/zero_shot/")
sc.pl.umap(embed_adata_CP, 
           color=[cell_type_key, batch_key], 
           frameon=False, 
           wspace=0.4, 
           legend_loc="on data",
           legend_fontsize=6,
           title=["scGPT zero-shot: cell type", "scGPT zero-shot: batch label"],
           save="CP_integration_Brain_metasis_nolabel_combined.pdf")

embed_adata_CP.obs['batch']=embed_adata_CP.obs['batch'].replace("0","Primary")
embed_adata_CP.obs['batch']=embed_adata_CP.obs['batch'].replace("1","Metastasis")

sc.pl.umap(embed_adata_CP, 
           color=["batch"], 
           frameon=False, 
           wspace=0.4, 
        #    legend_loc="on data",
           legend_fontsize=10,
           title=["scGPT zero-shot: Primary/Metastasis"],
           save="CP_Brain_metastasis_batches_cancer_type.pdf")

sc.pl.umap(embed_adata_CP, 
           color=["type","batch"], 
           frameon=False, 
           wspace=0.4, 
           legend_loc="on data",
           legend_fontsize=10,
           title=["scGPT zero-shot: Cancer Type", "scGPT zero-shot: Primary/Metastasis"],
           save="CP_Brain_metastasis_batches_cancer_type.pdf")

sc.pl.umap(embed_adata_CP, 
           color=["cellType"], 
           frameon=False, 
           wspace=0.4, 
           legend_loc="on data",
           legend_fontsize=10,
           title=["scGPT zero-shot: Celltype"],
           save="CP_Brain_metastasis_celltype.pdf")

# Perform Leiden clustering with a resolution of 0.5
sc.tl.leiden(embed_adata_CP, resolution=0.8)

sc.pl.umap(embed_adata_CP, 
           color=['leiden'], 
           frameon=False, 
           wspace=0.4,
           legend_loc="on data",
           legend_fontsize=10,     
           title=["Leiden Clustering"],
           save="CP_Brain_clustering.pdf", label = True)

"""
Calculate the metrics for integration results
"""
def scib_eval(adata, batch_key, cell_type_key, embed_key):
    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=cell_type_key,
        embed=embed_key,
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )
    result_dict = results[0].to_dict()
    
    # compute avgBIO metrics
    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )
    
    # compute avgBATCH metrics
    result_dict["avg_batch"] = np.mean(
        [
            result_dict["graph_conn"],
            result_dict["ASW_label/batch"],
        ]
    )
    
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}
    
    return result_dict


scib_result_dict = scib_eval(
    embed_adata_CP,
    batch_key=batch_key,
    cell_type_key=cell_type_key,
    embed_key="X_scGPT",
)

print("AvgBIO: {:.4f}".format(scib_result_dict["avg_bio"]))
print("AvgBATCH: {:.4f}".format(scib_result_dict["avg_batch"]))

# Modify the 'var' and 'obs' columns in the raw data (if raw exists)
if embed_adata_CP.raw is not None:
    embed_adata_CP.raw = embed_adata_CP.raw.copy()  # Create a modifiable copy
    # Rename '_index' in the raw.var or raw.obs if it exists
    if '_index' in embed_adata_CP.raw.var.columns:
        embed_adata_CP.raw.var = embed_adata_CP.raw.var.rename(columns={'_index': 'index_renamed'})

embed_adata_CP.write('/diazlab/data3/.abhinav/projects/Brain_metastasis/prim_metastasis/zero_shot/saveh5ad/Brain_metastasis_zero_shot.h5ad')

##### Reference Mapping ########
# We provide two mode of reference mapping in the following:
# Using a customized reference dataset with provided annotations. Mapping unkown cells in a query set to this reference dataset.
# This illustrated the use case for users who already have annotations for similar samples and want to quickly transfer the annotation 
# to the newly collected samples.
# Using our previously collected over 33 million cells from CellXGene as reference. Mapping unkown cells in a query set to this reference atlas. 
# This illustrates a generic use case for users who want to map their own data to a large reference atlas. For example, this can be a fast first
# step to understand the cell composition of a newly collected samples.
# According to your use case, you may only need apply one of the two modes.
# Note: please note that the reference mapping is a new experimental feature.

# extra dependency for similarity search
try:
    import faiss
    faiss_imported = True
except ImportError:
    faiss_imported = False
    print(
        "faiss not installed! We highly recommend installing it for fast similarity search."
    )
    print("To install it, see https://github.com/facebookresearch/faiss/wiki/Installing-Faiss")

warnings.filterwarnings("ignore", category=ResourceWarning)

### You can use own reference dataset to perform the reference mapping. However for SHH MB I donot have any reference mapping so I have use CellXGene Atlas
# We have previously built the index for all the cells in normal or cancer samples, over 33 million cells in total. You can find the code to build the index 
# at build_atlas_index_faiss.py. We applied careful tuning to eventually well balance between the accuracy and efficiency. Now the actual building process takes less than 3 minutes and we choose to use only 16 bytes to store the vector per cell, which leads to 808 MB for the whole index of all the millions of cells. Please download the faiss index folder from https://drive.google.com/drive/folders/1q14U50SNg5LMjlZ9KH-n-YsGRi8zkCbe?usp=sharing.

# 1. Load optimized scGPT model (pre-trained or fine-tuned) and data
# 2. Retrieve scGPT's gene embeddings
# 3. Extract gene programs from scGPT's gene embedding network
# 4. Visualize gene program activations on dataset of interest
# 5. Visualize the interconnectivity of genes within select gene programs

#### Gene Regulatory Network inference
import copy
import json
import os
from pathlib import Path
import sys
import warnings

import torch
from anndata import AnnData
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import tqdm
import gseapy as gp

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")
import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

set_seed(42)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_hvg = 1200
n_bins = 51
mask_value = -1
pad_value = -2
n_input_bins = n_bins

# Specify model path; here we load the pre-trained scGPT blood model
model_dir = Path("/diazlab/data3/.abhinav/projects/Brain_metastasis/scgpt/resources/scGPT_CP")
model_config_file = model_dir / "args.json"
model_file = model_dir / "best_model.pt"
vocab_file = model_dir / "vocab.json"

vocab = GeneVocab.from_file(vocab_file)
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)

# Retrieve model parameters from config files
with open(model_config_file, "r") as f:
    model_configs = json.load(f)
print(
    f"Resume model from {model_file}, the model args will override the "
    f"config {model_config_file}."
)
embsize = model_configs["embsize"]
nhead = model_configs["nheads"]
d_hid = model_configs["d_hid"]
nlayers = model_configs["nlayers"]
n_layers_cls = model_configs["n_layers_cls"]

gene2idx = vocab.get_stoi()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  # size of vocabulary
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    pad_value=pad_value,
    n_input_bins=n_input_bins,
)

try:
    model.load_state_dict(torch.load(model_file))
    print(f"Loading all model params from {model_file}")
except:
    # only load params that are in the model and match the size
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_file)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    for k, v in pretrained_dict.items():
        print(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

model.to(device)

### Loading the SHH dataset
adata = sc.read_h5ad("/diazlab/data3/.abhinav/projects/SHH/snRNA/removed_samples_BO/saveRDS_obj/snRNA_diet.h5ad")
ori_batch_col = "sample"
adata.obs["cell_types"] = adata.obs["cell_types"].astype(str)
data_is_raw = False

# Preprocess the data following the scGPT data pre-processing pipeline
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
    binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
    result_binned_key="X_binned",  # the key in adata.layers to store the binned data
)
preprocessor(adata, batch_key="sample")


# Step 2: Retrieve scGPT’s gene embeddings
# Note that technically scGPT’s gene embeddings are data independent. Overall, the pre-trained foundation model contains 30+K genes. Here for simplicity,
# we focus on a subset of HVGs specific to the data at hand.

# Retrieve the data-independent gene embeddings from scGPT
gene_ids = np.array([id for id in gene2idx.values()])
gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
gene_embeddings = gene_embeddings.detach().cpu().numpy()

# Filter on the intersection between the SHH MB HVGs found in step 1.2 and scGPT's 30+K foundation model vocab
gene_embeddings = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys()) if gene in adata.var.index.tolist()}
print('Retrieved gene embeddings for {} genes.'.format(len(gene_embeddings)))

# Construct gene embedding network
embed = GeneEmbedding(gene_embeddings)

# Perform Louvain clustering with desired resolution; here we specify resolution=40
gdata = embed.get_adata(resolution=40)
# Retrieve the gene clusters
metagenes = embed.get_metagenes(gdata)

# Obtain the set of gene programs from clusters with #genes >= 5
mgs = dict()
for mg, genes in metagenes.items():
    if len(genes) > 4:
        mgs[mg] = genes

mgs
#endregion
