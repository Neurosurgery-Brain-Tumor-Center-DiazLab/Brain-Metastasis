''' Attention-based GRN Inference on Fine-tuned Model
Here we use the fine-tuned blood model on the Adamson perturbation dataset as an example of the cell-state specific GRN inference via attention weights.
scGPT outputs attention weights on the individual cell level, which can be further aggregated by cell states. In this particular example, we compare the
most influenced genes between a transcription factor repression condition (perturbed) and the control. However, this attention-based GRN inference is not
restricted to perturbation-based discoveries. It can also be used to compare between cell states in general, such as healthy v.s. diseased,
undifferentiated v.s. differentiated, as a broader application.
Users may perform scGPT's attention-based GRN inference in the following steps:
 1. Load fine-tuned scGPT model and data
 2. Retrieve scGPT's attention weights by condition (i.e., cell states)
 3. Perform scGPT's rank-based most influenced gene selection
 4. Validate the most influenced gene list against existing databases
NOTE in advance: to run this tutorial notebook, you may need to download the fine-tuned model from link and the list of targets of BHLHE40 from CHIP-Atlas for
evaluation from link. -->
'''
# conda activate /diazlab/data3/.abhinav/tools/miniconda3/envs/py_r_env

#region Zero Shot GRN
# https://github.com/bowang-lab/scGPT/blob/main/tutorials/Tutorial_GRN.ipynb
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


# Step 1: Load pre-trained model and dataset
# Specify model path; here we load the pre-trained scGPT blood model
model_dir = Path("/diazlab/data3/.abhinav/projects/Brain_metastasis/scgpt/resources/scGPT_CP/")
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

# TransformerModel(
#   (encoder): GeneEncoder(
#     (embedding): Embedding(60697, 512, padding_idx=60694)
#     (enc_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#   )
#   (value_encoder): ContinuousValueEncoder(
#     (dropout): Dropout(p=0.5, inplace=False)
#     (linear1): Linear(in_features=1, out_features=512, bias=True)
#     (activation): ReLU()
#     (linear2): Linear(in_features=512, out_features=512, bias=True)
#     (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#   )
#   (transformer_encoder): TransformerEncoder(
#     (layers): ModuleList(
#       (0-11): 12 x TransformerEncoderLayer(
#         (self_attn): MultiheadAttention(
#           (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
#         )
#         (linear1): Linear(in_features=512, out_features=512, bias=True)
#         (dropout): Dropout(p=0.5, inplace=False)
#         (linear2): Linear(in_features=512, out_features=512, bias=True)
#         (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         (dropout1): Dropout(p=0.5, inplace=False)
#         (dropout2): Dropout(p=0.5, inplace=False)
#       )
#     )
#   )
#   (decoder): ExprDecoder(
#     (fc): Sequential(
#       (0): Linear(in_features=512, out_features=512, bias=True)
#       (1): LeakyReLU(negative_slope=0.01)
#       (2): Linear(in_features=512, out_features=512, bias=True)
#       (3): LeakyReLU(negative_slope=0.01)
#       (4): Linear(in_features=512, out_features=1, bias=True)
#     )
#   )
#   (cls_decoder): ClsDecoder(
#     (_decoder): ModuleList(
#       (0): Linear(in_features=512, out_features=512, bias=True)
#       (1): ReLU()
#       (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#       (3): Linear(in_features=512, out_features=512, bias=True)
#       (4): ReLU()
#       (5): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#     )
#     (out_layer): Linear(in_features=512, out_features=1, bias=True)
#   )
#   (sim): Similarity(
#     (cos): CosineSimilarity()
#   )
#   (creterion_cce): CrossEntropyLoss()
# )

# 1.2 Load dataset of interest

# Specify data path; here we load the Immune Human dataset
adata = sc.read_h5ad("/diazlab/data3/.abhinav/projects/Brain_metastasis/prim_metastasis/saveh5ad/adata_combined_processed.h5ad")
ori_batch_col = "str_batch"
adata.obs['celltype']=adata.obs['Celltypes'].astype(str)

### Since data is already processed
# Retrieve the data-independent gene embeddings from scGPT
gene_ids = np.array([id for id in gene2idx.values()])
gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
gene_embeddings = gene_embeddings.detach().cpu().numpy()

# Filter on the intersection between the our dataset HVGs found in step 1.2 and scGPT's 30+K foundation model vocab
gene_embeddings = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys()) if gene in adata.var.index.tolist()}
print('Retrieved gene embeddings for {} genes.'.format(len(gene_embeddings)))

# Construct gene embedding network
embed = GeneEmbedding(gene_embeddings)

# Step 3: Extract gene programs from gene embedding network
# 3.1 Perform Louvain clustering on the gene embedding network
# Perform Louvain clustering with desired resolution; here we specify resolution=40
gdata = embed.get_adata(resolution=40)
# Retrieve the gene clusters
metagenes = embed.get_metagenes(gdata)

# Obtain the set of gene programs from clusters with #genes >= 10
mgs = dict()
for mg, genes in metagenes.items():
    if len(genes) > 10:
        mgs[mg] = genes

savedir = "/diazlab/data3/.abhinav/projects/Brain_metastasis/GRN/zero_shot/"
os.chdir(savedir)
os.makedirs("figure", exist_ok = True)

embed.score_metagenes(adata, metagenes)

sns.set(font_scale=0.2)
plt.figure(figsize=(15, 15))  # adjust height as needed
embed.plot_metagenes_scores(adata, mgs, "celltype")
plt.savefig("metagene_scores.pdf", dpi=300, bbox_inches='tight')  # or .pdf, .svg, etc.

sns.set(font_scale=0.35)
plt.figure(figsize=(15, 15))  # adjust height as needed
embed.plot_metagenes_scores(adata, mgs, "histology")
plt.savefig("metagene_scores_histology.pdf", dpi=300, bbox_inches='tight')  # or .pdf, .svg, etc.

hist_order=adata.obs['histology'].unique()
adata.obs['histology_order'] = pd.Categorical(
    adata.obs['histology'],
    categories=hist_order,
    ordered=True
)

sns.set(font_scale=0.30)
plt.figure(figsize=(15, 15))  # adjust height as needed
embed.plot_metagenes_scores(adata, mgs, "histology_order")
plt.savefig("metagene_scores_histology_order.pdf", dpi=300, bbox_inches='tight')  # or .pdf, .svg, etc.

adata.obs['histology_subtype'] = adata.obs['histology'].str.split('_').str[-1]
sns.set(font_scale=0.35)
plt.figure(figsize=(15, 15))  # adjust height as needed
embed.plot_metagenes_scores(adata, mgs, "histology_subtype")
plt.savefig("metagene_scores_histology_subtypes_order.pdf", dpi=300, bbox_inches='tight')  # or .pdf, .svg, etc.


required_col = [m + "_SCORE" for m in mgs.keys()]
score_df = adata.obs.groupby('histology_order')[required_col].mean().T

from scipy.stats import zscore
score_df_scaled = score_df.apply(zscore, axis=1)

score_df_scaled = (score_df - score_df.mean(axis=1).values[:, None]) / score_df.std(axis=1).values[:, None]

# Min-max scaling: (x - min) / (max - min) per row
score_df_scaled = (score_df - score_df.min(axis=1).values[:, None]) / (score_df.max(axis=1).values[:, None] - score_df.min(axis=1).values[:, None])

plt.figure(figsize=(5, 5))  # Adjust size as needed
sns.heatmap(score_df_scaled, cmap='Blues')

plt.title("Min-Max Scaled Metagene Scores by Histology")
plt.tight_layout()
plt.savefig("minmax_scaled_metagene_scores.png", dpi=300)
plt.show()

# Plot clustered heatmap
g = sns.clustermap(
    score_df_scaled,
    cmap='Blues',             # or 'vlag', 'viridis', etc.
    figsize=(5, 10),
    standard_scale=None,      # You already scaled it manually
    row_cluster=True,
    col_cluster=False,
    # dendrogram_ratio=(0.1, 0.1),  # Adjust dendrogram size
    # cbar_pos=(0.02, 0.8, 0.03, 0.18)  # Optional: reposition colorbar
)


# Set font size of tick labels
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=10)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=10)

# Rotate x-axis labels if needed
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90,ha='center',fontsize=10)

# Save the full clustered heatmap
g.savefig("row_clustered_metagene_scores.pdf", dpi=300)

sorted_mgs = sorted(mgs.items(), key=lambda x: int(x[0]))
df = pd.DataFrame(sorted_mgs, columns=['Key', 'Value'])
df.to_csv("metagenes.csv",index = False)

histology_values = {
    'Uterine_BrM', 'Breast_NonBrM', 'Stomach_BrM', 'Renal_NonBrM',
    'Ovarian_NonBrM', 'Stomach_NonBrM', 'Colorectal_BrM', 'Esophageal_BrM',
    'Lung_BrM', 'Breast_BrM', 'Ovarian_BrM', 'Lung_NonBrM', 'Renal_BrM',
    'Colorectal_NonBrM', 'Melanoma_NonBrM', 'Unknown_BrM', 'Melanoma_BrM',
    'RMS_NonBrM'
}

sorted_histology = sorted(histology_values, key=lambda x: x.endswith('_BrM'))

adata.obs['histology_order'] = pd.Categorical(
    adata.obs['histology_order'],
    categories=sorted_histology,
    ordered=True
)

g = sns.clustermap(
    score_df_scaled.loc[:,sorted_histology],
    cmap='Blues',           # or 'vlag', 'viridis', etc.
    figsize=(5, 10),
    standard_scale=None,      # You already scaled it manually
    row_cluster=True,
    col_cluster=False,
    # dendrogram_ratio=(0.1, 0.1),  # Adjust dendrogram size
    # cbar_pos=(0.02, 0.8, 0.03, 0.18)  # Optional: reposition colorbar
)


# Set font size of tick labels
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=10)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=10)

# Rotate x-axis labels if needed
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90,ha='center',fontsize=10)

# Save the full clustered heatmap
g.savefig("row_clustered_metagene_scores_ordered_non_to_Brm.pdf", dpi=300)

### celltypes
required_col = [m + "_SCORE" for m in mgs.keys()]
score_df_celltype = adata.obs.groupby('celltype')[required_col].mean().T

# Min-max scaling: (x - min) / (max - min) per row
score_df_scaled = (score_df_celltype - score_df_celltype.min(axis=1).values[:, None]) / (score_df_celltype.max(axis=1).values[:, None] - score_df_celltype.min(axis=1).values[:, None])

plt.figure(figsize=(5, 5))  # Adjust size as needed
sns.heatmap(score_df_scaled, cmap='Blues')
plt.title("Min-Max Scaled Metagene Scores by Histology")
plt.tight_layout()
plt.savefig("figure/celltype_score_df_scaled.pdf", dpi=300)
plt.show()

# Plot clustered heatmap
g = sns.clustermap(
    score_df_scaled,
    cmap='Blues',             # or 'vlag', 'viridis', etc.
    figsize=(7, 10),
    standard_scale=None,      # You already scaled it manually
    row_cluster=True,
    col_cluster=True,
    # dendrogram_ratio=(0.1, 0.1),  # Adjust dendrogram size
    # cbar_pos=(0.02, 0.8, 0.03, 0.18)  # Optional: reposition colorbar
)

# Set font size of tick labels
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=10)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=10)

# Rotate x-axis labels if needed
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90,ha='center',fontsize=10)

# Save the full clustered heatmap
g.savefig("figure/celltype_row_clustered_metagene_scores.pdf", dpi=300)

sorted_mgs = sorted(mgs.items(), key=lambda x: int(x[0]))
df = pd.DataFrame(sorted_mgs, columns=['Key', 'Value'])
df.to_csv("metagenes.csv",index = False)

histology_values = {
    'Uterine_BrM', 'Breast_NonBrM', 'Stomach_BrM', 'Renal_NonBrM',
    'Ovarian_NonBrM', 'Stomach_NonBrM', 'Colorectal_BrM', 'Esophageal_BrM',
    'Lung_BrM', 'Breast_BrM', 'Ovarian_BrM', 'Lung_NonBrM', 'Renal_BrM',
    'Colorectal_NonBrM', 'Melanoma_NonBrM', 'Unknown_BrM', 'Melanoma_BrM',
    'RMS_NonBrM'
}

sorted_histology = sorted(histology_values, key=lambda x: x.endswith('_BrM'))

adata.obs['histology_order'] = pd.Categorical(
    adata.obs['histology_order'],
    categories=sorted_histology,
    ordered=True
)

g = sns.clustermap(
    score_df_scaled.loc[:,sorted_histology],
    cmap='Blues',           # or 'vlag', 'viridis', etc.
    figsize=(5, 10),
    standard_scale=None,      # You already scaled it manually
    row_cluster=True,
    col_cluster=False,
    # dendrogram_ratio=(0.1, 0.1),  # Adjust dendrogram size
    # cbar_pos=(0.02, 0.8, 0.03, 0.18)  # Optional: reposition colorbar
)


# Set font size of tick labels
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=10)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=10)

# Rotate x-axis labels if needed
g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90,ha='center',fontsize=10)

# Save the full clustered heatmap
g.savefig("row_clustered_metagene_scores_ordered_non_to_Brm.pdf", dpi=300)
#endregion


#region visualize network connectivity

# Retrieve gene program 3 which contains the CD3 gene set
CD_genes = mgs['8']
print(CD_genes)
# Compute cosine similarities among genes in this gene program
df = pd.DataFrame(columns=['Gene', 'Similarity', 'Gene1'])
all_dfs = []
for i in tqdm.tqdm(CD_genes):
    df = embed.compute_similarities(i, CD_genes)
    df['Gene1'] = i
    all_dfs.append(df)


df_CD = pd.concat(all_dfs, ignore_index=True)
df_CD_sub = df_CD[df_CD['Similarity']<0.99].sort_values(by='Gene') # Filter out edges from each gene to itself

# Creates a graph from the cosine similarity network
input_node_weights = [(row['Gene'], row['Gene1'], round(row['Similarity'], 2)) for i, row in df_CD_sub.iterrows()]
G = nx.Graph()
G.add_weighted_edges_from(input_node_weights)

# Plot the cosine similarity network; strong edges (> select threshold) are highlighted
thresh = 0.4
plt.figure(figsize=(10, 10))
widths = nx.get_edge_attributes(G, 'weight')

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > thresh]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= thresh]

pos = nx.spring_layout(G, k=0.4, iterations=15, seed=3)

width_large = {}
width_small = {}
for i, v in enumerate(list(widths.values())):
    if v > thresh:
        width_large[list(widths.keys())[i]] = v*10
    else:
        width_small[list(widths.keys())[i]] = max(v, 0)*10

nx.draw_networkx_edges(G, pos,
                       edgelist = width_small.keys(),
                    #    width=list(width_small.values()),
                       edge_color='lightblue',
                       alpha=0.8)
nx.draw_networkx_edges(G, pos,
                       edgelist = width_large.keys(),
                    #    width = list(width_large.values()),
                       alpha = 0.5,
                       edge_color = "blue",
                      )
# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels
# d = nx.get_edge_attributes(G, "weight")
# edge_labels = {k: d[k] for k in elarge}
# nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.savefig("GRN_score_score8.pdf", dpi=300)

# Step 6: Reactome pathway analysis
# Meta info about the number of terms (tests) in the databases
df_database = pd.DataFrame(
data = [['GO_Biological_Process_2021', 6036],
['GO_Molecular_Function_2021', 1274],
['Reactome_2022', 1818]],
columns = ['dataset', 'term'])


# Select desired database for query; here use Reactome as an example
databases = ['Reactome_2022']
m = df_database[df_database['dataset'].isin(databases)]['term'].sum()
# p-value correction for total number of tests done
p_thresh = 0.05/m

# Perform pathway enrichment analysis using the gseapy package in the Reactome database
df = pd.DataFrame()
enr_Reactome = gp.enrichr(gene_list=CD_genes,
                          gene_sets=databases,
                          organism='Human',
                          outdir='test/enr_Reactome_score8',
                          cutoff=0.5)
out = enr_Reactome.results
out = out[out['P-value'] < p_thresh]
# df = df.append(out, ignore_index=True)


def analyze_gene_program(CD_genes, program_name):
    print(f"Processing gene program: {program_name}")
    # Step 1: Compute cosine similarities
    all_dfs = []
    for gene in tqdm.tqdm(CD_genes):
        df = embed.compute_similarities(gene, CD_genes)
        df['Gene1'] = gene
        all_dfs.append(df)
    df_CD = pd.concat(all_dfs, ignore_index=True)
    # Filter out self-similarity edges
    df_CD_sub = df_CD[df_CD['Similarity'] < 0.99].sort_values(by='Gene')
    # Step 2: Build graph from similarity network
    input_node_weights = [
        (row['Gene'], row['Gene1'], round(row['Similarity'], 2))
        for _, row in df_CD_sub.iterrows()
    ]
    G = nx.Graph()
    G.add_weighted_edges_from(input_node_weights)
    # Step 3: Plot cosine similarity network
    thresh = 0.4
    plt.figure(figsize=(10, 10))
    widths = nx.get_edge_attributes(G, 'weight')
    elarge = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] > thresh]
    esmall = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] <= thresh]
    pos = nx.spring_layout(G, k=0.4, iterations=15, seed=3)
    width_large = {}
    width_small = {}
    for i, v in enumerate(list(widths.values())):
        if v > thresh:
            width_large[list(widths.keys())[i]] = v * 10
        else:
            width_small[list(widths.keys())[i]] = max(v, 0) * 10
    nx.draw_networkx_edges(G, pos,
                           edgelist=width_small.keys(),
                        #    width=list(width_small.values()),
                           edge_color='lightblue',
                           alpha=0.8)
    nx.draw_networkx_edges(G, pos,
                           edgelist=width_large.keys(),
                        #    width=list(width_large.values()),
                           alpha=0.5,
                           edge_color="blue")
    # Node labels bigger font size
    nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
    # Remove edge weight labels (commented out)
    # d = nx.get_edge_attributes(G, "weight")
    # edge_labels = {k: d[k] for k in elarge}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=15)
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.savefig(f"GRN_score_{program_name}.pdf", dpi=300)
    plt.close()
    # Step 4: Reactome pathway analysis
    df_database = pd.DataFrame(
        data=[['GO_Biological_Process_2021', 6036],
              ['GO_Molecular_Function_2021', 1274],
              ['Reactome_2022', 1818]],
        columns=['dataset', 'term']
    )
    databases = ['Reactome_2022']
    m = df_database[df_database['dataset'].isin(databases)]['term'].sum()
    p_thresh = 0.05 / m
    enr_Reactome = gp.enrichr(gene_list=CD_genes,
                              gene_sets=databases,
                              organism='Human',
                              outdir=f'test/enr_Reactome_{program_name}',
                              cutoff=0.5)
    out = enr_Reactome.results
    out = out[out['P-value'] < p_thresh]
    print(f"Reactome enrichment done for {program_name}, significant pathways found: {out.shape[0]}")
    return df_CD, out

# Suppose your gene programs are in a dict `mgs`
selected_programs = ['2','15']  # put your desired keys here

for prog_name in selected_programs:
    try:
        print(f"Running program: {prog_name}")
        gene_list = mgs[prog_name]
        df_CD, enr_results = analyze_gene_program(gene_list, prog_name)
    except Exception as e:
        print(f"❌ Error with program {prog_name}: {e}")

#endregion


#region fine tuning
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
from gears import PertData, GEARS

from scipy.sparse import issparse
import scipy as sp
from einops import rearrange
from torch.nn.functional import softmax
from tqdm import tqdm
import pandas as pd

from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")

import scgpt as scg
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.utils import set_seed
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.preprocess import Preprocessor

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

## initialization
set_seed(42)
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
n_hvg = 1200
n_bins = 51
mask_value = -1
pad_value = -2
n_input_bins = n_bins


# Step 1: Load fine-tuned model and dataset
# 1.1 Load fine-tuned model
# We are going to load a fine-tuned model for the gene interaction analysis on Adamson dataset. The fine-tuned model can be downloaded via this link. The dataset will be loaded in the next step 1.2.
# To reproduce the provided fine-tuned model. Please followw the integration fin-tuning pipeline to fine-tune the pre-trained blood model on the Adamson perturbation dataset.
# Note that in the fine-tuning stage, we did not perform highly vairable gene selection but trained on the 5000+ genes present in the Adamson dataset. This is to provide flexbility in the
# inference stage to investigate changes in attention maps across different perturbation conditions.

# Specify model path; here we load the scGPT blood model fine-tuned on adamson
model_dir = Path("/diazlab/data3/.abhinav/projects/Brain_metastasis/scgpt/resources/finetuned_scGPT_adamson/")
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

ntokens = len(vocab)
model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    vocab=vocab,
    pad_value=pad_value,
    n_input_bins=n_input_bins,
    use_fast_transformer=True,
)

# TransformerModel(
#   (encoder): GeneEncoder(
#     (embedding): Embedding(36574, 512, padding_idx=36571)
#     (enc_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#   )
#   (value_encoder): ContinuousValueEncoder(
#     (dropout): Dropout(p=0.5, inplace=False)
#     (linear1): Linear(in_features=1, out_features=512, bias=True)
#     (activation): ReLU()
#     (linear2): Linear(in_features=512, out_features=512, bias=True)
#     (layers): ModuleList(
#       (0-11): 12 x TransformerEncoderLayer(
#         (self_attn): MultiheadAttention(
#           (out_proj): NonDynamicallyQuantizableLinear(in_features=512, out_features=512, bias=True)
#         )
#         (linear1): Linear(in_features=512, out_features=512, bias=True)
#         (dropout): Dropout(p=0.5, inplace=False)
#         (linear2): Linear(in_features=512, out_features=512, bias=True)
#         (norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         (norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#         (dropout1): Dropout(p=0.5, inplace=False)
#         (dropout2): Dropout(p=0.5, inplace=False)
#       )
#     )
#   )
#   (decoder): ExprDecoder(
#     (fc): Sequential(
#       (0): Linear(in_features=512, out_features=512, bias=True)
#       (1): LeakyReLU(negative_slope=0.01)
#       (2): Linear(in_features=512, out_features=512, bias=True)
#       (3): LeakyReLU(negative_slope=0.01)
#       (4): Linear(in_features=512, out_features=1, bias=True)
#     )
#   )
#   (cls_decoder): ClsDecoder(
#     (_decoder): ModuleList(
#       (0): Linear(in_features=512, out_features=512, bias=True)
#       (1): ReLU()
#       (2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#       (3): Linear(in_features=512, out_features=512, bias=True)
#       (4): ReLU()
#       (5): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#     )
#     (out_layer): Linear(in_features=512, out_features=1, bias=True)
#   )
#   (sim): Similarity(
#     (cos): CosineSimilarity()
#   )
#   (creterion_cce): CrossEntropyLoss()
# )

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

# 1.2 Load dataset of interest
data_dir = Path("../data")
pert_data = PertData(data_dir)
pert_data.load(data_name="adamson")
adata = sc.read(data_dir / "adamson/perturb_processed.h5ad")
ori_batch_col = "control"
adata.obs["celltype"] = adata.obs["condition"].astype("category")
adata.obs["str_batch"] = adata.obs["control"].astype(str)
data_is_raw = False
