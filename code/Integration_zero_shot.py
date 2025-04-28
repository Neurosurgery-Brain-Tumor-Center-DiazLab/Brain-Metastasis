# Testing if CUDA is installed
import torch
print(torch.__version__)  # Should match the installed version
print(torch.cuda.is_available())  # Should return True if CUDA is set up correctly
print("CUDA available:", torch.cuda.is_available())

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

