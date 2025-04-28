library(Seurat)
library(SeuratDisk)
library(zellkonverter)
obj <- readRDS("/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdataBrM_Samples.rds")
obj <- NormalizeData(obj)
SaveH5Seurat(obj, filename = "/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdataBrM_Samples.h5Seurat")
Convert("/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdataBrM_Samples.h5Seurat", dest = "h5ad") #### error due to different number of cellnames

# Are all cell names the same?
length(Cells(obj))
dim(obj@assays$RNA@layers$counts)
length(rownames(obj@meta.data))
length(setdiff(Cells(obj), rownames(obj@meta.data)))

#### Since all metadata and object names are not same so making the object again
ct_matrix <- obj@assays$RNA@layers$counts
rownames(ct_matrix) <- rownames(obj@assays$RNA@features@.Data)
colnames(ct_matrix) <- rownames(obj@meta.data)

obj2 <- CreateSeuratObject(counts = ct_matrix, min.cells = 3, meta.data = obj@meta.data)
obj2 <- NormalizeData(obj2)
obj2@assays$RNA$data <- GetAssayData(obj2, layer = "data")
SaveH5Seurat(obj2, filename = "/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdataBrM_Samples.h5Seurat", overwrite = TRUE)
Convert("/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdataBrM_Samples.h5Seurat", dest = "h5ad", overwrite = TRUE) #### error due to different number of cellnames
### This also failed

### Trying this new one
library(Seurat)
library(Matrix)
counts <- obj2@assays$RNA@counts
Matrix::writeMM(counts, file = "/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdata/brain_met_count.mtx")
write.csv(counts, file = "/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdata/brain_met_count.csv", row.names = TRUE, quote = F)
write.csv(rownames(counts), "/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdata/genes.csv", row.names = FALSE)
write.csv(colnames(counts), "/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdata/barcodes.csv", row.names = FALSE)
write.csv(obj2@meta.data, "/diazlab/data3/.abhinav/projects/Brain_metastasis/metastasis/rawdata/metadata.csv", row.names = TRUE, quote = F)


### Applying the same for the Primary samples
nonbrm <- readRDS("/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/rawdata/NonBrM_Samples.rds")
library(Seurat)
library(Matrix)
ct_matrix <- nonbrm@assays$RNA@layers$counts
rownames(ct_matrix) <- rownames(nonbrm@assays$RNA@features@.Data)
colnames(ct_matrix) <- rownames(nonbrm@meta.data)

nonbrm_obj2 <- CreateSeuratObject(counts = ct_matrix, min.cells = 3, meta.data = nonbrm@meta.data)

nonbrm_obj2@assays$RNA$counts <- GetAssayData(nonbrm_obj2, layer = "counts")
nonbrm_obj2 <- NormalizeData(nonbrm_obj2)
nonbrm_obj2@assays$RNA$data <- GetAssayData(nonbrm_obj2, layer = "data")

counts <- nonbrm_obj2@assays$RNA$counts
Matrix::writeMM(counts, file = "/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/rawdata/brain_met_count.mtx")
# write.csv(counts, file = "/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/rawdata/brain_met_count.csv", row.names = TRUE, quote = F)
write.csv(rownames(counts), "/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/rawdata/genes.csv", row.names = FALSE)
write.csv(colnames(counts), "/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/rawdata/barcodes.csv", row.names = FALSE)
write.csv(nonbrm_obj2@meta.data, "/diazlab/data3/.abhinav/projects/Brain_metastasis/primary/rawdata/metadata.csv", row.names = TRUE, quote = F)
