# Load necessary libraries
library(DESeq2)
library(tidyverse)

# Load metadata
metadata <- read.csv('../data/MAGE_metadata.txt')

# Load the gene count matrix
count_data <- read.csv('../data/gene_count_matrix.csv', row.names = 1)

# --- Correct the column names ---
# Remove the '_stringtieRef' suffix from the count data column names to match the metadata 'Run' column
original_colnames <- colnames(count_data)
cleaned_colnames <- gsub("_stringtieRef", "", original_colnames)
colnames(count_data) <- cleaned_colnames

# --- Filter and Match Data ---
# Filter metadata for the groups of interest: DCM and Non-Failing Donor
# Hypertrophic cardiomyopathy (HCM)' and 'Peripartum cardiomyopathy (PPCM)', have been excluded from the current analysis.
filtered_metadata <- metadata %>%
  filter(etiology %in% c("Dilated cardiomyopathy (DCM)", "Hypertrophic cardiomyopathy (HCM)", "Non-Failing Donor"))

# Find the common samples between the cleaned count matrix column names and the metadata 'Run' column
common_samples <- intersect(colnames(count_data), filtered_metadata$Run)

# Filter the count data and metadata to keep only the common samples
count_data_filtered <- count_data[, common_samples]
metadata_filtered <- filtered_metadata %>%
  filter(Run %in% common_samples)

# --- Ensure Correct Order and Format ---
# Order the metadata to match the column order of the count data
metadata_ordered <- metadata_filtered[match(colnames(count_data_filtered), metadata_filtered$Run), ]

# Create the colData dataframe for DESeq2
coldata <- data.frame(
  condition = factor(metadata_ordered$etiology)
)
rownames(coldata) <- metadata_ordered$Run

# Convert the filtered count data to a matrix of integers for DESeq2
count_matrix <- as.matrix(count_data_filtered)
storage.mode(count_matrix) <- "integer"

# --- Run Normalization ---
# Create a DESeqDataSet object
dds <- DESeqDataSetFromMatrix(countData = count_matrix,
                              colData = coldata,
                              design = ~ condition)

# Estimate size factors before VST, as recommended
dds <- estimateSizeFactors(dds)

# Apply the Variance Stabilizing Transformation
vsd <- vst(dds, blind = FALSE)

# Extract and save the normalized counts
write.csv(assay(vsd), file = '../data/normalized_gene_counts.csv')

print("Normalization complete with correct metadata and column name matching. Normalized counts saved to ../data/normalized_gene_counts.csv")