# Load necessary libraries
library(DESeq2)
library(tidyverse)
library(ggrepel) # For non-overlapping labels

# --- Load and Prepare Data ---
metadata <- read.csv('../data/MAGE_metadata.txt')
count_data <- read.csv('../data/gene_count_matrix.csv', row.names = 1)

# Clean column names
original_colnames <- colnames(count_data)
cleaned_colnames <- gsub("_stringtieRef", "", original_colnames)
colnames(count_data) <- cleaned_colnames

# Filter for DCM and Non-Failing Donor groups
filtered_metadata <- metadata %>%
  filter(etiology %in% c("Dilated cardiomyopathy (DCM)", "Non-Failing Donor"))

# Find common samples
common_samples <- intersect(colnames(count_data), filtered_metadata$Run)

# Filter data to common samples
count_data_filtered <- count_data[, common_samples]
metadata_filtered <- filtered_metadata %>%
  filter(Run %in% common_samples)

# Order metadata
metadata_ordered <- metadata_filtered[match(colnames(count_data_filtered), metadata_filtered$Run), ]

# Create colData
coldata <- data.frame(
  condition = factor(metadata_ordered$etiology)
)
rownames(coldata) <- metadata_ordered$Run

# Convert to integer matrix
count_matrix <- as.matrix(count_data_filtered)
storage.mode(count_matrix) <- "integer"

# --- Differential Expression Analysis ---
# Create a full DESeqDataSet object
dds <- DESeqDataSetFromMatrix(countData = count_matrix,
                              colData = coldata,
                              design = ~ condition)

# Estimate size factors
dds <- estimateSizeFactors(dds)

# Run the DESeq analysis
dds <- DESeq(dds)

# Get the results
res <- results(dds, contrast=c("condition", "Dilated cardiomyopathy (DCM)", "Non-Failing Donor"))

# Order results by adjusted p-value
res_ordered <- res[order(res$padj), ]

# --- Filter for Significant Genes and Annotate Them ---
res_df <- as.data.frame(res_ordered)
significant_res_df <- res_df %>%
  filter(padj < 0.05 & abs(log2FoldChange) > 1)

# Annotate only the significant genes
library(biomaRt)
en_mart <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")
gene_ids <- rownames(significant_res_df)

gene_symbols <- getBM(attributes = c('ensembl_gene_id', 'hgnc_symbol'),
                      filters = 'ensembl_gene_id',
                      values = gene_ids,
                      mart = en_mart)

# Merge annotations with significant results
significant_res_df$ensembl_gene_id <- rownames(significant_res_df)
annotated_significant_df <- merge(significant_res_df, gene_symbols, by = "ensembl_gene_id", all.x = TRUE)

# Reorder and save the annotated significant results
annotated_significant_df <- annotated_significant_df %>%
  dplyr::select(hgnc_symbol, ensembl_gene_id, everything())

write.csv(annotated_significant_df, file = "../data/DEG_results.csv", row.names = FALSE)

print("Differential Expression Analysis complete. Annotated SIGNIFICANT results saved to ../data/DEG_results.csv")

# --- Visualization ---
# For context, the plot will still show all genes, but we will label the top significant ones.
res_df_plot <- as.data.frame(res_ordered)

# Get the top 10 genes from our significant, annotated list to label on the plot
top_10_genes <- annotated_significant_df %>%
  filter(!is.na(hgnc_symbol) & hgnc_symbol != "") %>%
  slice_min(order_by = padj, n = 10)

# Add a label column to the full dataset for plotting
res_df_plot$label <- ""
res_df_plot$label[match(top_10_genes$ensembl_gene_id, rownames(res_df_plot))] <- top_10_genes$hgnc_symbol

volcano_plot <- ggplot(res_df_plot, aes(x = log2FoldChange, y = -log10(padj))) +
  geom_point(aes(color = ifelse(padj < 0.05 & abs(log2FoldChange) > 1, "Significant", "Not Significant")), alpha = 0.4) +
  scale_color_manual(values = c("Significant" = "red", "Not Significant" = "grey")) +
  geom_text_repel(aes(label = label), size = 3.5, box.padding = 0.5, max.overlaps = Inf) +
  theme_minimal() +
  labs(title = "Volcano Plot of DCM vs. Non-Failing Donor",
       x = "Log2 Fold Change",
       y = "-Log10 Adjusted P-value",
       color = "Significance") +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed") +
  geom_vline(xintercept = c(-1, 1), linetype = "dashed")
# Save the plot
ggsave("../data/volcano_plot.png", plot = volcano_plot, width = 10, height = 8)

print("Volcano plot saved to ../data/volcano_plot.png")