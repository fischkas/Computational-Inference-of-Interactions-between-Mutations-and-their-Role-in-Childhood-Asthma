library(tidyverse)
library(data.table)
library(png)
library(plyr)
library(ggtext)
library(normentR)
library(formattable)
library(numbers)

gene_annotation <- fread("rs_hugo_conversion_nearest.txt")
severe_bim <- fread("bmfile.bim") %>%
  dplyr::rename(CHR=V1, SNP=V2, POS=V3, BP=V4, A1=V5, A2=V6)

snp2gene <- left_join(severe_bim, gene_annotation, by = "SNP" ) %>%
  dplyr::select(CHR, SNP, BP, A1, A2, startGene, endGene, Gene, dist)



snp2gene %>%
  distinct(SNP) %>%
  rownames_to_column() %>%
  mutate(snp_row = as.numeric(rowname)-1) -> snp_rows

snp2gene %>%
  distinct(Gene) %>%
  rownames_to_column() %>%
  mutate(gene_col = as.numeric(rowname)-1) -> gene_cols


write.table(gene_cols, "gene_cols.tsv",
            row.names = F, col.names = T, quote = F, sep = "\t")



left_join(snp2gene, snp_rows, by = "SNP") %>%
  dplyr::select(SNP, Gene, snp_row) -> data_snp_row


left_join(data_snp_row, gene_cols, by = "Gene") %>%
  dplyr::select(snp_row, gene_col) %>%
  mutate(snp = snp_row,
         gene = gene_col) %>%
  dplyr::select(snp, gene) -> connectivity


write.table(connectivity, "connectivty_pruned.tsv",
            row.names = F, col.names = T, quote = F, sep = "\t")
