options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)

exp="largescale"

idir <- sprintf("results/%s/", exp)

setwd("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/experiments_real/results_hpc/")
ifile.list <- list.files(idir)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim = ",", col_types = cols())
}))
results.raw$t_test <- results.raw$t_inference-results.raw$t_universal

key.values <- c("t_train", "t_missing", "t_sample_calib", "t_test", "t_universal")
key.labels <- c("Point pred.", "Missingness est.", "Calib. sampling", "Test inference", "One-time comp.")

results <- results.raw %>%
  pivot_longer(cols=c("t_train", "t_missing", "t_sample_calib", "t_universal",  "t_test"), names_to='Key', values_to='Value') %>%
  mutate(Key = factor(Key, key.values, key.labels)) %>%
  group_by(Calib_queries, k, Key) %>%
  summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))%>%
  filter(k==6)

