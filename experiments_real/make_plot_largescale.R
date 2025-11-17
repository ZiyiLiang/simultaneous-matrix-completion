options(width=160)

library(dplyr)
library(tidyr)
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

# Compute inference time for 100 test pairs
n_test <- 2000
results.raw$t_test <- (results.raw$t_inference-results.raw$t_universal)/n_test * 100

# Compute total inference time and model fitting time 
results.raw$t_fitting <- results.raw$t_train + results.raw$t_missing
results.raw$t_inference <- results.raw$t_sample_calib + results.raw$t_universal + results.raw$t_test

key.values <- c("t_train", "t_missing", "t_sample_calib", "t_universal", "t_test", "t_fitting", "t_inference")
key.labels <- c("Point pred.", "Missingness est.", "Calib. sampling", "One-time comp.", "Test inference", "Total fitting", "Total inference")

results <- results.raw %>%
  pivot_longer(cols=c("t_train", "t_missing", "t_sample_calib", "t_universal",  "t_test","t_fitting", "t_inference"), names_to='Key', values_to='Value') %>%
  mutate(Key = factor(Key, key.values, key.labels)) %>%
  filter(k==6) %>%
  group_by(Calib_queries, Key) %>%
  summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
  

# Reshape and organize the data
latex_table <- results %>%
  # Filter relevant rows
  filter(Key %in% c("Point pred.", "Missingness est.", "Total fitting",
                    "Calib. sampling", "One-time comp.", "Test inference", "Total inference")) %>%
  # Select relevant columns
  select(Calib_queries, Key, Value) %>%
  # Pivot wider to get keys as columns
  pivot_wider(names_from = Key, values_from = Value) %>%
  # Rename columns for clarity
  rename(
    `Point Pred.` = `Point pred.`,
    `Missingness Est.` = `Missingness est.`,
    `Total Fitting` = `Total fitting`,
    `Calib. Sampling` = `Calib. sampling`,
    `One-time Comp.` = `One-time comp.`,
    `Test Inference` = `Test inference`,
    `Total Inference` = `Total inference`
  ) %>%
  # Reorder columns
  select(Calib_queries, 
         `Point Pred.`, `Missingness Est.`, `Total Fitting`,
         `Calib. Sampling`, `One-time Comp.`, `Test Inference`, `Total Inference`)

# Create LaTeX table with kableExtra
latex_output <- latex_table %>%
  kbl(format = "latex", 
      booktabs = TRUE,
      digits = 2,
      col.names = c("Calib Queries", 
                    "Point Pred.", "Missingness Est.", "Total",
                    "Calib. Sampling", "One-time Comp.", "Test Inference", "Total"),
      align = c("l", rep("c", 7))) %>%
  add_header_above(c(" " = 1, 
                     "Model Fitting Time" = 3, 
                     "Inference Time" = 4)) %>%
  kable_styling(latex_options = c("striped", "hold_position"))

# Print the LaTeX code
latex_output