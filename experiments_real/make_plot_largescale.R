options(width=160)

library(dplyr)
library(tidyr)
library(tidyverse)
library(kableExtra)
library(lubridate) 

# --- Load Data ---
exp="largescale"

idir <- sprintf("results/%s/", exp)

setwd("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/experiments_real/results_hpc/")
ifile.list <- list.files(idir)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim = ",", col_types = cols())
}))

# --- Define Helper for Time Formatting ---
format_hms <- function(seconds) {
  if(any(is.na(seconds))) return("NA")

  total_seconds <- round(as.numeric(seconds))
  
  h <- total_seconds %/% 3600
  m <- (total_seconds %% 3600) %/% 60
  s <- total_seconds %% 60
  
  # Format: H:MM:SS (e.g., 0:01:39)
  sprintf("%d:%02d:%02d", h, m, s)
}

# --- Compute Metrics ---
n_test <- 2000

# Fitting totals
results.raw$t_fit_svd   <- results.raw$t_svd + results.raw$t_missing
results.raw$t_fit_svdpp <- results.raw$t_svdpp + results.raw$t_missing

# Inference totals
results.raw$t_test_marginal <- (results.raw$t_inference - results.raw$t_universal)
results.raw$t_test_100 <- (results.raw$t_test_marginal / n_test) * 100
results.raw$t_inf_total <- results.raw$t_sample_calib + results.raw$t_universal + results.raw$t_test_100

# --- Aggregate Data ---
results_agg <- results.raw %>%
  filter(k == 6) %>%
  group_by(Calib_queries) %>%
  summarise(
    # Aggregating raw seconds
    val_svd_pt   = mean(t_svd, na.rm=TRUE),
    val_svdpp_pt = mean(t_svdpp, na.rm=TRUE),
    val_missing  = mean(t_missing, na.rm=TRUE),
    val_fit_svd  = mean(t_fit_svd, na.rm=TRUE),
    val_fit_svdpp= mean(t_fit_svdpp, na.rm=TRUE),
    
    val_calib    = mean(t_sample_calib, na.rm=TRUE),
    val_univ     = mean(t_universal, na.rm=TRUE),
    val_test100  = mean(t_test_100, na.rm=TRUE),
    val_inf_tot  = mean(t_inf_total, na.rm=TRUE)
  ) %>%
  ungroup() %>%
  mutate(across(starts_with("val_"), Vectorize(format_hms)))

# --- Generate Table (a): Fitting ---
latex_a <- results_agg %>%
  select(Calib_queries, 
         val_missing, 
         val_svd_pt, val_svdpp_pt, 
         val_fit_svd, val_fit_svdpp) %>%
  kbl(format = "latex", booktabs = TRUE, 
      align = c("l", "c", "c", "c", "c", "c"),
      col.names = c("$n$",  
                    "Weight est.",   
                    "SVD", "SVD++", 
                    "SVD", "SVD++"),
      escape = FALSE) %>%
  add_header_above(c(" " = 2, 
                     "Point Prediction" = 2, 
                     "Total Fitting" = 2)) %>%
  kable_styling(latex_options = c("hold_position"))

# --- Generate Table (b): Inference ---
latex_b <- results_agg %>%
  select(Calib_queries, val_calib, val_univ, val_test100, val_inf_tot) %>%
  kbl(format = "latex", booktabs = TRUE, 
      align = c("l", "c", "c", "c", "c"),
      col.names = c("$n$", 
                    "Sampling", 
                    "Pre-comp.", 
                    "Test (100)", 
                    "Total"),
      escape = FALSE) %>%
  kable_styling(latex_options = c("hold_position"))

# Print
print(latex_a)
print(latex_b)