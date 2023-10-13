options(width=160)

library(tidyverse)
library(kableExtra)

setwd("~/GitHub/conformal-matrix-completion")
idir <- "results/exp_hpm"
ifile.list <- list.files(idir)

# Output directory
fig.dir <- "results/figures"

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

key.values <- c("Pair_coverage", "Coverage", "Size")
key.labels <- c("Paired coverage", "Coverage", "Size")

Method.values <- c("conformal", "naive", "bonferroni", "uncorrected")
Method.labels <- c("Conformal", "Naive", "Bonferroni", "Uncorrected")

color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
shape.scale <- c(15, 4, 8, 1)


results_hpm <- results.raw %>%
  mutate(Method = factor(Method, Method.values, Method.labels)) %>%
  pivot_longer(cols=c(`Pair_coverage`, `Coverage`,`Size`), names_to='Key', values_to='Value') %>%
  mutate(Key = factor(Key, key.values, key.labels)) %>%
  group_by(Method, n1, n2, r_true, r_guess, prob_obs, Key) %>%
  summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))%>%
#  pivot_wider(names_from = 'Key', values_from = 'Value') %>%
  filter(n1 == 500, n2 == 500, r_true == 20, r_guess==20, Key == 'Size')

