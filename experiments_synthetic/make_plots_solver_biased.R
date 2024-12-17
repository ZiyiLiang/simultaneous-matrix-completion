options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)
library(dplyr)


setwd("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/experiments_synthetic/results_hpc")
idir <- "results/exp_solver_biased/"
ifile.list <- list.files(idir)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

idir_als <- "results/exp_biased_obs/"
ifile_als.list <- list.files(idir_als)

results.als <- do.call("rbind", lapply(ifile_als.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir_als, ifile), delim=",", col_types=cols())
}))

results.als$Solver <- 'als'
results.als <- filter(results.als, scale==0.14)

combined_results <- rbind(
  results.als %>% select(intersect(names(.), names(results.raw))),
  results.raw %>% select(intersect(names(.), names(results.als)))
)


Method.values <- c("conformal", "Bonferroni", "Uncorrected")
Method.labels <- c("Simultaneous", "Bonferroni", "Unadjusted")

#color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
color.scale <- c( "blue", "#56b5e9", "#CC66CC" )
shape.scale <- c(15, 4, 8, 1)
alpha.scale <- c(1, 0.5, 0.8)

plot_full = FALSE

if (plot_full){
  key.values <- c("Query_coverage", "Coverage", "Size", "Inf_prop")
  key.labels <- c("Group cov.", "Coverage", "Avg. width", "Inf_prop")
  height <- 3.5
  fig.dir <- "C:/Users/liang/Documents/GitHub/conformal-matrix-completion/results/figures/exp_solver_biased_full/"
}else{
  key.values <- c("Query_coverage","Size")
  key.labels <- c("Group cov.","Avg. width")
  height <- 2.5
  fig.dir <- "C:/Users/liang/Documents/GitHub/conformal-matrix-completion/results/figures/exp_solver_biased/"
}
dir.create(fig.dir, showWarnings = FALSE)


results_filtered <- combined_results %>% filter(scale==0.14, sd==0.1)

if (plot_full){
  results <- results_filtered %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Coverage", "Size", "Inf_prop"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, Solver, scale,k, Key,sd) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  results <- results_filtered %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, Solver, scale,k, Key,sd) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}


## Make nice plots for paper
make_plot <- function(results, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))
  df.placeholder <- tibble(Key=c("Query_coverage"), Value=c(1, 0.7)) %>%
    mutate(Key = factor(Key, key.values, key.labels))

  pp <- results %>%
    # mutate(k = paste0("K: ", k))%>%
    mutate(Solver = toupper(Solver)) %>%
    ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
    geom_point(alpha=0.75) +
    geom_line() +
    geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.006) +
    geom_hline(data=df.nominal, aes(yintercept=Value)) +
    geom_hline(data=df.placeholder, aes(yintercept=Value), alpha=0) +
    ggh4x::facet_grid2(Key~Solver, scales="free_y", independent = "y") +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    scale_alpha_manual(values=alpha.scale) +
    xlab("Group size K") +
    ylab("") +
    theme_bw()
  if (sv == TRUE){
    ggsave(sprintf("%s/exp_solver_biased.pdf", fig.dir), pp, device=NULL, width=6.5, height=height)}
  else{
    print(pp)
  }
}

make_plot(results, sv=TRUE)

