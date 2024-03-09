options(width=160)

library(tidyverse)
library(kableExtra)


key.values <- c("Query_coverage", "Coverage", "Size", "Inf_prop")
key.labels <- c("Query coverage", "Coverage", "Size", "Inf_prop")

Method.values <- c("conformal", "Bonferroni", "Uncorrected")
Method.labels <- c("Simultaneous", "Bonferroni", "Individual")

color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
shape.scale <- c(15, 4, 8, 1)


generate_results <- function(exp, full_graph) {
  setwd("~/GitHub/conformal-matrix-completion/experiments_real/results_hpc/")
  idir <- sprintf("results/exp_uniform_%s/", exp)
  
  ifile.list <- list.files(idir)
  
  # Output directory
  if (full_graph == FALSE) {
    fig.dir <- sprintf("~/GitHub/conformal-matrix-completion/results/figures/exp_uniform_%s/", exp)
    img_height <- 2
  } else {
    fig.dir <- sprintf("~/GitHub/conformal-matrix-completion/results/figures/exp_uniform_%s_full/", exp)
    img_height <- 3
  }
  
  dir.create(fig.dir, showWarnings = FALSE)
  
  results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
    df <- read_delim(sprintf("%s/%s", idir, ifile), delim = ",", col_types = cols())
  }))
  
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols = c(`Query_coverage`, `Coverage`, `Size`, `Inf_prop`), names_to = 'Key', values_to = 'Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, k, r, Key) %>%
    summarise(num = n(), Value.se = sd(Value, na.rm = TRUE) / sqrt(n()), Value = mean(Value, na.rm = TRUE))
  
  if (full_graph == FALSE){
    results <- results %>%
      filter(Key != "Coverage", Key != "Inf_prop")
  }
  
  return(results)
}


## Make nice plots for paper
make_plot <- function(results, exp, full_graph=FALSE, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  df.ghost <- tibble(Key=c("Query_coverage", "Size"), Value=c(0.9,10), Method="conformal") %>%
    #  df.ghost <- tibble(Key=c("Query_coverage", "Coverage", "Size","Inf_prop"), Value=c(0.9,0.9,10,0), Method="conformal") %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    mutate(Key = factor(Key, key.values, key.labels)) 
  pp <- results %>%
    mutate(r = sprintf("Guessed rank: %i", r))%>%
    ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
    geom_point(alpha=0.75) +
    geom_line() +
    geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.5) +
    geom_hline(data=df.nominal, aes(yintercept=Value)) +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    facet_grid(Key~r, scales="free") +
    xlab("Query size K") +
    ylab("") +
    theme_bw()
  print(pp)
}

full_graph=TRUE
results <- generate_results("movielens", full_graph)
make_plot(results, "movielens", full_graph, sv=FALSE)