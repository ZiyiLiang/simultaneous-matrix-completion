options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)
library(dplyr)


setwd("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/experiments_synthetic/results_hpc")
idir <- "results/exp_est_biased/"
ifile.list <- list.files(idir)

# Output directory
fig.dir <- "C:/Users/liang/Documents/GitHub/conformal-matrix-completion/results/figures/exp_est_biased"
dir.create(fig.dir, showWarnings = FALSE)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

Method.values <- c("conformal", "est")
Method.labels <- c("Oracle", "Estimated")

#color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
color.scale <- c( "blue", "#56b5e9", "#CC66CC" )
shape.scale <- c(15, 4, 8, 1)
alpha.scale <- c(1, 0.5, 0.8)

plot_full = FALSE

if (plot_full){
  key.values <- c("Query_coverage", "Coverage", "Size", "Inf_prop")
  key.labels <- c("Group cov.", "Coverage", "Avg. width", "Inf_prop")
  height <- 3.5
}else{
  key.values <- c("Query_coverage","Size")
  key.labels <- c("Group cov.","Avg. width")
  height <- 2.5
}

results_filtered <- results.raw %>%
                    filter(k==5)

if (plot_full){
  results <- results_filtered %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Coverage", "Size","Inf_prop"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, k, r_est, scale, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  results <- results_filtered%>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    #pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, k, r_est, scale, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}


r_est_labeller <- as_labeller(function(x) paste("guessed rank:", x))

## Make nice plots for paper
make_plot <- function(results, val, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  df.placeholder <- tibble(Key=c("Query_coverage"), Value=c(1, 0.7)) %>%
    mutate(Key = factor(Key, key.values, key.labels))
  
  pp <- results %>%
    filter(r_est %in% val)%>%
    ggplot(aes(x=scale, y=Value, color=Method, shape=Method)) +
    geom_point(alpha=0.9) +
    geom_line() +
    geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.006) +
    geom_hline(data=df.nominal, aes(yintercept=Value)) +
    geom_hline(data=df.placeholder, aes(yintercept=Value), alpha=0) +
    ggh4x::facet_grid2(Key~r_est, scales="free_y", independent = "y", labeller = labeller(r_est = r_est_labeller)) +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    scale_alpha_manual(values=alpha.scale) +
    scale_x_reverse() +
    xlab("Missingness heterogeneity") +
    ylab("") +
    theme_bw()
  if (sv == TRUE){
    ggsave(sprintf("%s/exp_est_biased.pdf", fig.dir), pp, device=NULL, width=6.5, height=height)
  }else{
    print(pp)
  }
}

val <-c(3,5,7)
make_plot(results, val, sv=TRUE)