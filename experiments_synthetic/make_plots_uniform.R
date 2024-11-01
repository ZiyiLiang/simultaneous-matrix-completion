options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)


setwd("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/experiments_synthetic/results_hpc")
idir <- "results/exp_uniform/"
ifile.list <- list.files(idir)

# Output directory
fig.dir <- "C:/Users/liang/Documents/GitHub/conformal-matrix-completion/results/figures/exp_uniform"
dir.create(fig.dir, showWarnings = FALSE)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))


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
}else{
  key.values <- c("Query_coverage","Size")
  key.labels <- c("Group cov.","Avg. width")
  height <- 2.5
}

if (plot_full){
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Coverage", "Size", "Inf_prop"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, gamma_n, gamma_m, mu,k, Key,r) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, gamma_n, gamma_m, mu,k, Key,r) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}

## Make nice plots for paper
make_plot <- function(results, exp, val, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  df.placeholder <- tibble(Key=c("Query_coverage"), Value=c(1, 0.7)) %>%
    mutate(Key = factor(Key, key.values, key.labels))
#   
  if (exp=="vary_k"){
    pp <- results %>%
      filter(mu %in% val)%>%
      mutate(mu = paste0("\U03BC: ", mu)) %>%
      mutate(mu = factor(mu, levels = paste0("\U03BC: ", val))) %>%
      ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.9) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.3) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      geom_hline(data=df.placeholder, aes(yintercept=Value), alpha=0) +
      ggh4x::facet_grid2(Key~mu, scales="free_y", independent = "y") +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      scale_alpha_manual(values=alpha.scale) +
      xlab("Group size K") +
      ylab("") +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_uniform_%s.pdf", fig.dir, exp), pp, device=NULL, width=6.5, height=height)}
    else{print(pp)}
  }
  else{
    pp <- results %>%
      filter(k %in% val)%>%
      filter(mu %in% seq(6,30,3))%>%
      mutate(k = paste0("K: ", k))%>%
      ggplot(aes(x=mu, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.9) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.3) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      geom_hline(data=df.placeholder, aes(yintercept=Value), alpha=0) +
      ggh4x::facet_grid2(Key~k, scales="free_y", independent = "y") +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      scale_alpha_manual(values=alpha.scale) +
      xlab("Column-wise dependency") +
      ylab("") +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_uniform_%s.pdf", fig.dir, exp), pp, device=NULL, width=6.5, height=height)}
    else{print(pp)}
  }
}

exp_list <- c("vary_k", "vary_mu")
k_list <- c(2,5,8)
mu_list <-  c(9,15, 30)

for (exp in exp_list) {
  if (exp == "vary_k"){
    val <- mu_list
  }else{
    val <- k_list
  }
  make_plot(results, exp, val)
}
