options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)

exp = "wsc"
setwd("~/GitHub/conformal-matrix-completion/experiments_synthetic/results_hpc/")
idir <- sprintf("results/exp_conditional_%s/", exp)
ifile.list <- list.files(idir)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim = ",", col_types = cols())
}))

Method.values <- c("conditional", "unconditional")
Method.labels <- c("Conditional", "Marginal")

#color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
color.scale <- c( "blue", "#56b5e9", "#CC66CC" )
shape.scale <- c(15, 4, 8, 1)
alpha.scale <- c(1, 0.5, 0.8)

plot_full = FALSE

if (plot_full){
  key.values <- c("Query_coverage", "Coverage", "Size", "Inf_prop")
  key.labels <- c("Query cov.", "Coverage", "Size", "Inf_prop")
  height <- 3.5
  fig.dir <- "~/GitHub/conformal-matrix-completion/results/figures/exp_conditional_full/"
}else{
  key.values <- c("Query_coverage","Size")
  key.labels <- c("Query cov.","Size")
  height <- 2.5
  fig.dir <-"~/GitHub/conformal-matrix-completion/results/figures/exp_conditional/"
}
dir.create(fig.dir, showWarnings = FALSE)

if (plot_full){
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Coverage", "Size", "Inf_prop"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, k, delta, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, k, delta, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}


## Make nice plots for paper
make_plot <- function(results, exp, val, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))
  
  
  if (exp=="vary_delta"){
    pp <- results %>%
      filter(k %in% val)%>%
      filter(delta != 0.10)%>%
      mutate(k = paste0("k: ", k))%>%
      ggplot(aes(x=delta, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.006) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      ggh4x::facet_grid2(Key~k, scales="free") +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      scale_alpha_manual(values=alpha.scale) +
      xlab("Delta") +
      ylab("") +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_conditional_%s.pdf", fig.dir,exp), pp, device=NULL, width=6, height=height)}
    else{
      print(pp)
    }
  }
  if (exp=="vary_k"){
    pp <- results %>%
      filter(delta %in% val)%>%
      mutate(delta = paste0("proportion: ", delta))%>%
      ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.3) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      ggh4x::facet_grid2(Key~delta, scales="free") +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      scale_alpha_manual(values=alpha.scale) +
      xlab("Query size K") +
      ylab("") +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_conditional_%s.pdf", fig.dir,exp), pp, device=NULL, width=6, height=height)}
    else{
      print(pp)
    }
  }
}

exp_list <- c("vary_k", "vary_delta")
k_list <- seq(2, 4, by = 1)
delta_list <-  c(0.12, 0.16, 0.20)
for (exp in exp_list) {
  if (exp == "vary_k"){
    val = delta_list
  }else{
    val = k_list
  }
  make_plot(results, exp, val)
}
