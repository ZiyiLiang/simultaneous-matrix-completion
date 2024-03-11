options(width=160)

library(tidyverse)
library(kableExtra)

setwd("~/GitHub/conformal-matrix-completion/experiments_synthetic/results_hpc")
idir <- "results/exp_uniform/"
ifile.list <- list.files(idir)

# Output directory
fig.dir <- "~/GitHub/conformal-matrix-completion/results/figures/exp_uniform"
dir.create(fig.dir, showWarnings = FALSE)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))


Method.values <- c("conformal", "Bonferroni", "Uncorrected")
Method.labels <- c("Simultaneous", "Bonferroni", "Individual")

color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
shape.scale <- c(15, 4, 8, 1)

plot_full = FALSE

if (plot_full){
  key.values <- c("Query_coverage", "Coverage", "Size", "Inf_prop")
  key.labels <- c("Query coverage", "Coverage", "Size", "Inf_prop")
  height <- 5
}else{
  key.values <- c("Query_coverage","Size")
  key.labels <- c("Query coverage","Size")
  height <- 3
}

if (plot_full){
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Coverage", "Size", "Inf_prop"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, gamma_n, gamma_m, mu,k, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, gamma_n, gamma_m, mu,k, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
  
}

## Make nice plots for paper
make_plot <- function(results, exp, val, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
#   df.ghost <- tibble(Key=c("Query_coverage", "Size"), Value=c(0.9,10), Method="conformal") %>%
# #  df.ghost <- tibble(Key=c("Query_coverage", "Coverage", "Size","Inf_prop"), Value=c(0.9,0.9,10,0), Method="conformal") %>%
#     mutate(Method = factor(Method, Method.values, Method.labels)) %>%
#     mutate(Key = factor(Key, key.values, key.labels)) 
#   
  if (exp=="vary_k"){
    pp <- results %>%
      filter(mu %in% val)%>%
      mutate(mu = paste0("\U03BC: ", mu))%>%
      ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.5) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      facet_grid(Key~mu, scales="free") +
      xlab("Query size K") +
      ylab("") +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_uniform_%s.pdf", fig.dir, exp), pp, device=NULL, width=5.5, height=height)}
    else{pp}
  }
  else{
    pp <- result %>%
      filter(k %in% val)%>%
      ggplot(aes(x=mu, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.5) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      facet_grid(Key~k, scales="free") +
      xlab("Column-wise magnitude") +
      ylab("") +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_uniform_%s.pdf", fig.dir, exp), pp, device=NULL, width=5.5, height=height)}
    else{pp}
  }
}

exp_list <- c("vary_k", "vary_mu")
k_list <- 2:8
mu_list <-  seq(0, 30, by = 3)

results <- generate_results(plot_full)
for (exp in exp_list) {
  if (exp == "vary_k") {
    l <- mu_list
  }
  else {
    l <- k_list
  }
  for (val in l){
    make_plot(exp, val)
  }
}

