options(width=160)

library(tidyverse)
library(kableExtra)


key.values <- c("Query_coverage", "Coverage", "Size", "Inf_prop")
key.labels <- c("Query coverage", "Coverage", "Size", "Inf_prop")

Method.values <- c("conditional", "unconditional")
Method.labels <- c("Conditional", "Unconditional")

color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
shape.scale <- c(15, 4, 8, 1)


generate_results <- function(exp, full_graph) {
    setwd("~/GitHub/conformal-matrix-completion/experiments_synthetic/results_hpc/")
    idir <- sprintf("results/exp_conditional_%s/", exp)
    
    ifile.list <- list.files(idir)
    
    # Output directory
    if (full_graph == FALSE) {
      fig.dir <- sprintf("~/GitHub/conformal-matrix-completion/results/figures/exp_conditional_%s/", exp)
      img_height <- 2
    } else {
      fig.dir <- sprintf("~/GitHub/conformal-matrix-completion/results/figures/exp_conditional_%s_full/", exp)
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
      group_by(Method, k, delta, Key) %>%
      summarise(num = n(), Value.se = sd(Value, na.rm = TRUE) / sqrt(n()), Value = mean(Value, na.rm = TRUE))
    
    if (full_graph == FALSE){
      results <- results %>%
        filter(Key != "Coverage", Key != "Inf_prop")
    }
    
    return(results)
}




## Make nice plots for paper
make_plot <- function(results, exp, val, full_graph=FALSE, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))
  if (full_graph){
    df.ghost <- tibble(Key=c("Query_coverage", "Coverage", "Size","Inf_prop"), Value=c(0.9,0.9,10,0), Method="conditional") %>%
      mutate(Method = factor(Method, Method.values, Method.labels)) %>%
      mutate(Key = factor(Key, key.values, key.labels)) 
  }
  else{
    df.ghost <- tibble(Key=c("Query_coverage", "Size"), Value=c(0.9,10), Method="conditional") %>%
      mutate(Method = factor(Method, Method.values, Method.labels)) %>%
      mutate(Key = factor(Key, key.values, key.labels))
  }
  
  
  if (exp=="vary_delta"){
    pp <- results %>%
      filter(k==val)%>%
      ggplot(aes(x=delta, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.006) +
      # geom_errorbar(aes(ymin=Value.lq, ymax=Value.uq), width=0.3) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      facet_wrap(.~Key, scales="free") +
      xlab("Delta") +
      ylab(sprintf("k=%i", val)) +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_conditional_%s_k%i.pdf", exp, val, fig.dir), pp, device=NULL, width=5.5, height=img_height)}
    else{
      print(pp)
    }
  }
  if (exp=="vary_k"){
    pp <- results %>%
      filter(delta==val)%>%
      ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.3) +
      # geom_errorbar(aes(ymin=Value.lq, ymax=Value.uq), width=0.3) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      facet_wrap(.~Key, scales="free") +
      xlab("Query size K") +
      ylab(sprintf("delta=%.2f", val)) +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_conditional_%s_delta%.2f.pdf", exp, val, fig.dir), pp, device=NULL, width=5.5, height=img_height)}
    else{
      print(pp)
    }
  }
}

results <- generate_results("wsc", full_graph)

# Graphing parameters
exp_list <- c("vary_k", "vary_delta")
full_graph <- TRUE
sv <- FALSE
# for (exp in exp_list) {
#   results <- generate_results(exp, full_graph)
#   make_plot(results, exp, sv=sv)
# }

k_list <- seq(1, 4, by = 1)
delta_list <-  seq(0.1, 0.2, by = 0.02)

for (exp in exp_list) {
  if (exp == "vary_k") {
    l <- delta_list
  }
  else {
    l <- k_list
  }
  for (val in l){
    make_plot(results, exp, val, full_graph = full_graph, sv=sv)
  }
}