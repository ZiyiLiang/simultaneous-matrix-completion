options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)

plot_full = TRUE
est=FALSE
exp="movielens"
#exp="books"

if (est){
  idir <- sprintf("results/exp_uniform_est_%s/", exp)
} else {
  idir <- sprintf("results/exp_uniform_%s/", exp)
}

setwd("~/GitHub/conformal-matrix-completion/experiments_real/results_hpc/")
ifile.list <- list.files(idir)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim = ",", col_types = cols())
}))

Method.values <- c("conformal", "Bonferroni", "Uncorrected")
Method.labels <- c("Simultaneous", "Bonferroni", "Individual")

#color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
color.scale <- c( "blue", "#56b5e9", "#CC66CC" )
shape.scale <- c(15, 4, 8, 1)
alpha.scale <- c(1, 0.5, 0.8)


if (plot_full){
  key.values <- c("Query_coverage", "Coverage", "Size", "Inf_prop")
  key.labels <- c("Query cov.", "Coverage", "Size", "Inf_prop")
  height <- 4.5
  fig.dir <- sprintf("~/GitHub/conformal-matrix-completion/results/figures/exp_uniform_%s_full/", exp)
}else{
  key.values <- c("Query_coverage","Size")
  key.labels <- c("Query cov.","Size")
  height <- 2.5
  fig.dir <- sprintf("~/GitHub/conformal-matrix-completion/results/figures/exp_uniform_%s/", exp)
}
dir.create(fig.dir, showWarnings = FALSE)

if (plot_full){
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Coverage", "Size", "Inf_prop"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method,  r,k, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method,  r,k, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}


## Make nice plots for paper
make_plot <- function(results, exp, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  pp <- results %>%
    filter(!(r %in% c(10, 15)))%>%
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
  if (sv == TRUE){
    ggsave(sprintf("%s/exp_uniform_%s.pdf", fig.dir, exp), pp, device=NULL, width=6.5, height=height)}
  else{
    print(pp)
  }
}

make_plot(results, exp=exp, sv=FALSE)