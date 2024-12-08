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
  key.values <- c("Query_coverage","Size","avg_gap")
  key.labels <- c("Group cov.","Avg. width", "Avg. gap")
  height <- 2.5
}

results_filtered <- results.raw %>%
                    filter(k==5)%>%
                    filter(prop_obs==0.3)
#results_filtered <- results.raw

if (plot_full){
  results <- results_filtered %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Coverage", "Size", "avg_gap","Inf_prop"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, k, r_est, scale, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  results <- results_filtered%>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Size", "avg_gap"), names_to='Key', values_to='Value') %>%
    #pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, k, r_est, scale, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}

if (plot_full){
  results <- results_filtered %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Coverage", "Size", "avg_gap","Inf_prop"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, k, r_est, const, scale, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  results <- results_filtered%>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Size", "avg_gap"), names_to='Key', values_to='Value') %>%
    #pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method,const, k, r_est, scale, Key) %>%
    summarise(
      num = n(),
      Value= median(Value, na.rm = TRUE),
      MAD = mad(Value, na.rm = TRUE),
      Value.se = 1.253 * MAD / sqrt(n())
    )
}

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
    geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.05) +
    geom_hline(data=df.nominal, aes(yintercept=Value)) +
    geom_hline(data=df.placeholder, aes(yintercept=Value), alpha=0) +
    ggh4x::facet_grid2(Key~r_est, scales="free_y", independent = "y") +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    scale_alpha_manual(values=alpha.scale) +
    xlab("Scale") +
    ylab("") +
    theme_bw()
  if (sv == TRUE){
    ggsave(sprintf("%s/exp_est_biased.pdf", fig.dir), pp, device=NULL, width=5.4, height=height)
  }else{
    print(pp)
  }
}

#val <-c(5,10,20,30)
val <-c(1,3,5,7)
val<-unique(results_filtered$r_est)


plot.alpha <- 0.9
df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
  mutate(Key = factor(Key, key.values, key.labels))    
df.placeholder <- tibble(Key=c("Query_coverage"), Value=c(1, 0.7)) %>%
  mutate(Key = factor(Key, key.values, key.labels))

pp <- results %>%
#  filter(const== 20)%>%
#  filter(r_est!=30)%>%
#  filter(scale %in% c(0, 2, 4,6,8))%>%
  ggplot(aes(x=scale, y=Value, color=Method, shape=Method)) +
  geom_point(alpha=0.9) +
  geom_line() +
  geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.01) +
  geom_hline(data=df.nominal, aes(yintercept=Value)) +
  geom_hline(data=df.placeholder, aes(yintercept=Value), alpha=0) +
  ggh4x::facet_grid2(Key~r_est, scales="free_y", independent = "y") +
  scale_color_manual(values=color.scale) +
  scale_shape_manual(values=shape.scale) +
  scale_alpha_manual(values=alpha.scale) +
  xlab("Scale") +
  ylab("") +
  theme_bw()
pp

ggsave(sprintf("%s/exp_est_biased_logistic.pdf", fig.dir), pp, device=NULL, width=6.5, height=3.5)