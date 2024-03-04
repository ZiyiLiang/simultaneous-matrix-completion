options(width=160)

library(tidyverse)
library(kableExtra)

setwd("~/GitHub/conformal-matrix-completion/experiments_synthetic/results_hpc")
idir <- "results/exp_biased_obs/"
ifile.list <- list.files(idir)

# Output directory
fig.dir <- "~/GitHub/conformal-matrix-completion/results/figures/exp_biased_obs/"
dir.create(fig.dir, showWarnings = FALSE)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

key.values <- c("Query_coverage", "Coverage", "Size","Inf_prop")
key.labels <- c("Query coverage", "Coverage", "Size", "Inf_prop")

Method.values <- c("conformal", "Bonferroni", "Uncorrected")
Method.labels <- c("Simultaneous", "Bonferroni", "Uncorrected")

color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
shape.scale <- c(15, 4, 8, 1)


# results <- results.raw %>%
#   mutate(Method = factor(Method, Method.values, Method.labels)) %>%
#   pivot_longer(cols=c(`Query_coverage`, `Coverage`,`Size`), names_to='Key', values_to='Value') %>%
#   mutate(Key = factor(Key, key.values, key.labels)) %>%
#   group_by(Method, scale,k, Key) %>%
#   summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T)) %>%
#   filter(Key!='Coverage')

results <- results.raw %>%
  mutate(Method = factor(Method, Method.values, Method.labels)) %>%
  pivot_longer(cols=c(`Query_coverage`, `Coverage`,`Size`, `Inf_prop`), names_to='Key', values_to='Value') %>%
  mutate(Key = factor(Key, key.values, key.labels)) %>%
  group_by(Method, scale,k, Key) %>%
  summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))

# results_filtered <- results.raw %>%
#   mutate(Method = factor(Method, Method.values, Method.labels)) %>%
#   pivot_longer(cols=c(`Query_coverage`, `Coverage`,`Size`), names_to='Key', values_to='Value') %>%
#   mutate(Key = factor(Key, key.values, key.labels)) %>%
#   group_by(Method, gamma_n, gamma_m, mu, Key) %>%
#   filter(Method=='Uncorrected')%>%
#   summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))


# results_filtered <- results.raw %>%
#   filter(Method=='Uncorrected')



## Make nice plots for paper
make_plot <- function(exp, val, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  df.ghost <- tibble(Key=c("Query_coverage", "Coverage", "Size","Inf_prop"), Value=c(0.9,0.9,10,0), Method="conformal") %>%
  #df.ghost <- tibble(Key=c("Query_coverage", "Size"), Value=c(0.9,10), Method="conformal") %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    mutate(Key = factor(Key, key.values, key.labels)) 
  
  if (exp=="vary_k"){
    pp <- results %>%
      filter(scale==val)%>%
      ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.5) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      facet_wrap(.~Key, scales="free") +
      xlab("Query size K") +
      ylab("") +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_biased_obs_%s_scale%.1f.pdf", fig.dir, exp, val), pp, device=NULL, width=5.5, height=2)}
    else{
      pp
    }
  }
  else{
    pp <- results %>%
      filter(k==val)%>%
      ggplot(aes(x=scale, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.5) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      facet_wrap(.~Key, scales="free") +
      xlab("Missingness scale") +
      ylab("") +
      theme_bw()
    if (sv == TRUE){
      ggsave(sprintf("%s/exp_biased_obs_%s_k%i.pdf", fig.dir, exp, val), pp, device=NULL, width=5.5, height=2)}
    else{
      pp
    }
  }
}

exp_list <- c("vary_k", "vary_scale")
k_list <- seq(2, 15, by = 2)
scale_list <-  seq(0, 1.6, by = 0.2)


for (exp in exp_list) {
  if (exp == "vary_k") {
    l <- scale_list
  }
  else {
    l <- k_list
  }
  for (val in l){
    make_plot(exp, val)
  }
}
