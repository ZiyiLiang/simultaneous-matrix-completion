options(width=160)

library(tidyverse)
library(kableExtra)

setwd("~/GitHub/conformal-matrix-completion/experiments_synthetic/results_hpc")
idir <- "results/exp_uniform/"
ifile.list <- list.files(idir)

# Output directory
fig.dir <- "~/GitHub/conformal-matrix-completion/results/figures"

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

key.values <- c("Query_coverage", "Coverage", "Size")
key.labels <- c("Query coverage", "Coverage", "Size")

Method.values <- c("conformal", "Bonferroni", "Uncorrected")
Method.labels <- c("Conformal", "Bonferroni", "Uncorrected")

color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
shape.scale <- c(15, 4, 8, 1)


results_hpm <- results.raw %>%
  mutate(Method = factor(Method, Method.values, Method.labels)) %>%
  pivot_longer(cols=c(`Query_coverage`, `Coverage`,`Size`), names_to='Key', values_to='Value') %>%
  mutate(Key = factor(Key, key.values, key.labels)) %>%
  group_by(Method, gamma_n, gamma_m, mu,k, Key) %>%
  summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T)) 
# %>%
#   filter(Key!='Coverage')

results_filtered <- results.raw %>%
  mutate(Method = factor(Method, Method.values, Method.labels)) %>%
  pivot_longer(cols=c(`Query_coverage`, `Coverage`,`Size`), names_to='Key', values_to='Value') %>%
  mutate(Key = factor(Key, key.values, key.labels)) %>%
  group_by(Method, gamma_n, gamma_m, mu, Key) %>%
  filter(Method=='Uncorrected')%>%
  summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))


# results_filtered <- results.raw %>%
#   filter(Method=='Uncorrected')



## Make nice plots for paper
make_plot <- function(exp, val, xmax=2000) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  df.ghost <- tibble(Key=c("Query_coverage", "Coverage", "Size"), Value=c(0.9,0.9,10), Method="conformal") %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    mutate(Key = factor(Key, key.values, key.labels)) 
  
  if (exp=="vary_k"){
    pp <- results_hpm %>%
      filter(mu==val)%>%
      ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se)) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      geom_point(data=df.ghost, aes(x=0.5,y=Value), alpha=0) +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      facet_wrap(.~Key, scales="free") +
      xlab("Query size K") +
      ylab("") +
      theme_bw()
    ggsave(sprintf("%s/exp_uniform_%s_mu%i.pdf", fig.dir, exp, val), pp, device=NULL, width=8, height=2)
  }
  else{
    pp <- results_hpm %>%
      filter(k==val)%>%
      ggplot(aes(x=mu, y=Value, color=Method, shape=Method)) +
      geom_point(alpha=0.75) +
      geom_line() +
      geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se)) +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      geom_point(data=df.ghost, aes(x=0.5,y=Value), alpha=0) +
      scale_color_manual(values=color.scale) +
      scale_shape_manual(values=shape.scale) +
      facet_wrap(.~Key, scales="free") +
      xlab("Column-wise magnitude") +
      ylab("") +
      theme_bw()
    ggsave(sprintf("%s/exp_uniform_%s_k%i.pdf", fig.dir, exp, val), pp, device=NULL, width=8, height=2)
    
  }
}

exp_list <- c("vary_k", "vary_mu")
k_list <- 2:10
mu_list <-  seq(0, 30, by = 3)


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
