options(width=160)

library(tidyverse)
library(kableExtra)

setwd("~/GitHub/conformal-matrix-completion")
idir <- "results/exp_residual_hpm"
ifile.list <- list.files(idir)

# Output directory
fig.dir <- "results/figures"

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))

key.values <- c("Pair_coverage", "Coverage", "Size")
key.labels <- c("Paired coverage", "Coverage", "Size")

Method.values <- c("conformal", "naive", "bonferroni", "uncorrected")
Method.labels <- c("Conformal", "Naive", "Bonferroni", "Uncorrected")

color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
shape.scale <- c(15, 4, 8, 1)


results_hpm <- results.raw %>%
  mutate(Method = factor(Method, Method.values, Method.labels)) %>%
  pivot_longer(cols=c(`Pair_coverage`, `Coverage`,`Size`), names_to='Key', values_to='Value') %>%
  mutate(Key = factor(Key, key.values, key.labels)) %>%
  group_by(Method, n1, n2, r_true, r_guess, gamma_n, gamma_m, noise_model, a, b, mu, Key) %>%
  summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))%>%
  #  pivot_wider(names_from = 'Key', values_from = 'Value') %>%
  filter(n1 == 100, n2 == 100, Key!='Coverage')

results_filtered <- results.raw %>%
  filter(n1 == 100, n2 == 100, gamma_n==0.5)


## Make nice plots for paper
make_plot <- function(xmax=2000) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Pair_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  df.ghost <- tibble(Key=c("Pair_coverage","Size"), Value=c(0.9,10), Method="conformal") %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  pp <- results_hpm %>%
    filter(gamma_n == 0.5) %>%
    ggplot(aes(x=mu, y=Value, color=Method, shape=Method)) +
    geom_point(alpha=0.75) +
    geom_line() +
    #geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se)) +
    geom_hline(data=df.nominal, aes(yintercept=Value)) +
    geom_point(data=df.ghost, aes(x=30,y=Value), alpha=0) +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    facet_wrap(.~Key, scales="free") +
    #scale_x_continuous(trans='log10', lim=c(500,xmax), breaks=c(500,1000,2000)) +
    #        scale_y_continuous(trans='log10') +
    xlab("Row-wise noise magnitude") +
    ylab("") +
    theme_bw()
  pp
  #ggsave(sprintf("%s/exp_residual_hpm.pdf", fig.dir), pp, device=NULL, width=5.5, height=2)
}

make_plot()
