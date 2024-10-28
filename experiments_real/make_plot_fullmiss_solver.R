options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)

exp="movielens"

idir <- sprintf("results/est_%s_nnm_fullmiss/", exp)
setwd("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/experiments_real/results_hpc/")
ifile.list <- list.files(idir)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim = ",", col_types = cols())
}))

Method.values <- c("conformal", "Bonferroni", "Uncorrected")
Method.labels <- c("Simultaneous", "Bonferroni", "Unadjusted")

#color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
color.scale <- c( "blue", "#56b5e9", "#CC66CC" )
shape.scale <- c(15, 4, 8, 1)
alpha.scale <- c(1, 0.5, 0.8)



key.values <- c("Size")
key.labels <- c("Avg. width")
fig.dir <- sprintf("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/results/figures/%s/", exp)

dir.create(fig.dir, showWarnings = FALSE)

results <- results.raw %>%
  mutate(Method = factor(Method, Method.values, Method.labels)) %>%
  pivot_longer(cols=c("Size"), names_to='Key', values_to='Value') %>%
  mutate(Key = factor(Key, key.values, key.labels)) %>%
  group_by(Method,  r,k, Key) %>%
  summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
# %>%
#   filter(Method != "Unadjusted")



## Make nice plots for paper
make_plot <- function(results, exp, xmax=2000, sv=TRUE) {
  pp <- results %>%
    filter(Method!="Unadjusted")%>%
    mutate(r = sprintf("Guessed rank: %i", r))%>%
    ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
    geom_point(alpha=0.75) +
    geom_line() +
    geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.5) +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    facet_grid(Key~r, scales="free") +
    xlab("Group size K") +
    ylab("") +
    theme_bw()
  if (sv == TRUE){
    ggsave(sprintf("%s/%s_fullmiss.pdf", fig.dir, exp), pp, device=NULL, width=6.5, height=1.6)}
  else{
    print(pp)
  }
}

make_plot(results, exp=exp, sv=TRUE)