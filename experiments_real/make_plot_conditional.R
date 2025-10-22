options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)

fullmiss <- TRUE

if (fullmiss){
  idir <- sprintf("results/exp_movielens_conditional_fullmiss")
  exp <- "movielens_conditional_fullmiss"
  height <- 1.6
}else{
  idir <- sprintf("results/exp_movielens_conditional")
  exp <- "movielens_conditional"
  height <- 2.5
}

setwd("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/experiments_real/results_hpc/")
ifile.list <- list.files(idir)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim = ",", col_types = cols())
}))

Method.values <- c("conditional", "unconditional")
Method.labels <- c( "Conditional", "Marginal")

#color.scale <- c("#566be9", "#56b5e9", "#CC79A7", "orange")
color.scale <- c( "blue", "#56b5e9")
shape.scale <- c(15, 4 )
alpha.scale <- c(1, 0.5)

fig.dir <- sprintf("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/results/figures/%s/", exp)
dir.create(fig.dir, showWarnings = FALSE)


if (fullmiss){
  key.values <- c("Size")
  key.labels <- c("Avg. width")
  
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method,  n1, n2, genre, k, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  key.values <- c("Query_coverage","Size")
  key.labels <- c("Group cov.","Avg. width")
  
  results <- results.raw %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage","Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method,  n1, n2, genre, k, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}


## Make nice plots for paper
make_plot <- function(results, k_list, genres, exp, xmax=2000, sv=TRUE, fullmiss=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))   
  df.placeholder <- tibble(Key=c("Query_coverage"), Value=c(0.95, 0.85)) %>%
    mutate(Key = factor(Key, key.values, key.labels))
  
  pp <- results %>%
    filter(genre %in% genres)%>%
    filter(k %in% k_list)%>%
    ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
    geom_point(alpha=0.75) +
    geom_line() +
    geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.5)
  
  if (fullmiss==0){
    pp <- pp +
      geom_hline(data=df.nominal, aes(yintercept=Value)) +
      geom_hline(data=df.placeholder, aes(yintercept=Value), alpha=0)
  }
  
  pp <- pp +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    facet_grid(Key~genre, scales="free") +
    xlab("Group size K") +
    ylab("") +
    theme_bw()
  if (sv == TRUE){
    ggsave(sprintf("%s/movielens_%s.pdf", fig.dir, exp), pp, device=NULL, width=6.5, height=height)}
  else{
    print(pp)
  }
}

<<<<<<< Updated upstream
k_list <- c(2,3,4,5)
genres <- c("Children's", "Crime", "Drama", "Romance")
make_plot(results, k_list, genres, exp, sv=TRUE, fullmiss=fullmiss)
=======
make_plot(results, exp, sv=FALSE)
>>>>>>> Stashed changes

