options(width=160)

library(tidyverse)
library(kableExtra)
library(ggplot2)
library(dplyr)


setwd("C:/Users/liang/Documents/GitHub/conformal-matrix-completion/experiments_synthetic/results_hpc")
idir <- "results/exp_solver_uniform/"
ifile.list <- list.files(idir)

# Output directory
fig.dir <- "C:/Users/liang/Documents/GitHub/conformal-matrix-completion/results/figures/exp_solver_uniform"
dir.create(fig.dir, showWarnings = FALSE)

results.raw <- do.call("rbind", lapply(ifile.list, function(ifile) {
  df <- read_delim(sprintf("%s/%s", idir, ifile), delim=",", col_types=cols())
}))


Method.values <- c("conformal", "Bonferroni", "Uncorrected")
Method.labels <- c("Simultaneous", "Bonferroni", "Unadjusted")

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
  key.values <- c("Query_coverage","Size")
  key.labels <- c("Group cov.","Avg. width")
  height <- 2.5
}

results_filtered <- results.raw %>% filter(mu==15)
  
if (plot_full){
  results <- results_filtered %>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Coverage", "Size", "Inf_prop"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, Solver, k, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}else{
  results <- results_filtered%>%
    mutate(Method = factor(Method, Method.values, Method.labels)) %>%
    pivot_longer(cols=c("Query_coverage", "Size"), names_to='Key', values_to='Value') %>%
    mutate(Key = factor(Key, key.values, key.labels)) %>%
    group_by(Method, Solver, k, Key) %>%
    summarise(num=n(), Value.se = sd(Value, na.rm=T)/sqrt(n()), Value=mean(Value, na.rm=T))
}

runtime <- results_filtered %>%
    group_by(Solver) %>%
    summarise(
      mean_runtime = mean(Solver_runtime, na.rm = TRUE),
      se_runtime = sd(Solver_runtime, na.rm = TRUE) / sqrt(n())
      )

frob_error <- results_filtered %>%
  group_by(Solver) %>%
  summarise(
    mean_frob = mean(Frobenius_error, na.rm = TRUE),
    se_frob = sd(Frobenius_error, na.rm = TRUE) / sqrt(n())
  )

frob <- results_filtered%>% 
  select("Solver", "Frobenius_error")

# Plot histograms of Frobenius error for each Solver
ggplot(frob, aes(x = Frobenius_error, fill = Solver)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "dodge") +
  facet_wrap(~ Solver, scales = "free") +
  labs(title = "Histogram of Frobenius Error by Solver",
       x = "Frobenius Error",
       y = "Frequency") +
  theme_minimal() +
  theme(legend.position = "none")  # Hide legend since facet wrap shows solver

frob_counts <- frob %>%
  group_by(Solver) %>%
  summarise(count_frob_gte_1 = sum(Frobenius_error >= 1))

print(frob_counts)

## Make nice plots for paper
make_plot <- function(results, solvers, xmax=2000, sv=TRUE) {
  plot.alpha <- 0.9
  df.nominal <- tibble(Key=c("Query_coverage"), Value=plot.alpha) %>%
    mutate(Key = factor(Key, key.values, key.labels))    
  df.placeholder <- tibble(Key=c("Query_coverage"), Value=c(1, 0.7)) %>%
    mutate(Key = factor(Key, key.values, key.labels))
  
  pp <- results %>%
    mutate(Solver = recode(Solver, "pmf" = "als"))%>%
    ggplot(aes(x=k, y=Value, color=Method, shape=Method)) +
    geom_point(alpha=0.9) +
    geom_line() +
    geom_errorbar(aes(ymin=Value-Value.se, ymax=Value+Value.se), width=0.3) +
    geom_hline(data=df.nominal, aes(yintercept=Value)) +
    geom_hline(data=df.placeholder, aes(yintercept=Value), alpha=0) +
    ggh4x::facet_grid2(Key~Solver, scales="free_y", independent = "y") +
    scale_color_manual(values=color.scale) +
    scale_shape_manual(values=shape.scale) +
    scale_alpha_manual(values=alpha.scale) +
    xlab("Group size K") +
    ylab("") +
    theme_bw()
  if (sv == TRUE){
    ggsave(sprintf("%s/exp_solver_uniform.pdf", fig.dir), pp, device=NULL, width=6.5, height=height)
  }else{
    print(pp)
  }
}



sv <- FALSE
solver_list <- c("pmf","nnm", "svt")
make_plot(results, solver_list)
