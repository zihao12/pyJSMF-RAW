rm(list = ls())
set.seed(123)

## todo: spoeed up
## tcrossproduct is much much faster
## use https://brettklamer.com/diversions/statistical/faster-blas-in-r/

findS <- function(Cbar, K){
  R = t(Cbar)
  S = replicate(K, 0)
  diagR = replicate(K, 0)
  
  g = colSums(R**2)
  for(k in 1:K){
    maxCol = which.max(g)
    maxSquaredSum = max(g)
    S[k] = maxCol
    u = R[, maxCol]
    u = u / sqrt(maxSquaredSum)
    diagR[k] = sqrt(maxSquaredSum)
    R = R -  outer(u,(t(R) %*% u)[, 1])
    
    dg = (Cbar %*% u)**2
    g = g - dg
  }
  return(list(S = S, diagR = diagR))
}

compute_C <- function(X){
  out = t(X) %*% X - diag(colSums(X))
  return(out/sum(out))
}



## testing 
simulate_multinomial_counts_local <- function(L, F, s){
  n = nrow(L)
  p = nrow(F)
  Pi = L %*% t(F)
  X = matrix(0, nrow = n, ncol = p)
  for(i in 1:n){
    X[i,] = rmultinom(n = 1, size = s, prob = Pi[i, ])
  }
  return(X)
}

Yfile = "../../ebpmf_data_analysis/output/fastTopics_fit/sla_small.txt"
datafile="../../ebpmf_data_analysis/output/fastTopics_fit/sla_small_fitted.Rds"
vocabfile="../../ebpmf_data_analysis/output/fastTopics_fit/sla_small_vocab.txt"
titlefile="../../ebpmf_data_analysis/data/SLA/title.sla.txt"

data = readRDS(datafile)
L = data$fit_sub$L
F = data$fit_sub$F
s = mean(data$fit_sub$s)
# X =  simulate_multinomial_counts_local(L, F, s)
# C = compute_C(X)

C = tcrossprod(F, L)
C = tcrossprod(C) / nrow(L)
Cbar = diag(1/rowSums(C)) %*% C 

out = findS(Cbar, 6)

out$S




