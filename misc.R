match_topics <- function(F1, F2, use_top = NULL){
  K = ncol(F1)
  p = nrow(F1)
  id = replicate(K, NaN)
  for(i in 1:K){
    f1 = F1[,i]
    if(is.null(use_top)){
      idx = 1:p
    }else{
      idx = sort(f1, index.return = TRUE, decreasing = TRUE)$ix[1:use_top]
    }
    dist_min = Inf
    for(j in 1:K){
      dist = sum((f1[idx] - F2[idx,j])^2)
      if(dist < dist_min){
        dist_min = dist
        matched = j
      }
    }
    id[i] <- matched
  }
  return(id)
}

# Simulate counts from the multinomial topic model with factors F,
# loadings L and sample sizes s.
simulate_multinom_counts <- function (L, F, s) {
  n <- nrow(L)
  m <- nrow(F)
  X <- matrix(0,n,m)
  P <- tcrossprod(L,F)
  for (i in 1:n)
    X[i,] <- rmultinom(1,s[i],P[i,])
  return(X)
}

