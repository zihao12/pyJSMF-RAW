## generate some summary of various anchor word methods on realbows and simbows data
out_dir = "../output"

realbows_ll = list()
for(dataname in c("sla", "nips", "kos")){
  numiter=1000
  ks <- c(5, 10, 15, 20, 25, 30, 50)
  methods = c("pn", "rec1_pn", "rec2_pn", "rec12_pn", "rec", "vanila", "nndsvd" , "random")
  which_best = c()
  for(k in ks){
    Loglik <- data.frame(matrix(ncol = length(methods), nrow = numiter))
    colnames(Loglik) <- methods
    for(method in methods){
      modelfile=sprintf("%s/%s_fitted_init_%s_k%d.Rds", out_dir, dataname, method, k)
      Loglik[[method]] = readRDS(modelfile)$progress$loglik
    }
    which_best <- c(which_best, names(sort(Loglik[1, ], decreasing = TRUE))[1])
  }
  realbows_ll[[dataname]] = data.frame(k = ks, which_best = which_best)
}


simbows_ll = list()
simbows_ll_val = list()
for(dataname in c("sla", "nips", "kos")){
  n_eval = 101
  ks <- c(5, 10, 15, 20, 25, 30, 50)
  methods = c("pn", "rec1_pn", "rec2_pn", "rec12_pn", "rec", "vanila", "nndsvd" , "random")

  overfits = c()
  which_best = c()
  which_best_val = c()
  for(k in ks){
    Loglik_val <- data.frame(matrix(ncol = length(methods), nrow = n_eval))
    Loglik <- data.frame(matrix(ncol = length(methods), nrow = n_eval))
    colnames(Loglik_val) <- methods
    colnames(Loglik) <- methods
    for(method in methods){
      modelfile=sprintf("%s/%s_sim_fitted_init_%s_k%d.Rds", out_dir, dataname, method, k)
      model=readRDS(modelfile)
      Loglik_val[[method]] = model$Loglik$val
      Loglik[[method]] = model$Loglik$train
    }
    which_best <- c(which_best, names(sort(Loglik[1, ], decreasing = TRUE))[1])
    which_best_val <- c(which_best_val, names(sort(Loglik_val[1, ], decreasing = TRUE))[1])
  }
  simbows_ll[[dataname]] = data.frame(k = ks, which_best = which_best)
  simbows_ll_val[[dataname]] = data.frame(k = ks, which_best = which_best_val)
}

saveRDS(list(realbows_ll = realbows_ll,simbows_ll = simbows_ll, simbows_ll_val = simbows_ll_val), "bows_summary.Rds")