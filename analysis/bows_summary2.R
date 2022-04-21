rm(list = ls())
out_dir = "../output"

ks <- c(5, 10, 15, 20, 25, 30, 50)
datanames = c("sla", "nips", "kos")
methods = c("pn", "rec1_pn", "rec2_pn", "rec12_pn", "rec", "vanila", "nndsvd" , "random")
numiter=200

# ## real data, initialized, then fitted with scd
# Loglik <- data.frame(matrix(ncol = numiter, nrow = 0))
# dn_ = c()
# k_ = c()
# mn_ = c()
# for(dataname in datanames){
#   for(k in ks){    
#     for(method in methods){
#       modelfile=sprintf("%s/%s_fitted_init_%s_k%d.Rds", out_dir, dataname, method, k)
#       ll = readRDS(modelfile)$progress$loglik[1:numiter]
#       Loglik = rbind(Loglik, ll)
#       dn_ = c(dn_, dataname)
#       k_ = c(k_, k)
#       mn_ = c(mn_, method)
#     }
#   }
# }
# realbows_scd = cbind(dn_, k_, mn_, Loglik)
# colnames(realbows_scd) = c("data", "k", "method", paste0("iter", 1:numiter))
# write.csv(realbows_scd, "realbows_scd.csv", row.names = FALSE)

# ## real data, initialized, then fitted with em
# Loglik <- data.frame(matrix(ncol = numiter, nrow = 0))
# dn_ = c()
# k_ = c()
# mn_ = c()
# for(dataname in datanames){
#   for(k in ks){    
#     for(method in methods){
#       modelfile=sprintf("%s/%s_fitted_em_init_%s_k%d.Rds", out_dir, dataname, method, k)
#       ll = readRDS(modelfile)$progress$loglik[1:numiter]
#       Loglik = rbind(Loglik, ll)
#       dn_ = c(dn_, dataname)
#       k_ = c(k_, k)
#       mn_ = c(mn_, method)
#     }
#   }
# }
# realbows_em = cbind(dn_, k_, mn_, Loglik)
# colnames(realbows_em) = c("data", "k", "method", paste0("iter", 1:numiter))
# write.csv(realbows_em, "realbows_em.csv", row.names = FALSE)

## simulated data, initialized, then fitted with em
# n_eval = 101
# progress <- data.frame(matrix(ncol = numiter, nrow = 0))
# Loglik <- data.frame(matrix(ncol = n_eval, nrow = 0))
# Loglik_val <- data.frame(matrix(ncol = n_eval, nrow = 0))
# dn_ = c()
# k_ = c()
# mn_ = c()
# for(dataname in datanames){
#   for(k in ks){    
#     for(method in methods){
#       modelfile=sprintf("%s/%s_sim_fitted_em_init_%s_k%d.Rds", out_dir, dataname, method, k)
#       model=readRDS(modelfile)
#       Loglik_val = rbind(Loglik_val, model$Loglik$val)
#       Loglik = rbind(Loglik, model$Loglik$train)
#       progress = rbind(progress, model$fitted$progress$loglik[1:numiter])
#       dn_ = c(dn_, dataname)
#       k_ = c(k_, k)
#       mn_ = c(mn_, method)
#     }
#   }
# }
# simbows_em = cbind(dn_, k_, mn_, progress)
# colnames(simbows_em) = c("data", "k", "method", paste0("iter", 1:numiter))
# write.csv(simbows_em, "simbows_em.csv", row.names = FALSE)

# simbows_em_val = cbind(dn_, k_, mn_, Loglik_val)
# colnames(simbows_em_val) = c("data", "k", "method", paste0("iter", 1:n_eval))
# write.csv(simbows_em_val, "simbows_em_val.csv", row.names = FALSE)

# simbows_em_train = cbind(dn_, k_, mn_, Loglik)
# colnames(simbows_em_train) = c("data", "k", "method", paste0("iter", 1:n_eval))
# write.csv(simbows_em_train, "simbows_em_train.csv", row.names = FALSE)

n_eval = 101
progress <- data.frame(matrix(ncol = numiter, nrow = 0))
Loglik <- data.frame(matrix(ncol = n_eval, nrow = 0))
Loglik_val <- data.frame(matrix(ncol = n_eval, nrow = 0))
dn_ = c()
k_ = c()
mn_ = c()
for(dataname in datanames){
  for(k in ks){    
    for(method in methods){
      modelfile=sprintf("%s/%s_sim_fitted_init_%s_k%d.Rds", out_dir, dataname, method, k)
      model=readRDS(modelfile)
      Loglik_val = rbind(Loglik_val, model$Loglik$val)
      Loglik = rbind(Loglik, model$Loglik$train)
      progress = rbind(progress, model$fitted$progress$loglik[1:numiter])
      dn_ = c(dn_, dataname)
      k_ = c(k_, k)
      mn_ = c(mn_, method)
    }
  }
}
simbows_scd = cbind(dn_, k_, mn_, progress)
colnames(simbows_scd) = c("data", "k", "method", paste0("iter", 1:numiter))
write.csv(simbows_scd, "simbows_scd.csv", row.names = FALSE)

simbows_scd_val = cbind(dn_, k_, mn_, Loglik_val)
colnames(simbows_scd_val) = c("data", "k", "method", paste0("iter", 1:n_eval))
write.csv(simbows_scd_val, "simbows_scd_val.csv", row.names = FALSE)

simbows_scd_train = cbind(dn_, k_, mn_, Loglik)
colnames(simbows_scd_train) = c("data", "k", "method", paste0("iter", 1:n_eval))
write.csv(simbows_scd_train, "simbows_scd_train.csv", row.names = FALSE)