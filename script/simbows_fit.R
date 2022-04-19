args = commandArgs(trailingOnly=TRUE)
dataname=args[1]
method=args[2]
k=as.integer(args[3])
numiter=100

# dataname="nips"
# method="pn"
# k=10
# numiter=100

library(reticulate)
library(fastTopics)

init_dir = "../output"
data_dir = "../dataset/sim_bows"
out_dir = "../output"

datafile=sprintf("%s/%s_sim_k%d_X.mtx", data_dir, dataname, k)
valfile=sprintf("%s/%s_sim_k%d_Xval.mtx", data_dir, dataname, k)

initfile=sprintf("%s/%s_sim_init_%s_k%d.pkl", init_dir, dataname, method, k)
outfile=sprintf("%s/%s_sim_fitted_init_%s_k%d.Rds", init_dir, dataname, method, k)

## load data
X = Matrix::readMM(datafile)
X <- as(X, "dgCMatrix")
Xval = Matrix::readMM(valfile)
Xval <- as(Xval, "dgCMatrix")
## load init
init = py_load_object(initfile, pickle = "pickle")
F0 = t(init$H)
L0 = init$W
rm(init)

Loglik <- data.frame(matrix(ncol = 2, nrow = numiter + 1))
colnames(Loglik) <- c("train", "val")

fitted = init_poisson_nmf(X, F0, L0, verbose = "detailed")
Loglik[1, 1] = sum(loglik_poisson_nmf(X, fitted))
Loglik[1, 2] = sum(loglik_poisson_nmf(Xval, fitted))

for(i in 1:numiter){
    fitted = fit_poisson_nmf(X, fit0 = fitted, numiter = 10, method = "scd")
    Loglik[i+1, 1] = sum(loglik_poisson_nmf(X, fitted))
    Loglik[i+1, 2] = sum(loglik_poisson_nmf(Xval, fitted))
}

out = list(fitted=fitted, Loglik=Loglik)
saveRDS(out, outfile)

