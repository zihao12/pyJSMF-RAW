args = commandArgs(trailingOnly=TRUE)
dataname=args[1]
method=args[2]
k=as.integer(args[3])
numiter=1000

# dataname="nips"
# method="pn"
# k=10
# numiter=1000

library(reticulate)
library(fastTopics)

init_dir = "../output"
data_dir = "../dataset/real_bows"
out_dir = "../output"

datafile=sprintf("%s/%s_X.mtx", data_dir, dataname)
initfile=sprintf("%s/%s_init_%s_k%d.pkl", init_dir, dataname, method, k)
outfile=sprintf("%s/%s_fitted_init_%s_k%d.Rds", init_dir, dataname, method, k)

## load data
X = Matrix::readMM(datafile)
X <- as(X, "dgCMatrix")
## load init
init = py_load_object(initfile, pickle = "pickle")
F0 = t(init$H)
L0 = init$W
rm(init)

fit0 = init_poisson_nmf(X, F0, L0, verbose = "detailed")
fitted = fit_poisson_nmf(X, fit0 = fit0, numiter = numiter, method = "scd")
saveRDS(fitted, outfile)
