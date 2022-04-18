from scipy import sparse, io

Y = io.mmread("docword.sla.txt.old")
Y = Y.tocsr().tocoo()
with open("docword.sla.txt","w") as file:

    file.write(f"{Y.shape[0]}\n")
    file.write(f"{Y.shape[1]}\n")
    file.write(f"{Y.data.sum()}\n")
    for i in range(Y.data.shape[0]):
        file.write(f"{Y.row[i] + 1} {Y.col[i] + 1} {Y.data[i]}\n")

