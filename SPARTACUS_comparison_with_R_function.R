library(ClustVarLV)
#setwd("C:/Users/T460s/Documents/Git_projects/SPARTACUS/tests")

# matrixA

matrixA <- read.csv2("matrixA.csv", sep = ",")[,-1]
Z = CLV(matrixA, method = 1)
write.csv(Z$partition4$clusters, 'R_labels_matrixA.csv', row.names = F, quote = F)

#matrixB

matrixB <- read.csv2("matrixB.csv", sep = ",")[,-1]
Z = CLV(matrixB, method = 1,nmax = 101, maxiter = 101)
write.csv(Z$partition8$clusters, 'R_labels_matrixB.csv', row.names = F, quote = F)