# Classifying handwritten digits with K-Means and SVM
#
# Author: Alexandre Vilcek
# Version: 1.0

library(MASS)
library(Matrix)
library(e1071)

load_mnist <- function(filepath) {
  load_image_file <- function(filename) {
    f = file(paste0(filepath,filename),'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    nx = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=nx*nrow*ncol,size=1,signed=F)
    ret = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(paste0(filepath,filename),'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train_X <<- load_image_file('train-images.idx3-ubyte')
  test_X <<- load_image_file('t10k-images.idx3-ubyte')
  train_y <<- load_label_file('train-labels.idx1-ubyte')
  test_y <<- load_label_file('t10k-labels.idx1-ubyte')
}

show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}

brt_ctr_norm <- function(img) {
  round((img-apply(img,1,mean))/sqrt(apply(img,1,var)+10))
}

whiten <- function(img,epsilon) {
  mu <- apply(img,2,mean)
  img <- t(apply(img,1,function(x) x-mu))
  A <- t(img)%*%img
  S <- svd(A)
  V <- S$u
  D <- S$d
  wM <- sqrt(dim(img)[1]-1)*V%*%sqrt(ginv(diag(D)+diag(length(D))*epsilon))%*%t(V)
  list(img%*%wM,mu,wM)
}

extract_patches <- function(data,s_patch,n_patches) {
  patches <- matrix(0,n_patches,s_patch^2)
  s_data <- sqrt(dim(data)[2])
  i <- 1
  while(i<=n_patches) {
    if(i%%10000==0) {
      cat('Extracting patch: ',as.integer(i),' of ',as.integer(n_patches),'\n')
    }
    p <- sample(1:(s_data-s_patch+1),2)
    patch <- matrix(data[((i-1)%%s_data)+1,],nrow=s_data,byrow=T)
    patch <- patch[p[1]:(p[1]+s_patch-1),p[2]:(p[2]+s_patch-1)]
    if(sum(patch)!=0) {
      patches[i,] <-as.vector(t(patch))
      i <- i+1
    }
  }
  patches
}

s_kmeans <- function(X,k,iterations) {
  x2 <- apply(X,1,function(x) sum(x^2))
  centroids <- matrix(rnorm(k*dim(X)[2])*0.1,nrow=k)
  BATCH_SIZE <- 1000
  for(itr in 1:iterations) {
    cat('K-means iteration ',itr,' of ',iterations,'\n')
    c2 <- apply(centroids,1,function(x) 0.5*(sum(x^2)))
    summation <- matrix(0,dim(X)[2],nrow=k)
    counts <- matrix(0,k,nrow=1)
    loss <- 0
    for(i in seq(1,dim(X)[1],by=BATCH_SIZE)) {
      lastIndex <- min(i+BATCH_SIZE-1, dim(X)[1])
      m <- lastIndex-i+1
      val <- apply(apply(centroids%*%t(X[i:lastIndex,]),2,function(x) x-c2),2,max)
      labels <- apply(apply(centroids%*%t(X[i:lastIndex,]),2,function(x) x-c2),2,which.max)
      loss <- loss+sum(0.5*x2[i:lastIndex]-val)
      S <- sparseMatrix(i=c(1:m),j=labels,x=1, dims=c(m,k))
      summation <- summation+t(S)%*%X[i:lastIndex,]
      counts <- counts+apply(S,2,sum)
    }
    centroids <- apply(summation,2,function(x) x/counts)
    badIndex <- which(counts==0)
    centroids[badIndex,] <- 0
  }
  centroids
}

im2col <- function(A,m,n) {
  s <- T
  for(j in 1:(dim(A)[2]-n+1)) {
    for(i in 1:(dim(A)[1]-m+1)) {
      if(s) {
        B <- as.vector(A[i:(i+m-1),j:(j+n-1)])
      } else {
        B <- rbind(B,as.vector(A[i:(i+m-1),j:(j+n-1)]))
      }
      s <- F
    }
  }
  B
}

extract_features <- function(X,centroids,rf_size,img_dim,M,WM) {
  XC <- matrix(0,dim(X)[1],dim(centroids)[1]*4)
  for(i in 1:dim(X)[1]) {
    if(i%%100==0) {
      cat('Extracting features: ',as.integer(i),' of ',as.integer(dim(X)[1]),'\n')
    }
    patches <- im2col(matrix(X[i,],img_dim[1],img_dim[2]),rf_size,rf_size)
    patches <- brt_ctr_norm(patches)
    patches <- t(apply(patches,1,function(x) x-M))%*%WM
    #patches <- whiten(patches,0.1)
    xx <- apply(patches,1,function(x) sum(x^2))
    cc <- apply(centroids,1,function(x) sum(x^2))
    xc <- patches%*%t(centroids)
    z <- sqrt(matrix(cc,dim(xc)[1],dim(xc)[2],byrow=T)+(matrix(xx,dim(xc)[1],dim(xc)[2])-2*xc))
    v <- apply(z,1,min)
    inds <- apply(z,1,which.min)
    mu <- apply(z,1,mean)
    patches <- matrix(mu,dim(z)[1],dim(z)[2])-z
    patches[patches<0] = 0
    prows <- img_dim[1]-rf_size+1
    pcols <- img_dim[2]-rf_size+1
    patches <- array(patches,dim=(c(prows,pcols,dim(centroids)[1])))
    halfr <- round(prows/2)
    halfc <- round(pcols/2)
    q1 <- apply(patches[1:halfr,1:halfc,],3,sum)
    q2 <- apply(patches[(halfr+1):prows,1:halfc,],3,sum)
    q3 <- apply(patches[1:halfr,(halfc+1):pcols,],3,sum)
    q4 <- apply(patches[(halfr+1):prows,(halfc+1):pcols,],3,sum)
    XC[i,] <- c(q1,q2,q3,q4)
  }
  XC
}

energy_correlation <- function(Z, epsilon) {
  ncol <- dim(Z)[2]
  indice <- combn(ncol,2)
  ZC <- matrix(0,3,dim(indice)[2])
  for(i in 1:dim(indice)[2]) {
    zj <- Z[,indice[1,i]]
    zk <- Z[,indice[2,i]]
    ZW <- whiten(cbind(zj,zk),epsilon)[[1]]
    zj <- ZW[,1]
    zk <- ZW[,2]
    num <- sum(zj^2*zk^2-1)
    den <- sqrt(sum(zj^4-1)*sum(zk^4-1))
    ZC[1,i] <- indice[1,i]
    ZC[2,i] <- indice[2,i]
    ZC[3,i] <- num/den
  }
  ZC
}

load_mnist("./MNIST/")
idx_train <- sample(1:dim(train_X)[1], trunc(dim(train_X)[1]*50/100))
idx_test <- sample(1:dim(test_X)[1], trunc(dim(test_X)[1]*50/100))
s_train_X <- train_X[idx_train,]
s_train_y <- train_y[idx_train]
s_test_X <- train_X[idx_test,]
s_test_y <- train_y[idx_test]

epsilon <- 0.01
patches <- extract_patches(train_X,7,500000)
n_patches <- brt_ctr_norm(patches)
w_params <- whiten(n_patches,epsilon)
n_patches <- w_params[[1]]

n_centroids <- s_kmeans(n_patches,1600,30)

train_XC <- extract_features(s_train_X, n_centroids, 7, c(28,28),w_params[[2]],w_params[[3]]);
train_XC_mean <- apply(train_XC,2,mean)
train_XC_sd <- sqrt(apply(train_XC,2,var)+0.01)
train_XC <- (train_XC-matrix(train_XC_mean,dim(train_XC)[1],dim(train_XC)[2],byrow=T))/train_XC_sd

test_XC <- extract_features(s_test_X, n_centroids, 7, c(28,28),w_params[[2]],w_params[[3]]);
test_XC_mean <- apply(test_XC,2,mean)
test_XC_sd <- sqrt(apply(test_XC,2,var)+0.01)
test_XC <- (test_XC-matrix(test_XC_mean,dim(test_XC)[1],dim(test_XC)[2],byrow=T))/test_XC_sd

model <- svm(train_XC, s_train_y, type='C-classification', cost=50, cachesize=512)
pred <- predict(model, test_XC)
print(table(pred, s_test_y))
print(1-(length(pred[pred==s_test_y])/length(pred)))

