
library(caret) 

train_data<-read.table("optdigits.tra", sep ="," )

test_data<-read.table("optdigits.tes", sep = ",")

y_train=matrix(train_data[,65], nrow=3823, ncol=1)


y_test = test_data[,65]
df_y_t<-data.frame(y_test)


dim(train_data)
dim(test_data)

X_test = test_data[,(1:64)]
X_train = train_data[,(1:64)]
dim(X_test)

colors = c("red", "yellow", "green", "violet", "orange", "blue", "pink", "cyan","black","skyblue")
hist(y_train, xlim = c(-2,10), col = colors, ylim = c(0,500), main="Histogram of training sets for each classes",
     xpd=TRUE,las=2, lwd=2)
axis(1, 0:9, las=2)

var_of_testset<-NULL #variance for test set
number_of_class=10
for (i in 1:64){ 
    var_of_testset[i]=var(X_test[,i])
    
}


var_of_trainset<-NULL
number_of_class=10
for (i in 1:64){ 
    var_of_trainset[i]=var(X_train[,i])
    
}




vectorize_var<-matrix(var_of_trainset, nrow=64, ncol=1)
vectorize_test<-matrix(var_of_testset, nrow=64, ncol=1)#test set

zero_var=which(vectorize_var==0)
zero_test=which(vectorize_test==0)


df<-data.frame(X_train)

dt<-data.frame(X_test)

 df<-df[,-1]
 df<-df[,-39]
 dt<-dt[,-1]
 dt<-dt[,-39]
print("The Feature with variance of Zero has been deleted!!")
print("number of features:")
ncol(df)
ncol(dt)
vr<-NULL

for(i in 1:62)
    {
   vr[i]<-var(df[,i])
    
}

ss=matrix(nrow = 10,ncol = 62)


#train_classify<-NULL
#y_sample<-NULL
mean_matrix=matrix(0,nrow=10,ncol =62)
CovarianceMatrix <- array(0, dim=c(10,62,62))

#freqency_of_casses=matrix(NA, nrow=10,ncol = 1)
common_co=matrix(0,nrow = 62,ncol = 62)
df_y_sample<-NULL
nume_of_e_class<-matrix(0,nrow = 10,ncol = 1)

#jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000
mean_matrix<-matrix(1,nrow=10,ncol =62)

arr0 <- array(1, dim=c(10,62,62))


for(j in 0:9)
    {
    train_classify<-df[df_y_sample[i,1],]
    
y_sample<-which(y_train==j)
    
df_y_sample<-data.frame(y_sample)
k=nrow(df_y_sample)
nume_of_e_class[j+1,1]<-nrow(df_y_sample) #store the number of each class data point frequency   

for(i in 1:k)
    {
    
    train_classify[i,]<-df[df_y_sample[i,1],]

}
    c<-train_classify
     mean_matrix[j+1,]<-apply(train_classify,2,mean)

  SubtractedMatrix<-NULL
  SubtractedMatrix = matrix(0, nrow = k, ncol = 62)
  
    
    
  for(i in 1:k)
  {
    SubtractedMatrix[i,]=as.matrix(train_classify[i,]-mean_matrix[j+1,])
  }
  #Generating Covariance matrix
  CovarianceMatrix[j+1,,]<-as.matrix(t(SubtractedMatrix))%*%as.matrix(SubtractedMatrix)/k

   
   
    y_sample<-NULL
    
     }

dim(CovarianceMatrix)

pri_prob<-matrix(0,nrow = 10,ncol = 1)

#priror probability of class for commoncovariance
c_freq<-c(376, 389, 380,389,387,376,377,387,380,382) 
#pri_prob<-matrix(0,nrow = 10,ncol = 1)
Num_sample=3823
for(i in 0:9)
    {
    pri_prob[i+1,1]<-c_freq[i+1]/Num_sample
}

p_diag=matrix(1:62,ncol =1,nrow = 62)

#common covariance

common_co=matrix(0,nrow = 62,ncol = 62)
for(i in 0:9)
    {
    common_co<-common_co+(pri_prob[i+1,1]*CovarianceMatrix[i+1,,])
}


mean_co<-mean(diag(common_co))
df_diag_co<-data.frame(diag(common_co))

mean_co


diag_com1<-data.frame(diag(common_co))
var_com=apply(diag_com1,2,mean)


dim(mean_matrix)


#discriminant function
discriminant_fu <- function(X, M, S, P) {
    
g = matrix(0,nrow = 3823,ncol = 10)
   for (i in 0:9){
       
   
       
   g[,i+1] = -1/2 * sum(((as.matrix(t(X)) - as.vector(M[i+1]))^2)/S) +  log10(P[i+1])
       
     
       
          

}
    return (g) 
  
}


discriminant_fu <- function(TrainData_X, MeanData, VarianceData, PriorData)
{
  g = matrix(0,nrow = dim(TrainData_X)[1],ncol = 10)
  for(i in 0:9)
  {
    mean_tmp <- as.vector(MeanData[i+1,])
    g[,i+1] = -(1/2) * colSums(((as.matrix(t(TrainData_X)) - mean_tmp)^2)/VarianceData)+log(PriorData[i+1])

  }
  return(g)
}

dim(df)

#finding the discriminant in two mode (Q1a and Q1b)
g_train = discriminant_fu(df, mean_matrix, diag(common_co), pri_prob)
g_train2 = discriminant_fu(dt, mean_matrix, diag(common_co), pri_prob)
g_train3 = discriminant_fu(df, mean_matrix, mean_co, pri_prob)
g_train4 = discriminant_fu(dt, mean_matrix, mean_co, pri_prob)#test mode

df_g<-data.frame(g_train)
gmax<-apply(df_g,1,max)
max_g<-matrix(0,nrow =3823,ncol = 1 )
max_g_t<-matrix(0,nrow = 1797,ncol = 1)
num_train<-3823
max_g1<-matrix(0,nrow =3823,ncol = 1 )
max_g_t1<-matrix(0,nrow = 1797,ncol = 1)
nume_of_testset=1797
numee_of_trainset=3823


for(i in 1:3823) #for train set 
    {
   max_g[i,1]<-which.max(g_train[i,]) # find the maximum of g for each row
}
train_output<-max_g-1
counter=0
for(i in 1:3823)
    {
    if(train_output[i,1]==y_train[i,1]){ #find the number of true prediction
      counter=counter+1  
    }
    
}

accuracy=counter/numee_of_trainset
error=1-accuracy

print("rate of error for whole classes")
print(error)
print("Accuracy of whole classes for training data:")
print(accuracy)
for(j in 0:9)
    {
    
    
 
}



expected<-as.matrix(y_train)
dim(expected)
dim(expected)


results6 <- confusionMatrix(data=train_output, reference=expected)
results6

for(i in 1:1797) #for test set 
    {
   max_g_t[i,1]<-which.max(g_train2[i,])
}
train_output<-max_g_t-1
counter2=0
for(i in 1:1797)
    {
    if(train_output[i,1]==df_y_t[i,1]){
      counter2=counter2+1  
    }
    
}

counter2/nume_of_testset

counter2
accuracy5<-counter2/nume_of_testset
print("rate of error for whole classes")
print(1-accuracy5)
print("Accuracy of whole classes for training data:")
print(accuracy)


expected1<-as.matrix(df_y_t)

results <- confusionMatrix(data=train_output, reference=expected1)
results

counter3=0
for(i in 1:3823) #for train set hypersphere
    {
   max_g1[i,1]<-which.max(g_train3[i,])
}
train_output3<-max_g1-1

for(i in 1:3823)
    {
    if(train_output3[i,1]==y_train[i,1]){
      counter3=counter3+1  
    }
    
}

accuracy2=counter3/numee_of_trainset
error2=1-accuracy2

print("rate of error for whole classes")
print(error2)
print("Accuracy of whole classes for training data:")
print(accuracy2)
for(j in 0:9)
    {
    
    

}


expected<-as.matrix(y_train)


results2 <- confusionMatrix(data=train_output3, reference=expected)
results2

counter4=0
for(i in 1:1797) #for test set with hyperspher
    {
   max_g_t1[i,1]<-which.max(g_train4[i,])
}
train_output4<-max_g_t1-1

for(i in 1:1797)
    {
    if(train_output4[i,1]==df_y_t[i,1]){
      counter4=counter4+1  
    }
    
}
#index_predict<-NULL
#train_output<-NULL
print("accuracy:")
accuracy4=counter4/nume_of_testset
#train_output
print(accuracy4)

print("error of test sets")

error4=1-accuracy4
print(error4)


library(caret)


expected1<-as.matrix(df_y_t)

results <- confusionMatrix(data=train_output4, reference=expected1)
results






