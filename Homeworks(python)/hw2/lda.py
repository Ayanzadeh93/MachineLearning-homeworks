# import the required libraries
import os
import numpy as np
import matplotlib.pyplot as plt

# load train and test set
train = np.loadtxt(open(os.path.join('data', 'optdigits.tra'), "rb"), delimiter=",")
test =np.loadtxt(open(os.path.join('data', 'optdigits.tes'), "rb"), delimiter=",")

# print the shape of train and test set
print ("\nTraining set shape: {}".format(train.shape))
print ("Test set shape: {}".format(test.shape))




####################################################
####################################################
#												   #
#		Some code to prevent python				   #
#		from outputing ComplexWarning			   #
#												   #
####################################################
####################################################
import warnings
warnings.filterwarnings("ignore")





# Slicing features and labels from data
X_train = train[:,:64]
y_train = train[:,64]
X_test = test[:,:64]
y_test = test[:,64]


# print shape of features and labels
print("\nX_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))


# get the number of classes, features, train, and test samples
num_class = len(set(y_train))
num_features = len(X_train[1])
num_train = len(y_train)
num_test = len(y_test)

#

# create an empty matrix for mean matrix of classes
M = np.zeros((num_class, num_features))


# create an empty matrix for overall mean
M_overall = np.zeros((num_class, num_features))


# create an empty matrix for within-class scatter
Sw = np.zeros((num_features, num_features))

# create an empty matrix for between-class scatter
Sb = np.zeros((num_features, num_features))


# calculate within and between class scatter
for i in range(num_class):
    
    # boolean masking of class i
    r = (y_train==i)

    
    # extracting class i features from training set
    X_class = X_train[np.where(r)[0], :]
    
    # calculating mean of features for class i
    M[i] = np.mean(X_class, axis=0)

    
    # claculate the within_class_scatter
    Sw += (X_class - M[i]).T.dot(X_class - M[i])
    
    # calculate the overall mean
    M_overall += (1/num_class) * M[i]


    # calculate the between-class scatter
    Sb += np.sum(r) * (M[i] - M_overall).T.dot(M[i] - M_overall)
    
    
# caculate the inverse of within_class scatter
Sw_pinv = np.linalg.pinv(Sw)


# calculating dot product of within-class scatter matrix invers with between_class scatter
Sw_pinv_Sb = Sw_pinv.dot(Sb)



# find the eigenvalues and eigenvectors
eigenValues, eigenVectors = np.linalg.eig(Sw_pinv_Sb)


# sort the eigenvalues and eigenvectors in decreasing order
idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]


# Select 2 largest eigenvectors
W = eigenVectors[:,:2]


# transform the train and test set to 2 dimentional space
X_train_2d = X_train.dot(W)
X_test_2d = X_test.dot(W)


# plot the scatter plot of training set after LDA
plt.figure()
fig = plt.gcf()
fig.set_size_inches(14, 14)
plt.scatter(X_train_2d[:,0], X_train_2d[:,1], c='y')

np.random.seed(41)
# generate ramdom integers to label some samples
random_samples =  np.random.randint(low=0, high=num_train-1, size=150)

for num in random_samples:
    plt.annotate(str(y_train[num]), xy=(X_train_2d[num,0], X_train_2d[num,1]))

plt.title("Train set Optdigits after LDA")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.savefig('trainoptdigits.png')



# plot the scatter plot of test set after LDA
plt.figure()
fig = plt.gcf()
fig.set_size_inches(14, 14)
plt.scatter(X_test_2d[:,0], X_test_2d[:,1], c='c')

# generate ramdom integers to label some samples
random_samples =  np.random.randint(low=0, high=num_test-1, size=150)

for num in random_samples:
    plt.annotate(str(y_test[num]), xy=(X_test_2d[num,0], X_test_2d[num,1]))



plt.title("Test set Optdigits after LDA")
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.savefig('testoptdigits.png')

