# load the required libraries
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


f = open("output.txt", "w")
# load train and test data
train = np.loadtxt(open(os.path.join('data', 'optdigits.tra'), "rb"), delimiter=",")
test =np.loadtxt(open(os.path.join('data', 'optdigits.tes'), "rb"), delimiter=",")

print("\nOriginal train and test data set shapes:")
print ("Training set shape: {}".format(train.shape))
print ("Test set shape: {}\n".format(test.shape))


# Slicing features and labels from train and test data
X_train = train[:,:64]
y_train = train[:,64]
X_test = test[:,:64]
y_test = test[:,64]



# plot class distribution bar chart.
tick_label = ['0','1','2','3','4','5','6','7','8','9']
plt.bar(range(10), np.histogram(y_train)[0], tick_label=tick_label)
plt.title('Class distribution in training set')
plt.bar(range(10), np.histogram(y_test)[0], tick_label=tick_label)
plt.title('Classes distribution')
plt.xlabel('classes')
plt.ylabel('Number of samples')
plt.legend(['Training set', 'Test set'])
plt.savefig('classdistribution.png')
print ('Class distribution bar chart is saved.\n')


# calculate the variance of features in train set
print('Removing features with zero variance.')
X_train_var = np.var(X_train, axis=0)


# boolean masking of variances > 0
zero_var_idx = np.where(X_train_var>0)[0]

# remove features with zero variance from train and test data
X_train = X_train[:,zero_var_idx]
X_test = X_test[:,zero_var_idx]
print ("train set shape after removing features with zero variance: {}".format(X_train.shape))
print ("test set shape after removing features with zero variance: {}\n".format(X_test.shape))       


# print number of class and features
num_class = len(set(y_train))
num_features = X_train.shape[1]
print ('There are {} classes.'.format(num_class))
print ('Each sample has {} features.\n'.format(num_features))


# calculate probability of classes
P = []
for i in range(10):
    P.append(np.sum(y_train==i)/y_train.shape[0])



# create an empty matrix for mean
M = np.zeros((num_class, num_features))

# create an empty matrix for covariance matrix
S_mat = np.zeros((num_class, num_features, num_features))

# create an empty matrix for common covariance
S_common = np.zeros((num_features, num_features))

# find the mean vector and covariace of classes
for i in range(num_class):
    
    # find the indices for class i (boolean masking)
    r = (y_train==i)
    
    # extract class i samples
    X_class = X_train[np.where(r)[0],:]
    
    # calculate mean vector of class i 
    M[i] = np.mean(X_class, axis=0)
    
    # calculate covariance matrix of class i
    S_mat[i] = ((X_class - M[i]).T.dot(X_class - M[i]))/np.sum(r)
    
    # calculate common covariance matrix
    S_common += P[i]*S_mat[i]





# calculate the diagonal of common covariance matrix 
S_diag = np.diag(S_common).reshape(num_features, 1)



# calculate the common variance
S = np.mean(np.diag(S_common))




# training method
def train(X, M, S, P):


	# create empty matrix to store discriminant functions values
	g = np.zeros((X.shape[0], num_class))


	# calculate discriminant functions
	for i in range(num_class):
		g[:,i] = -1/2 * np.sum(np.square(X - M[i])/S.T, axis=1) + np.log(P[i])
	return g


# prediction method
def predict(y_true, g): # add , title
	class_error = []
	y_pred = np.argmax(g, axis=1)
	acc = np.sum(y_pred == y_true)/len(y_true)
	error = 1 - acc
	print("Accuracy: {}".format(acc))
	acc_per_class =[]
	for i in range(num_class):

		r = (y_true==i)

		class_acc = (np.sum(y_true[np.where(r)] == y_pred[np.where(r)])/len(y_true[np.where(r)]))
		class_error.append(1 - class_acc)

		print("Accuracy per class {}: {:2f}".format(i, class_acc))

	return y_pred, error, class_error




# Visualizing Confusion Matrix taken from:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')






# train for common covariance diagonal (Shared, Axis-aligned)
g_train = train(X=X_train, M=M, S=S_diag, P=P)

"""
Use mean matrix, covariance matrix, probabilities of classes,
 learned from train set to calculate discriminant functions of test set
"""
g_test = train(X=X_test, M=M, S=S_diag, P=P)


print("Train set accuracy with diagonal common covariance (Shared, Axis-aligned):")
# predict and print the result of classifier for train set
y_pred, error, class_error = predict(y_true=y_train, g=g_train)


# write the training error to file
f.write("Training error of Q1a: {:4f} \n".format(error))



# Visualize and save confusion matrix for train set
conf_matrix_train_Q1a = confusion_matrix(y_train, y_pred)
plt.figure()
plot_confusion_matrix(conf_matrix_train_Q1a, classes=tick_label,
                      title="Confusion Matrix of train set (Shared, Axis-aligned)")
plt.savefig("conf_matrix_train_Shared_Axis_aligned.png")
#plt.show()
	






print("\nTest set accuracy with diagonal common covariance (Shared, Axis-aligned):")
# predict and print the result of classifier for test set
y_pred, error, class_error_q1a = predict(y_true=y_test, g=g_test)


# write the test error to file
f.write("Test error of Q1a: {:4f} \n".format(error))

# Visualize and save confusion matrix for test set
conf_matrix_test_Q1a = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(conf_matrix_test_Q1a, classes=tick_label,
                      title="Confusion Matrix of test set (Shared, Axis-aligned)")

plt.savefig("conf_matrix_test_Shared_Axis_aligned.png")
#plt.show()








# train for common common variance (Shared, Hyperspheric) Q1b
g_train = train(X=X_train, M=M, S=S, P=P)


"""
Use mean matrix, covariance matrix, probabilities of classes,
 learned from train set to calculate discriminant functions of test set
"""
g_test = train(X=X_test, M=M, S=S, P=P)



print("\nTrain set accuracy with common variance (Shared, Hyperspheric):")
# predict and print the result of classifier for train set
y_pred, error, class_error = predict(y_true=y_train, g=g_train)



# write the training error to file
f.write("\n\n")
f.write("Training error of Q1b: {:4f} \n".format(error))


# Visualize and save confusion matrix for train set
conf_matrix_train_Q1b = confusion_matrix(y_train, y_pred)
plt.figure()
plot_confusion_matrix(conf_matrix_train_Q1b, classes=tick_label,
                      title="Confusion Matrix of train set (Shared, Hyperspheric)")

plt.savefig("conf_matrix_train_shared_Hyperspheric.png")
#plt.show()




print("\nTest set accuracy with common variance (Shared, Hyperspheric):")
# predict and print the result of classifier for test set
y_pred, error, class_error_q1b = predict(y_true=y_test, g=g_test)


# write the test error to file
f.write("Test error of Q1b: {:4f} \n".format(error))
f.write("\n\n")



# Visualize and save confusion matrix for test set
conf_matrix_test_Q1b = confusion_matrix(y_test, y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(conf_matrix_test_Q1b, classes=tick_label,
                     title="Confusion Matrix of test set (Shared, Hyperspheric)")

plt.savefig("conf_matrix_test_shared_Hyperspheric.png")
#plt.show()

# writing Test error for each class for Q1a
f.write("Test error for each class for Q1a:\n")
for i in range(len(class_error_q1a)):
	f.write("Class {} error: {:4f} \n".format(i, class_error_q1a[i]))




# writing Test error for each class for Q1b
f.write('\n\n')
f.write("Test error for each class for Q1b:\n")
for i in range(len(class_error_q1b)):
	f.write("Class {} error: {:4f} \n".format(i, class_error_q1b[i]))


# write the confusion matrix of the Q1a
f.write('\n\n')
f.write("Confusion matrix of the Q1a:\n\n")
f.write(np.array2string(conf_matrix_test_Q1a, separator=', '))


# write the confusion matrix of the Q1a
f.write('\n\n\n')
f.write("Confusion matrix of the Q1b:\n\n")
f.write(np.array2string(conf_matrix_test_Q1b, separator=', '))





            



