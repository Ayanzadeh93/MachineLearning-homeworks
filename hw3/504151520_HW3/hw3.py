# import the required libraries
import os
import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import tree
from sklearn.neural_network import MLPClassifier

# set the random state
np.random.seed(seed=300)

# load train and test data
train = np.loadtxt(open(os.path.join('input', 'optdigits.tra'), "rb"), delimiter=",")
test =np.loadtxt(open(os.path.join('input', 'optdigits.tes'), "rb"), delimiter=",")

# Define labels for confusion matrix figure
tick_label = ['0','1','2','3','4','5','6','7','8','9']

# Slicing features and labels from train and test data
X_train = train[:,:64]
y_train = train[:,64]
X_test = test[:,:64]
y_test = test[:,64]


# spliting train set into 90% train and 10% validation set
Xtr, Xval, ytr, yval = train_test_split(X_train, y_train, test_size=0.10)



# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# define a function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
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
        print(title)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title)




# compute accuracy per class
def acc_per_class(conf_train, conf_test, flag=False):
	if flag:
		print('Class    train accuracy     test accuracy')
	else:
		print('Class    Before Removal     After Removal')
	for i in range(10):
		train_acc = float(conf_train[i,i])/np.sum(conf_train[i,:])
		test_acc = float(conf_test[i,i])/np.sum(conf_test[i,:])
		print(' {}       {:.4f}           {:.4f}'.format(i, train_acc, test_acc))      


##################################################
#												 #
# tuning the hyper-parameter k of KNN Classifier #
#												 #
##################################################	 

# k values (1, 3, 5, 7, ... , 19)
k_array = np.arange(1,20,2)

# an empty list to store validation accuracies
val_scores_knn = []

# compute knn classifier accuracy for each k
print("\n"+50*"#")
print('Hyper-parameters tunning for knn...')
print(50*"#")
best_knn = None
best_acc_knn = -1
for k in k_array:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr, ytr)
    val_acc_knn = knn.score(Xval, yval) # accuracy for validation set
    val_scores_knn.append(val_acc_knn)
    if val_acc_knn > best_acc_knn:
    	best_knn = knn
    	best_acc_knn = val_acc_knn

# choose the optimal k
best_k = k_array[val_scores_knn.index(max(val_scores_knn))]


print ("Best n_neighbors: {}\n".format(best_k))

# Best model accuracy on validation set
print("Validation accuracy (KNN): {:.4f}".format(best_knn.score(Xval, yval)))



# compute knn train time
print('\nComputing knn training time...')
start = time.time()
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
print("n_neighbors: {}, training took {:.4f} seconds.\n".format(best_k, time.time() - start))

y_pred_train_knn = knn.predict(X_train)
conf_knn_train = confusion_matrix(y_train, y_pred_train_knn)
plot_confusion_matrix(conf_knn_train, classes=tick_label, title="Confusion Matrix of KNN (Train)")



# compute knn test time
start = time.time()
print('\nComputing knn test time...')
y_pred_test_knn = knn.predict(X_test)
print("n_neighbors: {} test took {:.4f} seconds.\n".format(best_k, time.time() - start))
print("Test accuracy (KNN): {:.4f}\n".format(knn.score(X_test, y_test)))

# plot test set confusion matrix
conf_knn_test = confusion_matrix(y_test, y_pred_test_knn)
plot_confusion_matrix(conf_knn_test, classes=tick_label, title="Confusion Matrix of KNN (Test)")



##################################################
#												 #
# 	Tuning the hyper-parameter: max depth of 	 #
#				Decision Tree					 #
#												 #
##################################################	 

# set an aray of max depth values
max_depth_array = np.arange(1,51)

# an empty list to store validation accuracies
val_scores_dtree = []

# compute dtree classifier accuracy for each max depth
print("\n\n"+50*"#")
print('Hyper-parameters tunning for Decision Tree...')
print(50*"#")
best_dtree = None
best_acc_dtree = -1
for d in max_depth_array:
    dtree = tree.DecisionTreeClassifier(max_depth=d)
    dtree.fit(Xtr, ytr)
    val_acc_dtree = dtree.score(Xval, yval) # accuracy for validation set
    val_scores_dtree.append(val_acc_dtree)
    if val_acc_dtree > best_acc_dtree:
    	best_dtree = dtree
    	best_acc_dtree = val_acc_dtree

# choose the best max depth
best_depth = max_depth_array[val_scores_dtree.index(max(val_scores_dtree))]


print ("Best max_depth: {}\n".format(best_depth))


# Classification accuracy on validation set
print("Validation accuracy (Decision Tree): {:.4f}\n".format(best_dtree.score(Xval, yval)))


# compute Deision tree train time
print('\nComputing Decision Tree training time...')
start = time.time()
dtree = tree.DecisionTreeClassifier(max_depth=best_depth)
dtree.fit(X_train, y_train)
print("max_depth: {}, training took {:.4f} seconds.\n".format(best_depth, time.time() - start))


# train set confusion matrix
y_pred_train_dtree = dtree.predict(X_train)
conf_dtree_train = confusion_matrix(y_train, y_pred_train_dtree)
plot_confusion_matrix(conf_dtree_train, classes=tick_label, title="Confusion Matrix of Decision Tree (Train)")


# compute Deision tree test time
start = time.time()
print('\nComputing Decision Tree test time...')
y_pred_test_dtree = dtree.predict(X_test)
print("max_depth: {}, test took {:.4f} seconds.\n".format(best_depth, time.time() - start))
print("Test accuracy (Decision Tree): {:.4f}\n".format(dtree.score(X_test, y_test)))

# plot test set confusion matrix
conf_dtree_test = confusion_matrix(y_test, y_pred_test_dtree)
plot_confusion_matrix(conf_dtree_test, classes=tick_label, title="Confusion Matrix of Decision Tree (Test)")


##################################################
#												 #
# 			Tuning the hyper-parameter for 		 #
#				linear discrimination:  		 #
#				regularization penalty			 #
#												 #
##################################################	 

# set regularization penalty
regs = [0.0001, 0.001, 0.01, 0.1, 1, 10]

# an empty list to store validation accuracies
val_scores_sgd = []

# compute linear classifier accuracy for each regularization strength
print('\n'+50*'#')
print('Hyper-parameters tunning for linear classifier...')
print(50*'#')
best_sgd = None
best_acc_sgd = -1
for reg in regs:
    sgd = linear_model.SGDClassifier(alpha=reg)
    sgd.fit(Xtr, ytr)
    val_acc_sgd = sgd.score(Xval, yval) # accuracy for validation set
    val_scores_sgd.append(val_acc_sgd)
    if val_acc_sgd > best_acc_sgd:
    	best_sgd = sgd
    	best_acc_sgd = val_acc_sgd

# choose the best regularization penalty
best_reg = regs[val_scores_sgd.index(max(val_scores_sgd))]
print ("Best alpha: {}\n".format(best_reg))


# Validation accuracy
y_pred_val_sgd = best_sgd.predict(Xval)
print("Validation accuracy (Linear classifier): {:.4f}".format(best_sgd.score(Xval, yval)))


# compute linear classifier train time
print('\nComputing linear classifier training time...')
start = time.time()
sgd = linear_model.SGDClassifier(alpha=best_reg)
sgd.fit(X_train, y_train)
print("alpha: {}, training took {:.4f} seconds.\n".format(best_reg, time.time() - start))

# train set confusion matrix
y_pred_train_sgd = sgd.predict(X_train)
conf_sgd_train = confusion_matrix(y_train, y_pred_train_sgd)
plot_confusion_matrix(conf_sgd_train, classes=tick_label,title="Confusion Matrix of linear classifier (Train)")


# compute linear classifier test time
start = time.time()
print('\nComputing linear classifier test time...')
y_pred_test_sgd = sgd.predict(X_test)
print("alpha: {}, test took {:.4f} seconds.\n".format(best_reg, time.time() - start))
print("Test accuracy (linear classifier): {:.4f}\n".format(sgd.score(X_test, y_test)))

# plot test set confusion matrix
conf_sgd_test = confusion_matrix(y_test, y_pred_test_sgd)
plot_confusion_matrix(conf_sgd_test, classes=tick_label,title="Confusion Matrix of linear classifier (Test)")



##################################################
#												 #
# 			Tuning the hyper-parameter:			 #
#				Multilayer perceptron			 #
#												 #
##################################################	 


# compute mlp classifier accuracy for each set of parameters
print('\n'+50*'#')
print('Hyper-parameters tunning for MLP...')
print(50*'#')

best_mlp = None
best_acc_mlp = -1
best_hls = []
best_reg = []
hls = [64, 128, 256, (64,64), (128,128), (256,256)]
regs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
for hl in hls:
    for reg in regs:
    	mlp = MLPClassifier(solver='adam', alpha=reg, hidden_layer_sizes=hl)
    	mlp.fit(Xtr, ytr)
    	val_acc_mlp = mlp.score(Xval, yval) # accuracy for validation set
    	val_scores_sgd.append(val_acc_sgd)
    	if val_acc_mlp > best_acc_mlp:
    		best_mlp = mlp
    		best_acc_mlp = val_acc_mlp
    		best_hls = hl
    		best_reg = reg

# choose the best number of components
#best_n_components = n_components_array[val_scores_lda.index(max(val_scores_lda))]

# print best hyper-parameters
print ("Best hidden_layer_sizes: {}, Best alpha: {}\n".format(best_hls, best_reg))


# Validation accuracy
print("Validation accuracy (MLP): {:.4f}".format(best_mlp.score(Xval, yval)))


# compute MLP train time
print('\nComputing MLP training time...')
start = time.time()
mlp = MLPClassifier(solver='adam', alpha=best_reg, hidden_layer_sizes=best_hls)
mlp.fit(X_train, y_train)
print("hidden_layer_sizes: {}, alpha: {}, training took {:.4f} seconds.\n".format(best_hls, best_reg,\
														 time.time() - start))


# plot train set confusion matrix
y_pred_train_mlp = mlp.predict(X_train)
conf_mlp_train = confusion_matrix(y_train, y_pred_train_mlp)
plot_confusion_matrix(conf_mlp_train, classes=tick_label,title="Confusion Matrix of MLP (Train)")


# compute MLP test time
start = time.time()
print('\nComputing MLP test time...')
y_pred_test_mlp = mlp.predict(X_test)
print("hidden_layer_sizes: {}, alpha: {}, test took {:.4f} seconds.".format(best_hls, best_reg,\
														 time.time() - start))
# print test accuracy MLP
print("Test accuracy (MLP): {:.4f}\n".format(mlp.score(X_test, y_test)))

# plot test set confusion matrix
conf_mlp_test = confusion_matrix(y_test, y_pred_test_mlp)
plot_confusion_matrix(conf_mlp_test, classes=tick_label,title="Confusion Matrix of MLP (Test)")




# compute accuracy per class KNN 
print('\n\n'+40*'#')
print('Accuracy per class (KNN)')
acc_per_class(conf_knn_train, conf_knn_test, flag=True)


# compute accuracy per class Decision Tree
print('\n\n'+40*'#')
print('Accuracy per class (Decision Tree)')
acc_per_class(conf_dtree_train, conf_dtree_test, flag=True)

# compute accuracy per class Linear Classifier
print('\n\n'+40*'#')
print('Accuracy per class (Linear Classifier)')
acc_per_class(conf_sgd_train, conf_sgd_test, flag=True)

# compute accuracy per class MLP
print('\n\n'+40*'#')
print('Accuracy per class (MLP)')
acc_per_class(conf_mlp_train, conf_mlp_test, flag=True)


##################################################
#												 #
#				Removing noisy instances 		 #
#												 #
##################################################

# Find noisy instances index per model
idx_knn_noisy = [(y_train != y_pred_train_knn)]
idx_dtree_noisy = [(y_train != y_pred_train_dtree)]
idx_linear_noisy = [(y_train != y_pred_train_sgd)]
idx_mlp_noisy = [(y_train != y_pred_train_mlp)]

print('\n\n\n'+50*'#')
print('Number of noisy instances per model.')
print('knn: {}'.format(np.sum(idx_knn_noisy)))
print('decision tree: {}'.format(np.sum(idx_dtree_noisy)))
print('linear classifier: {}'.format(np.sum(idx_linear_noisy)))
print('mlp: {}'.format(np.sum(idx_mlp_noisy)))


# Identify index of all noisy instances
idx_noisy = [(y_train != y_pred_train_sgd) | (y_train != y_pred_train_dtree) |\
             (y_train != y_pred_train_knn) | (y_train != y_pred_train_mlp)]

print('\nTotal number of missclassified instaces: {}'.format(np.sum(idx_noisy)))

# Find the percentage of train data that are misclassified
noisy_percent = np.sum(idx_noisy)/float(y_train.shape[0])*100
print('{:.2f} percent of train data are missclasified.'.format(noisy_percent))

# Eliminate all noisy instances
idx_correct = (idx_noisy[0] == False)
X_train_new = X_train[idx_correct]
y_train_new = y_train[idx_correct]


# spliting new train set into 90% train and 10% validation set
Xtr, Xval, ytr, yval = train_test_split(X_train_new, y_train_new, test_size=0.10)



##################################################
#												 #
# tuning the hyper-parameter k of KNN Classifier #
#                After Removal                   # 
#												 #
##################################################	 

# k values (1, 3, 5, 7, ... , 19)
k_array = np.arange(1,20,2)

# an empty list to store validation accuracies
val_scores_knn = []

# compute knn classifier accuracy for each k
print("\n"+50*"#")
print('Hyper-parameters tunning for knn after removal...')
print(50*"#")
best_knn = None
best_acc_knn = -1
for k in k_array:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xtr, ytr)
    val_acc_knn = knn.score(Xval, yval) # accuracy for validation set
    val_scores_knn.append(val_acc_knn)
    if val_acc_knn > best_acc_knn:
    	best_knn = knn
    	best_acc_knn = val_acc_knn

# choose the optimal k
best_k = k_array[val_scores_knn.index(max(val_scores_knn))]


print ("Best n_neighbors after removal: {}\n".format(best_k))

# Best model accuracy on validation set
print("Validation accuracy after removal (KNN): {:.4f}".format(best_knn.score(Xval, yval)))



# compute knn train time
print('\nComputing knn training time after removal ...')
start = time.time()
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_new, y_train_new)
print("n_neighbors: {}, training took {:.4f} seconds.\n".format(best_k, time.time() - start))

y_pred_train_knn = knn.predict(X_train_new)
conf_knn_train_re = confusion_matrix(y_train_new, y_pred_train_knn)
plot_confusion_matrix(conf_knn_train_re, classes=tick_label, title="Confusion Matrix of KNN after removal (Train)")



# compute knn test time
start = time.time()
print('\nComputing knn test time after removal ...')
y_pred_test_knn = knn.predict(X_test)
print("n_neighbors: {} test took {:.4f} seconds.\n".format(best_k, time.time() - start))
print("Test accuracy after removal (KNN): {:.4f}\n".format(knn.score(X_test, y_test)))

# plot test set confusion matrix
conf_knn_test_re = confusion_matrix(y_test, y_pred_test_knn)
plot_confusion_matrix(conf_knn_test_re, classes=tick_label, title="Confusion Matrix of KNN after removal (Test)")





##################################################
#												 #
# 	Tuning the hyper-parameter: max depth of 	 #
#				Decision Tree After removal		 #
#												 #
##################################################	 

# set an aray of max depth values
max_depth_array = np.arange(1,51)

# an empty list to store validation accuracies
val_scores_dtree = []

# compute dtree classifier accuracy for each max depth
print("\n\n"+50*"#")
print('Hyper-parameters tunning for Decision Tree after removal...')
print(50*"#")
best_dtree = None
best_acc_dtree = -1
for d in max_depth_array:
    dtree = tree.DecisionTreeClassifier(max_depth=d)
    dtree.fit(Xtr, ytr)
    val_acc_dtree = dtree.score(Xval, yval) # accuracy for validation set
    val_scores_dtree.append(val_acc_dtree)
    if val_acc_dtree > best_acc_dtree:
    	best_dtree = dtree
    	best_acc_dtree = val_acc_dtree

# choose the best max depth
best_depth = max_depth_array[val_scores_dtree.index(max(val_scores_dtree))]


print ("Best max_depth after removal: {}\n".format(best_depth))


# Classification accuracy on validation set
print("Validation accuracy (Decision Tree) after removal: {:.4f}\n".format(best_dtree.score(Xval, yval)))


# compute Deision tree train time
print('\nComputing Decision Tree training time after removal ...')
start = time.time()
dtree = tree.DecisionTreeClassifier(max_depth=best_depth)
dtree.fit(X_train_new, y_train_new)
print("max_depth: {}, training took {:.4f} seconds.\n".format(best_depth, time.time() - start))


# train set confusion matrix
y_pred_train_dtree = dtree.predict(X_train_new)
conf_dtree_train_re = confusion_matrix(y_train_new, y_pred_train_dtree)
plot_confusion_matrix(conf_dtree_train_re, classes=tick_label, title="Confusion Matrix of Decision Tree after removal (Train)")


# compute Deision tree test time
start = time.time()
print('\nComputing Decision Tree test time after removal ...')
y_pred_test_dtree = dtree.predict(X_test)
print("max_depth: {}, test took {:.4f} seconds.\n".format(best_depth, time.time() - start))
print("Test accuracy (Decision Tree) after removal: {:.4f}\n".format(dtree.score(X_test, y_test)))

# plot test set confusion matrix
conf_dtree_test_re = confusion_matrix(y_test, y_pred_test_dtree)
plot_confusion_matrix(conf_dtree_test_re, classes=tick_label, title="Confusion Matrix of Decision Tree after removal (Test)")



##################################################
#												 #
# 			Tuning the hyper-parameter for 		 #
#	     linear discrimination after removal:  	 #
#				regularization penalty			 #
#												 #
##################################################	 

# set regularization penalty
regs = [0.0001, 0.001, 0.01, 0.1, 1, 10]

# an empty list to store validation accuracies
val_scores_sgd = []

# compute linear classifier accuracy for each regularization strength
print('\n'+50*'#')
print('Hyper-parameters tunning for linear classifier after removal...')
print(50*'#')
best_sgd = None
best_acc_sgd = -1
for reg in regs:
    sgd = linear_model.SGDClassifier(alpha=reg)
    sgd.fit(Xtr, ytr)
    val_acc_sgd = sgd.score(Xval, yval) # accuracy for validation set
    val_scores_sgd.append(val_acc_sgd)
    if val_acc_sgd > best_acc_sgd:
    	best_sgd = sgd
    	best_acc_sgd = val_acc_sgd

# choose the best regularization penalty
best_reg = regs[val_scores_sgd.index(max(val_scores_sgd))]
print ("Best alpha: {}\n".format(best_reg))


# Validation accuracy
y_pred_val_sgd = best_sgd.predict(Xval)
print("Validation accuracy (Linear classifier) after removal: {:.4f}".format(best_sgd.score(Xval, yval)))


# compute linear classifier train time
print('\nComputing linear classifier training time after removal...')
start = time.time()
sgd = linear_model.SGDClassifier(alpha=best_reg)
sgd.fit(X_train_new, y_train_new)
print("alpha: {}, training took {:.4f} seconds.\n".format(best_reg, time.time() - start))

# train set confusion matrix
y_pred_train_sgd = sgd.predict(X_train_new)
conf_sgd_train_re = confusion_matrix(y_train_new, y_pred_train_sgd)
plot_confusion_matrix(conf_sgd_train_re, classes=tick_label,title="Confusion Matrix of linear classifier after removal (Train)")


# compute linear classifier test time
start = time.time()
print('\nComputing linear classifier test time after removal...')
y_pred_test_sgd = sgd.predict(X_test)
print("alpha: {}, test took {:.4f} seconds.\n".format(best_reg, time.time() - start))
print("Test accuracy after removal (linear classifier): {:.4f}\n".format(sgd.score(X_test, y_test)))

# plot test set confusion matrix
conf_sgd_test_re = confusion_matrix(y_test, y_pred_test_sgd)
plot_confusion_matrix(conf_sgd_test_re, classes=tick_label,title="Confusion Matrix of linear classifier after removal (Test)")




##################################################
#												 #
# 	Tuning the hyper-parameter after removal:	 #
#				Multilayer perceptron			 #
#												 #
##################################################	 


# compute mlp classifier accuracy for each set of parameters
print('\n'+50*'#')
print('Hyper-parameters tunning for MLP after removal...')
print(50*'#')
best_mlp = None
best_acc_mlp = -1
best_hls = []
best_reg = []
hls = [64, 128, 256, (64,64), (128,128), (256,256)]
regs = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
for hl in hls:
    for reg in regs:
    	mlp = MLPClassifier(solver='adam', alpha=reg, hidden_layer_sizes=hl)
    	mlp.fit(Xtr, ytr)
    	val_acc_mlp = mlp.score(Xval, yval) # accuracy for validation set
    	val_scores_sgd.append(val_acc_sgd)
    	if val_acc_mlp > best_acc_mlp:
    		best_mlp = mlp
    		best_acc_mlp = val_acc_mlp
    		best_hls = hl
    		best_reg = reg

# choose the best number of components
#best_n_components = n_components_array[val_scores_lda.index(max(val_scores_lda))]

# print best hyper-parameters
print ("Best hidden_layer_sizes: {}, Best alpha: {}\n".format(best_hls, best_reg))


# Validation accuracy
print("Validation accuracy after removal (MLP): {:.4f}".format(best_mlp.score(Xval, yval)))


# compute MLP train time
print('\nComputing MLP training time after removal ...')
start = time.time()
mlp = MLPClassifier(solver='adam', alpha=best_reg, hidden_layer_sizes=best_hls)
mlp.fit(X_train_new, y_train_new)
print("hidden_layer_sizes: {}, alpha: {}, training took {:.4f} seconds.\n".format(best_hls, best_reg,\
														 time.time() - start))


# plot train set confusion matrix
y_pred_train_mlp = mlp.predict(X_train_new)
conf_mlp_train_re = confusion_matrix(y_train_new, y_pred_train_mlp)
plot_confusion_matrix(conf_mlp_train_re, classes=tick_label,title="Confusion Matrix of MLP after removal (Train)")


# compute MLP test time
start = time.time()
print('\nComputing MLP test time after removal ...')
y_pred_test_mlp = mlp.predict(X_test)
print("hidden_layer_sizes: {}, alpha: {}, test took {:.4f} seconds.".format(best_hls, best_reg,\
														 time.time() - start))
# print test accuracy MLP
print("Test accuracy (MLP): {:.4f}\n".format(mlp.score(X_test, y_test)))

# plot test set confusion matrix
conf_mlp_test_re = confusion_matrix(y_test, y_pred_test_mlp)
plot_confusion_matrix(conf_mlp_test_re, classes=tick_label,title="Confusion Matrix of MLP after removal (Test)")





# compute accuracy per class before and after removal for test set KNN 
print('\n'+40*'#')
print('Accuracy per class (KNN)')
acc_per_class(conf_knn_test, conf_knn_test_re)


# compute accuracy per class before and after removal for test set  Decision Tree
print('\n'+40*'#')
print('Accuracy per class (Decision Tree)')
acc_per_class(conf_dtree_test, conf_dtree_test_re)

# compute accuracy per class before and after removal for test set  Linear Classifier
print('\n'+40*'#')
print('Accuracy per class (Linear Classifier)')
acc_per_class(conf_sgd_test, conf_sgd_test_re)

# compute accuracy per class before and after removal for test set  MLP
print('\n'+40*'#')
print('Accuracy per class (MLP)')
acc_per_class(conf_mlp_test, conf_mlp_test_re)

