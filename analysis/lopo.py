# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:59:30 2017

PrEPare : Leave-One-Participant-Out Evaluation
UMass Amherst & Swedish Medical

This script trains and evaluates a classifier for identifying 
pill intake and other hand-to-mouth gestures (e.g. drinking) in 
a leave-one-participant-out fashion.

More precisely, given (X,y) data from K participants, we perform 
a K-fold cross-validation, where for each patient, the training 
data corresponds to the data from all other patients and the 
test data corresponds to that patient's data.

Leave-one-participant-out (LOPO) evaluation provides a measure 
that describes how well the classifier will generalize on 
never-before-seen patients. This is the most fitting evaluation 
technique, because we want the classifier to work for new 
patients.

In general, expect to see a drop in performance with respect to 
the usual cross-validation approach, where the K folds are over 
data points and not over subjects.

                            ROC Curve

Performance is reported in the form of receiver operating characteristic 
curves (ROC), which generalizes well to multiple classes. The ROC curve
is the true positive rate plotted against the false positive rate, where 
the upper-left point (0,1) corresponds to the optimal performance, 
that is a false positive rate of 0 and a true positive rate of 1.
Thus the area under the curve (AUC or AUROC) can be used to describe 
the performance concisely in a single value.

The ROC curve is specified for each class individually and the 
least optimal curve (the line y=x) is also plotted for reference. 
To get a single curve, the curves can be averaged in one of two ways:

    (1) Macro-average : The true positive rates and the false 
        positive are separately averaged across classes. This is 
        more informative if the classes are not balanced.
    
    (2) Micro-average : The true positive and false positive 
        rates are recomputed by counting the true positives 
        and false positives for each class, and dividing by 
        the total number of instances. In the presence of class 
        imbalance, this will give an artificially high value, 
        similar to how accuracy would be uninformative.
        
The accuracy, precision, recall and F1 score are also reported for 
completeness.

See http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

"""

import pickle
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from itertools import cycle
from scipy import interp
import os
from multiprocessing import Process, Queue
import constants

plot_roc = False
if len(sys.argv) > 1:
    if sys.argv[1] == '-p':
        plot_roc = True
        
def binarize(y):
    # for two-class case only
    y = np.vstack((1-y, y))
    y = np.transpose(y)
    return y
        
queue = Queue()
        
data_dir = constants.PROCESSING_DIR

fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = 3 # can change to 2 classes, e.g. pill vs. other or drinking vs. other
random_state = np.random.RandomState(0)
C = 32.0
if n_classes > 2:
    classifier = OneVsRestClassifier(svm.LinearSVC(C=C, random_state=random_state), n_jobs=-1)
else:
    classifier = svm.LinearSVC(C=C) # for efficiency, don't use a OneVsRest for 2 classes

def evaluate_leave_one_out(queue, patient_to_omit):
    print "Evaluating classifier omitting patient {}".format(patient_to_omit)
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    for j in [1,2,3,4] + range(97,143):
        subject_dir = os.path.join(data_dir, str(j))
        with open(os.path.join(subject_dir, constants.SEGMENTS_FN), "rb") as f:
            X_i,y_i = pickle.load(f)
            if j == patient_to_omit:
                X_test = X_i
                y_test = y_i
            else:
                if X_train == None:
                    X_train = X_i
                    y_train = y_i
                else:
                    X_train = np.append(X_train, X_i, axis=0)
                    y_train = np.append(y_train, y_i, axis=0)
    if n_classes > 2:
        # must binarize the labels for multi-class, so that each row is a sequence 
        # of 0s with a single 1 in the index corresponding to the class
        y_train = label_binarize(y_train, classes=[0, 1, 2])
        y_test = label_binarize(y_test, classes=[0, 1, 2])
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        # for the binary case, we can't give binarized input to the classifier, so 
        # do it after training the classifier. This is necessary for computing the receiver-operator curve
        y_test = binarize(y_test)
        y_score = binarize(y_score)

# grid search was used to find the best value of C for the data. For efficiency, I omit this 
# and assume C=32.0 which was the best value found in all cases so far.
#classifier = OneVsRestClassifier(BaggingClassifier(svm.SVC(kernel='linear', C=32.0, probability=True, random_state=random_state), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
#C_array = np.logspace(5, 16, 12, base=2).tolist()
#cv = StratifiedKFold(y_train, n_folds=3)
#gs = GridSearchCV(classifier, {'C' : C_array}, cv=cv, scoring="f1_weighted", n_jobs=6)
#gs = gs.fit(X_train, y_train)
#print gs.best_params_
#classifier = svm.SVC(kernel='linear', random_state=random_state, **gs.best_params_)

    if n_classes > 2:
        binarized_labels = label_binarize(np.argmax(y_score, axis=1), classes=range(n_classes))
    else:
        binarized_labels = binarize(np.argmax(y_score, axis=1))
    print "precision, recall, fscore omitting patient {} : ".format(patient_to_omit), precision_recall_fscore_support(y_test, binarized_labels)
    
    queue.put((y_test, y_score))

Y_test = np.zeros((0,n_classes))
Y_score = np.zeros_like(Y_test)

subprocesses = []
NP = 0
for j in [1,2,3,4] + range(97,143):
    p = Process(target=evaluate_leave_one_out, args=(queue, j))
    NP += 1
    print 'delegated task to subprocess %s' % NP
    p.start()
    subprocesses.append(p)

# aggregate predictions and ground-truth over each left-out participant
for i in range(NP):
    print "Process {} complete.".format(i)
    (y_test, y_score) = queue.get()
    Y_test = np.append(Y_test, y_test, axis=0)
    Y_score = np.append(Y_score, y_score, axis=0)

while subprocesses:
    subprocesses.pop().join()
            
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:,i], Y_score[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

if plot_roc:
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.show()
    
    
# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    lw=2
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()

#classifier = classifier.fit(X,y)
#
#with open('classifier.pickle', 'wb') as f:
#    pickle.dump(classifier, f)
