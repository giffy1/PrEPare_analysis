# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 13:59:30 2017

PrEPare : Model Training
UMass Amherst & Swedish Medical

This script trains a classifier using ALL labelled pill-intake data 
and saves the classifier to disk, for use in server side predictions.

"""

import pickle
from sklearn.ensemble import RandomForestClassifier as RF
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import os
import constants
        
data_dir = constants.PROCESSING_DIR
output_dir = constants.OUTPUT_DIR

with open(os.path.join(data_dir, constants.ALL_SEGMENTS_FN), "rb") as f:
    X,y = pickle.load(f)
    
n_classes = 3 # can change to 2 classes, e.g. pill vs. other or drinking vs. other

random_state = np.random.RandomState(0)
C = 32.0
if n_classes > 2:
    classifier = OneVsRestClassifier(svm.LinearSVC(C=C, random_state=random_state), n_jobs=-1) # -1 jobs indicates maximum parallelization
else:
    classifier = svm.LinearSVC(C=C) # for efficiency, don't use a OneVsRest for 2 classes

classifier = classifier.fit(X,y)

with open(os.path.join(constants.OUTPUT_DIR, constants.CLASSIFIER_FN), 'wb') as f:
    pickle.dump(classifier, f)
