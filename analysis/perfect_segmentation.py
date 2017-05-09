# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:24:22 2017

@author: snoran
"""


import numpy as np
import os
from matplotlib import pyplot as plt
import sys
from features import extract_features
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import pickle
from util import reorient, reset_vars, farey, slidingWindow, running_avg
from visualize import visualize_data, plot_histogram, show_instances, visualize_signals
from load_data import load_data
from quaternions import normalize, q_conjugate, q_mult, qv_mult, q
from fusion import Fusion
import pandas as pd
from math import radians
from madgwick_py.madgwickahrs import MadgwickAHRS
from scipy import interpolate
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from peakdetect import peakdetect

ratio = 1
if len(sys.argv) > 1:
    ratio = int(sys.argv[1])

n_features=-1 #265#23
Xp = np.zeros((0,0))
Xn = np.zeros((0,0))

np.random.seed(0)

data_dir = '../corrected_data/'
for j in [1,2,3,4] + range(97,112):
    print "Loading " + os.path.join(data_dir, str(j))
    with open(os.path.join(data_dir, str(j), "accelerometer.csv"), "rb") as f:
        wearable_accel_i = np.loadtxt(f, delimiter=",")  
    
    with open(os.path.join(data_dir, str(j), "gyroscope.csv"), "rb") as f:
        wearable_gyro_i = np.loadtxt(f, delimiter=",")  
        
    with open(os.path.join(data_dir, str(j), "labels.txt"), "rb") as f:
        labels = np.loadtxt(f, delimiter=',', comments='#').astype(int)
        
    with open(os.path.join(data_dir, str(j), "trajectory.csv"), "rb") as f:
        T = np.loadtxt(f, delimiter=',', comments='#')
        
    with open(os.path.join(data_dir, str(j), "quaternions.csv"), "rb") as f:
        Q = np.loadtxt(f, delimiter=',', comments='#')
    
    t = wearable_gyro_i[:,0]
    if len(labels.shape) == 1:
        labels = labels.reshape((1,-1))
    label_col = np.zeros((len(wearable_gyro_i),1))
    for i,label in enumerate(labels):
        indexes = (t >= label[0]) & (t <= label[1])
        if i > 0:
            no_label_indexes = (t >= labels[i-1][1]) & (t <= label[0])
            label_col[no_label_indexes] = 0
        elif i == 0:
            no_label_indexes = (t <= label[0])
            label_col[no_label_indexes] = 0
        if i == len(labels) - 1:
            no_label_indexes = (t >= label[1])
            label_col[no_label_indexes] = 0
        label_col[indexes] = int(label[2])
        
    get_pill_indexes = np.where(labels[:,2] == 2)
    get_pill = labels[get_pill_indexes, :2][0]
        
    hand_to_mouth_indexes = np.where(labels[:,2] > 3)
    hand_to_mouth = labels[hand_to_mouth_indexes, :2][0]
        
    dist = np.zeros((len(T)))
    prev_t0 = 0
    for i,(s,e) in enumerate(get_pill):
        t0 = np.where(t >= e)[0][0] # first timestamp after getting pill out
        dist[prev_t0:t0] = prev_t0
        prev_t0 = t0
    
    dist[prev_t0:] = prev_t0
    dist = np.sqrt(np.sum(np.square(T[:,1:]-T[dist.astype(int),1:]), axis=1))
    
    # extract features for each perfectly segmented hand-to-mouth gesture
    for i,(s,e) in enumerate(hand_to_mouth):
        indexes = (t >= s) & (t <= e)
        x = extract_features(wearable_accel_i[indexes,1:], wearable_gyro_i[indexes,1:], dist[indexes], Q[indexes,1:])
        if n_features == -1:
            n_features = len(x)
            Xp = np.zeros((0,n_features))
            Xn = np.zeros((0,n_features)) 
            
        Xp = np.append(Xp, np.reshape(x, (1,-1)), axis=0)
    
    window_size=50
    for start in range(0, len(wearable_accel_i)-20*window_size, window_size):
        skip=False
        window_size = 0
        while window_size <= 25: # min size
            window_size = int(np.random.normal(175, 50))
        for (s,e) in hand_to_mouth:
            if ((t[start] > s and t[start] < e) or (t[start+window_size] > s and t[start+window_size] < e) or (t[start] < s and t[start+window_size] > e) or (t[start] > s and t[start+window_size] < e)):
                skip = True
        if not skip:
            x = extract_features(wearable_accel_i[start:start+window_size,1:],wearable_gyro_i[start:start+window_size,1:], dist[start:start+window_size], Q[start:start+window_size,1:])
            Xn = np.append(Xn, np.reshape(x, (1,-1)), axis=0)
        window_size = 50

Np = len(Xp)
Nn = ratio * Np
np.random.shuffle(Xn)
X = np.vstack((Xp, Xn[:Nn]))
y = np.asarray([1] * Np + [0] * Nn)
print X.shape, y.shape
print set(y)
print sum(y)

with open("dataset.pickle", "wb") as f:
    pickle.dump([X, y], f)