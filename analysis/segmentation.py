# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:30:01 2016

@author: snoran

PrEPare : Analysis 

Segments candidate gestures from the trajectory and extracts their feature 
vector and label.

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
import pandas as pd
from math import radians
from madgwick_py.madgwickahrs import MadgwickAHRS
from scipy import interpolate
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from peakdetect import peakdetect
from multiprocessing import Process, Queue
import constants

plot_dist = True
        
queue = Queue()

data_dir = constants.PROCESSING_DIR
    
# extract features from some random data just to check feature length
r = np.random.rand(50,4)
x = extract_features(r[:,:-1], r[:,:-1], r[:,0], r)
n_features = len(x)

classes = ['other', 'pill', 'drinking']

def segment(t, trajectory, rest_points):
    
    velocity = np.sqrt(np.sum(np.square(np.diff(trajectory[:,1:], axis=0)),axis=1)) # TODO: doesn't work if using T_smooth?

    # compute distance of each point from most recent point at which the patient took the pill out:
    dist = np.zeros((len(trajectory))) # distance-from-rest-point signal
    prev_t0 = 0
    for i,rest_point in enumerate(rest_points):
        t0 = np.where(t >= rest_point)[0][0] # first timestamp after taking pill out
        dist[prev_t0:t0] = prev_t0 # store at each point the most recent point
        prev_t0 = t0
    dist[prev_t0:] = prev_t0 
    dist = np.sqrt(np.sum(np.square(trajectory[:,1:]-trajectory[dist.astype(int),1:]), axis=1)) # compute distance of trajectory from most recent point
    
    # compute peaks in the distance-from-rest-point signal: The lookahead parameter describes how close peaks can be
    max_peaks, min_peaks = peakdetect(dist, lookahead=50)
    
    # append to min and max peaks a variable indicating what kind of critical point it is (0=min, 1=max), so we can combine into a single list:
    for p in min_peaks:
        p.append(0)
    for p in max_peaks:
        p.append(1)
    critical_points = min_peaks + max_peaks # combine into single list
    critical_points.sort(key=lambda x: x[0]) # order by timestamp
    
    segments = []
    
    #### Look for trough-peak-trough & peak-trough-peak pattern ####    
    
    is_prev_peak = False
    is_prev_prev_trough = False
    start = 0
    for i,p in enumerate(critical_points):
        # trough-peak-trough pattern
        is_trough = p[-1] == 0
        if is_trough and is_prev_peak and is_prev_prev_trough:
            # trough-peak-trough pattern
            end = p[0]
            segments.append((start,end))
        elif not is_trough and not is_prev_peak and not is_prev_prev_trough:
            # peak-trough-peak pattern
            end = p[0]
            segments.append((start,end))
        elif p[-1] == 1:
            start = p[0]
        if i > 0:
            is_prev_prev_trough = not is_prev_peak
        is_prev_peak = p[-1] == 1
        
    return dist, max_peaks, min_peaks, segments

def segmentation(queue, patient_id):
    """
    Segments the data stream for the given patient, adding the 
    results to the queue to allow for multi-processing.
    """
    subject_dir = os.path.join(data_dir, str(patient_id))
    print "Loading data from {}.".format(os.path.join(data_dir, str(patient_id)))
    with open(os.path.join(subject_dir, constants.WEARABLE_ACCELEROMETER_FN), "rb") as f:
        wearable_accel_i = np.loadtxt(f, delimiter=",")  
    
    with open(os.path.join(subject_dir, constants.WEARABLE_GYROSCOPE_FN), "rb") as f:
        wearable_gyro_i = np.loadtxt(f, delimiter=",")  
        
    with open(os.path.join(subject_dir, constants.LABELS_FN), "rb") as f:
        labels = np.loadtxt(f, delimiter=',', comments='#').astype(int)
        
    with open(os.path.join(subject_dir, constants.TRAJECTORY_FN), "rb") as f:
        trajectory = np.loadtxt(f, delimiter=',', comments='#')
        
    with open(os.path.join(subject_dir, constants.QUATERNION_FN), "rb") as f:
        quaternions = np.loadtxt(f, delimiter=',', comments='#')
    
    t = wearable_gyro_i[:,0]
    if len(labels.shape) == 1:
        labels = labels.reshape((1,-1))
        
    get_pill_indexes = np.where(labels[:,2] == 2)
    get_pill = labels[get_pill_indexes, :2][0] # labels for taking the pill out
        
    hand_to_mouth_indexes = np.where(labels[:,2] > 3)
    hand_to_mouth = labels[hand_to_mouth_indexes, :][0] # labels for hand-to-mouth gestures (pill intake or drinking water)

#    from pandas import ewma # smooth?
#    alpha=0.1
#    dist = ewma(dist, com=1/alpha-1)
    
    X = np.zeros((0,n_features))
    y = np.zeros((0,))
        
    dist, _, _, segments = segment(t, trajectory, get_pill[:,-1])
    
    for (start,end) in segments:
        x = extract_features(wearable_accel_i[start:end,1:], wearable_gyro_i[start:end,1:], dist[start:end], quaternions[start:end,1:])
                
        label = 0
        for (s,e,v) in hand_to_mouth:
            if t[int(start+(end-start)/2)] > s and t[int(start+(end-start)/2)] < e:
                label = v-3 # 1 for pill taking, 2 for drinking                
        
        X = np.append(X, np.reshape(x, (1,-1)), axis=0)
        y = np.append(y, label)

    
    print "Found classes {} for patient {}".format(set(y), patient_id)
    for y_i in set(y):
        print "Found {} instances of class {} ({}) for patient {}.".format(sum(y==y_i),y_i, classes[int(y_i)], patient_id)     
    
    print "Saving candidate segments for patient {}.".format(patient_id)
    with open(os.path.join(subject_dir, constants.SEGMENTS_FN), "wb") as f:
        pickle.dump([X, y], f)
    queue.put((X, y))
    
if __name__=='__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '-p':
            plot_dist=True
    
    subprocesses = []
    NP = 0
    for j in [1,2,3,4] + range(97,143): # participants 1-4 are Swedish folks, 97-112 are HIV patients
        p = Process(target=segmentation, args=(queue, j))
        NP += 1
        print 'delegated task to subprocess %s' % NP
        p.start()
        subprocesses.append(p)
     
    X = np.zeros((0,n_features))
    y = np.zeros((0,))
    # merge data after processes have finished:  
    for i in range(NP):
        subdata = queue.get()
        for (x,label) in zip(*subdata):
            X = np.append(X, np.reshape(x, (1,-1)), axis=0)
            y = np.append(y, label)
    
    while subprocesses:
        subprocesses.pop().join()
    
    print "Found classes {}".format(set(y))
    for y_i in set(y):
        print "Found {} instances of class {} ({}).".format(sum(y==y_i),y_i, classes[int(y_i)]) 
    
    # although we have already saved segments for each patient, save the combined result as well:
    with open(os.path.join(data_dir, constants.ALL_SEGMENTS_FN), "wb") as f:
        pickle.dump([X, y], f)