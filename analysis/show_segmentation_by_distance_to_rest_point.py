# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:47:29 2017

PrEPare : Visualizing Segments and Peak Detection
UMass Amherst & Swedish Medical

This script plots the segmentation results for visualization for a 
particular subject. It specifically plots the distance from the 
most recent rest point signal, where rest points are defined as the 
point when the subject takes the pill out of the bottle/mediset.

It also plots the peaks and troughs detected and at y=0.1 it plots 
the ground-truth labels for hand-t-mouth gestures (red for pill intake 
gestures, green for drinking). All peak-trouth-peak and trough-peak-trough 
windows are considered to be candidate segments, so the critical points
can be used to compare the ground-truth with the identified segments.

Run with the command
        
        show_segmentation_by_distance_to_rest_point.py -p=###
        
where ### corresponds to the patient ID. If omitted, the default is 
subject 1.

"""

from segmentation import segment
import os
import constants
import numpy as np
import matplotlib.pyplot as plt
import sys

data_dir = constants.PROCESSING_DIR

def show_segmentation(patient_id):
    """
    Visualizes the segmentation given by the peak-trough-peak 
    and trough-peak-trough patterns found in the distance 
    from most recent rest point signal.
    """
    subject_dir = os.path.join(data_dir, str(patient_id))
    print "Loading data from {}.".format(os.path.join(data_dir, str(patient_id)))
        
    with open(os.path.join(subject_dir, constants.LABELS_FN), "rb") as f:
        labels = np.loadtxt(f, delimiter=',', comments='#').astype(int)
        
    with open(os.path.join(subject_dir, constants.TRAJECTORY_FN), "rb") as f:
        T = np.loadtxt(f, delimiter=',', comments='#')
    
    t = T[:,0]
    if len(labels.shape) == 1:
        labels = labels.reshape((1,-1))
        
    get_pill_indexes = np.where(labels[:,2] == 2)
    get_pill = labels[get_pill_indexes, :2][0] # labels for taking the pill out
        
    hand_to_mouth_indexes = np.where(labels[:,2] > 3)
    hand_to_mouth = labels[hand_to_mouth_indexes, :][0] # labels for hand-to-mouth gestures (pill intake or drinking water)
    K = len(hand_to_mouth)

#    from pandas import ewma # smooth?
#    alpha=0.1
#    dist = ewma(dist, com=1/alpha-1)
        
    dist, max_peaks, min_peaks, segments = segment(t, T, get_pill[:,-1])
    
    pmin = zip(*max_peaks)
    pmax = zip(*min_peaks)
    
    colors = {4 : 'r-', 5 : 'g-'} # different color for pill-intake (=4) and water drinking gestures (=5)

    plt.figure()
    print t.shape
    print dist.shape
    plt.plot(t[:len(dist)], dist)
    
    for k in range(K):
        s = hand_to_mouth[k,0]
        e = hand_to_mouth[k,1]
        plt.plot([s, e], [0.1, 0.1], colors[hand_to_mouth[k,-1]])
        
    plt.plot(t[list(pmin[0])], pmin[1], 'g*')
    plt.plot(t[list(pmax[0])], pmax[1], 'r*')
    
    plt.show()
  
patient_id = 1
if len(sys.argv) > 1:
    if '-p=' in sys.argv[1]:
        patient_id = int(sys.argv[1][3:])
show_segmentation(patient_id)