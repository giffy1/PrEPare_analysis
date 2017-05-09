# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:08:49 2016

@author: cs390mb

This is the solution code for extracting features over accelerometer 
windows. It doesn't include all the features mentioned in the 
assignment, but it does have mean/variance, histogram features, 
and dominant frequencies over the magnitude.

"""

import numpy as np
from scipy.stats import stats

def _compute_mean_features(window):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(window, axis=0)
    
def _compute_min_features(window):
    """
    Computes the min x, y and z acceleration over the given window. 
    """
    return np.min(window, axis=0)
   
def _compute_max_features(window):
    """
    Computes the max x, y and z acceleration over the given window. 
    """
    return np.max(window, axis=0)

def _compute_var_features(window):
    """
    Computes the variance along the x-, y- and z-axes 
    over the given window.
    """
    return np.std(window, axis=0)
    
def _compute_fft_of_magnitude_features(magnitude):
    """
    Computes the first 2 dominant frequencies of the magnitude 
    signal, given by the real-valued FFT.
    """
    n_freq=32
    sp = np.fft.fft(magnitude, n=n_freq)
    freq = np.fft.fftfreq(n_freq)
    return freq[sp.argsort()][:2]
    
def _compute_histogram_features(window):
    return np.histogram(window, bins=4)[0]
    
def _compute_skew(window):
    return stats.skew(window, axis=0, bias=False)
    
def _compute_kurtosis(window):
    return stats.kurtosis(window, axis=0, bias=False)
    
def _compute_zero_crossing_count(window):
    b = np.sign(window)
    return np.sum(np.diff(b, axis=0)!=0, axis=0)
    
def _compute_zero_crossing_rate(window):
    return _compute_zero_crossing_count(window)/ float(len(window))
    
def _compute_amplitude(window):
    return np.max(window, axis=0) - np.min(window, axis=0)
    
def extract_features(accelWindow, gyroWindow, distWindow, quaternionWindow):
    x = []

    window_len = len(distWindow)
    peak_index = np.argmax(distWindow) #point at mouth
    ascending_accel = accelWindow[:peak_index,:]
    descending_accel = accelWindow[peak_index:,:]
    
    ascending_gyro = gyroWindow[:peak_index,:]
    descending_gyro = gyroWindow[peak_index:,:]  
    
    
    if len(ascending_accel) == 0 or len(descending_accel) == 0:
#        print "MAX POINT AT BEGINNING OR END"
        peak_index = np.argmin(distWindow)
        ascending_accel = accelWindow[:peak_index,:]
        descending_accel = accelWindow[peak_index:,:]
    
        ascending_gyro = gyroWindow[:peak_index,:]
        descending_gyro = gyroWindow[peak_index:,:]  

    x=extract_features_for_window(x, accelWindow)
    x=extract_features_for_window(x, gyroWindow)
    
    x=extract_features_for_window(x, ascending_accel)
    x=extract_features_for_window(x, ascending_gyro)
    
    x=extract_features_for_window(x, descending_accel)
    x=extract_features_for_window(x, descending_gyro)
        
    g = 9.8    
    gX = accelWindow[:,0] / g
    gY = accelWindow[:,1] / g
    gZ = accelWindow[:,2] / g
    
    qW = quaternionWindow[:,0]
    qX = quaternionWindow[:,1]
    qY = quaternionWindow[:,2]
    qZ = quaternionWindow[:,3]
    
    roll = (np.arctan2(2*qY*qW + 2*qX*qZ, 1 - 2*qY*qY - 2*qZ*qZ)*180)/np.pi # using quaternions
#    roll = (np.arctan2(-gY, gZ)*180)/np.pi # using gyro
    
    yaw = (np.arctan2(2*qX*qW + 2*qY*qZ, 1 - 2*qX*qX - 2*qZ*qZ)*180)/np.pi # using quaternions

    pitch = (np.arcsin(2*qX*qY + 2*qZ*qW)*180)/np.pi # we can do this?? from quaternions again 
#    pitch = (np.arctan2(gX, np.sqrt(gY * gY + gZ * gZ))*180)/np.pi # using gyro
    
    roll_deciles = np.percentile(roll, np.arange(0, 100, 10))
    pitch_deciles = np.percentile(pitch, np.arange(0, 100, 10))
    yaw_deciles = np.percentile(yaw, np.arange(0, 100, 10))
    x = np.append(x, roll_deciles)
    x = np.append(x, pitch_deciles)
    x = np.append(x, yaw_deciles)
    x = np.append(x, roll[-1] - roll[0])
    x = np.append(x, window_len) # total duration
    x = np.append(x, peak_index) # ascending duration
    x = np.append(x, window_len - peak_index) # descending duration
    x = np.append(x, np.abs(distWindow[-1] - distWindow[0])) # difference in starting / end position relative to common start point
    return x

def extract_features_for_window(x, window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature matrix X.
    
    Make sure that X is an N x d matrix, where N is the number 
    of data points and d is the number of features.
    
    """
    
    magnitude = np.sqrt(np.sum(np.square(window), axis=1))
        
    x = np.append(x, _compute_mean_features(window))
    x = np.append(x, _compute_var_features(window))
    x = np.append(x, _compute_min_features(window))
    x = np.append(x, _compute_max_features(window))
    x = np.append(x, _compute_skew(window))
    x = np.append(x, _compute_kurtosis(window))
    x = np.append(x, _compute_zero_crossing_rate(window))
    x = np.append(x, _compute_zero_crossing_rate(window - np.mean(window, axis=0)))
    x = np.append(x, _compute_amplitude(window))
    x = np.append(x, _compute_mean_features(magnitude))
    x = np.append(x, _compute_var_features(magnitude))
    x = np.append(x, _compute_min_features(magnitude))
    x = np.append(x, _compute_max_features(magnitude))
    x = np.append(x, _compute_skew(magnitude))
    x = np.append(x, _compute_kurtosis(magnitude))
    x = np.append(x, _compute_zero_crossing_rate(magnitude))
    x = np.append(x, _compute_zero_crossing_rate(magnitude - np.mean(magnitude, axis=0)))
    x = np.append(x, _compute_amplitude(magnitude))
    
#    x = np.append(x, _compute_fft_of_magnitude_features(magnitude))
    x = np.append(x, _compute_histogram_features(window))
    
    return x