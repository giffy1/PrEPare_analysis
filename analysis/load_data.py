# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:30:01 2016

PrEPare : Reading Raw Data
UMass Amherst & Swedish Medical

This script is used to load the raw sensor data, metadata and corresponding 
labels.

"""

import numpy as np
import os

def load_data(directory):
    """
    Loads labelled sensor data from the given subject directory.
    
    Returns the data, which is a dictionary whose keys are the (device, sensor) 
    pair, e.g. ("ACCELEROMETER", "WEARABLE") and whose values are the 
    corresponding data streams, along with the labels, which include 
    the start timestamp, end timestamp and type of each gesture.
    
        :param directory The directory containing the data files for 
        a particular user.
      
    The data directory has a particular structure and sould not be 
    modified. When generating new files, they should be generated 
    into a separate directory. The files in a subject's data directory 
    include:
        
        ACCELEROMETER_METAWEAR_PATIENT1_###.csv
        ACCELEROMETER_WEARABLE_PATIENT1_###.csv
        GYROSCOPE_WEARABLE_PATIENT1_###.csv
        labels.txt
        VIDEO###.mp4
        
    where in every case ### refers to the timestamp at which that sensor 
    stream was started.
    
    Note that the patient number is always 1, regardless of which subject 
    it is, because the people collecting the data didn't change the patient 
    number, but made sure that the data was organized by folder instead.
    
    """
    video_start = 0
    
    data = {}

    labels = None
    labels_fn = 'labels.txt'
    if not os.path.exists(os.path.join(directory, labels_fn)):
        print("WARNING: No labels found in {}.".format(directory))
    else:
        with open(os.path.join(directory, labels_fn), "rb") as f:
            labels = np.loadtxt(f, delimiter=',', comments='#').astype(int)
    
    files = os.listdir(directory)
        
    for fname in files:
        if fname.startswith("VIDEO"):
            video_start = int(fname[5:].split(".")[0])
    
    # no video found or saved incorrectly:
    if video_start == 0:
        print("WARNING: No annotation video found in {}.".format(directory))
        
    for fname in files:
        if (fname.endswith(".csv")):
            meta = fname.split("_")
            if len(meta) != 4:
                print "WARNING : File {} has invalid name. Expected it to be in form SENSOR_DEVICE_PATIENT#_TIMESTAMP.csv".format(fname)
                continue
            
            sensor = meta[0]
            device = meta[1]
            
            with open(os.path.join(directory, fname), "rb") as f:
                sensor_data = np.loadtxt(f, delimiter=',')  
            
            t = sensor_data[:,0]
            start_index = np.where(t>=video_start)[0][0] # only load data points after video start
                
            sensor_data = sensor_data[start_index:,:]
            sensor_data[:,0] = sensor_data[:,0] - video_start
                
            data[(sensor, device)] = sensor_data
            
    return data, labels
    