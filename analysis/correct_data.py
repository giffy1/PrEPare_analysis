# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:14:38 2017

PrEPare : Data Correction
UMass Amherst & Swedish Medical

This script makes a number of corrections to the raw data, including

    (1) synchronizing the wearable sensor data (accelerometer and gyroscope) 
    with the video from which the start and end labels of gestures were acquired.
    
    (2) removing duplicate points from the data streams that were inadvertantly 
    written twice during data collection.
    
    (3) dividing the data streams by a factor of 9.8, which was inadvertantly 
    multiplied during data collection.
    
    (4) re-orientating the coordinate systems to be consistent across subjects.
    
Note that the time shifts are slightly different per participant, but close 
to 100 ms in all cases. The shifts were determined empircally by aligning 
the ground-truth labels acquired from the video to the sensor signatures 
in the visualized trajectory.

For re-orienting the coordinate systems, I notated the sign of each accelerometer 
axis when the hand was flat on the table, palm down. Each axes of the sensor 
stream is then multiplied by +1 or -1 accordingly, such that for every participant 
this wrist posture corresponds to all positive values. 

I am not sure whether to apply the orientation corrections, because I found better 
performance without doing it, perhaps due to orientation-invariant features.

Corrections are done using multi-processing, one process per subject.

"""

from load_data import load_data
import numpy as np
import os
from multiprocessing import Process
import constants
from shutil import copyfile

old_data_dir = constants.DATA_DIR
new_data_dir = constants.PROCESSING_DIR

# TODO : 3 : 115
# TODO: Look at Patient #97, it seems strange; 102, 103, 104, 107 possibly a slightly longer delay
# maps patient ID to a timestamp shift, used for correction:
shifts = {1 : 55, 2 : 55, 3 : 75, 4 : 55, 97 : 55, 98 : 55, 99 : 55, 100 : 55, 101 : 55, 102 : 75, 
          103 : 75, 104 : 75, 105 : 55, 106 : 55, 107 : 75, 108 : 55, 109 : 55, 110 : 55, 111 : 55}
          
for i in range(112, 143):
    shifts[i] = 55

#orientation_corrections = {1 : [-1,-1,-1], 2 : [1,-1,-1], 3 : [1,1,-1], 4 : [1,1,-1], 97 : [1,-1,-1], 98 : [1,1,-1], 
#                            99 : [1,-1,-1], 100 : [1,1,-1], 101 : [1,-1,-1], 102 : [-1,1,-1], 103 : [1,-1,-1], 
#                            104 : [1,1,-1], 105 : [1,1,-1], 106 : [1,-1,-1], 107 : [-1,1,-1], 108 : [1,1,-1], 
#                            109 : [1,-1,-1], 110 : [-1,-1,1], 111 : [1,1,-1]}

def correct_data(patient_id):
    """
    Corrects the data for a particular subject. This makes a number of corrections:
    
    (1) It shifts the data points to handle snychronization issues.
    (2) It rotates the coordinate system to be consistent across subjects.
    (3) It removes duplicate points from the accelerometer & gyroscope streams.
    (4) It divides both streams by unnecessary 9.8 multiplicative factor.
    
    :param patient_id The ID of the patient.
    
    """
    
    old_subject_dir = os.path.join(old_data_dir, str(patient_id))
    new_subject_dir = os.path.join(new_data_dir, str(patient_id))
    print "Correcting data for patient {} from directory {}...".format(patient_id, old_subject_dir) 
    data_i, labels = load_data(old_subject_dir)

    wearable_gyro_i = data_i[constants.WEARABLE_GYROSCOPE]
    wearable_accel_i = data_i[constants.WEARABLE_ACCELEROMETER]
    metawear_accel_i = data_i[constants.METAWEAR_ACCELEROMETER]
        
    # remove multiplicative factor of 9.8:
    wearable_accel_i[:,1:]=wearable_accel_i[:,1:]/constants.GRAVITY
    wearable_gyro_i[:,1:]=wearable_gyro_i[:,1:]/constants.GRAVITY
    metawear_accel_i[:,1:]=metawear_accel_i[:,1:]/constants.GRAVITY
    
    # make sure the wearable accelerometer and gyroscope streams are of the same length:
    min_length = min(len(wearable_accel_i), len(wearable_gyro_i))
    wearable_accel_i = wearable_accel_i[:min_length, :]
    wearable_gyro_i = wearable_gyro_i[:min_length, :]
    
    # remove duplicates:
    wearable_gyro_i = wearable_gyro_i[::2,:]
    wearable_accel_i = wearable_accel_i[::2,:]
    metawear_accel_i = metawear_accel_i[::2,:]
    
    offset = shifts[patient_id]
    # use timestamps up to offset before end of stream, use sensor data starting at offset to end of stream (this corrects the offset):
    wearable_gyro_i = np.hstack((wearable_gyro_i[:-offset, :1], wearable_gyro_i[offset:, 1:]))
    wearable_accel_i = np.hstack((wearable_accel_i[:-offset, :1], wearable_accel_i[offset:, 1:]))
    metawear_accel_i = np.hstack((metawear_accel_i[:-offset, :1], metawear_accel_i[offset:, 1:]))

    # re-orient coordinate system to make it consistent across subjects
#    wearable_accel_i[:,1] = wearable_accel_i[:,1] * orientation_corrections[patient_id][0]
#    wearable_gyro_i[:,1] = wearable_gyro_i[:,1] * orientation_corrections[patient_id][0]
#    wearable_accel_i[:,2] = wearable_accel_i[:,2] * orientation_corrections[patient_id][1]
#    wearable_gyro_i[:,2] = wearable_gyro_i[:,2] * orientation_corrections[patient_id][1]
#    wearable_accel_i[:,3] = wearable_accel_i[:,3] * orientation_corrections[patient_id][2]
#    wearable_gyro_i[:,3] = wearable_gyro_i[:,3] * orientation_corrections[patient_id][2]

    print "Saving corrected data for patient {} to directory {}...".format(patient_id, new_subject_dir) 
    
    if not os.path.exists(new_subject_dir):
        os.makedirs(new_subject_dir)

    # save corrected data streams
    with open(os.path.join(new_subject_dir, constants.WEARABLE_ACCELEROMETER_FN), "wb") as f:
        np.savetxt(f, wearable_accel_i, delimiter=",")
        
    with open(os.path.join(new_subject_dir, constants.WEARABLE_GYROSCOPE_FN), "wb") as f:
        np.savetxt(f, wearable_gyro_i, delimiter=",")
        
    with open(os.path.join(new_subject_dir, constants.METAWEAR_ACCELEROMETER_FN), "wb") as f:
        np.savetxt(f, metawear_accel_i, delimiter=",")

    copyfile(os.path.join(old_subject_dir, "labels.txt"), os.path.join(new_subject_dir, "labels.txt"))

subject_ids = [1,2,3,4] + range(97,143) # participants 1-4 are Swedish folks, 97-112 are HIV patients
for j in subject_ids:
    p = Process(target=correct_data, args=(j,))
    p.start()
