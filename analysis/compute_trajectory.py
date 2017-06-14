# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 16:35:48 2017

PrEPare : Trajectory Computation
UMass Amherst & Swedish Medical

This script computes and writes to disk the trajectory from the 
accelerometer and gyroscope streams, using Madgwick's fusion algorithm.

"""

import numpy as np
import os
from madgwick_py.madgwickahrs import MadgwickAHRS
from quaternions import qv_mult
from scipy import interpolate
from multiprocessing import Process
import constants

data_dir = constants.PROCESSING_DIR

# patients 102 and 105 don't have good calibration points; patients 1 and 103 switch arms halfway through
calibration_times = {1 : -1, 2 : 0, 3 : 30000, 4 : 137000, 97 : 10000, 98 : 34000, 99 : 0, 100 : 0, 101 : 34000, 
                     102 : 11000, 103 : -1, 104 : 39000, 105 : 0, 106 : 28000, 107 : 0, 108 : 70000, 109 : 0, 110 : 31000, 111 : 0}
                     
orientation_corrections = { 1 : [-1,-1,-1], 2 : [1,-1,-1], 3 : [1,1,-1], 4 : [1,1,-1], 97 : [1,-1,-1], 98 : [1,1,-1], 
                            99 : [1,-1,-1], 100 : [1,1,-1], 101 : [1,-1,-1], 102 : [-1,1,-1], 103 : [1,-1,-1], 
                            104 : [1,1,-1], 105 : [1,1,-1], 106 : [1,-1,-1], 107 : [-1,1,-1], 108 : [1,1,-1], 
                            109 : [1,-1,-1], 110 : [-1,-1,1], 111 : [1,1,-1]}

def compute_trajectory(patient_id):
    """
    Computes the trajectory and quaternions by fusing the given subject's 
    accelerometer and gyroscope data and writes them to disk. It also 
    writes the smoothed trajectory to disk.
    
    Smoothing is done using B-spline interpolation with a smoothing 
    factor of s=2. The trajectory is computed using Madgwick's fusion 
    algorithm.
    
        :param patient_id The ID of the patient.    
    
    """
    subject_dir = os.path.join(data_dir, str(patient_id))
    print "Loading {}".format(subject_dir)
    
    with open(os.path.join(subject_dir, constants.WEARABLE_ACCELEROMETER_FN), "rb") as f:
        wearable_accel_i = np.loadtxt(f, delimiter=",")  
        
    with open(os.path.join(subject_dir, constants.WEARABLE_GYROSCOPE_FN), "rb") as f:
        wearable_gyro_i = np.loadtxt(f, delimiter=",")  
                
    t = wearable_gyro_i[:,0]
        
    w = (1,0,0) # assumed initial position
    fuse = MadgwickAHRS(sampleperiod=1./constants.SAMPLING_RATE)
    trajectory = []
    quaternions = []
    for i in range(len(wearable_accel_i)):
        fuse.update_imu(wearable_gyro_i[i,1:4] * constants.DEG_TO_RAD, wearable_accel_i[i,1:4])
        w2 = qv_mult(fuse.quaternion.q, w)
        quaternions.append(fuse.quaternion.q)
#         w2 = np.multiply(w2, orientation_corrections[patient_id]) # apply orientation correction
        trajectory.append(w2)
        
    T = np.asarray(trajectory)
    Q = np.asarray(quaternions)
        
    # uncomment to see the calibration points:
#    t_calibrate = np.where(t >= calibration_times[patient_id])[0][0] # first point after calibration timestamp
#    print "Patient {} has wearable accelerometer {} at calibration time {}".format(patient_id, wearable_accel_i[t_calibrate, 1:], calibration_times[patient_id])
    
    print "Saving trajectory and quaternion data for patient {}.".format(patient_id)   
    
    trajectory_with_timestamps = np.hstack((t.reshape((-1,1)), T))
    with open(os.path.join(subject_dir, constants.TRAJECTORY_FN), "wb") as f:
        np.savetxt(f, trajectory_with_timestamps, delimiter=",", fmt="%1f")
        
    quaternions_with_timestamps = np.hstack((t.reshape((-1,1)), Q))
    with open(os.path.join(subject_dir, constants.QUATERNION_FN), "wb") as f:
        np.savetxt(f, quaternions_with_timestamps, delimiter=",", fmt="%1f")
    
    tck, u = interpolate.splprep([T[:,0],T[:,1],T[:,2]], s=2)
    x_knots, y_knots, z_knots = interpolate.splev(tck[0], tck)
    u_fine = np.linspace(0,1,len(T))
    x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
    T_smooth = np.vstack((x_fine, y_fine, z_fine)).transpose()
    
    smooth_trajectory_with_timestamps = np.hstack((t.reshape((-1,1)), T_smooth))
    with open(os.path.join(subject_dir, constants.SMOOTH_TRAJECTORY_FN), "wb") as f:
        np.savetxt(f, smooth_trajectory_with_timestamps, delimiter=",", fmt="%1f")
        
for j in [1,2,3,4] + range(97,143): # participants 1-4 are Swedish folks, 97-112 are HIV patients
    p = Process(target=compute_trajectory, args=(j,))
    p.start()
