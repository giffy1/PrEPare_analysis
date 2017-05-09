# -*- coding: utf-8 -*-
"""
Created on Tue May  9 09:30:45 2017

@author: snoran
"""

import os
import numpy as np

DATA_DIR = os.path.join('..', 'data')
PROCESSING_DIR = os.path.join('..', 'corrected_data')
OUTPUT_DIR = os.path.join('..', 'outputs')

GRAVITY = 9.8

WEARABLE_ACCELEROMETER = ("ACCELEROMETER", "WEARABLE")
WEARABLE_GYROSCOPE = ("GYROSCOPE", "WEARABLE")
METAWEAR_ACCELEROMETER = ("ACCELEROMETER", "METAWEAR")

WEARABLE_ACCELEROMETER_FN = 'wearable_accelerometer.csv'
WEARABLE_GYROSCOPE_FN = 'wearable_gyroscope.csv'
METAWEAR_ACCELEROMETER_FN = 'metawear_accelerometer.csv'
TRAJECTORY_FN = 'trajectory.csv'
QUATERNION_FN = 'quaternions.csv'
SMOOTH_TRAJECTORY_FN = 'smooth_trajectory.csv'
LABELS_FN = 'labels.txt'
CLASSIFIER_FN = 'classifier.pickle'

SEGMENTS_FN = 'segments.pickle'
ALL_SEGMENTS_FN = 'all_segments.pickle'

SAMPLING_RATE = 72.

DEG_TO_RAD = np.pi / 180.