# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 17:05:12 2016

@author: cs390mb

My Activities Main Python Module

This is the main entry point for the Python analytics.

You should modify the user_id field passed into the Client instance, 
so that your connection to the server can be authenticated. Also, 
uncomment the lines relevant to the current assignment. The 
map_data_to_function() function simply maps a sensor type, one of 
"SENSOR_ACCEL", "SENSOR_AUDIO" or "SENSOR_CLUSTERING_REQUEST", 
to the appropriate function for analytics.

"""

from client import Client
import json
from quaternions import qv_mult
from madgwick_py.madgwickahrs import MadgwickAHRS
import numpy as np

# instantiate the client, passing in a valid user ID:
user_id = "42.8d.7c.7d.69.92.f8.eb.dc.b5"
c = Client(user_id)

start_time = -1

fuse = MadgwickAHRS(sampleperiod=1./72)
accel_vals = [0,0,0]
count = 0

def compute_trajectory(data, send_notification):
    print "got accel", data['t'], data['x'], data['y'], data['z']
    global start_time
    global accel_vals
    if start_time == -1:
        start_time = data['t']
    global count
    count += 1
    print count
    accel_vals[0] = data['x']
    accel_vals[1] = data['y']
    accel_vals[2] = data['z']
    print "accel", (data['t'] - start_time) / 10**6
    return
#    c.send_socket.send(json.dumps({'user_id' : user_id, 'sensor_type' : 'SENSOR_TRAJECTORY', 'data': {'t' : data['t'], 'x' : position[0], 'y' : position[1], 'z' : position[2]}}) + "\n")
    
def got_quat(data, send_notification):
    return
#    global start_time 
#    if start_time == -1:
#        start_time = data['t']
#    w = data['w']
#    x = data['x']
#    y = data['y']
#    z = data['z']
#    position = qv_mult((w,x,y,z), (1,0,0))
#    c.send_socket.send(json.dumps({'user_id' : user_id, 'sensor_type' : 'SENSOR_TRAJECTORY', 'data': {'t' : data['t'], 'x' : position[0], 'y' : position[1], 'z' : position[2]}}) + "\n")
    
def got_gyro(data, send_notification):
    print "got gyro", data['t'], data['x'], data['y'], data['z']
    global start_time
    global accel_vals
    if start_time == -1:
        start_time = data['t']
    fuse.update_imu(np.asarray([data['x'], data['y'], data['z']])*np.pi / 180., accel_vals)
    position = qv_mult(fuse.quaternion, (1, 0, 0))
    c.send_socket.send(json.dumps({'user_id' : user_id, 'sensor_type' : 'SENSOR_TRAJECTORY', 'data': {'t' : data['t'], 'x' : position[0], 'y' : position[1], 'z' : position[2]}}) + "\n")
    print "gyro", (data['t'] - start_time) / 10**6
    return

c.map_data_to_function("SENSOR_ACCEL", compute_trajectory)
c.map_data_to_function("SENSOR_GYRO", got_gyro)
#c.map_data_to_function("SENSOR_QUATERNION", got_quat)

# connect to the server to begin:
c.connect()