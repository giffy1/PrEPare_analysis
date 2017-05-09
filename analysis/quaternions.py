# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 07:57:27 2016

@author: snoran

http://stackoverflow.com/questions/4870393/rotating-coordinate-system-via-a-quaternion
"""

import numpy as np

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v
    
def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z
    
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)
    
def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]
    
def q(theta, axis):
#    axis = normalize(axis)
    sin = np.sin(theta / 2)
    cos = np.cos(theta / 2)
    return cos, sin * axis[0], sin * axis[1], sin * axis[2]