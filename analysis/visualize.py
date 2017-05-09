# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:30:01 2016

@author: snoran

PrEPare : Analysis 

"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
    
formats = ['k-', 'r-', 'g-', 'b-', 'c-', 'm-']

def visualize_data(data, labels=None, title=''):
    plt.figure()

    subplot_index=0
    
    for i, k in enumerate(data.keys()):
        t = data[k][:,0]
        z = np.sqrt(np.sum(np.square(data[k][:,1:4]),axis=1))
        
        subplot_index+=1
        if i == 0:
            ax=plt.subplot(3, 1, subplot_index)
        else:
            ax=plt.subplot(3, 1, subplot_index,sharex=ax)
        if labels != None:
            if len(labels.shape)==1:
                labels = labels.reshape((1,-1))
            for i,label in enumerate(labels):
                indexes = (t >= label[0]-50) & (t <= label[1]+50)
                #+/-50 ensures that there are no discontinuities
                if i > 0:
                    no_label_indexes = (t >= labels[i-1][1]) & (t <= label[0])
                    ax.plot(t[no_label_indexes], z[no_label_indexes], formats[0])
                elif i == 0:
                    no_label_indexes = (t <= label[0])
                    ax.plot(t[no_label_indexes], z[no_label_indexes], formats[0])
                if i == len(labels) - 1:
                    no_label_indexes = (t >= label[1])
                    ax.plot(t[no_label_indexes], z[no_label_indexes], formats[0])
                ax.plot(t[indexes], z[indexes], formats[int(label[2])])
        else:
            ax.plot(t, z, formats[0])
        ax.set_title(k[0] + " " + k[1])

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
def show_instances(data, label=1):
    subplot_index=0
    labels = (data[:,-1]==label).astype(int)
    print set(labels)
    diff = labels[1:] - labels[:-1]
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if len(starts) == 0:
        return
    plt.figure()
    for i in range(len(starts)):
        subplot_index+=1
        ax = plt.subplot(2, 4, subplot_index)
        s = starts[i]
        e = ends[i]
        ax.plot(data[s:e,0], data[s:e,1], 'r-')
        ax.plot(data[s:e,0], data[s:e,2], 'g-')
        ax.plot(data[s:e,0], data[s:e,3], 'b-')
        idx = np.argwhere(np.diff(np.sign(data[s:e,1] - data[s:e,2])) != 0).reshape(-1) + 0
        ax.plot(data[idx+s,0],data[idx+s,1], 'go')
    plt.tight_layout()
    plt.show()
    
def plot_histogram(lst, n_bins, bin_range, title, xlabel, ylabel):
	plt.figure()
	hist, bins = np.histogram(lst, bins = n_bins, range=bin_range)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center', width=width)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()
 
def visualize_signals(t, y1, y2, labels=None, label_col=None, accX=None, title=None):
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    if labels != None:
        if len(labels.shape)==1:
            labels = labels.reshape((1,-1))
        for i,label in enumerate(labels):
            indexes = (t >= label[0]-50) & (t <= label[1]+50)
            #+/-50 ensures that there are no discontinuities
            if i > 0:
                no_label_indexes = (t >= labels[i-1][1]) & (t <= label[0])
                ax1.plot(t[no_label_indexes], y1[no_label_indexes], 'k-')
                ax2.plot(t[no_label_indexes], y2[no_label_indexes], 'k-')
            elif i == 0:
                no_label_indexes = (t <= label[0])
                ax1.plot(t[no_label_indexes], y1[no_label_indexes], 'k-')
                ax2.plot(t[no_label_indexes], y2[no_label_indexes], 'k-')
            if i == len(labels) - 1:
                no_label_indexes = (t >= label[1])
                ax1.plot(t[no_label_indexes], y1[no_label_indexes], 'k-')
                ax2.plot(t[no_label_indexes], y2[no_label_indexes], 'k-')
                
            ax1.plot(t[indexes], y1[indexes], formats[int(label[2])])
            ax2.plot(t[indexes], y2[indexes], formats[int(label[2])])
    else:
        ax1.plot(t, y1, 'k-')
        ax2.plot(t, y2, 'k-')
    plt.title(title)
        
    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0).reshape(-1) + 0
    if (y1[0] < y2[0]):
        idx = idx[1::2]
    else:
        idx = idx[::2]
    
    time_threshold = 50
    count = 0
    total = 0
    for i,e in enumerate(idx[1:]):
        s = idx[i]
#        print "elapsed: " + str(e-s)
#        print "sum: " + str(np.sum(np.sign(accX[s:e])))
        v= np.mean(label_col[s:e]>=4)
        dt = e - s
#        print "dt : " + str(dt)
#        if (np.sum(np.sign(accX[s:e])) > 0) or 
        if dt < time_threshold:
            if v > 0:
                print "lost one! " + str(v)
            continue
        total+=1
        if v > 0:
            print "dt : " + str(dt)
#            print "gesture! " + str(v)
            count+=1
            #print(accX[s:e])
        ax1.plot(t[s:e],[1]*(e-s), 'g-')
        ax1.plot(t[s],1, 'ko')
        ax1.plot(t[e],1, 'ko')
    print count
    print total
    print(count / float(total)) 
    # sliding window of 1s with no overlap would give (for subject 1) 1500 
    # segments, whereas this approach produces only 350, 22 of which are positive.
        
    

    plt.show()