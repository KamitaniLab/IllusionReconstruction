#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 10:57:11 2022

@author: fcheng
"""

import numpy as np

def cal_orientation_index(theta, R):
    """
    calculate OI of unit for theta, using unit responses to orientations R
    (larger OI, higher selectivity)
    1st dimension of R and the length of theta are equivalent
    """
    
    n_theta = len(theta)  
    
    # response to orthogonal orientation
    R_orth = np.roll(R, int(n_theta/2),axis=0)
    
    OI = (R-R_orth)/R
    
    return OI


def cal_CirVar(theta, R):
    """
    calculate circular variance that quantify orientation selectivity for unit, 
    (smaller CirVar, higher selectivity)
    using unit responses to orientations R
    R and theta (degree) are the numpy arrays of the same length
    theta: 1 x num_orient
    R: num_orient x num_unit
    L: (num_unit,)
    """
    
    itheta = np.array([complex(0, 2*t/180*np.pi) for t in theta]).reshape(-1,1)
    L = np.sum(R*np.exp(itheta),axis=0)/np.sum(R,axis=0)
    
    return 1-np.abs(L)