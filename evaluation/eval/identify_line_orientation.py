#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:29:34 2021

@author: fcheng
"""
import numpy as np

from skimage.transform import radon
from skimage import draw

def principalOrient_Radon_var(image, r_list, theta=None):
    """
    Identify the principal orientation in `img` from a list of orientations `theta`,
    based on variance along each orientation
    
    Inputs:
    r_list - the array of radius of region where lines of orientations overlap (usually small)
    
    Ouput:
    orientation -  the principal orientation

    """
    # set pixels out of circle to 0
    l = image.shape[0]
    radius = l//2
    [rr, cc] = draw.ellipse(radius-1, radius-1, radius, radius) 
    img_new = np.zeros((l, l))
    img_new[rr, cc] = image[rr, cc]
    
    # radon transform
    R = radon(img_new, theta=theta, circle=True, preserve_range=False)
    
    centerPos = R.shape[0]//2
    n = len(theta)
    # w: an array of weights corresponding to orientations that indicate how 
    # likely it is the principal orientation
    w = np.zeros(n)
    for r in r_list:
        startPos = int(centerPos-r)
        endPos = int(centerPos+r)
        v = np.var(R[startPos:endPos, :], axis=0)
        
        v = np.concatenate((v[90:180], v[0:90]),axis=0) 
        w = w + v/np.sum(v)
    
    maxw =  np.max(w) 
    orientation = np.where(w == maxw)[0][0]
    #print(orientation, maxw)

    return orientation, maxw




def create_region_mask(img_size):
    """
    Create mask of 4 disk regions in an image
    """
    # Initialize mask
    # 'illusory orientation, inducer orientation'
    keys = ['0 90', '90 0', '45 0', '45 90', '135 0', '135 90']
    mask = dict.fromkeys(keys)
   
    radius = np.floor(img_size/4)-1
    ##############################################################
    # illusory orientation = 0 or 90
    y = np.floor(img_size/2)
    # left disk
    D1 = np.zeros((img_size, img_size))
    [rr, cc] = draw.ellipse(y, radius+1, radius, radius)
    D1[rr,cc] = 1
    # right disk
    D2 = np.zeros((img_size, img_size))
    [rr, cc] = draw.ellipse(y, img_size-1-radius, radius, radius)
    D2[rr,cc] = 1
    # top disk
    D3 = np.zeros((img_size, img_size))
    [rr, cc] = draw.ellipse(radius+1, y, radius, radius)
    D3[rr,cc] = 1
    # bottom disk
    D4 = np.zeros((img_size, img_size))
    [rr, cc] = draw.ellipse(img_size-1-radius, y, radius, radius)
    D4[rr,cc] = 1
    
    # [[masks overlap illusory line], [masks overlap only inducer line]]
    mask['0 90'] = [[D1, D2], [D3, D4]]
    mask['90 0'] = [[D3, D4], [D1, D2]]
    
    ##############################################################
    # illusory orientation = 45, inducer orientation = 0 or 90
    offset = np.ceil(radius/np.sqrt(2))
    # topright disk
    D1 = np.zeros((img_size, img_size))
    [rr, cc] = draw.ellipse(y-offset, y+offset, radius, radius)
    D1[rr,cc] = 1
    # bottomleft disk
    D2 = np.zeros((img_size, img_size))
    [rr, cc] = draw.ellipse(y+offset, y-offset, radius, radius)
    D2[rr,cc] = 1
    # lefttop disk
    D3 = np.zeros((img_size, img_size))
    [rr, cc] = draw.ellipse(y-offset, y-offset, radius, radius)
    D3[rr,cc] = 1
    # rightbottom disk
    D4 = np.zeros((img_size, img_size))
    [rr, cc] = draw.ellipse(y+offset, y+offset, radius, radius)
    D4[rr,cc] = 1
   
    # [[masks overlap illusory line], [masks overlap only inducer line]]
    mask['45 0'] = [[D1, D2], [D3, D4]]
    mask['45 90'] = [[D1, D2], [D3, D4]]
    mask['135 0'] = [[D3, D4], [D1, D2]]
    mask['135 90'] = [[D3, D4], [D1, D2]]
    
   
    return mask


def create_center_region_mask(img_size):
    """
    Create mask of 4 disk regions in an image
    """
   
    radius = np.floor(img_size/4)-1
    ##############################################################
    # illusory orientation = 0 or 90
    y = np.floor(img_size/2)
    # cneter disk
    D = np.zeros((img_size, img_size))
    [rr, cc] = draw.ellipse(y, y, radius, radius)
    D[rr,cc] = 1
      
    # [mask contain both illusory/inducer line]
    mask = D
    
    return mask
    

def crop_region(img, mask):
    """
    crop smallest square region out of image containing mask
    and set pixel outside mask to 0
    """
    
    # crop square
    img = img[np.any(mask, axis=1)] # rows
    img = img[:,np.any(mask, axis=0)] # columns
    
    return img
    
  