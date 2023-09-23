#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:04:35 2021

@author: fcheng
"""
import os
import cv2

import numpy as np


from eval.image_process import img_process, normalise_img

def MakeRegressor(image_name, path_to_regressor, regressor, img_size, normalization=1, redness=None , 
                  interception=True, save_regressor_img = None, save_path_img = None):
    """
    Read regressor images and integrate them into a matrix 

    Parameters
    ----------
    image_name : string
        Name of the presented image.
    path_to_regressor : string
        the directory that saves regressor images.
    regressor: list
        list of regressors to be used

    Returns
    -------
    None.

    """
    
    
    if redness is None:
        if interception == True:
            regmat = np.zeros((img_size**2*3, len(regressor)+1))
        else:
            regmat = np.zeros((img_size**2*3, len(regressor)))
    else:  
        if interception == True:
            regmat = np.zeros((img_size**2, len(regressor)+1))
        else:
            regmat = np.zeros((img_size**2, len(regressor)))
        
        maptype = redness

        
    for i,r in enumerate(regressor):
        
        img_path = os.path.join(path_to_regressor, r+'-'+image_name+'.tiff')
        regressor_img = img_process(img_path, img_size=img_size, redness=redness)
        
        if not save_regressor_img is None:
            sav_file = os.path.join(save_path_img, maptype+'-'+r + '-' + image_name + '.tiff') 
            if not os.path.exists(sav_file):
                x_save = np.reshape(regressor_img,[-1,1])
                x_save = np.reshape(x_save, [img_size, img_size])
                cv2.imwrite(sav_file, normalise_img(x_save))
        

        regressor_vec = regressor_img.flatten()/255
        
        if normalization is not None:
            mean = np.mean(regressor_vec)
            std = np.std(regressor_vec, ddof=1)
    
            regmat[:,i] = (regressor_vec - mean)/std
        else:
            regmat[:,i] = regressor_vec
            
    # error term
    if interception == True:
        regmat[:,-1] = 1
    
    return regmat

        
