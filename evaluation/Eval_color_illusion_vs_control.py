#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 10:10:25 2022

@author: fcheng
"""


import glob
import os

import numpy as np
import pandas as pd


from eval.image_process import img_process
from eval.make_regressor import MakeRegressor

# save dirs
save_dir = "./results/evaluation"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

    
image_keys = ['Ehrenstein', 'Varin']  
# Target images                                                                                                                                                                         
images_set = dict.fromkeys(image_keys)
images_set['Ehrenstein'] = [
           'fillingin001_small_unionjack_lumi0p3_sat0p8_connected',
           'fillingin002_small_unionjack_lumi0p3_sat0p8_disconnected',
           'fillingin004_large_unionjack_lumi0p3_sat0p8_connected',
           'fillingin005_large_unionjack_lumi0p3_sat0p8_disconnected',
           'fillingin007_small_cross_lumi0p3_sat0p7_connected',
           'fillingin008_small_cross_lumi0p3_sat0p7_disconnected',
           'fillingin010_large_cross_lumi0p3_sat0p7_connected',
           'fillingin011_large_cross_lumi0p3_sat0p7_disconnected'
           ]
           
images_set['Varin'] =  [
                        'neonVarinImg001_IllusSurf_alpha_0p7_sat_0p3_bgray',     
                        'neonVarinImg002_innerKanizsa_alpha_0p7_sat_0p3_bgray',    
                        ]
# stimulus type
stimtypes = ['Illusion', 'Control']
stimType_list = dict.fromkeys(image_keys)
stimType_list['Ehrenstein'] = ['Illusion', 'Control', 'Illusion', 'Control', 
                               'Illusion', 'Control', 'Illusion', 'Control']
stimType_list['Varin'] = ['Illusion', 'Control']

# pattern type
pattern_list = dict.fromkeys(image_keys)
pattern_list['Ehrenstein'] = ['unionjack', 'unionjack', 'unionjack', 'unionjack',
                              'cross', 'cross', 'cross', 'cross']
pattern_list['Varin'] = ['Illusory surface','Inner Kanizsa']

# size 
size_list = dict.fromkeys(image_keys)
size_list['Ehrenstein'] = [3, 3, 9, 9, 3, 3, 9, 9]
size_list['Varin'] = [6,6] 

# parameters
img_size = 227
n = img_size**2
normalization = None

# presented images
stimuli_dir_root = './data'
source_image_dir = os.path.join(stimuli_dir_root, 'test_image')
source_image_ext = "tif"

# recon from true features
path = './results/reconstruction/recon_images/GAN'
recon_true_folder = 'stimulus_feature'

# recon from decoded features
sbjs = ['S1','S2','S3','S4','S5','S6','S7']
rois = ['VC', 'V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA']

# regressor 
path_to_regressor = './data/regressor/'
maptype = 'Redness'
label = 'stimulus + red surface'
regressor = ['stimulus', 'red_surface']

# Main #######################################################################
for figtype in image_keys:
    
    # images information
    images = images_set[figtype]
    stimtype = stimType_list[figtype]

    # Initialization fot dataframe
    Weight = []
    stimType = []
    Model = []
    Map = []
    ROI = []
    Subject = []
    reconType = []
    Trial = []
    stimName = []
    

    # prepare response vector
    n_sample = n
    # prepare predictor matrix 
    k = len(regressor)
   
    for i, image in enumerate(images):
        x = np.zeros((n_sample, k+1))
        y = np.zeros((n_sample, 1))
        
        # make predictor matrix for each stimulus type  
        X = MakeRegressor(image, path_to_regressor, regressor, img_size, 
                          normalization=normalization, redness=maptype, interception=False)
        # prepare response vector
        img_path = os.path.join(path, recon_true_folder, 'recon_image_normalized-'+image + ".tiff")  
        Y = img_process(img_path, img_size, redness=maptype)     
   
        y[:, 0] = Y.flatten()/255
        x = X    
 
        # linear regression
        w = np.linalg.lstsq(x , y, rcond=None)[0]
        Weight.append(w)
        print("w:{}".format(w))
        

        # model parameter
        Model.append(label)
        stimType.append(stimtype[i])
        stimName.append(image)
                       
        # fmri parameter
        reconType.append('Recon-stimulus features')
        ROI.append(None)
        Subject.append(None)
        Trial.append(None)
    

    # recon from decoded features --------------------------------------
    for sbj in sbjs:

        print('Subject:{0}'.format(sbj))
        
        if figtype == 'Varin' and sbj=='S4':
            continue
     
        for roi in rois:
            print('ROI:{0}'.format(roi))
            

            # prepare response vector
            n_sample = n 
            # prepare predictor matrix 
            k = len(regressor)
            

            for i, image in enumerate(images):

                # make predictor matrix for each stimulus type  
                X = MakeRegressor(image, path_to_regressor, regressor, img_size, 
                                  normalization=normalization, redness=maptype, interception=False)
                # prepare response vector
                img_path = os.path.join(path, sbj, roi, 'recon_image_normalized-'+image + "*.tiff")
                imgfiles = sorted(glob.glob(img_path))
                for f, fn in enumerate(imgfiles): 
                    x = np.zeros((n_sample, k+1))
                    y = np.zeros((n_sample, 1))
                    Y = img_process(fn, img_size, redness=maptype)     

                    y[:, 0] = Y.flatten()/255
                    x = X    
             
                    # linear regression
                    w = np.linalg.lstsq(x , y, rcond=None)[0]
                    Weight.append(w)
                    print("w:{}".format(w))
                    
 
                    # model parameter
                    Model.append(label)
                    stimType.append(stimtype[i])
                    stimName.append(image)
                                   
                    # fmri parameter
                    reconType.append('Recon-decoded features')
                    ROI.append(roi)
                    Subject.append(sbj)
                    Trial.append(f+1)
                    
        
    # save results -----------------------------------------------------------
    save_title = 'Regression_color_'+ figtype
    reg_pkl_file = os.path.join(save_dir, save_title+'.pkl')  
    reg = pd.DataFrame.from_dict({ 
                                    'Beta coefficient': Weight, 
                                    'Trial':Trial, 'stimName': stimName, 'stimType':stimType,'reconType':reconType,
                                   'Model':Model,'ROI': ROI, 'Subject': Subject, 
                                  })
    reg.to_pickle(reg_pkl_file)
               
              
    
print('All done')

    















