#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:22:54 2021

@author: fcheng
"""

import glob
import os

import numpy as np
import pandas as pd


from eval.image_process import img_process
from eval.identify_line_orientation import principalOrient_Radon_var, create_region_mask, crop_region 



# save dirs
save_dir = "./results/evaluation"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_title = 'Principal_orientation_local'


# Target images
images =  ['abuttingLine001_linespace_1p2_angle_000_045diag',
           'abuttingLine002_linespace_1p2_angle_000_045diag_pos',
           'abuttingLine003_linespace_1p2_angle_000_135diag',
           'abuttingLine004_linespace_1p2_angle_000_135diag_pos',
           'abuttingLine005_linespace_1p2_angle_000_vline',
           'abuttingLine006_linespace_1p2_angle_000_vline_pos',
           'abuttingLine007_linespace_1p2_angle_090_045diag',
           'abuttingLine008_linespace_1p2_angle_090_045diag_pos',
           'abuttingLine009_linespace_1p2_angle_090_135diag',
           'abuttingLine010_linespace_1p2_angle_090_135diag_pos',
           'abuttingLine011_linespace_1p2_angle_090_hline',
           'abuttingLine012_linespace_1p2_angle_090_hline_pos',
           'abuttingLine013_linespace_2p4_angle_000_vline',
           'abuttingLine014_linespace_2p4_angle_000_vline_pos',
           'abuttingLine015_linespace_2p4_angle_090_hline',
           'abuttingLine016_linespace_2p4_angle_090_hline_pos',
           'abuttingLine017_linespace_6_angle_000_vline',
           'abuttingLine018_linespace_6_angle_000_vline_pos',
           'abuttingLine019_linespace_6_angle_090_hline',
           'abuttingLine020_linespace_6_angle_090_hline_pos'
          ]

stim_type_list = ['Illusion', 'Positive control', 'Illusion', 'Positive control', 'Illusion', 'Positive control',
             'Illusion', 'Positive control', 'Illusion', 'Positive control', 'Illusion', 'Positive control',
             'Illusion', 'Positive control', 'Illusion', 'Positive control', 'Illusion', 'Positive control',
             'Illusion', 'Positive control']
orient_inducer_list = [0, 0, 0, 0, 0, 0, 90, 90, 90, 90, 90, 90, 0, 0, 90, 90, 0, 0, 90, 90]
orient_illus_list = [45, 45, 135, 135, 90, 90, 45, 45, 135, 135, 
                0, 0, 90, 90, 0, 0, 90, 90, 0, 0]
line_space_list = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2,
              2.4, 2.4, 2.4, 2.4, 6, 6, 6, 6]

# parameters
img_size = 227
theta = np.linspace(0., 180., 180, endpoint=False)
r_list = [5]
mask_dict = create_region_mask(img_size) # crop 4 regions
regiontypes = ['Illusory', 'Non-illusory']

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

# Main #######################################################################

# Initialization
Principal_orientation = []
Orientation_category = []
ROI = []
Subject = []
stimType = []
reconType = []
Line_space = []
stimName = []
Trial = []
regionType = []
Inducer_orientation = []
Illusory_orientation = []


# presented images ------------------------------------------
for i, image in enumerate(images):
    
    img_path = os.path.join(source_image_dir, image+'.'+source_image_ext)
    img = img_process(img_path, img_size, gray=1)
    
    # identify principal orientation
    orient_inducer = orient_inducer_list[i]
    orient_illus = orient_illus_list[i]
    key = str(orient_illus)+' '+str(orient_inducer)
    mask_list = mask_dict[key]
    for m, masks in enumerate(mask_list):
        for mask in masks:
            # region type
            regionType.append(regiontypes[m])
            
            # identify orientation
            crop_image = crop_region(img, mask)
            [orientation, w] = principalOrient_Radon_var(crop_image, r_list, theta)
            Principal_orientation.append(orientation)
        
            # image parameter
            stimName.append(image)
            Trial.append(0)
            stimType.append(stim_type_list[i])
            Line_space.append(line_space_list[i])
            Inducer_orientation.append(orient_inducer)
            Illusory_orientation.append(orient_illus)
            
            # fmri parameter
            reconType.append('Stimulus')
            ROI.append('None')
            Subject.append('None')
            
            # decide category of principal orientation: illusory or inducer
            similarity_inducer = abs(np.cos((orient_inducer-orientation)/180*np.pi))
            similarity_illusory = abs(np.cos((orient_illus-orientation)/180*np.pi))
            if similarity_illusory > similarity_inducer:
                Orientation_category.append('Illusory')
            elif similarity_illusory < similarity_inducer:
                Orientation_category.append('Inducer')
            else:
                Orientation_category.append('Not clear')
print("Stimulus Done")

# recon from true features --------------------------------------
for i, image in enumerate(images):
    
    img_path = os.path.join(path, recon_true_folder, 'recon_image_normalized-'+image + ".tiff")
    img = img_process(img_path, img_size, gray=1)
    
    # identify principal orientation
    orient_inducer = orient_inducer_list[i]
    orient_illus = orient_illus_list[i]
    key = str(orient_illus)+' '+str(orient_inducer)
    mask_list = mask_dict[key]
    for m, masks in enumerate(mask_list):
        for mask in masks:
            # region type
            regionType.append(regiontypes[m])
            
            # identify orientation
            crop_image = crop_region(img, mask)
            [orientation, w] = principalOrient_Radon_var(crop_image, r_list, theta)
            Principal_orientation.append(orientation)
    
            # image parameter
            stimName.append(image)
            Trial.append(0)
            stimType.append(stim_type_list[i])
            Line_space.append(line_space_list[i])
            Inducer_orientation.append(orient_inducer)
            Illusory_orientation.append(orient_illus)
            
            # fmri parameter
            reconType.append('Recon-stimulus features')
            ROI.append('None')
            Subject.append('None')
            
            # decide category of principal orientation: illusory or inducer
            similarity_inducer = abs(np.cos((orient_inducer-orientation)/180*np.pi))
            similarity_illusory = abs(np.cos((orient_illus-orientation)/180*np.pi))
            if similarity_illusory > similarity_inducer:
                Orientation_category.append('Illusory')
            elif similarity_illusory < similarity_inducer:
                Orientation_category.append('Inducer')
            else:
                Orientation_category.append('Not clear')
                    
print("Recon-stimulus features Done")


# recon from decoded features --------------------------------------
for sbj in sbjs:

    print('Subject:{0}'.format(sbj))
 
    for k, roi in enumerate(rois):
        print('ROI:{0}'.format(roi))
        for i, image in enumerate(images):
        
            img_path = os.path.join(path, sbj, roi, 'recon_image_normalized-'+image + "*.tiff")
            imgfiles = sorted(glob.glob(img_path))
            for j, fn in enumerate(imgfiles):
                img = img_process(fn, img_size, gray=1)
                
                # identify principal orientation
                orient_inducer = orient_inducer_list[i]
                orient_illus = orient_illus_list[i]
                key = str(orient_illus)+' '+str(orient_inducer)
                mask_list = mask_dict[key]
                for m, masks in enumerate(mask_list):
                    for mask in masks:
                        # region type
                        regionType.append(regiontypes[m])
                        
                        # identify orientation
                        crop_image = crop_region(img, mask)
                        [orientation, w] = principalOrient_Radon_var(crop_image, r_list, theta)
                        Principal_orientation.append(orientation)
                        
                        # image parameter
                        stimName.append(image)
                        Trial.append(j+1)
                        stimType.append(stim_type_list[i])
                        Line_space.append(line_space_list[i])
                        Inducer_orientation.append(orient_inducer)
                        Illusory_orientation.append(orient_illus)
                        
                        # fmri parameter
                        reconType.append('Recon-decoded features')
                        ROI.append(roi)
                        Subject.append(sbj)
                        
                        # decide category of principal orientation: illusory or inducer
                        similarity_inducer = abs(np.cos((orient_inducer-orientation)/180*np.pi))
                        similarity_illusory = abs(np.cos((orient_illus-orientation)/180*np.pi))
                        if similarity_illusory > similarity_inducer:
                            Orientation_category.append('Illusory')
                        elif similarity_illusory < similarity_inducer:
                            Orientation_category.append('Inducer')
                        else:
                            Orientation_category.append('Not clear')



# save results -----------------------------------------------------------
po_pkl_file = os.path.join(save_dir, save_title+'.pkl')   
PrOr = pd.DataFrame.from_dict({'Principal orientation': Principal_orientation, 
                              'Orientation category': Orientation_category,
                              'ROI': ROI, 'Subject': Subject, 'Trial':Trial, 
                              'stimName': stimName, 'stimType':stimType, 
                              'reconType':reconType, 'regionType': regionType,
                              'Line space':Line_space, 
                              'Inducer orientation': Inducer_orientation,
                              'Illusory orientation':Illusory_orientation
                              })
PrOr.to_pickle(po_pkl_file)
           
          

print('All done')
