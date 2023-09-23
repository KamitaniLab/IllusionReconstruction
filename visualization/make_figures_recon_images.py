#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:09:10 2022

@author: fcheng
"""


#coding:utf-8

import os
import numpy as np
import PIL
from PIL import ImageFont
import cv2


from plot.image_process import img_process, gammaCorrection



# save dirs
save_dir = "./results/plots"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_ext = 'pdf'

# fmri
roi = 'VC'
sbjs = ['S1', 'S2']

drawtypes = ['stimulus','Recon-stimulus', 'Recon-decoded'] 

# Target images
images = [ 
            'abuttingLine005_linespace_1p2_angle_000_vline',
            'abuttingLine003_linespace_1p2_angle_000_135diag',
            'abuttingLine017_linespace_6_angle_000_vline',
            'fillingin004_large_unionjack_lumi0p3_sat0p8_connected',
            'fillingin005_large_unionjack_lumi0p3_sat0p8_disconnected', 
            'fillingin007_small_cross_lumi0p3_sat0p7_connected',
            'fillingin008_small_cross_lumi0p3_sat0p7_disconnected',
            'neonVarinImg001_IllusSurf_alpha_0p7_sat_0p3_bgray',     
            'neonVarinImg002_innerKanizsa_alpha_0p7_sat_0p3_bgray']



# stimulus images
stimuli_dir_root = './data'
source_image_dir = os.path.join(stimuli_dir_root, 'test_image')
source_image_ext = "tif"

true_image_filepath_list = []
for a_image in images:
    a_filepath = os.path.join(source_image_dir, a_image + "." + source_image_ext) 
    if os.path.exists(a_filepath):
        true_image_filepath_list.append(a_filepath)

# recon images settings
# recon from stimulus features
recon_dir = './results/reconstruction/recon_images/GAN'


# trial number
n_trial = 4
trials = {'S1': np.array([[6,9,11,20], [5, 8,12,18], [8,9,12,18],
                    [3,7,8,10], [3,7,8,10],[2,8,13,17], [2,8,13,17],                     
                    [5,15,17,19], [5,15,17,19]]),
          'S2': np.array([[3,8,11,15], [5,9,13,15], [7,9,10,15],
                    [5,11,12,15],[3,7,11,15],[1,10,15,16],[4,10,13,15],
                    [2,6,14,18],[6,8,10,18]])}

####################
def save_img(image, save_jpg_file):
    # cast to int
    image = image.astype('uint8')
    # convert ndarray to image object
    image_obj = PIL.Image.fromarray(image)
    
    # Save
    image_obj.save(save_jpg_file, quality=98)
    print("Save:", save_jpg_file)

### settings ##############################
# font settings
font_diff = 2
font_size = 36
font_color = (0, 0, 255)
font_bg_color = (255, 255, 255)
font = ImageFont.load_default()

# position, image size settings
h_mergin = 2
w_mergin = 2
img_size = (80, 80)

font_family_path = "/usr/share/fonts/google-crosextra-carlito/Carlito-Bold.ttf"

### make figure ##############################
#------------------------------------
# make canvas
#------------------------------------
ncol = 1+1+n_trial*len(sbjs)
nImg_col = ncol
nImg_row = len(images) 
size_x = (img_size[0]+h_mergin)*nImg_row+h_mergin
size_y = (img_size[1]+w_mergin)*nImg_col+w_mergin
image = np.ones( (size_x, size_y, 3) )*255

    
# Get true image set
true_image_id_list = [os.path.basename(t).split(".")[0] for t in true_image_filepath_list]

print("True image num:", len(true_image_filepath_list))
print("True image:", true_image_filepath_list)
stimulus_selection_size = len(true_image_id_list)
 

for drawtype in drawtypes:

    if drawtype == 'stimulus':
        
        #------------------------------------
        # draw true image
        #------------------------------------
        for j in range(stimulus_selection_size):
            img_true_filepath = true_image_filepath_list[j]
            img_true = img_process(img_true_filepath,img_size)
            img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2RGB)
            img_true = gammaCorrection(img_true)
            yi = j // nImg_row
            xi = j % nImg_row
            x = xi * (img_size[1]+h_mergin)+h_mergin
            y = yi * (img_size[1]+w_mergin)+w_mergin
            image[ x:(x+img_size[0]), y:(y+img_size[1]), : ] = np.array(img_true)[:,:,:]

        
    elif drawtype == 'Recon-stimulus':
        #------------------------------------
        # draw recon from stimulus feature
        #------------------------------------
        recon_true_image_filepath_list = []
        for a_image in images:
            a_filepath = os.path.join(recon_dir, 'stimulus_feature', 'recon_image_normalized-'+a_image + ".tiff" ) 
            if os.path.exists(a_filepath):
                recon_true_image_filepath_list.append(a_filepath)
        
        for j in range(stimulus_selection_size):
            img_true_filepath = recon_true_image_filepath_list[j]
            img_true = PIL.Image.open(img_true_filepath)
            img_true = img_true.resize((img_size[0], img_size[1]), PIL.Image.LANCZOS)
            yi = j // nImg_row +1
            xi = j % nImg_row
            x = xi * (img_size[1]+h_mergin)+h_mergin 
            y = yi * (img_size[1]+w_mergin)+w_mergin
            image[ x:(x+img_size[0]), y:(y+img_size[1]), : ] = np.array(img_true)[:,:,:]            
    
    elif drawtype == 'Recon-decoded':

        #------------------------------------
        # draw recon from decoded feature
        #------------------------------------
        for b, sbj in enumerate(sbjs):
            for s in range(n_trial):
                for j in range(stimulus_selection_size):
    
                        
                    recon_dir_sbj = os.path.join(recon_dir, sbj, roi)
    
                    img_recon_filepath = os.path.join(recon_dir_sbj,  
                                      "recon_image_normalized-%s.tiff" % (true_image_id_list[j]+'_trial'+'{:02d}'.format(trials[sbj][j,s])))
                    if os.path.exists(img_recon_filepath):
                        img_recon = PIL.Image.open(img_recon_filepath)
                        img_recon = img_recon.resize((img_size[0], img_size[1]), PIL.Image.LANCZOS)
                        xi = j % nImg_row
                        yi = s + (j // nImg_row) * ncol+2+b*n_trial
                        x = xi * (img_size[1]+h_mergin)+h_mergin
                        y = yi * (img_size[1]+w_mergin)+w_mergin
                        image[ x:(x+img_size[0]), y:(y+img_size[1]), : ] = np.array(img_recon)[:,:,:]
                    else:
                        print("Not foud image:", img_recon_filepath)
                    
                        
save_jpg_file = os.path.join(save_dir, "Fig2.%s" % (save_ext))
save_img(image, save_jpg_file)
    
    
     
