#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:53:55 2023

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
sbjs = ['S1']


# Target images
images = [ 
            'fillingin004_large_unionjack_lumi0p3_sat0p8_connected',
            'fillingin005_large_unionjack_lumi0p3_sat0p8_disconnected', 
           ]



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
recon_dir = './results/reconstruction/recon_images/diffusion'


# trial number
n_trial = 4
trials = {'S1': np.array([
                    [3,7,8,10], [3,7,8,10]                   
                    ])}
# seed
seeds = [0,1,7,30,64, 78, 222, 666, 1111, 2022, 2023]
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
 

    
#------------------------------------
# draw recon image from decoded features
#------------------------------------
for sbj in sbjs:
    ncol = len(seeds)
    nImg_col = ncol
    nImg_row = nImg_row*n_trial 
    size_x = (img_size[0]+h_mergin)*nImg_row+h_mergin
    size_y = (img_size[1]+w_mergin)*nImg_col+w_mergin
    image = np.ones( (size_x, size_y, 3) )*255
    for s in range(len(seeds)):
        for j in range(stimulus_selection_size):
            for t in range(n_trial):
                j1 = j*n_trial+t
                recon_dir_sbj = os.path.join(recon_dir, sbj, roi)
                img_recon_filepath = os.path.join(recon_dir_sbj, 
                                                  "recon_image_normalized-%s.tiff" % (true_image_id_list[j]+'_trial'+str(trials[sbj][j,t]).zfill(2)+'-seed'+str(seeds[s])))
                if os.path.exists(img_recon_filepath):
                    img_recon = PIL.Image.open(img_recon_filepath)
                    img_recon = img_recon.resize((img_size[0], img_size[1]), PIL.Image.LANCZOS)
                    xi = j1 % nImg_row
                    yi = s + (j1 // nImg_row) * ncol
                    x = xi * (img_size[1]+h_mergin)+h_mergin
                    y = yi * (img_size[1]+w_mergin)+w_mergin
                    image[ x:(x+img_size[0]), y:(y+img_size[1]), : ] = np.array(img_recon)[:,:,:]
                else:
                    print("Not foud image:", img_recon_filepath)
                   
                
                    
                        
save_jpg_file = os.path.join(save_dir, "figsupp_diffusion.%s" % (save_ext))
save_img(image, save_jpg_file)
    
    
     
