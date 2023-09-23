#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 19:17:40 2022

@author: fcheng
"""

'''Feature-to-Generator reconstruction.'''


import glob
import os
import sys

from bdpy.dl.torch.models import AlexNetGenerator
from bdpy.dataform import Features, DecodedFeatures
from bdpy.feature import normalize_feature
from bdpy.util import dump_info, average_elemwise
import numpy as np
import PIL.Image
import scipy.io as sio
import torch


import PIL
from PIL import ImageDraw
from PIL import ImageFont


def recon_fg_from_manipulateDNN(modify_list, unit_pos_list, level_list, 
                                decoded_features_dir, net, savdir_name, images=None, 
                                generator_input_layer='relu6', relu_normalization = None):
# Settings ###################################################################

    # GPU settings ---------------------------------------------------------------
    
    device0 = 'cuda:0'
    device1 = 'cuda:0'
    # Data settings --------------------------------------------------------------

    image_mean_file = '/home/mu/data/models_shared/caffe/bvlc_reference_caffenet/ilsvrc_2012_mean.npy'
    feature_std_file = '/home/mu/data/models_shared/caffe/bvlc_reference_caffenet/estimated_feat_std/estimated_cnn_feat_std_bvlc_reference_caffenet_ImgSize_227x227_chwise_dof1.mat'

    # Reconstruction data settings
    recon_table = [
        {
            'decoded_features_dir': decoded_features_dir,
            'network': net,
            'images': images,
            'output_dir': os.path.join('../results/recon_images', savdir_name),
        },
    ]

    # Generator input features settings ------------------------------------------

    # Network settings -----------------------------------------------------------
    
    generator_dir_root = '/home/mu/data/models_shared/pytorch/bvlc_reference_caffenet/generators/ILSVRC2012_Training/'+generator_input_layer

    # Delta degrees of freedom when calculating SD
    # This should be match to the DDoF used in calculating
    # SD of true DNN features (`feat_std0`)
    std_ddof = 1
    
    # Axis for channel in the DNN feature array
    channel_axis = 0
    
    # Generator
    generator_conf = {
        'parameters':      os.path.join(generator_dir_root, 'generator.pt'),
        'input_layer':     'feat',       # Input layer for generator net
        'output_layer':    'generated',  # Output layer for generator net
        'act_range_shape': 'unit',       # {'unit', 'channel'}
    }


    # Functions ##################################################################
    
    def clip_extreme_value(img, pct=1):
        '''clip the pixels with extreme values'''
        if pct < 0:
            pct = 0.
    
        if pct > 100:
            pct = 100.
    
        img = np.clip(img, np.percentile(img, pct/2.),
                      np.percentile(img, 100-pct/2.))
        return img
    
    
    def normalise_img(img):
        '''Normalize the image.
        Map the minimum pixel to 0; map the maximum pixel to 255.
        Convert the pixels to be int
        '''
        img = img - img.min()
        if img.max() > 0:
            img = img * (255.0/img.max())
        img = np.uint8(img)
        return img
    
    
    def img_deprocess(img, img_mean=np.float32([104, 117, 123])):
        '''convert from Caffe's input image layout'''
        return np.dstack((img + np.reshape(img_mean, (3, 1, 1)))[::-1])


    # Main #######################################################################
    
    for recon in recon_table:
    
        # Initialize models-------------------------------------------------------
    
        # Average image of ImageNet
        img_mean = np.load(image_mean_file)
        img_mean = np.mean(img_mean, axis=(1, 2)).astype(np.float32)
    
        # Generator network
        net_gen = AlexNetGenerator(device=(device0, device1))
        net_gen.load_state_dict(torch.load(generator_conf['parameters']))
    
    
        # Feature SD estimated from true CNN features of 10000 images
        feat_std0 = sio.loadmat(feature_std_file)
    
        # Setup results directory ------------------------------------------------
        save_dir_root = recon['output_dir']
        if not os.path.exists(save_dir_root):
            os.makedirs(save_dir_root)
    
        # Save runtime information -----------------------------------------------
        dump_info(save_dir_root, script=__file__)
    
        # Reconstrucion ----------------------------------------------------------
        decoded_feature_root_dir = recon['decoded_features_dir']
        network = recon['network']
        images_list = recon['images']
    
        # Load the decoded CNN features
        decfeat = Features(os.path.join(decoded_feature_root_dir, network))
    
        save_dir = save_dir_root
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
        # Get images if images_list is None
        if images_list is None:
            matfiles = glob.glob(os.path.join(decoded_feature_root_dir, network, generator_input_layer, '*.mat'))
            images_list = [os.path.splitext(os.path.basename(fl))[0] for fl in matfiles]
    
        # Images loop
        for i, image_label in enumerate(images_list):
            print('Image: ' + image_label)
    
            #--------------------------
            # Generator input features
            #--------------------------
            gen_input_feat = decfeat.get_features(layer=generator_input_layer)[i, :]
    
            # Normalization with ReLU features
            if not relu_normalization is None:
                gen_input_feat = np.maximum(gen_input_feat, 0)
                gen_input_feat = normalize_feature(gen_input_feat,
                                                   channel_wise_mean=True, channel_wise_std=True,
                                                   channel_axis=channel_axis,
                                                   shift='self', scale=feat_std0[relu_normalization],
                                                   std_ddof=std_ddof)
            
            for u, unit_pos in enumerate(unit_pos_list):
                
                for lev in level_list:
                    gen_input = gen_input_feat.copy()
                    gen_input[unit_pos] = gen_input[unit_pos]*lev
                    
                    #----------------
                    # Reconstruction
                    #----------------
                    recon_img_tensor = net_gen(torch.tensor(gen_input).to(device0))
                    recon_img = recon_img_tensor.cpu().detach().numpy()
                    recon_img = recon_img[0, :, :, :]
                    recon_img = img_deprocess(recon_img, img_mean)
            
                    #------------------
                    # Save the results
                    # To better display the image, clip pixels with extreme values (0.02% of
                    # pixels with extreme low values and 0.02% of the pixels with extreme high
                    # values). And then normalise the image by mapping the pixel value to be
                    # within [0,255].
                    recon_image_normalized_file = os.path.join(save_dir, 'recon_image_normalized' + '-' + image_label +\
                                                               '_'+str(modify_list[u])+'-'+str(lev).replace('.','p')+'.tiff')
                    PIL.Image.fromarray(normalise_img(clip_extreme_value(recon_img, pct=0.04))).save(recon_image_normalized_file)
                

    
    print('All done')
    

def visualize_recon_from_manipulateDNN(recon_folder, condition_list, savdir_name, save_title, save_ext, images=None):

    save_dir = os.path.join("../plot/recon_images", savdir_name)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # true images
    image_dataset = 'IllusNat210312'
    stimuli_dir_root = '/home/nu/fcheng/illusion_python'
    source_image_dir = os.path.join(stimuli_dir_root, image_dataset, 'Image')
    source_image_ext = "tif"


    # recon from decoded features
    recon_dir = os.path.join('../results/recon_images', recon_folder)

    # Get true image set
    if images is None:
        true_image_filepath_list = glob.glob(os.path.join(source_image_dir,"*." + source_image_ext))
    else:
        true_image_filepath_list = []
        for a_image in images:
            a_filepath = os.path.join(source_image_dir, a_image + "." + source_image_ext) 
            if os.path.exists(a_filepath):
                true_image_filepath_list.append(a_filepath)
     
        
    #true_image_filepath_list = sorted(true_image_filepath_list)
    true_image_id_list = [os.path.basename(t).split(".")[0] for t in true_image_filepath_list]
    
    print("True image num:", len(true_image_filepath_list))
    print("True image:", true_image_filepath_list)
        
    ### settings ##############################
    
    # position, image size settings
    row_size = len(images)
    h_mergin = 2
    w_mergin = 2
    img_size = (80, 80)
    
    
    ### make figure ##############################
    #------------------------------------
    # make canvas
    #------------------------------------
    stimulus_selection_size = len(true_image_id_list)
    turn_num = int(np.ceil(stimulus_selection_size/float(row_size)))
    add = 1
    ncol = len(condition_list) + add
    nImg_col = ncol * turn_num
    
    nImg_row = row_size 
    size_x = (img_size[0]+h_mergin)*nImg_row
    size_y = (img_size[1]+w_mergin)*nImg_col
    image = np.zeros( (size_x, size_y, 3) )
    
    #------------------------------------
    # draw true image
    #------------------------------------
    for j in range(stimulus_selection_size):
        img_true_filepath = true_image_filepath_list[j]
        img_true = PIL.Image.open(img_true_filepath)
        img_true = img_true.resize((img_size[0], img_size[1]), PIL.Image.LANCZOS)

        xi = j % nImg_row
        yi = (j // nImg_row) * ncol
        x = xi * (img_size[1]+h_mergin) 
        y = yi * (img_size[1]+w_mergin)

        image[ x:(x+img_size[0]), y:(y+img_size[1]), : ]
        np.array(img_true)
        np.array(img_true)[:,:,:]
        image[ x:(x+img_size[0]), y:(y+img_size[1]), : ] = np.array(img_true)[:,:,:]
    
    
    #------------------------------------
    # draw recon image from decoded features
    #------------------------------------
    for s, condition in enumerate(condition_list):
        cond = condition[0]
        lev = condition[1]
    
        for j in range(stimulus_selection_size):
            img_recon_filepath = os.path.join(recon_dir, 
                                              "recon_image_normalized-%s.tiff" % (true_image_id_list[j]+\
                                                                                  '_'+str(cond)+'-'+str(lev).replace('.','p')))
            if os.path.exists(img_recon_filepath):
                img_recon = PIL.Image.open(img_recon_filepath)
                img_recon = img_recon.resize((img_size[0], img_size[1]), PIL.Image.LANCZOS)
                xi = j % nImg_row
                yi = s + add + (j // nImg_row) * ncol
                x = xi * (img_size[1]+h_mergin) 
                y = yi * (img_size[1]+w_mergin) 
                image[ x:(x+img_size[0]), y:(y+img_size[1]), : ] = np.array(img_recon)[:,:,:]
            else:
                print("Not foud image:", img_recon_filepath)
    # cast to int
    image = image.astype('uint8')
    
    
    #------------------------------------
    # For printing characters
    #------------------------------------
    # convert ndarray to image object
    image_obj = PIL.Image.fromarray(image)
    #draw = PIL.ImageDraw.Draw(image_obj)
    
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_jpg_file = os.path.join(save_dir, "recon_image_%s.%s" % (save_title, save_ext))
    
    # Save
    image_obj.save(save_jpg_file, quality=98)
    print("Save:", save_jpg_file)
        

    
# images =  ['abuttingLine001_linespace_1p2_angle_000_045diag',
#            #'abuttingLine002_linespace_1p2_angle_000_045diag_pos',
#            'abuttingLine003_linespace_1p2_angle_000_135diag',
#            #'abuttingLine004_linespace_1p2_angle_000_135diag_pos',
#            'abuttingLine005_linespace_1p2_angle_000_vline',
#            #'abuttingLine006_linespace_1p2_angle_000_vline_pos',
#            'abuttingLine007_linespace_1p2_angle_090_045diag',
#            #'abuttingLine008_linespace_1p2_angle_090_045diag_pos',
#            'abuttingLine009_linespace_1p2_angle_090_135diag',
#            #'abuttingLine010_linespace_1p2_angle_090_135diag_pos',
#            'abuttingLine011_linespace_1p2_angle_090_hline'
#            #'abuttingLine012_linespace_1p2_angle_090_hline_pos',
#            #'abuttingLine013_linespace_2p4_angle_000_vline',
#            #'abuttingLine014_linespace_2p4_angle_000_vline_pos',
#            #'abuttingLine015_linespace_2p4_angle_090_hline',
#            #'abuttingLine016_linespace_2p4_angle_090_hline_pos',
#            #'abuttingLine017_linespace_6_angle_000_vline',
#            #'abuttingLine018_linespace_6_angle_000_vline_pos',
#            #'abuttingLine019_linespace_6_angle_090_hline',
#            #'abuttingLine020_linespace_6_angle_090_hline_pos'
#            ]
    
    
# 'fillingin001_small_unionjack_lumi0p3_sat0p8_connected',
# 'fillingin002_small_unionjack_lumi0p3_sat0p8_disconnected',
# 'fillingin003_small_unionjack_lumi0p3_sat0p8_uniform',
# 'fillingin004_large_unionjack_lumi0p3_sat0p8_connected',
# 'fillingin005_large_unionjack_lumi0p3_sat0p8_disconnected',
# 'fillingin006_large_unionjack_lumi0p3_sat0p8_uniform',   
# 'fillingin007_small_cross_lumi0p3_sat0p7_connected',
# 'fillingin008_small_cross_lumi0p3_sat0p7_disconnected',
# 'fillingin009_small_cross_lumi0p3_sat0p7_uniform',
# 'fillingin010_large_cross_lumi0p3_sat0p7_connected',
# 'fillingin011_large_cross_lumi0p3_sat0p7_disconnected',
# 'fillingin012_large_cross_lumi0p3_sat0p7_uniform',    
 # 'neonVarinImg001_IllusSurf_alpha_0p7_sat_0p3_bgray',     
 # 'neonVarinImg002_innerKanizsa_alpha_0p7_sat_0p3_bgray',
 # 'neonVarinImg003_Kanizsa_alpha_0p7_sat_0p3_bgray',
 # 'neonVarinImg004_Disk_alpha_0p7_sat_0p3_bgray',     
 # 'neonVarinImg005_realSurf_alpha_0p7_sat_0p3_bgray',
 # 'neonVarinImg006_realRect_alpha_0p7_sat_0p3_bgray',  