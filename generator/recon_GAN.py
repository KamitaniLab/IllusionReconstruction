'''Feature-to-Generator reconstruction.'''


import glob
import os
from itertools import product

from bdpy.dl.torch.models import AlexNetGenerator
from bdpy.dataform import DecodedFeatures
from bdpy.feature import normalize_feature
from bdpy.util import dump_info
import numpy as np
import PIL.Image
import scipy.io as sio
import torch


# Settings ###################################################################

# GPU settings ---------------------------------------------------------------

device0 = 'cuda:0'
device1 = 'cuda:0'

# Data settings --------------------------------------------------------------

# Target images to be reconstructed
# If None, reconstruct all images.
images = None

image_mean_file = './generator/GAN/ilsvrc_2012_mean.npy'
feature_std_file = './generator/GAN/estimated_cnn_feat_std_bvlc_reference_caffenet_ImgSize_227x227_chwise_dof1.mat'

# Reconstruction data settings
recon_table = [
  
    {
        # List of subjects
        'subjects': ['S1','S2','S3','S4','S5','S7'],

        # List of ROIs
        'rois': ['VC', 'V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA'],

        # Path to decoded feature directoryi
        'decoded_features_dir': os.path.join('./results/feature-decoding',
                                              'Illusion_single_trial',
                                              'derivatives/decoded_features/',
                                              'deeprecon-fmd-mscoco_fmriprep_5rep_500voxel_bvlc_reference_caffenet_allunits_fastl2lir_alpha100',
                                              'decoded_features'),

        # DNN name (decoded features)
        'network': 'caffe/bvlc_reference_caffenet',

        # List of images
        'images': images,

        # Path to output directory
        'output_dir': os.path.join('./results/reconstruction/recon_images/GAN'),
    },
   {
        # List of subjects
        'subjects': ['S6'],

        # List of ROIs
        'rois': ['VC', 'V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA'],

        # Path to decoded feature directoryi
        'decoded_features_dir': os.path.join('./results/feature-decoding',
                                              'Illusion_single_trial',
                                              'derivatives/decoded_features/',
                                              'deepreconS6-fmd-mscoco_fmriprep_5rep_500voxel_bvlc_reference_caffenet_allunits_fastl2lir_alpha100',
                                              'decoded_features'),

        # DNN name (decoded features)
        'network': 'caffe/bvlc_reference_caffenet',

        # List of images
        'images': images,

        # Path to output directory
        'output_dir': os.path.join('./results/reconstruction/recon_images/GAN'),
    },
]

# Generator input features settings ------------------------------------------

# Decoded DNN features as the input of the generator
generator_input_layer = 'fc6'


# crop generated image
crop_size = (227, 227)

# Network settings -----------------------------------------------------------

generator_dir_root = './generator/GAN/'

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
    'output_size':   (256,256)
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

    # Initialize CNN ---------------------------------------------------------

    # Average image of ImageNet
    img_mean = np.load(image_mean_file)
    img_mean = np.mean(img_mean, axis=(1, 2)).astype(np.float32)

    # crop
    l0 = (generator_conf['output_size'][0]-crop_size[0])//2
    l1 = (generator_conf['output_size'][1]-crop_size[1])//2
    u0 = (generator_conf['output_size'][0]+crop_size[0])//2
    u1 = (generator_conf['output_size'][1]+crop_size[1])//2

    # Generator network
    net_gen = AlexNetGenerator(device=(device0, device1))
    net_gen.load_state_dict(torch.load(generator_conf['parameters']))

    # Input/output layer for generator net
    input_layer_gen = generator_conf['input_layer']
    output_layer_gen = generator_conf['output_layer']

    # Feature SD estimated from true CNN features of 10000 images
    feat_std0 = sio.loadmat(feature_std_file)

    # Setup results directory ------------------------------------------------
    save_dir_root = recon['output_dir']
    if not os.path.exists(save_dir_root):
        os.makedirs(save_dir_root)

    # Save runtime information -----------------------------------------------
    dump_info(save_dir_root, script=__file__)

    # Reconstrucion ----------------------------------------------------------
    subjects_list = recon['subjects']
    rois_list = recon['rois']

    decoded_feature_root_dir = recon['decoded_features_dir']
    network = recon['network']
    images_list = recon['images']

    # Load the decoded CNN features
    decfeat = DecodedFeatures(os.path.join(decoded_feature_root_dir, network), squeeze=True)

    for subject, roi in product(subjects_list, rois_list):
        print('----------------------------------------')
        print('Subject: ' + subject)
        print('ROI:     ' + roi)
        print('')

        save_dir = os.path.join(save_dir_root, subject, roi)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get images if images_list is None
        if images_list is None:
            matfiles = glob.glob(os.path.join(decoded_feature_root_dir, network, generator_input_layer, subject, roi, '*.mat'))
            images_list0 = [os.path.splitext(os.path.basename(fl))[0] for fl in matfiles]
            
        # Images loop
        for image_label in images_list0:
            print('Image: ' + image_label)

            #--------------------------
            # Generator input features
            #--------------------------
            gen_input_feat = decfeat.get(layer=generator_input_layer, subject=subject, roi=roi, image=image_label)

            gen_input_feat = normalize_feature(gen_input_feat,
                                               channel_wise_mean=True, channel_wise_std=True,
                                               channel_axis=channel_axis,
                                               shift='self', scale=feat_std0[generator_input_layer],
                                               std_ddof=std_ddof)
            #----------------
            # Reconstruction
            #----------------
            recon_img_tensor = net_gen(torch.tensor(gen_input_feat).to(device0))
            recon_img = recon_img_tensor.cpu().detach().numpy()
            recon_img = recon_img[0, :, l0:u0, l1:u1]
            recon_img = img_deprocess(recon_img, img_mean)
            
            #------------------
            # Save the results
            #------------------

            # Save the raw reconstructed image
            recon_image_mat_file = os.path.join(save_dir, 'recon_image' + '-' + image_label + '.mat')
            sio.savemat(recon_image_mat_file, {'recon_image': recon_img})

            # To better display the image, clip pixels with extreme values (0.02% of
            # pixels with extreme low values and 0.02% of the pixels with extreme high
            # values). And then normalise the image by mapping the pixel value to be
            # within [0,255].
            recon_image_normalized_file = os.path.join(save_dir, 'recon_image_normalized' + '-' + image_label + '.tiff')
            PIL.Image.fromarray(normalise_img(clip_extreme_value(recon_img, pct=0.04))).save(recon_image_normalized_file)

    print('Done')

print('All done')
