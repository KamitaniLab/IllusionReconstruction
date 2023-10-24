'''DNN Feature decoding - decoders training script'''


from __future__ import print_function

from itertools import product
import os
from time import time
import warnings
import argparse

import bdpy
from bdpy.dataform import Features, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTraining
from bdpy.util import makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np
import yaml


# Main #######################################################################

def main(conf):

     # Settings ###############################################################

     # Brain data
     subjects_list = conf['training fmri']
     label_key = conf['label key']
     num_training_samples = conf['num_training_samples']
     rois_list = conf['rois']
     num_voxel = conf['rois voxel num']

     # Image features
     features_dir_list = conf['training feature dir']
     network = conf['network']
     features_list = conf['layers']
     features_list = features_list[::-1]  # Start training from deep layers


     # Model parameters
     alpha = conf['alpha']

     # Results directory
     results_dir_root = os.path.join(conf['feature decoder dir'], conf['analysis name'])

     # Misc settings
     chunk_axis = conf['chunk axis']
     # If Y.ndim >= 3, Y is divided into chunks along `chunk_axis`.
     # Note that Y[0] should be sample dimension.


     # Main ###################################################################

     analysis_basename = os.path.splitext(os.path.basename(__file__))[0] + '-' + conf['analysis name']

     # Print info -------------------------------------------------------------
     print('Subjects:         %s' % list(subjects_list.keys()))
     print('ROIs:               %s' % list(rois_list.keys()))
     print('Target features: %s' % network)
     print('Layers:            %s' % features_list)
     print('')

     # Load data --------------------------------------------------------------
     print('----------------------------------------')
     print('Loading data')

     data_brain = {}
     
     data_features_list = []
     for features_dir in features_dir_list:
         print('Load', os.path.join(features_dir, network))
         data_features = Features(os.path.join(features_dir, network))
         data_features_list.append(data_features)
     

     # Initialize directories -------------------------------------------------
     makedir_ifnot(results_dir_root)
     makedir_ifnot(os.path.join(results_dir_root, network))
     makedir_ifnot('tmp')

     # Analysis loop ----------------------------------------------------------
     print('----------------------------------------')
     print('Analysis loop')

     for feat, sbj, roi in product(features_list, subjects_list, rois_list):
         print('--------------------')
         print('Feature:     %s' % feat)
         print('Subject:     %s' % sbj)
         print('ROI:         %s' % roi)
         print('Num voxels: %d' % num_voxel[roi])

         # Setup
         # -----
         analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
         results_dir = os.path.join(results_dir_root, network, feat, sbj, roi, 'model')
         makedir_ifnot(results_dir)

         # Check whether the analysis has been done or not.
         info_file = os.path.join(results_dir, 'info.yaml')
         if os.path.exists(info_file):
               with open(info_file, 'r') as f:
                   info = yaml.safe_load(f)
               while info is None:
                   warnings.warn('Failed to load info from %s. Retrying...'
                                    % info_file)
                   with open(info_file, 'r') as f:
                        info = yaml.safe_load(f)
               if '_status' in info and 'computation_status' in info['_status']:
                   if info['_status']['computation_status'] == 'done':
                        print('%s is already done and skipped' % analysis_id)
                        continue

         # Preparing data
         # --------------
         print('Preparing data')

         start_time = time()
     
         # Load brain data
         print('------------------------------------------------------')
         print('Preparing brain data')
         if sbj not in data_brain.keys():
               bdata_list = []
               for dat_file in subjects_list[sbj]:
                   print('Load', dat_file)
                   bdata_list.append(bdpy.BData(dat_file))
                   data_brain[sbj] = bdata_list
         else:
               print('Already loaded...skip')

         # Brain data
         x_list=[]; x_labels_list = [];
         for i, bdobj in enumerate(data_brain[sbj]):
             x_list.append(bdobj.select(rois_list[roi])[:num_training_samples[i], :])    # Brain data
             x_labels_list.append(bdobj.get_label(label_key)[:num_training_samples[i]])  # Labels
     
         x = np.vstack(x_list)
         x_labels = np.hstack(x_labels_list)
         print("x:", x.shape)
         print("x_labels:", x_labels.shape)

         # Target features and image labels (file names)
         y_list=[]; y_labels_list=[];
         for i, data_features in enumerate(data_features_list):
               y_list.append(data_features.get_features(feat))  # Target DNN features
               y_labels_list.append(data_features.labels)        # Labels
              
         y = np.vstack(y_list)
         y_labels = np.hstack(y_labels_list)
         print("y:", y.shape)
         print("y_labels", y_labels.shape)
         del y_list, y_labels_list
    
         print('Elapsed time (data preparation): %f' % (time() - start_time))


         # Calculate normalization parameters
         # ----------------------------------

         # Normalize X (fMRI data)
         x_mean = np.mean(x, axis=0)[np.newaxis, :]  # np.newaxis was added to match Matlab outputs
         x_norm = np.std(x, axis=0, ddof=1)[np.newaxis, :]

         # Normalize Y (DNN features)
         y_mean = np.mean(y, axis=0)[np.newaxis, :]
         y_norm = np.std(y, axis=0, ddof=1)[np.newaxis, :]

         # Y index to sort Y by X (matching samples)
         # -----------------------------------------
         y_index = np.array([np.where(np.array(y_labels) == xl) for xl in x_labels]).flatten()

         # Save normalization parameters
         # -----------------------------
         print('Saving normalization parameters.')
         norm_param = {'x_mean': x_mean, 'y_mean': y_mean,
                          'x_norm': x_norm, 'y_norm': y_norm}
         save_targets = [u'x_mean', u'y_mean', u'x_norm', u'y_norm']
         for sv in save_targets:
               save_file = os.path.join(results_dir, sv + '.mat')
               if not os.path.exists(save_file):
                   try:
                        save_array(save_file, norm_param[sv], key=sv, dtype=np.float32, sparse=False)
                        print('Saved %s' % save_file)
                   except Exception:
                        warnings.warn('Failed to save %s. Possibly double running.' % save_file)

         # Preparing learning
         # ------------------
         model = FastL2LiR()
         model_param = {'alpha':  alpha,
                           'n_feat': num_voxel[roi]}

         # Distributed computation setup
         # -----------------------------
         makedir_ifnot('./tmp')
         distcomp_db = os.path.join('./tmp', analysis_basename + '.db')
         distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)

         # Model training
         # --------------
         print('Model training')
         start_time = time()

         train = ModelTraining(model, x, y)
         train.id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
         train.model_parameters = model_param

         train.X_normalize = {'mean': x_mean,
                                  'std': x_norm}
         train.Y_normalize = {'mean': y_mean,
                                  'std': y_norm}
         train.Y_sort = {'index': y_index}

         train.dtype = np.float32
         train.chunk_axis = chunk_axis
         train.save_format = 'bdmodel'
         train.save_path = results_dir
         train.distcomp = distcomp

         train.run()

         print('Total elapsed time (model training): %f' % (time() - start_time))

     print('%s finished.' % analysis_basename)


# Entry point ################################################################

if __name__ == '__main__':
     parser = argparse.ArgumentParser()
     parser.add_argument(
         'conf',
         type=str,
         help='analysis configuration file',
     )
     args = parser.parse_args()

     conf_file = args.conf

     with open(conf_file, 'r') as f:
         conf = yaml.safe_load(f)

     main(conf)
