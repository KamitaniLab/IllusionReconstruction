'''DNN Feature decoding - decoders test (prediction) script'''


from __future__ import print_function

import glob
from itertools import product
import os
from time import time
import argparse

import bdpy
from bdpy.dataform import Features, load_array, save_array
from bdpy.distcomp import DistComp
from bdpy.ml import ModelTest
from bdpy.util import get_refdata, makedir_ifnot
from fastl2lir import FastL2LiR
import numpy as np
import yaml


# Main #######################################################################

def main(conf):

    # Settings ###############################################################

    # Brain data
    subjects_list = conf['test fmri']
    label_key = conf['label key']
    rois_list = conf['rois']
    num_voxel = conf['rois voxel num']

    # Image features
    if 'test feature dir' in conf:
        features_dir = conf['test feature dir'][0]
    network = conf['network']
    features_list = conf['layers']
    features_list = features_list[::-1]  # Start training from deep layers

    if 'feature index file' in conf:
        feature_index_file = os.path.join(conf['training feature dir'][0], network, conf['feature index file'])
    else:
        feature_index_file = None

    # Model parameters
    alpha = conf['alpha']

    # Trained models
    models_dir_root = os.path.join(conf['feature decoder dir'], conf['analysis name'])

    # Results directory
    results_dir_root = os.path.join(conf['decoded feature dir'], conf['analysis name'])

    # Misc settings
    chunk_axis = conf['chunk axis']
    # If Y.ndim >= 3, Y is divided into chunks along `chunk_axis`.
    # Note that Y[0] should be sample dimension.


    # Main ###################################################################

    analysis_basename = os.path.splitext(os.path.basename(__file__))[0]

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_brain = {sbj: bdpy.BData(dat_file[0])
                  for sbj, dat_file in subjects_list.items()}
    if 'test feature dir' in conf:     
        if feature_index_file is not None:
            data_features = Features(os.path.join(features_dir, network), feature_index=feature_index_file)
        else:
            data_features = Features(os.path.join(features_dir, network))

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir_root)
    makedir_ifnot(os.path.join(results_dir_root, 'decoded_features', network))
    makedir_ifnot(os.path.join(results_dir_root, 'profile_correlation', network))
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for feat, sbj, roi in product(features_list, subjects_list, rois_list):
        print('--------------------')
        print('Feature:    %s' % feat)
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)

        # Distributed computation setup
        # -----------------------------
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_dir_prediction = os.path.join(results_dir_root, 'decoded_features', network, feat, sbj, roi)
        if 'test feature dir' in conf:
            results_dir_accuracy = os.path.join(results_dir_root, 'profile_correlation', network, feat, sbj, roi)

        if os.path.exists(results_dir_prediction):
            print('%s is already done. Skipped.' % analysis_id)
            continue

        distcomp_db = os.path.join('./tmp', analysis_basename + '.db')
        distcomp = DistComp(backend='sqlite3', db_path=distcomp_db)
        if not distcomp.lock(analysis_id):
            print('%s is already running. Skipped.' % analysis_id)
            continue

        # Preparing data
        # --------------
        print('Preparing data')

        start_time = time()

        # Brain data
        x = data_brain[sbj].select(rois_list[roi])       # Brain data
        x_labels = data_brain[sbj].get_label(label_key)  # Labels

        # Target features and image labels
        if 'test feature dir' in conf:
            y = data_features.get_features(feat)  # Target DNN features
            y_labels = data_features.labels       # Labels
        
        
        if 'single_trial' in results_dir_root:
            # add trial number to labels
            x_labels_unique, counts = np.unique(x_labels, return_counts=True)
            x_label_dict = dict(zip(x_labels_unique, counts))
            x_labels_pred = []
            for label in x_labels:
                trial_num = x_label_dict[label]
                x_labels_pred.append(label+'_trial'+str(trial_num).zfill(2))
                x_label_dict[label]-=1
        elif 'avg_trials' in results_dir_root:
            # Averaging brain data
            x_labels_unique, counts = np.unique(x_labels, return_counts=True)
            num_rep = min(counts)        
            print('Average %d repetitions' % num_rep)
            x = np.vstack([np.mean(x[(np.array(x_labels) == lb).flatten(), :][:num_rep, :], axis=0) for lb in x_labels_unique])
            x_labels_pred = x_labels_unique
 

        print('Elapsed time (data preparation): %f' % (time() - start_time))

        # Save feature index
        # ------------------
        if feature_index_file is not None:
            feature_index_save_file = os.path.join(results_dir_prediction, 'feature_index.mat')
            data_features.save_feature_index(feature_index_save_file)
            print('Saved %s' % feature_index_file)
            feature_index_save_file = os.path.join(results_dir_accuracy, 'feature_index.mat')
            data_features.save_feature_index(feature_index_save_file)
            print('Saved %s' % feature_index_file)

        # Model directory
        # ---------------
        model_dir = os.path.join(models_dir_root, network, feat, sbj, roi, 'model')

        # Preprocessing
        # -------------
        x_mean = load_array(os.path.join(model_dir, 'x_mean.mat'), key='x_mean')  # shape = (1, n_voxels)
        x_norm = load_array(os.path.join(model_dir, 'x_norm.mat'), key='x_norm')  # shape = (1, n_voxels)
        y_mean = load_array(os.path.join(model_dir, 'y_mean.mat'), key='y_mean')  # shape = (1, shape_features)
        y_norm = load_array(os.path.join(model_dir, 'y_norm.mat'), key='y_norm')  # shape = (1, shape_features)

        x = (x - x_mean) / x_norm

        # Prediction
        # ----------
        print('Prediction')

        start_time = time()

        model = FastL2LiR()

        test = ModelTest(model, x)
        test.model_format = 'bdmodel'
        test.model_path = model_dir
        test.dtype = np.float32
        test.chunk_axis = chunk_axis

        y_pred = test.run()

        print('Total elapsed time (prediction): %f' % (time() - start_time))

        # Postprocessing
        # --------------
        y_pred = y_pred * y_norm + y_mean

        if 'test feature dir' in conf:
            # Calculate prediction accuracy
            # -----------------------------
            print('Prediction accuracy')

            start_time = time()

            y_pred_2d = y_pred.reshape([y_pred.shape[0], -1])
            y_true_2d = y.reshape([y.shape[0], -1])

            y_true_2d = get_refdata(y_true_2d, np.array(y_labels), x_labels_unique)

            n_units = y_true_2d.shape[1]

            accuracy = np.array([np.corrcoef(y_pred_2d[:, i].flatten(), y_true_2d[:, i].flatten())[0, 1] for i in range(n_units)])
            accuracy = accuracy.reshape((1,) + y_pred.shape[1:])

            print('Total elapsed time (prediction accuracy): %f' % (time() - start_time))

        # Save results
        # ------------
        print('Saving results')

        start_time = time()

        makedir_ifnot(results_dir_prediction)
        # Predicted features
        for i, label in enumerate(x_labels_pred):
            # Predicted features
            feat = np.array([y_pred[i,]])  # To make feat shape 1 x M x N x ...

            # Save file name
            save_file = os.path.join(results_dir_prediction, '%s.mat' % label)

            # Save
            save_array(save_file, feat, key='feat', dtype=np.float32, sparse=False)

        print('Saved %s' % results_dir_prediction)

        if 'test feature dir' in conf:
            makedir_ifnot(results_dir_accuracy)
            # Prediction accuracy
            save_file = os.path.join(results_dir_accuracy, 'accuracy.mat')
            save_array(save_file, accuracy, key='accuracy', dtype=np.float32, sparse=False)
            print('Saved %s' % save_file)

        print('Elapsed time (saving results): %f' % (time() - start_time))

        distcomp.unlock(analysis_id)

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
