#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 21:03:49 2021

@author: fcheng
"""
import numpy as np
from itertools import product

def CalSimilarityIndex(image, maskinfo):
    """
    calculate similarity index for image, given two masks
    Output:
        Index - a float number, positive if image is more similar to mask1
    """
    
    # read mask information
    mask1 = maskinfo[0]
    mask2 = maskinfo[1]
    d = maskinfo[2]
    
    # read image
    
    Index = (np.corrcoef(image.flatten(), mask2.flatten())[0,1] - np.corrcoef(image.flatten(), mask1.flatten())[0,1])/d
    
    return Index




def makeDataFrame4pooledSubjects(input_df, group_key):
    
    df = input_df
    
    col_names = df.columns
    n_col = len(col_names)
    
    # add mean as rows
    sbjs = df.Subject.unique()[1:]
    rois = df.ROI.unique()[1:]
    reconType_list = df.reconType.unique()
    
    group_list = df[group_key].unique()
    
    for reconType in reconType_list:
        if reconType == 'Recon-decoded features':
            for group, sbj, roi in product(group_list, sbjs, rois):
                tmp_df = df[(df.reconType == reconType)&(df[group_key]==group)&(df.Subject==sbj)&(df.ROI==roi)]
                row_values = ['None']*n_col
                for j,col in enumerate(col_names):
                    if 'Index' in col:
                        row_values[j] = tmp_df[col].mean()
                        if np.isnan(row_values[j]):
                            print(sbj, roi, reconType)
                    elif col == 'ROI':
                        row_values[j] = roi
                    elif col == 'Subject':
                        row_values[j] = sbj
                    elif col == group_key:
                        row_values[j] = group
                    elif col == 'stimName':
                        row_values[j] = 'mean '+group_key
                    elif col == 'reconType':
                        row_values[j] = reconType
                        
                new_row = {'Index of contour&color': row_values[0],
                           'Index of color': row_values[1],'Index of contour': row_values[2],
                           'ROI':row_values[3], 'Subject':row_values[4],'Trial': row_values[5], 'stimName':row_values[6],
                           'stimType':row_values[7], 'reconType':row_values[8], 'Size':row_values[9], 'Pattern':row_values[10]
                           }
                df = df.append(new_row, ignore_index=True)
        else: 
            for group in group_list:
                tmp_df = df[(df.reconType == reconType)&(df[group_key]==group)]
                row_values = ['None']*n_col
                for j,col in enumerate(col_names):
                    if 'Index' in col:
                        row_values[j] = tmp_df[col].mean()
                        if np.isnan(row_values[j]):
                            print(sbj, roi, reconType)
                    elif col == group_key:
                        row_values[j] = group
                    elif col == 'stimName':
                        row_values[j] = 'mean '+group_key
                    elif col == 'reconType':
                        row_values[j] = reconType
                        
                new_row = {'Index of contour&color': row_values[0],
                           'Index of color': row_values[1],'Index of contour': row_values[2],
                           'ROI':row_values[3], 'Subject':row_values[4],'Trial': row_values[5], 'stimName':row_values[6],
                           'stimType':row_values[7], 'reconType':row_values[8], 'Size':row_values[9], 'Pattern':row_values[10]
                           }
                df = df.append(new_row, ignore_index=True)
                
    return df
    