#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:43:08 2022

@author: fcheng
"""

import numpy as np
import pandas as pd
from itertools import product


    
def makeDataFrame_diff_orientation_1stStage(df, illusion_type, plotdict):

    x = []
    pct_illus = []
    pct_inducer = []
    pct_all = []
    recon_types = []
    Subject = []
    ROI = []
    Illusory_orient = []
    Line_space = []
   
    keys = plotdict.keys()
    for key in keys:
       
        condition = plotdict[key]
        reconType = condition[0]
        roi = condition[1]
        sbj = condition[2]
        line_space_list = condition[3]
        illus_orient_list = condition[4]
        
        if sbj != 'Subjects pooled':
            df2 = df[(df.ROI==roi)&(df.Subject==sbj)&(df.stimType==illusion_type)&(df.reconType==reconType)]
        else:
            df2 = df[(df.ROI==roi)&(df.stimType==illusion_type)&(df.reconType==reconType)]

        for illus_orient,line_space in product(illus_orient_list, line_space_list): 
        #for line_space in line_space_list:

            tmp = df2[(df2['Illusory orientation'].isin(illus_orient))&(df2['Line space']==line_space)]
            #tmp = df2[(df2['Line space']==line_space)]
            t = tmp['Orientation category'].value_counts()
            pct = 100*t/t.sum()
            pct = pct.reindex(['Illusory', 'Inducer'], fill_value=0)
            #pct = pct.reindex(['Illusory', 'Inducer',  'Not clear'], fill_value=0)
           
            pct_illus.append(100*pct[0]/(pct[0]+pct[1]))
            pct_inducer.append(100*pct[1]/(pct[0]+pct[1]))
            pct_all.append(100)
            ROI.append(roi)
            Subject.append(sbj)
            recon_types.append(reconType)
            Line_space.append(line_space)
            Illusory_orient.append(illus_orient)
            x.append(key)

    pct_diff_orientation = {'Illusory': pct_illus, 'Inducer': pct_inducer, 'All': pct_all, 
                            'ROI':ROI, 'Subject':Subject, 'x':x, 'Orientation difference':Illusory_orient,
                           'reconType': recon_types, 'Line space':Line_space}

    df_plot = pd.DataFrame(data=pct_diff_orientation)
    return df_plot  


    

def makeDataFrame_diff_orientation(df, illusion_type, regionType_list, plotdict):

    x = []
    pct_illus = []
    pct_inducer = []
    pct_all = []
    region_types = []
    recon_types = []
    Subject = []
    ROI = []
    Illusory_orient = []
    Line_space = []
   
    keys = plotdict.keys()
    for key in keys:
       
        condition = plotdict[key]
        reconType = condition[0]
        roi = condition[1]
        sbj = condition[2]
        line_space_list = condition[3]
        illus_orient_list = condition[4]
        
        if sbj != 'Subjects pooled':
            df2 = df[(df.reconType==reconType)&(df.ROI==roi)&(df.Subject==sbj)&(df.stimType==illusion_type)]
        else:
            df2 = df[(df.reconType==reconType)&(df.ROI==roi)&(df.stimType==illusion_type)]

        for illus_orient,line_space,regionType in product(illus_orient_list, line_space_list, regionType_list): 

            tmp = df2[(df2.regionType==regionType)&(df2['Illusory orientation'].isin(illus_orient))&(df2['Line space']==line_space)]
            t = tmp['Orientation category'].value_counts()
            pct = 100*t/t.sum()
            pct = pct.reindex(['Illusory', 'Inducer',  'Not clear'], fill_value=0)
           
            pct_illus.append(100*pct[0]/(pct[0]+pct[1]))
            pct_inducer.append(100*pct[1]/(pct[0]+pct[1]))
            pct_all.append(100)
            ROI.append(roi)
            Subject.append(sbj)
            region_types.append(regionType)
            recon_types.append(reconType)
           
            if 90 not in illus_orient and 45 in illus_orient:
                Illusory_orient.append(45)
            elif 45 not in illus_orient and 90 in illus_orient:
                Illusory_orient.append(90)
            else:
                Illusory_orient.append(0)
            Line_space.append(line_space)
            x.append(key)

    pct_diff_orientation = {'Illusory': pct_illus, 'Inducer': pct_inducer, 'All': pct_all, 
                            'ROI':ROI, 'Subject':Subject, 'x':x, 'Orientation difference':Illusory_orient,
                           'reconType': recon_types, 'regionType': region_types, 'Line space':Line_space}

    df_plot = pd.DataFrame(data=pct_diff_orientation)
    return df_plot  


def makeDataFrame4combine_globalLocal(df, x_list, l_base=50):
    # take difference between region types
    X = []
    y = []
    ytype = []
    
    for x in x_list: 
        if 'regionType' in df:
            for r in ['Illusory','Non-illusory']:
                pct = df[(df.x==x)&(df.regionType==r)].Illusory.values
                y.append(pct[0]-l_base)
                ytype.append(r)
                X.append(x)
        else:
            pct = df[(df.x==x)].Illusory.values
            y.append(pct[0]-l_base)
            ytype.append('Global')
            X.append(x)
        
        
               
    pct_region_diff = {'x':X, 'y': y, 'Scale':ytype}
    df_plot = pd.DataFrame(data=pct_region_diff)
    return df_plot

def makeDataFrame_color_weight(df, yname, stimtypes, rois, sbjs):
     # make dataframe   
    X = []
    Y = []
    hue = []
    Subject = []

    for stimtype in stimtypes:   
        
        for roi in rois:
            for sbj in sbjs:
                y1 = df[(df.Subject==sbj)&(df.ROI==roi)&(df['stimType']==stimtype)][yname].values
                for i, yi in enumerate(y1):

                    if np.isnan(yi[1]):
                        y2=0
                    y2 = yi[1][0]
                    print(y2)
                    Y.append(y2)
                    X.append(roi)
                    hue.append(stimtype)
                    Subject.append(sbj)
                  
     
    df_plot = pd.DataFrame.from_dict({'Y':Y, 'X': X, 'hue':hue, 'Subject':Subject})           
    return df_plot