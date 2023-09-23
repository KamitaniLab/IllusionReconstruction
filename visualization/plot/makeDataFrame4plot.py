#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:43:08 2022

@author: fcheng
"""
import pandas as pd
from itertools import product

def makeDataFrame_pct_illus(df, illusion_type, regionType_list, plotdict, key):
    
    """
    make dataframe that is ready for plot
    """
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
    keyvals = plotdict.keys()
    for k, keyval in enumerate(keyvals):
        condition = plotdict[keyval]
        reconType = condition[0]
        rois = condition[1]
        sbjs = condition[2]
        line_space_list = condition[3]
        illus_orient_list = condition[4]
        
        if sbjs != 'Subjects pooled':
            df2 = df[(df.reconType==reconType)&(df.Subject.isin(sbjs))&(df.stimType==illusion_type)&(df[key]==keyval)]
        else:
            df2 = df[(df.reconType==reconType)&(df.stimType==illusion_type)&(df[key]==keyval)]
              
        for sbj, roi, illus_orient,line_space,regionType in product(sbjs, rois, illus_orient_list, line_space_list, regionType_list):
            
            tmp = df2[(df2.regionType==regionType)&(df2['Illusory orientation'].isin(illus_orient))&(df['Line space']==line_space)\
                      &(df.ROI==roi)&(df.Subject==sbj)]
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
            if key == 'Volume':
                x.append(keyval*2)
            else:
                x.append(keyval)

    pct_diff_orientation = {'Illusory': pct_illus, 'Inducer': pct_inducer, 'All': pct_all, 
                            'ROI':ROI, 'Subject':Subject, 'x':x, 'Orientation difference':Illusory_orient,
                            'reconType': recon_types, 'regionType': region_types, 'Line space':Line_space}

    df_plot = pd.DataFrame(data=pct_diff_orientation)
    return df_plot