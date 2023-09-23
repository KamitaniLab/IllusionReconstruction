#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:08:55 2022

@author: fcheng
"""


import numpy as np


from statsmodels.stats.weightstats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest

def compare_two_weight_samples(df, figtype, rois, sbjs_dict):
    
    sbjs = sbjs_dict[figtype]
    pvals_dict = dict.fromkeys(sbjs)
    
    pvals = dict.fromkeys(rois)
    for r,roi in enumerate(rois):
        print(roi)
        df2 = df[(df.Subject.isin(sbjs))&(df.ROI==roi)]
        # illusion
        y = df2[df2['stimType']=='Illusion']['Beta coefficient'].values
        for i, yi in enumerate(y):
            if np.isnan(yi[1]):
                y[i] = 0
            else:
                y[i] = yi[1][0]
        y1 = np.array(y)

        # Control
        y = df2[df2['stimType']=='Control']['Beta coefficient'].values
        for i, yi in enumerate(y):
            if np.isnan(yi[1]):
                y[i] = 0
            else:
                y[i] = yi[1][0]
        
        y2 = np.array(y)

        print(len(y1), len(y2))
        [stat,pval,df0]=ttest_ind(y1, y2,alternative="larger", usevar="unequal")

        pvals[roi] = [pval]
       
    pvals_dict['pooled'] = pvals   
    
    for sbj in sbjs:
        print(sbj)
        pvals = dict.fromkeys(rois)
        for r,roi in enumerate(rois):
            print(roi)
            df2 = df[(df.Subject==sbj)&(df.ROI==roi)]
            # illusion
            y = df2[df2['stimType']=='Illusion']['Beta coefficient'].values
            for i, yi in enumerate(y):
                if np.isnan(yi[1]):
                    y[i] = 0
                else:
                    y[i] = yi[1][0]
           
            y1 = np.array(y)
            
            # Control
            y = df2[df2['stimType']=='Control']['Beta coefficient'].values
            for i, yi in enumerate(y):
                if np.isnan(yi[1]):
                    y[i] = 0
                else:
                    y[i] = yi[1][0]
           
            y2 = np.array(y)
            
            print(len(y1), len(y2))
            [stat,pval,df0]=ttest_ind(y1, y2,alternative="larger", usevar="unequal")
            
            pvals[roi] = [pval, np.mean(y1)-np.mean(y2)]
            
        pvals_dict[sbj] = pvals   
    return pvals_dict



def compare_two_proportions(df, illusion_type, rois, sbjs):
    
    pvals_dict = dict.fromkeys(sbjs)
    for sbj in sbjs:
        print(sbj)
        pvals = dict.fromkeys(rois)
        for r,roi in enumerate(rois):
            print(roi)
            stat = df[(df.stimType==illusion_type)&(df.ROI==roi)&(df.Subject==sbj)]
            stat = stat.dropna() #Drop the nan values
            num_illusoryRegion = stat[stat.regionType=="Illusory"].shape[0]
            num_non_illusory = stat[stat.regionType=="Non-illusory"].shape[0]

            prop = stat.groupby("regionType")["Orientation category"].agg([lambda z: np.sum(z=="Illusory"), "size"])
            prop.columns = ['proportions_illusory','total_counts']
            print(prop)

            count = np.array([prop.proportions_illusory['Illusory'], prop.proportions_illusory['Non-illusory']])
            nobs = np.array([num_illusoryRegion, num_non_illusory])
            stat, pval = proportions_ztest(count, nobs,alternative='larger')

            
            pvals[roi] = [pval]

        pvals_dict[sbj] = pvals
    return pvals_dict





