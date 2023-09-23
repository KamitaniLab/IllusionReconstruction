#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 17:13:33 2023

@author: fcheng
"""


import pandas as pd
import seaborn as sns

import os

from plot.makeDataFrame4plot import makeDataFrame_color_weight
from plot.barplot import barplot_dotline
from plot.stats import compare_two_weight_samples

save_dir = "./results/plots/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
        

# settings
figtypes = ['Ehrenstein', 'Varin']
stimtypes = ['Illusion', 'Control']
maptype = 'Redness'
ytype = 'Illusory_surface'
alpha = 0.05

rois = ['VC','V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA']
sbjs = {'Ehrenstein':['S1','S2','S3','S4','S5', 'S6', 'S7'],
            'Varin':['S1','S2','S3','S5', 'S6', 'S7']}
dotcolor_tmp =sns.color_palette('deep')
dotcolor = {'Ehrenstein': dotcolor_tmp, 'Varin': [dotcolor_tmp[0], dotcolor_tmp[1], dotcolor_tmp[2], 
                                                 dotcolor_tmp[4], dotcolor_tmp[5], dotcolor_tmp[6]]}
col2 =['#0000CD','#87CEEB','#20B2AA', '#FFD700', '#DDA0DD', '#FFA500','#D2691E']  
linecolor = {'Ehrenstein': col2, 'Varin': [col2[0], col2[1], col2[2], 
                                           col2[4], col2[5], col2[6]]}

# multiple regression
model = 'stimulus + red surface'
yname = 'Beta coefficient'
avg = 'sbj'


# draw coeffcient for each configuration
for figtype in figtypes:
  
    si_fn = './results/evaluation/Regression_color_'+figtype+'.pkl'
    df = pd.read_pickle(si_fn)   
    df_tmp = df[(df.Model==model)&(df.ROI.isin(rois))] 
    
    # perform statistical analysis
    pvals = compare_two_weight_samples(df_tmp, figtype, rois, sbjs)
    for sbj in sbjs[figtype]:
        for roi in rois:
            if pvals[sbj][roi][0]<alpha:
                pvals[sbj][roi] = 1
            elif pvals[sbj][roi][0]>alpha:
                pvals[sbj][roi] = 0
    
    # prepare data
    df2 = makeDataFrame_color_weight(df_tmp, yname, stimtypes, rois, sbjs[figtype])

    # create color map 
    col_palette1 = sns.light_palette("gray", n_colors=6,reverse=True)
    col_palette2 = sns.dark_palette("gray", n_colors=12,reverse=True)

    # draw bars
    xlabel = 'ROI' # x-axis label
    xticks = rois
    
   
    ylabel = 'beta illus surface'   
    ymax = 0.36
    yticks = [0,0.1,0.2,0.3]
    ymin = -0.02

    save_title = 'Fig4_'+figtype+'.pdf' # file title
    
    barplot_dotline(df2, save_dir, save_title,  xlabel, xticks, ylabel, yname, yticks=yticks, ymin=ymin, ymax=ymax, 
                barhuename='hue', barhuenameval=stimtypes, 
                barhue=[col_palette2[2], col_palette1[4]], barwidth = 0.27, ci=None, 
                changeerrbarcol=True, changeerrcol=[ col_palette1[2]],
                linehue=True, linehuewidth=2, linehuecolor=linecolor[figtype], pvals = pvals,
                dot=True, dothuename="Subject", dothuelist=sbjs[figtype], dotstyle='o', dotsize=8, dotcolor=linecolor[figtype],
                dpi=300, width=1, height=5, labelsize=30, ticksize=30, legend=True)
        
  