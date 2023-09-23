#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:04:11 2023

@author: fcheng
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd


from plot.makeDataFrame4plot import makeDataFrame_diff_orientation,makeDataFrame_diff_orientation_1stStage,makeDataFrame4combine_globalLocal
from plot.stats import compare_two_proportions
from plot.barplot import change_width

        

# read principal orientation for each single-trial reconstruction
result_path = './results/evaluation'
po_fn = os.path.join(result_path, 'Principal_orientation_local.pkl')
po_fn_1st = os.path.join(result_path, 'Principal_orientation_global.pkl')
po = pd.read_pickle(po_fn)
po_1st = pd.read_pickle(po_fn_1st)

# save dir
save_dir = "./results/plots/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

              
# plot parameters    
sbjs = ['S1','S2','S3','S4','S5', 'S6', 'S7']
rois = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA']


xlabels = rois
barplot=dict.fromkeys(xlabels)
# reconType, roi, sbj, line space 
for roi in rois:
    barplot[roi] = ['Recon-decoded features',roi, 'Subjects pooled', [1.2], [[0,45,90,135]]]

regionType = ['Illusory','Non-illusory']
illusion_type = 'Illusion'
l_base=0
# local 
df2 = makeDataFrame_diff_orientation(po,illusion_type, regionType, barplot)
df2 = makeDataFrame4combine_globalLocal(df2, xlabels ,l_base)
# global 
df3 = makeDataFrame_diff_orientation_1stStage(po_1st,illusion_type, barplot)
df3 = makeDataFrame4combine_globalLocal(df3, xlabels,l_base)
df2_bar = pd.concat([df3,df2], axis=0)

# calculate p value
alpha=0.05
df_pval = po[po['Line space']==1.2]
pvals = compare_two_proportions(df_pval, illusion_type, rois, sbjs)
for sbj in sbjs:
    for roi in rois:
        if pvals[sbj][roi][0] > alpha:
            pvals[sbj][roi] = 0
        elif pvals[sbj][roi][0] < alpha:
            pvals[sbj][roi] = 1
        
# main -------------------------------------------------------------------       
a = plt.figure(figsize=(15,4.5), dpi=80)

heights = [25, 50,75]
for h in heights:
    plt.axhline(h, linestyle='-',color='gray',alpha=0.5,linewidth=0.8,zorder=1)

col = plt.cm.gray(np.linspace(0.4,0.85,3))
col2 =['#0000CD','#87CEEB','#20B2AA', '#FFD700', '#DDA0DD', '#FFA500','#D2691E']   
xticks =[]
for r in range(len(rois)):
    xticks.append(r*3+0.5)
    xticks.append(r*3+1.4)
    xticks.append(r*3+2)

df2_bar = df2_bar.sort_values(by=['x','Scale'])
df2_bar = pd.concat([df2_bar[9:],df2_bar[3:6],df2_bar[:3],df2_bar[6:9]])
ax1 = plt.bar(np.array(xticks),df2_bar.y.values, color=np.tile(col,(len(rois),1)),zorder=2)

change_width(ax1, .6)

plt.legend(labels = ['Global', 'Illusory','Non-illusory'],
            bbox_to_anchor=(1, 1), loc='upper left', fontsize=16, frameon=False)

plt.xlabel("")
plt.ylabel("% closer to \n illusory orientation", fontsize=18)
plt.ylim(20,85),
plt.yticks(heights, fontsize=16)


plt.xticks(xticks, 
           ['','V1','','','V2','','','V3','','', 'V4', '','','LOC', '','','FFA', '','','PPA',''],
           fontsize=16, rotation=0)
sns.despine(a)

# plot individual subjects
ax_list = ax1.patches
for i in range(len(ax_list)):
    
    if np.mod(i,3)==0:
        p1 = ax_list[i]
        x1 = p1.get_width()/2 + p1.get_x()
       
        barplot_line=dict.fromkeys(sbjs)
        for sbj in sbjs:
            barplot_line[sbj] = ['Recon-decoded features', rois[int(i/3)], sbj, [1.2], [[0,45,90,135]]]
        df3 = makeDataFrame_diff_orientation_1stStage(po_1st,illusion_type, barplot_line)        
        
        for s, sbj in enumerate(sbjs):
            h1 = df3.loc[(df3.Subject==sbj)].Illusory.values-l_base
            plt.plot(x1,h1,'o',markeredgecolor=col2[s],markerfacecolor=[1,1,1], markersize=6,markeredgewidth=1.5,zorder=3)    
           
    elif np.mod(i,3)==1:
        p1 = ax_list[i]
        p2 = ax_list[i+1]
        x1 = p1.get_width()/2 + p1.get_x()
        x2 = p2.get_width()/2 + p2.get_x()
        
        barplot_line=dict.fromkeys(sbjs)
        for sbj in sbjs:
            barplot_line[sbj] = ['Recon-decoded features', rois[int(np.floor(i/3))], sbj, [1.2], [[0,45,90,135]]]     
        df2 = makeDataFrame_diff_orientation(po,illusion_type, regionType, barplot_line)
        
        for s, sbj in enumerate(sbjs):
            h1 = df2.loc[(df2.Subject==sbj)&(df2.regionType=='Illusory')].Illusory.values
            h2 = df2.loc[(df2.Subject==sbj)&(df2.regionType=='Non-illusory')].Illusory.values
            if pvals[sbj][rois[int(i/3)]] == 0:
                plt.plot(x1,h1,'o',markeredgecolor= col2[s],markerfacecolor=[1,1,1], markersize=6,markeredgewidth=1.5,zorder=3)
                plt.plot(x2,h2,'o',markeredgecolor= col2[s],markerfacecolor=[1,1,1], markersize=6,markeredgewidth=1.5,zorder=3)
            else:
                # indicate statistical significance
                plt.plot(x1,h1,'o',markeredgecolor= col2[s],markerfacecolor=col2[s], markersize=6,markeredgewidth=1.5,zorder=3)
                plt.plot(x2,h2,'o',markeredgecolor= col2[s],markerfacecolor=col2[s], markersize=6,markeredgewidth=1.5,zorder=3)
            plt.plot([x1,x2],[h1,h2], color=col2[s], linewidth=1.5,zorder=3)    
           
            
save_title = 'Fig3F.pdf'
plt.savefig(os.path.join(save_dir, save_title), bbox_inches='tight')


