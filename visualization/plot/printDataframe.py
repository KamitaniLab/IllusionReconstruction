#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 14:07:21 2022

@author: fcheng
"""
import numpy as np
import matplotlib.pyplot as plt
import six

def render_table(df, savfile, col_width=3.0, row_height=0.625, font_size=14, 
                 header_color = 'gray', row_colors=['#f1f1f2', 'w'], edge_color='w',
                 bbox = [0,0,1,1], header_columns=0, ax=None, **kwargs):
    if ax is None:
        size = (np.array(df.shape[::-1]) + np.array([0,1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize = size)
        ax.axis('off')
        
    table = ax.table(cellText=df.values, bbox=bbox, colLabels=df.columns, **kwargs)
    
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    
    for k, cell in six.iteritems(table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    
    plt.savefig(savfile, bbox_inches='tight')
    
    return ax

def render_table_significancy_and_effect_size(df, savfile, col_width=3.0, row_height=0.625, font_size=14, 
                 header_color = 'gray', row_colors=['#f1f1f2', 'w'], edge_color='w',
                 bbox = [0,0,1,1], header_columns=0, ax=None, **kwargs):
    if ax is None:
        size = (np.array(df.shape[::-1]) + np.array([0,1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize = size)
        ax.axis('off')
        
    table = ax.table(cellText=df.values, bbox=bbox, colLabels=df.columns, **kwargs)
    
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    
    for k, cell in six.iteritems(table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    
    plt.savefig(savfile, bbox_inches='tight')
    
    return ax