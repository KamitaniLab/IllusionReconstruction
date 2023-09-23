#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 17:53:00 2022

@author: fcheng
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def change_width(ax, new_value) :
    """
    Change bar width
    """
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)
        

        

def barplot_dotline(df, save_dir, save_title,  xlabel, xticks, ylabel, yname, ymin=None, ymax=None, yinterval=None, yticks=None, title=None,
                    barhuename=None, barhuenameval=None, barhue=None, barwidth = 0.5, ci=None, errcolor='black', pvals = None,
                       dot=None, dothuename="Subject", dothuelist=None, dotstyle='o', dotsize=4, dotedgewidth =1.5, dotcolor="black", dotcolor2=None,
                       linehue=None, linehuewidth=2, linehuecolor='black', 
                       dpi=80, width=1.5, height=8, titlesize=20, labelsize=20, ticksize=18, 
                       legend=None, display=None, plot_type='bar', split=False, cut=2, scale='width', inner='quartile', 
                       changebarcol=None, changecol = None, changeerrbarcol=None, changeerrcol = None,
                       medianprops={"color":"coral", "linewidth":2}):
 
   """
   Draw bar plot with optional dots/lines
   dot: Draw dots if dot=True
   linehue: Draw lines between bars of different hues, if linehue=True
   """
   
   if barhuenameval is None:
       a = plt.figure(figsize=(width*len(xticks),height), dpi=dpi)
   else:
       a = plt.figure(figsize=(width*len(xticks)*len(barhuenameval),height), dpi=dpi)
   
   # plot bar
   ax=sns.barplot(x='X', y='Y', data=df, ci=ci,  hue=barhuename, errcolor=errcolor,
                     orient="v", palette=barhue, saturation=0.5,zorder=2)
  
   change_width(ax, barwidth)
   
   if not changebarcol is None:
       for i, patch in enumerate(ax.patches):
           if i%2==1:
               patch.set_color(changecol[int((i-1)/2)])
               
   if not changeerrbarcol is None:
       for i, line in enumerate(ax.get_lines()):
           if i<=len(ax.get_lines())/2:
               line.set_color(changeerrcol[0])     
           else:
               line.set_color(changeerrcol[0])   
               
   # plot dashed line
   if not yticks is None:
       heights = np.array(yticks)
   elif (not ymin is None) and (not ymax is None):
       heights = np.arange(ymin, ymax, yinterval) # w un
   else:
       heights = np.array([0])
   for h in heights:
       plt.axhline(h, linestyle='-',color='gray',alpha=0.5,linewidth=0.8,zorder=1)
   
   # plot dots and line
   # draw lines between different xticks
   ax_list = ax.patches
   interval = len(xticks)
   for i in range(len(ax_list)):
     
        if i>=0 and i<interval:
             
             p1 = ax_list[i]
             p2 = ax_list[i+interval]
             x1 = p1.get_width()/2 + p1.get_x()
             x2 = p2.get_width()/2 + p2.get_x()
             if len(barhuenameval)==3:
                
                 p3 = ax_list[i+2*interval]
                 x3 = p3.get_width()/2 + p3.get_x()
             
             if isinstance(dothuelist, dict):
                 dothuelist2 = dothuelist[xticks[i]]
             else:
                 dothuelist2 = dothuelist
             for j, dothue in enumerate(dothuelist2):    
             
                 h1 = df.loc[(df['X']==xticks[i])&(df[dothuename]==dothue)&(df[barhuename]==barhuenameval[0])].Y.values
                 h2 = df.loc[(df['X']==xticks[i])&(df[dothuename]==dothue)&(df[barhuename]==barhuenameval[1])].Y.values
                 if len(barhuenameval)==3:
                     h3 = df.loc[(df['X']==xticks[i])&(df[dothuename]==dothue)&(df[barhuename]==barhuenameval[2])].Y.values
                 
                 if isinstance(dotcolor, str):
                     color = dotcolor
                 elif len(dotcolor) == 2:
                     color = dotcolor[i]
                 else:
                     color = dotcolor[j]
                     
                 if len(h1)>1:
                     h1 = np.mean(h1)
                     h2 = np.mean(h2)
                     if len(barhuenameval)==3:
                         h3 = np.mean(h3)
                 if pvals[dothue][xticks[i]] == 0:    
                     plt.plot(x1,h1,dotstyle,markeredgecolor=color,markerfacecolor=[1, 1, 1], markersize=dotsize,markeredgewidth=dotedgewidth)
                     plt.plot(x2,h2,dotstyle,markeredgecolor=color, markerfacecolor=[1, 1, 1], markersize=dotsize,markeredgewidth=dotedgewidth)
                 else:
                     plt.plot(x1,h1,dotstyle,markeredgecolor=color,markerfacecolor=color, markersize=dotsize,markeredgewidth=dotedgewidth)
                     plt.plot(x2,h2,dotstyle,markeredgecolor=color, markerfacecolor=color, markersize=dotsize,markeredgewidth=dotedgewidth)
                 if len(barhuenameval)==3:
                     plt.plot(x3,h3,dotstyle,markeredgecolor=color, markerfacecolor=[1, 1, 1], markersize=dotsize,markeredgewidth=dotedgewidth)
                 if not linehue is None:
                     #draw lines between different bars on the same xtick 
                     if isinstance(linehuecolor, str):
                         linecolor = linehuecolor
                     elif len(linehuecolor) == 2:
                         linecolor = linehuecolor[i]
                     else:
                         linecolor = linehuecolor[j]
                     if pvals[dothue][xticks[i]] == 0:
                         plt.plot([x1,x2],[h1,h2],'-', color=linecolor, linewidth=linehuewidth, label=dothue)
                     else:
                         plt.plot([x1,x2],[h1,h2], color=linecolor, linewidth=linehuewidth, label=dothue)
                     if len(barhuenameval)==3:
                         plt.plot([x2,x3],[h2,h3], color=linecolor, linewidth=linehuewidth, label=dothue)
                     if not legend is None:
                         handles, labels = ax.get_legend_handles_labels()
                         display_bar =  tuple([len(dothuelist)*interval, len(dothuelist)*interval+1])
                         display_line = tuple(d for d in range(len(dothuelist)))
                         ax.legend([handle for i,handle in enumerate(handles) if i in display_bar]+\
                                   [handle for i,handle in enumerate(handles) if i in display_line],
                                  [label for i,label in enumerate(labels) if i in display_bar]+\
                                   [label for i,label in enumerate(labels) if i in display_line],
                                  bbox_to_anchor=(1, 1), loc='upper left', fontsize=labelsize,frameon=False)
                     else:
                         plt.legend([],[],frameon=False)
 

   # labels and ticks
   plt.ylabel(ylabel,fontsize=labelsize)
   plt.xlabel(xlabel,fontsize=labelsize)
   
   ax.set_xticklabels(xticks)
   
   plt.xticks(fontsize=ticksize, rotation=0)
   if yticks is None:
       plt.yticks(fontsize=ticksize)
   else:
       plt.yticks(heights, labels=yticks, fontsize=ticksize)
   if (not ymin is None) and (not ymax is None):
       plt.ylim((ymin, ymax)) 
   plt.title(title, fontsize=titlesize)
   sns.despine(a)                                                                                  
   
   plt.savefig(os.path.join(save_dir, save_title), bbox_inches='tight')
   
