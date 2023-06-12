from __future__ import print_function
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

def make_dir(dirname,verbose=True):
    try:
        os.makedirs(dirname)
        if verbose==True: print("Created folder:",dirname)
    except OSError:
        if verbose==True: print(dirname,"already exists.")

def ax_apply_settings(ax,ticksize=None):
    """
    Apply axis settings that I keep applying
    """
    ax.minorticks_on()
    if ticksize is None:
        ticksize=12
    ax.tick_params(pad=3,labelsize=ticksize)
    ax.grid(lw=0.5,alpha=0.5)

def get_cmap_colors(cmap='jet',p=None,N=10):
    """

    """
    cm = plt.get_cmap(cmap)
    if p is None:
        return [cm(i) for i in np.linspace(0,1,N)]
    else:
        normalize = matplotlib.colors.Normalize(vmin=min(p), vmax=max(p))
        colors = [cm(normalize(value)) for value in p]
        return colors

def ax_set_linewidth(ax,linewidth=2):
    """
    Change the line width of the matplotlib axes
    """
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(linewidth)

