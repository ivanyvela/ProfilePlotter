# -*- coding: utf-8 -*-
"""
Created on 9 October 09:09:32 2024
@author: ivanyvela
"""

#TODO: embed the getAarhusColormap function properly within the matplotlib cmap integration and put the logarithmic ticks in the colorbar legend
#TODO: profiler.py:344: RuntimeWarning: something is dividing by NaN and consuming time in the Model.createProfiles, most likely the interpolation
#TODO: profiler might be a bad name, when we also have an instrument by the same name?
#TODO: the interpotaling functions is too slow, and it consumes most of the time
#TODO: The dynamic calculation of square, text size and spacing of Plot.addBoreholeLegend uses a single formula for all elements. Works well for 2 to 8 lithologies
#TODO: Fit the legend into a TEMprofile figure. Figure out the coordinate system to plot things on in a sandbox 
#TODO: allow to choose which DOI is to be plotted
#TODO: review the interpolation method. It can be both done faster and there needs to be an option for just grabbing the nearest neighbour to the profile
#Inmediate TODOS:
#TODO: PC_01 only has profiler data in it, and therefore gives an error. see #TODO: Model.createProfiles uses ttem_model_idx. How about if no tTEM in it?
#TODO: profiler and borehole elevation problems were fixed but weird data structure left behind
#TODO: see LLM Profiler_Inefficiency
#TODO: Plot tTEM models with xm distance check---> check against actual tTEM map
#TODO: make profiler class specially thinking about the profileMap method
#TODO: when plotting sTEM or profiler data, the DOI is not very clear
#TODO: all profiles get a certain axis fontseize no matter their length, but then because the long ones do not fit into an A4 page, they need to get squeezed and the fontsize looks small, which looks amateur on reports


import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import re
import contextily as cx
import rasterio
from rasterio.plot import show
import textwrap
from matplotlib import cm
from rasterio.windows import from_bounds
from rasterio.plot import show

import textwrap