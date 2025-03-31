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
#g

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

class Model:

    def __init__(self):
        self.ttem_models = []
        self.stem_models = []
        self.profiler_models = [] 
        self.profiles = []
        self.boreholes = []
        self.profile_filenames = []

    def loadXYZ(self, xyz_path, return_mod_df=False, mod_name=None, model_type='tTEM'):
        """
        bragalert: This function is now 30 lines shorter and 15% faster
        """
        tem_model = {}

        # Open the file and process lines to find the header end dynamically
        row_idx = 0
        with open(xyz_path, "r") as f:
            for line in f:
                row_idx += 1

                if 'DATA TYPE' in line:
                    tem_model['instrument'] = f.readline().strip().replace("/DT", "")
                    row_idx += 1  # Move past the next line after 'DATA TYPE'

                if 'COORDINATE SYSTEM' in line:
                    espg_code = re.search(r'epsg:(\d+)', f.readline()).group(1)
                    tem_model['epsg'] = 'epsg:' + espg_code
                    row_idx += 1  # Move past the next line after 'COORDINATE SYSTEM'

                if 'LINE_NO' in line:
                    row_idx -= 1
                    # Stop reading when the data section starts
                    break

        # Read the rest of the file into a pandas DataFrame
        mod_df = pd.read_csv(xyz_path, delimiter=None, skiprows=row_idx)
        
        # Process the DataFrame: split first column, rename columns, drop unnecessary ones
        col_names = mod_df.columns[0].split()[1:]
        mod_df = mod_df[mod_df.columns[0]].str.split(expand=True)
        mod_df.columns = col_names
        mod_df = mod_df.drop(columns=['DATE', 'TIME'], errors='ignore').astype(float)

        # Extract relevant data into numpy arrays
        rho_cols = [col for col in mod_df.columns if 'RHO' in col and 'STD' not in col]
        depth_cols = [col for col in mod_df.columns if 'DEP_BOT' in col and 'STD' not in col]

        rhos = mod_df[rho_cols].values
        depths = mod_df[depth_cols].values
        doi_con = mod_df['DOI_CONSERVATIVE'].values
        doi_standard = mod_df['DOI_STANDARD'].values
        x = mod_df['UTMX'].values
        y = mod_df['UTMY'].values
        elev = mod_df['ELEVATION'].values
        residual = mod_df['RESDATA'].values
        line_num = mod_df['LINE_NO'].astype(int).values

        # Store data in the model dictionary
        if mod_name:
            tem_model['mod_name'] = mod_name
        else:
            tem_model['mod_name'] = len(self.ttem_models)

        tem_model.update({
            'x': x,
            'y': y,
            'elev': elev,
            'rhos': rhos,
            'depths': depths,
            'doi_con': doi_con,
            'doi_standard': doi_standard,
            'residual': residual,
            'line_num': line_num,
        })

        if return_mod_df:
            tem_model['mod_df'] = mod_df

        # Append the loaded model to the appropriate list
        if model_type == 'tTEM':
            self.ttem_models.append(tem_model)
        elif model_type == 'sTEM':
            self.stem_models.append(tem_model)
        elif model_type in ['Profiler', 'profiler']:
            self.profiler_models.append(tem_model)
        else:
            print('Model type not recognized, no data loaded.')

        
    def combineModels(self, idx):
        
        idx = np.sort(idx)
        
        new_tem_model = self.tem_models[idx[0]]
        
        for i in idx[1:]:
            
            
            new_tem_model['x'] = np.append(new_tem_model['x'], self.tem_models[i]['x'])
            new_tem_model['y'] = np.append(new_tem_model['y'], self.tem_models[i]['y'])
            new_tem_model['elev'] = np.append(new_tem_model['elev'], self.tem_models[i]['elev'])
            new_tem_model['rhos'] = np.append(new_tem_model['rhos'], self.tem_models[i]['rhos'], axis=0)
            new_tem_model['depths'] = np.append(new_tem_model['depths'], self.tem_models[i]['depths'], axis=0)
            
            new_tem_model['doi_con'] = np.append(new_tem_model['doi_con'], self.tem_models[i]['doi_con'])
            new_tem_model['doi_standard'] = np.append(new_tem_model['doi_standard'], self.tem_models[i]['doi_standard'])
            
            new_tem_model['residual'] = np.append(new_tem_model['residual'], self.tem_models[i]['residual'])
            
            new_tem_model['line_num'] = np.append(new_tem_model['line_num'], self.tem_models[i]['line_num'])

           
        #del self.tem_models[idx[1:]]
        
            
        
            
            #new_tem_model['y'] = y
            #new_tem_model['elev'] = elev
            #new_tem_model['rhos'] = rhos
            #new_tem_model['depths'] = depths
            #tem_model['doi_con'] = doi_con
            #tem_model['doi_standard'] = doi_standard
            #tem_model['residual'] = residual
            #tem_model['line_num'] = line_num
        
        
    
        #new_tem_model = [self.tem_models[idx[0]]]
        
       # for i in idx[1:]:
            
        #    new_tem_model.extend(self.tem_models[i])
            
         #   self.tem_models.pop(i)
        
        #self.tem_models[idx[0]] = new_tem_model

    def loadProfileCoords(self, profile_coord_paths, file_type='csv'):

        if file_type == 'csv':

           for profile_coord_path in profile_coord_paths:
                filename = profile_coord_path.replace('\\', '/').split('/')[-1].replace('.csv', '')
                self.profile_filenames.append(filename)
                # Read the CSV file into a DataFrame
                df = pd.read_csv(profile_coord_path)

                # Initialize variables for x and y columns
                x_col = None
                y_col = None

                # Loop through the columns to find the ones that match 'X'/'x' and 'Y'/'y'
                for col in df.columns:
                    if col.lower() == 'x':
                        x_col = col
                    elif col.lower() == 'y':
                        y_col = col

                # If both x and y columns are found, extract the values
                if x_col and y_col:
                    profile = {}
                    profile['x'] = df[x_col].values
                    profile['y'] = df[y_col].values

                    # Append the profile to the profiles list
                    self.profiles.append(profile)
                else:
                    print(f"Error: Could not find 'X' and 'Y' columns in {profile_coord_path}")

        elif file_type == 'shp':

            for profile_coord_path in profile_coord_paths:
                # Read the shapefile using geopandas
                gdf = gpd.read_file(profile_coord_path)
                filename = profile_coord_path.replace('\\', '/').split('/')[-1].replace('.shp', '')
                self.profile_filenames.append(filename)
                
                for idx, feature in gdf.iterrows():
                    geometry = feature.geometry
                    if isinstance(geometry, LineString):
                        # Extract the coordinates
                        coords = list(geometry.coords)
                        profile = {
                            'x': [coord[0] for coord in coords],
                            'y': [coord[1] for coord in coords]
                        }
                        # Append the profile to the profiles list
                        self.profiles.append(profile)

        else:
            print('File type was not recognised, choose "csv" or "shp".')


    def readShpFile(self, shp_file_path):

        gdf = gpd.read_file(shp_file_path)

        sections = []

        for index, row in gdf.iterrows():
            geometry = row['geometry']

            # Check if the geometry is a LineString
            if geometry.geom_type == 'LineString':
                # Access the LineString coordinates
                section_coords = list(geometry.coords)

                sections.append(np.array(section_coords))

        return sections

    def interpCoords(self, x_p, y_p, distance=10):
        interpolated_points = []

        for i in range(len(x_p)-1):
            x1 = x_p[i]
            y1 = y_p[i]
            x2 = x_p[i+1]
            y2 = y_p[i+1]
            dx = x2 - x1
            dy = y2 - y1
            segments = int(np.sqrt(dx**2 + dy**2) / distance)

            if segments != 0:

                for j in range(segments + 1):
                    xi = x1 + dx * (j / segments)
                    yi = y1 + dy * (j / segments)
                    interpolated_points.append((xi, yi))

        interpolated_points = np.array(interpolated_points)

        xi, yi = interpolated_points[:, 0], interpolated_points[:, 1]
        dists = (np.diff(xi) ** 2 + np.diff(yi) ** 2) ** 0.5
        dists = np.cumsum(np.insert(dists, 0, 0))
        idx = np.unique(dists, return_index=True)[1]

        return xi[idx], yi[idx], dists[idx]

    def interpIDW(self, x, y, z, xi, yi, power=2, interp_radius=10):
        """

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.
        xi : TYPE
            DESCRIPTION.
        yi : TYPE
            DESCRIPTION.
        power : TYPE, optional
            DESCRIPTION. The default is 2.
        interp_radius : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        """
        # Calculate distances between grid points and input points
        dists = np.sqrt((x[:, np.newaxis] - xi[np.newaxis, :])**2 +
                        (y[:, np.newaxis] - yi[np.newaxis, :])**2)

        # Calculate weights based on distances and power parameter
        weights = 1.0 / (dists + np.finfo(float).eps)**power

        # Set weights to 0 for points outside the specified radius
        weights[dists > interp_radius] = 0

        # Normalize weights for each grid point
        weights /= np.sum(weights, axis=0)

        # Interpolate values using weighted average
        zi = np.sum(z[:, np.newaxis] * weights, axis=0)

        return zi

    def interpLIN(self, x, z):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.

        Returns
        -------
        zi : TYPE
            DESCRIPTION.

        """

        # Create interpolation function, excluding nan values
        interp_func = interp1d(x[~np.isnan(z)], z[~np.isnan(z)],
                               kind='linear', fill_value="extrapolate")
        zi = z.copy()
        zi[np.isnan(z)] = interp_func(x[np.isnan(z)])

        if np.isnan(zi[-1]):
            zi[-1] = zi[-2]

        return zi

    def createProfiles(self, ttem_model_idx, profile_idx='all', model_spacing=10, interp_radius=40):
        #this only create profiles that contain tTEM models
        #we need to create an option where the interpolation method can be chosen: nearest neighbout, krigging, IDW...
        if profile_idx == 'all':
            profile_idx = range(0, len(self.profiles))
        elif type(profile_idx) != list:
            profile_idx = [profile_idx]

        rhos = self.ttem_models[ttem_model_idx]['rhos']
        depths = self.ttem_models[ttem_model_idx]['depths']
        x = self.ttem_models[ttem_model_idx]['x']
        y = self.ttem_models[ttem_model_idx]['y']
        elev = self.ttem_models[ttem_model_idx]['elev'] 
        doi = self.ttem_models[ttem_model_idx]['doi_standard'] 

        for idx in profile_idx:
            # Add debug statement to print the profile filename
            filename = self.profile_filenames[idx]
            #print(f"Processing profile: {filename}")

            x_p = self.profiles[idx]['x'] 
            y_p = self.profiles[idx]['y'] 

            xi, yi, dists = self.interpCoords(x_p, y_p, distance=model_spacing)

            


            n_layers = rhos.shape[1]
            n_models = len(xi)
            rhos_new = np.zeros((n_models, n_layers))

            for i in range(rhos.shape[1]):
                rhos_new[:, i] = self.interpIDW(x, y, rhos[:, i], xi, yi, power=2,
                                                interp_radius=interp_radius)

            depths_new = np.repeat(depths[0, :][None, :], n_models, axis=0)

            # Debug statement before interpolation
            #print(f"Interpolating elevation for profile: {filename}")
            #print(f"xi: {xi}")
            #print(f"yi: {yi}")
            #print(f"dists: {dists}")
            #print(f"elev: {elev}")

            elev_new = self.interpIDW(x, y, elev, xi, yi, power=2, interp_radius=interp_radius)

            # Debug statement before calling interpLIN
            #print(f"Interpolated elevation: {elev_new}")
            
            try:
                elev_new = self.interpLIN(dists, elev_new)
            except ValueError as e:
                print(f"Error occurred with profile: {filename}")
                print(f"dists: {dists}")
                print(f"elev_new: {elev_new}")
                print(f"Interpolated elevation: {elev_new}")
                print(f"xi: {xi}")
                print(f"yi: {yi}")
                print(f"dists: {dists}")
                print(f"elev: {elev}")
                raise e  # Re-raise the exception after logging

            doi_new = self.interpIDW(x, y, doi, xi, yi, power=2, interp_radius=interp_radius)
            doi_new = self.interpLIN(dists, doi_new)

            #use dictionary's update method here: 
            self.profiles[idx]['rhos'] = rhos_new
            self.profiles[idx]['depths'] = depths_new
            self.profiles[idx]['elev'] = elev_new
            self.profiles[idx]['doi'] = doi_new
            self.profiles[idx]['distances'] = dists
            self.profiles[idx]['xi'] = xi
            self.profiles[idx]['yi'] = yi


    def loadBoreholes(self, borehole_paths, file_type='dat'):
        """

        Parameters
        ----------
        borehole_path : TYPE
            DESCRIPTION.

        Returns
        -------
        borehole_dict : TYPE
            DESCRIPTION.

        """

        if file_type == 'dat':
            for borehole_path in borehole_paths:
                bh_df = pd.read_csv(borehole_path, sep='\t')

                bh_dict = {}

                bh_dict['id'] = bh_df['id'][0]
                bh_dict['n_layers'] = len(bh_df)
                bh_dict['x'] = bh_df['utm_x'].values[0]
                bh_dict['y'] = bh_df['utm_y'].values[0]
                bh_dict['elevation'] = bh_df['elev'].values[0]
                bh_dict['top_depths'] = bh_df['top_depths'].values
                bh_dict['bot_depths'] = bh_df['bot_depths'].values
                bh_dict['colors'] = bh_df['colors'].values
                bh_dict['descriptions'] = bh_df['lith_descriptions']
                bh_dict['lith_names'] = bh_df['lith_names'].values

                self.boreholes.append(bh_dict)

        elif file_type == 'xlsx':
            for borehole_path in borehole_paths:
                excel_file = pd.ExcelFile(borehole_path)
                sheet_names = excel_file.sheet_names

                for sheet_name in sheet_names:
                    bh_dict = {}
                    bh_df = excel_file.parse(sheet_name)
                    bh_dict['id'] = bh_df['id'][0]
                    bh_dict['n_layers'] = len(bh_df)
                    bh_dict['x'] = bh_df['utm_x'].values[0]
                    bh_dict['y'] = bh_df['utm_y'].values[0]
                    if 'elev' in bh_dict:
                        bh_dict['elevation'] = bh_df['elev'].values[0]
                    else:
                        bh_dict['elevation'] = 0
                    bh_dict['top_depths'] = bh_df['top_depths'].values
                    bh_dict['bot_depths'] = bh_df['bot_depths'].values
                    bh_dict['colors'] = bh_df['colors'].values
                    bh_dict['lith_names'] = bh_df['lith_names'].values
                    self.boreholes.append(bh_dict)

        else:
            print('File type was not recognised, choose "csv" or "xlsx".')
