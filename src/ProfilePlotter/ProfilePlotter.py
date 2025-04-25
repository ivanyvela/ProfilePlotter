# -*- coding: utf-8 -*-
"""
Created on 9 October 09:09:32 2024
@author: ivanyvela
"""


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


class Plot:


    def __init__(self, model):
        self.model = model

    def getAarhusColormap(self):
        cols = [(0, 0, 190),
                (0, 75, 220),
                (0, 150, 235),
                (0, 200, 255),
                (80, 240, 255),
                (30, 210, 0),
                (180, 255, 30),
                (255, 255, 0),
                (255, 195, 0),
                (255, 115, 0),
                (255, 0, 0),
                (255, 0, 120),
                (140, 40, 180),
                (165, 70, 220),
                (195, 130, 240),
                (230, 155, 240)]
        
        # Normalize the RGB values to [0, 1] as required by Matplotlib
        return ListedColormap(np.array(cols)/255)

    def profileMapper(self, ax=None, background='imagery'):

        if background == 'imagery':
            source = cx.providers.Esri.WorldImagery
            ttem_color = 'w'
            stem_color = 'w'

        elif background == 'osm':
            source = cx.providers.OpenStreetMap.Mapnik
            ttem_color = 'k'
            stem_color = 'w'

        else:
            print('Background not recognised, specify either "imagery" or "osm".')

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 8))
        else:
            fig = ax.figure

        for i in range(len(self.model.ttem_models)):

            if i == 0:
                ax.scatter(self.model.ttem_models[i]['x'],
                           self.model.ttem_models[i]['y'],
                           marker='.', c=ttem_color, s=1, label='tTEM data')

            else:
                ax.scatter(self.model.tem_models[i]['x'],
                           self.model.tem_models[i]['y'],
                           marker='.', c=ttem_color, s=1)

        for i in range(len(self.model.stem_models)):

            if i == 0:
                ax.scatter(self.model.stem_models[i]['x'],
                           self.model.stem_models[i]['y'],
                           marker='.', c=stem_color, s=10, ec='k',
                           label='sTEM data')

            else:
                ax.scatter(self.model.tem_models[i]['x'],
                           self.model.tem_models[i]['y'],
                           marker='.', c=ttem_color, s=10, ec='k')

        if len(self.model.profiles) > 10:
            colorscale = cm.get_cmap('jet', len(self.model.profiles))

            cols = colorscale(np.linspace(0, 1, len(self.model.profiles)))

            for i in range(len(self.model.profiles)):
                ax.plot(self.model.profiles[i]['x'],
                        self.model.profiles[i]['y'], c=cols[i], lw=2, alpha=0.8,
                        label='Profile ' + str(i+1))

        else:
            for i in range(len(self.model.profiles)):
                ax.plot(self.model.profiles[i]['x'],
                        self.model.profiles[i]['y'],lw=2, alpha=0.8,
                        label='Profile ' + str(i+1))


        cx.add_basemap(ax, crs=self.model.ttem_models[0]['epsg'],  source=source,
                       attribution_size=2)

        #ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        ax.grid()
        leg = ax.legend()
        leg.legendHandles[0]._sizes = [20]
        ax.set_xlabel('Distance [m]')
        fig.tight_layout()


    def getColors(self, rhos, vmin, vmax, cmap=plt.cm.viridis, n_bins=16,
                  log=True, discrete_colors=False):
        """
        Return colors from a color scale based on numerical values

        Parameters
        ----------
        rhos : TYPE
            DESCRIPTION.
        vmin : TYPE
            DESCRIPTION.
        vmax : TYPE
            DESCRIPTION.
        cmap : TYPE, optional
            DESCRIPTION. The default is 'viridis'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if log:
            rhos = np.log10(rhos)
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)

        # Determine color of each polygon
        cmaplist = [cmap(i) for i in range(cmap.N)]

        if discrete_colors:
            cmap = LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(vmin, vmax, n_bins)
            norm = BoundaryNorm(bounds, cmap.N)

        else:
            cmap = LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, 256)

            # define the bins and normalize
            bounds = np.linspace(vmin, vmax, 256)
            norm = BoundaryNorm(bounds, 256)

        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        return scalar_map.to_rgba(rhos)


    def TEMProfile(self, profile_idx, doi=True, ax=None, vmin=1, vmax=1000,
                   scale=10, contour=False, cmap=None, log=True,
                   flip=False, cbar=True, cbar_orientation='vertical',
                   zmin=None, zmax=None, xmin=None, xmax=None, plot_title='',
                   cbar_label='Resistivity [Ohm.m]'):

        # Retrieve the filename for the current profile
        filename = self.model.profile_filenames[profile_idx]

        rhos = self.model.profiles[profile_idx]['rhos']
        depths = self.model.profiles[profile_idx]['depths']
        elev = self.model.profiles[profile_idx]['elev']
        dists = self.model.profiles[profile_idx]['distances']
        doi = self.model.profiles[profile_idx]['doi']

        if rhos.shape[1] == depths.shape[1]:

            depths = depths[:,:-1]

        self.plot2D(rhos=rhos, depths=depths, elev=elev, dists=dists,
                    doi=doi, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                    zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax,
                    plot_title=plot_title, scale=scale, 
                    cbar_orientation=cbar_orientation,filename=filename)


    def plot2D(self, rhos, depths, elev=None, dists=None, doi=None,
               ax=None, vmin=1, vmax=1000, contour=False, scale=10,
               cmap=None, n_bins=16, discrete_colors=False,
               log=True, flip=False, cbar=True, cbar_orientation='vertical',
               zmin=None, zmax=None, xmin=None, xmax=None, plot_title='',
               cbar_label='Resistivity [Ohm.m]', filename=None):

        # Set the default colormap if none is provided as the Aarhus Res, or else some of matplotlibs can be used 
        # like plt.cm.viridis
        if cmap is None:
            cmap = self.getAarhusColormap()
        # Add extra distance, otherwise problem later on, check why this is...
        if dists is not None:
            dists = np.append(dists, dists[-1])
            plot_model_idx = False
        else:
            plot_model_idx = True

        # Add 0 m depth to depths, could be buggy
        depths = -np.c_[np.zeros(depths.shape[0]), depths]

        n_layers = rhos.shape[1]
        n_models = rhos.shape[0]

        # Transform data and lims into log
        if log:
            rhos = np.log10(rhos)
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)

        if plot_model_idx:
            x = np.arange(0, rhos.shape[0]+1)
            elev = np.zeros_like(x)[:-1]

        else:
            x = dists

        if flip:
            x = x[-1] - x[::-1]
            elev = elev[::-1]
            rhos = rhos[::-1, :]
            doi = doi[::-1]

        # Create boundary of polygons to be drawn
        xs = np.tile(np.repeat(x, 2)[1:-1][:, None], n_layers+1)

        depths = np.c_[np.zeros(depths.shape[0]), depths]
        ys = np.repeat(depths, 2, axis=0) + np.repeat(elev, 2, axis=0)[:, None]
        verts = np.c_[xs.flatten('F'), ys.flatten('F')]

        n_vert_row = verts.shape[0]
        connection = np.c_[np.arange(n_vert_row).reshape(-1, 2),
                           2*(n_models) +
                           np.arange(n_vert_row).reshape(-1, 2)[:, ::-1]]

        ie = (connection >= len(verts)).any(1)
        connection = connection[~ie, :]
        coordinates = verts[connection]

        
        # Determine color of each polygon
        cmaplist = [cmap(i) for i in range(cmap.N)]

        if discrete_colors:
            cmap = LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(vmin, vmax, n_bins)
            norm = BoundaryNorm(bounds, cmap.N)

        else:
            cmap = LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, 256)

            # define the bins and normalize
            bounds = np.linspace(vmin, vmax, 256)
            norm = BoundaryNorm(bounds, 256)

        # Create polygon collection
        coll = PolyCollection(coordinates, array=rhos.flatten('F'),
                              cmap=cmap, norm=norm, edgecolors=None)

        coll.set_clim(vmin=vmin, vmax=vmax)

        # Add polygons to plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        else:
            fig = ax.figure
        ax.text(0.5, 1.02, filename, fontsize=8, ha='center', transform=ax.transAxes)

        if contour:
            max_depth = 100
            centroid = np.mean(coordinates, axis=1)
            centroidx = centroid[:, 0].reshape((-1, n_models))
            centroidz = centroid[:, 1].reshape((-1, n_models))
            xc = np.vstack([centroidx[0, :], centroidx, centroidx[-1, :]])
            zc = np.vstack([np.zeros(n_models), centroidz, -
                           np.ones(n_models)*max_depth])
            val = np.c_[rhos[:, 0], rhos, rhos[:, -1]].T

            levels = np.linspace(vmin, vmax, 15)

            ax.contourf(xc, zc, val, cmap=cmap, levels=levels, extend='both')

        else:
            ax.add_collection(coll)

        # Blank out models below doi
        if doi is not None:
            doi = (np.repeat(elev, 2) - np.repeat(doi, 2)).tolist()

            doi.append(-1000)
            doi.append(-1000)
            doi.append(doi[-1])

            x_doi = xs[:, 0].tolist()

            x_doi.append(x_doi[-1])
            x_doi.append(x_doi[0])
            x_doi.append(x_doi[0])

            ax.fill(np.array(x_doi),  np.array(doi), edgecolor="none",
                    facecolor='w', alpha=0.8)

        if dists is not None:
            ax.set_xlabel('Distance [m]\n')
        else:
            ax.set_xlabel('Index')
            


        ax.set_ylabel('Elevation [m]')

        if cbar:

            if cbar_orientation == 'vertical':
                cbar = fig.colorbar(coll, label=cbar_label, ax=ax,
                                    orientation=cbar_orientation, shrink=0.8)

            else:
                cbar = fig.colorbar(coll, label=cbar_label, ax=ax,
                                    orientation=cbar_orientation, shrink=0.7,
                                    fraction=0.06, pad=0.2)

            if log:
                tick_locs = np.arange(int(np.floor(vmin)), int(np.ceil(vmax)))

                if tick_locs[-1] < vmax:
                    tick_locs = np.append(tick_locs, vmax)

                if tick_locs[0] < vmin:
                    tick_locs = np.append(vmin, tick_locs[1:])

                cbar.set_ticks(tick_locs)
                cbar.set_ticklabels(np.round(10**tick_locs).astype(int))
                cbar.ax.minorticks_off()

            else:
                tick_locs = np.arange(vmin, vmax+0.00001, int(vmax-vmin)/4)
                cbar.set_ticks(tick_locs)
                cbar.set_ticklabels(np.round(tick_locs))
                cbar.ax.minorticks_off()

        if zmin is None:
            zmin = np.nanmin(elev)+np.min(depths)

        if zmax is None:
            zmax = np.nanmax(elev)

        if xmin is None:
            xmin = 0

        if xmax is None:
            if dists is not None:
                xmax = dists[-1]
            else:
                xmax = n_models


        ax.set_ylim([zmin, zmax])
        ax.set_xlim([xmin, xmax])


        if len(plot_title) != 0:
            ax.set_title(plot_title, pad=15)

        ax.set_aspect(scale)

        ax.grid(which='both')

     
        fig.tight_layout()

        plt.draw()#Force the rendering, otherwise the x_ticks might not get set right for some cases 

        self.x_ticks = ax.get_xticks()
        print('x_ticks position 4:', self.x_ticks)     


    def TEMSounding(self, model_type, model_idx, sounding_idx, vmin=0, vmax=1000, ax=None):

        if model_type == 'tTEM':
            rhos = self.model.ttem_models[model_idx]['rhos'][sounding_idx, :]
            depths = self.model.ttem_models[model_idx]['depths'][sounding_idx, :]
            doi = self.model.ttem_models[model_idx]['doi_con'][sounding_idx]

        else:
            rhos = self.model.stem_models[model_idx]['rhos'][sounding_idx, :]
            depths = self.model.stem_models[model_idx]['depths'][sounding_idx, :]
            doi = self.model.stem_models[model_idx]['doi_con'][sounding_idx]

        if len(rhos) == len(depths):

            depths = depths[:-1]

        self.plot1D(rhos, depths, doi, vmin=vmin, vmax=vmax, ax=ax)


    def plot1D(self, rhos, depths, doi, log=True, vmin=0, vmax=1000,
               title=None, label=None, ax=None,
               col='k', ymin=None, ymax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 8))
        else:
            fig = ax.figure

        if doi is not None:
            idx = np.where(depths > doi)[0][0]
            ax.step(rhos[:idx], -np.insert(depths, 0, 0)[:idx], where='pre', c=col, label=label)

            ax.step(rhos[idx-1:], -np.insert(depths, 0, 0)[idx-1:],
                    where='pre', c='grey', ls='-',  alpha=0.8)
        else:
            ax.step(rhos, -np.insert(depths, 0, 0), where='pre', c=col, label=label)

        if log == True:
            ax.set_xscale('log')
            if vmin == 0:
                vmin = 1

        ax.set_xlim([vmin, vmax])
        ax.set_ylim([ymin, ymax])

        if label is not None:

            ax.legend()

        if title is not None:

            ax.set_title(title)

        ax.set_xlabel('Resistivity [Ohm.m]')

        ax.set_ylabel('Elevation [m]')

        ax.grid(True, which='major')
        fig.tight_layout()


    def findNearest(self, dict_data, x, y, dists=None, elev=None):
        """

        Parameters
        ----------
        dict_data : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        dists : TYPE
            DESCRIPTION.
        elev : TYPE
            DESCRIPTION.

        Returns
        -------
        dist_loc : TYPE
            DESCRIPTION.
        elev_loc : TYPE
            DESCRIPTION.
        min_dist : TYPE
            DESCRIPTION.

        """

        x_loc = dict_data['x']
        y_loc = dict_data['y']

        idx = np.argmin(((x_loc - x) ** 2 + (y_loc - y) ** 2) ** 0.5)
        min_dist = np.min(((x_loc - x) ** 2 + (y_loc - y) ** 2) ** 0.5)

        if elev is not None:
            elev = elev[idx]
        else:
            elev = dict_data['elevation']

        if dists is not None:

            dist_loc = dists[idx]

        else:
            dist_loc = np.nan

        return dist_loc, elev, min_dist, idx


    def addBorehole(self, bh_idx, ax, bh_width=0.2,
                    text_size=12, x_start=None, x_end=None):
        """


        Parameters
        ----------
        bh_list : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        bh_width : TYPE, optional
            DESCRIPTION. The default is dists[-1]/100.

        Returns
        -------
        None.

        """

        bh_dict = self.model.boreholes[bh_idx]

        if x_start is None:
            x_start = ax.get_xlim()[0]
            x_end = 10**(np.log10(ax.get_xlim()[1]) * bh_width)

        for i in range(bh_dict['n_layers']):

            coordinates = np.array(([x_start, -bh_dict['top_depths'][i]],
                                    [x_end, -bh_dict['top_depths'][i]],
                                    [x_end, -bh_dict['bot_depths'][i]],
                                    [x_start, -bh_dict['bot_depths'][i]],
                                    [x_start, -bh_dict['top_depths'][i]]))

            p = Polygon(coordinates, facecolor=bh_dict['colors'][i],
                        edgecolor='k', lw=0)

            ax.add_patch(p)

        coordinates = np.array(([x_start, -bh_dict['top_depths'][0]],
                                [x_end, -bh_dict['top_depths'][0]],
                                [x_end, -bh_dict['bot_depths'][-1]],
                                [x_start, -bh_dict['bot_depths'][-1]],
                                [x_start, -bh_dict['top_depths'][-1]]))

        p = Polygon(coordinates, facecolor='none', edgecolor='k', lw=1)

        ax.add_patch(p)


    def addBoreholes(self, profile_idx, ax, elev=None,
                    search_radius=150, bh_width=None, add_label=False,
                    text_size=12, shift=5, print_msg=False, alpha=1.0):
        """
        Add boreholes to a profile plot with transparency and labels.
        Returns a list of boreholes that were plotted.
        """
        xi = self.model.profiles[profile_idx]['xi']
        yi = self.model.profiles[profile_idx]['yi']
        dists = self.model.profiles[profile_idx]['distances']
        elevs = self.model.profiles[profile_idx]['elev']

        if bh_width is None:
            bh_width = dists[-1] / 40

        plotted_boreholes = []  # List to store the boreholes that were plotted

        for bh in self.model.boreholes:
            dist_loc, elev, min_dist, idx = self.findNearest(bh, xi, yi, dists, elevs)
            elev = bh['elevation']
            if min_dist < search_radius:
                x1 = dist_loc - bh_width / 2
                x2 = dist_loc + bh_width / 2

                for i in range(bh['n_layers']):
                    verts = np.array(([x1, elev - bh['top_depths'][i]],
                                    [x2, elev - bh['top_depths'][i]],
                                    [x2, elev - bh['bot_depths'][i]],
                                    [x1, elev - bh['bot_depths'][i]],
                                    [x1, elev - bh['top_depths'][i]]))

                    p = Polygon(verts, facecolor=bh['colors'][i], lw=0, alpha=alpha)
                    ax.add_patch(p)

                # Add boundary around log
                verts = np.array(([x1, elev - bh['top_depths'][0]],
                                [x2, elev - bh['top_depths'][0]],
                                [x2, elev - bh['bot_depths'][-1]],
                                [x1, elev - bh['bot_depths'][-1]],
                                [x1, elev - bh['top_depths'][0]]))

                p = Polygon(verts, facecolor='none', edgecolor='k', lw=1)
                ax.add_patch(p)

                if add_label:
                    ax.text(dist_loc, elev + shift, f"BH {bh['id']}",
                            horizontalalignment='center', weight='bold',
                            verticalalignment='top', fontsize=text_size)

                if print_msg:
                    print(f'Borehole {bh["id"]} is {min_dist / 1000:.3f} km from profile, it was included.')

                plotted_boreholes.append(bh)  # Add to the list of plotted boreholes

            else:
                if print_msg:
                    print(f'Borehole {bh["id"]} is {min_dist / 1000:.3f} km from profile, it was not included.')

        return profile_idx, plotted_boreholes   # Return the list of plotted boreholes


    def addNMRSoundings(self, profile_idx, nmr_list, param, ax, vmin=1, vmax=1000, elev=None,
                        log=True, cmap=plt.cm.viridis, n_bins=16, discrete_colors=False,
                        search_radius=100, model_width=None, print_msg=False):
        """

        Parameters
        ----------
        bh_list : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        bh_width : TYPE, optional
            DESCRIPTION. The default is dists[-1]/100.

        Returns
        -------
        None.

        """

        xi = self.model.profiles[profile_idx]['xi']
        yi = self.model.profiles[profile_idx]['yi']
        dists = self.model.profiles[profile_idx]['distances']
        elevs = self.model.profiles[profile_idx]['elev']

        n_models = len(nmr_list)

        if model_width is None:
            model_width = dists[-1] / 60

        for nmr in nmr_list:

            dist_loc, elev, min_dist, idx = self.findNearest(nmr, xi, yi,
                                                             dists, elevs)

            if min_dist < search_radius:
                x1 = dist_loc - model_width/2
                x2 = dist_loc + model_width/2

                for i in range(nmr['n_layers']-1):
                    verts = np.array(([x1, elev - nmr['top_depths'][i]],
                                      [x2, elev - nmr['top_depths'][i]],
                                      [x2, elev - nmr['bot_depths'][i]],
                                      [x1, elev - nmr['bot_depths'][i]],
                                      [x1, elev - nmr['top_depths'][i]]))

                    nmr['colors'] = self.getColors(nmr[param],
                                                   vmin=vmin, vmax=vmax,
                                                   log=log, cmap=cmap,
                                                   discrete_colors=discrete_colors)

                    if nmr['bot_depths'][i] > nmr['doi']:
                        p = Polygon(verts, facecolor=nmr['colors'][i],
                                    alpha= 0.3, lw=0)

                    else:
                        p = Polygon(verts, facecolor=nmr['colors'][i],
                                    lw=0)

                    ax.add_patch(p)

                verts = np.array(([x1, elev - nmr['top_depths'][0]],
                                  [x2, elev - nmr['top_depths'][0]],
                                  [x2, elev - nmr['bot_depths'][-1]],
                                  [x1, elev - nmr['bot_depths'][-1]],
                                  [x1, elev - nmr['top_depths'][0]]))

                p = Polygon(verts, facecolor='none', edgecolor='k', lw=0.5)

                ax.add_patch(p)
                
                if print_msg:
                    print('\033[1mTEM sounding %s is %.3f km from profile, it was included.\033[0m' % (nmr['id'], min_dist/1000))
                    

            else:
                if print_msg:
                    print('TEM sounding %s is %.3f km from profile, it was not included.' % (nmr['id'], min_dist/1000))

        if log:
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)
            

    def addTEMSoundings(self, profile_idx, stem_model_idx, ax, vmin=1, vmax=1000, elev=None,
                        log=True, cmap=plt.cm.turbo, n_bins=16, discrete_colors=False,
                        search_radius=100, model_width=None, print_msg=False):
        """

        Parameters
        ----------
        bh_list : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        bh_width : TYPE, optional
            DESCRIPTION. The default is dists[-1]/100.

        Returns
        -------
        None.

        """

        xi = self.model.profiles[profile_idx]['xi']
        yi = self.model.profiles[profile_idx]['yi']
        dists = self.model.profiles[profile_idx]['distances']
        elevs = self.model.profiles[profile_idx]['elev']
  

        n_models = len(self.model.stem_models[stem_model_idx]['x'])

        if model_width is None:
            model_width = dists[-1] / 60

        for i in range(n_models):

            sounding = {}

            sounding['id'] = str(i+1)

            sounding['x'] = self.model.stem_models[stem_model_idx]['x'][i]
            sounding['y'] = self.model.stem_models[stem_model_idx]['y'][i]
            sounding['rhos'] = self.model.stem_models[stem_model_idx]['rhos'][i]
            sounding['depths'] = self.model.stem_models[stem_model_idx]['depths'][i]
            sounding['doi'] = self.model.stem_models[stem_model_idx]['doi_con'][i]
            sounding['n_layers'] = len(sounding['rhos'])
            sounding['top_depths'] = np.insert(sounding['depths'], 0, 0)[:-1]
            sounding['bot_depths'] = sounding['depths']
            sounding['elev'] = self.model.stem_models[stem_model_idx]['elev'][i]

            dist_loc, elev, min_dist, idx = self.findNearest(sounding,
                                                             xi, yi,
                                                             dists, elevs)

            if min_dist < search_radius:
                x1 = dist_loc - model_width/2
                x2 = dist_loc + model_width/2

                for i in range(sounding['n_layers']-1):
                    verts = np.array(([x1, sounding['elev'] - sounding['top_depths'][i]],
                                      [x2, sounding['elev'] - sounding['top_depths'][i]],
                                      [x2, sounding['elev'] - sounding['bot_depths'][i]],
                                      [x1, sounding['elev'] - sounding['bot_depths'][i]],
                                      [x1, sounding['elev'] - sounding['top_depths'][i]]))

                    sounding['colors'] = self.getColors(sounding['rhos'],
                                                        vmin=vmin, vmax=vmax,
                                                        log=log, cmap=cmap,
                                                        discrete_colors=discrete_colors)

                    if sounding['bot_depths'][i] > sounding['doi']:
                        p = Polygon(verts, facecolor=sounding['colors'][i],
                                    alpha= 0.3, lw=0)

                    else:
                        p = Polygon(verts, facecolor=sounding['colors'][i],
                                    lw=0)

                    ax.add_patch(p)

                verts = np.array(([x1, sounding['elev'] - sounding['top_depths'][0]],
                                  [x2, sounding['elev'] - sounding['top_depths'][0]],
                                  [x2, sounding['elev'] - sounding['bot_depths'][-1]],
                                  [x1, sounding['elev'] - sounding['bot_depths'][-1]],
                                  [x1, sounding['elev'] - sounding['top_depths'][0]]))

                p = Polygon(verts, facecolor='none', edgecolor='k', lw=0.5)

                ax.add_patch(p)

                if print_msg:
                    print('\033[1mTEM sounding %s is %.3f km from profile, it was included.\033[0m' % (sounding['id'], min_dist/1000))

            else:
                if print_msg:
                    print('TEM sounding %s is %.3f km from profile, it was not included.' % (sounding['id'], min_dist/1000))

        if log:
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)

    def addProfilerSoundings(self, profile_idx, profiler_model_idx, ax, vmin=1, vmax=1000, elev=None,
                        log=True, cmap=plt.cm.turbo, n_bins=16, discrete_colors=False,
                        search_radius=100, model_width=None, print_msg=False):
  

        xi = self.model.profiles[profile_idx]['xi']
        yi = self.model.profiles[profile_idx]['yi']
        dists = self.model.profiles[profile_idx]['distances']
        elevs = self.model.profiles[profile_idx]['elev']
  

        n_models = len(self.model.profiler_models[profiler_model_idx]['x'])

        if model_width is None:
            model_width = dists[-1] / 60

        for i in range(n_models):

            sounding = {}

            sounding['id'] = str(i+1)

            sounding['x'] = self.model.profiler_models[profiler_model_idx]['x'][i]
            sounding['y'] = self.model.profiler_models[profiler_model_idx]['y'][i]
            sounding['rhos'] = self.model.profiler_models[profiler_model_idx]['rhos'][i]
            sounding['depths'] = self.model.profiler_models[profiler_model_idx]['depths'][i]
            sounding['doi'] = self.model.profiler_models[profiler_model_idx]['doi_con'][i]
            sounding['n_layers'] = len(sounding['rhos'])
            sounding['top_depths'] = np.insert(sounding['depths'], 0, 0)[:-1]
            sounding['bot_depths'] = sounding['depths']
            sounding['elev'] = self.model.profiler_models[profiler_model_idx]['elev'][i]

            dist_loc, elev, min_dist, idx = self.findNearest(sounding,
                                                             xi, yi,
                                                             dists, elevs)

            if min_dist < search_radius:
                x1 = dist_loc - model_width/2
                x2 = dist_loc + model_width/2

                for i in range(sounding['n_layers']-1):
                    verts = np.array(([x1, sounding['elev'] - sounding['top_depths'][i]],
                                      [x2, sounding['elev'] - sounding['top_depths'][i]],
                                      [x2, sounding['elev'] - sounding['bot_depths'][i]],
                                      [x1, sounding['elev'] - sounding['bot_depths'][i]],
                                      [x1, sounding['elev'] - sounding['top_depths'][i]]))

                    sounding['colors'] = self.getColors(sounding['rhos'],
                                                        vmin=vmin, vmax=vmax,
                                                        log=log, cmap=cmap,
                                                        discrete_colors=discrete_colors)

                    if sounding['bot_depths'][i] > sounding['doi']:
                        p = Polygon(verts, facecolor=sounding['colors'][i],
                                    alpha= 0.3, lw=0)

                    else:
                        p = Polygon(verts, facecolor=sounding['colors'][i],
                                    lw=0)

                    ax.add_patch(p)

                verts = np.array(([x1, sounding['elev'] - sounding['top_depths'][0]],
                                  [x2, sounding['elev'] - sounding['top_depths'][0]],
                                  [x2, sounding['elev'] - sounding['bot_depths'][-1]],
                                  [x1, sounding['elev'] - sounding['bot_depths'][-1]],
                                  [x1, sounding['elev'] - sounding['top_depths'][0]]))

                p = Polygon(verts, facecolor='none', edgecolor='k', lw=0.5)

                ax.add_patch(p)

                if print_msg:
                    print('\033[1mProfiler sounding %s is %.3f km from profile, it was included.\033[0m' % (sounding['id'], min_dist/1000))

            else:
                if print_msg:
                    print('Profiler sounding %s is %.3f km from profile, it was not included.' % (sounding['id'], min_dist/1000))

        if log:
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)


    def lithKey(self, ax=None, max_line_length=50, title='Geological Key',
                
                label_names=None, drop_idx=None):

        lith_cols = []
        lith_names = []
        top_depths = []

        # Iterate through the list of dictionaries and extract unique values
        for bh in self.model.boreholes:
            lith_cols.append(bh['colors'].tolist())
            lith_names.append(bh['lith_names'].tolist())
            top_depths.append(bh['top_depths'].tolist())

        lith_cols = np.concatenate(lith_cols)
        lith_names = np.concatenate(lith_names)
        top_depths = np.concatenate(top_depths)

        unique_cols = np.unique(lith_cols)

        lith_key = []
        lith_depth = []
        for col in unique_cols:

            idx = np.where(lith_cols == col)[0]

            lith_key.append(np.unique(lith_names[idx]))
            lith_depth.append(np.mean(top_depths[idx]))

        idx = np.argsort(lith_depth)
        
        print(idx)

        print(lith_key)

        lith_key = np.array(lith_key)[idx]
        unique_cols = np.array(unique_cols)[idx]

        if label_names is not None:
            
            lith_key = label_names

        if ax is None:

            fig, ax = plt.subplots(1, 1)
            
        if drop_idx is not None:
            label_names.pop(drop_idx)
            unique_cols=unique_cols[drop_idx+1:]
            #print(lith_key)
            #lith_key.pop(drop_idx)
            
        # Iterate through the geological units and plot colored squares
        y_position = 0.5  # Initial y-position for the first square
        for i in range(len(unique_cols)):

            # Plot a colored square
            ax.add_patch(plt.Rectangle((0, y_position), 0.4, 0.37, color=unique_cols[i]))
            if len(lith_key) > 1:
                label = ' / '.join(lith_key[i])
                
            else:
                label = lith_key[i]
                print(label)
            wrapped_label = '\n'.join(textwrap.wrap(label, max_line_length))

            ax.text(0.6, y_position + 0.15, wrapped_label, va='center', fontsize=12)

            # Update the y-position for the next square
            y_position -= 0.7

        # Set axis limits and labels
        ax.set_xlim(0, 4)
        ax.set_ylim(y_position, 1)
        ax.axis('off')  # Turn off axis
        ax.set_title(title)


    def addBoreholeLegend(self, plotted_boreholes_tuple, title="Borehole Legend", text_size=12, max_char=150):
            

        # Unpack the tuple to get profile_idx and plotted_boreholes
        profile_idx, plotted_boreholes = plotted_boreholes_tuple

        # Retrieve the filename for the current profile
        filename = self.model.profile_filenames[profile_idx]

        # Parameters for BoreholeLegendSizing
        min_text_size = 10  # Ensuring readability
        max_text_size = 16
        titlesize_factor = 1.25
        min_square_size = 0.5  # Ensure squares aren't too small
        max_square_size = 0.8  # Limit max square size to avoid overlap
        min_spacing = 0.2
        max_spacing = 0.5

        min_text_position = 0.8  # Move closer for fewer lithologies
        max_text_position = 1.0  # Standard position
        
        relevant_lithologies = {} #= number of different colors present in the profile

        # Collect lithologies from the provided boreholes
        for bh in plotted_boreholes:
            for lith_name, color in zip(bh['lith_names'], bh['colors']):
                if color not in relevant_lithologies:
                    relevant_lithologies[color] = []
                if lith_name not in relevant_lithologies[color]:
                    relevant_lithologies[color].append(lith_name)
        # Sort colors to have consistent order
        sorted_colors = sorted(relevant_lithologies.keys(), key=lambda c: relevant_lithologies[c])

        # Calculate dynamic sizes
        num_elements = len(relevant_lithologies)
        text_size = self.boreholeLegendSizing(num_elements, min_text_size, max_text_size)
        square_size = self.boreholeLegendSizing(num_elements, min_square_size, max_square_size, reverse=True)
        spacing = self.boreholeLegendSizing(num_elements, min_spacing, max_spacing)
        text_position = self.boreholeLegendSizing(num_elements, min_text_position, max_text_position, reverse=True)

        # start separate figure, set the legend's position and spacing
        fig, ax = plt.subplots(1, 1, figsize=(20, 5 + 2 * (num_elements - 2) / 6))
        ax.set_aspect('equal')
        ax.set_ylim(-num_elements - 1, 1)
        y_position = 0

        for color in relevant_lithologies:
            lith_names = relevant_lithologies[color]
            
            # Wrap the text if it is too long
            wrapped_label = textwrap.fill(' / '.join(lith_names), width=110)
            
            # Plot the color rectangle
            ax.add_patch(plt.Rectangle((0, y_position - 0.1), square_size, square_size, color=color))
            
            # Calculate dynamic vertical adjustment based on the square size
            vertical_adjustment = (square_size - (square_size/ num_elements+2)*0.1) / 2 # move the square size minus 10% of the ratio of squares to vertical axis size (num_elements+2)
            
            # Plot the wrapped text with dynamic adjustment
            ax.text(text_position, y_position + vertical_adjustment, wrapped_label, va='center', fontsize=text_size)
            
            # Update the y-position based on the number of lines
            num_lines = wrapped_label.count('\n') + 1
            y_position -= square_size + spacing

        # Adjust title font size dynamically as well
        title_font_size = text_size*titlesize_factor
        filename_font_size = title_font_size / 2


        # Set axis limits and labels
        ax.set_title('Borehole Legend', fontsize=title_font_size, loc='left', weight='bold')
        # Add the filename next to the title with smaller font size
        ax.text(-0.05, 1.08, f" {filename}", fontsize=filename_font_size, ha='left', va='center', transform=ax.transAxes, weight='normal')
        ax.axis('off')

        return fig, ax #necessary for saving the figures in your implementation

      
    def boreholeLegendSizing(self, num_elements, min_size, max_size, reverse=False):
        scale_factor = max(0, min(1, (8 - num_elements) / 6))  # 8 can be adjusted to your maximum expected elements
        if reverse:
            return max_size - scale_factor * (max_size - min_size)
        else:
            return min_size + scale_factor * (max_size - min_size) 
        

    def profileMap(self, profile_idx, tif_file=None, length_tick_size=100, modeltype_1_size=1, 
                modeltype_2_size=10, borehole_size=10, boreholetext_size=8, 
                legend_size=10, legend_text_size=10, legend_location=None, 
                buffer=0.1, ax=None):
        
        profile = self.model.profiles[profile_idx]
        profile_name = self.model.profile_filenames[profile_idx]
        xi = profile['xi']
        yi = profile['yi']
        dists = profile['distances']
        
        # Getting EPSG code from the tTEM model
        epsg_code = self.model.ttem_models[0].get('epsg', 'epsg:4326')

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        # Define map extent (+buffer)
        x_min, x_max = min(profile['x']), max(profile['x'])
        y_min, y_max = min(profile['y']), max(profile['y'])
        x_min_buff = x_min - buffer * (x_max - x_min)
        x_max_buff = x_max + buffer * (x_max - x_min)
        y_min_buff = y_min - buffer * (y_max - y_min)
        y_max_buff = y_max + buffer * (y_max - y_min)

        if tif_file:
            self.add_geotiff_background(ax, tif_file, x_min_buff, x_max_buff, y_min_buff, y_max_buff)
        else:
            cx.add_basemap(ax, crs=epsg_code, source=cx.providers.Esri.WorldImagery, attribution=False)
        
        ax.set_xlim(x_min_buff, x_max_buff)
        ax.set_ylim(y_min_buff, y_max_buff)

        x_size = x_max_buff - x_min_buff
        y_size = y_max_buff - y_min_buff




        # Plot profile line without dots
        ax.plot(profile['x'], profile['y'], c='black', lw=2, label=profile_name, zorder=10)

        # Compute tangent vectors (dx/ds, dy/ds)
        dx_ds = np.gradient(xi, dists)
        dy_ds = np.gradient(yi, dists)

        # Compute normal vectors (perpendicular to tangent)
        normal_x = -dy_ds
        normal_y = dx_ds

        # Normalize normal vectors
        magnitude = np.hypot(normal_x, normal_y)
        normal_x /= magnitude
        normal_y /= magnitude

        # Define the length of the marker lines
        line_length = max(x_size,y_size)/100  # Adjust as needed
        """ normalize this to the length of the longest axis """

    

        # Use x_ticks to add markers
        for xtick in self.x_ticks:
            # Get the position along the profile
            x_pos = np.interp(xtick, dists, xi)
            y_pos = np.interp(xtick, dists, yi)
            
            # Interpolate the normal vector at xtick
            n_x = np.interp(xtick, dists, normal_x)
            n_y = np.interp(xtick, dists, normal_y)
            
            # Compute the start and end points of the marker line
            x_start = x_pos - (line_length / 2) * n_x
            y_start = y_pos - (line_length / 2) * n_y
            x_end = x_pos + (line_length / 2) * n_x
            y_end = y_pos + (line_length / 2) * n_y
            
            # Draw the line
            ax.plot([x_start, x_end], [y_start, y_end], color='black', linewidth=2)

        # Define an offset for the distance ticks relative to the profile line (this will scale with the axis size)
        offset_factor = -0.03  # Proportional offset factor
        text_shift_factor = 0.02  # Proportional shift factor for centering the text
 
        offset_distance = offset_factor * max(x_size, y_size)
        text_shift_distance = text_shift_factor * max(x_size, y_size)

        # Function to normalize the angle to face "upwards" (between 0 and 180 degrees)
        def adjust_text_angle(angle):
            if angle < -90 or angle > 90:
                return angle + 180  # Flip the text direction
            return angle

        # Adding distance ticks at the start of the profile (0m)
        n_x_start = normal_x[0]
        n_y_start = normal_y[0]
        t_x_start = dx_ds[0]  # Tangent x-component at the start
        t_y_start = dy_ds[0]  # Tangent y-component at the start

        x_pos_start = profile['x'][0] + offset_distance * n_x_start - text_shift_distance * t_x_start
        y_pos_start = profile['y'][0] + offset_distance * n_y_start - text_shift_distance * t_y_start
        # Set rotation perpendicular to the tangent, adjust the angle to always face upwards
        start_angle = np.degrees(np.arctan2(n_y_start, n_x_start)) + 90
        start_angle_adjusted = adjust_text_angle(start_angle)
        ax.text(x_pos_start, y_pos_start, '0 m', color='black', fontsize=10, 
                rotation=start_angle_adjusted, ha='center', va='center')

        # Adding distance ticks at the end of the profile
        n_x_end = normal_x[-1]
        n_y_end = normal_y[-1]
        t_x_end = dx_ds[-1]  # Tangent x-component at the end
        t_y_end = dy_ds[-1]  # Tangent y-component at the end

        x_pos_end = profile['x'][-1] + offset_distance * n_x_end - text_shift_distance * t_x_end
        y_pos_end = profile['y'][-1] + offset_distance * n_y_end - text_shift_distance * t_y_end
        # Set rotation perpendicular to the tangent, adjust the angle to always face upwards
        end_angle = np.degrees(np.arctan2(n_y_end, n_x_end)) + 90
        end_angle_adjusted = adjust_text_angle(end_angle)
        ax.text(x_pos_end, y_pos_end, f'{int(dists[-1])} m', color='black', fontsize=10, 
                rotation=end_angle_adjusted, ha='center', va='center')

       # Plot tTEM models
        for ttem in self.model.ttem_models:
            mask = (ttem['x'] >= x_min_buff) & (ttem['x'] <= x_max_buff) & (ttem['y'] >= y_min_buff) & (ttem['y'] <= y_max_buff)
            ax.scatter(ttem['x'][mask], ttem['y'][mask], s=modeltype_1_size, c='blue', label='tTEM')

        # Plot sTEM models
        for stem in self.model.stem_models:
            mask = (stem['x'] >= x_min_buff) & (stem['x'] <= x_max_buff) & (stem['y'] >= y_min_buff) & (stem['y'] <= y_max_buff)
            ax.scatter(stem['x'][mask], stem['y'][mask], s=modeltype_2_size, c='red', label='sTEM')
        # Plot profiler models
        for profiler in self.model.profiler_models:
            mask = (profiler['x'] >= x_min_buff) & (profiler['x'] <= x_max_buff) & (profiler['y'] >= y_min_buff) & (profiler['y'] <= y_max_buff)
            ax.scatter(profiler['x'][mask], profiler['y'][mask], s=modeltype_2_size, c='gold', label='profiler')
        # Plot boreholes
        for borehole in self.model.boreholes:
            if x_min_buff <= borehole['x'] <= x_max_buff and y_min_buff <= borehole['y'] <= y_max_buff:
                ax.scatter(borehole['x'], borehole['y'], s=borehole_size, c='cyan', label='Borehole')
                ax.text(borehole['x']+10, borehole['y']+10, f'{borehole["id"]}', fontsize=boreholetext_size, color='cyan')
        
        # Get the legend handles and labels once
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))

        if legend_location == None:

            # Decide legend location and bbox_to_anchor based on axis ratio
            if y_size / x_size > 10:
                legend_location = 'upper left'
                bbox_to_anchor = (1, 0.5)
            elif x_size / y_size > 10:
                legend_location = 'upper center'
                bbox_to_anchor = (0.5, 1.15)
            else:
                # Use the existing legend_location, which you might have passed as an argument
                # Ensure legend_location is defined
                bbox_to_anchor = None  # No need for bbox_to_anchor in this case
                legend_location = 'upper right'

            # Call ax.legend() once
            ax.legend(unique_labels.values(), unique_labels.keys(),
                    loc=legend_location, bbox_to_anchor=bbox_to_anchor,
                    fontsize=legend_text_size)
        else:
            bbox_to_anchor = None
            ax.legend(unique_labels.values(), unique_labels.keys(),
            loc=legend_location, bbox_to_anchor=bbox_to_anchor,
            fontsize=legend_text_size)
        
        

        # Display the EPSG code in the bottom right below the Easting label
        ax.text(1, -0.1, f'EPSG: {epsg_code}', transform=ax.transAxes, fontsize=6,
                verticalalignment='top', horizontalalignment='right')

        # Control the size of the axis labels
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_xlabel("Easting [m]", fontsize=12)
        ax.set_ylabel("Northing [m]", fontsize=12)

        # Dynamically reducing tick labels based on axis size
      

        def round_to_nearest_100(value):
            return int(np.round(value / 100.0) * 100)

        if y_size/x_size>7:
            # Set one ticks at approximately mid from the edges, rounded to nearest 100
            x_ticks = [round_to_nearest_100(x_min + 0.2 * x_size)]
            ax.set_xticks(x_ticks)

        elif y_size/x_size>3:
            # Set two ticks at approximately 20% and 80% from the edges, rounded to nearest 100
            x_ticks = [round_to_nearest_100(x_min + 0.2 * x_size), round_to_nearest_100(x_min + 0.8 * x_size)]
            ax.set_xticks(x_ticks)
        
        
        if x_size/y_size>10:
            # Set one ticks at approximately mid from the edges, rounded to nearest 100
            y_ticks = [round_to_nearest_100(y_min + 0.2 * y_size)]
            ax.set_yticks(y_ticks)        
        
        elif x_size/y_size>5:
            # Set two ticks at approximately 20% and 80% from the edges, rounded to nearest 100
            y_ticks = [round_to_nearest_100(y_min + 0.2 * y_size), round_to_nearest_100(y_min + 0.8 * y_size)]
            ax.set_yticks(y_ticks)



        return fig, ax


    def add_geotiff_background(self, ax, tif_file, x_min, x_max, y_min, y_max):
        
        try:
            with rasterio.open(tif_file) as src:
                # Calculate pixel coordinates for the bounding box
                row_min, col_min = src.index(x_min, y_max)  # Top-left corner
                row_max, col_max = src.index(x_max, y_min)  # Bottom-right corner

                # Define the window (row_start, row_stop, col_start, col_stop)
                window = ((row_min, row_max), (col_min, col_max))

                # Read the data within the window for all bands
                data = src.read(window=window)

                # Check if it's multi-band (RGB)
                if src.count == 4:
                    # Assuming the image is RGB (bands 1, 2, 3 are R, G, B respectively)
                    rgb_data = data[:3, :, :]  # Extract the first three bands (R, G, B)
                    
                    # Display the RGB image
                    show(rgb_data, transform=src.window_transform(window), ax=ax)
                else:
                    # For single-band images, display as is
                    show(data, transform=src.window_transform(window), ax=ax)
                
                # Explicitly set the axis limits to match the profile's bounding box
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

        except ValueError as e:
            # Catching the error and notifying the user
            print(f"Warning: {e}. No background map will be shown for this profile. There could be that no GEOTIFF was loaded, or that \
                  it is not big enough")
