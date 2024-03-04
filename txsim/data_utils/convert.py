import numpy as np
import pandas as pd
import warnings

import geopandas as gpd
from shapely.geometry import Polygon
from rasterio import features
from rasterio import Affine

def create_polygon(x_str, y_str):
    '''
     Function to create a Polygon from x, y coordinates
     Arguments: 
         x_str: string of commma separated floats denoting the x coordinates of the polygon
         y_str: string of comma sepatated floats denoting the y coordinates of the polygon
     Returns:
         Polygon object (shapely.geometry.Polygon)
    '''
    x = [float(value) for value in x_str.split(',')]
    y = [float(value) for value in y_str.split(',')]
    return Polygon(list(zip(x, y)))

def assign_spots_to_cells(cell_df,df,n_z_planes):
    '''
    Function to assign RNA spots to cells given cell boundaries (i.e. assigning points to polygons)
    Arguments:
        cell_df: pandas dataframe containing cell boundaries 
            necessary columns: 'cell_id', 'boundaryX_z0', 'boundaryY_z0', ..., 'boundaryX_zN', 'boundaryY_zN', where N= n_z_planes-1
            cell boundaries are polygons stored as a string of comma separated floats
        df: pandas dataframe containing RNA spot coordinates
            necessary columns: 'x', 'y', 'z'
        n_z_planes: int the number of z planes.
    Returns:
        results_dfs: list of pandas dataframes for each z_plane where RNA spots are assigned to their corresponding cell_id
    '''



    # List to store the results of spatial joins for each z-plane
    result_dfs = []

    # Iterate over z planes
    for z in range(n_z_planes):
        mask = cell_df['boundaryX_z' + str(z)].apply(lambda x: isinstance(x, str))
        rows_to_process = [(row, z) for index, row in cell_df[mask].iterrows()]

        # Create polygons for the current z-plane
        polygons_z = [create_polygon(row['boundaryX_z' + str(z)], row['boundaryY_z' + str(z)]) for index, row in cell_df[mask].iterrows()]

        # Create a GeoDataFrame from the list of Polygons for the current z-plane
        gdf_polygons_z = gpd.GeoDataFrame(geometry=polygons_z)

        # Add a new column 'cell_id' to gdf_polygons_z with cell IDs from cell_df
        gdf_polygons_z['cell_id'] = cell_df.loc[mask, 'cell_id'].tolist()

        # Filter points for the current z-plane
        df_z = df[df['z'] == z].copy() 
        
        # Create GeoDataFrame with points for the current z-plane
        df_z['geometry'] = gpd.points_from_xy(df_z['x'], df_z['y'])
        gdf_points_z = gpd.GeoDataFrame(df_z, geometry='geometry')

        # Spatial join for the current z-plane
        result_z = gpd.sjoin(gdf_points_z, gdf_polygons_z, how='left', op='within')

        # Append the result DataFrame to the list
        result_dfs.append(result_z)
    return results_dfs



def convert_polygons_to_label_image(df, X_column, Y_column,complete_img_size_x,complete_img_size_y):
    '''
    Create label image from a dataframe with polygons
    Note that polygon coordinates should not be negative, if so please shift the coordinates before applying the function 
    Arguments
    ---------
    df:                     pandas.Dataframe
                            Dataframe containing polygon coordinates
    X_column:               str
                            Name of the column with x coordinates of the polygons (coordinates are a string of comma separated values)
    Y_column:               str
                            Name of the column with y coordinates of the polygons (coordinates are a string of comma separated values)
    complete_img_size_x:    int 
                            image size in x
    complete_img_size_y:    int  
                            image size in y

    Returns:
    ----------
    cell_id_image:          np.array 
                            Array containing the label image where each polygon has a different integer label

    '''
    # Decode polygon coordinates and filter out NaN values
    df = df.dropna(subset=[X_column])
    df = df.dropna(subset=[Y_column])
    df.loc[:, 'polygon_coordinates'] = df.apply(lambda row: create_polygon(row[X_column], row[Y_column]), axis=1)
   

    if df.empty:
        print("No valid polygons found.")
        return
    
   
    cell_id_image = np.zeros((complete_img_size_y, complete_img_size_x), dtype=np.uint32)

    cell_id = 1 
    for polygon in df["polygon_coordinates"]:
        try: 
            minx, miny, maxx, maxy = polygon.bounds
            if (int(maxx - minx)==0) or (int(maxy-miny)==0): # Skip polygons with zero width or height
                print(f"Skipping invalid Polygon at cell_id = {cell_id}")
                continue
            minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)
            image = features.rasterize([(polygon, cell_id)],
                              out_shape=(maxy-miny, maxx-minx),
                              transform=Affine.translation(minx, miny),
                              fill=0,
                              dtype=np.uint32)
        except AttributeError:
            print(f"Invalid Polygon at cell_id = {cell_id}")
            continue
        cell_id_image[miny:maxy, minx:maxx] = np.where(
                cell_id_image[miny:maxy,minx:maxx] == 0, image, cell_id_image[miny:maxy, minx:maxx])
        cell_id = cell_id + 1
    return cell_id_image
    
    
def convert_coordinates_to_pixel_space(
    df: pd.DataFrame,
    xy_resolution: float,
    x_col: str = "x",
    y_col: str = "y",
    z_col: str = "z",
    scale_z_as_xy: bool = True,
    z_resolution: float = None,
):
    """ Convert coordinates to pixel space.
    
    Note that the z dimension is scaled as the xy dimension by default. In real pixel space the 3d distance would not be
    euclidean anymore. Most processing methods that work on the spots coordinates assume euclidean distance. Therefore 
    we scale the z dimension to match the xy dimension.
    
    Arguments
    ---------
    df : pandas.DataFrame
        DataFrame containing coordinates.
    xy_resolution : float
        Resolution of the image in microns.
    x : str
        Name of the column containing x-coordinates.
    y : str
        Name of the column containing y-coordinates.
    z : str
        Name of the column containing z-coordinates.
    scale_z_as_xy : bool
        Whether to scale the z dimension as the xy dimension.
    z_resolution : float
        Resolution of the z dimension in microns.
        
    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing coordinates in pixel space.
        
    """

    # Convert coordinates to pixel space.
    df[x_col] = df[x_col] / xy_resolution
    df[y_col] = df[y_col] / xy_resolution
    if z_col in df.columns:
        if scale_z_as_xy:
            df[z_col] = df[z_col] / xy_resolution
        else:
            df[z_col] = df[z_col] / z_resolution
    else:
        warnings.warn('z column not found in DataFrame. Not scaling z dimension.')

    return df
