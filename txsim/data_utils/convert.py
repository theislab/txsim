import numpy as np
import pandas as pd
import warnings


def convert_polygons_to_label_image(df, tech="Xenium") -> np.array:
    """
    
    #TODO: decide on arguments, instead of tech maybe sth else? And add some extra fct/wrapper for tech.
    
    """
    
    # TODO: Implement this function. (have it laying around somewhere.)
    
    
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
