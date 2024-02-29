import numpy as np


def _get_bin_ids(df, region_range, bins, x_col='x', y_col='y'):
    """ Assigns bin IDs to each spot based on its x and y coordinates (-1 for points outside of region_range).
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the x and y coordinates.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
        
    Returns
    -------
    pd.DataFrame
        The input dataframe with two additional columns 'y_bin' and 'x_bin' containing the bin IDs.
        
    """
    # Unpack region_range and bins
    (y_min, y_max), (x_min, x_max) = region_range
    ny, nx = bins
    
    # Get mask for points outside of region_range
    mask = (df[y_col] < y_min) | (df[y_col] > y_max) | (df[x_col] < x_min) | (df[x_col] > x_max)

    # Create bin edges
    y_bins = np.linspace(y_min, y_max, ny)
    x_bins = np.linspace(x_min, x_max, nx)

    # Assign bins to y and x coordinates (-1 to start indexing from 0)
    df['y_bin'] = np.digitize(df[y_col], y_bins) - 1  
    df['x_bin'] = np.digitize(df[x_col], x_bins) - 1  
    
    # Set bins outside of region_range to -1
    df.loc[mask, 'y_bin'] = -1
    df.loc[mask, 'x_bin'] = -1

    return df