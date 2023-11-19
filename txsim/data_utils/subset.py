import warnings
from pathlib import Path
from typing import List, Tuple
import tifffile
import numpy as np
import pandas as pd



def generate_and_save_data_subset(
        img_paths : List[str],
        spots_path : str,
        save_dir : str,
        n_pixel_x : int = 2000,
        n_pixel_y : int = 2000,
        max_ct_heterogeneity : bool = True,
        x_min : int = None,
        y_min : int = None,
        x_key : str = "x",
        y_key : str = "y",
    ) -> None:
    """ Generate a subset of the data 
    
    Arguments
    ---------
    img_paths : list of str
        Paths to the images.
    spots_path : str
        Path to the spots.
    save_dir : str
        Directory to save the subset to.
    n_pixel_x : int
        Number of pixels in x direction.
    n_pixel_y : int
        Number of pixels in y direction.
    max_ct_heterogeneity : bool
        Whether to select the crop with the highest cell type heterogeneity. If not given set x_min and y_min.
    x_min : int
        Minimum x coordinate of the crop. If not given set max_ct_heterogeneity to True.
    y_min : int
        Minimum y coordinate of the crop. If not given set max_ct_heterogeneity to True.
        
    Returns
    -------
    None
    """
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    spots = pd.read_csv(spots_path)
    
    if max_ct_heterogeneity:
        with tifffile.TiffFile(img_paths[0]) as tif:
            img_shape = tif.pages[0].shape
    
        y_min, x_min = get_crop_with_high_ct_heterogeneity(
            spots, n_pixel_x=n_pixel_x, n_pixel_y=n_pixel_y,img_shape=img_shape
        )

    # Generate subset of spots table
    spots = spots[
        (spots[x_key] >= x_min) & (spots[x_key] < x_min + n_pixel_x) & 
        (spots[y_key] >= y_min) & (spots[y_key] < y_min + n_pixel_y)
    ].copy()
    spots["x"] = spots["x"] - x_min
    spots["y"] = spots["y"] - y_min
    spots.to_csv(Path(save_dir, Path(spots_path).name), index=False)
    
    # Generate subset of images
    for img_path in img_paths:
        with tifffile.TiffFile(img_path) as tif:
            img = tif.pages[0].asarray()
            img = img[y_min:y_min+n_pixel_y, x_min:x_min+n_pixel_x]
        tifffile.imwrite(Path(save_dir, Path(img_path).name), img)
    


def get_crop_with_high_ct_heterogeneity(
        spots_df:pd.DataFrame, 
        n_pixel_x:int, 
        n_pixel_y:int, 
        img_shape:Tuple[int],
        spots_df_key:str="celltype",
    ) -> Tuple[int, int]:
    """ Get crop with high cell type heterogeneity
    
    Arguments
    ---------
    spots_df : pandas.DataFrame
        DataFrame containing spots.
    n_pixel_x : int
        Number of pixels in x direction.
    n_pixel_y : int
        Number of pixels in y direction.
    img_shape : tuple
        y, x shape of the image.
        
    Returns
    -------
    y_min, x_min : int
        Coordinates of crop
    """

    df = spots_df

    # Iterate through full image in crops
    # For each crop measure the number of cells of the cell type with the lowest number of cells (save in a 2d array)

    n_x = img_shape[1] // n_pixel_x #+ bool(img_shape[1] % n_pixel_x)
    n_y = img_shape[0] // n_pixel_y #+ bool(img_shape[0] % n_pixel_y)

    df[spots_df_key] = df[spots_df_key].astype("category")

    min_cell_counts = np.zeros((n_x, n_y))
    counts_i_j = []

    for i in range(n_x):
        for j in range(n_y):
            df_crop = df[(df["x"] > i*n_pixel_x) & (df["x"] < (i+1)*n_pixel_x) & (df["y"] > j*n_pixel_y) & (df["y"] < (j+1)*n_pixel_y)].copy()
            min_cell_counts[i,j] = df_crop[spots_df_key].value_counts().min()
            counts_i_j.append([df_crop[spots_df_key].value_counts().min(), i, j])

    x_idx = counts_i_j[np.array(counts_i_j)[:, 0].argmax()][1]
    y_idx = counts_i_j[np.array(counts_i_j)[:, 0].argmax()][2]

    ## Value counts in crop
    #i = x_idx
    #j = y_idx
    #df_crop = df[(df["x"] > i*n_pixel_x) & (df["x"] < (i+1)*n_pixel_x) & (df["y"] > j*n_pixel_y) & (df["y"] < (j+1)*n_pixel_y)].copy()
    #df_crop[spots_df_key].value_counts()

    x_min = x_idx*n_pixel_x
    x_max = (x_idx+1)*n_pixel_x
    y_min = y_idx*n_pixel_y
    y_max = (y_idx+1)*n_pixel_y

    return y_min, x_min



##########
# Tiling #
##########

def get_tile_intervals(x:int,y:int,nx:int,ny:int,x_len:int,y_len:int,extend_n_pixels:int):
    """Get x,y start and end points of a tile
    
    Arguments
    ---------
    x: int
        x-id of the tile
    y: int
        y-id of the tile
    nx: int
        Number of tiles along x (total number of tiles = nx * ny)
    ny: int
        Number of tiles along y (total number of tiles = nx * ny)
    x_len: int
        Image x-length
    y_len: int
        Image y-length
    extend_n_pixels: int
        Nr of pixels to add at the tile borders (don't extend further than (0,x_len) and (0,y_len))
        
    Returns
    -------
    x_min, x_max, y_min, y_max: ints
        Interval start and end points
    offset_x, offset_y: 
        Tile offsets due to `extend_n_pixels`. Can be either 0 (tile at left/top border) or `extend_n_pixels`.
    
    """
    
    tile_len_x = x_len // nx
    tile_len_y = y_len // ny
    
    if extend_n_pixels > min(tile_len_y,tile_len_x):
        raise ValueError(
            f"The tiles are smaller ({(tile_len_y,tile_len_x)}) than the tile extension ({extend_n_pixels})"
        )
    
    x_min = (x * tile_len_x) - extend_n_pixels if x > 0 else 0
    x_max = ((x+1) * tile_len_x) + extend_n_pixels if x < (nx-1) else x_len
    y_min = (y * tile_len_y) - extend_n_pixels if y > 0 else 0
    y_max = ((y+1) * tile_len_y) + extend_n_pixels if y < (ny-1) else y_len
    
    offset_x = extend_n_pixels if x > 0 else 0
    offset_y = extend_n_pixels if y > 0 else 0
    
    return x_min, x_max, y_min, y_max, offset_x, offset_y


def get_tile_mask(spots, x_min, x_max, y_min, y_max, x_col="x", y_col="y"):
    """
    """
    mask = (spots[x_col].round() >= x_min) & (spots[x_col].round() < x_max) 
    mask &=(spots[y_col].round() >= y_min) & (spots[y_col].round() < y_max)
    return mask

def get_nr_of_spots_in_tile(spots, x_min, x_max, y_min, y_max, x_col="x", y_col="y"):
    """
    """
    mask = get_tile_mask(spots, x_min, x_max, y_min, y_max, x_col=x_col, y_col=y_col)
    return mask.sum()

def get_spots_tile(spots: pd.DataFrame, x_min: int, x_max: int, y_min: int, y_max:int, x_col="x", y_col="y"):
    """ Subset spots table to tile and translate coordinates
    """
    mask = get_tile_mask(spots, x_min, x_max, y_min, y_max, x_col=x_col, y_col=y_col)
    spots = spots.loc[mask]
    spots[x_col] -= x_min
    spots[y_col] -= y_min
    return spots
    
    
    

def find_optimal_tile_division_for_nspots_limit(img_shape, spots, n_spots_max=10000000, extend_n_pixels=20):
    """
    
    """
    
    y_len, x_len = img_shape
    
    n_spots = len(spots)
    
    # Start with the minimal expected number of tiles
    n_tiles = n_spots // n_spots_max + bool(n_spots % n_spots_max)
    
    while n_spots > n_spots_max:
        
        # Get x, y tile splitting that produces the most square like tiles (to minimize tile perimeter)
        ny, nx = find_optimal_tile_division(img_shape, n_tiles)
        
        print(f"n_tiles = {n_tiles}, n_spots = {n_spots}, nx = {nx}, ny = {ny}")
        
        for x in range(nx):
            for y in range(ny):
                
                # Get tile interval (including tile extensions)
                x_min, x_max, y_min, y_max, _, _ = get_tile_intervals(x, y, nx, ny, x_len, y_len, extend_n_pixels)
                          
                # Update n_spots (Start with first tile, update if tile with higher n_spots is found)
                n_spots_tile = get_nr_of_spots_in_tile(spots, x_min, x_max, y_min, y_max, x_col="x", y_col="y")
                if (n_spots_tile > n_spots) or (x==0 and y==0):
                    n_spots = n_spots_tile
                
                print(f"\tx = {x}, y = {y}, n_spots_tile = {n_spots_tile}, n_spots = {n_spots}")
                    
        n_tiles += 1
                    
    return ny, nx





def find_multiplication_factors(n):
    """Find pairs of factors of an integer n."""
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append((i, n // i))
    return factors

def find_optimal_tile_division(img_shape, n):
    """
    Determine the optimal division of a rectangle (image) into tiles with the goal of making 
    the tiles as square-like as possible.

    Arguments
    ---------
    img_shape:
        shape of image.
    n: The number of tiles to divide the rectangle into.

    Returns
    -------
    tuple: A pair (divisions_y, divisions_x) indicating the number of divisions 
           along the y-axis and x-axis, respectively, to achieve the most square-like tiles.
    """
    
    rect_y, rect_x = img_shape
    
    longer_side_x = (rect_x > rect_y)
    if longer_side_x:
        width, length = (rect_x, rect_y)
    else:
        width, length = (rect_y, rect_x)
        
    # Find all factor pairs of tile_count
    factor_pairs = find_multiplication_factors(n)

    # Initialize variables to store the optimal division
    optimal_divisions_wid = None
    optimal_divisions_len = None
    min_diff = float('inf')

    # Iterate through each pair of factors
    for divisions_len, divisions_wid in factor_pairs:
        # Calculate aspect ratios for each division
        tile_aspect_ratio = (width / divisions_wid) / (length / divisions_len)
        aspect_ratio_difference = abs(tile_aspect_ratio - 1)

        # Update the optimal divisions if this one is closer to square
        if aspect_ratio_difference < min_diff:
            min_diff = aspect_ratio_difference
            optimal_divisions_wid = divisions_wid
            optimal_divisions_len = divisions_len

    if longer_side_x:
        optimal_divisions_y, optimal_divisions_x = (optimal_divisions_len, optimal_divisions_wid)
    else:
        optimal_divisions_y, optimal_divisions_x = (optimal_divisions_wid, optimal_divisions_len)
        
    return (optimal_divisions_y, optimal_divisions_x)