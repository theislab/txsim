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
