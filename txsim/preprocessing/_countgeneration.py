import numpy as np
import anndata as ad
from anndata import AnnData
import pandas as pd
import scanpy as sc
from typing import Optional

def generate_adata(
    molecules: str,
    cell_types: Optional[str] = None
) -> AnnData:
    """Generate an AnnData object with counts from molecule data

    Parameters
    ----------
    molecules : str
        File name of CSV containing genes and cell assignments
    cell_types : Optional[str]
        File name containing cell type for each cell, by default None

    Returns
    -------
    AnnData
        Populated count matrix
    """
    
    #Read assignments
    spots = pd.read_csv(molecules)
    spots = spots[spots['cell'] > 0]

    #Generate blank, labelled count matrix
    X = np.zeros([ len(pd.unique(spots['cell'])), len(pd.unique(spots['Gene'])) ])
    adata = ad.AnnData(X, dtype = X.dtype)
    adata.obs['cell_id'] = pd.unique(spots['cell'])
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = pd.unique(spots['Gene'])

    #Populate matrix using assignments
    for gene in adata.var_names:
        cts = spots[spots['Gene'] == gene ]['cell'].value_counts()
        adata[:, gene] = cts.reindex(adata.obs['cell_id'], fill_value = 0)

    if cell_types is not None:
        temp = pd.read_csv(cell_types, header=None, index_col = 0)
        adata.obs['celltype'] = pd.Categorical(temp[1][adata.obs['cell_id']])

    return adata

def calculate_alpha_area(
    adata: AnnData,
    molecules: str,
    alpha: float = 0
) -> np.ndarray:
    """Calculate and store the alpha shape area of the cell given a set of points (genes). 
    Uses the Alpha Shape Toolboox: https://alphashape.readthedocs.io/en/latest/readme.html 

    Parameters
    ----------
    adata : AnnData
        AnnData object with cells as `obs` and genes as `var`
    molecules : str
        File name of CSV containing genes and cell assignments
    alpha : float, optional
        The alpha parameter a, used to calculate the alpha shape, by default 0

    Returns
    -------
    np.ndarray
        Returns the area vector stored in `adata.obs['alpha_area']` as a numpy array
    """    

    import alphashape
    from descartes import PolygonPatch
    
    #Read assignments
    spots = pd.read_csv(molecules)
    spots = spots[spots['cell'] != 0]

    # Calculate alpha shape
    # If there are <3 molecules for a cell, use the mean area per molecule
    # times the number of molecules in the cell
    area_vec = np.zeros([adata.n_obs])
    shape_vec = []
    for i in range(adata.n_obs):
        dots = pd.concat(
            [spots[spots['cell'] == adata.obs['cell_id'][i]].x,
            spots[spots['cell'] == adata.obs['cell_id'][i]].y],
            axis=1
        )
        pts = list(dots.itertuples(index=False, name=None))
        alpha_shape = alphashape.alphashape(pts,alpha)
        shape_vec.append(alpha_shape)
        #If possible, take area of alpha shape
        if(len(pts) > 2):
            area_vec[i] = alpha_shape.area
        else:
            area_vec[i] = np.nan
    #Normalize each cell by the number of molecules assigned to it
    mean_area = np.nanmean(area_vec / np.sum(adata.X, axis=1) )
    #Use this mean area to fill in NaN values
    area_vec[np.isnan(area_vec)] = mean_area * np.sum(adata.X, axis=1)[np.isnan(area_vec)]
    adata.obs['alpha_area'] = area_vec
    adata.obs['alpha_shape'] = shape_vec
    return area_vec