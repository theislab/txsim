import numpy as np
from numpy import ndarray
import anndata as ad
from anndata import AnnData
import pandas as pd
from pandas import DataFrame
import scanpy as sc
from typing import Optional

from ._ctannotation import run_majority_voting, run_ssam #, run_all_ct_methods

def generate_adata(
    molecules: DataFrame,
    ct_method: str = 'majority',
    ct_certainty_threshold: float = 0.7,
    adata_sc: Optional[str] = None,
    ct_assign_output: Optional[str] = None,
    all_ct_methods: bool = False,
    prior_pct: float = 0.5
    #ct_manual_markers: Optional[str] = None, # ToDo: marker genes based annotation, input as csv with cell type and markers
    #ct_scrna_reference: Optional[str] = None, # ToDo: path to adata with single cell rna-seq data for automated marker gene detection
    
) -> AnnData:
    """Generate an AnnData object with counts from molecule data and assign cell types

    Parameters
    ----------
    molecules : DataFrame
        DataFrame containing genes and cell assignments
    ct_method : Optional[str]
        Method to use for cell type annotation. Output will be added to ``adata.obs['ct_<ct_method>']`` and duplicated in  ``adata.obs['celltype']``. Valid entries are ``['majority', 'ssam', 'pciSeq']``
    ct_certainty_threshold : Optional[float]
        To be set if ``ct_method`` provides a certainty measure of the cell type assignment. Threshold will be applied when annotations are set as ``adata.obs['celltype']`` to keep all annotations in ``adata.obs['ct_<ct_method>']``. For ``ct_method=='majority'`` the certainty refers to the percent of spots per cell assigned to a celltype proir to segmentatation, for ``ct_method=='ssam'`` it refers to **TODO!**
    ct_assign_output : Optional[str]
        File name containing cell type for each cell generated by the spot to cell assignment method, currently only implemented for pciSeq, default is ``'None'``
    all_ct_methods : Optional[bool]
        If set to True, all available cell type annotation methods are ran, results added as ``adata.obs['ct_<ct_method>']`` and the method chosen as ct_method is duplicated to ``adata.obs['celltype']``

    Returns
    -------
    AnnData
        Populated count matrix
    """    
    
    #Read assignments, calculate percentage of non-assigned spots (pct_noise) and save raw version of spots
    spots = molecules #TODO bad naming
    pct_noise = sum(spots['cell'] <= 0)/len(spots['cell'])
    spots_raw = spots.copy() # save raw spots to add to adata.uns and set 0 to None
    spots_raw.loc[spots_raw['cell']==0,'cell'] = None
    spots = spots[spots['cell'] > 0] #What is happening here

    #Generate blank, labelled count matrix
    X = np.zeros([ len(pd.unique(spots['cell'])), len(pd.unique(spots['Gene'])) ])
    adata = ad.AnnData(X, dtype = X.dtype)
    adata.obs['cell_id'] = pd.unique(spots['cell'])
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = pd.unique(spots['Gene'])
    adata.obs['centroid_x'] = 0
    adata.obs['centroid_y'] = 0
    
    #Populate matrix using assignments
    for cell_id in adata.obs['cell_id']:
        cts = spots[spots['cell'] == cell_id ]['Gene'].value_counts()
        adata[adata.obs['cell_id'] == cell_id, :] = cts.reindex(adata.var_names, fill_value = 0)
        adata.obs.loc[adata.obs['cell_id'] == cell_id,'centroid_x'] = spots[spots['cell'] == cell_id ]['x'].mean()
        adata.obs.loc[adata.obs['cell_id'] == cell_id,'centroid_y'] = spots[spots['cell'] == cell_id ]['y'].mean()
    
    #TEMP: save intermediate adata
    #adata.write_h5ad('data/adata_st_temp.h5ad')
    
    

    #Save additional information about the data
    adata.uns['spots'] = spots_raw
    adata.uns['pct_noise'] = pct_noise
    adata.layers['raw_counts'] = adata.X.copy()

    #Calculate some basic statistics
    adata.obs['n_counts']= np.sum(adata.layers['raw_counts'], axis = 1)
    adata.obs['n_genes']= np.sum(adata.layers['raw_counts']>0, axis = 1)
    adata.var['n_counts']= np.sum(adata.layers['raw_counts'], axis=0)
    adata.var['n_cells']= np.sum(adata.layers['raw_counts']>0, axis = 0)

    return adata

def calculate_alpha_area(
    adata: AnnData,
    alpha: float = 0
) -> ndarray:
    """Calculate and store the alpha shape area of the cell given a set of points (genes). 
    Uses the Alpha Shape Toolboox: https://alphashape.readthedocs.io/en/latest/readme.html 

    Parameters
    ----------
    adata : AnnData
        AnnData object with cells as `obs` and genes as `var`, and spots as `adata.uns['spots']`
    alpha : float, optional
        The alpha parameter a, used to calculate the alpha shape, by default 0. If -1, optimal alpha 
        parameter will be calculated.

    Returns
    -------
    ndarray
        Returns the area vector stored in `adata.obs['alpha_area']` as a numpy array
    """    

    import alphashape
    from descartes import PolygonPatch
    import shapely
    import json
    
    #Read assignments
    spots = adata.uns['spots']

    # Calculate alpha shape
    # If there are <3 molecules for a cell, use the mean area
    area_vec = np.zeros([adata.n_obs])
    shape_vec = []
    for i in range(adata.n_obs):
        dots = pd.concat(
            [spots[spots['cell'] == adata.obs['cell_id'][i]].x,
            spots[spots['cell'] == adata.obs['cell_id'][i]].y],
            axis=1
        )
        pts = list(dots.itertuples(index=False, name=None))
        #If alpha is negative, find optimal alpha, else use parameter/convex hull
        if alpha < 0 and len(pts) > 3:
            opt_alpha = alphashape.optimizealpha(pts, max_iterations=100, lower = 0, upper = 10, silent=False)
            alpha_shape = alphashape.alphashape(pts, opt_alpha)
        elif alpha < 0:
            alpha_shape = alphashape.alphashape(pts,0)
        else:    
            alpha_shape = alphashape.alphashape(pts,alpha)
        
        shape_vec.append(json.dumps(shapely.geometry.mapping(alpha_shape)))
        #If possible, take area of alpha shape
        if(len(pts) > 3):
            area_vec[i] = alpha_shape.area
        else:
            area_vec[i] = np.nan
    
    #Find mean cell area and fill in NaN and very small values
    #mean_area = np.nanmean(area_vec / np.sum(adata.X, axis=1) )
    #area_vec[np.isnan(area_vec)] = mean_area * np.sum(adata.X, axis=1)[np.isnan(area_vec)]
    mean_area = np.nanmean(area_vec)
    area_vec[np.isnan(area_vec)] = mean_area
    area_vec[np.where(area_vec < 1)] = mean_area

    adata.obs['alpha_area'] = area_vec
    adata.obs['alpha_shape'] = shape_vec
    return area_vec



def aggregate_count_matrices(
    adata_list: list,
    rep_list: list = None
) -> AnnData:
    
    # Jut in case the replicates aren't read in the right order
    if rep_list is None: rep_list = range(1, len(adata_list)+1)
    rep_idx = 1

    # Copy in the first anndata, and set the replicate number
    adata = adata_list[0].copy()
    spots = adata.uns["spots"].copy()

    adata.obs["replicate"] = rep_list[0]
    spots['replicate'] = rep_list[0]

    adata_list.pop(0)

    for new_adata in adata_list: 
        new_spots = new_adata.uns["spots"].copy()

        # Ensures the cell id's will be unique, and that the spots['cell'] will match adata.obs['cell_id']
        # Uses the max not 'nunique' of previous cell id since numbering is not perfectly sequential 
        new_spots.loc[(new_spots['cell']  > 0), "cell"] = new_spots['cell'][new_spots['cell']  > 0] + spots['cell'].max()
        new_adata.obs['cell_id'] = new_adata.obs['cell_id'] + spots['cell'].max()

        # Set replicate number and increase index
        new_spots['replicate'] = rep_list[rep_idx]
        new_adata.obs['replicate'] = rep_list[rep_idx]
        rep_idx+=1
        
        # Concatenate anndata and dataframe
        spots = pd.concat((spots,new_spots), ignore_index = True)
        adata = adata.concatenate(new_adata)

    adata.uns['spots'] = spots
    
    return adata

def filter_cells(
    adata: AnnData,
    min_counts: int = 10, 
    min_cell_percentage: float = 0.8, 
    min_area: Optional[float] = None, 
    max_area: Optional[float] = None,
    obs_key: str = "passed_QC",
) -> None:
    """Filter cells based on their counts and area.

    Args:
        min_counts: Minimum number of counts per cell.
        min_cell_percentage: Minimum percentage of cells to keep.
        min_area: Minimum area of a cell.
        max_area: Maximum area of a cell.
        obs_key: Key to store the QC results in the obs slot.

    Returns:
        Adds a boolean column to the obs slot of the AnnData object.
        
    """
    #Filter counts
    adata.obs[obs_key] = adata.obs["n_counts"] >= min_counts
    #If filtered cells is below the min_cell_percentage
    if sum(adata.obs[obs_key]) / len(adata.obs) < min_cell_percentage:
        print(f"Only {round(100 * sum(adata.obs[obs_key]) / len(adata.obs))}% of cells passed min_counts QC (min: {round(100*min_cell_percentage)}%)")
        #Find largest value that gives enough cells
        counts = adata.obs["n_counts"].value_counts().sort_index(ascending=False)
        new_min_counts = counts.index[np.argmax(np.cumsum(counts) > adata.n_obs*min_cell_percentage)]
        #Refilter with new value
        adata.obs[obs_key] = adata.obs["n_counts"] >= new_min_counts
        print(f"New min_counts: {new_min_counts}, now {round(100 * sum(adata.obs[obs_key]) / len(adata.obs))}% of cells pass QC")
        print("Stopping QC since min cell percentage reached")
        return
        
    prev_obs = adata.obs[obs_key].copy()
    if min_area is not None: adata.obs[obs_key] &= adata.obs["area"] >= min_area

    if sum(adata.obs[obs_key]) / len(adata.obs) < min_cell_percentage:
        print(f"Only {round(100 * sum(adata.obs[obs_key]) / len(adata.obs))}% of cells passed min_area QC (min: {round(100*min_cell_percentage)}%)")
        #Find largest value that gives enough cells
        counts = adata[prev_obs].obs["area"].value_counts().sort_index(ascending=False)
        new_min_area = counts.index[np.argmax(np.cumsum(counts) > adata.n_obs*min_cell_percentage)]
        #Refilter with new value
        adata.obs[obs_key] = prev_obs
        adata.obs[obs_key] &= adata.obs["area"] >= new_min_area
        print(f"New min_area: {new_min_area}, now {round(100 * sum(adata.obs[obs_key]) / len(adata.obs))}% of cells pass QC")
        print("Stopping QC since min cell percentage reached")
        return

    prev_obs = adata.obs[obs_key].copy()
    if max_area is not None: adata.obs[obs_key] &= adata.obs["area"] <= max_area

    if sum(adata.obs[obs_key]) / len(adata.obs) < min_cell_percentage:
        print(f"Only {round(100 * sum(adata.obs[obs_key]) / len(adata.obs))}% of cells passed max_area QC (min: {round(100*min_cell_percentage)}%)")
        #Find lowest value that gives enough cells
        counts = adata[prev_obs].obs["area"].value_counts().sort_index()
        print(np.cumsum(counts))
        new_max_area = counts.index[np.argmax(np.cumsum(counts) > adata.n_obs*min_cell_percentage)]
        #Refilter with new value
        adata.obs[obs_key] = prev_obs
        print(adata.obs)
        adata.obs[obs_key] &= adata.obs["area"] <= new_max_area
        print(f"New max_area: {new_max_area}, now {round(100 * sum(adata.obs[obs_key]) / len(adata.obs))}% of cells pass QC")
        print("Stopping QC since min cell percentage reached")
        return
    