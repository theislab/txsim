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
    ct_method: str,
    ct_certainty_threshold: float = 0.7,
    adata_sc: Optional[str] = None,
    ct_assign_output: Optional[str] = None,
    all_ct_methods: bool = False,
    prior_pct: float = 0.5
    #ct_manual_markers: Optional[str] = None, # ToDo: marker genes based annotation, input as csv with cell type and markers
    #ct_scrna_referecne: Optional[str] = None, # ToDo: path to adata with single cell rna-seq data for automated marker gene detection
    
) -> AnnData:
    """Generate an AnnData object with counts from molecule data and assign cell types

    Parameters
    ----------
    molecules : DataFrame
        DataFrame containing genes and cell assignments
    ct_method : str
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
    # spots = spots[spots['cell'] > 0] #What is happening here

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
    
    #Add celltype according to ct_method and check if all methods should be implemented
    if (ct_method == 'None'): ct_method = 'majority'
    if (ct_method == 'majority' or all_ct_methods):
        adata = run_majority_voting(adata, spots)
    if (ct_method == 'ssam' or all_ct_methods):
        adata = run_ssam(adata, spots, adata_sc = adata_sc)
    if (ct_method == 'pciSeq' or all_ct_methods):
        assert ct_assign_output is not None, 'Cell annotation file of assignment method not found.'
        temp = pd.read_csv(ct_assign_output, header=None, index_col = 0)
        adata.obs['ct_pciSeq'] = pd.Categorical(temp[1][adata.obs['cell_id']])

    # ToDo (second prio)
    # elif ct_method == 'manual_markers':
    #     adata = run_manual_markers(adata, spots)
    # elif ct_method == 'scrna_markers':
    #     adata = run_scrna_markers(adata, spots, rna_adata)
    else:
        print('No valid cell type annotation method')
    
    # Take over primary ct annotation method to adata.obs['celltype'] and apply certainty threshold
    # Add methods, if they provide certainty measure
    if ct_method in ['majority', 'ssam']: 
        ct_list = adata.obs['ct_'+str(ct_method)].copy()
        ct_list[adata.obs['ct_'+str(ct_method)+'_cert'] < ct_certainty_threshold] = "Unknown"
        adata.obs['celltype'] = ct_list
    else:
        adata.obs['celltype'] = adata.obs['ct_'+str(ct_method)]

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