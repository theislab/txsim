import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.sparse import issparse
from typing import Optional


def cell_density(
    adata_sp: AnnData,
    scaling_factor: float = 1.0,
    img_shape: Optional[tuple] = None,
    ct_key: str = "celltype",
    pipeline_output: bool = True
) -> float:
    """Calculates the area of the region imaged using convex hull and divide total number of cells/area.
    
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    scaling_factor: float
        XY position should be in µm. If not, a multiplicative scaling factor can be provided.
    img_shape: tuple
        Provide an image shape for area calculation instead of convex hull
    ct_key: str
        Key in adata.obs that contains cell type information. Only needed if pipeline_output is False.
    pipeline_output : float, optional
        Generic argument for txsim metrics. Boolean for whether to return only the summary statistic or additional
        metric specific outputs. (Here: no additional outputs)
        
    Returns
    -------
    density : float
       Cell density (cells per area unit)
    """   
    if scaling_factor == 1.0:
        pos = adata_sp.uns['spots'].loc[:,['x','y']].values
    else:
        pos = adata_sp.uns['spots'].loc[:,['x','y']].values.copy() * scaling_factor
    
    if img_shape is not None:
        area = img_shape[0] * img_shape[1]
    else:
        hull = ConvexHull(pos) #TODO: test for large dataset, maybe better to use cell positions for scalability
        area = hull.area
        
    density= adata_sp.n_obs/area #*10e6 #TODO: Check if there can be numerical issues due to large area!!!
    
    if pipeline_output:
        return density
    
    density_per_celltype = adata_sp.obs[ct_key].value_counts()/area
    
    return density, density_per_celltype

def proportion_of_assigned_reads(adata_sp: AnnData,pipeline_output=True):
    """Proportion of assigned reads
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Generic argument for txsim metrics. Boolean for whether to return only the summary statistic or additional
        metric specific outputs. (Here: no additional outputs)
        
    Returns
    -------
    proportion_assigned : float
       Proportion of reads assigned to cells / all reads decoded
    """
    if issparse(adata_sp.layers['raw']):
        proportion_assigned=adata_sp.layers['raw'].sum()/adata_sp.uns['spots'].shape[0]
    else:
        proportion_assigned=np.sum(adata_sp.layers['raw'])/adata_sp.uns['spots'].shape[0]
    return proportion_assigned


def reads_per_cell(adata_sp: AnnData, statistic: str = "mean", pipeline_output=True):
    """ Get mean/median number of reads per cell
    
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data. Integer counts are expected in 
        adata_sp.layers['raw'].
    pipeline_output : float, optional
        Generic argument for txsim metrics. Boolean for whether to return only the summary statistic or additional
        metric specific outputs. (Here: no additional outputs)
        
    Returns
    -------
    median_cells : float
       Median_number_of_reads_x_cell
    """   
    if issparse(adata_sp.layers['raw']) and statistic == "mean":
        return np.mean(adata_sp.layers['raw'].sum(axis=1))
    elif issparse(adata_sp.layers['raw']) and statistic == "median":
        return np.median(np.asarray(adata_sp.layers['raw'].sum(axis=1)).flatten())
    elif statistic == "mean":
        return np.mean(np.sum(adata_sp.layers['raw'],axis=1))
    elif statistic == "median":
        return np.median(np.sum(adata_sp.layers['raw'],axis=1))
    else:
        raise ValueError("Please choose either 'mean' or 'median' for statistic")


def number_of_genes(adata_sp: AnnData,pipeline_output=True):
    """ Size of the gene panel present in the spatial dataset
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    number_of genes : float
       Number of genes present in the spatial dataset
    """   
    number_of_genes=adata_sp.shape[1]
    return number_of_genes

def number_of_cells(adata_sp: AnnData,pipeline_output=True):
    """ Number of cells present in the spatial dataset
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    number_of cells : float
       Number of cells present in the spatial dataset
    """   
    number_of_cells=adata_sp.shape[0]
    return number_of_cells

def percentile_5th_reads_cells(adata_sp: AnnData,pipeline_output=True):
    """5th percentile of number of reads/cells in the spatial experiment
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
       Median_number_of_reads_x_cell
    """   
    pctile5=np.percentile(np.sum(adata_sp.layers['raw'],axis=1),5)
    return pctile5

def mean_genes_cells(adata_sp: AnnData,pipeline_output=True):
    """Mean number of genes/cell in the spatial experiment
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
       Mean number of genes per cell
    """   
    mean_genesxcell=np.mean(np.sum((adata_sp.layers['raw']>0)*1,axis=1))
    return mean_genesxcell

def percentile_95th_genes_cells(adata_sp: AnnData,pipeline_output=True):
    """Percentile 95 of genes/cell in the spatial experiment
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
       Percentile 95 of genes per cell
    """   
    percentile95_genesxcell=np.percentile(np.sum((adata_sp.layers['raw']>0)*1,axis=1),95)
    return percentile95_genesxcell

def percentile_5th_genes_cells(adata_sp: AnnData,pipeline_output=True):
    """Percentile 5 of genes/cell in the spatial experiment
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
       Percentile 5 of genes per cell
    """   
    percentile5_genesxcell=np.percentile(np.sum((adata_sp.layers['raw']>0)*1,axis=1),5)
    return percentile5_genesxcell

def median_genes_cells(adata_sp: AnnData,pipeline_output=True):
    """Median of genes/cell in the spatial experiment
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
       Median of genes per cell
    """   
    median_genesxcell=np.median(np.sum((adata_sp.layers['raw']>0)*1,axis=1))
    return median_genesxcell




def percentile_95th_reads_cells(adata_sp: AnnData,pipeline_output=True):
    """5th percentile of number of reads/cells in the spatial experiment
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    pipeline_output : float, optional
        Boolean for whether to use the 
    Returns
    -------
    median_cells : float
       Median_number_of_reads_x_cell
    """   
    pctile95=np.percentile(np.sum(adata_sp.layers['raw'],axis=1),95)
    return pctile95
    