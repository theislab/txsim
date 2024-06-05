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
) -> float | tuple[float, pd.Series]:
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
    pipeline_output : bool
        Generic argument for txsim metrics. Boolean for whether to return only the summary statistic or additional
        metric specific outputs. Here it is used to return the density per cell type.
        
    Returns
    -------
    density : float
       Cell density (cells per area unit)
    if pipeline_output is False, also returns:
    density_per_celltype : pd.Series
        Cell density per cell type (cells per area unit)
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

def proportion_of_assigned_reads(
    adata_sp: AnnData,
    ct_key: str = "celltype",
    gene_key: str = "Gene",
    pipeline_output=True
) -> float | tuple[float, pd.Series, pd.Series]:
    """Proportion of assigned reads
    
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    ct_key: str
        Key in adata.obs that contains cell type information. Only needed if pipeline_output is False.
    gene_key: str
        Key in adata.uns['spots'] that contains gene symbols. Only needed if pipeline_output is False.
    pipeline_output : bool
        Generic argument for txsim metrics. Boolean for whether to return only the summary statistic or additional
        metric specific outputs. Here it is used to return the proportion of assigned reads per gene and per cell type.
        
    Returns
    -------
    proportion_assigned : float
       Proportion of reads assigned to cells relative to all reads decoded.
    if pipeline_output is False, also returns:
    n_spots_per_gene : pd.Series
        Proportion of reads assigned to cells per gene (i.e. relative to all reads decoded per gene)
    n_spots_per_celltype : pd.Series
        Proportion of reads assigned to each cell type relative to all reads decoded.

    """
    if issparse(adata_sp.layers['raw']):
        proportion_assigned = adata_sp.layers['raw'].sum()/adata_sp.uns['spots'].shape[0]
    else:
        proportion_assigned = np.sum(adata_sp.layers['raw'])/adata_sp.uns['spots'].shape[0]
        
    if pipeline_output:
        return proportion_assigned
    
    # Proportion of assigned reads per gene
    n_spots_per_gene = pd.DataFrame(adata_sp.uns["spots"][gene_key].value_counts()).rename(columns={"count": "total"})
    genes_diff = set(adata_sp.var_names) - set(n_spots_per_gene.index)
    assert len(genes_diff) == 0, f"Genes {genes_diff} in adata_sp.var_names are not present in adata_sp.uns['spots']."
    n_spots_per_gene["assigned"] = 0
    n_spots_per_gene.loc[adata_sp.var_names, "assigned"] = np.array(adata_sp.layers['raw'].sum(axis=0)).flatten()
    proportion_assigned_per_gene = n_spots_per_gene["assigned"] / n_spots_per_gene["total"]
    
    # Proportion of reads assigned to each cell type
    obs_df = pd.DataFrame(data = {
        "celltype": adata_sp.obs[ct_key],
        "assigned": np.array(adata_sp.layers['raw'].sum(axis=1)).flatten()
    })
    n_spots_per_celltype = obs_df.groupby("celltype", observed=True).sum()
    proportion_assigned_to_ct = n_spots_per_celltype["assigned"] / adata_sp.uns['spots'].shape[0]
    
    return proportion_assigned, proportion_assigned_per_gene, proportion_assigned_to_ct


def reads_per_cell(
    adata_sp: AnnData, 
    statistic: str = "mean", 
    ct_key: str = "celltype",
    pipeline_output=True
) -> float | tuple[float, pd.Series, pd.Series]:
    """ Get mean/median number of reads per cell
    
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data. Integer counts are expected in 
        adata_sp.layers['raw'].
    statistic: str
        Whether to calculate mean or median reads per cell. Options: "mean" or "median"
    ct_key: str
        Key in adata.obs that contains cell type information. Only needed if pipeline_output is False.
    pipeline_output : bool
        Generic argument for txsim metrics. Boolean for whether to return only the summary statistic or additional
        metric specific outputs. Here it is used to return the mean/median number of reads per cell per gene and per 
        cell type.
        
    Returns
    -------
    mean_reads : float
       Mean or medium number of reads per cell
    if pipeline_output is False, also returns:
    mean_reads_per_gene : pd.Series
        Mean or median number of reads per cell per gene
    mean_reads_per_celltype : pd.Series
        Mean or median number of reads per cell per cell type
    """   
    if issparse(adata_sp.layers['raw']) and statistic == "mean":
        mean_reads = float(np.mean(adata_sp.layers['raw'].sum(axis=1)))
    elif issparse(adata_sp.layers['raw']) and statistic == "median":
        mean_reads = float(np.median(np.asarray(adata_sp.layers['raw'].sum(axis=1)).flatten()))
    elif statistic == "mean":
        mean_reads = float(np.mean(np.sum(adata_sp.layers['raw'],axis=1)))
    elif statistic == "median":
        mean_reads = float(np.median(np.sum(adata_sp.layers['raw'],axis=1)))
    else:
        raise ValueError("Please choose either 'mean' or 'median' for statistic")
    
    if pipeline_output:
        return mean_reads
    
    # Mean/median number of reads per cell per gene
    if statistic == "mean":
        mean_reads_per_gene = pd.Series(
            index=adata_sp.var_names, data=np.array(adata_sp.layers['raw'].mean(axis=0)).flatten()
        )
    elif issparse(adata_sp.layers['raw']) and statistic == "median":
        mean_reads_per_gene = pd.Series(
            index=adata_sp.var_names, data=np.median(adata_sp.layers['raw'].toarray(),axis=0)
        )
    else:
        mean_reads_per_gene = pd.Series(
            index=adata_sp.var_names, data=np.median(adata_sp.layers['raw'],axis=0)
        )
        
    # Mean/median number of reads per cell per cell type
    obs_df = pd.DataFrame(data = {
        "celltype": adata_sp.obs[ct_key],
        "counts": np.array(adata_sp.layers['raw'].sum(axis=1)).flatten()
    })
    if statistic == "mean":
        mean_reads_per_celltype = obs_df.groupby("celltype", observed=True).mean()["counts"]
    else:
        mean_reads_per_celltype = obs_df.groupby("celltype", observed=True).median()["counts"]
        
    return mean_reads, mean_reads_per_gene, mean_reads_per_celltype
      
      
def genes_per_cell(
    adata_sp: AnnData, 
    statistic: str = "mean", 
    ct_key: str = "celltype",
    pipeline_output=True
) -> float | tuple[float, pd.Series]:
    """ Get mean/median number of genes per cell
    
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data. Integer counts are expected in 
        adata_sp.layers['raw'].
    statistic: str
        Whether to calculate mean or median genes per cell. Options: "mean" or "median"
    ct_key: str
        Key in adata.obs that contains cell type information. Only needed if pipeline_output is False.
    pipeline_output : bool
        Generic argument for txsim metrics. Boolean for whether to return only the summary statistic or additional
        metric specific outputs. Here it is used to return the mean/median number of genes per cell per cell type.
        
    Returns
    -------
    mean_genes : float
       Mean or medium number of genes per cell
    if pipeline_output is False, also returns:
    mean_genes_per_celltype : pd.Series
        Mean or median number of genes per cell per cell type
    """
    if issparse(adata_sp.layers['raw']):
        n_genes_per_cell = np.array((adata_sp.layers['raw'] > 0).sum(axis=1)).flatten()
    else:
        n_genes_per_cell = (adata_sp.layers['raw'] > 0).sum(axis=1)
        
    if statistic == "mean":
        mean_genes = float(np.mean(n_genes_per_cell))
    elif statistic == "median":
        mean_genes = float(np.median(n_genes_per_cell))
        
    if pipeline_output:
        return mean_genes
    
    # Mean/median number of genes per cell per cell type
    obs_df = pd.DataFrame(data = {
        "celltype": adata_sp.obs[ct_key],
        "counts": n_genes_per_cell
    })
    if statistic == "mean":
        mean_genes_per_celltype = obs_df.groupby("celltype", observed=True).mean()["counts"]
    else:
        mean_genes_per_celltype = obs_df.groupby("celltype", observed=True).median()["counts"]
        
    return mean_genes, mean_genes_per_celltype
        

def number_of_genes(
    adata_sp: AnnData,
    ct_key: str = "celltype", 
    pipeline_output=True
) -> int | tuple[int, pd.Series]:
    """ Size of the gene panel present in the spatial dataset
    
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    ct_key: str
        Key in adata.obs that contains cell type information. Only needed if pipeline_output is False.
    pipeline_output : bool
        Generic argument for txsim metrics. Boolean for whether to return only the summary statistic or additional
        metric specific outputs. Here it is used to return the number of genes per cell type (genes with at least one
        count in the given cell type).
        
    Returns
    -------
    number_of genes : float
       Number of genes present in the spatial dataset
    if pipeline_output is False, also returns:
    number_of_genes_per_celltype : pd.Series
        Number of genes per cell type (genes with at least one count in the given cell type)
    """
    number_of_genes=adata_sp.n_vars
    if pipeline_output:
        return number_of_genes
    
    # Number of genes per cell type
    gene_in_ct = pd.DataFrame(index=adata_sp.obs[ct_key].unique(), columns=adata_sp.var_names)
    for ct in adata_sp.obs[ct_key].unique():
        gene_in_ct.loc[ct] = adata_sp[adata_sp.obs[ct_key]==ct].layers['raw'].sum(axis=0) > 0
        
    number_of_genes_per_celltype = gene_in_ct.sum(axis=1)
    
    return number_of_genes, number_of_genes_per_celltype
    

def number_of_cells(adata_sp: AnnData,ct_key: str = "celltype", pipeline_output = True) -> int | tuple[int, pd.Series]:
    """ Number of cells present in the spatial dataset
    
    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    ct_key: str
        Key in adata.obs that contains cell type information. Only needed if pipeline_output is False.
    pipeline_output : bool
        Generic argument for txsim metrics. Boolean for whether to return only the summary statistic or additional
        metric specific outputs. Here it is used to return the number of cells per cell type.

    Returns
    -------
    number_of cells : float
       Number of cells present in the spatial dataset
    if pipeline_output is False, also returns:
    number_of_cells_per_celltype : pd.Series
        Number of cells per cell type
    """   
    number_of_cells=adata_sp.n_obs
    if pipeline_output:
        return number_of_cells

    # Number of cells per cell type
    number_of_cells_per_celltype = adata_sp.obs[ct_key].value_counts()
    
    return number_of_cells, number_of_cells_per_celltype

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
    