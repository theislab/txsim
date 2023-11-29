import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
from _util import check_crop_exists
from _util import get_bin_edges



def relative_expression_similarity_across_genes_local(
    adata_sp : AnnData, 
    adata_sc : AnnData,
    x_min : int, 
    x_max : int, 
    y_min : int, 
    y_max : int, 
    image : np.ndarray, 
    bins : int = 10,            #wie in np.histogram2d, oder anderes Typing?
    key : str='celltype', 
    layer : str='lognorm',
    min_total_cells : int = 1,      #welche Werte?
    min_n_cells_per_ct : int = 1
    ):

    """Caculate in each gridfield the relative expression similarity across genes. 
    
    If too few cells are in a gridfield, we assign NaN. Further we only include
    celltypes with enough cells per celltype.

    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    x_min : int, 
    x_max : int, 
    y_min : int, 
    y_max : int, 
    image : np.ndarray, 
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    key : str='celltype', optional
        name of the column containing the cell type information
    layer : str='lognorm'
    min_total_cells: int, optional
        if less than min_total_cells cells are provided, NaN is assigned. By default 1
    min_n_cells_per_ct: int, optional
        only celltypes with at least min_n_cells_per_ct members are considered. By default 1

    Returns
    -------
    gridfield_metric: array of floats  
    """

    range = check_crop_exists(x_min,x_max,y_min,y_max,image)
    bins_x, bins_y = get_bin_edges(range, bins)

    n_bins_x = len(bins_x) - 1
    n_bins_y = len(bins_y) - 1

    ### SET UP
    # set the .X layer of each of the adatas to be log-normalized counts
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]

    # take the intersection of genes in adata_sp and adata_sc, as a list
    intersect_genes = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))
    n_intersect_genes = len(intersect_genes)

    # subset adata_sc and adata_sp to only include genes in the intersection of adata_sp and adata_sc 
    adata_sc=adata_sc[:,intersect_genes]
    adata_sp=adata_sp[:,intersect_genes]
            
    # find the intersection of unique celltypes in adata_sc and adata_sp
    intersect_celltypes = list(set(adata_sp.obs["celltype"]).intersection(set(adata_sc.obs["celltype"])))

    adata_sc = adata_sc[adata_sc.obs[key].isin(intersect_celltypes), :]
    adata_sp = adata_sp[adata_sp.obs[key].isin(intersect_celltypes), :]

    sp_X = adata_sp.X.toarray()
    exp_sp = pd.DataFrame(sp_X,columns=intersect_genes)
    exp_sp.index = adata_sp.obs.index

    gridfield_metric = np.zeros((n_bins_y,n_bins_x))

    i, j = 0, 0
    for x_start, x_end in zip(bins_x[:-1], bins_x[1:]):
        i = 0
        for y_start, y_end in zip(bins_y[:-1], bins_y[1:]):    
            df = adata_sp.obs[["x", "y"]]
            df = df[
            (df["x"] >= x_start)
            & (df["x"] < x_end)
            & (df["y"] >= y_start)
            & (df["y"] < y_end)
            ]

            if len(df) < min_total_cells:
                gridfield_metric[i,j] = np.nan   
                i += 1
                continue  

            sp_local = exp_sp.loc[df.index,:] 
            sp_local[key] = adata_sp.obs.loc[df.index,:][key]           

            n_cells_per_ct = sp_local[key].value_counts()
            eligible_ct = n_cells_per_ct.loc[n_cells_per_ct >= min_n_cells_per_ct].index.tolist()

            sp_local = sp_local.loc[sp_local[key].isin(eligible_ct),:]
            sum_sp_local = np.sum(np.sum(sp_local.iloc[:,:-1]))

            adata_sc_local = adata_sc[adata_sc.obs[key].isin(eligible_ct), :]
            sc_X = adata_sc_local.X.toarray()
            sc_local = pd.DataFrame(sc_X, columns = intersect_genes)
            sc_local.index = adata_sc_local.obs.index
            sc_local[key] = adata_sc_local.obs[key]
            sum_sc_local = np.sum(np.sum(sc_local.iloc[:,:-1]))

            if sum_sp_local != 0:
                mean_celltype_sp_normalized=(sp_local.groupby(key).mean().dropna())*(n_intersect_genes)**2     
            else: 
                mean_celltype_sp_normalized=0

            if sum_sc_local != 0:
                mean_celltype_sc_normalized=(sc_local.groupby(key).mean().dropna())*(n_intersect_genes)**2
            else: 
                mean_celltype_sc_normalized=0

            mean_celltype_sp_normalized=mean_celltype_sp_normalized.to_numpy()
            pairwise_diff_sp = mean_celltype_sp_normalized[:,:,np.newaxis] - mean_celltype_sp_normalized[:,np.newaxis,:]

            mean_celltype_sc_normalized=mean_celltype_sc_normalized.to_numpy()
            pairwise_diff_sc = mean_celltype_sc_normalized[:,:,np.newaxis] - mean_celltype_sc_normalized[:,np.newaxis,:]

            delta = np.sum(np.abs(pairwise_diff_sp-pairwise_diff_sc))
            gridfield_metric[i,j]  = 1-delta/(2*np.sum(np.abs(pairwise_diff_sc)))
            i+=1
        j+=1     

    return gridfield_metric





def relative_expression_similarity_across_cell_type_clusters(
    adata_sp : AnnData, 
    adata_sc : AnnData,
    x_min : int, 
    x_max : int, 
    y_min : int, 
    y_max : int, 
    image : np.ndarray, 
    bins : int = 10,                #wie in np.histogram2d, oder anderes Typing?
    key : str='celltype', 
    layer : str='lognorm',
    min_total_cells : int = 1,      #welche Werte?
    min_n_cells_per_ct : int = 1
    ):

    """Caculate in each gridfield the relative expression similarity across cell type clusters. 
    
    If too few cells are in a gridfield, we assign NaN. Further we only include
    celltypes with enough cells per celltype.

    Parameters
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    x_min : int, 
    x_max : int, 
    y_min : int, 
    y_max : int, 
    image : np.ndarray, 
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    key : str
        name of the column containing the cell type information
    layer: str='lognorm'
    min_total_cells: int, optional
        if less than min_total_cells cells are provided, NaN is assigned. By default 50
    min_n_cells_per_ct: int, optional
        only celltypes with at least min_n_cells_per_ct members are considered. By default 10

    Returns
    -------
    gridfield_metric: array of floats  
    """

    
    range = check_crop_exists(x_min,x_max,y_min,y_max,image)
    bins_x, bins_y = get_bin_edges(range, bins)

    n_bins_x = len(bins_x) - 1
    n_bins_y = len(bins_y) - 1

    ### SET UP
    # set the .X layer of each of the adatas to be log-normalized counts
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]

    # take the intersection of genes in adata_sp and adata_sc, as a list
    intersect_genes = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))

    # subset adata_sc and adata_sp to only include genes in the intersection of adata_sp and adata_sc 
    adata_sc=adata_sc[:,intersect_genes]
    adata_sp=adata_sp[:,intersect_genes]
            
    # find the intersection of unique celltypes in adata_sc and adata_sp
    intersect_celltypes = list(set(adata_sp.obs["celltype"]).intersection(set(adata_sc.obs["celltype"])))
    n_intersect_celltypes = len(intersect_celltypes)

    adata_sc = adata_sc[adata_sc.obs[key].isin(intersect_celltypes), :]
    adata_sp = adata_sp[adata_sp.obs[key].isin(intersect_celltypes), :]

    sp_X = adata_sp.X.toarray()
    exp_sp = pd.DataFrame(sp_X,columns=intersect_genes)
    exp_sp.index = adata_sp.obs.index

    gridfield_metric = np.zeros((n_bins_y,n_bins_x))

    i, j = 0, 0
    for x_start, x_end in zip(bins_x[:-1], bins_x[1:]):
        i = 0
        for y_start, y_end in zip(bins_y[:-1], bins_y[1:]):    
            df = adata_sp.obs[["x", "y"]]
            df = df[
            (df["x"] >= x_start)
            & (df["x"] < x_end)
            & (df["y"] >= y_start)
            & (df["y"] < y_end)
            ]

            if len(df) < min_total_cells:
                gridfield_metric[n_bins_y-1-i,j] = np.nan   
                i += 1
                continue  

            sp_local = exp_sp.loc[df.index,:] 
            sp_local[key] = adata_sp.obs.loc[df.index,:][key]           

            n_cells_per_ct = sp_local[key].value_counts()
            eligible_ct = n_cells_per_ct.loc[n_cells_per_ct >= min_n_cells_per_ct].index.tolist()

            sp_local = sp_local.loc[sp_local[key].isin(eligible_ct),:]
            sum_sp_local = np.sum(np.sum(sp_local.iloc[:,:-1]))

            adata_sc_local = adata_sc[adata_sc.obs[key].isin(eligible_ct), :]
            sc_X = adata_sc_local.X.toarray()
            sc_local = pd.DataFrame(sc_X, columns = intersect_genes)
            sc_local.index = adata_sc_local.obs.index
            sc_local[key] = adata_sc_local.obs[key]
            sum_sc_local = np.sum(np.sum(sc_local.iloc[:,:-1]))

            if sum_sp_local != 0:
                mean_celltype_sp_normalized=(sp_local.groupby(key).mean().dropna())*(n_intersect_celltypes)**2     
            else: 
                mean_celltype_sp_normalized=0

            if sum_sc_local != 0:
                mean_celltype_sc_normalized=(sc_local.groupby(key).mean().dropna())*(n_intersect_celltypes)**2
            else: 
                mean_celltype_sc_normalized=0

            mean_celltype_sp_normalized=mean_celltype_sp_normalized.T.to_numpy()
            pairwise_diff_sp = mean_celltype_sp_normalized[:,:,np.newaxis] - mean_celltype_sp_normalized[:,np.newaxis,:]

            mean_celltype_sc_normalized=mean_celltype_sc_normalized.T.to_numpy()
            pairwise_diff_sc = mean_celltype_sc_normalized[:,:,np.newaxis] - mean_celltype_sc_normalized[:,np.newaxis,:]

            delta = np.sum(np.abs(pairwise_diff_sp-pairwise_diff_sc))
            gridfield_metric[n_bins_y-1-i,j]  = 1-delta/(2*np.sum(np.abs(pairwise_diff_sc)))
            i+=1
        j+=1     

    return gridfield_metric