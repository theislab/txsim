import numpy as np
import anndata as ad
from typing import Tuple
from ..metrics import knn_mixing_per_cell_score

#TODO: "negative_marker_purity_reads", "negative_marker_purity_cells", "coexpression_similarity", 
#    "relative_expression_similarity_across_genes", "relative_expression_similarity_across_celltypes",


def _get_knn_mixing_grid(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    obs_key: str = "celltype",
    cells_x_col: str = "x",
    cells_y_col: str = "y",
    **kwargs
) -> np.ndarray:
    """Calculate the average knn mixing score within each grid bin.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated AnnData object containing spatial transcriptomics data.
    adata_sc : AnnData
        Annotated AnnData object containing dissociated single cell transcriptomics data.
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.
    kwargs : dict
        Additional keyword arguments for the txsim.metrics.knn_mixing_per_cell_score function.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the average knn mixing scores in each grid bin.
    """

    knn_mixing_per_cell_score(adata_sp, adata_sc, obs_key = obs_key, **kwargs)

    knn_mixing_score_key = kwargs["key_added"] if ("key_added" in kwargs) else "knn_mixing_score"
    
    df_cells = adata_sp.obs[[cells_y_col, cells_x_col, knn_mixing_score_key]]
    
    H = np.histogram2d(
        df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, 
        range=region_range, weights=df_cells[knn_mixing_score_key]
    )[0]
    
    # Normalize by the number of cells in each bin
    H = H / np.histogram2d(df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, range=region_range)[0]
    
    return H



    
    def get_relative_expression_between_celltypes(adata_sp: ad, adata_sc: ad, region_range: Tuple[Tuple[float, float], Tuple[float, float]], obs_key: str = "celltype",cells_x_col: str = "x",
    cells_y_col: str = "y", layer:str='lognorm'):
  """Calculate the efficiency deviation present between the genes in the panel. 
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from scRNAseq data
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells
    layer: str (default: 'lognorm')
        layer of ```AnnData`` to use to compute the metric
     

    Returns
    -------
    overall_metric: float
        similarity of relative gene expression across all genes and celltypes, b/t the scRNAseq and spatial data
    per_gene_metric: float
        similarity of relative gene expression per gene across all celltypes, b/t the scRNAseq and spatial data
    per_celltype_metric: float
        similarity of relative gene expression per celltype across all genes, b/t the scRNAseq and spatial data
  
    """
   ### SET UP

    # set the .X layer of each of the adatas to be log-normalized counts
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]

    # take the intersection of genes in adata_sp and adata_sc, as a list
    intersect = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))

    # subset adata_sc and adata_sp to only include genes in the intersection of adata_sp and adata_sc
    adata_sc=adata_sc[:,intersect].copy()
    adata_sp=adata_sp[:,intersect].copy()

    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.X = a.X.toarray()
            
    # Filter cells by region range
    x_min, x_max = region_range[1]
    y_min, y_max = region_range[0]

    # Apply filtering for adata_sp
    adata_sp = adata_sp[adata_sp.obs[cells_x_col].between(x_min, x_max) &
                        adata_sp.obs[cells_y_col].between(y_min, y_max)]

    # find the unique celltypes in adata_sc that are also in adata_sp
    unique_celltypes=adata_sc.obs.loc[adata_sc.obs[obs_key].isin(adata_sp.obs[obs_key]),obs_key].unique()

    #### FIND MEAN GENE EXPRESSION PER CELL TYPE FOR EACH MODALITY
    # get the adata_sc cell x gene matrix as a pandas dataframe (w gene names as column names)
    exp_sc=pd.DataFrame(adata_sc.X,columns=adata_sc.var.index)

    # get the adata_sp cell x gene matrix as a pandas dataframe (w gene names as column names)
    exp_sp=pd.DataFrame(adata_sp.X,columns=adata_sp.var.index)

    # add "celltype" label column to exp_sc & exp_sp cell x gene matrices
    exp_sc[obs_key]=list(adata_sc.obs[obs_key])
    exp_sp[obs_key]=list(adata_sp.obs[obs_key])

    # delete all cells from the exp matrices if they aren't in the set of intersecting celltypes b/t sc & sp data
    exp_sc=exp_sc.loc[exp_sc[obs_key].isin(unique_celltypes),:]
    exp_sp=exp_sp.loc[exp_sp[obs_key].isin(unique_celltypes),:]

    #get metrics
    overall_metric, per_gene_metric, per_celltype_metric = find_gene_expression_per_celltype(adata_sp, adata_sc, exp_sc, exp_sp,'louvain')

    return overall_metric, per_gene_metric, per_celltype_metric




    def find_gene_expression_per_celltype(adata_sp: ad, adata_sc: ad, exp_sc: pd.DataFrame(), exp_sp: pd.DataFrame(), obs_key: str = "celltype",cells_x_col: str = "x",
    cells_y_col: str = "y", layer:str='lognorm'):
    """Calculate the efficiency deviation present between the genes in the panel. 
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from scRNAseq data
    exp_sc: pd.DataFrame()
        cell x gene matrix as pandas DataFrame for scRNAseq data
    exp_sp: pd.DataFrame()
        cell x gene matrix as pandas Dataframefor spatial data
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells
    layer: str (default: 'lognorm')
        layer of ```AnnData`` to use to compute the metric
     

    Returns
    -------
    overall_metric: float
        similarity of relative gene expression across all genes and celltypes, b/t the scRNAseq and spatial data
    per_gene_metric: float
        similarity of relative gene expression per gene across all celltypes, b/t the scRNAseq and spatial data
    per_celltype_metric: float
        similarity of relative gene expression per celltype across all genes, b/t the scRNAseq and spatial data
  
    """
  #### FIND MEAN GENE EXPRESSION PER CELL TYPE FOR EACH MODALITY
    
    # find the mean expression for each gene for each celltype in sc and sp data
    mean_celltype_sp=exp_sp.groupby(obs_key).mean()
    mean_celltype_sc=exp_sc.groupby(obs_key).mean()

    # sort genes in alphabetical order
    mean_celltype_sc=mean_celltype_sc.loc[:,mean_celltype_sc.columns.sort_values()]
    mean_celltype_sp=mean_celltype_sp.loc[:,mean_celltype_sp.columns.sort_values()]

    #### CALCULATE PAIRWISE RELATIVE DISTANCES BETWEEN CELL TYPES
    mean_celltype_sc_np = mean_celltype_sc.T.to_numpy()
    pairwise_distances_sc = mean_celltype_sc_np[:,:,np.newaxis] - mean_celltype_sc_np[:,np.newaxis,:]
    pairwise_distances_sc = pairwise_distances_sc.transpose((1,2,0)) #results in np.array of dimensions (num_celltypes, num_celltypes, num_genes)

    mean_celltype_sp_np = mean_celltype_sp.T.to_numpy()
    pairwise_distances_sp = mean_celltype_sp_np[:,:,np.newaxis] - mean_celltype_sp_np[:,np.newaxis,:]
    pairwise_distances_sp = pairwise_distances_sp.transpose((1,2,0)) #results in np.array of dimensions (num_celltypes,num_celltypes, num_genes)

    #### NORMALIZE THESE PAIRWISE DISTANCES BETWEEN CELL TYPES
    #calculate sum of absolute distances
    abs_diff_sc = np.absolute(pairwise_distances_sc)
    abs_diff_sum_sc = np.sum(abs_diff_sc, axis=(0,1))

    abs_diff_sp = np.absolute(pairwise_distances_sp)
    abs_diff_sum_sp = np.sum(abs_diff_sp, axis=(0,1))

    norm_factor_sc = (1/(mean_celltype_sc.T.shape[1]**2)) * abs_diff_sum_sc
    norm_factor_sp = (1/(mean_celltype_sp.T.shape[1]**2)) * abs_diff_sum_sp


    #perform normalization
    norm_pairwise_distances_sc = np.divide(pairwise_distances_sc, norm_factor_sc)
    norm_pairwise_distances_sp = np.divide(pairwise_distances_sp, norm_factor_sp)


    pairwise_distances_sc[:,:,norm_factor_sc!=0] = np.divide(pairwise_distances_sc[:,:,norm_factor_sc!=0],
                                                             norm_factor_sc[norm_factor_sc!=0])
    # exclude the ones with norm_factor_sc, norm_factor_sp with zero
    pairwise_distances_sp[:,:,norm_factor_sp!=0] = np.divide(pairwise_distances_sp[:,:,norm_factor_sp!=0],
                                                             norm_factor_sp[norm_factor_sp!=0])
    norm_pairwise_distances_sc = pairwise_distances_sc
    norm_pairwise_distances_sp = pairwise_distances_sp

    ##### CALCULATE OVERALL SCORE,PER-GENE SCORES, PER-CELLTYPE SCORES
    overall_score = np.sum(np.absolute(norm_pairwise_distances_sp - norm_pairwise_distances_sc), axis=None)
    overall_metric = 1 - (overall_score/(2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=None)))

    per_gene_score = np.sum(np.absolute(norm_pairwise_distances_sp - norm_pairwise_distances_sc), axis=(0,1))
    per_gene_metric = 1 - (per_gene_score/(2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=(0,1))))
    per_gene_metric = pd.DataFrame(per_gene_metric, index=mean_celltype_sc.columns, columns=['score']) #add back the gene labels


    #per_gene_metric = pd.DataFrame(per_gene_metric, index=mean_celltype_sc.T.columns, columns=['score']) #add back the gene labels

    per_celltype_score = np.sum(np.absolute(norm_pairwise_distances_sp - norm_pairwise_distances_sc), axis=(1,2))
    per_celltype_metric = 1 - (per_celltype_score/(2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=(1,2))))
    per_celltype_metric = pd.DataFrame(per_celltype_metric, index=mean_celltype_sc.index, columns=['score']) #add back the celltype labels


    return overall_metric, per_gene_metric, per_celltype_metric
