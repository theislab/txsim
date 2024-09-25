import warnings
from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy import stats


def coexpression_similarity(
        adata_sp: AnnData,
        adata_sc: AnnData,
        min_cells: int = 20,
        thresh: float = 0,
        layer: str = 'lognorm',
        key: str = 'celltype',
        by_celltype: bool = False,
        correlation_measure: str = "pearson",
        pipeline_output: bool = True,
) -> float | tuple[float, pd.Series, pd.Series]:
    """Calculate the mean difference between correlation matrices of spatial and scRNAseq data.
    
    Parameters
    ----------
    spatial_data : AnnData
        annotated ``AnnData`` object with counts from spatial data
    seq_data : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    min_cells : int, optional
        Minimum number of cells in which a gene should be detected to be considered
        expressed. By default 20. If `by_celltype` is True, the filter is applied for each cell type individually.
    thresh : float, optional
        threshold for significant pairs from scRNAseq data. Pairs with correlations
        below the threshold (by magnitude) will be ignored when calculating mean, by
        default 0
    layer : str, optional
        name of layer used to calculate coexpression similarity. Should be the same
        in both AnnData objects
        default lognorm
    key : str
        name of the column containing the cell type information
    by_celltype: bool
        run analysis by cell type? If False, computation will be performed using the
        whole gene expression matrix
    correlation_measure: str
        metric used to evaluate the correlation of gene expression, mutual, spearman or pearson.
        default pearson
    pipeline_output: bool, optional (default: True)
        flag whether the metric is run as part of the pipeline or not. If not, then
        coexpression similarity matrices are returned for each modality. Otherwise,
        only the mean absolute difference of the upper triangle of the coexpression
        matrices is returned as a single score.
        
    Returns
    -------
    float
        coexpression summary metric (mean of absolute correlation difference between spatial and scRNAseq data). If 
        `by_celltype` is True, the summary metric is the mean of the mean coexpression similarity per cell type.
    if pipeline_output is False also returns:
    pd.Series
        coexpression similarity per gene (mean over cell types if by_celltype is True)
    pd.Series
        coexpression similarity per cell type. (empty if by_celltype is False)
    """

    SUPPORTED_CORR = ["mutual", "pearson", "spearman"]
    assert correlation_measure in SUPPORTED_CORR, f"Invalid correlation measure {correlation_measure}"
    if correlation_measure == "mutual":
        raise NotImplementedError("Mutual information is not yet supported")

    # Reduce to shared genes
    genes_unfiltered = adata_sc.var_names.intersection(adata_sp.var_names)
    adata_sc = adata_sc[:, genes_unfiltered].copy()
    adata_sp = adata_sp[:, genes_unfiltered].copy()

    # Filter genes based on number of cells that express them
    genes_sp = adata_sp.var_names[sc.pp.filter_genes(adata_sp, min_cells=min_cells, inplace=False)[0]]
    genes_sc = adata_sc.var_names[sc.pp.filter_genes(adata_sc, min_cells=min_cells, inplace=False)[0]]

    # Get common genes
    genes = genes_sp.intersection(genes_sc)
    
    if (not len(genes)):
        print("No expressed genes are shared in both modalities")
        if pipeline_output:
            return np.nan 
        else: 
            return [np.nan, pd.Series(index=genes_unfiltered, data=np.nan), pd.Series(dtype=float)]

    if not by_celltype:
        # Get matrices for commonly expressed genes
        mat_sp = adata_sp[:, genes].layers[layer]
        mat_sc = adata_sc[:, genes].layers[layer]

        # Calculate coexpression similarity matrix
        coexp_sim_mat = coexpression_similarity_matrix(mat_sp, mat_sc, thresh, correlation_measure)
        
        # Calculate summary metric
        coexp_sim = np.nanmean(coexp_sim_mat[np.triu_indices(len(coexp_sim_mat), k=1)])
        
        if pipeline_output:
            return coexp_sim
        else:
            return [coexp_sim, pd.Series(index=genes, data=np.nanmean(coexp_sim_mat, axis=1)), pd.Series(dtype=float)]
    else:        
        # Get shared cell types
        shared_cts = list(set(adata_sc.obs[key].unique()).intersection(set(adata_sp.obs[key].unique())))
        
        # Init coexpression similarity matrices
        coexp_sim_matrices = np.zeros((len(shared_cts), len(genes), len(genes)), dtype=float)
        coexp_sim_matrices[:,:,:] = np.nan

        for ct_idx, ct in enumerate(shared_cts):
            # Adatas for the given cell type
            adata_sc_ct = adata_sc[adata_sc.obs[key] == ct, :]
            adata_sp_ct = adata_sp[adata_sp.obs[key] == ct, :]

            # Filter genes based on number of cells that express them
            genes_sp_ct = adata_sp_ct.var_names[sc.pp.filter_genes(adata_sp_ct, min_cells=min_cells, inplace=False)[0]]
            genes_sc_ct = adata_sc_ct.var_names[sc.pp.filter_genes(adata_sc_ct, min_cells=min_cells, inplace=False)[0]]
            
            # Get common genes
            genes_ct = genes_sp_ct.intersection(genes_sc_ct)
            genes_mask = np.isin(genes, genes_ct)
            
            # Skip cell type if no expressed genes are shared
            if len(genes_ct) == 0:
                print(f"No expressed genes are shared in both modalities for cell type {ct}")
                continue
                
            # Get matrices for commonly expressed genes
            mat_sp = adata_sp_ct[:, genes_ct].layers[layer]
            mat_sc = adata_sc_ct[:, genes_ct].layers[layer]
            
            # Calculate coexpression similarity matrix
            coexp_sim_mat = coexpression_similarity_matrix(mat_sp, mat_sc, thresh, correlation_measure)
            coexp_sim_matrices[ct_idx][np.ix_(genes_mask, genes_mask)] = coexp_sim_mat
            
        # Calculate per cell type score
        coexp_sim_per_ct = pd.Series(index=shared_cts, data=np.nan)
        for ct_idx, ct in enumerate(shared_cts):
            coexp_sim_mat = coexp_sim_matrices[ct_idx, :, :]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                coexp_sim_per_ct.loc[ct] = np.nanmean(coexp_sim_mat[np.triu_indices(len(coexp_sim_mat), k=1)])
        
        # Calculate per gene score (RuntimeWarning expected for gene pairs with NaN values in all cell types)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            coexp_sim_mat_ct_avg = np.nanmean(coexp_sim_matrices, axis=0)
            coexp_sim_per_gene = pd.Series(index=genes, data=np.nanmean(coexp_sim_mat_ct_avg, axis=0))
        
        # Calculate summary metric
        coexp_sim = coexp_sim_per_ct.mean()
        
        if pipeline_output:
            return coexp_sim
        else:
            return [coexp_sim, coexp_sim_per_gene, coexp_sim_per_ct]


def coexpression_similarity_matrix(
        mat_sp: np.ndarray | scipy.sparse.csr_matrix, 
        mat_sc: np.ndarray | scipy.sparse.csr_matrix, 
        thresh: float, 
        correlation_measure: str
) -> np.ndarray:
    """ Calculate the coexpression similarity matrix between spatial and scRNAseq data.
    
    The treshold to filter out pairs of low correlations is only applied to the scRNAseq data. The reasoning is that 
    scRNAseq should have less correlation artifacts and is understood as the "ground truth".
    
    Parameters
    ----------
    mat_sp : np.ndarray | scipy.sparse.csr_matrix
        spatial data matrix. Note that the matrix should be in the same order as the scRNAseq data matrix.
    mat_sc : np.ndarray | scipy.sparse.csr_matrix
        scRNAseq data matrix. Note that the matrix should be in the same order as the spatial data matrix.
    thresh : float
        threshold for significant pairs from scRNAseq data. Pairs with correlations below the threshold (by magnitude) 
        will be set to NaN.
    correlation_measure : str
        Metric for the correlation measure refering to coexpression. Supported are "pearson" and "spearman".
        
    Returns
    -------
    np.ndarray
        Coexpression similarity matrix (1 - absolute difference of the correlation matrices/2 of spatial and scRNAseq 
        data). Diagonal is set to NaN.
    """
    
    # Calculate correlation matrices
    if correlation_measure == 'pearson':
        coexp_sp = get_pearson_correlation_matrix(mat_sp)
        coexp_sc = get_pearson_correlation_matrix(mat_sc)
    elif correlation_measure == 'spearman':
        coexp_sp = get_spearman_correlation_matrix(mat_sp)
        coexp_sc = get_spearman_correlation_matrix(mat_sc)
        
    # Gene pair filter based on threshold applied to correlations of scRNAseq data
    mask = np.abs(coexp_sc) < thresh
        
    # Calculate difference between modalities
    coexp_diff_matrix = np.abs(coexp_sc - coexp_sp)
    coexp_diff_matrix[mask] = np.nan
    
    # Transform to similarity matrix
    coexp_sim_matrix = 1 - coexp_diff_matrix / 2
    
    # Set diagonal to NaN
    np.fill_diagonal(coexp_sim_matrix, np.nan)
    
    return coexp_sim_matrix
    

def get_pearson_correlation_matrix(mat: np.ndarray | scipy.sparse.csr_matrix) -> np.ndarray:
    """ Calculate the Pearson correlation matrix of the input matrix.
    
    Parameters
    ----------
    mat : np.ndarray | scipy.sparse.csr_matrix
        input matrix (e.g. adata.X)
        
    Returns
    -------
    np.ndarray
        Pearson correlation matrix of genes
    """
    mat_ = mat.toarray().astype(float) if scipy.sparse.issparse(mat) else mat.astype(float)
    if mat_.shape[1] > 1:
        return np.corrcoef(mat_, rowvar=False)
    else:
        return np.ones((1,1), dtype=np.float32)

def get_spearman_correlation_matrix(mat: np.ndarray | scipy.sparse.csr_matrix) -> np.ndarray:
    """ Calculate the Spearman correlation
    
    Parameters
    ----------
    mat : np.ndarray | scipy.sparse.csr_matrix
        input matrix (e.g. adata.X)
        
    Returns
    -------
    np.ndarray
        Spearman correlation matrix of genes
    """
    mat_ = mat.toarray().astype(float) if scipy.sparse.issparse(mat) else mat.astype(float)
    
    if mat.shape[1] > 2: # spearman automatically returns a matrix for 3 and more genes
        return stats.spearmanr(mat_).correlation.astype(np.float32)
    elif mat.shape[1] == 2: # still get matrix output for 2 or 1 genes
        spearmanr = np.ones((mat.shape[1], mat.shape[1]), dtype=np.float32)
        spearmanr[0,1] = stats.spearmanr(mat_[:,0], mat_[:,1]).statistic.astype(np.float32)
        spearmanr[1,0] = spearmanr[0,1]
        return spearmanr
    else:
        return np.ones((1,1), dtype=np.float32)
            
        

############################################################################################################

def compute_mutual_information(spt_mat, seq_mat, common, thresh, pipeline_output):
    """
    Computing normalised mutual information between all pairs of random variables(?) in the expression data.
    
    NOTE: untested.
    
    MI is often used as a generalized correlation measure. It can be used for measuring co-expression, especially non-linear     
    associations. MI is well defined for discrete or categorical variables.
    
    Normalized Mutual Information (NMI) is a normalization of the Mutual Information (MI) score to 
    scale the results between 0 (no mutual information) and 1 (perfect correlation). 
    
    The function requires column-wise comparison and calculation for each variable(gene), so we transposed the adata.X before.
    
    The output is a matrix of (n_gene, n_gene) in the common expression data.
    """
    
    import warnings
    warnings.warn("The pyitlib package which is required for the current mutual information implementation is currently not in txsim's dependencies since it has too strong dependency restrictions on scikit-learn.")
    
    from pyitlib import discrete_random_variable as drv
    # Apply distance metric

    print("  - Spatial data...")
    if scipy.sparse.issparse(spt_mat):
        sim_spt = drv.information_mutual_normalised(spt_mat.toarray())
    else:    
        sim_spt = drv.information_mutual_normalised(spt_mat)
        
    print("  - Single-cell data...\n")
    if scipy.sparse.issparse(seq_mat):
        sim_seq = drv.information_mutual_normalised(seq_mat.toarray())
    else:    
        sim_seq = drv.information_mutual_normalised(seq_mat)
    
    output = compute_correlation_difference(sim_spt, sim_seq, common, thresh) if pipeline_output \
        else [sim_spt, sim_seq, common]

    return output
