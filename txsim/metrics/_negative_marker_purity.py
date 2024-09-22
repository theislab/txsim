import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse

# TODO: probably better to combine both metrics to one function with argument `flavor in ["reads", "cells"]`
# TODO: Write test
# TODO: Investigate the importance of setting max_ratio_cells, minimum_exp, and min_number_cells

def negative_marker_purity_cells(
    adata_sp: AnnData, 
    adata_sc: AnnData, 
    key: str='celltype',
    layer: str='raw',
    pipeline_output: bool=True
) -> float | tuple[float, pd.Series, pd.Series]:
    """ Negative marker purity aims to measure read leakeage between cells in spatial datasets. 
    
    For this, we calculate the increase in positive cells assigned in spatial datasets to pairs of genes-celltyes with 
    no/very low expression in scRNAseq.
    
    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    key : str
        Celltype key in adata_sp.obs and adata_sc.obs
    layer : str
        Layer of ``AnnData`` to use to compute the metric
    pipeline_output : float, optional
        Boolean for whether to use the function in the pipeline or not
    Returns
    -------
    negative marker purity : float
       Increase in proportion of positive cells assigned in spatial data to pairs of genes-celltyes with no/very low expression in scRNAseq
    """   
    
    # Set threshold parameters
    min_number_cells=10 # minimum number of cells belonging to a cluster to consider it in the analysis
    max_ratio_cells=0.005 # maximum ratio of cells expressing a marker to call it a negative marker gene-ct pair
    
    # Subset adata_sc to genes of spatial data
    adata_sc = adata_sc[:,adata_sp.var_names].copy()
    
    # TMP fix for sparse matrices, ideally we don't convert, and instead have calculations for sparse/non-sparse
    for a in [adata_sc, adata_sp]:
        if issparse(a.layers[layer]):
            a.layers[layer] = a.layers[layer].toarray()
    
    # Get cell types that we find in both modalities
    shared_celltypes = adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    
    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[shared_celltypes]
    celltype_count_sp = adata_sp.obs[key].value_counts().loc[shared_celltypes]
    ct_filter = (celltype_count_sc >= min_number_cells) & (celltype_count_sp >= min_number_cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()
    
    # Return nan if too few cell types were found
    if len(celltypes) < 2:
        print("Not enough cell types (>1) eligible to calculate negative marker purity")
        negative_marker_purity = np.nan
        if pipeline_output==True:
            return negative_marker_purity
        else:
            purity_per_gene = pd.Series(index=adata_sp.var_names, data=np.nan)
            purity_per_celltype = pd.Series(index=celltypes, data=np.nan)
            return negative_marker_purity, purity_per_gene, purity_per_celltype
    
    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    adata_sp = adata_sp[adata_sp.obs[key].isin(celltypes)]
    
    # besides the threshold parameter till here neg. marker purity reads and cells are the same
    
    
    # Get ratio of positive cells per cell type
    pos_exp_sc = pd.DataFrame(adata_sc.layers[layer] > 0,columns=adata_sp.var_names)
    pos_exp_sp = pd.DataFrame(adata_sp.layers[layer] > 0,columns=adata_sp.var_names)
    pos_exp_sc['celltype'] = list(adata_sc.obs[key])
    pos_exp_sp['celltype'] = list(adata_sp.obs[key])
    ratio_celltype_sc = pos_exp_sc.groupby('celltype').mean()
    ratio_celltype_sp = pos_exp_sp.groupby('celltype').mean()
    
    # Get gene-cell type pairs with negative marker expression
    neg_marker_mask = np.array(ratio_celltype_sc < max_ratio_cells)
    
    # Return nan if no negative markers were found
    if np.sum(neg_marker_mask) < 1:
        print("No negative markers were found in the sc data reference.")
        negative_marker_purity = np.nan
        if pipeline_output==True:
            return negative_marker_purity
        else:
            purity_per_gene = pd.Series(index=adata_sp.var_names, data=np.nan)
            purity_per_celltype = pd.Series(index=celltypes, data=np.nan)
            return negative_marker_purity, purity_per_gene, purity_per_celltype
    
    # Get pos cell ratios in negative marker-cell type pairs
    lowvals_sc = np.full_like(neg_marker_mask, np.nan, dtype=np.float32)
    lowvals_sc = ratio_celltype_sc.values[neg_marker_mask]
    lowvals_sp = ratio_celltype_sp.values[neg_marker_mask]
    
    # Take the mean over the normalized expressions of the genes' negative cell types
    mean_sc_low_ratio = np.nanmean(lowvals_sc)
    mean_sp_low_ratio = np.nanmean(lowvals_sp)
    
    # Calculate summary metric
    negative_marker_purity = 1
    if mean_sp_low_ratio > mean_sc_low_ratio:
        negative_marker_purity -= (mean_sp_low_ratio - mean_sc_low_ratio)
    
    if pipeline_output:
        return negative_marker_purity
    else:
        # Calculate per gene and per cell type purities
        purities = (ratio_celltype_sc - ratio_celltype_sp).clip(0,None)
        purities.values[~neg_marker_mask] = np.nan
        purities = purities.loc[~(purities.isnull().all(axis=1)), ~(purities.isnull().all(axis=0))]
        purity_per_gene = purities.mean(axis=0, skipna=True)
        purity_per_celltype = purities.mean(axis=1, skipna=True)
        
        return negative_marker_purity, purity_per_gene, purity_per_celltype


def negative_marker_purity_reads(
        adata_sp: AnnData, 
        adata_sc: AnnData, 
        key: str='celltype', 
        layer: str='raw', 
        pipeline_output: bool=True
):
    """ Negative marker purity aims to measure read leakeage between cells in spatial datasets. 
    
    For this, we calculate the increase in reads assigned in spatial datasets to pairs of genes-celltyes with no/very low expression in scRNAseq
    
    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    key : str
        Celltype key in adata_sp.obs and adata_sc.obs
    layer : str
        Layer of ``AnnData`` to use to compute the metric
    pipeline_output : float, optional
        Boolean for whether to use the function in the pipeline or not
    Returns
    -------
    negative marker purity : float
       Increase in proportion of reads assigned in spatial data to pairs of genes-celltyes with no/very low expression in scRNAseq
    """   
    
    # Set threshold parameters
    min_number_cells=10 #minimum number of cells belonging to a cluster to consider it in the analysis
    minimum_exp=0.005 #maximum relative expression allowed in a gene in a cluster to consider the gene-celltype pair the analysis 
    
    # Subset adata_sc to genes of spatial data
    adata_sc = adata_sc[:,adata_sp.var_names].copy()
    
    # TMP fix for sparse matrices, ideally we don't convert, and instead have calculations for sparse/non-sparse
    for a in [adata_sc, adata_sp]:
        if issparse(a.layers[layer]):
            a.layers[layer] = a.layers[layer].toarray()
    
    # Get cell types that we find in both modalities
    shared_celltypes = adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    
    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[shared_celltypes]
    celltype_count_sp = adata_sp.obs[key].value_counts().loc[shared_celltypes]
    ct_filter = (celltype_count_sc >= min_number_cells) & (celltype_count_sp >= min_number_cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()
    
    # Return nan if too few cell types were found
    if len(celltypes) < 2:
        print("Not enough cell types (>1) eligible to calculate negative marker purity")
        negative_marker_purity = np.nan
        if pipeline_output==True:
            return negative_marker_purity
        else:
            purity_per_gene = pd.Series(index=adata_sp.var_names, data=np.nan)
            purity_per_celltype = pd.Series(index=celltypes, data=np.nan)
            return negative_marker_purity, purity_per_gene, purity_per_celltype
    
    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    adata_sp = adata_sp[adata_sp.obs[key].isin(celltypes)]
    
    # Get mean expression per cell type
    exp_sc = pd.DataFrame(adata_sc.layers[layer],columns=adata_sp.var_names)
    exp_sp = pd.DataFrame(adata_sp.layers[layer],columns=adata_sp.var_names)
    exp_sc['celltype'] = list(adata_sc.obs[key])
    exp_sp['celltype'] = list(adata_sp.obs[key])
    mean_celltype_sc = exp_sc.groupby('celltype').mean()
    mean_celltype_sp = exp_sp.groupby('celltype').mean()
    
    # Get mean expressions relative to mean over cell types (this will be used for filtering based on minimum_exp)
    mean_ct_sc_rel = mean_celltype_sc.div(mean_celltype_sc.mean(axis=0),axis=1)
    mean_ct_sp_rel = mean_celltype_sp.div(mean_celltype_sp.mean(axis=0),axis=1)
    # Potential TODO: this step is actually a tricky one 
    #       - at first it was normalised wrt sum instead of mean, imo it's not a good choice since for a high number of
    #         cell types we then always get many cell types < minimum_exp
    #       - we could normalize to the mean expression in cells that do express the gene probably best to have this 
    #         mean per cell type to account for cell type proportion differences. However, if spatial data has a lot of 
    #         leakage we add many more cells into the mean calculation which lowers the mean. 
    #       - Let's stick with the current one for now, however others should be explored at some point
    
    # Get normalized mean expressions over cell types (this will be summed up over negative cell types)
    mean_ct_sc_norm = mean_celltype_sc.div(mean_celltype_sc.sum(axis=0),axis=1)
    mean_ct_sp_norm = mean_celltype_sp.div(mean_celltype_sp.sum(axis=0),axis=1)
    # Explanation: The idea is to measure which ratio of mean expressions is shifted towards negative clusters.
    #              With this approach the metric stays between 0 and 1
    
    
    # Get gene-cell type pairs with negative marker expression
    #neg_marker_mask = np.array(mean_ct_sc_rel < minimum_exp) #NOTE: Old criteria, lead to too few negative markers
    neg_marker_mask = np.array(mean_ct_sc_norm < minimum_exp) #TODO: Add an extra function to retrieve the negative 
    # marker-cell type pairs, and use that function across NMP formulations (including the local one). Also delete
    # the lines above that are not needed anymore. This new constraint is simpler than the previous one. Check the 
    # negative marker selection on different datasets with the plotting function to see how well it generalises.
    
    # Return nan if no negative markers were found
    if np.sum(neg_marker_mask) < 1:
        print("No negative markers were found in the sc data reference.")
        negative_marker_purity = np.nan
        if pipeline_output==True:
            return negative_marker_purity
        else:
            purity_per_gene = pd.Series(index=adata_sp.var_names, data=np.nan)
            purity_per_celltype = pd.Series(index=celltypes, data=np.nan)
            return negative_marker_purity, purity_per_gene, purity_per_celltype
    
    # Get marker expressions in negative marker-cell type pairs
    lowvals_sc = np.full_like(neg_marker_mask, np.nan, dtype=np.float32)
    lowvals_sc = mean_ct_sc_norm.values[neg_marker_mask]
    lowvals_sp = mean_ct_sp_norm.values[neg_marker_mask]
    
    # Take the mean over the normalized expressions of the genes' negative cell types
    mean_sc_lowexp = np.nanmean(lowvals_sc)
    mean_sp_lowexp = np.nanmean(lowvals_sp)
    
    # Calculate summary metric
    negative_marker_purity = 1
    if mean_sp_lowexp > mean_sc_lowexp:
        negative_marker_purity -= (mean_sp_lowexp - mean_sc_lowexp)
    
    
    if pipeline_output:
        return negative_marker_purity
    else:
        # Calculate per gene and per cell type purities
        purities = (mean_ct_sp_norm - mean_ct_sc_norm).clip(0,None)
        purities.values[~neg_marker_mask] = np.nan
        purities = purities.loc[~(purities.isnull().all(axis=1)), ~(purities.isnull().all(axis=0))]
        purity_per_gene = purities.mean(axis=0, skipna=True)
        purity_per_celltype = purities.mean(axis=1, skipna=True)
        
        return negative_marker_purity, purity_per_gene, purity_per_celltype

    
    
def negative_marker_purity_reads_FP_based_optimum(
        adata_sc: AnnData, 
        spots: pd.DataFrame, 
        FP_rate: float, 
        key: str = "celltype",
        gene_key: str = "Gene",
    ):
    """ Calculate the negative marker purity for an optimal segmentation of all annotated reads based on a false positive rate
    
    The false positive rate refers to rates e.g. based on control probes as used in e.g. Xenium ("Negative control 
    probe rate"). This refers to falsely detected spots. Note that e.g. falsely decoded spots as e.g. in MERFISH are
    conceptually different. For these rates a different compuatation approach would be needed.
    
    The computation goes as follows. txsim input spots tables come with expert annotated cell type annotations mapped to 
    spots. We only take into account those spots that belong to cell types occuring in adata_sc.obs[key]. The negative
    marker-cell type pairs are calculated as in the NMP metrics. For each cell type all spots of negative markers are
    deleted except of FP_rate * n_spots_g_ct (n_spots_g_ct = the number of spots of the given negative marker in the
    given cell type). Finally the negative marker purity is calculated as usual.
    
    Arguments
    ---------
    adata_sc: AnnData
        Annotated ``AnnData`` object with counts scRNAseq data. Raw counts are expected in adata_sc['raw'].
    spots: pd.DataFrame
        The spots table with columns spots["Gene"] and spots[key].
    FP_rate: float
        False positive rate of counts in the given dataset.
    key: str
        Obs key for cell types of `adata.obs` and `spots`.
    gene_key: str
        Gene columns name of `spots`
    
    Returns
    -------
    float
        Negative marker purity for the theoretical case of ideal segmentations except of falsly positive detected reads
        of the peak calling. (Negative marker purity := 1 - (X_neg,sp - X_neg,sc), where X_neg,m is the normalized mean
        expression over negative marker-cell type pairs in modality m)
    float
        Like the first return value, except that we neglect the comparison term of scRNAseq, i.e. 
        Negative marker purity := 1 - X_neg,sp.
        
        
    """
    
    celltypes_sc = adata_sc.obs[key].unique()
    spots = spots.loc[spots[key].isin(celltypes_sc)]
    shared_celltypes = [ct for ct in spots[key].unique() if ct in celltypes_sc]
    genes = spots[gene_key].unique().tolist()
    assert np.all([g in adata_sc.var_names for g in genes]); "All genes in `spots[gene_key]` must occur in `adata_sc.var_names`"
    
    # Get negative markers of adata_sc #TODO: Provide a shared function for this
    min_number_cells=10 
    minimum_exp=0.005 #maximum relative expression allowed in a gene in a cluster to consider the gene-celltype pair the analysis 
    # Subset adata_sc to genes of spatial data
    adata_sc = adata_sc[:,genes]
    if issparse(adata_sc.layers["raw"]):
        adata_sc.layers["raw"] = adata_sc.layers["raw"].toarray()
    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[shared_celltypes]
    ct_filter = (celltype_count_sc >= min_number_cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()
    # Return nan if too few cell types were found
    if len(celltypes) < 2:
        print("Not enough cell types (>1) eligible to calculate negative marker purity")
        negative_marker_purity = 'nan'
        return negative_marker_purity
    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    # Get mean expression per cell type
    exp_sc = pd.DataFrame(adata_sc.layers['raw'],columns=genes)
    exp_sc['celltype'] = list(adata_sc.obs[key])
    mean_celltype_sc = exp_sc.groupby('celltype').mean()
    # Get normalized mean expressions over cell types (this will be summed up over negative cell types)
    mean_ct_sc_norm = mean_celltype_sc.div(mean_celltype_sc.sum(axis=0),axis=1)
    # Get gene-cell type pairs with negative marker expression
    #neg_marker_mask = np.array(mean_ct_sc_rel < minimum_exp) #NOTE: Old criteria, lead to too few negative markers
    neg_marker_mask = np.array(mean_ct_sc_norm.loc[celltypes,genes] < minimum_exp) 
    
    # NOTE: This here is the actual interesting part.
    # We apply a different approach for the spatial reads here, as we don't have segmented cells
    # To achieve the cell type balance as in the NMP formulation we need a weighted sum of reads
    counts_per_ct = [(spots[key]==ct).sum() for ct in celltypes]
    min_n_counts = min(counts_per_ct)
    ct_weights = [min_n_counts/count for count in counts_per_ct]
    sp_counts_per_gene_and_ct = pd.crosstab(spots[key],spots[gene_key]).loc[celltypes,genes]
    sp_counts_per_gene_and_ct = sp_counts_per_gene_and_ct.mul(ct_weights, axis=0)
    # Throw out negative marker counts in cells where they shouldn't be (except of FP_rate * n_counts)
    tmp = sp_counts_per_gene_and_ct.values.copy()
    tmp[neg_marker_mask] *= FP_rate
    sp_counts_per_gene_and_ct.loc[:,:] = tmp
    mean_ct_sp_norm = sp_counts_per_gene_and_ct.div(sp_counts_per_gene_and_ct.sum(axis=0),axis=1)
    
    #TODO: this should then just call some final NMP fct maybe(?)
    # Get marker expressions in negative marker-cell type pairs
    lowvals_sc = np.full_like(neg_marker_mask, np.nan, dtype=np.float32)
    lowvals_sc = mean_ct_sc_norm.values[neg_marker_mask]
    lowvals_sp = mean_ct_sp_norm.values[neg_marker_mask]
    
    # Take the mean over the normalized expressions of the genes' negative cell types
    mean_sc_lowexp = np.nanmean(lowvals_sc)
    mean_sp_lowexp = np.nanmean(lowvals_sp)
    
    # Calculate summary metric
    negative_marker_purity = 1
    if mean_sp_lowexp > mean_sc_lowexp:
        negative_marker_purity -= (mean_sp_lowexp - mean_sc_lowexp)
    
    return negative_marker_purity, (1 - mean_sp_lowexp)

