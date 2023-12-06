from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy import stats


# TODO Change how normalization happens and consider using log1p
def coexpression_similarity(
        spatial_data: AnnData,
        seq_data: AnnData,
        min_cells: int = 20,
        thresh: float = 0,
        layer: str = 'lognorm',
        key: str = 'celltype',
        by_celltype: bool = False,
        correlation_measure: str = "pearson",
        pipeline_output: bool = True,
):
    """Calculate the mean difference of normalised mutual information matrix values
    
    Parameters
    ----------
    spatial_data : AnnData
        annotated ``AnnData`` object with counts from spatial data
    seq_data : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    min_cells : int, optional
        Minimum number of cells in which a gene should be detected to be considered
        expressed. By default 20
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
        mean of upper triangular difference matrix
    matrices: list
        list containing coexpression similarity matrix for each modality and gene names
    """

    # Make a copy of the anndata to prevent overwriting the original pnes
    _seq_data = seq_data.copy()
    _spatial_data = spatial_data.copy()

    # Create matrix only with intersected genes between sp and sc data
    common = _seq_data.var_names.intersection(_spatial_data.var_names)
    seq = _seq_data[:, common]
    spt = _spatial_data[:, common]
    # the number of common genes in the sp and sc data
    print(len(common), "genes are shared in both modalities")

    # if we compare the coexpression similarity based on whole gene expression matrix
    if not by_celltype:
        """
        Extract genes based on number of cells or counts and 
        keep genes that have at least `min_counts` cells to express.
        {modality}_expressed saves the remaining genes after filtering
        """
        spatial_expressed = spt.var_names[sc.pp.filter_genes(spt, min_cells=min_cells, inplace=False)[0]]
        seq_expressed = seq.var_names[sc.pp.filter_genes(seq, min_cells=min_cells, inplace=False)[0]]

        print(round(len(spatial_expressed) / len(common) * 100), "% of genes are expressed in the spatial modality",
              sep='')
        print(round(len(seq_expressed) / len(common) * 100), "% of genes are expressed in the single-cell modality",
              sep='')

        # intersected expressed genes between sp and sc data
        common_exp = spatial_expressed.intersection(seq_expressed)

        if (not len(common_exp)):
            print("No expressed genes are shared in both modalities")
            output = None if pipeline_output else [None, None, None]
        else:
            print(len(common_exp), "out of", len(common), 'genes will be used')

            # subset the commonly expressed genes
            spt_exp = spt[:, common_exp]
            seq_exp = seq[:, common_exp]

            spt_mat = spt_exp.layers[layer]
            seq_mat = seq_exp.layers[layer]

            print("Calculating co-expression similarity")

            # for mutual information, we use transposed matrix
            if correlation_measure == 'pearson':
                output = compute_pearson_correlation(spt_mat, seq_mat, common_exp, thresh, pipeline_output)
            elif correlation_measure == 'spearman':
                output = compute_spearman_correlation(spt_mat, seq_mat, common_exp, thresh, pipeline_output)
            else:
                output = compute_mutual_information(spt_mat.T, seq_mat.T, common_exp, thresh, pipeline_output)
    else:
        output = {}
        # Determine cell type populations across both modalities
        ct_spt = list(np.unique(np.array(spt.obs[key])))
        ct_seq = list(np.unique(np.array(seq.obs[key])))

        common_types = [x for x in ct_spt if x in ct_seq]

        # spt_ct_counts = spt.obs[key].value_counts()
        # seq_ct_counts = seq.obs[key].value_counts()
        # print(type(spt_ct_counts[spt_ct_counts > min_cells]))

        for c in common_types:
            print("[%s]" % c)

            # Extract expression data layer by the specific cell type 
            spt_ct = spt[spt.obs[key] == c, :]
            seq_ct = seq[seq.obs[key] == c, :]

            """
            Extract genes based on number of cells or counts and 
            keep genes that have at least `min_counts` cells to express.
            {modality}_expressed saves the remaining genes after filtering
            
            """
            spatial_expressed = spt_ct.var_names[sc.pp.filter_genes(spt_ct, min_cells=min_cells, inplace=False)[0]]
            seq_expressed = seq_ct.var_names[sc.pp.filter_genes(seq_ct, min_cells=min_cells, inplace=False)[0]]

            print(round(len(spatial_expressed) / len(common) * 100), "% of genes are expressed in the spatial modality",
                  sep='')
            print(round(len(seq_expressed) / len(common) * 100), "% of genes are expressed in the single-cell modality",
                  sep='')

            common_exp = spatial_expressed.intersection(seq_expressed)

            if (not len(common_exp)):
                print("No expressed genes are shared in both modalities. Returning None value\n")
                output[c] = None
            else:
                print(len(common_exp), "out of", len(common), 'genes will be used')

                spt_exp = spt_ct[:, common_exp]
                seq_exp = seq_ct[:, common_exp]

                spt_mat = spt_exp.layers[layer]
                seq_mat = seq_exp.layers[layer]

                print("Calculating co-expression similarity")
                if correlation_measure == 'pearson':
                    output[c] = compute_pearson_correlation(spt_mat, seq_mat, common_exp, thresh, pipeline_output)
                elif correlation_measure == 'spearman':
                    output[c] = compute_spearman_correlation(spt_mat, seq_mat, common_exp, thresh, pipeline_output)
                else:
                    output[c] = compute_mutual_information(spt_mat.T, seq_mat.T, common_exp, thresh, pipeline_output)

    return output


def compute_mutual_information(spt_mat, seq_mat, common, thresh, pipeline_output):
    """
    Computing normalised mutual information between all pairs of random variables(?) in the expression data.
    
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


def compute_pearson_correlation(spt_mat, seq_mat, common, thresh, pipeline_output):
    # Calculate pearson correlation of the gene pairs in the matrix

    print("  - Spatial data...")
    if scipy.sparse.issparse(spt_mat):
        sim_spt = np.corrcoef(spt_mat.toarray(), rowvar=False)
    else:    
        sim_spt = np.corrcoef(spt_mat.astype(float), rowvar=False)
        
    print("  - Single-cell data...\n")
    if scipy.sparse.issparse(seq_mat):
        sim_seq = np.corrcoef(seq_mat.toarray(), rowvar=False)
    else:    
        sim_seq = np.corrcoef(seq_mat.astype(float), rowvar=False)

    output = compute_correlation_difference(sim_spt, sim_seq, common, thresh) if pipeline_output \
        else [sim_spt, sim_seq, common]

    return output


def compute_spearman_correlation(spt_mat, seq_mat, common, thresh, pipeline_output):
    # Calculate spearman correlation of the gene pairs in the matrix

    print("  - Spatial data...")
    if scipy.sparse.issparse(spt_mat):
        sim_spt = stats.spearmanr(spt_mat.toarray().astype(float)).statistic
    else:
        sim_spt = stats.spearmanr(spt_mat.astype(float)).statistic
   
    print("  - Single-cell data...\n")
    if scipy.sparse.issparse(seq_mat):
        sim_seq = stats.spearmanr(seq_mat.toarray().astype(float)).statistic
    else:
        sim_seq = stats.spearmanr(seq_mat.astype(float)).statistic

    output = compute_correlation_difference(sim_spt, sim_seq, common, thresh) if pipeline_output \
        else [sim_spt, sim_seq, common]

    return output

def compute_correlation_difference(sim_spt, sim_seq, common, thresh):
    # Evaluate NaN values for each gene in every modality

    ## Spatial
    nan_res = np.sum(np.isnan(sim_spt), axis=0)
    # If one gene has NA values among all its pars
    if any(nan_res == len(common)):
        genes_nan = common[nan_res == len(common)]
        genes_nan = genes_nan.tolist()
        print("The following genes in the spatial modality resulted in NaN values:")
        for i in genes_nan: print(i)

    ## Single cell
    nan_res = np.sum(np.isnan(sim_seq), axis=0)
    if any(nan_res == len(common)):
        genes_nan = common[nan_res == len(common)]
        genes_nan = genes_nan.tolist()
        print("The following genes in the single-cell modality resulted in NaN values:")
        for i in genes_nan: print(i)
    """
    Significant pairs from scRNAseq data can be filtered by the threshold. Pairs with correlations
    below the threshold (by magnitude, abstract) will be ignored when calculating mean.
    """

    # If threshold, mask values with NaNs
    sim_seq[np.abs(sim_seq) < np.abs(thresh)] = np.nan
    sim_seq[np.tril_indices(len(common))] = np.nan

    # Calculate difference between modalities
    diff = sim_seq - sim_spt
    mean = np.nanmean(np.absolute(diff)) / 2
    output = mean
    return output
