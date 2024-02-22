import numpy as np
import anndata as ad
from typing import Tuple
from scipy.sparse import issparse
import pandas as pd
from ..metrics import knn_mixing_per_cell_score
from ..metrics import relative_pairwise_gene_expression

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

def _get_relative_expression_similarity_across_genes_grid(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    obs_key: str = "celltype",
    layer: str = 'lognorm',
    cells_x_col: str = "x",
    cells_y_col: str = "y",
    normalization: str = "global",
    contribution: bool = True,
):
    """Calculate the similarity of pairwise gene expression differences for all pairs of genes in the panel, between the two modalities 
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from scRNAseq data
    key: str (default: 'celltype')
        .obs column of ``AnnData`` that contains celltype information
    layer: str (default: 'lognorm')
        layer of ```AnnData`` to use to compute the metric
    pipeline_output: bool (default: True)
        whether to return only the overall metric (if False, will return the overall metric, per-gene metric and per-celltype metric)
    Returns
    -------
    overall_metric: np.ndarray
        overall similarity of relative pairwise gene expression for all pairs of genes in the panel, b/t the scRNAseq and spatial data
    per_gene_metric: np.ndarray
        similarity of relative pairwise gene expression per gene, b/t the scRNAseq and spatial data
    per_celltype_metric: np.ndarray
        similarity of relative pairwise gene expression per celltype, b/t the scRNAseq and spatial data

    """   #TODO: Correct the docstring
    assert normalization in ["global", "local"], "normalization must be either 'global' or 'local'"

    ### SET UP
    # set the .X layer of each of the adatas to be log-normalized counts
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]

    # take the intersection of genes present in adata_sp and adata_sc, as a list
    intersect = list(set(adata_sp.var_names).intersection(set(adata_sc.var_names)))

    # subset adata_sc and adata_sp to only include genes in the intersection of adata_sp and adata_sc
    adata_sc = adata_sc[:, intersect].copy()
    adata_sp = adata_sp[:, intersect].copy()

    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.layers[layer] = a.layers[layer].toarray()

    # only consider cells within the specified region
    adata_sp = adata_sp[(adata_sp.obs[cells_y_col] >= region_range[0][0]) &
                        (adata_sp.obs[cells_y_col] < region_range[0][1]) &
                        (adata_sp.obs[cells_x_col] >= region_range[1][0]) &
                        (adata_sp.obs[cells_x_col] < region_range[1][1])]

    # add "bin" label columns to adata_sp
    adata_sp.obs["bin_y"] = pd.cut(adata_sp.obs[cells_y_col], bins=bins[0], labels=False)
    adata_sp.obs["bin_x"] = pd.cut(adata_sp.obs[cells_x_col], bins=bins[1], labels=False)

    # create empty matrices to store the overall, per-celltype and per-gene metrics
    overall_metric_matrix = np.zeros((bins[0], bins[1]))
    celltype_order = adata_sp.obs[obs_key].unique()
    per_celltype_metric_matrix = np.zeros((bins[0], bins[1], len(celltype_order)))
    per_gene_metric_matrix = np.zeros((bins[0], bins[1], adata_sp.shape[1]))

    for y_bin in adata_sp.obs["bin_y"].unique():
        for x_bin in adata_sp.obs["bin_x"].unique():
            # subset the spatial data to only include cells in the current grid field
            adata_sp_local = adata_sp[(adata_sp.obs["bin_y"] == y_bin) & (adata_sp.obs["bin_x"] == x_bin)]

            # find the unique celltypes in the grid field, that are both in the adata_sc and in the adata_sp
            unique_celltypes=adata_sc.obs.loc[adata_sc.obs[obs_key].isin(adata_sp_local.obs[obs_key]),obs_key].unique()

            #### CALCULATE EACH GENE'S MEAN EXPRESSION PER CELL TYPE
            # get the adata_sc cell x gene matrix as a pandas dataframe (w gene names as column names)
            exp_sc = pd.DataFrame(adata_sc.layers[layer], columns=adata_sc.var.index)

            # get the adata_sp cell x gene matrix as a pandas dataframe, once for the local grid field and once for the entire dataset
            exp_sp_local = pd.DataFrame(adata_sp_local.layers[layer], columns=adata_sp_local.var.index)
            if normalization == "global":
                exp_sp_global = pd.DataFrame(adata_sp.layers[layer], columns=adata_sp.var.index)

            # add "celltype" label column to exp_sc & exp_sp cell x gene matrices
            exp_sc[obs_key] = list(adata_sc.obs[obs_key])
            exp_sp_local[obs_key] = list(adata_sp_local.obs[obs_key])
            if normalization == "global":
                exp_sp_global[obs_key] = list(adata_sp.obs[obs_key])

            # delete all cells from the exp matrices if they aren't in the set of intersecting celltypes b/t sc & sp data
            exp_sc = exp_sc.loc[exp_sc[obs_key].isin(unique_celltypes), :]
            exp_sp_local = exp_sp_local.loc[exp_sp_local[obs_key].isin(unique_celltypes), :]
            if normalization == "global":
                exp_sp_global = exp_sp_global.loc[exp_sp_global[obs_key].isin(unique_celltypes), :]

            # find the mean expression for each gene for each celltype in sc and sp data
            mean_celltype_sc = exp_sc.groupby(obs_key).mean()
            mean_celltype_sp_local = exp_sp_local.groupby(obs_key).mean()
            if normalization == "global":
                mean_celltype_sp_global = exp_sp_global.groupby(obs_key).mean()

            # sort genes in alphabetical order
            mean_celltype_sc = mean_celltype_sc.loc[:, mean_celltype_sc.columns.sort_values()]
            mean_celltype_sp_local = mean_celltype_sp_local.loc[:, mean_celltype_sp_local.columns.sort_values()]
            if normalization == "global":
                mean_celltype_sp_global = mean_celltype_sp_global.loc[:, mean_celltype_sp_global.columns.sort_values()]

            #### CALCULATE EXPRESSION DIFFERENCES BETWEEN ALL PAIRS OF GENES FOR EACH CELLTYPE
            mean_celltype_sc_np = mean_celltype_sc.to_numpy()
            pairwise_distances_sc = mean_celltype_sc_np[:, :, np.newaxis] - mean_celltype_sc_np[:, np.newaxis, :]
            pairwise_distances_sc = pairwise_distances_sc.transpose(
                (1, 2, 0))  # results in np.array of dimensions (num_genes, num_genes, num_celltypes)

            mean_celltype_sp_np_local = mean_celltype_sp_local.to_numpy()
            pairwise_distances_sp_local = mean_celltype_sp_np_local[:, :, np.newaxis] - mean_celltype_sp_np_local[:, np.newaxis, :]
            pairwise_distances_sp_local = pairwise_distances_sp_local.transpose(
                (1, 2, 0))  # results in np.array of dimensions (num_genes, num_genes, num_celltypes)

            if normalization == "global":
                mean_celltype_sp_np_global = mean_celltype_sp_global.to_numpy()
                pairwise_distances_sp_global = mean_celltype_sp_np_global[:, :, np.newaxis] - mean_celltype_sp_np_global[:, np.newaxis, :]
                pairwise_distances_sp_global = pairwise_distances_sp_global.transpose(
                    (1, 2, 0))  # results in np.array of dimensions (num_genes, num_genes, num_celltypes)

            #### NORMALIZE PAIRWISE EXPRESSION DIFFERENCES
            ## normalization is performed by dividing by the sum of the absolute values of all differences between pairs of genes
            ## furthermore, to ensure that the values are comparable across datasets with different numbers of genes, we scale the result by a factor of
            ## num_genes^2
            # calculate sum of absolute distances
            abs_diff_sc = np.absolute(pairwise_distances_sc)
            abs_diff_sum_sc = np.sum(abs_diff_sc, axis=(0, 1))

            if normalization == "local":
                abs_diff_sp = np.absolute(pairwise_distances_sp_local)
            elif normalization == "global":
                abs_diff_sp = np.absolute(pairwise_distances_sp_global)
            abs_diff_sum_sp = np.sum(abs_diff_sp, axis=(0, 1))

            # calculate normalization factor
            norm_factor_sc = mean_celltype_sc.shape[1] ** 2 * abs_diff_sum_sc
            norm_factor_sp = mean_celltype_sc.shape[1] ** 2 * abs_diff_sum_sp

            # perform normalization
            # exclude the ones with norm_factor_sc, norm_factor_sp with zero
            pairwise_distances_sc[:, :, norm_factor_sc != 0] = np.divide(pairwise_distances_sc[:, :, norm_factor_sc != 0],
                                                                         norm_factor_sc[norm_factor_sc != 0])
            # the following is the key difference for calculating the local metric version, as we divide the LOCAL pairwise
            # distances by the global or local normalization factor
            pairwise_distances_sp_local[:, :, norm_factor_sp != 0] = np.divide(pairwise_distances_sp_local[:, :, norm_factor_sp != 0],
                                                                         norm_factor_sp[norm_factor_sp != 0])
            norm_pairwise_distances_sc = pairwise_distances_sc
            norm_pairwise_distances_sp_local = pairwise_distances_sp_local

            ##### CALCULATE OVERALL SCORE,PER-GENE SCORES, PER-CELLTYPE SCORES
            # First, sum over the differences between modalities in relative pairwise gene expression distances
            # The overall metric is then bounded at a maximum of 1, representing perfect similarity of relative gene expression between modalities.
            ## Furthermore, the metric is constructed such that, when its value is 0, this represents perfect dissimilarity of
            ## relative gene expression between modalities (such that each gene's expression value in each gene pair is swapped).
            overall_score = np.sum(np.absolute(norm_pairwise_distances_sp_local - norm_pairwise_distances_sc), axis=None)
            overall_metric = 1 - (overall_score / (2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=None)))
            overall_metric_matrix[y_bin, x_bin] = overall_metric

            # We can further compute the metric on a per-gene and per-celltype basis
            per_gene_score = np.sum(np.absolute(norm_pairwise_distances_sp_local - norm_pairwise_distances_sc), axis=(1, 2))
            per_gene_metric = 1 - (per_gene_score / (2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=(1, 2))))
            per_gene_metric = pd.DataFrame(per_gene_metric, index=mean_celltype_sc.columns,
                                           columns=['score'])  # add back the gene labels
            per_gene_metric_matrix[y_bin, x_bin] = np.squeeze(per_gene_metric)

            per_celltype_score = np.sum(np.absolute(norm_pairwise_distances_sp_local - norm_pairwise_distances_sc), axis=(0, 1))
            per_celltype_metric = 1 - (per_celltype_score / (2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=(0, 1))))
            per_celltype_metric = pd.DataFrame(per_celltype_metric, index=mean_celltype_sc.index,
                                               columns=['score'])  # add back the celltype labels
            # bring in order of celltype_order and fill missing celltypes with np.nan
            per_celltype_metric = per_celltype_metric.reindex(celltype_order)
            per_celltype_metric_matrix[y_bin, x_bin] = np.squeeze(per_celltype_metric)

    def local_metric_contribution(local_metric_matrix, global_metric):
        nr_grid_fields = bins[0] * bins[1]
        metric_contribution_matrix = np.zeros((bins[0], bins[1]))
        for y_bin in adata_sp.obs["bin_y"].unique():
            for x_bin in adata_sp.obs["bin_x"].unique():
                # calculate the difference in the local metric matrix explained by the grid field
                diff_explained_by_grid_field = (nr_grid_fields * (1 - local_metric_matrix[y_bin, x_bin])
                                                / (nr_grid_fields - local_metric_matrix.sum()))

                # calculate the proportion of the global metric that is explained by the grid field
                diff_to_explain = 1 - global_metric
                metric_contribution_matrix[y_bin, x_bin] = 1 - (diff_explained_by_grid_field * diff_to_explain)

        return metric_contribution_matrix


    # calculate the contribution of each grid field to the overall metric, if contribution is set to True
    if contribution:
        # calculate global metrics (overall, per-gene, and per-celltype)
        overall_metric, per_gene_metric, per_celltype_metric = relative_pairwise_gene_expression(adata_sp, adata_sc, key=obs_key, pipeline_output=False)

        overall_metric_matrix = local_metric_contribution(overall_metric_matrix, overall_metric)

        for gene in per_gene_metric_matrix.index:
            per_gene_metric_matrix.loc[gene] = local_metric_contribution(per_gene_metric_matrix.loc[gene], per_gene_metric.loc[gene])

        for celltype in per_celltype_metric_matrix.index:
            per_celltype_metric_matrix.loc[celltype] = local_metric_contribution(per_celltype_metric_matrix.loc[celltype], per_celltype_metric.loc[celltype])


    # remove the bin columns from adata_sp.obs
    adata_sp.obs = adata_sp.obs.drop(columns=["bin_y", "bin_x"])

    return overall_metric_matrix, per_gene_metric_matrix, per_celltype_metric_matrix

