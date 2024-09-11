import numpy as np
import anndata as ad
from typing import Tuple
import pandas as pd
from scipy.sparse import issparse

from ..metrics import knn_mixing_per_cell_score
from ..metrics import mean_proportion_deviation
from ..metrics import relative_pairwise_gene_expression
from ..metrics import relative_pairwise_celltype_expression
from ..metrics._jensen_shannon_distance import jensen_shannon_distance
from ._utils import _get_bin_ids

#TODO: "negative_marker_purity_reads", "negative_marker_purity_cells", "coexpression_similarity", 
#    "relative_expression_similarity_across_celltypes",


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


def _get_celltype_proportions_grid(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    abs_score: bool = True,
    ct_set: str = "union",
    obs_key: str = "celltype",
    cells_x_col: str = "x",
    cells_y_col: str = "y"
) -> np.ndarray:
    """Calculate the average difference in cell type proportions within each grid bin.

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
    abs_score : bool, default True
        Whether to return the absolute score between 0 and 1 (higher means proportions more consistent with sc data)
        or a relative score between -1 and +1
        (negative score means cell types that are less abundant in spatial than sc data are more common per grid field,
        positive score cell types that are more abundant in spatial than sc data are more common).
    ct_set : str, default "union"
        How to combine the different cell types from both data sets.
        Supported: ["union", "intersection"]
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the average difference in cell type proportions in each grid bin.
    """

    # Get cell type proportion deviation scores per cell type
    _, df_props = mean_proportion_deviation(adata_sp, adata_sc, ct_set=ct_set, obs_key=obs_key, pipeline_output=False)

    # Map each cell's cell type to its corresponding proportion deviation
    df_cells = adata_sp.obs[[cells_x_col, cells_y_col, obs_key]]
    df_cells["sp_minus_sc"] = df_cells[obs_key].map(df_props["sp_minus_sc"]).astype(float)

    # Count number of cells per grid field for mean calculation
    ct_grid_counts = np.histogram2d(df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, range=region_range)[0]

    # Sum cell scores per grid field
    ct_score_hist = np.full_like(ct_grid_counts, 0)
    ct_score_hist = np.histogram2d(
        df_cells[cells_y_col], df_cells[cells_x_col], bins=bins, range=region_range, 
        weights=df_cells["sp_minus_sc"].abs() if abs_score else df_cells["sp_minus_sc"]
    )[0]

    # Get mean scores by normalizing with number of cells per grid field
    ct_score_hist = ct_score_hist / ct_grid_counts

    return ct_score_hist


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
    """Calculate the similarity of pairwise gene expression differences for all pairs of genes in the panel, between the
    two modalities, within each grid bin.
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from scRNAseq data
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    layer: str (default: 'lognorm')
        Layer of ``AnnData`` to use to compute the metric.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.
    normalization: str (default: 'global')
        The type of normalization to use for computing the metric. If set to 'global', the entire spatial dataset is used
        to normalize the pairwise gene expression differences for the spatial modality.
        If set to 'local', only the local grid field is used to normalize the pairwise gene expression differences.
        Can be either 'global' or 'local'.
    contribution: bool (default: True)
        Set to True to calculate the contribution of each grid field to the overall metric, or False to calculate the metric itself.

    Returns
    -------
    overall_metric: np.ndarray
        Matrix containing the local overall similarity of relative pairwise gene expression for all pairs of genes in the panel,
        b/t the scRNAseq and spatial data.
    """
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

    # get bin ids
    adata_sp.obs = _get_bin_ids(adata_sp.obs, region_range, bins, cells_x_col, cells_y_col)

    # only consider cells within the specified region
    adata_sp_region_range = adata_sp[(adata_sp.obs["y_bin"] != -1) & (adata_sp.obs["x_bin"] != -1)]

    # create an empty matrix to store the computed metric for each grid field
    overall_metric_matrix = np.zeros((bins[0], bins[1]))

    for y_bin in adata_sp_region_range.obs["y_bin"].unique():
        for x_bin in adata_sp_region_range.obs["x_bin"].unique():
            # subset the spatial data to only include cells in the current grid field
            adata_sp_local = adata_sp_region_range[(adata_sp_region_range.obs["y_bin"] == y_bin) & (adata_sp_region_range.obs["x_bin"] == x_bin)]

            # find the unique celltypes in the grid field, that are both in the adata_sc and in the adata_sp
            unique_celltypes = adata_sc.obs.loc[adata_sc.obs[obs_key].isin(adata_sp_local.obs[obs_key]),obs_key].unique()

            # If there are no cells in the grid field or no overlap between cell types in sc and sp data, set the local metric to NaN
            if len(unique_celltypes) == 0:
                overall_metric_matrix[y_bin, x_bin] = np.nan
                continue

            #### CALCULATE EACH GENE'S MEAN EXPRESSION PER CELL TYPE
            # get the adata_sc cell x gene matrix as a pandas dataframe (w gene names as column names)
            exp_sc = pd.DataFrame(adata_sc.layers[layer], columns=adata_sc.var.index)

            # get the adata_sp cell x gene matrix as a pandas dataframe, once for the local grid field and once for the entire dataset
            exp_sp_local = pd.DataFrame(adata_sp_local.layers[layer], columns=adata_sp_local.var.index)

            # add "celltype" label column to exp_sc & exp_sp cell x gene matrices
            exp_sc[obs_key] = list(adata_sc.obs[obs_key])
            exp_sp_local[obs_key] = list(adata_sp_local.obs[obs_key])

            # delete all cells from the exp matrices if they aren't in the set of intersecting celltypes b/t sc & sp data
            exp_sc = exp_sc.loc[exp_sc[obs_key].isin(unique_celltypes), :]
            exp_sp_local = exp_sp_local.loc[exp_sp_local[obs_key].isin(unique_celltypes), :]

            # find the mean expression for each gene for each celltype in sc and sp data
            mean_celltype_sc = exp_sc.groupby(obs_key).mean()
            mean_celltype_sp_local = exp_sp_local.groupby(obs_key).mean()

            # sort genes in alphabetical order
            mean_celltype_sc = mean_celltype_sc.loc[:, mean_celltype_sc.columns.sort_values()]
            mean_celltype_sp_local = mean_celltype_sp_local.loc[:, mean_celltype_sp_local.columns.sort_values()]

            #### CALCULATE EXPRESSION DIFFERENCES BETWEEN ALL PAIRS OF GENES FOR EACH CELLTYPE
            mean_celltype_sc_np = mean_celltype_sc.to_numpy()
            pairwise_distances_sc = mean_celltype_sc_np[:, :, np.newaxis] - mean_celltype_sc_np[:, np.newaxis, :]
            pairwise_distances_sc = pairwise_distances_sc.transpose(
                (1, 2, 0))  # results in np.array of dimensions (num_genes, num_genes, num_celltypes)

            mean_celltype_sp_np_local = mean_celltype_sp_local.to_numpy()
            pairwise_distances_sp_local = mean_celltype_sp_np_local[:, :, np.newaxis] - mean_celltype_sp_np_local[:, np.newaxis, :]
            pairwise_distances_sp_local = pairwise_distances_sp_local.transpose(
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
                # prepare entire spatial dataset (not just in the region range) to compute the global normalization factor
                exp_sp_global = pd.DataFrame(adata_sp.layers[layer], columns=adata_sp.var.index)
                exp_sp_global[obs_key] = list(adata_sp.obs[obs_key])
                exp_sp_global = exp_sp_global.loc[exp_sp_global[obs_key].isin(unique_celltypes), :]
                mean_celltype_sp_global = exp_sp_global.groupby(obs_key).mean()
                mean_celltype_sp_global = mean_celltype_sp_global.loc[:, mean_celltype_sp_global.columns.sort_values()]

                mean_celltype_sp_np_global = mean_celltype_sp_global.to_numpy()
                pairwise_distances_sp_global = mean_celltype_sp_np_global[:, :,
                                               np.newaxis] - mean_celltype_sp_np_global[:, np.newaxis, :]
                pairwise_distances_sp_global = pairwise_distances_sp_global.transpose(
                    (1, 2, 0))  # results in np.array of dimensions (num_genes, num_genes, num_celltypes)

                abs_diff_sp = np.absolute(pairwise_distances_sp_global)

            abs_diff_sum_sp = np.sum(abs_diff_sp, axis=(0, 1))

            # calculate normalization factor
            norm_factor_sc = (1/(mean_celltype_sc.shape[1] ** 2)) * abs_diff_sum_sc
            norm_factor_sp = (1/(mean_celltype_sc.shape[1] ** 2)) * abs_diff_sum_sp

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

            ##### CALCULATE OVERALL SCORE MATRIX
            # First, sum over the differences between modalities in relative pairwise gene expression distances
            # The overall metric is then bounded at a maximum of 1, representing perfect similarity of relative gene expression between modalities.
            ## Furthermore, the metric is constructed such that, when its value is 0, this represents perfect dissimilarity of
            ## relative gene expression between modalities (such that each gene's expression value in each gene pair is swapped).
            overall_score = np.sum(np.absolute(norm_pairwise_distances_sp_local - norm_pairwise_distances_sc), axis=None)
            overall_metric = 1 - (overall_score / (2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=None)))
            overall_metric_matrix[y_bin, x_bin] = overall_metric

    # calculate the contribution of each grid field to the overall metric, if contribution is set to True
    if contribution:
        # calculate global metric for the entire spatial dataset (not just in the region range)
        overall_metric = relative_pairwise_gene_expression(adata_sp, adata_sc, key=obs_key, pipeline_output=True)

        nr_grid_fields = np.sum(~np.isnan(overall_metric_matrix))
        metric_contribution_matrix = np.zeros((bins[0], bins[1]))
        for y_bin in adata_sp_region_range.obs["y_bin"].unique():
            for x_bin in adata_sp_region_range.obs["x_bin"].unique():
                # calculate the difference in the local metric matrix explained by the grid field
                diff_explained_by_grid_field = (nr_grid_fields * (1 - overall_metric_matrix[y_bin, x_bin])
                                                / (nr_grid_fields - np.nansum(overall_metric_matrix)))

                # calculate the proportion of the global metric that is explained by the grid field
                diff_to_explain = 1 - overall_metric
                metric_contribution_matrix[y_bin, x_bin] = 1 - (diff_explained_by_grid_field * diff_to_explain)

        return metric_contribution_matrix

    return overall_metric_matrix


def _get_relative_expression_similarity_across_celltypes_grid(
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
    """Calculate the similarity of gene expression differences for all pairs of celltypes in the panel, between the
    two modalities, within each grid bin.
    ----------
    adata_sp : AnnData
        annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        annotated ``AnnData`` object with counts from scRNAseq data
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    obs_key : str, default "celltype"
        The column name in adata_sp.obs and adata_sc.obs for the cell type annotations.
    layer: str (default: 'lognorm')
        Layer of ``AnnData`` to use to compute the metric.
    cells_x_col : str, default "x"
        The column name in adata_sp.obs for the x-coordinates of cells.
    cells_y_col : str, default "y"
        The column name in adata_sp.obs for the y-coordinates of cells.
    normalization: str (default: 'global')
        The type of normalization to use for computing the metric. If set to 'global', the entire spatial dataset is used
        to normalize the pairwise gene expression differences for the spatial modality.
        If set to 'local', only the local grid field is used to normalize the pairwise gene expression differences.
        Can be either 'global' or 'local'.
    contribution: bool (default: True)
        Whether to calculate the contribution of each grid field to the overall metric or the metric itself.

    Returns
    -------
    overall_metric: np.ndarray
        Matrix containing the local overall similarity of relative pairwise expression between celltypes for all pairs of celltypes in the panel,
        b/t the scRNAseq and spatial data
    """
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

    # get bin ids
    adata_sp.obs = _get_bin_ids(adata_sp.obs, region_range, bins, cells_x_col, cells_y_col)

    # only consider cells within the specified region
    adata_sp_region_range = adata_sp[(adata_sp.obs["y_bin"] != -1) & (adata_sp.obs["x_bin"] != -1)]

    # create an empty matrix to store the computed metric for each grid field
    overall_metric_matrix = np.zeros((bins[0], bins[1]))

    for y_bin in adata_sp_region_range.obs["y_bin"].unique():
        for x_bin in adata_sp_region_range.obs["x_bin"].unique():
            # subset the spatial data to only include cells in the current grid field
            adata_sp_local = adata_sp_region_range[(adata_sp_region_range.obs["y_bin"] == y_bin) & (adata_sp_region_range.obs["x_bin"] == x_bin)]

            # find the unique celltypes in the grid field, that are both in the adata_sc and in the adata_sp
            unique_celltypes = adata_sc.obs.loc[adata_sc.obs[obs_key].isin(adata_sp_local.obs[obs_key]),obs_key].unique()

            # If there are not at least two shared cell types in the grid field set the local metric to NaN
            if len(unique_celltypes) <= 1:
                overall_metric_matrix[y_bin, x_bin] = np.nan
                continue

            #### CALCULATE EACH GENE'S MEAN EXPRESSION PER CELL TYPE
            # get the adata_sc cell x gene matrix as a pandas dataframe (w gene names as column names)
            exp_sc = pd.DataFrame(adata_sc.layers[layer], columns=adata_sc.var.index)

            # get the adata_sp cell x gene matrix as a pandas dataframe, once for the local grid field and once for the entire dataset
            exp_sp_local = pd.DataFrame(adata_sp_local.layers[layer], columns=adata_sp_local.var.index)

            # add "celltype" label column to exp_sc & exp_sp cell x gene matrices
            exp_sc[obs_key] = list(adata_sc.obs[obs_key])
            exp_sp_local[obs_key] = list(adata_sp_local.obs[obs_key])

            # delete all cells from the exp matrices if they aren't in the set of intersecting celltypes b/t sc & sp data
            exp_sc = exp_sc.loc[exp_sc[obs_key].isin(unique_celltypes), :]
            exp_sp_local = exp_sp_local.loc[exp_sp_local[obs_key].isin(unique_celltypes), :]

            # find the mean expression for each gene for each celltype in sc and sp data
            mean_celltype_sc = exp_sc.groupby(obs_key).mean()
            mean_celltype_sp_local = exp_sp_local.groupby(obs_key).mean()

            # sort genes in alphabetical order
            mean_celltype_sc = mean_celltype_sc.loc[:, mean_celltype_sc.columns.sort_values()]
            mean_celltype_sp_local = mean_celltype_sp_local.loc[:, mean_celltype_sp_local.columns.sort_values()]

            #### CALCULATE EXPRESSION DIFFERENCES BETWEEN ALL PAIRS OF GENES FOR EACH CELLTYPE
            mean_celltype_sc_np = mean_celltype_sc.T.to_numpy()
            pairwise_distances_sc = mean_celltype_sc_np[:, :, np.newaxis] - mean_celltype_sc_np[:, np.newaxis, :]
            pairwise_distances_sc = pairwise_distances_sc.transpose(
                (1, 2, 0))  # results in np.array of dimensions (num_celltypes, num_celltypes, num_genes)

            mean_celltype_sp_np_local = mean_celltype_sp_local.T.to_numpy()
            pairwise_distances_sp_local = mean_celltype_sp_np_local[:, :, np.newaxis] - mean_celltype_sp_np_local[:, np.newaxis, :]
            pairwise_distances_sp_local = pairwise_distances_sp_local.transpose(
                (1, 2, 0))  # results in np.array of dimensions (num_celltypes, num_celltypes, num_genes)

            #### NORMALIZE PAIRWISE EXPRESSION DIFFERENCES
            ## normalization is performed by dividing by the sum of the absolute values of all differences between pairs of cellttypes
            ## furthermore, to ensure that the values are comparable across datasets with different numbers of genes, we scale the result by a factor of
            ## num_genes^2
            # calculate sum of absolute distances
            abs_diff_sc = np.absolute(pairwise_distances_sc)
            abs_diff_sum_sc = np.sum(abs_diff_sc, axis=(0, 1))

            if normalization == "local":
              abs_diff_sp = np.absolute(pairwise_distances_sp_local)
            elif normalization == "global":
              # prepare entire spatial dataset (not just in the region range) to compute the global normalization factor
              exp_sp_global = pd.DataFrame(adata_sp.layers[layer], columns=adata_sp.var.index)
              exp_sp_global[obs_key] = list(adata_sp.obs[obs_key])
              exp_sp_global = exp_sp_global.loc[exp_sp_global[obs_key].isin(unique_celltypes), :]
              mean_celltype_sp_global = exp_sp_global.groupby(obs_key).mean()
              mean_celltype_sp_global = mean_celltype_sp_global.loc[:, mean_celltype_sp_global.columns.sort_values()]

              mean_celltype_sp_np_global = mean_celltype_sp_global.T.to_numpy()
              pairwise_distances_sp_global = mean_celltype_sp_np_global[:, :,
                                               np.newaxis] - mean_celltype_sp_np_global[:, np.newaxis, :]
              pairwise_distances_sp_global = pairwise_distances_sp_global.transpose(
                    (1, 2, 0))  # results in np.array of dimensions (num_celltypes, num_celltypes, num_genes)

              abs_diff_sp = np.absolute(pairwise_distances_sp_global)

            abs_diff_sum_sp = np.sum(abs_diff_sp, axis=(0, 1))

            # calculate normalization factor
            norm_factor_sc = (1/(mean_celltype_sc.T.shape[1]**2)) * abs_diff_sum_sc
            norm_factor_sp = (1/(mean_celltype_sc.T.shape[1]**2)) * abs_diff_sum_sp

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

            ##### CALCULATE OVERALL SCORE MATRIX
            # First, sum over the differences between modalities in relative pairwise gene expression distances
            # The overall metric is then bounded at a maximum of 1, representing perfect similarity of relative gene expression between modalities.
            ## Furthermore, the metric is constructed such that, when its value is 0, this represents perfect dissimilarity of
            ## relative gene expression between modalities.
            overall_score = np.sum(np.absolute(norm_pairwise_distances_sp_local - norm_pairwise_distances_sc), axis=None)
            overall_metric = 1 - (overall_score / (2 * np.sum(np.absolute(norm_pairwise_distances_sc), axis=None)))
            overall_metric_matrix[y_bin, x_bin] = overall_metric



    # calculate the contribution of each grid field to the overall metric, if contribution is set to True
    if contribution:
        # calculate global metric for the entire spatial dataset (not just in the region range)
        overall_metric = relative_pairwise_celltype_expression(adata_sp, adata_sc, key=obs_key, pipeline_output=True)

        nr_grid_fields = np.sum(~np.isnan(overall_metric_matrix))
        metric_contribution_matrix = np.zeros((bins[0], bins[1]))
        for y_bin in adata_sp_region_range.obs["y_bin"].unique():
            for x_bin in adata_sp_region_range.obs["x_bin"].unique():
                # calculate the difference in the local metric matrix explained by the grid field
                diff_explained_by_grid_field = (nr_grid_fields * (1 - overall_metric_matrix[y_bin, x_bin])
                                                / (nr_grid_fields - np.nansum(overall_metric_matrix)))

                # calculate the proportion of the global metric that is explained by the grid field
                diff_to_explain = 1 - overall_metric
                metric_contribution_matrix[y_bin, x_bin] = 1 - (diff_explained_by_grid_field * diff_to_explain)

        return metric_contribution_matrix

    return overall_metric_matrix

def _get_jensen_shannon_distance_grid(
    adata_sp: ad.AnnData,
    adata_sc: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    obs_key: str = "celltype",
    layer: str = 'lognorm',
    cells_x_col: str = "x",
    cells_y_col: str = "y",
    min_number_cells:int=10, # the minimal number of cells per celltype to be considered
    smooth_distributions:str='no_smoothing',
    window_size:int=7,
    sigma:int=2,
    correct_for_cell_number_dependent_decay:bool=False,
    filter_out_double_zero_distributions:bool=True,
    decay_csv_enclosing_folder='output'
):
### SET UP
    # set the .X layer of each of the adatas to be log-normalized counts
    adata_sp.X = adata_sp.layers[layer]
    adata_sc.X = adata_sc.layers[layer]

    # sparse matrix support
    for a in [adata_sc, adata_sp]:
        if issparse(a.X):
            a.layers[layer] = a.layers[layer].toarray()

    # get bin ids
    adata_sp.obs = _get_bin_ids(adata_sp.obs, region_range, bins, cells_x_col, cells_y_col)

    # only consider cells within the specified region
    adata_sp_region_range = adata_sp[(adata_sp.obs["y_bin"] != -1) & (adata_sp.obs["x_bin"] != -1)]

    # create an empty matrix to store the computed metric for each grid field
    overall_metric_matrix = np.zeros((bins[0], bins[1]))

    for y_bin in adata_sp_region_range.obs["y_bin"].unique():
        for x_bin in adata_sp_region_range.obs["x_bin"].unique():
            # subset the spatial data to only include cells in the current grid field
            adata_sp_local = adata_sp_region_range[(adata_sp_region_range.obs["y_bin"] == y_bin) & 
                                                   (adata_sp_region_range.obs["x_bin"] == x_bin)].copy()

            # pipeline output=True, # TODO add functions per gene and per celltype
            jsd = jensen_shannon_distance(adata_sc = adata_sc, 
                                    adata_sp = adata_sp_local,
                                    key=obs_key, 
                                    layer=layer, 
                                    min_number_cells=min_number_cells,
                                    smooth_distributions=smooth_distributions,
                                    window_size=window_size,
                                    sigma=sigma, 
                                    pipeline_output=True,
                                    correct_for_cell_number_dependent_decay=correct_for_cell_number_dependent_decay,
                                    filter_out_double_zero_distributions=filter_out_double_zero_distributions,
                                    decay_csv_enclosing_folder=decay_csv_enclosing_folder)
            overall_metric_matrix[y_bin, x_bin]  = jsd

    return overall_metric_matrix