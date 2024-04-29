import numpy as np
import pandas as pd
import anndata as ad
import math
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from ._util import check_crop_exists
from ._util import get_bin_edges


"""
Main Intersection over Difference Expression Correlation (MIDEC)
---------------------------------------------------
source - annotated segmentation (e.g. expert annotation)
target - target segmentation (e.g. automatic segmentation)

Correlation between the gene expression vector in the main 
intersection and the difference (source - target) is computed for 
each cell. The result of the metric is the mean of the correlation 
coefficients across all cells detected in source.
The functionaliry of the metric also includes the substitution of
the default correlation (Pearson) with Spearman's rank correlation
and other metrics (MSE, absolute error).
"""

# Main functions

def global_MIDEC(adata_sp_1 : ad.AnnData,
                                adata_sp_2 : ad.AnnData,
                                min_number_of_spots_per_cell : int = 10,
                                min_number_of_cells : int = 10,
                                upper_threshold : float = 0.75,
                                lower_threshold : float = 0.25,
                                source_segmentation_name : str = 'source',
                                target_segmentation_name : str = 'target',
                                spots_x_col: str = 'x',
                                spots_y_col: str = 'y',
                                cell_col_name: str = 'cell',
                                gene_col_name: str = 'Gene',
                                which_metric = 'pearson'):
    """
    Main Intersection over Difference Expression Correlation (MIDEC).

    Calculate pearson correlation between the gene expression vectors 1 and 2:

    1. gene expression vector of the main overlap (intersection between one source 
    segmentation cell and one target segmentation cell). The intersection with the most
    spots is considered as the main overlap (intersection).
    2. gene expression vector from the rest of the source segmentation cell which is not 
    containing the intersection (or in other words, the difference)

    is computed for each cell.

    The result of the metric is the mean of the correlation coefficients across all cells in 
    the source segmentation.

    Input
    ----------
    spots_df : pandas.DataFrame
        DataFrame with columns 'x', 'y', 'Gene'.
    source_segmentation : numpy.ndarray
        Segmentation array of the source segmentation.
    target_segmentation : numpy.ndarray
        Segmentation array of the target segmentation.
    min_number_of_spots_per_cell : int
        Minimum number of spots per cell. Default is 10.
    upper_threshold : float
        Upper threshold for share of spots in the intersection (<1.0). Default is 0.75.
    lower_threshold : float
        Lower threshold for share of spots in the intersection. Default is 0.25.
    source_segmentation_name : str
        Name of the column with the source segmentation. Default is 'source'.
    target_segmentation_name : str
        Name of the column with the target segmentation. Default is 'target'
    spots_x_col : str
        Name of the column with the x coordinates of the spots. Default is 'x'.
    spots_y_col : str
        Name of the column with the y coordinates of the spots. Default is 'y'.
    which_metric : str
        Name of the metric to be used. Default is 'pearson'. Other options are 
        'spearman', 'MAE' (mean absolute error), and 'MSE' (mean square error). 
    
    Output
    -------
    metric : float
        Metric value, correllation for 'pearson' and 'spearman', or error for
        'MAE' and 'MSE'.
    """
    # if the thresholds make no sense - throw an error
    if upper_threshold >= 1:
        raise ValueError('upper_threshold must be < 1')
    elif upper_threshold <= lower_threshold:
        raise ValueError('upper_threshold must be > lower_threshold')
    elif lower_threshold <= 0:
        raise ValueError('lower_threshold must be > 0')
    
    # assert that one of the metrics is chosen
    assert which_metric in ['pearson', 'spearman', 'MAE', 'MSE'], 'Invalid metric'

    # calculate the metric once when 1 is the source and 2 is the target, and once the other way around
    result = []
    for source, target in [(adata_sp_1, adata_sp_2), (adata_sp_2, adata_sp_1)]:
        # get the spots from both AnnData objects
        source_spots = source.uns['spots'].copy()
        target_spots = target.uns['spots'].copy()

        # rename the cell column to source_segmentation_name and make it integer type
        spots_df = source_spots.rename(columns={cell_col_name: source_segmentation_name}).copy()
        # add the cell column from target segmentation to the spots_df
        spots_df[target_segmentation_name] = target_spots[cell_col_name].copy()
        # fill the NaN values with 0 and make the column integer type
        spots_df[source_segmentation_name] = spots_df[source_segmentation_name].fillna(0).astype(int)
        spots_df[target_segmentation_name] = spots_df[target_segmentation_name].fillna(0).astype(int)

        metric, cell_number = main_inter_diff_expression_correlation(spots_df=spots_df,
                                    min_number_of_spots_per_cell=min_number_of_spots_per_cell,
                                    min_number_of_cells=min_number_of_cells,
                                    upper_threshold=upper_threshold,
                                    lower_threshold=lower_threshold,
                                    source_segmentation_name=source_segmentation_name,
                                    target_segmentation_name=target_segmentation_name,
                                    spots_x_col=spots_x_col,
                                    spots_y_col=spots_y_col,
                                    gene_col_name=gene_col_name,
                                    which_metric=which_metric)
        result.append((metric, cell_number))
    return result

def local_MIDEC(adata_sp_source : ad.AnnData,
                adata_sp_target : ad.AnnData,
                image : np.ndarray,
                x_min : int,
                x_max : int,
                y_min : int,
                y_max : int,
                min_number_of_spots_per_cell = 10,
                min_number_of_cells = 10,
                upper_threshold = 0.75,
                lower_threshold = 0.25,
                source_segmentation_name='source',
                target_segmentation_name = 'target',
                spots_x_col='x',
                spots_y_col='y',
                cell_col_name='cell',
                gene_col_name='Gene',
                bins = 10,
                which_metric = 'pearson'):
    """
    Local Main Intersection over Difference Expression Correlation (MIDEC).

    For a given segmentation, calculate MIDEC for each bin in a local crop.

    Calculate pearson correlation between the gene expression vectors 1 and 2:

    1. gene expression vector of the main overlap (intersection between one source 
    segmentation cell and one target segmentation cell). The intersection with the most
    spots is considered as the main overlap (intersection).
    2. gene expression vector from the rest of the source segmentation cell which is not 
    containing the intersection (or in other words, the difference)

    is computed for each cell.

    Input
    ----------
    adata_sp_source : anndata.AnnData 
        AnnData object containing the spots with the source segmentation in adata.uns['spots']
    adata_sp_target : anndata.AnnData
        AnnData object containing the spots with the target segmentation in adata.uns['spots']
    source_segmentation : numpy.ndarray
        Segmentation array of the source segmentation.
    x_min : int
        Minimum x coordinate of the segment of interest.
    x_max : int
        Maximum x coordinate of the segment of interest.
    y_min : int
        Minimum y coordinate of the segment of interest.
    y_max : int
        Maximum y coordinate of the segment of interest.
    min_number_of_spots_per_cell : int
        Minimum number of spots per cell. Default is 10.
    min_number_of_cells : int
        Minimum number of cells for the metric to be calculated. Default is 10.
    upper_threshold : float
        Upper threshold for share of spots in the intersection (<1.0). Default is 0.75.
    lower_threshold : float
        Lower threshold for share of spots in the intersection. Default is 0.25.
    source_segmentation_name : str
        Name of the column with the source segmentation. Default is 'source'.
    target_segmentation_name : str
        Name of the column with the target segmentation. Default is 'target'.
    bins : int
        Number of bins for the crop for one axes. Default is 10.
    
    Output
    -------
    gridfield_metric : numpy.ndarray
        Local correlation for each segment of the gridfield in a given crop.
    """
    # if the thresholds make no sense - throw an error
    if upper_threshold >= 1:
        raise ValueError('upper_threshold must be < 1')
    elif upper_threshold <= lower_threshold:
        raise ValueError('upper_threshold must be > lower_threshold')
    elif lower_threshold <= 0:
        raise ValueError('lower_threshold must be > 0')

    # 1. get one spot table with both segmentations
    source_spots = adata_sp_source.uns['spots']
    target_spots = adata_sp_target.uns['spots']
    # rename the column 'cell' to source_segmentation_name and make it integer type
    spots_df = source_spots.rename(columns={cell_col_name: source_segmentation_name}).copy()
    # add the column 'cell' from target segmentation to the spots_df
    spots_df[target_segmentation_name] = target_spots[cell_col_name].copy()
    # fill the NaN values with 0 and make the column integer type
    spots_df[source_segmentation_name] = spots_df[source_segmentation_name].fillna(0).astype(int)
    spots_df[target_segmentation_name] = spots_df[target_segmentation_name].fillna(0).astype(int)

    # 2. check crop, get bins
    check_crop_exists(image, x_min, x_max, y_min, y_max)
    bins_x, bins_y = get_bin_edges([[x_min, x_max], [y_min, y_max]], bins)
    # make the values in bins_x and bins_y integers
    bins_x = bins_x.astype(int)
    bins_y = bins_y.astype(int)
    # define the size of the gridfield_metric (np.array) 
    n_bins_x = len(bins_x) - 1
    n_bins_y = len(bins_y) - 1

    # 3. initialize the np.array that will hold the metric for each segment of the gridfield
    gridfield_metric = np.zeros((n_bins_x, n_bins_y))

    # 4. generate the gridfield_metric
    j = 0
    for x_start, x_end in zip(bins_x[:-1], bins_x[1:]):
        i = 0
        for y_start, y_end in zip(bins_y[:-1], bins_y[1:]):
            
            # crop the spots_df
            spots_crop = spots_df[(spots_df[spots_x_col].astype(int) >= x_start) &
                                     (spots_df[spots_x_col].astype(int) < x_end) &
                                     (spots_df[spots_y_col].astype(int) >= y_start) &
                                     (spots_df[spots_y_col].astype(int) < y_end)].copy()
            
            # adjust the coordinates of the cropped dataframe
            spots_crop[spots_x_col] = spots_crop[spots_x_col] - x_start
            spots_crop[spots_y_col] = spots_crop[spots_y_col] - y_start
            spots_crop = spots_crop.reset_index(drop=True)
            
            # calculate the metric for the crop
            gridfield_metric[i, j] = main_inter_diff_expression_correlation(spots_df=spots_crop,
                                                                            min_number_of_spots_per_cell=min_number_of_spots_per_cell,
                                                                            min_number_of_cells=min_number_of_cells,
                                                                            upper_threshold=upper_threshold,
                                                                            lower_threshold=lower_threshold,
                                                                            source_segmentation_name=source_segmentation_name,
                                                                            target_segmentation_name=target_segmentation_name,
                                                                            spots_x_col=spots_x_col,
                                                                            spots_y_col=spots_y_col,
                                                                            gene_col_name=gene_col_name,
                                                                            which_metric=which_metric)
            i += 1
        j += 1 
    return gridfield_metric


# Helper functions
def main_inter_diff_expression_correlation(spots_df : pd.DataFrame,
                                min_number_of_spots_per_cell : int,
                                min_number_of_cells : int,
                                upper_threshold : float,
                                lower_threshold : float,
                                source_segmentation_name : str,
                                target_segmentation_name : str,
                                spots_x_col: str,
                                spots_y_col: str,
                                gene_col_name: str,
                                which_metric : str):
    """
    Main Intersection over Difference Expression Correlation (MIDEC).

    Calculate pearson correlation between the gene expression vectors 1 and 2:

    1. gene expression vector of the main overlap (intersection between one source 
    segmentation cell and one target segmentation cell). The intersection with the most
    spots is considered as the main overlap (intersection).
    2. gene expression vector from the rest of the source segmentation cell which is not 
    containing the intersection (or in other words, the difference)

    is computed for each cell.

    The result of the metric is the mean of the correlation coefficients across all cells in 
    the source segmentation.

    Input
    ----------
    spots_df : pandas.DataFrame
        DataFrame with columns 'x', 'y', 'Gene', source_segmentation_name and target_segmentation_name.
    source_segmentation : numpy.ndarray
        Segmentation array of the source segmentation.
    target_segmentation : numpy.ndarray
        Segmentation array of the target segmentation.
    min_number_of_spots_per_cell : int
        Minimum number of spots per cell. Default is 10.
    upper_threshold : float
        Upper threshold for share of spots in the intersection (<1.0). Default is 0.75.
    lower_threshold : float
        Lower threshold for share of spots in the intersection. Default is 0.25.
    source_segmentation_name : str
        Name of the column with the source segmentation. Default is 'source'.
    target_segmentation_name : str
        Name of the column with the target segmentation. Default is 'target'.
    
    Output
    -------
    correlation : float
        Pearson correlation coefficient between the intersection gene expression
        vector and the difference gene expression vector.
    """
    spots_df = allocate_spots_to_pixels(spots_df, spots_x_col=spots_x_col, spots_y_col=spots_y_col)

    # get the ids from the source segmentation
    source_segmentation_ids = np.unique(spots_df[source_segmentation_name])
    # remove the 0 (background) from the ids
    source_segmentation_ids = source_segmentation_ids[source_segmentation_ids != 0]

    values = []

    for cell_id_source in source_segmentation_ids:
        # check if the cell has enough spots
        spots_source = spots_df[(spots_df[source_segmentation_name] == cell_id_source) &
                        (spots_df[target_segmentation_name] != 0)]
        if spots_source.shape[0] <= min_number_of_spots_per_cell:
            continue

        # calculate the intersection and the difference gene expression vectors for one cell
        main_overlap, rest = calculate_main_overlap_and_rest_one_cell(cell_id_source, 
                                                                        spots_df, 
                                                                        upper_threshold=upper_threshold,
                                                                        lower_threshold=lower_threshold,
                                                                        source_column_name=source_segmentation_name,
                                                                        target_column_name=target_segmentation_name,
                                                                        gene_col_name=gene_col_name)
        
        # if there is no overlap or no rest, continue with the next cell
        if main_overlap is None or rest is None:
            continue
        if which_metric == 'pearson':
            # calculate the pearson correlation coefficient between intersection and difference
            value = pearsonr(main_overlap, rest)[0]
        elif which_metric == 'spearman':
            value = spearmanr(main_overlap, rest)[0]
        elif which_metric == 'MAE':
            main_overlap_normalized = main_overlap/np.linalg.norm(main_overlap)
            rest_normalized = rest/np.linalg.norm(rest)
            value = np.abs(np.subtract(main_overlap_normalized, rest_normalized)).mean()
        elif which_metric == 'MSE':
            main_overlap_normalized = main_overlap/np.linalg.norm(main_overlap)
            rest_normalized = rest/np.linalg.norm(rest)
            value = np.square(np.subtract(main_overlap_normalized, rest_normalized)).mean()
        # ensure that the value is not NaN, because for example pearsonr can return None
        if value is not math.nan:
            values.append(value)

    # calculate the mean for the overall value
    if len(values) < min_number_of_cells:
        metric = None
    else:
        metric = np.mean(values)

    return metric, len(values) # return the metric and the number of cells that were considered

def allocate_spots_to_pixels(spots_df, spots_x_col, spots_y_col):
    """
    Allocate spots to pixels.
    
    Input
    ----------
    spots_df : pandas.DataFrame
        DataFrame with columns spots_x_col and spots_y_col
    
    Output
    -------
    spots_df_copy : pandas.DataFrame
        DataFrame with columns 'pixel_x' and 'pixel_y', spots_x_col and spots_y_col
    """
    spots_df['pixel_x'] = spots_df[spots_x_col].astype(int)
    spots_df['pixel_y'] = spots_df[spots_y_col].astype(int)
    return spots_df

def add_segmentation_to_spots_df(spots_df : pd.DataFrame, 
                                 segmentation : np.ndarray,
                                 segmentation_name='segmentation'):
    """
    Add segmentation to spots_df.
    
    Input
    ----------
    spots_df : pandas.DataFrame
        DataFrame with columns 'pixel_x','pixel_y'
    segmentation : numpy.ndarray
        Segmentation array
    
    Output
    -------
    spots_df : pandas.DataFrame
        DataFrame with columns 'pixel_x','pixel_y', segmentation_name
    """
    spots_df[segmentation_name] = segmentation[spots_df['pixel_y'], spots_df['pixel_x']]
    return spots_df


def calculate_main_overlap_and_rest_one_cell(cell_id_source : int,
                                    spots_df : pd.DataFrame,
                                    upper_threshold : float,
                                    lower_threshold : float,
                                    source_column_name : str,
                                    target_column_name : str,
                                    gene_col_name : str):
    """
    Calculate the gene expression vectors for the main overlap (maximum overlap)
    with a cell in the target segmentation. Only cells with a minimum number of
    min_number_of_spots_per_cell are considered. The main overlap is defined as
    the amount of spots of a cell in the source segmentation that overlap with
    the maximum number of spots of a cell in the target segmentation.The overlap
    can have a maximum of upper_threshold and a minimum of lower_threshold of the 
    total number of spots of the cell in the source segmentation.

    Input
    ----------
    cell_id_source : int
        Cell id of the cell in the source segmentation.
    spots_df : pandas.DataFrame
        DataFrame with columns gene_col_name, source_column_name, target_column_name.
    upper_threshold : float
        Upper threshold for share of spots in the main overlap. Default is 0.75.
    lower_threshold : float
        Lower threshold for share of spots in the main overlap. Default is 0.25.
    source_column_name : str
        Name of the column with the source segmentation. Default is 'source'.
    target_column_name : str
        Name of the column with the target segmentation. Default is 'target'.
    
    Output
    -------
    expression_vector_main_overlap : numpy.ndarray
        Expression vector for the main overlap.
    expression_vector_rest : numpy.ndarray
        Expression vector for the rest of the spots in the source cell.
    
    """
    # get the spots of the cell in the source segmentation 
    # which are allocated to some cell in the target segmentation
    spots_source = spots_df[(spots_df[source_column_name] == cell_id_source) &
                        (spots_df[target_column_name] != 0)]

    # group by cell in the target segmentation
    spots_source_grouped = spots_source.groupby(target_column_name).count()

    # find out which target cell has the most spots in common with the source cell
    cell_id_target = spots_source_grouped[gene_col_name].idxmax()

    # main overlap spots
    spots_main_overlap = spots_source[spots_source[target_column_name] == cell_id_target]
    # rest spots
    spots_rest = spots_df[(spots_df[target_column_name] != cell_id_target) &
                                  (spots_df[source_column_name] == cell_id_source)]
    
    # check for the threshold
    main_overlap_part = spots_main_overlap.shape[0]/(spots_main_overlap.shape[0] + spots_rest.shape[0])

    if main_overlap_part < lower_threshold or main_overlap_part > upper_threshold:
        return None, None
    else:
        # generate the list of all spots in the spots_df
        all_spots = spots_df[gene_col_name].unique()
        # generate the expression vectors for the main overlap and the rest
        expression_vector_main_overlap = np.zeros(len(all_spots))
        expression_vector_rest = np.zeros(len(all_spots))
        for spot in spots_main_overlap[gene_col_name]:
            expression_vector_main_overlap[np.where(all_spots == spot)] += 1
        for spot in spots_rest[gene_col_name]:
            expression_vector_rest[np.where(all_spots == spot)] += 1

        return expression_vector_main_overlap, expression_vector_rest
  


