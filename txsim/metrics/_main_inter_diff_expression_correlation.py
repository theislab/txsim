import numpy as np
import pandas as pd

"""
Main Intersection over Difference Expression Correlation (MIDEC)
---------------------------------------------------
source - annotated segmentation (e.g. expert annotation)
target - target segmentation (e.g. automatic segmentation)

Pearson correlation between the gene expression vector in the main 
intersection and the difference (source - target) is computed for 
each cell. The result of the metric is the mean of the correlation 
coefficients across all cells detected in source.
"""

# Main function

def main_inter_diff_expression_correlation(spots_df : pd.DataFrame,
                                source_segmentation : np.ndarray,
                                target_segmentation : np.ndarray,
                                min_number_of_spots_per_cell : int = 10,
                                upper_threshold : float = 0.75,
                                lower_threshold : float = 0.25,
                                source_segmentation_name : str = 'source',
                                target_segmentation_name : str = 'target'):
    """
    Calculate pearson correlation between the gene expression vectors 1 and 2:

    1. gene expression vector of the main overlap (intersection between one source 
    segmentation cell and one target segmentation cell). The intersection with the most
    spots is considered as the main overlap (intersection).
    2. gene expression vector from the rest of the source segmentation cell which is not 
    containing the intersection (or in other words, the difference)

    is computed for each cell.
    
    Only cells with a minimum number of min_number_of_spots_per_cell 
    are considered. The intercection area has an upper_threshold (<1) and a lower_threshold 
    for the ratio (amount_of_spots_in_intersection)/(amount_of_spots_in_source_cell).

    The result of the metric is the mean of the correlation coefficients across all cells in 
    the source segmentation.

    Input
    ----------
    spots_df : pandas.DataFrame
        DataFrame with columns 'x', 'y', 'gene'.
    source_segmentation : numpy.ndarray
        Segmentation array of the source segmentation.
    target_segmentation : numpy.ndarray
        Segmentation array of the target segmentation.
    min_number_of_spots_per_cell : int
        Minimum number of spots per cell. Default is 10.
    upper_threshold : float
        Upper threshold for share of spots in the intersection. Default is 0.75.
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

    # allocate spots to pixels
    spots_df = allocate_spots_to_pixels(spots_df)
    # add segmentations to spots_df
    spots_df = add_segmentation_to_spots_df(spots_df, source_segmentation, source_segmentation_name)
    spots_df = add_segmentation_to_spots_df(spots_df, target_segmentation, target_segmentation_name)
    # get the ids from the source segmentation
    source_segmentation_ids = np.unique(source_segmentation)
    # remove the 0 (background) from the ids
    source_segmentation_ids = source_segmentation_ids[source_segmentation_ids != 0]

    pearson_correlations = []

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
                                                                              target_column_name=target_segmentation_name)
        
        # if there is no overlap or no rest, continue with the next cell
        if main_overlap is None or rest is None:
            continue
        # calculate the pearson correlation coefficient between intersection and difference
        pearson_correlation = np.corrcoef(main_overlap, rest)[0, 1]

        # add the pearson correlation coefficient to the list
        pearson_correlations.append(pearson_correlation)

    # calculate the mean of the pearson correlation coefficients
    correlation = np.mean(pearson_correlations)
    print(len(pearson_correlations))

    return correlation

# Helper functions

def allocate_spots_to_pixels(spots_df):
    """
    Allocate spots to pixels.
    
    Input
    ----------
    spots_df : pandas.DataFrame
        DataFrame with columns 'x', 'y' and 'gene'.
    
    Output
    -------
    spots_df : pandas.DataFrame
        DataFrame with columns 'x', 'y', 'gene', 'pixel_x' and 'pixel_y'.
    """
    spots_df['pixel_x'] = spots_df['x'].astype(int)
    spots_df['pixel_y'] = spots_df['y'].astype(int)
    return spots_df


def add_segmentation_to_spots_df(spots_df, 
                                 segmentation,
                                 segmentation_name='segmentation'):
    """
    Add segmentation to spots_df.
    
    Input
    ----------
    spots_df : pandas.DataFrame
        DataFrame with columns 'x', 'y', 'gene', 'pixel_x' and 'pixel_y'.
    segmentation : numpy.ndarray
        Segmentation array
    
    Output
    -------
    spots_df : pandas.DataFrame
        DataFrame with columns 'x', 'y', 'gene', 'pixel_x', 'pixel_y' and 'segmentation'.
    """
    spots_df[segmentation_name] = segmentation[spots_df['pixel_y'], spots_df['pixel_x']]
    return spots_df


def calculate_main_overlap_and_rest_one_cell(cell_id_source : int,
                                    spots_df : pd.DataFrame,
                                    upper_threshold : float = 0.75,
                                    lower_threshold : float = 0.25,
                                    source_column_name : str = 'source',
                                    target_column_name : str = 'target'):
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
        DataFrame with columns 'x', 'y', 'gene', 'pixel_x', 'pixel_y', source_column_name and target_column_name.
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
    # if upper threshold is set to >=1, throw an error
    if upper_threshold >= 1:
        raise ValueError('Upper_threshold must be < 1')

    # get the spots of the cell in the source segmentation 
    # which are allocated to some cell in the target segmentation
    spots_source = spots_df[(spots_df[source_column_name] == cell_id_source) &
                        (spots_df[target_column_name] != 0)]

    # group by cell in the target segmentation
    spots_source_grouped = spots_source.groupby(target_column_name).count()

    # find out which target cell has the most spots in common with the source cell
    cell_id_target = spots_source_grouped['gene'].idxmax()

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
        all_spots = spots_df['gene'].unique()
        # generate the expression vectors for the main overlap and the rest
        expression_vector_main_overlap = np.zeros(len(all_spots))
        expression_vector_rest = np.zeros(len(all_spots))
        for spot in spots_main_overlap['gene']:
            expression_vector_main_overlap[np.where(all_spots == spot)] += 1
        for spot in spots_rest['gene']:
            expression_vector_rest[np.where(all_spots == spot)] += 1

        return expression_vector_main_overlap, expression_vector_rest
  


