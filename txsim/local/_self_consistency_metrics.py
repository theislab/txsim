def calc_annotation_similarity(adata_sp1: ad, adata_sp2: ad, region_range=None):
    # Convert AnnData object to DataFrame
    adata1_spots = adata_sp1.uns['spots'].copy()
    adata2_spots = adata_sp2.uns['spots'].copy()

    # Filter spots based on region_range
    if region_range is not None:
        x_min, x_max = region_range[0]
        y_min, y_max = region_range[1]
        # Filter spots based on x and y coordinates within the specified range
        adata1_spots = adata1_spots[(adata1_spots['x'] >= x_min) & (adata1_spots['x'] <= x_max) &
                                     (adata1_spots['y'] >= y_min) & (adata1_spots['y'] <= y_max)]
        adata2_spots = adata2_spots[(adata2_spots['x'] >= x_min) & (adata2_spots['x'] <= x_max) &
                                     (adata2_spots['y'] >= y_min) & (adata2_spots['y'] <= y_max)]

    # Get indices of spots present in both adata1 and adata2
    common_indices = adata1_spots.index.intersection(adata2_spots.index)
    # Create temporary copy of cell_id -> celltype dictionary, and add "None" value
    temp_cell_dict1 = adata1_spots[['cell_id','celltype']].set_index('cell_id')['celltype'].copy()
    temp_cell_dict1.loc[np.nan] = 'None_1'
    temp_cell_dict2 = adata2_spots[['cell_id','celltype']].set_index('cell_id')['celltype'].copy()
    temp_cell_dict2.loc[np.nan] = 'None_2'

    # Use the common indices to get cell types from the dictionaries
    common_cell_ids = adata1_spots.loc[common_indices]['cell_id'].values

    celltypes1 = temp_cell_dict1.loc[common_cell_ids].values
    celltypes2 = temp_cell_dict2.loc[common_cell_ids].values


    # Calculate similarity
    similarity = np.mean(celltypes1 == celltypes2)
    return similarity

def _get_ARI_spot_clusters(
    adata_sp1: ad.AnnData,
    adata_sp2: ad.AnnData,
    region_range: Tuple[Tuple[float, float], Tuple[float, float]],
    bins: Tuple[int, int],
    spots_x_col: str = "x",
    spots_y_col: str = "y",
):
    ''' Calculate ...(?)
    Parameters
    ----------
    adata_sp1 : AnnData
        Annotated ``AnnData`` object with counts from spatial data and spots from clustering1
    adata_sp2 : AnnData
        Annotated ``AnnData`` object with counts from spatial data and spots from clustering2
    region_range : Tuple[Tuple[float, float], Tuple[float, float]]
        The range of the grid specified as ((y_min, y_max), (x_min, x_max)).
    bins : Tuple[int, int]
        The number of bins along the y and x axes, formatted as (ny, nx).
    uns_key : str
        Key where to find the data containing the spots information in both adata.uns
    ann_key : str
        Key where the annotation for the cell IDs are found in adata.uns[uns_key]
    spots_x_col : str, default "x"
        The column name in adata.uns[uns_key] for the x-coordinates of spots. Must be the same for both datasets.
    spots_y_col : str, default "y"
        The column name in adata.uns[uns_key] for the y-coordinates of spots. Must be the same for both datasets.
    pipeline_output : float, optional
        Boolean for whether to use the function in the pipeline or not
    Returns
    -------
    ARI_per_bins : float
        Adjusted rand Index for every bin
        Increase in proportion of positive cells assigned in spatial data to pairs of genes-celltyes with no/very low expression in scRNAseq
    '''
    # dataframes of uns restricted to region_range
    df1 = adata_sp1.uns['spots'][((adata_sp1.uns['spots'][spots_y_col] >= region_range[0][0])&(adata_sp1.uns['spots'][spots_y_col] <= region_range[0][1]))&
                                 ((adata_sp1.uns['spots'][spots_x_col] >= region_range[1][0])&(adata_sp1.uns['spots'][spots_x_col] <= region_range[1][1]))]
    df2 = adata_sp2.uns['spots'][((adata_sp1.uns['spots'][spots_y_col] >= region_range[0][0])&(adata_sp1.uns['spots'][spots_y_col] <= region_range[0][1]))&
                                 ((adata_sp1.uns['spots'][spots_x_col] >= region_range[1][0])&(adata_sp1.uns['spots'][spots_x_col] <= region_range[1][1]))]
    # assign bin to each xy coordinate and save in adata.uns
    adata_sp1.uns['spots']["bin_y"] = pd.cut(adata_sp1.uns['spots'][spots_y_col], bins=bins[0], labels=False)
    adata_sp1.uns['spots']["bin_x"] = pd.cut(adata_sp1.uns['spots'][spots_x_col], bins=bins[1], labels=False)
    adata_sp2.uns['spots']["bin_y"] = pd.cut(adata_sp2.uns['spots'][spots_y_col], bins=bins[0], labels=False)
    adata_sp2.uns['spots']["bin_x"] = pd.cut(adata_sp2.uns['spots'][spots_x_col], bins=bins[1], labels=False)
    df1 = adata_sp1.uns['spots']
    df2 = adata_sp2.uns['spots']
    assert (len(df1)==len(df2)), "AnnData Objects do not have the same number of genes in the specified region."
    # group by bins
    df1 = df1.groupby(["bin_y", "bin_x"])['celltype']
    df2 = df2.groupby(["bin_y", "bin_x"])['celltype']
    # new Dataframe, size as bins
    annotation_similarity_per_bins = np.zeros(bins, float)

    #TODO: "ARI_spot_clusters", "annotation_similarity" 
    # calculate ARI for every bin
    for (y, x), group1 in df1:
        group2 = df2.get_group((y, x))
        c1 = group1.values
        c2 = group2.values
        annotation_similarity_per_bins[y, x] = np.mean(c1 == c2)

    return annotation_similarity_per_bins
#TODO: "ARI_spot_clusters", "annotation_similarity" 
