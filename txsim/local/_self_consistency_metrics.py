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
#TODO: "ARI_spot_clusters", "annotation_similarity" 
