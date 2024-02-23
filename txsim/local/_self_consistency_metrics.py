import sklearn.metrics
#TODO: "ARI_spot_clusters", "annotation_similarity" 
def _get_ARI_spot_clusters(
        adata_sp1: ad.AnnData,
        adata_sp2: ad.AnnData,
        region_range: Tuple[Tuple[float, float], Tuple[float, float]],
        bins: Tuple[int, int],
        uns_key: str = "spots",
        ann_key: str = "cell_id",
        spots_x_col: str = "x",
        spots_y_col: str = "y",
        pipeline_output: bool=True,
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
        df1 = adata_sp1.uns[uns_key][((adata_sp1.uns[uns_key][spots_y_col] >= region_range[0][0])&(adata_sp1.uns[uns_key][spots_y_col] <= region_range[0][1]))&
                                     ((adata_sp1.uns[uns_key][spots_x_col] >= region_range[1][0])&(adata_sp1.uns[uns_key][spots_x_col] <= region_range[1][1]))]
        df2 = adata_sp2.uns[uns_key][((adata_sp1.uns[uns_key][spots_y_col] >= region_range[0][0])&(adata_sp1.uns[uns_key][spots_y_col] <= region_range[0][1]))&
                                     ((adata_sp1.uns[uns_key][spots_x_col] >= region_range[1][0])&(adata_sp1.uns[uns_key][spots_x_col] <= region_range[1][1]))]
        # assign bin to each xy coordinate and save in adata.uns
        adata_sp1.uns[uns_key]["bin_y"] = pd.cut(adata_sp1.uns[uns_key][spots_y_col], bins=bins[0], labels=False)
        adata_sp1.uns[uns_key]["bin_x"] = pd.cut(adata_sp1.uns[uns_key][spots_x_col], bins=bins[1], labels=False)
        adata_sp2.uns[uns_key]["bin_y"] = pd.cut(adata_sp2.uns[uns_key][spots_y_col], bins=bins[0], labels=False)
        adata_sp2.uns[uns_key]["bin_x"] = pd.cut(adata_sp2.uns[uns_key][spots_x_col], bins=bins[1], labels=False)
        df1 = adata_sp1.uns[uns_key]
        df2 = adata_sp2.uns[uns_key]
        assert (len(df1)==len(df2)), "AnnData Objects do not have the same number of genes in the specified region."
        # group by bins
        df1 = df1.groupby(["bin_y", "bin_x"])[ann_key]
        df2 = df2.groupby(["bin_y", "bin_x"])[ann_key]
        # new Dataframe, size as bins
        ARI_per_bins = np.zeros(bins, float)

        # calculate ARI for every bin
        for (y, x), group1 in df1:
            group2 = df2.get_group((y, x))
            c1 = group1.values
            c2 = group2.values
            ARI_per_bins[y, x] = sklearn.metrics.adjusted_rand_score(c1, c2)

        return ARI_per_bins
