def knn_mixing_per_ctype(
    adata_st: AnnData, 
    adata_sc: AnnData, 
    pipeline_output: bool = True,
    obs_key: str = "celltype",
    k: int = 45,
    ct_filter_factor: float = 5,
    by_celltype: bool = False,
) -> Union[float, Tuple[str, dict]]: 
    """Compute score for knn mixing of modalities
    
    Procedure: Concatenate sc and st data. Compute PCA on dataset union. Compute assortativity of each knn graph on 
    each shared cell type. Take the mean over shared cell types. The final score is `-assortativity + 1`. The score
    ranges from 0 (worst, i.e. no mixing) to 1 (maximal mixing) and could theoretically go to 2. We clip at 1 since
    above 1 is not expected (more mixing with other modality than with itself).
    
    Arguments
    ---------
    adata_st:
        Spatial data.
    adata_sc:
        Single cell data.
    pipeline_output:
        Whether to return only a summary score or additionally also cell type level scores.
    k:
        Number of neighbors for knn graphs.
    ct_filter_factor:
        Cell types with fewer cells than `ct_filter_factor` x `k` per modality are filtered out.
        
    Returns
    -------
    if pipeline_output
        knn mixing score
    else
        - knn mixing score
        - score per cell type
    
    """
    
    #adata_sc = adata_sc[:,adata_st.var_names] -> not even needed, concat does this
    
    # Concate sc and spatial
    adata_st.obs["modality"] = "spatial"
    adata_sc.obs["modality"] = "sc"
    adata = ad.concat([adata_st, adata_sc], join='inner')
    
    # Set counts to log norm data
    adata.X = adata.layers["lognorm"]      
    
    # Calculate PCA (general or per cell type)
    if not by_celltype:
        assert (adata.obsm is None) or ('X_pca' not in adata.obsm), "PCA already exists."
        sc.tl.pca(adata)    
        
    else:
        c_type = adata.obs['celltype'].unique()
        anndatas = []
        for val in c_type:
            dt = adata[adata.obs['celltype'] == val]
            assert (dt.obsm is None) or ('X_pca' not in dt.obsm), "PCA already exists."
            sc.tl.pca(dt)
            anndatas.append(dt)
        adata = ad.concat([anndata for anndata in anndatas], join='inner')
    
    # get cell type groups
    sc_cts = set(adata_sc.obs["celltype"].cat.categories)
    st_cts = set(adata_st.obs["celltype"].cat.categories)
    all_cts = list(sc_cts.union(st_cts))
    shared_cts = list(sc_cts.intersection(st_cts))
    #st_only_cts = list(st_cts - sc_cts)
    #sc_only_cts = list(sc_cts - st_cts)
    
    # Get adata per shared cell type
    scores = {ct:np.nan for ct in all_cts}
    for ct in shared_cts:
        enough_cells = (adata.obs.loc[adata.obs[obs_key]==ct,"modality"].value_counts() > (ct_filter_factor * k)).all()
        if enough_cells:
            a = adata[adata.obs[obs_key]==ct]
            sc.pp.neighbors(a,n_neighbors=k)
            G = nx.Graph(incoming_graph_data=a.obsp["connectivities"])
            nx.set_node_attributes(G, {i:a.obs["modality"].values[i] for i in range(G.number_of_nodes())}, "modality")
            scores[ct] = np.clip(-nx.attribute_assortativity_coefficient(G, "modality") + 1, 0, 1)
            
    mean_score = np.mean([v for _,v in scores.items() if v is not np.nan])
    
    if pipeline_output:
        return mean_score
    else:
        return mean_score, scores
