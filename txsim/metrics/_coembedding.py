import numpy as np
import networkx as nx
import anndata as ad
import scanpy as sc
from typing import Union, Tuple

def knn_mixing(
    adata_sp: ad.AnnData, 
    adata_sc: ad.AnnData, 
    pipeline_output: bool = True,
    obs_key: str = "celltype",
    k: int = 45,
    ct_filter_factor: float = 5,
) -> Union[float, Tuple[str, dict]]: 
    """Compute score for knn mixing of modalities
    
    Procedure: Concatenate sc and st data. Compute PCA on dataset union. Compute assortativity of each knn graph on 
    each shared cell type. Take the mean over shared cell types. The final score is `-assortativity + 1`. The score
    ranges from 0 (worst, i.e. no mixing) to 1 (maximal mixing) and could theoretically go to 2. We clip at 1 since
    above 1 is not expected (more mixing with other modality than with itself).
    
    Arguments
    ---------
    adata_sp:
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

    # Concatenate sc and spatial
    adata_sp.obs["modality"] = "spatial"
    adata_sc.obs["modality"] = "sc"
    adata = ad.concat([adata_sp, adata_sc], join='inner')
    
    # Set counts to log norm data
    adata.X = adata.layers["lognorm"]
    
    # Calculate PCA (Note: we could also think about pca per cell type...)
    assert (adata.obsm is None) or ('X_pca' not in adata.obsm), "PCA already exists."
    sc.tl.pca(adata)
    
    # get cell type groups
    sc_cts = set(adata_sc.obs[obs_key].cat.categories)
    st_cts = set(adata_sp.obs[obs_key].cat.categories)
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
    
#TODO: fix NumbaDeprecationWarning

def knn_mixing_per_cell_score(
    adata_sp: ad.AnnData, 
    adata_sc: ad.AnnData, 
    obs_key: str = "celltype", 
    key_added: str = "knn_mixing_score",
    k: int = 45,
    ct_filter_factor: float = 2
) -> None:
    """Compute knn mixing scores per cell

    For this we concatenate the spatial and single cell datasets, compute the neighbors graph for eligible celltypes, 
    get the expected value for the modality ratio, compute the scores as the actual measured ratio for each cell.

    Arguments
    ----------
    adata_sp:
        Spatial data.
    adata_sc:
        Single cell data.
    obs_key:
        adata.obs key for cell type annotations.
    key_added:
        adata.obs key where knn mixing scores are saved.
    k:
        Number of neighbors for knn graphs.
    ct_filter_factor:
        Cell types with fewer cells than `ct_filter_factor` x `k` per modality are filtered out.
    
    Returns
    -------
    nothing - just added scores to `adata.obs[key_added]`
    
    
    """

    adata_sp.obs["modality"] = "spatial"
    sp_obs_names = adata_sp.obs_names.tolist()
    adata_sp.obs_names = [f"sp_{i}" for i in adata_sp.obs_names] # just to make sure that the names are unique
    adata_sc.obs["modality"] = "sc"
    adata = ad.concat([adata_sp, adata_sc])
 
    adata_sp.obs[key_added] = np.zeros(adata_sp.n_obs)  

    # Set counts to log norm data
    adata.X = adata.layers["lognorm"]
    
    # Calculate PCA (Note: we could also think about pca per cell type...)
    assert (adata.obsm is None) or ('X_pca' not in adata.obsm), "PCA already exists."
    sc.tl.pca(adata)
    
    # get cell type groups
    sc_cts = set(adata_sc.obs[obs_key].cat.categories)
    st_cts = set(adata_sp.obs[obs_key].cat.categories)
    shared_cts = list(sc_cts.intersection(st_cts))         

    # Get ratio per shared cell type
    for ct in shared_cts:
        enough_cells = (adata.obs.loc[adata.obs[obs_key]==ct,"modality"].value_counts() > (ct_filter_factor * k)).all()    
        if enough_cells:
            a = adata[adata.obs[obs_key]==ct]
            exp_val = (a.obs.loc[a.obs["modality"]=="sc"].shape[0])/a.obs.shape[0]  
            sc.pp.neighbors(a,n_neighbors=k)
            G = nx.Graph(incoming_graph_data=a.obsp["connectivities"])
            nx.set_node_attributes(G, {i:a.obs["modality"].values[i] for i in range(G.number_of_nodes())}, "modality")   

            ct_df = np.zeros(a.obs.shape[0])
            f = np.vectorize(lambda x: x/exp_val if x>=0 and x<=exp_val else x/(exp_val-1)+1/(1-exp_val))
            i = 0
            for cell in G.nodes():
                ct_df[i] = sum(1 for neighbor in G.neighbors(cell) if G.nodes[neighbor]["modality"]=="sc")
                ct_df[i] = ct_df[i]/G.degree(cell)     
                i += 1 
            
            a.obs[key_added] = f(ct_df)
            adata_sp.obs.loc[adata_sp.obs[obs_key] == ct, key_added] = a.obs.loc[a.obs["modality"]=="spatial", key_added]

    adata_sp.obs_names = sp_obs_names