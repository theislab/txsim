import numpy as np
import networkx as nx
import anndata as ad
import pandas as pd
from anndata import AnnData
import scanpy as sc
from typing import Union, Tuple

#TODO: fix Warnings in get_knn_mixing_score

def knn_mixing(
    adata_st: AnnData, 
    adata_sc: AnnData, 
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
    
    # Calculate PCA (Note: we could also think about pca per cell type...)
    assert (adata.obsm is None) or ('X_pca' not in adata.obsm), "PCA already exists."
    sc.tl.pca(adata)
    
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
    
def get_knn_mixing_score(adata_st: AnnData, adata_sc: AnnData, obs_key: str = "celltype", k: int = 45,ct_filter_factor: float = 2):
    """Get column in adata_sp.obs with knn mixing score.

    For this we concatenate the spatial and single cell datasets, compute the neighborsgraph for eligible celltypes, get the expected value for the
    modality ratio, compute the actual ratio for each cell and assign the knn mixing score.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    """
    adata_st.obs["modality"] = "spatial"
    adata_sc.obs["modality"] = "sc"
    adata = ad.concat([adata_st, adata_sc], join='inner')  
    adata.X = adata.X.astype(np.float32)
    adata.obs_names = adata.obs_names.astype(str) + "_" + adata.obs["modality"].astype(str)
    adata.obs = adata.obs.reset_index()

    # Set counts to log norm data
    adata.X = adata.layers["lognorm"]
    
    # Calculate PCA (Note: we could also think about pca per cell type...)
    assert (adata.obsm is None) or ('X_pca' not in adata.obsm), "PCA already exists."
    sc.tl.pca(adata)
    
    # get cell type groups
    sc_cts = set(adata_sc.obs["celltype"].cat.categories)
    st_cts = set(adata_st.obs["celltype"].cat.categories)
    shared_cts = list(sc_cts.intersection(st_cts))         

    # Get ratio per shared cell type
    df = pd.DataFrame(columns=["index","knn mixing score"])
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
                ct_df[i] = sum(1 for neighbor in G.neighbors(cell) if G.nodes[cell]["modality"]=="sc")  #number_modality_sc
                ct_df[i] = ct_df[i]/G.degree(cell)      #ratio: number modality sc / total cells
                i += 1 

            df_t = a.obs[a.obs['index'].str.endswith('spatial')]    	    #TODO: früher subsetten
            df_t["index"] = df_t["index"].apply(lambda s: s.rsplit('_',1)[0])
            ct_df = ct_df[a.obs['index'].str.endswith('spatial')]
            
            daf = pd.DataFrame({"index": df_t["index"], "knn mixing score":f(ct_df)})
            df = pd.concat([df,daf])

    adata_st.obs = pd.merge(adata_st.obs.reset_index(),df,left_on="index",right_on="index",how="left")