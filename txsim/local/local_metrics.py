import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
import networkx as nx
import anndata as ad
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.sparse import isspmatrix 
import itertools
from typing import List

#TODO: fix Warnings in get_knn_mixing_score
#TODO: fix Warnings in get_cells_location

#helper function
def check_crop_exists(x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray):
    """Check if crop coordinates exist.
    
    For this, we check if either (x_min, x_max, y_min, y_max) or an image was provided. If not, we raise a ValueError. 

    Parameters
    ----------
    x_min: int, x_max: int, y_min: int, y_max: int
        crop coordinates
    image: np.ndarray

    Returns
    -------
    if no ValueError was raised, returns range  
    """
    if (x_min is None or x_max is None or y_min is None or y_max is None) and image is None:
        raise ValueError("please provide an image or crop")         
        
    if x_min is not None and x_max is not None and y_min is not None and y_max is not None:
        range = [[x_min,x_max],[y_min,y_max]]
    
    else:
        range = [[0,image.shape[0]],[0,image.shape[1]]]
        
    return range 


#helper function
def matrix_colorbar_plot(matrix: np.ndarray, title: str, x_min: int, x_max: int, y_min: int, y_max: int, vmin, vmax, 
                         smooth: float = 0, show_ticks: bool = False):
    """Display (smoothed and cropped) matrix as an image with a colorbar and title.
    
    Parameters
    ----------
    matrix: np.ndarray
        data
    title: str
    x_min: int, x_max: int, y_min: int, y_max: int
        crop coordinates
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """
    
    matrix = gaussian_filter(matrix,sigma=smooth)
    fig = plt.figure()
    ax = fig.add_subplot(title = title)
    plot = plt.imshow(matrix, vmin=vmin, vmax=vmax, interpolation='nearest', extent=[x_min, x_max, y_max, y_min])
    fig.colorbar(plot)
    
    if not show_ticks:
        ax.tick_params(which='both', bottom=False, left=False, labelbottom = False, labelleft = False)



def get_negative_marker_dict(adata_sp: AnnData, adata_sc: AnnData, key: str='celltype'):
    """Add dictionary of negative marker genes for different celltypes.
    
    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    key : str
        Celltype key in adata_sp.obs and adata_sc.obs

    Returns
    -------
    neg_marker_dict : dict[string: list]
        Dictionary with celltypes as keys and corresponding negative marker genes as values
    """

    # Set threshold parameters
    min_number_cells=10 # minimum number of cells belonging to a cluster to consider it in the analysis
    max_ratio_cells=0.005 # maximum ratio of cells expressing a marker to call it a negative marker gene-ct pair

    #check that genes in spatial data is subset of genes in single cell data
    adata_sp = adata_sp[:,adata_sp.var_names.isin(adata_sc.var_names)]

    # Subset adata_sc to genes of spatial data
    adata_sc = adata_sc[:,adata_sp.var_names]           
    
    adata_sc.layers["raw"] = adata_sc.X

    # TMP fix for sparse matrices, ideally we don't convert, and instead have calculations for sparse/non-sparse
    if isspmatrix(adata_sc.layers["raw"]):                                  
        adata_sc.layers["raw"] = adata_sc.layers["raw"].toarray()

    # Get cell types that we find in both modalities
    shared_celltypes = adata_sc.obs.loc[adata_sc.obs[key].isin(adata_sp.obs[key]),key].unique()
    
    # Filter cell types by minimum number of cells
    celltype_count_sc = adata_sc.obs[key].value_counts().loc[shared_celltypes]
    celltype_count_sp = adata_sc.obs[key].value_counts().loc[shared_celltypes]   
    ct_filter = (celltype_count_sc >= min_number_cells) & (celltype_count_sp >= min_number_cells)
    celltypes = celltype_count_sc.loc[ct_filter].index.tolist()             
    
    
    # Filter cells to eligible cell types
    adata_sc = adata_sc[adata_sc.obs[key].isin(celltypes)]
    genes = adata_sc.var_names
    
    # Get ratio of positive cells per cell type
    pos_exp_sc = pd.DataFrame(adata_sc.layers["raw"] > 0,columns=adata_sp.var_names)     
    pos_exp_sc['celltype'] = list(adata_sc.obs[key])
    ratio_celltype_sc = pos_exp_sc.groupby('celltype').mean()
    
    # Get gene-cell type pairs with negative marker expression
    neg_marker_mask = np.array(ratio_celltype_sc < max_ratio_cells)
    

    neg_marker_mask = pd.DataFrame(neg_marker_mask, columns = genes, index = ratio_celltype_sc.index)   
    neg_marker_dict = {celltype: list(row.index[row]) for celltype, row in neg_marker_mask.iterrows()}

    return neg_marker_dict



def get_spot_assignment_col(adata_sp: AnnData, adata_sc: AnnData):

    """Add column spot_assignment to adata_sp.uns["spots"].

       spot_assignment is "spot in wrong celltype", if celltype of cell is in the keys of the neg_marker_dict and gene is a negative marker for the celltype. 
       spot_assignment is "unassigned" if the corresponding cell entry is NaN.
       spot_assignment is "no negative marker" if gene is  not a negative marker for any celltype in neg_marker_dict.
       spot_assignment is "spot in correct celltype" otherwise.
      
       Parameters
       ----------
       adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
       adata_sc : AnnData
        Annotated ``AnnData`` object with counts scRNAseq data
    """ 
    #check that genes in spatial data is subset of genes in single cell data
    adata_sp = adata_sp[:,adata_sp.var_names.isin(adata_sc.var_names)]

    df = adata_sp.uns["spots"]  
   
    neg_marker_dict = get_negative_marker_dict(adata_sp, adata_sc)

    celltypes = list(neg_marker_dict.keys())   

    df["spot_assignment"] = "spot in correct celltype"

    all_neg_markers = set(itertools.chain(*neg_marker_dict.values()))
    #gene is not a negative marker for any celltype in neg_marker_dict and celltype is not in celltypes 
    df.loc[(~df["Gene"].isin(all_neg_markers))& (~df["celltype"].isin(celltypes)), "spot_assignment"] = "no negative marker"  

    for ct in celltypes:
      df.loc[(df["Gene"].isin(neg_marker_dict[ct])) & (df["celltype"] == ct), 'spot_assignment'] = "spot in wrong celltype" 

    df.loc[df["cell"]!=df["cell"],"spot_assignment"] = "unassigned"   #check for NaNs



def get_cells_location(adata_sp: AnnData, adata_sc: AnnData):
    """Add x,y coordinate columns of cells to adata_sp.obs.

        Parameters
        ----------
        adata_sp : AnnData
            Annotated ``AnnData`` object with counts from spatial data
        adata_sc : AnnData
            Annotated ``AnnData`` object with counts scRNAseq data
    """

    get_spot_assignment_col(adata_sp,adata_sc)
    spots = adata_sp.uns["spots"]
    df_cells = spots.loc[spots["spot_assignment"]!="unassigned"]      
    df_cells = df_cells.groupby(["cell"])[["x","y"]].mean()
    df_cells = df_cells.reset_index().rename(columns={'cell':'cell_id'})
    adata_sp.obs = pd.merge(df_cells,adata_sp.obs,left_on="cell_id",right_on="cell_id",how="inner")



def get_knn_mixing_score(adata_st: AnnData, adata_sc: AnnData, obs_key: str = "celltype", k: int = 45,ct_filter_factor: float = 2):
    """Get column in adata_sp.obs with knn mixing score.

    For this we concatenate the spatial and single cell datasets, compute the neighborsgraph for eligible celltypes, get the expected value for the
    modality ratio, compute the actual ratio for each cell and assign a the knn mixing score.

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