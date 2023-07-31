import numpy as np
import anndata as ad
from anndata import AnnData
import pandas as pd
import scanpy as sc


def run_majority_voting(
    adata_st: AnnData, 
    spots: pd.DataFrame
) -> AnnData:

    """Add cell type annotation by majority voting of available annotations listed in ``spots['celltype']``

    Parameters
    ----------
    adata_st : AnnData
        File name of CSV containing genes and cell assignments
    spots : pd.DataFrame
        File name containing cell type for each cell, by default None
        
    Returns
    -------
    AnnData
        Anndata object with cell type annotation in ``adata_st.obs['ct_majority']`` and certainty as ``adata_st.obs['ct_majority_cert']`` 
    """

    for cell_id in adata_st.obs['cell_id']:
        cts = spots[spots['cell'] == cell_id ]['Gene'].value_counts()
        
        if 'celltype' in spots.columns:
            mode = spots[spots['cell'] == cell_id ]['celltype'].mode()
            adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_majority'] = mode.values[0]
            adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_majority_cert'] = (spots[spots['cell'] == cell_id ]['celltype'].value_counts()[mode].values[0] / sum(cts))
            
        else:
            print('No celltypes available in spots object')

    return adata_st


        
def run_ssam(
    
    adata_st: AnnData,
    spots: pd.DataFrame,
    adata_sc: pd.DataFrame,
    um_p_px: float = 0.325,
    
) -> AnnData:
    """Add cell type annotation by ssam.

    Parameters
    ----------
    coordinates : pd.DataFrame
        DataFrame containing all spots with gene labels and coordinates
    adata_st : AnnData
        AnnData object of the spatial transcriptomics data
    adata_sc : str
        Path to the sc transcriptomics AnnData
    um_p_px : float
        Conversion factor micrometer per pixel. Adjust to data set
        
    Returns
    -------
    AnnData
        Anndata object with cell type annotation in ``adata_st.obs['ct_ssam']`` and ``adata_st.obs['ct_ssam_cert']``, whereby the latter is the percentage of spots per cell with consistent cell type assignment.
    """
    
    import plankton.plankton as pl
    from plankton.utils import ssam
    
    x =  spots.x.values 
    y =  spots.y.values 
    g =  spots.Gene.values
    sdata = pl.SpatialData( spots.Gene,
                            spots.x*um_p_px,
                            spots.y*um_p_px )
    adata_sc=adata_sc[:,adata_st.var_names]
    exp=pd.DataFrame(adata_sc.X,columns=adata_sc.var_names)
    exp['celltype']=list(adata_sc.obs['celltype'])
    signatures=exp.groupby('celltype').mean().transpose()
    # 'cheat-create' an anndata set:
    adata = AnnData(signatures.T)
    adata.X = np.array(np.nan_to_num(signatures.T))
    adata.obs['celltype'] = adata.obs.index
    # pl.ScanpyDataFrame(sdata,adata)
    sdata = pl.SpatialData(
                            spots.Gene,
                            spots.x*um_p_px,
                            spots.y*um_p_px,
    #                         pixel_maps={'DAPI':bg_map},
                            scanpy=adata
                            )
    sdata = sdata.clean()
    signatures = sdata.scanpy.generate_signatures()
    signatures[signatures.isna()]=0
    #signatures.to_csv('signatures.csv')
    ctmap = ssam(sdata,signatures=signatures,kernel_bandwidth=4,patch_length=1500,threshold_cor=0.2,threshold_exp=0.1)
    # sample the map's values at all molecule locations:
    values_at_xy = ctmap.get_value(sdata.x,sdata.y)
    # assign tissue label to sampled values:
    celltype_labels = adata.obs['celltype'].values[values_at_xy]
    celltype_labels[ctmap.get_value(sdata.x,sdata.y)==-1]='other'
    # add to sdata frame:
    sdata['celltype']= celltype_labels
    sdata['celltype'] = sdata.celltype.astype('category')
    
    # Assign based on majority vote
    spots['celltype'] = sdata['celltype']
    
    for cell_id in adata_st.obs['cell_id']:
        cts = spots[spots['cell'] == cell_id ]['Gene'].value_counts()
        mode = spots[spots['cell'] == cell_id ]['celltype'].mode()
        adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_ssam'] = mode.values[0]
        adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_ssam_cert'] = \
        (spots[spots['cell'] == cell_id ]['celltype'].value_counts()[mode].values[0] / sum(cts))
    
    return adata_st

def annotate_celltypes(
    adata: AnnData,
    adata_sc: AnnData,
    ct_method: str = 'majority',
    ct_threshold: float = 0.7,
    prior_celltypes : pd.DataFrame = None,
    hyperparams: dict = {}
) -> AnnData:
    #all_ct_methods = False
    #TODO potentially fix how threshold is measured
    #Add celltype according to ct_method and check if all methods should be implemented
    if hyperparams.get('threshold') is not None: ct_threshold = hyperparams.get('threshold')
    ran_ct_method = False
    if (ct_method is None): ct_method = 'majority'
    if (ct_method == 'majority'):
        adata = run_majority_voting(adata, adata.uns['spots'])
        ran_ct_method = True
    elif (ct_method == 'ssam'):
        adata = run_ssam(adata, adata.uns['spots'], adata_sc = adata_sc)
        ran_ct_method = True
    elif (ct_method == 'pciSeqCT'):
        #TODO check if this actually works
        ct_method = 'pciSeq'
        adata.obs['ct_pciSeq'] = pd.Categorical(prior_celltypes['type'][adata.obs['cell_id']])
        adata.obs['ct_pciSeq_cert'] = prior_celltypes['prob'][adata.obs['cell_id']]
        ran_ct_method = True
    else:
        raise Exception(f'{ct_method} is not a valid cell type method')
    # ToDo (second prio)
    # elif ct_method == 'manual_markers':
    #     adata = run_manual_markers(adata, spots)
    # elif ct_method == 'scrna_markers':
    #     adata = run_scrna_markers(adata, spots, rna_adata)
    if not ran_ct_method: print('No valid cell type annotation method')
    
    # Take over primary ct annotation method to adata.obs['celltype'] and apply certainty threshold
    # Add methods, if they provide certainty measure
    if ct_method in ['majority', 'ssam']: 
        ct_list = adata.obs['ct_'+str(ct_method)].copy()
        ct_list[adata.obs['ct_'+str(ct_method)+'_cert'] < ct_threshold] = "Unknown" #TODO different hyperparams probably
        adata.obs['celltype'] = ct_list
    else:
        adata.obs['celltype'] = adata.obs['ct_'+str(ct_method)]

    return adata

def run_tangram(
        
    adata_st: AnnData,
    #TODO figure out adata_sc is the path or the adata file
    adata_sc: AnnData,
    sc_ct_labels: str = 'celltype',
    device: str = 'cpu',
    mode: str = 'cells',
    num_epochs: int = 1000,
    
) -> AnnData:
    """Run the Tangram algorithm.

    Parameters
    ----------
    adata_st : AnnData
        AnnData object of the spatial transcriptomics data
    adata_sc : str
        Path to the sc transcriptomics AnnDat
    sc_ct_labels : str
        Labels of the cell_type layer in the adata_sc
    device : str or torch.device 
        Optional. Default is 'cpu'.
    mode : str
        Optional. Tangram mapping mode. 'cells', 'clusters', 'constrained'. Default is 'cells'
    num_epochs : int 
        Optional. Number of epochs. Default is 1000
        
    Returns
    -------
    AnnData
        Anndata object with cell type annotation in ``adata_st.obs['ct_tangram']`` and ``adata_st.obs['ct_tangram_scores']``, whereby the latter is the noramlized scores, i.e. probability of each spatial cell to belong to a specific cell type assignment.
    """
    #import scanpy as sc
    import tangram as tg

    #TODO: check the layers in adata_sc
    # use log1p noramlized values  
    #adata_sc.X = adata_sc.layers['lognorm']
    
    adata_st_orig = adata_st.copy()

    # use all the genes from adata_st as markers for tangram
    markers = adata_st.var_names.tolist()
    
    # Removes genes that all entries are zero. Finds the intersection between adata_sc, adata_st and given marker gene list, save the intersected markers in two adatas
    # Calculates density priors and save it with adata_st
    tg.pp_adatas(
        adata_sc=adata_sc, 
        adata_sp=adata_st, 
        genes=markers,
        )
    
    # Map single cell data (`adata_sc`) on spatial data (`adata_st`).
    # density_prior (str, ndarray or None): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). 
    # use 'uniform' if the spatial voxels are at single cell resolution (e.g. MERFISH). 'rna_count_based', assumes that cell density is proportional to the number of RNA molecules.

    adata_map = tg.map_cells_to_space(
        adata_sc=adata_sc,
        adata_sp=adata_st,
        device=device,
        mode=mode,
        num_epochs=num_epochs,
        density_prior='uniform')
    
    #TODO: normalize the scores so they represent a probability to be a cell type 
    # Spatial prediction dataframe is saved in `obsm` `tangram_ct_pred` of the spatial AnnData
    tg.project_cell_annotations(
        adata_map = adata_map,
        adata_sp = adata_st, annotation=sc_ct_labels)
    
    # use original without extra layers generated from tangram
    adata_st_orig.obsm['ct_tangram_scores'] = adata_st.obsm['tangram_ct_pred']
    adata_st = adata_st_orig.copy()

    # Assign cell type predictions
    df = adata_st.obsm['tangram_ct_pred'].copy()
    df['MaxValue_Column'] = df.idxmax(axis=1)
    df = df['MaxValue_Column']
    adata_st.obs['ct_tangram'] = df

    return adata_st

