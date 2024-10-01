import numpy as np
import anndata as ad
from anndata import AnnData
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


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

    
    assert ('celltype' in spots.columns); 'No celltypes available in spots object'
    
    #Sort spots table by cell id and get table intervals for each cell
    spots = spots.sort_values("cell",ascending=True)
    cells_sorted = spots["cell"].values
    start_indices = np.flatnonzero(np.concatenate(([True], cells_sorted[1:] != cells_sorted[:-1])))
    cell_to_start_idx = pd.Series(start_indices, index=cells_sorted[start_indices])
    cell_to_end_idx = pd.Series(cell_to_start_idx.iloc[1:].tolist()+[len(spots)], index=cell_to_start_idx.index)
    
    for cell_id in adata_st.obs['cell_id']:
        start_idx = cell_to_start_idx.loc[cell_id]
        end_idx = cell_to_end_idx.loc[cell_id]
        spots_of_cell = spots.iloc[start_idx:end_idx]
        
        cts = spots_of_cell['Gene'].value_counts()
        mode = spots_of_cell['celltype'].mode()
        adata_st.obs.loc[str(cell_id), 'ct_majority'] = mode.values[0]
        adata_st.obs.loc[str(cell_id), 'ct_majority_cert'] = (spots_of_cell['celltype'].value_counts()[mode].values[0] / sum(cts))

    return adata_st

        
def run_ssam(
    adata_st: AnnData,
    spots: pd.DataFrame,
    adata_sc: pd.DataFrame,
    um_p_px: float = 0.325,
    cell_id_col: str = 'cell',
    gene_col: str = 'Gene',
    sc_ct_key: str = 'celltype',
    no_ct_assigned_value: str | None = 'None_sp',
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
    cell_id_col : str
        Name of the cell id column in the spots DataFrame
    gene_col : str
        Name of the gene column in the spots DataFrame
    sc_ct_key : str
        Name of the cell type column in the sc AnnData
    no_ct_assigned_value : str
        Value to assign to cells that are not assigned to any cell type
        
    Returns
    -------
    AnnData
        Anndata object with cell type annotation in ``adata_st.obs['ct_ssam']`` and ``adata_st.obs['ct_ssam_cert']``, whereby the latter is the percentage of spots per cell with consistent cell type assignment.
    """
    
    import plankton.plankton as pl
    from plankton.utils import ssam
    
    assert "other" not in adata_sc.obs[sc_ct_key].values, "cell type'other' not allowed in sc data"
    
    sdata = pl.SpatialData( spots[gene_col],
                            spots.x*um_p_px,
                            spots.y*um_p_px )
    adata_sc=adata_sc[:,adata_st.var_names]
    if issparse(adata_sc.X):
        adata_sc = adata_sc.copy()
        adata_sc.X = adata_sc.X.toarray()
    exp=pd.DataFrame(adata_sc.X,columns=adata_sc.var_names)
    exp['celltype']=list(adata_sc.obs[sc_ct_key])
    signatures=exp.groupby('celltype').mean().transpose()
    # 'cheat-create' an anndata set:
    adata = AnnData(signatures.T)
    adata.X = np.array(np.nan_to_num(signatures.T))
    adata.obs['celltype'] = adata.obs.index
    # pl.ScanpyDataFrame(sdata,adata)
    sdata = pl.SpatialData(
                            spots[gene_col],
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
    # TODO: Unsure why ssam can duplicate spots. And why some are missing. In general check if it's better
    #       to use the ssam package instead of plankton, maybe there the issue is solved.
    spots['celltype'] = no_ct_assigned_value
    sdata = sdata[~sdata.index.duplicated(keep='first')] 
    spots.loc[sdata.index,"celltype"] = sdata['celltype']
    #spots['celltype'] = sdata['celltype']
    
    adata_st.obs['ct_ssam'] = no_ct_assigned_value
    
    for cell_id in adata_st.obs['cell_id']:
        spots_of_cell = spots[spots[cell_id_col] == cell_id ]
        if spots_of_cell["celltype"].isna().all():
            # mode fails if all values are NaN. TODO: Potentially we want to convert NaNs to no_ct_assigned_value 
            # before the loop: currently cells with mainly NaNs are still assigned to the cell type with the highest 
            # number of spots
            adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_ssam'] = no_ct_assigned_value
            continue
        cts = spots_of_cell[gene_col].value_counts()
        mode = spots_of_cell['celltype'].mode()
        adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_ssam'] = mode.values[0]
        adata_st.obs.loc[adata_st.obs['cell_id'] == cell_id, 'ct_ssam_cert'] = \
        (spots[spots[cell_id_col] == cell_id ]['celltype'].value_counts()[mode].values[0] / sum(cts))
    
    adata_st.obs.loc[adata_st.obs['ct_ssam'] == 'other', 'ct_ssam'] = no_ct_assigned_value
    
    return adata_st


def run_tangram(
        
    adata_st: AnnData,
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
        Anndata object of the sc transcriptomics data
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
        Anndata object with cell type annotation in ``adata_st.obs['celltype']`` and ``adata_st.obs['score']``, whereby the latter is the noramlized scores, i.e. probability of each spatial cell to belong to a specific cell type assignment.
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
    
    # Spatial prediction dataframe is saved in `obsm` `tangram_ct_pred` of the spatial AnnData
    tg.project_cell_annotations(
        adata_map = adata_map,
        adata_sp = adata_st, annotation=sc_ct_labels)
    
    # use original without extra layers generated from tangram
    
    df = adata_st.obsm['tangram_ct_pred'].copy()
    adata_st = adata_st_orig.copy()


    adata_st.obs['celltype'] = df.idxmax(axis=1)


    # Normalize by row before setting the score
    normalized_df = df.div(df.sum(axis=1), axis=0)
    max_values = normalized_df.max(axis=1)
    adata_st.obs['score'] = max_values
    adata_st.obsm['ct_tangram_scores'] = normalized_df

   

    return adata_st
