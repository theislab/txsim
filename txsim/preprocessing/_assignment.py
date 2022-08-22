import skimage.io
import numpy as np
import pandas as pd
from typing import Optional, Tuple

def basic_assign(
    molecules: str,
    image: str
) -> pd.DataFrame:
    """Assign molecules to cells

    Parameters
    ----------
    molecules : str
        File name of molecule data CSV with columns: 'gene name', 'x coord', 'y coord'
    image : str
        File name of TIF with segmented cells where each cell is indicated by a different value

    Returns
    -------
    pd.DataFrame
        DataFrame containing the cell assignments as column 'cell'
    """
    #Read and format molecules
    spots = pd.read_csv(molecules)
    spots.rename(columns = {spots.columns[0]:'Gene',
                     spots.columns[1]:'x',
                     spots.columns[2]:'y'
                     } , inplace = True)

    #Read image
    seg = skimage.io.imread(image)

    #Assign molecules based value in segmentation array
    spots['cell'] = seg[spots.y.to_numpy(dtype=np.int64), spots.x.to_numpy(dtype=np.int64)]
    return spots

def run_pciSeq(
    molecules: str,
    image: str,
    sc_data: str,
    cell_type_key: str,
    opts: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use pciSeq to assign molecules to cells

    Parameters
    ----------
    molecules : str
        File name of molecule data CSV with the first 3 columns: 'gene', 'x', 'y'
    image : str
        File name of TIF with segmented cells where each cell is indicated by a different value
    sc_data : str
        File name of h5ad AnnData object 
    cell_type_key : str
        Key for the cell type in `sc_data.obs` 
    opts : Optional[dict]
        Options for pciSeq

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Returns:
            - Molecule to cell assignments
            - Cell types
    """    

    import pciSeq
    from scipy.sparse import coo_matrix
    import scanpy as sc

    
    #Read and format molecules, single cell data, and labels
    spots = pd.read_csv(molecules)
    spots.rename(columns = {spots.columns[0]:'Gene',
                     spots.columns[1]:'x',
                     spots.columns[2]:'y'
                     } , inplace = True)

    adata = sc.read_h5ad(sc_data)
    scdata = adata.X
    scdata  = pd.DataFrame(scdata.transpose())
    scdata.columns = adata.obs[cell_type_key]
    scdata.index = adata.var_names

    seg = skimage.io.imread(image)
    coo = coo_matrix(seg)

    if opts is None:
        opts = {}

    #Safety feature for spatial genes that aren't included in scRNAseq
    not_included = set(spots['Gene']) - set(adata.var_names)
    if len(not_included) > 0:
        print("Warning: the following genes will be exluded since they were in the spatial data but not RNAseq")
        print(list(not_included))
        if opts.get('exclude_genes') is None:
            opts['exclude_genes'] =  list(not_included)
        else:
            opts['exclude_genes'].extend(list(not_included))
            opts['exclude_genes'] = list(set(opts['exclude_genes']))
    
    #Run through pciSeq
    pciSeq.attach_to_log()
    if(opts is not None):
        cellData, geneData = pciSeq.fit(spots, coo, scdata, opts)
    else:
        cellData, geneData = pciSeq.fit(spots, coo, scdata)   

    #Save in correct format
    assignments = geneData["neighbour"]

    #Save cell types
    type_vec = []
    for i in cellData['Cell_Num']:
        type_vec.append(cellData['ClassName'][i][np.argmax(cellData['Prob'][i])])

    #Change the cell names to match the segmentation

    cell_id = np.unique(seg)
    assignments = cell_id[assignments]
    if opts.get('exclude_genes') is None:
        spots['cell'] = assignments
    else:
        spots['cell'] = 0
        spots.loc[~spots['Gene'].isin(opts['exclude_genes']), 'cell'] = assignments
    
    cell_types = pd.DataFrame(data=type_vec, index = cell_id[cell_id != 0])
    cell_types[cell_types == 'Zero'] = 'None'
    
    return spots, cell_types

def run_clustermap(
    molecules: str,
    image: str,
    hyperparams: Optional[dict] = None
) -> pd.DataFrame:
    """Use ClusterMap to assign molecules to cells

    Parameters
    ----------
    molecules : str
        File name of molecule data CSV with the first 3 columns: 'gene', 'x', 'y'
    image : str
        File name of TIF with DAPI signal for cells
    hyperparams : Optional[dict], optional
        Hyperparameters for ClusterMap as a dictionary, by default None. Should have 3 keys, `model`
        `preprocess`, `segmentation`, each containing a dictionary of key-word arguments

    Returns
    -------
    pd.DataFrame
        DataFrame containing the cell assignments as column 'cell'
    """


    from ClusterMap.clustermap import ClusterMap
    import tifffile

    #Make sure parameter dictionaries are not `None`
    if hyperparams is None: hyperparams = {}
    if hyperparams.get('model') is None: hyperparams['model'] = {'xy_radius':15}
    if hyperparams['model'].get('xy_radius') is None: hyperparams['model']['xy_radius']=15
    if hyperparams.get('preprocess') is None: hyperparams['preprocess'] = {}
    if hyperparams.get('segmentation') is None: hyperparams['segmentation'] = {}

    #Read and format input data
    dapi = tifffile.imread(image)
    num_dims=len(dapi.shape)
    spots = pd.read_csv(molecules)
    spots.rename(columns = {spots.columns[0]:'gene_name',
                        spots.columns[1]:'spot_location_1',
                        spots.columns[2]:'spot_location_2'
                        } , inplace = True)

    #Use gene id numbers instead of names
    genes, ids = np.unique(spots['gene_name'], return_inverse=True)
    spots['gene'] = ids+1
    spots = spots.astype({'spot_location_1':int, 'spot_location_2':int})
    gene_list=np.unique(ids)+1
    genes = pd.DataFrame(genes)

    #Create Model
    model = ClusterMap(spots=spots,dapi=dapi, gene_list=gene_list, num_dims=num_dims,z_radius=0,
        **(hyperparams['model']))

    print('Radius: ' + str(model.xy_radius))

    #Preprocess
    model.preprocess(**(hyperparams['preprocess']))

    #Segment cells
    #TODO hyperparam min_spot_per_cell somehow
    model.min_spot_per_cell = 2
    model.segmentation(**(hyperparams['segmentation']))

    #The original spots file is modified by Clustermap to include assignments
    #Copy and return spots-to-cell assignment
    assignments = spots.copy()    
    assignments.rename(columns = {'gene_name':'Gene',
                        'spot_location_1':'x',
                        'spot_location_2':'y',
                        'clustermap':'cell',
                        } , inplace = True)
    assignments.cell = assignments.cell+1

    return assignments
