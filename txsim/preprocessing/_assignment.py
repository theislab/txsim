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

def run_pciseq(
    molecules: str,
    image: str,
    sc_data: str,
    cell_type_key: str,
    opts: Optional[dict]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Use pciSeq to assign molecules to cells

    Parameters
    ----------
    molecules : str
        File name of molecule data CSV with columns: 'gene name', 'x coord', 'y coord'
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

    #Safety feature for spatial genes that aren't included in scRNAseq
    not_included = set(spots['Gene']) - set(adata.var_names)
    if len(not_included) > 0:
        if opts is None:
            opts = {'exclude_genes':  list(not_included) }
        elif opts.get('exclude_genes') is None:
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
    assignments = geneData[ ["Gene", "x", "y", "neighbour"] ]
    assignments.columns = ["Gene", "x", "y", "cell"]

    #Save cell types
    type_vec = []
    for i in cellData['Cell_Num']:
        type_vec.append(cellData['ClassName'][i][np.argmax(cellData['Prob'][i])])

    #Change the cell names to match the segmentation
    cell_id = np.unique(seg)
    assignments['cell'] = cell_id[assignments['cell']]
    cell_types = pd.DataFrame(data=type_vec, index = cell_id[cell_id != 0])
    cell_types[cell_types == 'Zero'] = 'None'
    
    return assignments, cell_types