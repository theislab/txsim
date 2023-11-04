import numpy as np
from numpy import ndarray
import pandas as pd
from pandas import DataFrame
import anndata as ad
from anndata import AnnData
from typing import Optional, Tuple

def basic_assign(
    molecules: DataFrame,
    image: ndarray
) -> DataFrame:
    """Assign molecules to cells based on segments

    Parameters
    ----------
    molecules : DataFrame
        Molecule data DataFrame with first 3 columns: 'gene', 'x', 'y'
    image : ndarray
        DAPI image as matrix with segmented cells where each cell is indicated by a different value

    Returns
    -------
    DataFrame
        DataFrame containing the cell assignments as column 'cell'
    """

    #Read and format molecules
    spots = molecules.copy()
    spots.rename(columns = {spots.columns[0]:'Gene',
                     spots.columns[1]:'x',
                     spots.columns[2]:'y'
                     } , inplace = True)

    #Assign molecules based value in segmentation array
    spots['cell'] = image[spots.y.to_numpy(dtype=np.int64), spots.x.to_numpy(dtype=np.int64)]
    return spots

def run_pciSeq(
    molecules: DataFrame,
    image: ndarray,
    sc_data: AnnData,
    cell_type_key: str,
    opts: Optional[dict] = None
) -> Tuple[DataFrame, DataFrame]:
    """Use pciSeq to assign molecules to cells

    Parameters
    ----------
    molecules : DataFrame
        Molecule data as pandas DataFrame with the first 3 columns: 'gene', 'x', 'y'
    image : ndarray
        DAPI image as matrix with segmented cells where each cell is indicated by a different value
    sc_data : AnnData
        AnnData object with scRNA-seq data
    cell_type_key : str
        Key for the cell type in `sc_data.obs` 
    opts : Optional[dict], optional
        Options for pciSeq, by default None

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        Returns two DataFrames:
            - Molecule to cell assignments
            - Cell types
    """

    import pciSeq
    from scipy.sparse import coo_matrix, issparse
    import scanpy as sc

    
    #Read and format molecules, single cell data, and labels
    spots = molecules.copy()
    spots.rename(columns = {spots.columns[0]:'Gene',
                     spots.columns[1]:'x',
                     spots.columns[2]:'y'
                     } , inplace = True)

    adata = sc_data.copy()
    scdata = adata.X if not issparse(adata.X) else adata.X.toarray()
    scdata  = pd.DataFrame(scdata.transpose())
    print(scdata.columns)
    print(adata.obs[cell_type_key])
    print(type(adata.X))
    print(type(scdata))
    print(scdata)
    scdata.columns = adata.obs[cell_type_key]
    scdata.index = adata.var_names

    coo = coo_matrix(image)

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
    prob_vec = []
    for i in cellData['Cell_Num']:
        type_vec.append(cellData['ClassName'][i][np.argmax(cellData['Prob'][i])])
        prob_vec.append(np.max(cellData['Prob'][i]))

    #Change the cell names to match the segmentation

    cell_id = np.unique(image)
    assignments = cell_id[assignments]
    if opts.get('exclude_genes') is None:
        spots['cell'] = assignments
    else:
        spots['cell'] = 0
        spots.loc[~spots['Gene'].isin(opts['exclude_genes']), 'cell'] = assignments
    
    cell_types = pd.DataFrame(data={'type':type_vec, 'prob':prob_vec}, index = cell_id[cell_id != 0])
    cell_types = cell_types.replace({'Zero':'None'})
    
    return spots, cell_types
