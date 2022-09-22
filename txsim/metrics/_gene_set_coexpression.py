import scanpy as scOA
import pandas as pd
import numpy as np
from anndata import AnnData
from . import coexpression_similarity

def gene_set_coexpression(
    spatial_data: AnnData,
    seq_data: AnnData,
    overlap_threshold: int = 5,
    min_cells: int = 20,
    pipeline_output: bool = True,
    **kwargs
) -> float:
    """Calculate coexpression score similarity on gene sets
    
    Parameters
    ----------
    spatial_data : AnnData
        annotated ``AnnData`` object with counts from spatial data
    seq_data : AnnData
        annotated ``AnnData`` object with counts scRNAseq data
    overlap_threshold : int, optional (Default: 5)
        minimal overlap of genes between a gene set and the common features bewteen
        the spatial and sequencing datasets for the gene set to be tested.
    min_cells : int, optional (Default: 20)
        The number of cells a gene must be expressed in at minimum to contribute to
        coexpression similarity score.
    pipeline_output : bool, optional (Default: True)
        flag whether to output a single value (True) or results broken down into
        gene sets

    Returns
    -------
    mean : float
        mean of coexpression score  across all tested gene sets. For further details
        on the coexpression score, see the `coexpression_score` function documentation

    Future extension
    ----------------
    Calculate the score per cell type and average over that
    """
    import omnipath as op

    print('Computing gene set coexpression...')

    # Get gene sets to test
    anns = op.requests.Annotations.get(resources='MSigDB')
    geneset_dict = _get_msigdb_collection(anns)

    # Filter gene sets that can be tested (gene expressed in at least X cells)
    vars_in_x_cells_st = sc.pp.filter_genes(spatial_data, min_cells = min_cells, inplace=False)[0]
    spatial_features = spatial_data.var_names[vars_in_x_cells_st].tolist()

    vars_in_x_cells_sc = sc.pp.filter_genes(seq_data, min_cells = min_cells, inplace=False)[0]
    seq_features = seq_data.var_names[vars_in_x_cells_sc].tolist()

    common_features = list(set(spatial_features).intersection(seq_features))

    sets_to_remove = list()

    for g in geneset_dict:
        overlap = len(set(geneset_dict[g]).intersection(common_features))
    
        if overlap < overlap_threshold:
            sets_to_remove.append(g)

    for g in sets_to_remove:
        del geneset_dict[g]
    

    # Get coexpression matrix
    mat_st, mat_sc, gene_ids = coexpression_similarity(
        spatial_data,
        seq_data,
        min_cells = min_cells,
        pipeline_output = False,
    )
    
    
    # Calculate score per gene set
    results = dict()

    for g in geneset_dict.keys():
        print(f'Testing gene set: {g}')

        val = _mtx_subset_and_diff(
            mat_st,
            mat_sc,
            gene_ids,
            geneset_dict[g],
        )

        results[g] = val
        
    if not pipeline_output:
        return results

    # Aggregate co-expression outputs
    return np.mean(list(results.values()))


def _mtx_subset_and_diff(
        mat_st: np.ndarray,
        mat_sc: np.ndarray,
        gene_ids: list,
        geneset: list,
) -> float:
    """
    Subset two matrices indexed by `gene_ids` to the genes in `geneset` and
    get the mean absolute difference of the upper triangle.
    """

    # Find gene overlap
    gene_ids = list(gene_ids)
    ids_in_set = [gene_ids.index(g) for g in geneset if g in gene_ids]

    # Subset matrices
    mat_st_sub = mat_st[ids_in_set,:][:,ids_in_set]
    mat_sc_sub = mat_sc[ids_in_set,:][:,ids_in_set]

    # Get upper triangle
    mat_st_sub[np.tril_indices(len(ids_in_set))] = np.nan
    mat_sc_sub[np.tril_indices(len(ids_in_set))] = np.nan

    mean_coexp_sc = np.nanmean(mat_sc_sub)
    mean_coexp_st = np.nanmean(mat_st_sub)
    print(f'Average geneset coexpression in spatial data: {mean_coexp_st}')
    print(f'Average geneset coexpression in seq data: {mean_coexp_sc}')

    # Absolute mean diff
    diff = mat_st_sub - mat_sc_sub
    res = np.nanmean(np.absolute(diff)) / 2

    return res



def _get_msigdb_collection(anns, collection='hallmark', verbose=False):
    """
    Parser for MSigDB gene sets from omnipath
    
    Returns:
        A str: list[str] dictionary of gene set names to HGNC gene names
    
    """
    from collections import defaultdict
    
    ids = anns[(anns.entity_type.isin(['protein'])) &
               (anns.label.isin(['collection'])) &
               (anns.value.isin([collection]))].record_id

    collection_anns = anns[(anns.entity_type.isin(['protein'])) & 
                           (anns.label.isin(['geneset'])) &
                           (anns.record_id.isin(ids))]
    
    if verbose:
        print(f'Number of genes: {len(set(collection_anns.genesymbol))}')
        print(f'Number of annotations: {len(collection_anns.genesymbol)}')
        print(f'Number of gene sets: {collection_anns}')

        
    geneset_dict = defaultdict(list)

    [geneset_dict[collection_anns['value'][i]].append(collection_anns['genesymbol'][i])
     for i in collection_anns.index];
    
    return geneset_dict
