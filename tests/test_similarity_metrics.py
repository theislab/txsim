import pytest
import numpy as np
import pandas as pd
import txsim as tx
from scipy.sparse import issparse, csr_matrix


#TODO: Test for negative_marker_purity_reads_FP_based_optimum, or remove it from the package

def set_gene_to_zero_for_celltype(adata, celltype, gene, layer="raw"):
    gene_idx = adata.var_names.tolist().index(gene)
    if issparse(adata.layers[layer]):
        adata.layers[layer] = adata.layers[layer].toarray()
        adata.layers[layer][adata.obs["celltype"]==celltype,gene_idx] *= 0
        adata.layers[layer] = csr_matrix(adata.layers[layer])
    else:
        adata.layers[layer][adata.obs["celltype"]==celltype,gene_idx] *= 0
    return adata


@pytest.mark.parametrize("adata_spatial, adata_sc, neg_marker", [
    ("adata_sp", "adata_sc_high_sim", True),
    ("adata_sp", "adata_sc_high_sim", False),
    ("adata_sp_not_sparse", "adata_sc_high_sim_not_sparse", True),
    ("adata_sp_not_sparse", "adata_sc_high_sim_not_sparse", False),
])
def test_negative_marker_purity_cells(adata_spatial, adata_sc, neg_marker, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    adata_sc = request.getfixturevalue(adata_sc)

    if neg_marker:
        adata_sc = set_gene_to_zero_for_celltype(adata_sc, "B cells", "UBB", layer="raw")
    
    purity, purity_per_gene, purity_per_celltype = tx.metrics.negative_marker_purity_cells(
        adata_spatial, adata_sc, pipeline_output = False
    )
    
    assert isinstance(purity, (float,np.float32))
    assert isinstance(purity_per_gene, pd.Series)
    assert isinstance(purity_per_celltype, pd.Series)
    assert purity_per_gene.dtype in [float, np.float32]
    assert purity_per_celltype.dtype in [float, np.float32]
    # >= 0, <= 1 for all that are not np.nan
    assert np.isnan(purity) or ((purity >= 0) and (purity <= 1))
    assert (purity_per_gene.loc[~purity_per_gene.isnull()] >= 0).all() 
    assert (purity_per_gene.loc[~purity_per_gene.isnull()] <= 1).all()
    assert (purity_per_celltype.loc[~purity_per_celltype.isnull()] >= 0).all()
    assert (purity_per_celltype.loc[~purity_per_celltype.isnull()] <= 1).all()
    if neg_marker:
        assert not np.isnan(purity)
        assert not purity_per_gene.isnull().all()
        assert not purity_per_celltype.isnull().all()
    
    
@pytest.mark.parametrize("adata_spatial, adata_sc, neg_marker", [
    ("adata_sp", "adata_sc_high_sim", True),
    ("adata_sp", "adata_sc_high_sim", False),
    ("adata_sp_not_sparse", "adata_sc_high_sim_not_sparse", True),
    ("adata_sp_not_sparse", "adata_sc_high_sim_not_sparse", False),
])
def test_negative_marker_purity_reads(adata_spatial, adata_sc, neg_marker, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    adata_sc = request.getfixturevalue(adata_sc)
    
    if neg_marker:
        adata_sc = set_gene_to_zero_for_celltype(adata_sc, "B cells", "UBB", layer="raw")
    
    purity, purity_per_gene, purity_per_celltype = tx.metrics.negative_marker_purity_reads(
        adata_spatial, adata_sc, pipeline_output = False
    )
    
    assert isinstance(purity, (float,np.float32))
    assert isinstance(purity_per_gene, pd.Series)
    assert isinstance(purity_per_celltype, pd.Series)
    assert purity_per_gene.dtype in [float, np.float32]
    assert purity_per_celltype.dtype in [float, np.float32]
    # >= 0, <= 1 for all that are not np.nan
    assert np.isnan(purity) or ((purity >= 0) and (purity <= 1))
    assert (purity_per_gene.loc[~purity_per_gene.isnull()] >= 0).all() 
    assert (purity_per_gene.loc[~purity_per_gene.isnull()] <= 1).all()
    assert (purity_per_celltype.loc[~purity_per_celltype.isnull()] >= 0).all()
    assert (purity_per_celltype.loc[~purity_per_celltype.isnull()] <= 1).all()
    if neg_marker:
        assert not np.isnan(purity)
        assert not purity_per_gene.isnull().all()
        assert not purity_per_celltype.isnull().all()
        
        
@pytest.mark.parametrize("adata_spatial, adata_sc, correlation_measure, by_ct, thresh", [
    ("adata_sp", "adata_sc_high_sim", "pearson", True, 0),
    ("adata_sp", "adata_sc_high_sim", "pearson", False, 0),
    ("adata_sp", "adata_sc_high_sim", "spearman", True, 0),
    ("adata_sp", "adata_sc_high_sim", "spearman", False, 0),
    ("adata_sp_not_sparse", "adata_sc_high_sim_not_sparse", "pearson", True, 0),
    ("adata_sp_not_sparse", "adata_sc_high_sim_not_sparse", "spearman", True, 0),
    ("adata_sp", "adata_sc_high_sim", "pearson", True, 0.3),
    ("adata_sp", "adata_sc_high_sim", "spearman", True, 0.2),
])
def test_coexpression_similarity(adata_spatial, adata_sc, correlation_measure, by_ct, thresh, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    adata_sc = request.getfixturevalue(adata_sc)

    coexp_sim, coexp_sim_per_gene, coexp_sim_per_celltype = tx.metrics.coexpression_similarity(
        adata_spatial, 
        adata_sc, 
        thresh=thresh, 
        by_celltype = by_ct, 
        correlation_measure = correlation_measure, 
        pipeline_output=False,
    )

    assert isinstance(coexp_sim, (float,np.float32))
    assert isinstance(coexp_sim_per_gene, pd.Series)
    assert isinstance(coexp_sim_per_celltype, pd.Series)
    assert coexp_sim_per_gene.dtype in [float, np.float32]
    assert coexp_sim_per_celltype.dtype in [float, np.float32]
    # >= 0, <= 1 for all that are not np.nan
    assert np.isnan(coexp_sim) or ((coexp_sim >= 0) and (coexp_sim <= 1))
    assert (coexp_sim_per_gene.loc[~coexp_sim_per_gene.isnull()] >= 0).all() 
    assert (coexp_sim_per_gene.loc[~coexp_sim_per_gene.isnull()] <= 1).all()
    assert (coexp_sim_per_celltype.loc[~coexp_sim_per_celltype.isnull()] >= 0).all()
    assert (coexp_sim_per_celltype.loc[~coexp_sim_per_celltype.isnull()] <= 1).all()


@pytest.mark.parametrize("adata_spatial, adata_sc, k, ct_filter_factor", [
    ("adata_sp", "adata_sc_high_sim", 5, 1),
    ("adata_sp_not_sparse", "adata_sc_high_sim_not_sparse", 5, 1),
    ("adata_sp", "adata_sc_high_sim", 45, 5),
])
def test_knn_mixing_score(adata_spatial, adata_sc, k, ct_filter_factor, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    adata_sc = request.getfixturevalue(adata_sc)
    
    knn_mixing_score, knn_mixing_score_per_celltype = tx.metrics.knn_mixing(
        adata_spatial, adata_sc, k=k, ct_filter_factor=ct_filter_factor, pipeline_output=False
    )
    
    assert isinstance(knn_mixing_score, (float,np.float32))
    assert isinstance(knn_mixing_score_per_celltype, pd.Series)
    assert knn_mixing_score_per_celltype.dtype in [float, np.float32]
    # >= 0, <= 1 for all that are not np.nan
    assert np.isnan(knn_mixing_score) or ((knn_mixing_score >= 0) and (knn_mixing_score <= 1))
    assert (knn_mixing_score_per_celltype.loc[~knn_mixing_score_per_celltype.isnull()] >= 0).all()
    assert (knn_mixing_score_per_celltype.loc[~knn_mixing_score_per_celltype.isnull()] <= 1).all()
    
    
@pytest.mark.parametrize("adata_spatial, adata_sc", [
    ("adata_sp", "adata_sc_high_sim"),
    ("adata_sp_not_sparse", "adata_sc_high_sim_not_sparse"),
])
def test_relative_pairwise_celltype_expression(adata_spatial, adata_sc, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    adata_sc = request.getfixturevalue(adata_sc)
    
    rel_ct_expr, rel_ct_expr_per_gene, rel_ct_expr_per_celltype = tx.metrics.relative_pairwise_celltype_expression(
        adata_spatial, adata_sc, pipeline_output=False
    )
    
    assert isinstance(rel_ct_expr, (float,np.float32))
    assert isinstance(rel_ct_expr_per_gene, pd.Series)
    assert isinstance(rel_ct_expr_per_celltype, pd.Series)
    assert rel_ct_expr_per_gene.dtype in [float, np.float32]
    assert rel_ct_expr_per_celltype.dtype in [float, np.float32]
    # >= 0, <= 1 for all that are not np.nan #NOTE: theoretically could be negative, but not with the given dataset
    assert np.isnan(rel_ct_expr) or ((rel_ct_expr >= 0) and (rel_ct_expr <= 1))
    assert (rel_ct_expr_per_gene.loc[~rel_ct_expr_per_gene.isnull()] >= 0).all()
    assert (rel_ct_expr_per_gene.loc[~rel_ct_expr_per_gene.isnull()] <= 1).all()
    assert (rel_ct_expr_per_celltype.loc[~rel_ct_expr_per_celltype.isnull()] >= 0).all()
    assert (rel_ct_expr_per_celltype.loc[~rel_ct_expr_per_celltype.isnull()] <= 1).all()
    

@pytest.mark.parametrize("adata_spatial, adata_sc", [
    ("adata_sp", "adata_sc_high_sim"),
    ("adata_sp_not_sparse", "adata_sc_high_sim_not_sparse"),
])
def test_relative_pairwise_gene_expression(adata_spatial, adata_sc, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    adata_sc = request.getfixturevalue(adata_sc)
    
    rel_gene_expr, rel_gene_expr_per_gene, rel_gene_expr_per_celltype = tx.metrics.relative_pairwise_gene_expression(
        adata_spatial, adata_sc, pipeline_output=False
    )
    
    assert isinstance(rel_gene_expr, (float,np.float32))
    assert isinstance(rel_gene_expr_per_gene, pd.Series)
    assert isinstance(rel_gene_expr_per_celltype, pd.Series)
    assert rel_gene_expr_per_gene.dtype in [float, np.float32]
    assert rel_gene_expr_per_celltype.dtype in [float, np.float32]
    # >= 0, <= 1 for all that are not np.nan
    assert np.isnan(rel_gene_expr) or ((rel_gene_expr >= 0) and (rel_gene_expr <= 1))
    assert (rel_gene_expr_per_gene.loc[~rel_gene_expr_per_gene.isnull()] >= 0).all()
    assert (rel_gene_expr_per_gene.loc[~rel_gene_expr_per_gene.isnull()] <= 1).all()
    assert (rel_gene_expr_per_celltype.loc[~rel_gene_expr_per_celltype.isnull()] >= 0).all()
    assert (rel_gene_expr_per_celltype.loc[~rel_gene_expr_per_celltype.isnull()] <= 1).all()
    