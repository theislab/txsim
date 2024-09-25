import pytest
import pandas as pd
import txsim as tx

#TODO: Add tests that check if sparse and none sparse adata give the same results

@pytest.mark.parametrize("adata_spatial", ["adata_sp", "adata_sp_not_sparse"])
def test_cell_density(adata_spatial, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    density, density_per_celltype = tx.quality_metrics.cell_density(adata_spatial, pipeline_output=False)
    assert isinstance(density, float)
    assert isinstance(density_per_celltype, pd.Series)
    assert density >= 0
    assert density_per_celltype.sum() == density
    assert (density_per_celltype >= 0).all()
    

@pytest.mark.parametrize("adata_spatial", ["adata_sp", "adata_sp_not_sparse"])
def test_proportion_of_assigned_reads(adata_spatial, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    reads_assigned, reads_assigned_per_gene, reads_assigned_per_ct = tx.quality_metrics.proportion_of_assigned_reads(
        adata_spatial, pipeline_output=False
    )
    
    assert isinstance(reads_assigned, float)
    assert isinstance(reads_assigned_per_gene, pd.Series)
    assert isinstance(reads_assigned_per_ct, pd.Series)
    # >= 0 for all
    assert reads_assigned >= 0
    assert (reads_assigned_per_gene >= 0).all()
    assert (reads_assigned_per_ct >= 0).all()
    # <= 1 for all
    assert reads_assigned <= 1
    assert (reads_assigned_per_gene <= 1).all()
    assert (reads_assigned_per_ct <= 1).all()
    # all genes and cell types in indices
    assert reads_assigned_per_gene.index.isin(adata_spatial.var_names).all()
    assert reads_assigned_per_ct.index.isin(adata_spatial.obs["celltype"].unique()).all()
    # sum of cell type proportions equals total proportion
    assert reads_assigned_per_ct.sum() == pytest.approx(reads_assigned)
    

@pytest.mark.parametrize("adata_spatial, statistic", [
    ("adata_sp", "mean"),
    ("adata_sp", "median"),
    ("adata_sp_not_sparse", "mean"),
    ("adata_sp_not_sparse", "median")
])
def test_reads_per_cell(adata_spatial, statistic, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    reads_per_cell, reads_per_cell_per_gene, reads_per_cell_per_ct = tx.quality_metrics.reads_per_cell(
        adata_spatial, statistic=statistic, pipeline_output=False
    )
    
    assert isinstance(reads_per_cell, float)
    assert isinstance(reads_per_cell_per_gene, pd.Series)
    assert isinstance(reads_per_cell_per_ct, pd.Series)
    # >= 0 for all
    assert reads_per_cell >= 0
    assert (reads_per_cell_per_gene >= 0).all()
    assert (reads_per_cell_per_ct >= 0).all()
    # all genes and cell types in indices
    assert reads_per_cell_per_gene.index.isin(adata_spatial.var_names).all()
    assert reads_per_cell_per_ct.index.isin(adata_spatial.obs["celltype"].unique()).all()
    # per gene <= total
    assert (reads_per_cell_per_gene <= reads_per_cell).all()
    
    
@pytest.mark.parametrize("adata_spatial, statistic", [
    ("adata_sp", "mean"),
    ("adata_sp", "median"),
    ("adata_sp_not_sparse", "mean"),
    ("adata_sp_not_sparse", "median")
])
def test_genes_per_cell(adata_spatial, statistic, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    genes_per_cell, genes_per_cell_per_ct = tx.quality_metrics.genes_per_cell(
        adata_spatial, statistic=statistic, pipeline_output=False
    )
    
    assert isinstance(genes_per_cell, float)
    assert isinstance(genes_per_cell_per_ct, pd.Series)
    # >= 0 for all
    assert genes_per_cell >= 0
    assert (genes_per_cell_per_ct >= 0).all()
    # <= adata.n_vars for all
    assert genes_per_cell <= adata_spatial.n_vars
    assert (genes_per_cell_per_ct <= adata_spatial.n_vars).all()
    # all cell types in indices
    assert genes_per_cell_per_ct.index.isin(adata_spatial.obs["celltype"].unique()).all()
    # min per cell type <= total & max per cell type >= total
    assert (genes_per_cell_per_ct.min() <= genes_per_cell)
    assert (genes_per_cell_per_ct.max() >= genes_per_cell)
    
    
@pytest.mark.parametrize("adata_spatial", ["adata_sp", "adata_sp_not_sparse"])
def test_number_of_genes(adata_spatial, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    n_genes, n_genes_per_ct = tx.quality_metrics.number_of_genes(adata_spatial, pipeline_output=False)
    
    assert isinstance(n_genes, int)
    assert isinstance(n_genes_per_ct, pd.Series)
    # >= 0 for all
    assert n_genes >= 0
    assert (n_genes_per_ct >= 0).all()
    # all cell types in indices
    assert n_genes_per_ct.index.isin(adata_spatial.obs["celltype"].unique()).all()
    # per cell type <= total
    assert (n_genes_per_ct <= n_genes).all()
    
    
@pytest.mark.parametrize("adata_spatial", ["adata_sp", "adata_sp_not_sparse"])
def test_number_of_cells(adata_spatial, request):
    adata_spatial = request.getfixturevalue(adata_spatial)
    n_cells, n_cells_per_ct = tx.quality_metrics.number_of_cells(adata_spatial, pipeline_output=False)
    
    assert isinstance(n_cells, int)
    assert isinstance(n_cells_per_ct, pd.Series)
    # >= 0 for all
    assert n_cells >= 0
    assert (n_cells_per_ct >= 0).all()
    # all cell types in indices
    assert n_cells_per_ct.index.isin(adata_spatial.obs["celltype"].unique()).all()
    # sum of cell type counts equals total count
    assert n_cells_per_ct.sum() == n_cells
    