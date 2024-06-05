import pytest
import anndata as ad
import numpy as np

@pytest.fixture
def adata_sp():
    adata = ad.read_h5ad("tests/_data/adata_sp_simulated.h5ad")
    return adata

@pytest.fixture
def adata_sp_not_sparse():
    adata = ad.read_h5ad("tests/_data/adata_sp_simulated.h5ad")
    adata = adata.copy()
    adata.X = adata.X.toarray()
    for key in adata.layers.keys():
        adata.layers[key] = adata.layers[key].toarray()
    return adata

@pytest.fixture
def adata_sc_high_sim():
    """adata with high (but not perfect) similarity to adata_sp_simulated"""
    adata = ad.read_h5ad("tests/_data/adata_sp_simulated.h5ad")
    np.random.seed(0)
    obs = np.random.choice(adata.obs_names, size=int(0.9*adata.n_obs), replace=True)
    adata = adata[obs]
    adata.obs.index = [f"sc_{i}" for i in range(adata.n_obs)]
    adata = adata.copy() # Important after subsetting
    for key in ["x","y","n_spots","grid_x","grid_y","area"]:
        del adata.obs[key]
    del adata.uns["spots"]
    return adata

@pytest.fixture
def adata_sc_high_sim_not_sparse(adata_sc_high_sim):
    adata = adata_sc_high_sim.copy()
    adata.X = adata.X.toarray()
    for key in adata.layers.keys():
        adata.layers[key] = adata.layers[key].toarray()
    return adata