import scanpy as sc
import txsim as tx

if __name__ == "__main__":

    # Simulate spatial adata
    sim = tx.simulation.Simulation()
    sim.simulate_spatial_data("IL32", n_groups=3, n_per_bin_and_ct=2, n_cols_cell_numb_increase=2, seed=0)
    sim.simulate_exact_positions(spot_sampling_type='uniform')
    sim.adata_sp.obs['celltype'] = sim.adata_sp.obs['louvain']
    del sim.adata_sp.obs['louvain']
    # Filter genes with 0 counts
    sc.pp.filter_genes(sim.adata_sp, min_cells=1)
    sim.adata_sp.layers['lognorm'] = sim.adata_sp.X
    sim.adata_sp.write("adata_sp_simulated.h5ad")
    #TODO: simulate image as well
    #TODO: adata_sp.uns['spots'].index looks weird -> clean up