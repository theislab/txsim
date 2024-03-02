import random
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
from tqdm import tqdm
import itertools


class Simulation:
    def __init__(self, n_celltypes=3, genes=["IL32", "CYBA", "UBB", "AKR1C3"], data_dir: str = "../data"):
        """Create a simulation object for spatial transcriptomics data.
        ----------
        n_celltypes: int
            Number of cell types to simulate spatial data for. The simulation will use the n_celltypes most frequent cell types in the PBMC3k dataset.
        genes: str | list
            List of genes to keep in the simulation. Specify "all" to keep all genes in the PBMC3k dataset.
        data_dir: str
            Path to data directory.

        Returns
        -------
        Simulation object
        """
        # Set up data directory
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.datasetdir = data_dir

        self.n_celltypes = n_celltypes
        self.adata_sc, self.celltypes = self._prepare_sc_data()

        if genes == "all":
            self.genes = self.adata_sc.var_names
        else:
            self.genes = genes

    def simulate_spatial_data(self, gene_to_simulate, n_groups=10):
        """
        Simulate spatial data based on the scRNA-seq data.
        The expression levels of this gene will increase across the y-axis of the simulated spatial data.
        ----------
        gene_to_simulate: str
            Gene to simulate spatial data for. Must be in the list of genes provided when creating the simulation object.
        n_groups: int
            Number of groups to split the expression levels into per cell type.

        Returns
        -------
        adata_sp: AnnData
            Annotated ``AnnData`` object with simulated spatial data. The resulting object will also be stored directly in the simulation object.
        """
        if gene_to_simulate not in self.genes:
            raise ValueError(
                f"Gene {gene_to_simulate} not in the list of genes {self.genes} provided when creating the simulation object.")

        ################################################
        # Define expression level groups per cell type #
        ################################################

        # df of cells with four genes
        df = pd.DataFrame(index=self.adata_sc.obs_names, columns=self.genes,
                          data=self.adata_sc[:, self.genes].X.toarray())
        df["celltype"] = self.adata_sc.obs["louvain"].values

        # Split into cell type dfs
        dfs = [df.loc[df["celltype"] == c].copy() for c in self.celltypes]

        # Add group column
        for df_ in dfs:
            n_obs = len(df_)
            n_per_group = n_obs // n_groups
            for gene in self.genes:
                df_.loc[df_.sort_values(gene).index, gene + "_group"] = np.repeat(np.arange(n_groups + 1), n_per_group)[
                                                                        :n_obs]
                df_.loc[df_[gene + "_group"] > n_groups - 1, gene + "_group"] = n_groups - 1

        # Concatenate back with initial order
        df = pd.concat(dfs).loc[self.adata_sc.obs_names]

        y_groups = np.arange(n_groups)

        #############################################
        # Define spatial distribution of cell types #
        #############################################

        N = 5  # general factor for number of cells per grid field per cell type
        F_CT = 4  # number of grid field columns over which we increase the number of cells of a cell type

        def stair(f_ct=F_CT, n=1, N=N):
            return np.concatenate(
                [np.arange(1, f_ct + 1) * N for _ in range(n)])  # NOTE: maybe quadratic increase more useful

        def zeros(f_ct=F_CT, n=1):
            return np.repeat(0, n * f_ct)

        x_n_cells_ct0 = np.concatenate(
            [stair(), zeros(), zeros(), stair(), stair(), zeros(), stair()])  # NOTE: simplify with itertools?
        x_n_cells_ct1 = np.concatenate([zeros(), stair(), zeros(), stair(), zeros(), stair(), stair()])
        x_n_cells_ct2 = np.concatenate([zeros(), zeros(), stair(), zeros(), stair(), stair(), stair()])
        x_n_cells = np.array([x_n_cells_ct0, x_n_cells_ct1, x_n_cells_ct2]).T

        obs1, pos1 = self._sample1(df, y_groups, x_n_cells, gene_to_simulate, self.celltypes, seed=0)

        adata_sp = self.adata_sc[obs1, self.genes].copy()
        adata_sp.obs["x"] = pos1[:, 0]
        adata_sp.obs["y"] = pos1[:, 1]

        adata_sp = self._simulate_spots(adata_sp)

        # filter out cells with no spots
        adata_sp.obs["n_spots"] = [len(adata_sp[adata_sp.obs_names == cell_id].uns["spots"]) for cell_id in adata_sp.obs_names]
        adata_sp = adata_sp[adata_sp.obs["n_spots"] > 0]

        self.adata_sp = adata_sp

        return adata_sp

    def simulate_exact_positions(self, radius_range=(0.01, 0.05), cell_sampling_type='uniform', spot_sampling_type='uniform',
                                  adata_sp=None, cell_spread=0.1):
        """Simulate exact cell and spot positions in each grid field.
        ----------
        radius_range: tuple
            Range of cell radii to sample from.
        cell_sampling_type: str
            Type of cell position sampling. Either 'uniform' or 'normal'.
        spot_sampling_type: str
            Type of spot position sampling. Either 'uniform' or 'normal'.
        adata_sp: AnnData
            Annotated ``AnnData`` object with spatial data. Leave blank to use the spatial data stored in the simulation object.
        cell_spread: float
            Spread of cell positions when using normal sampling.

        Returns
        -------
        adata_sp: AnnData
            Annotated ``AnnData`` object with simulated spatial data. The resulting object will also be stored directly in the simulation object.
        """
        assert cell_sampling_type in ['uniform', 'normal'], "cell_sampling_type must be 'uniform' or 'normal'"
        assert spot_sampling_type in ['uniform', 'normal'], "spot_sampling_type must be 'uniform' or 'normal'"

        if adata_sp is None and self.adata_sp is None:
            raise ValueError("No spatial data provided. Either provide adata_sp or run simulate_spatial_data first.")
        elif adata_sp is None:
            adata_sp = self.adata_sp

        def sample_cell_pos(sampling_type, radius):
            if sampling_type == 'uniform':
                x_cell = random.uniform(x + radius, x + 1 - radius)
                y_cell = random.uniform(y + radius, y + 1 - radius)
            elif sampling_type == 'normal':
                x_cell = random.gauss(x + 0.5, cell_spread)
                y_cell = random.gauss(y + 0.5, cell_spread)
            return x_cell, y_cell

        new_cell_pos_list = []  # Collect new cell positions
        new_spot_pos_list = []  # Collect new spot positions

        # Iterate over grid fields
        for x, y in tqdm(itertools.product(sorted(adata_sp.obs["x"].unique()), sorted(adata_sp.obs["y"].unique()))):
            cells_in_grid = adata_sp[(adata_sp.obs["x"] == x) & (adata_sp.obs["y"] == y)].obs_names

            # Simulate cell positions
            cell_radii = np.random.uniform(*radius_range, len(cells_in_grid))
            cell_areas = np.pi * cell_radii ** 2
            x_cells, y_cells = zip(*[sample_cell_pos(cell_sampling_type, r) for r in cell_radii])
            x_cells, y_cells = list(x_cells), list(y_cells)

            # Check for overlapping cells
            overlapping = True
            overlapping_counter = 0
            while overlapping:
                overlapping = False
                assert overlapping_counter < 250, "Could not find non-overlapping cell positions. Please decrease the possible cell radii."
                for i in range(len(x_cells)):
                    for j in range(i + 1, len(x_cells)):
                        distance = np.sqrt((x_cells[i] - x_cells[j]) ** 2 + (y_cells[i] - y_cells[j]) ** 2)
                        if distance < (cell_radii[i] + cell_radii[j]):
                            overlapping = True
                            overlapping_counter += 1
                            # If there is an overlap, sample new positions for one of the overlapping cells
                            x_cells[j], y_cells[j] = sample_cell_pos(cell_sampling_type, cell_radii[j])

            new_cell_pos_list.append(pd.DataFrame({"cell_id": cells_in_grid, "x": x_cells, "y": y_cells, "area": cell_areas}))

            # Simulate spot positions
            spots_df = adata_sp.uns["spots"][adata_sp.uns["spots"]["cell_id"].isin(cells_in_grid)]

            # sort in cells_in_grid order
            spots_df["cell_id"] = pd.Categorical(spots_df["cell_id"], categories=cells_in_grid, ordered=True)
            spots_df = spots_df.sort_values("cell_id")

            if spot_sampling_type == 'uniform':
                spots_alpha = 2 * np.pi * np.random.random(len(spots_df))
                spots_radii = (np.sqrt(np.random.random(len(spots_df))) * cell_radii.repeat(
                    spots_df.groupby('cell_id').size()))

                spot_x = spots_radii * np.cos(spots_alpha) + np.repeat(x_cells, spots_df.groupby('cell_id').size())
                spot_y = spots_radii * np.sin(spots_alpha) + np.repeat(y_cells, spots_df.groupby('cell_id').size())
            elif spot_sampling_type == 'normal':
                # spread for this spot is 0.25 * mean cell radius
                spot_spread = 0.25 * np.mean(cell_radii)
                spot_x = np.random.normal(np.repeat(x_cells, spots_df.groupby('cell_id').size()), spot_spread, len(spots_df))
                spot_y = np.random.normal(np.repeat(y_cells, spots_df.groupby('cell_id').size()), spot_spread, len(spots_df))

            new_spot_pos_list.append(
                pd.DataFrame({"Gene": spots_df["Gene"], "x": spot_x, "y": spot_y, "cell_id": spots_df["cell_id"], "celltype": spots_df["celltype"]}))


        # Update adata_sp
        adata_sp.obs["grid_x"] = adata_sp.obs["x"]
        adata_sp.obs["grid_y"] = adata_sp.obs["y"]

        # Add x position in order of cell_id and adata_sp.obs_names
        new_cell_pos = pd.concat(new_cell_pos_list).set_index('cell_id').loc[adata_sp.obs_names].reset_index()
        adata_sp.obs["x"] = new_cell_pos["x"].tolist()
        adata_sp.obs["y"] = new_cell_pos["y"].tolist()
        adata_sp.obs["area"] = new_cell_pos["area"].tolist()

        adata_sp.uns["spots"] = pd.concat(new_spot_pos_list)

        self.adata_sp = adata_sp
        return adata_sp

    def plot_hist(self, adata=None, hue_order=None):
        """Plot a histogram of the gene expression for each cell type.
        ----------
        adata: AnnData
            Annotated ``AnnData`` object storing scRNA-seq data. Leave blank to use the scRNA-seq data stored in the simulation object.
        hue_order: list
            Order of cell types to plot. If None, will use the order of cell types in the simulation object.

        Returns
        -------
        None
        """
        if adata is None:
            adata = self.adata_sc
        if hue_order is None:
            hue_order = self.celltypes

        df_expr = self._get_gene_expression_dataframe(adata, self.genes)

        sns.displot(
            data=df_expr, x="expr", hue="celltype", col="gene", hue_order=hue_order,
            kind="hist", stat="probability", common_norm=False, facet_kws=dict(sharey=False), height=3, aspect=1,
        )
        plt.show()

    def plot_kde(self, adata=None, hue_order=None):
        """Plot a kernel density estimate of the gene expression for each cell type.
        ----------
        adata: AnnData
            Annotated ``AnnData`` object storing scRNA-seq data. Leave blank to use the scRNA-seq data stored in the simulation object.
        hue_order: list
            Order of cell types to plot. If None, will use the order of cell types in the simulation object.

        Returns
        -------
        None
        """
        if adata is None:
            adata = self.adata_sc
        if hue_order is None:
            hue_order = self.celltypes

        df_expr = self._get_gene_expression_dataframe(adata, self.genes)

        sns.displot(
            data=df_expr, x="expr", hue="celltype", col="gene", hue_order=hue_order,
            kind="kde", common_norm=False, facet_kws=dict(sharey=False), height=3, aspect=1, warn_singular=False
        )
        plt.show()

    def spatial_heatmap_gene(self, gene, adata_sp=None, celltypes=None, flavor="mean", **kwargs):
        """ Plot a heatmap of the mean expression of a gene in each grid field.
        ----------
        gene: str
            Gene to plot.
        adata_sp: AnnData
            Annotated ``AnnData`` object with spatial data. Leave blank to use the spatial data stored in the simulation object.
        celltypes: list
            List of cell types to include in the plot. If None, will include all cell types.
        flavor: str
            Flavor of the heatmap, either "mean" or "sum". Whether to plot the mean or sum expression of the gene in each grid field.
        **kwargs: dict
            Additional arguments to pass to plt.pcolormesh.

        Returns
        -------
        None
        """
        assert flavor in ["mean", "sum"], "flavor must be 'mean' or 'sum'"

        if adata_sp is None and self.adata_sp is None:
            raise ValueError("No spatial data provided. Either provide adata_sp or run simulate_spatial_data first.")
        elif adata_sp is None:
            adata_sp = self.adata_sp

        if celltypes is not None:
            adata_sp = adata_sp[adata_sp.obs["louvain"].isin(celltypes)]

        plt.figure(figsize=(8, 2.5))

        if flavor == "sum":
            hist = np.histogram2d(adata_sp.obs["x"].values, adata_sp.obs["y"].values,
                                  weights=adata_sp[:, gene].X.toarray().flatten(),
                                  bins=(adata_sp.obs["x"].nunique(), adata_sp.obs["y"].nunique()))[0].T
            plt.pcolormesh(hist, **kwargs)

        else:
            plot_data = pd.DataFrame(data={
                "expr": adata_sp[:, gene].X.toarray().flatten(),
                "x": adata_sp.obs["x"].values,
                "y": adata_sp.obs["y"].values,
            })

            x_values = sorted(plot_data['x'].unique())
            y_values = sorted(plot_data['y'].unique())

            # Create a grid of mean expression values for each coordinate
            grid = np.zeros((len(y_values), len(x_values)))
            for i, y in enumerate(y_values):
                for j, x in enumerate(x_values):
                    grid[i, j] = plot_data[(plot_data['x'] == x) & (plot_data['y'] == y)]['expr'].mean()

            plt.pcolormesh(x_values, y_values, grid, **kwargs)

        plt.colorbar(label=f'Mean {gene} expression' if flavor == "mean" else f'Sum {gene} expression')
        plt.title(f'{gene} expression in each grid field')
        plt.show()

    def spatial_heatmap_metric(self, metric_matrix, metric_name="Metric", **kwargs):
        """Plot a heatmap of a metric in each grid field.
        ----------
        metric_matrix: np.array
            Matrix of the metric to plot.
        metric_name: str
            Name of the metric to plot.
        **kwargs: dict
            Additional arguments to pass to plt.pcolormesh.

        Returns
        -------
        None
        """
        plt.figure(figsize=(8, 2.5))
        plt.pcolormesh(metric_matrix, **kwargs)
        plt.colorbar(label=metric_name)
        plt.title(f'{metric_name} in each grid field')
        plt.show()

    def spatial_cell_plot(self, adata_sp=None, **kwargs):
        """Plot the spatial distribution of cells, colored by cell type.
        ----------
        adata_sp: AnnData
            Annotated ``AnnData`` object with spatial data. Leave blank to use the spatial data stored in the simulation object.
        **kwargs: dict
            Additional arguments to pass to sns.scatterplot.

        Returns
        -------
        None
        """
        if adata_sp is None and self.adata_sp is None:
            raise ValueError("No spatial data provided. Either provide adata_sp or run simulate_spatial_data first.")
        elif adata_sp is None:
            adata_sp = self.adata_sp

        sns.scatterplot(data=adata_sp.obs, x="x", y="y", hue="louvain", **kwargs)
        plt.title("Cell positions")
        plt.show()

    def spatial_spot_plot(self, adata_sp=None, **kwargs):
        """Plot the spatial distribution of spots, colored by cell type.
        ----------
        adata_sp: AnnData
            Annotated ``AnnData`` object with spatial data. Leave blank to use the spatial data stored in the simulation object.
        **kwargs: dict
            Additional arguments to pass to sns.scatterplot.

        Returns
        -------
        None
        """
        if adata_sp is None and self.adata_sp is None:
            raise ValueError("No spatial data provided. Either provide adata_sp or run simulate_spatial_data first.")
        elif adata_sp is None:
            adata_sp = self.adata_sp

        sns.scatterplot(data=adata_sp.uns["spots"], x="x", y="y", hue="celltype", **kwargs)
        plt.title("Spot positions")
        plt.show()

    def _prepare_sc_data(self):
        """Prepare scRNA-seq data for simulation.
        """
        # Take lognorm counts from raw adata, since the processed has scaled counts (we don't want scaled).
        adata_raw = sc.datasets.pbmc3k()
        sc.pp.normalize_total(adata_raw)
        sc.pp.log1p(adata_raw)

        # Get processed (includes cell type annotations)
        adata = sc.datasets.pbmc3k_processed().copy()
        adata.X = adata_raw[adata.obs_names, adata.var_names].X

        # Reduce to n cell types
        celltypes = adata.obs["louvain"].value_counts().iloc[:self.n_celltypes].index.tolist()
        adata = adata[adata.obs["louvain"].isin(celltypes)].copy()

        del adata.uns["rank_genes_groups"]

        # I have absolutely no clue why I keep getting more genes than there are in adata.var when running DE tests (only creating a new AnnData worked for me)
        adata = ad.AnnData(X=adata.X, var=adata.var, obs=adata.obs[["louvain"]])

        return adata, celltypes

    def _get_gene_expression_dataframe(self, adata, genes, ct_key="louvain"):
        """Get a dataframe of gene expression levels for each cell type.
        ----------
        adata: AnnData
            Annotated ``AnnData`` object storing scRNA-seq data
        genes: list
            List of genes to include in the dataframe.
        ct_key: str
            Key in adata.obs to use for cell type annotations.

        Returns
        -------
        df_expr: pd.DataFrame
            Dataframe of gene expression levels for each cell type.
        """
        df_expr = pd.DataFrame(data={
            "expr": np.concatenate([adata[:, g].X.toarray().flatten() for g in genes]),
            "gene": np.concatenate([np.repeat(g, adata.n_obs) for g in genes]),
            "celltype": np.concatenate([adata.obs[ct_key].values for _ in genes]),
        })

        return df_expr

    def _sample1(self, df, y_groups, x_n_cells, gene, celltypes, seed=0):
        """ Sample from cells in different expression level groups (rows) over different numbers of cells per cell type (column)
        ----------
        df: pd.DataFrame
            Dataframe of cells with gene expression levels and cell type annotations. Use _get_gene_expression_dataframe to create this dataframe.
        y_groups: list
            shape = (number of grid field rows)
            Array of groups (each group should occur exactly once, otherwise the cell state frequencies are not correct anymore)
        x_n_cells: np.array
            shape = (number of grid field columns, number of cell types)
            Matrix of number of cells per cell type that we change along each row of the grid field

        """
        np.random.seed(seed)

        obs = []
        positions = []

        for y, group in enumerate(y_groups):
            obs_pools = [df.loc[(df["celltype"] == ct) & (df[gene + "_group"] == group)].index.tolist() for ct in
                         celltypes]

            for x in range(x_n_cells.shape[0]):
                for ct_idx in range(x_n_cells.shape[1]):
                    obs_ = np.random.choice(obs_pools[ct_idx], x_n_cells[x, ct_idx], replace=True)
                    positions += [[x, y] for _ in obs_]
                    obs += list(obs_)

        return obs, np.array(positions)

    def _simulate_spots(self, adata_sp):
        """Simulate spots in spatial data. Note that this does not simulate the exact positions of cells and spots, it merely creates a spots dataframe in adata_sp.uns.
        ----------
        adata_sp: AnnData
            Annotated ``AnnData`` object with spatial data. The resulting object will also be stored directly in the simulation object.

        Returns
        -------
        adata_sp: AnnData
            Annotated ``AnnData`` object with simulated spot positions in adata_sp.uns.
        """
        raw_counts = sc.datasets.pbmc3k()
        adata_sp.layers["raw"] = raw_counts[adata_sp.obs_names, adata_sp.var_names].X

        # make obs names unique
        adata_sp.obs["sc_cell_id"] = adata_sp.obs_names
        adata_sp.obs_names_make_unique()

        # create empty spots dataframe
        spots = []

        for obs_name in adata_sp.obs_names.unique():
            # get raw counts for that obs name (possibly multiple cells)
            obs_counts_sparse = adata_sp[obs_name].layers["raw"]
            # get indices and counts of genes with non-zero counts for that observation
            _, gene_indices, gene_counts = scipy.sparse.find(obs_counts_sparse)

            if len(gene_counts) > 0:
                nr_cells = obs_counts_sparse.shape[0]

                # append spots data by repeating the gene names as often as they occur in the cell
                # repeat x and y coordinates of that cell as often as RNA molecules found that cell
                spots.append({
                    "Gene": np.repeat(adata_sp.var_names[gene_indices], gene_counts),
                    "x": np.repeat(adata_sp.obs.loc[obs_name, "x"], [sum(gene_counts) / nr_cells] * nr_cells),
                    "y": np.repeat(adata_sp.obs.loc[obs_name, "y"], [sum(gene_counts) / nr_cells] * nr_cells),
                    "cell_id": np.repeat(obs_name, sum(gene_counts)),
                    "celltype": np.repeat(adata_sp.obs.loc[obs_name, "louvain"], sum(gene_counts))
                })

        adata_sp.uns["spots"] = pd.concat(pd.DataFrame(s) for s in spots)

        # sanity check
        assert adata_sp.layers["raw"].sum() == adata_sp.uns["spots"].shape[0]

        return adata_sp
