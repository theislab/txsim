from anndata import AnnData
import numpy as np
from skimage.morphology import convex_hull_image 


def _uniform_cell(adata_sp: AnnData):
    """Compute how uniform spots are distributed over cells.
    
    Therefore we compare the observed and expected counts using the chi-quadrat-statistic.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    """

    df = adata_sp.uns["spots"]
    unique_cell_ids = adata_sp.obs["cell_id"].unique()
    adata_sp.obs["uniform_cell"] = np.nan

    for i in unique_cell_ids:
        spots = df.loc[df["cell"] == i].copy()
        spots["x"], spots["y"] = [spots["x"].round().astype(int), spots["y"].round().astype(int)]      
        [x_min, x_max, y_min, y_max] = [spots["x"].min(),spots["x"].max(),spots["y"].min(),spots["y"].max()]
        spots["x"], spots["y"] = spots["x"]-x_min, spots["y"]-y_min

        seg_mask = np.zeros((x_max-x_min+1,y_max-y_min+1))
        seg_mask[spots["x"].values.tolist(), spots["y"].values.tolist()] = 1     #?spot mir Koord. (x,y) wird geplottet bei (y,x)
        cell = convex_hull_image(seg_mask)

        # Count the number of spots in each quadrat
        n_quadrats_x, n_quadrats_y = x_max-x_min+1, y_max-y_min+1  # Define the number of quadrats in each dimension
        quadrat_counts = np.histogram2d(spots["x"], spots["y"], bins=[n_quadrats_x, n_quadrats_y])[0]

        # observed and expected counts
        observed_counts = quadrat_counts[cell]
        total_spots = len(spots)
        n_pixs = np.sum(cell)
        mean_pix = total_spots / n_pixs
        expected_counts = np.full_like(observed_counts, mean_pix)

        # Calculate the Chi-squared statistic
        chi2_statistic = np.sum((observed_counts - expected_counts)**2 / expected_counts)

        #delta peak: all spots in one pixel
        chi2_delta = (n_pixs-1)*mean_pix + (total_spots-mean_pix)**2/mean_pix       #richtig so?

        # Calculate a uniformness measure based on the Chi-squared statistic
        adata_sp.obs.loc[adata_sp.obs["cell_id"]==i,"uniform_cell"] = 1 - chi2_statistic / chi2_delta  

