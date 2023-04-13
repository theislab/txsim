import numpy as np
from anndata import AnnData
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns
from ..local.local_metrics import check_crop_exists
from ..local.local_metrics import matrix_colorbar_plot
from ..local.local_measurements import get_wrong_spot_ratio
from ..local.local_measurements import get_spot_density
from ..local.local_measurements import get_cell_density
from ..local.local_measurements import get_celltype_density
from ..local.local_measurements import get_number_of_celltypes
from ..local.local_measurements import get_major_celltype_perc
from ..local.local_measurements import get_summed_cell_area
from ..local.local_measurements import get_avg_knn_mixing
from ..local.local_measurements import get_correlation_matrices

def plot_spots(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, show_ticks: bool = False):
    """Plot gene spots.
     
     Spot is red if entry in 'spot_assignment' is "spot in wrong celltype", blue if "spot in correct celltype", grey if "unassigned", green if "no negative marker".

     Parameters
     ----------
     adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
     x_min : int, x_max : int, y_min : int, y_max : int 
          crop coordinates
     image : NDArray
          read from image of dapi stained cell-nuclei
     show_ticks : bool 
          default False, show no ticks or labels
    """

    df = adata_sp.uns["spots"]
    range = check_crop_exists(x_min,x_max,y_min,y_max,image)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()
    
    
    s_factor =  150000/((x_max-x_min)**2)                  

    plt.axis([x_min, x_max, y_max, y_min])        
    plt.imshow(image,cmap = "binary_r") 

    #filter spots
    df = df.loc[(df['x']>= x_min) & (df['x']<=x_max) & (df['y']>=y_min) & (df['y']<=y_max)]
    
    
    plt.scatter(df.loc[df['spot_assignment']=="unassigned","x"],df.loc[df['spot_assignment']=="unassigned","y"], s = 0.3*s_factor, color = "grey", label = "unassigned")
    plt.scatter(df.loc[df['spot_assignment']=="spot in correct celltype","x"],df.loc[df['spot_assignment']=="spot in correct celltype","y"], s = 0.5*s_factor, color = "blue", label = "spot in correct celltype")
    plt.scatter(df.loc[df['spot_assignment']=="spot in wrong celltype","x"],df.loc[df['spot_assignment']=="spot in wrong celltype","y"], s = 1*s_factor, color = "red", label = "spot in wrong celltype")
    plt.scatter(df.loc[df['spot_assignment']=="no negative marker","x"],df.loc[df['spot_assignment']=="no negative marker","y"], s = 1*s_factor, color = "green", label = "no negative marker")
    
    lgnd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1, markerscale=2, frameon=False)

    #fix size of legend spots 
    lgnd.legend_handles[0]._sizes = [10]
    lgnd.legend_handles[1]._sizes = [10]
    lgnd.legend_handles[2]._sizes = [10]
    lgnd.legend_handles[3]._sizes = [10]

    if not show_ticks:
        plt.tick_params(which='both', bottom=False, left=False, labelbottom = False, labelleft = False)


def plot_wrong_spot_ratio(adata_sp : AnnData, x_min: int, x_max: int, y_min: int, y_max: int,
                       image: np.ndarray, bins, smooth: float = 0, show_ticks: bool = False): 
    """Plot density of spots in wrong celltype.

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """

    matrix, range = get_wrong_spot_ratio(adata_sp,x_min,x_max,y_min,y_max,image,bins)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()
    
    title = "spots in wrong celltype density"
    vmin, vmax = [0,1]
    matrix_colorbar_plot(matrix, title, x_min, x_max, y_min, y_max, vmin, vmax, smooth, show_ticks)



def plot_spot_density(adata_sp : AnnData, spot_types: list, x_min: int, x_max: int, y_min: int, y_max: int,
                      image: np.ndarray, bins, smooth: float = 0, show_ticks: bool = False):
    """Plot density of specified spot types ("spot in wrong celltype", "spot in correct celltype", "unassigned", "no negative marker").

    Parameters
    ----------
    adata_sp : AnnData
        Annotated ``AnnData`` object with counts from spatial data
    spot_types: list[str]                    
        specify, which spot types ("spot in wrong celltype", "spot in correct celltype", "unassigned", "no negative marker") should be included in calculation
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """
    matrix, range = get_spot_density(adata_sp,spot_types,x_min,x_max,y_min,y_max,image,bins)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()
    
    spot_types_str = '-/'.join([f"'{spot}'" for spot in spot_types])
    title = f"density of {spot_types_str} spots"        
    vmin, vmax = [0,None]
    matrix_colorbar_plot(matrix, title, x_min, x_max, y_min, y_max, vmin, vmax, smooth, show_ticks)



def plot_cell_density(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int,
                      image: np.ndarray, bins, smooth: float = 0, show_ticks: bool = False):
    """Plot cell density.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """

    matrix, range = get_cell_density(adata_sp,x_min,x_max,y_min,y_max,image,bins)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()

    title = "cell density"
    
    vmin, vmax = [0,None]
    matrix_colorbar_plot(matrix, title, x_min, x_max, y_min, y_max, vmin, vmax, smooth, show_ticks)



def plot_celltype_density(adata_sp: AnnData, celltypes: list, x_min: int, x_max: int, y_min: int, y_max: int,
                      image: np.ndarray, bins, smooth: float = 0, show_ticks: bool = False):
    """Plot cell density

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    celltypes : list[str]
        list of celltypes
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : int or array_like or [int, int] or [array, array]
        The bin specification:
        If int, the number of bins for the two dimensions (nx=ny=bins).
        If array_like, the bin edges for the two dimensions (x_edges=y_edges=bins).
        If [int, int], the number of bins in each dimension (nx, ny = bins).
        If [array, array], the bin edges in each dimension (x_edges, y_edges = bins).
        A combination [int, array] or [array, int], where int is the number of bins and array is the bin edges.
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """
    number_of_celltypes = len(celltypes)
    
    for i in range(number_of_celltypes):
        matrix, range_ = get_celltype_density(adata_sp,celltypes[i],x_min,x_max,y_min,y_max,image,bins)
        x_min, x_max, y_min, y_max = np.ravel(range_).tolist()
        title = f"{celltypes[i]} density"
        vmin, vmax = [0,None]
        matrix_colorbar_plot(matrix, title, x_min, x_max, y_min, y_max, vmin, vmax, smooth, show_ticks)



def plot_number_of_celltypes(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins: Tuple[int,int], smooth: float = 0, show_ticks: bool = False):    
    """Plot number of celltypes
    
    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """
    matrix, range = get_number_of_celltypes(adata_sp,x_min,x_max,y_min,y_max,image,bins)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()
    
    title = "number of celltypes"
    vmin, vmax = [0,None]
    matrix_colorbar_plot(matrix, title, x_min, x_max, y_min, y_max, vmin, vmax, smooth, show_ticks)



def plot_major_celltype_perc(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins: Tuple[int,int], smooth: float = 0, show_ticks: bool = False):    
    """Plot major celltype percentage.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """
    
    matrix, range = get_major_celltype_perc(adata_sp,x_min,x_max,y_min,y_max,image,bins)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()

    title = "major celltype percentage"
    vmin, vmax = [0,None]
    matrix_colorbar_plot(matrix, title, x_min, x_max, y_min, y_max, vmin, vmax, smooth, show_ticks)



def plot_summed_cell_area(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins: Tuple[int,int], smooth: float = 0, show_ticks: bool = False):
    """Plot summed cell area.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """

    matrix, range = get_summed_cell_area(adata_sp,x_min,x_max,y_min,y_max,image,bins)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()

    title = "summed cell area"
    vmin, vmax = [0,None]
    matrix_colorbar_plot(matrix, title, x_min, x_max, y_min, y_max, vmin, vmax, smooth, show_ticks)



def plot_avg_knn_mixing(adata_sp: AnnData, x_min: int, x_max: int, y_min: int, y_max: int, image: np.ndarray, bins: Tuple[int,int], smooth: float = 0, show_ticks: bool = False):
    """Plot summed cell area.

    Parameters
    ----------
    adata_sp: AnnData
        Annotated ``AnnData`` object with counts from spatial data
    x_min : int, x_max : int, y_min : int, y_max : int 
        crop coordinates
    image : NDArray
        read from image of dapi stained cell-nuclei
    bins : [int,int]
        the number of bins in each dimension
    smooth : float = 0
        sigma parameter of scipy.ndimage.gaussian_filter function
    show_ticks : bool 
        default False, show no ticks or labels
    """

    matrix, range = get_avg_knn_mixing(adata_sp,x_min,x_max,y_min,y_max,image,bins)
    x_min, x_max, y_min, y_max = np.ravel(range).tolist()

    title = "average knn mixing"
    vmin, vmax = [0,1]
    matrix_colorbar_plot(matrix, title, x_min, x_max, y_min, y_max, vmin, vmax, smooth, show_ticks)


def plot_correlation_matrices_and_pairplot(adata_sp,x_min,x_max,y_min,y_max,image,bins,celltype):
    pearson_corr_matrix, spearman_corr_matrix, measurements_df = get_correlation_matrices(adata_sp,x_min,x_max,y_min,y_max,image,bins,celltype)
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title("measurements pearson-corrlations", fontsize = 10)
    ax2.set_title("measurements spearman-corrlations", fontsize = 10)
    plot1 = ax1.imshow(pearson_corr_matrix)
    plot2 = ax2.imshow(spearman_corr_matrix)

    labels = ["wrong spot ratio","spot density","cell density", "celltype density",
            "number of celltypes","major celltype perc","summed cell area", "avg knn mixing"]
    for ax in [ax1,ax2]:
        ax.set_xticks([0,1,2,3,4,5,6,7],labels)
        ax.set_yticks([0,1,2,3,4,5,6,7],labels)
        ax.set_xticklabels(labels, rotation=90)
        
    fig.colorbar(plot1,fraction=0.046, pad=0.04)    
    fig.colorbar(plot2,fraction=0.046, pad=0.04)    

    plt.subplots_adjust(wspace=1.5)        

    sns.pairplot(measurements_df)