import squidpy as sq
import numpy as np
from squidpy.im._container import ImageContainer
from squidpy.im._segment import SegmentationModel
from typing import Union,  Optional, Any, Mapping, Callable, Sequence, TYPE_CHECKING, Tuple
from squidpy._utils import NDArrayA

def segment_nuclei(
    img: ImageContainer,
    layer: Optional[str] = None,
    library_id: Union[str, Sequence[str], None] = None,
    method: Union[str, SegmentationModel, Callable[..., NDArrayA]] = "watershed",
    channel: Optional[int] = 0,
    chunks: Union[str, int, Tuple[int, int], None] = None,
    lazy: bool = False,
    layer_added: Optional[str] = None,
    copy: bool = False,
    **kwargs: Any,
) -> Optional[ImageContainer]:
    """Squidpy segment wrapper function
    Based on https://github.com/scverse/squidpy version 1.2.2
    This function will also smooth the image via ``process``

    Parameters
    ----------
    img : ImageContainer
        High-resolution image.
    layer : Optional[str], optional
        Image layer in `img` that should be processed. If None and only 1 layer is present, 
        it will be selected., by default None
    library_id : Union[str, Sequence[str], None], optional
        Name of the Z-dimension(s) that this function should be applied to. 
        For not specified Z-dimensions, the identity function is applied. 
        If None, all Z-dimensions are segmented separately, by default None
    method : Union[str, SegmentationModel, Callable[..., NDArrayA]], optional
        Segmentation method to use. Valid options are:
        ``watershed`` - skimage.segmentation.watershed().
        Alternatively, any callable() can be passed as long as it has the following signature: 
        ``numpy.ndarray (height, width, channels)`` -> ``numpy.ndarray (height, width[, channels])``,
        by default "watershed"
    channel : Optional[int], optional
        Channel index to use for segmentation. If None, use all channels, by default 0
    chunks : Union[str, int, Tuple[int, int], None], optional
        Number of chunks for dask. For automatic chunking, use ``chunks = 'auto'``, by default None
    lazy : bool, optional
        Whether to lazily compute the result or not. Only used when ``chunks != None``, by default False
    layer_added : Optional[str], optional
        Layer of new image layer to add into img object. If None, use 'segmented_{model}'., by default None
    copy : bool, optional
        If True, return the result, otherwise save it to the image container, by default False

    Returns
    -------
    Optional[ImageContainer]
        If copy = True, returns a new container with the segmented image in '{layer_added}'.
        Otherwise, modifies the img with the following key:
        ``squidpy.im.ImageContainer ['{layer_added}']``
    """  
      

    if (kwargs is not None) and ("blur_std" in kwargs):
        if kwargs["blur_std"] > 0:
            sq.im.process(
                img, 
                layer="image", 
                method="smooth", #skimage.filters.gaussian, 
                layer_added="image",
                sigma=kwargs["blur_std"],
                truncate=4.0,
            )
        del kwargs["blur_std"]
        
    #sq.im.process(
    #    img=img,
    #    layer=layer,
    #    method = "smooth",
    #    layer_added = "image_smooth"
    #)
    
    return sq.im.segment(img=img, layer= "image", library_id=library_id, method=method, 
        channel=channel, chunks=chunks, lazy=lazy, layer_added=layer_added, copy=copy, **kwargs)

# def segment_cellpose(
#     img: ImageContainer,
#     layer: Optional[str] = None,
#     library_id: Union[str, Sequence[str], None] = None,
#     channel: Optional[int] = 0,
#     chunks: Union[str, int, Tuple[int, int], None] = None,
#     lazy: bool = False,
#     layer_added: Optional[str] = None,
#     copy: bool = False,
#     **kwargs: Any,
# ) -> Optional[ImageContainer]:
#     """Squidpy segment wrapper function
#     Based on https://github.com/scverse/squidpy version 1.2.2
#     This function will also smooth the image via ``process``

#     Parameters
#     ----------
#     img : ImageContainer
#         High-resolution image.
#     layer : Optional[str], optional
#         Image layer in `img` that should be processed. If None and only 1 layer is present, 
#         it will be selected., by default None
#     library_id : Union[str, Sequence[str], None], optional
#         Name of the Z-dimension(s) that this function should be applied to. 
#         For not specified Z-dimensions, the identity function is applied. 
#         If None, all Z-dimensions are segmented separately, by default None
#     channel : Optional[int], optional
#         Channel index to use for segmentation. If None, use all channels, by default 0
#     chunks : Union[str, int, Tuple[int, int], None], optional
#         Number of chunks for dask. For automatic chunking, use ``chunks = 'auto'``, by default None
#     lazy : bool, optional
#         Whether to lazily compute the result or not. Only used when ``chunks != None``, by default False
#     layer_added : Optional[str], optional
#         Layer of new image layer to add into img object. If None, use 'segmented_{model}'., by default None
#     copy : bool, optional
#         If True, return the result, otherwise save it to the image container, by default False

#     Returns
#     -------
#     Optional[ImageContainer]
#         If copy = True, returns a new container with the segmented image in '{layer_added}'.
#         Otherwise, modifies the img with the following key:
#         ``squidpy.im.ImageContainer ['{layer_added}']``
#     """  
#     from cellpose import models
    
#     def cellpose(
#         img: NDArrayA, 
#         min_size: Optional[int] = 15
#     ) -> NDArrayA:
#         """Run cellpose and get masks

#         Parameters
#         ----------
#         img : NDArrayA
#             Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
#         min_size : Optional[int], optional
#             Minimum number of pixels per mask, can turn off with -1, by default 15

#         Returns
#         -------
#         NDArray
#             labelled image, where 0=no masks; 1,2,...=mask labels
#         """
        
#         model = models.Cellpose(model_type='nuclei')
#         res, _, _, _ = model.eval(
#             img,
#             channels=[0, 0],
#             diameter=None,
#             min_size=min_size,
#         )
#         return res

#     return sq.im.segment(img=img, layer=layer, library_id=library_id, method=cellpose, 
#             channel=channel, chunks=chunks, lazy=lazy, layer_added="segmented_cellpose", copy=copy, **kwargs)

def segment_cellpose(
    img: NDArrayA, 
    hyperparams: Optional[dict]
) -> NDArrayA:
    """Run cellpose and get masks

    Parameters
    ----------
    img : NDArrayA
        Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
    Returns
    -------
    NDArray
        labelled image, where 0=no masks; 1,2,...=mask labels
    """
    from cellpose import models
    
    # Set model type
    if (hyperparams is not None) and ("model_type" in hyperparams):
        model_type = hyperparams["model_type"]
    else:
        model_type = 'nuclei'
    
    # Init model
    model = models.Cellpose(model_type=model_type)
    
    # Predict
    if hyperparams is not None:
        if "model_type" in hyperparams:
            del hyperparams["model_type"]
        res, _, _, _ = model.eval(
            img,
            channels=[0, 0],
            **hyperparams
        )
    else:
        res, _, _, _ = model.eval(
            img,
            channels=[0, 0]
        )
        
    return res

def segment_watershed(
    img: NDArrayA,
    hyperparams: Optional[dict]
        
)-> NDArrayA:
    
    from skimage.segmentation import watershed
    from scipy.ndimage import distance_transform_edt
    from skimage.exposure import adjust_gamma, adjust_log, adjust_sigmoid, equalize_adapthist, equalize_hist, rescale_intensity
    from skimage.morphology import remove_small_objects, label,remove_small_holes
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu, threshold_triangle, gaussian, median, rank
    from skimage.morphology import (square, rectangle, diamond, disk, cube,
                            octahedron, ball, octagon, star)
    import skimage.filters.rank as rank
    from scipy import ndimage as ndi
    import matplotlib.pyplot as plt
    import inspect

    def filter_params(func, params):
        """Return only those params that are in the function's signature."""
        signature_params = set(inspect.signature(func).parameters)
        return {k: v for k, v in params.items() if k in signature_params}

    def visualize(image, segmentation_labels,img_name,display = False, *args, **kwargs):
        # Display the original image
        fig, axes = plt.subplots(figsize=(10, 10))
        # Display the original image with the watershed segmentation overlaid
        axes.imshow(image, cmap='gray') # Original image
        axes.imshow(segmentation_labels, cmap='nipy_spectral', alpha=0.5) # Segmentation labels overlaid with transparency
        axes.set_title("Watershed Over Original Image")
        #write image to file
        plt.savefig(img_name+'_watershed.png')
        if display:
            plt.show()


    def find_local_maxima(distance_transform, min_distance=5):
        local_maxima_coords = peak_local_max(distance_transform, min_distance=min_distance)
        local_maxima = np.zeros_like(distance_transform, dtype=bool)
        local_maxima[tuple(local_maxima_coords.T)] = True
        return label(local_maxima)

    # Define normalization function mappings
    normalization_functions = {
        "gamma": adjust_gamma,
        "log": adjust_log,
        "sigmoid": adjust_sigmoid,
        # Add other mappings as needed
    }

    # Define contrast adjustment function mappings
    contrast_adjustment_functions = {
        "equalize_adapthist": equalize_adapthist,
        "equalize_hist": equalize_hist,
        "rescale_intensity": rescale_intensity,
        # Add other mappings as needed
    }

    # Define blur function mappings
    blur_functions = {
        "gaussian": gaussian,
        "median": median,
        # Add other mappings as needed
    }

    # Define threshold function mappings
    threshold_functions = {
        "otsu": threshold_otsu,
        "triangle": threshold_triangle,
        "local_otsu": rank.otsu,
        # Add other mappings as needed
    }

    # Define distance transform function mappings
    distance_transform_functions = {
        "distance_transform_edt": distance_transform_edt,
        # Add other mappings as needed
    }

    # Define local maxima function mappings
    local_maxima_functions = {
        "find_local_maxima": find_local_maxima,
        # Add other mappings as needed
    }

    # Define post-processing function mappings
    post_processing_functions = {
        "remove_small_objects": remove_small_objects,
        "remove_small_holes": remove_small_holes,
        # Add other mappings as needed
    }

    footprints = {
        "square": square,
        "diamond": diamond,
        "disk": disk,
        "cube": cube,
        "octahedron": octahedron,
        "ball": ball,
        "star": star,
        # Add other mappings as needed
    }

    threshold_apply = lambda  img: img > threshold_value

    # Get parameters
    if hyperparams is not None:
        
        normalize_func = normalization_functions.get(hyperparams.get("normalize_func","gamma"))
        contrast_adjustment_func = contrast_adjustment_functions.get(hyperparams.get("contrast_adjustment_func","equalize_hist"))
        blur_func = blur_functions.get(hyperparams.get("blur_func","gaussian"))
        threshold_func = threshold_functions.get(hyperparams.get("threshold_func","otsu"))
        distance_transform_func = distance_transform_functions.get(hyperparams.get("distance_transform_func","distance_transform_edt"))
        local_maxima_func = local_maxima_functions.get(hyperparams.get("local_maxima_func","find_local_maxima"))
        #post_processing_func = post_processing_functions.get(hyperparams.get("post_processing_func"))

        # Normalize image
        if normalize_func is not None:

            normalize_params = {
                "gamma": hyperparams.get("normalize_gamma", 1),
                "gain": hyperparams.get("normalize_gain", 1),
                "inv": hyperparams.get("normalize_inv", False),
                "cutoff": hyperparams.get("normalize_cutoff", 0.5)
            }
            valid_params = filter_params(normalize_func, normalize_params)
            img = normalize_func(img,**valid_params)

        # Contrast adjustment
        if contrast_adjustment_func is not None:
            contrast_adjustment_params = {
                "kernel_size": hyperparams.get("contrast_adjustment_kernel_size"),
                "clip_limit": hyperparams.get("contrast_adjustment_clip_limit", 0.01),
                "nbins": hyperparams.get("contrast_adjustment_nbins", 256),
                "mask": hyperparams.get("contrast_adjustment_mask"),
                "in_range": hyperparams.get("contrast_adjustment_in_range", "image"),
                "out_range": hyperparams.get("contrast_adjustment_out_range", "dtype")
            }

            valid_params = filter_params(contrast_adjustment_func, contrast_adjustment_params)
            img = contrast_adjustment_func(img, **valid_params)
        # Blur image
        if blur_func is not None:
            blur_params = {
                "sigma": hyperparams.get("blur_sigma", 1),
                "output": hyperparams.get("blur_output"),
                "mode": hyperparams.get("blur_mode", 'nearest'),
                "cval": hyperparams.get("blur_cval", 0),
                "preserve_range": hyperparams.get("blur_preserve_range", False),
                "truncate": hyperparams.get("blur_truncate", 4.0)
            }

            valid_params = filter_params(blur_func, blur_params)
            img = blur_func(img, **valid_params)

        # Threshold (Masking), Apply thresholding to detect nuclei
        if threshold_func is not None:

            threshold_params = {
                "nbins": hyperparams.get("threshold_nbins", 256),
                "hist": hyperparams.get("threshold_hist"),
                "out": hyperparams.get("threshold_out"),
                "mask": hyperparams.get("threshold_mask"),
                "shift_x": hyperparams.get("threshold_shift_x", False),
                "shift_y": hyperparams.get("threshold_shift_y", False),
                "shift_z": hyperparams.get("threshold_shift_z", False),
                "footprint_func": hyperparams.get("threshold_footprint", "square"),
                "footprint_size": hyperparams.get("threshold_footprint_size", 10)
            }

            if threshold_func == rank.otsu: # Check if the method is local Otsu
                # Retrieve footprint or any other specific parameter for local Otsu
                footprint_func = footprints.get(threshold_params.get("footprint_func", "square"))
                # Footprint size 
                footprint_size = threshold_params.get("footprint_size", 10)
                # Create footprint
                footprint = footprint_func(footprint_size)
                # Here, use the local Otsu function as needed
                threshold_params["footprint"] = footprint

                valid_params = filter_params(threshold_func, threshold_params)
                threshold_value = threshold_func(img,**valid_params)
                #print(threshold_value)
                nuclei = img >= threshold_value # Applying the lambda function
                

            else: # For global thresholding methods
                valid_params = filter_params(threshold_func, threshold_params)
                threshold_value = threshold_func(img, **valid_params)
                nuclei = threshold_apply(img)

                    # Remove small objects that are not nuclei

            # Initialize an index to keep track of the number of post-processing steps
            post_processing_index = 1

            while True:
                
                
                # Extract the function and parameters for the current post-processing step
                post_processing_func_name = hyperparams.get(f"post_processing_func_{post_processing_index}")
                post_processing_func = post_processing_functions.get(post_processing_func_name)
                
                if post_processing_func is None:
                    break  # Exit the loop if there is no function for the current index

                # Construct a dictionary of parameters for the current post-processing step
                post_processing_params = {
                    "min_size": hyperparams.get(f"post_processing_min_size_{post_processing_index}", 64),
                    "area_threshold": hyperparams.get(f"post_processing_area_threshold_{post_processing_index}", 64),
                    "connectivity": hyperparams.get(f"post_processing_connectivity_{post_processing_index}", 1),
                    "out": hyperparams.get(f"post_processing_out_{post_processing_index}", None),
                }

                # Apply the post-processing function with the extracted parameters
                valid_params = filter_params(post_processing_func, post_processing_params)
                nuclei = post_processing_func(nuclei, **valid_params)
                
                # Increment the index to process the next post-processing step (if any)
                post_processing_index += 1

        # Perform distance transform
        if distance_transform_func is not None:

            distance_transform_params = {
        "sampling": hyperparams.get("distance_transform_sampling"),
        "return_distances": hyperparams.get("distance_transform_return_distances", True),
        "return_indices": hyperparams.get("distance_transform_return_indices", False),
        "distances": hyperparams.get("distance_transform_distances"),
        "indices": hyperparams.get("distance_transform_indices")
            }

            valid_params = filter_params(distance_transform_func, distance_transform_params)
            distance_transform = distance_transform_func(nuclei, **valid_params)

        # Find local maxima to use as markers for watershed
        if local_maxima_func is not None:
            local_maxima_params = {
        "min_distance": hyperparams.get("local_maxima_min_distance", 5)
            }

            local_maxima = local_maxima_func(distance_transform, **local_maxima_params)
        # Perform watershed segmentation

        segmentation_labels = watershed(-distance_transform, local_maxima, **hyperparams.get("watershed_params", {}),mask=nuclei, watershed_line=True)
        
        if "visualize" in hyperparams:
            
            visualize(img,segmentation_labels, hyperparams.get("visualize", "img"))
    else:
        min_distance = 10
        gaussian_sigma = 1
        min_size = 15

        # Smooth image
        img_smooth = gaussian(img, sigma=gaussian_sigma)
    
        # Get local maxima
        #local_maxi = peak_local_max(img_smooth, min_distance=min_distance)
    
        # Get distance transform
        distance = distance_transform_edt(img_smooth)

        # Get markers
        markers = find_local_maxima(distance)

        # Get threshold
        threshold_value = threshold_otsu(img_smooth)
        nuclei = threshold_apply(img_smooth) # Applying the lambda function
    
        # Perform watershed segmentation
        segmentation_labels = watershed(-distance, markers, mask=nuclei,  watershed_line=True)
    
        # Remove small objects
        segmentation_labels = remove_small_objects(segmentation_labels, min_size=min_size)

    return segmentation_labels

def segment_binning(
    img: NDArrayA,
    bin_size: int
) -> NDArrayA:

    # Get shape of image
    n = np.shape(img)[0]
    m = np.shape(img)[1]

    # Create grids of coordinates, and combine to form bins
    x = np.floor(np.mgrid[0:n, 0:m][0] / bin_size)
    y = np.floor(np.mgrid[0:n, 0:m][1] / bin_size)
    bins = x*(np.ceil(m/bin_size)) + y + 1

    return bins

def segment_stardist(
    img: NDArrayA,
    hyperparams: Optional[dict]
) -> NDArrayA:

    from csbdeep.utils import normalize
    from stardist.models import StarDist2D

    # Create a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')
    # Segment and normalize image 
    if hyperparams is not None:
        labels, _ = model.predict_instances(normalize(img), **hyperparams)
    else:
        labels, _ = model.predict_instances(normalize(img))
    return labels


