import squidpy as sq
from squidpy.im._container import ImageContainer
from squidpy.im._segment import SegmentationModel
from cellpose import models
from typing import Union,  Optional, Any, Mapping, Callable, Sequence, TYPE_CHECKING, Tuple
from squidpy._utils import NDArrayA

#TODO seperate cellpose and watershed into two seperate functions
def cellpose(
    img: NDArrayA, 
    min_size: Optional[int] = 15
) -> NDArrayA:
    """Run cellpose and get masks

    Parameters
    ----------
    img : NDArrayA
        Can be list of 2D/3D images, or array of 2D/3D images, or 4D image array
    min_size : Optional[int], optional
        Minimum number of pixels per mask, can turn off with -1, by default 15

    Returns
    -------
    NDArray
        labelled image, where 0=no masks; 1,2,...=mask labels
    """
    
    model = models.Cellpose(model_type='nuclei')
    res, _, _, _ = model.eval(
        img,
        channels=[0, 0],
        diameter=None,
        min_size=min_size,
    )
    return res

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
      
    if(method == 'cellpose'):
        return sq.im.segment(img=img, layer= layer, library_id=library_id, method=cellpose, 
            channel=channel, chunks=chunks, lazy=lazy, layer_added=layer_added, copy=copy, **kwargs)

    else:
        sq.im.process(
            img=img,
            layer=layer,
            method = "smooth",
            layer_added = "image_smooth"
        )
        
        return sq.im.segment(img=img, layer= "image_smooth", library_id=library_id, method=method, 
            channel=channel, chunks=chunks, lazy=lazy, layer_added=layer_added, copy=copy, **kwargs)
