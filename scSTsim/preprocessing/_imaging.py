import scanpy as sc
import anndata as ad
import squidpy as sq

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from cellpose import models

def cellpose(img, min_size=15):
    model = models.Cellpose(model_type='nuclei')
    res, _, _, _ = model.eval(
        img,
        channels=[0, 0],
        diameter=None,
        min_size=min_size,
    )
    return res

def segment_nuclei(img, 
    layer=None, 
    library_id=None, 
    method='watershed', 
    channel=0, 
    chunks=None, 
    lazy=False, 
    layer_added=None, 
    copy=False, 
    **kwargs
):
    """Squidpy segment wrapper function
    Based on https://github.com/scverse/squidpy version 1.2.2
    This function will also smooth the image via ``process``

    :param img: ``ImageContainer`` High-resolution image.
    :param layer: Image layer in img that should be processed. If None and only 1 layer is present, it will be selected.
    :param library_id: Name of the Z-dimension(s) that this function should be applied to. 
        For not specified Z-dimensions, the identity function is applied. 
        If None, all Z-dimensions are segmented separately.
    :param method: Segmentation method to use. Valid options are:
        ``watershed`` - skimage.segmentation.watershed().
        Alternatively, any callable() can be passed as long as it has the following signature: 
        ``numpy.ndarray (height, width, channels)`` -> ``numpy.ndarray (height, width[, channels])``
    :param channel: Channel index to use for segmentation. If None, use all channels.
    :param chunks: Number of chunks for dask. For automatic chunking, use chunks = 'auto'.
    :param lazy: Whether to lazily compute the result or not. Only used when chunks != None.
    :param layer_added: Layer of new image layer to add into img object. If None, use 'segmented_{model}'.
    :param thresh: Threshold for creation of masked image. The areas to segment should be contained in this mask. If None, it is determined by Otsu’s method. Only used if method = 'watershed'.
    :param geq: Treat thresh as upper or lower bound for defining areas to segment. 
        If geq = True, mask is defined as mask = arr >= thresh, meaning high values in arr denote areas to segment. 
        Only used if method = 'watershed'.
    :param copy: If True, return the result, otherwise save it to the image container.
    :param kwargs: Keyword arguments for the underlying model.
    :return: If copy = True, returns a new container with the segmented image in '{layer_added}'.
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
