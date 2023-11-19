


def dapi_image_not_empty():
    """TODO: Implement; maybe some more sophisticated test, checking if there is some cell structure in the image
    
    
    """
    
    
# Can we write a test that checks if a dapi image and a labels image somewhat match?
def dapi_and_segmentations_match():
    """TODO
    
    - segmentations capture a similar image range as the strong dapi signal (i had cases where the segmentation 
      was only a small part of the dapi image since I didn't scale the polygon coordinates)
    - check for reasonable overlap between dapi signal and segmentation: if there is a systematic offset, 
      the segmentation might be shifted
    
    """