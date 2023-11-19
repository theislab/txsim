
import tifffile
import xml.etree.ElementTree as ET


def get_image_shape(image_path):
    """
    """
    
    with tifffile.TiffFile(image_path) as tif:
        img_shape = tif.pages[0].shape
        
    return img_shape

def get_ome_schema(image_path):
    with tifffile.TiffFile(image_path) as tif:
        ome_xml = tif.ome_metadata
    ome_xml_root = ET.fromstring(ome_xml)

    # Extract the namespace associated with the OME tag
    ome_namespace = ome_xml_root.tag.split('}')[0].strip('{')

    return ome_namespace


def extract_physical_sizes(image_path):
    """ Extract physical sizes from OME-XML assuming schema version 2016-06.
    """
    
    schema2016 = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    
    assert str(image_path).endswith('.ome.tif'), 'Image must be an OME-TIFF.'
    schema = get_ome_schema(image_path)
    assert schema == schema2016, f"Unexpected schema: {schema}"
    
    with tifffile.TiffFile(image_path) as tif:
        ome_xml = tif.ome_metadata

    ome_xml_root = ET.fromstring(ome_xml)

    image_data = {}

    for image in ome_xml_root.findall('.//ns0:Image', namespaces={'ns0': schema2016}):
        image_name = image.attrib.get('Name', None)

        pixels_element = image.find('ns0:Pixels', namespaces={'ns0': schema2016})

        if pixels_element is not None:
            physical_size_x = pixels_element.attrib.get('PhysicalSizeX', None)
            physical_size_y = pixels_element.attrib.get('PhysicalSizeY', None)
            physical_size_z = pixels_element.attrib.get('PhysicalSizeZ', None)

            image_data[image_name] = {
                'PhysicalSizeX': physical_size_x,
                'PhysicalSizeY': physical_size_y,
                'PhysicalSizeZ': physical_size_z
            }

    return image_data


