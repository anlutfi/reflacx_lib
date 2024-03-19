import numpy as np
import pydicom


class DicomImgs:
    """Loads DICOM images to memory. Since more than one REFLACX datapoint
    can have the same MIMIC-CXR dicom_id, this class prevents loading the same
    one more than once.
    In the future, there will be unloading of images to free up memory"""
    # TODO don't be a liar in the docstring
    _imgs = {}
    _features = {}
    _feature_extractor = None
    
    @staticmethod
    def check_id(dicom_id):
        return dicom_id in DicomImgs._imgs
    
    
    @staticmethod
    def get_dicom_img(dicom_id, imgpath=None):
        assert dicom_id in DicomImgs._imgs or imgpath is not None
        if dicom_id not in DicomImgs._imgs:
            DicomImgs._imgs[dicom_id] = pydicom.read_file(imgpath).pixel_array
        return np.copy(DicomImgs._imgs[dicom_id])