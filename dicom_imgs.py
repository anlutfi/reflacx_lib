import numpy as np
import pydicom
from psutil import virtual_memory


class DicomImgs:
    """Loads DICOM images to memory. Since more than one REFLACX datapoint
    can have the same MIMIC-CXR dicom_id, this class prevents loading the same
    one more than once.
    Loaded images occupy at most a fixed percentage of available virtual memory.
    When exceeding limit, last accessed images will be unloaded first"""
        
    def __init__(self, max_ram_percent=60):
        """param:max_ram_percent sets the maximum consumption of virtual memory
        by the images. It calculates a constant limit based on the total free
        memory reported by psutil.virtual_memory at instantiation
        """
        assert 0 < max_ram_percent <= 100
        self.imgs = {}
        self.max_ram_usage = int(virtual_memory().free * max_ram_percent / 100)
        self.ram_usage = 0
        self.last_accessed = []
    
    def check_id(self, dicom_id):
        return dicom_id in self.imgs
    
    
    def get_dicom_img(self, dicom_id, imgpath=None):
        assert dicom_id in self.imgs or imgpath is not None
        if dicom_id not in self.imgs:
            img = pydicom.read_file(imgpath).pixel_array
            self.imgs[dicom_id] = img
            try:
                i = self.last_accessed.index(dicom_id)
                self.last_accessed.pop(i)
            except ValueError:
                pass

            self.last_accessed.insert(0, dicom_id)

            self.ram_usage += img.size
            while self.ram_usage > self.max_ram_usage:
                d_id = self.last_accessed.pop()
                self.ram_usage -= self.imgs[d_id].size
                self.imgs.pop(d_id)
            
        return np.copy(self.imgs[dicom_id])