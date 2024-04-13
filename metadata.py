import os
from tools import csv2dictlist
import json
import numpy as np
from random import randint

from reflacx_sample import ReflacxSample
from dicom_imgs import DicomImgs


class Metadata:
    def __init__(self,
                 reflacx_dir,
                 mimic_dir,
                 full_meta_path,
                 reflacx_main_data_dir='main_data',
                 heatmaps_search_term='heatmaps_phase_',
                 metadata_search_term='metadata',
                 exclude_invalid_eyetracking=True,
                 max_dicom_lib_ram_percent=60):
        
        self.imgs_lib = DicomImgs(max_ram_percent=max_dicom_lib_ram_percent)
        
        print("loading metadata")
        if os.path.exists(full_meta_path):
            with open(full_meta_path) as f:
                self.metadata = json.load(f)
            print("metadata loaded from file")
            return
        
        print("file not found, generating metadata from reflacx and mimic. This will take about 20 min.")
        
        main_data_dir = "{}{}{}".format(reflacx_dir,
                                        os.sep,
                                        reflacx_main_data_dir)
        reflacx_metadata = []
        for metadata_file in [item
                            for item in os.listdir(main_data_dir)
                            if metadata_search_term in item]:
            reflacx_metadata += csv2dictlist("{}{}{}".format(main_data_dir,
                                                                os.sep,
                                                                metadata_file))
        
        dicom_metadata = {}
        
        print("grouping reflacx metadata by dicom_id")
        for item in reflacx_metadata:
            if (item.pop('eye_tracking_data_discarded') in ['TRUE', 'True', 'true']
                and exclude_invalid_eyetracking):
                continue

            dicom_id = item.pop('dicom_id')
            if dicom_id not in dicom_metadata:
                dicom_metadata[dicom_id] = {}
            
            id = item.pop('id')
            if id not in dicom_metadata[dicom_id]:
                dicom_metadata[dicom_id][id] = {}
            
            item['image'] = "{}{}{}.dcm".format(mimic_dir, os.sep, dicom_id)
            eyetracking_path = "{}{}{}{}{}".format(reflacx_dir,
                                                os.sep,
                                                reflacx_main_data_dir,
                                                os.sep,
                                                id)
            for eyetracking_file in os.listdir(eyetracking_path):
                item[eyetracking_file.split('.')[0]] = "{}{}{}".format(eyetracking_path,
                                                                    os.sep,
                                                                    eyetracking_file)
            dicom_metadata[dicom_id][id] = item

        print("grouping heatmaps")
        for dir in [item for item in os.listdir(reflacx_dir) if heatmaps_search_term in item]:
            path = "{}{}{}".format(reflacx_dir, os.sep, dir)
            print("getting heatmaps from {}".format(path))
            for count, npy in enumerate(os.listdir(path)):
                if count % 100 == 0:
                    print("loading {}th npy".format(count))
                npy_path = "{}{}{}".format(path, os.sep, npy)
                heatmap = np.load(npy_path, allow_pickle=True).item()
                dicom_id = heatmap.pop('img_path').split('/')[-1].split('.')[0]
                id = heatmap.pop('id')
                dicom_metadata[dicom_id][id]['heatmaps'] = npy_path


        self.metadata = dicom_metadata

        with open(full_meta_path, 'w') as f:
            json.dump(self.metadata, f)
        print("done")

    
    def list_dicom_ids(self, n_samples=None, reverse=False, random_samples=False):
        if n_samples is None:
            n_samples = len(self.metadata)
        else:
            n_samples = min(n_samples, len(self.metadata))
        if not random_samples:
            result = (list(self.metadata.keys())[:n_samples]
                      if not reverse
                      else list(self.metadata.keys())[-n_samples:])
        else:
            keys = list(self.metadata.keys())
            result = [keys.pop(randint(0, len(keys) - 1))
                      for i in range(n_samples)]
            
        return result
    
    
    def list_reflacx_ids(self, dicom_id):
        if dicom_id in self.metadata:
            return list(self.metadata[dicom_id].keys())
        return []
    

    def get_sample(self, dicom_id, reflacx_id):
        try:
            return ReflacxSample(dicom_id,
                                 reflacx_id,
                                 self.metadata[dicom_id][reflacx_id],
                                 imgs_lib=self.imgs_lib)
        except KeyError:
            #TODO log missing dicom_id, reflacx_id pair
            return None
