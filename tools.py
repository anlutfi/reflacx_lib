import os
import numpy as np
import pandas as pd

def csv2dictlist(csv_file):
    """generate a list of dictionaries from a csv with a header for first line"""
    csv = pd.read_csv(csv_file)
    return [dict(row[1]) for row in csv.iterrows()]


def normalize(img, value_range=(0, 1), type=float, by_channel=False):
    """returns an image normalized in a given range and type
    if :param by_channel: is True, normalizes each color channel separately
    """
    result = img.astype(float)
    if len(img.shape) == 3 and by_channel:
        for i in range(result.shape[2]):
            result[:,:,i] = normalize(result[:, :, i], value_range)
    else:
        result -= np.min(img)
        imax = np.max(result)
        result /= imax if imax != 0 else 1
        result = (result * (value_range[1] - value_range[0]) + value_range[0])
    
    return result.astype(type)