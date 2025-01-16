import os
import time
import random
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import xarray as xr
import scipy
from typing import Union, Tuple
import regex as re

def get_folders(name_dataset):
    """
    Getting names folders with data cubes
    """    
    # Directory path
    data = os.path.join(
        os.getcwd(), name_dataset
    )

    # List to store the names of subfolders (sample IDs)
    sample_ids = []

    # Iterate over the items in the directory
    for item in os.listdir(data):
        item_path = os.path.join(data, item)
        if os.path.isdir(item_path):
            sample_ids.append(item)

    return sample_ids

def get_cubes(name_dataset, data, pattern):
    """
    Getting data cubes
    """    
    # Path to the subfolder/sample
    sample_path = os.path.join(name_dataset, data)

    # List and print all files in the sample subfolder
    files = os.listdir(sample_path)

    # Iterate over the files and load the .npy files.
    for file in files:
        if file.startswith(pattern) and file.endswith(".npy"):
            data_ = np.load(os.path.join(sample_path, file), allow_pickle= True)
            
    return data_

def rescale_volume(volume, low= 0, high= 100):
    """
    Rescaling 3D seismic volumes 0-255 range, clipping values between low and high percentiles
    """
    minval = np.percentile(volume, low)
    maxval = np.percentile(volume, high)
    volume = np.clip(volume, minval, maxval)
    volume = ((volume - minval) / (maxval - minval)) * 255

    return volume


def create_submission(
        sample_id: str, prediction: np.ndarray, submission_path: str, append: bool = True
):
    """Function to create submission file out of one test prediction at time

    Parameters:
        sample_id: id of survey used for perdiction
        prediction: binary 3D np.ndarray of predicted faults
        submission_path: path to save submission
        append: whether to append prediction to existing .npz or create new one

    Returns:
        None
    """

    if append:
        try:
            submission = dict(np.load(submission_path))
        except:
            print("File not found, new submission will be created.")
            submission = dict({})
    else:
        submission = dict({})

    # Positive value coordinates
    coordinates = np.stack(np.where(prediction > 0)).T
    coordinates = coordinates.astype(np.uint16)

    submission.update(dict([[sample_id, coordinates]]))

    np.savez(submission_path, **submission)
