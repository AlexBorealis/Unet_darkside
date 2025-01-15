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


def get_dice(gt_mask, pred_mask):
    # masks should be binary
    # DICE Score = (2 * Intersection) / (Area of Set A + Area of Set B)
    intersect = np.sum(pred_mask * gt_mask)
    total_sum = np.sum(pred_mask) + np.sum(gt_mask)
    if total_sum == 0:  # both samples are without positive masks
        dice = 1.0
    else:
        dice = (2 * intersect) / total_sum
    return dice


def get_submission_score(
        gt_submission_path, prediction_submission_path, mask_shape=(300, 300, 1259)
):
    # load submissions
    gt_submission = dict(np.load(gt_submission_path))
    prediction_submission = dict(np.load(prediction_submission_path))

    # prepare place to store per sample score
    global_scores = []
    for sample_id in gt_submission.keys():
        # reconstruct gt mask
        gt_mask = np.zeros(mask_shape)
        gt_coordinates = gt_submission[sample_id]
        if gt_coordinates.shape[0] > 0:
            gt_mask[
                gt_coordinates[:, 0], gt_coordinates[:, 1], gt_coordinates[:, 2]
            ] = 1

        # reconstruct prediction mask
        pred_mask = np.zeros(mask_shape)
        pred_coordinates = prediction_submission[sample_id]
        if pred_coordinates.shape[0] > 0:
            pred_mask[
                pred_coordinates[:, 0], pred_coordinates[:, 1], pred_coordinates[:, 2]
            ] = 1

        global_scores.append(get_dice(gt_mask, pred_mask))

    sub_score = sum(global_scores) / len(global_scores)

    return sub_score
