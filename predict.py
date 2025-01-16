import numpy as np
import os
import re
import tensorflow as tf
import utils as u

from tensorflow.keras import layers, models
from network import *

# Creation list of folders with seismic cube for prediction
test_data = u.get_folders(name_test_dataset)

# Reading testing file for predicting
seismic_test = u.rescale_volume(u.get_cubes(name_test_dataset, test_data[4], 'seismic'), low= low_clip, high= high_clip)

# Getting prediction on test data
fault_predicted = np.zeros((300, 300, 1259))

# Predict on test
for i in range(0, 300, 100):
    for j in range(0, 300, 100):
        # Test
        predict = model.predict(np.reshape(seismic_test[i:i+100, j:j+100, :], (1, 100, 100, 1259, 1)))
        predict = np.squeeze(predict)
        fault_predicted[i:i+100, j:j+100, :] = predict

# Creation name files
name_predicted = os.listdir(os.path.join(name_test_dataset, test_data[4]))
name_predicted = re.sub(r'seismicCubes_RFC_fullstack_', 'fault_segments_', name_predicted[0])
name_predicted = re.sub(r'.npy', '', name_predicted)
name_predicted

# Creation of .npz file with predictions of faults
u.create_submission(sample_id=test_data[4], 
                    prediction=fault_predicted, 
                    submission_path=name_predicted, 
                    append=False)
