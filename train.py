import gc
import numpy as np
import os
import tensorflow as tf
import utils as u

from tensorflow.keras import layers, models
from hyperparameters import *
from network import *

# Setting directory
os.chdir('/workspace/darkside_test')

# Creation list of folders with cubes for training
training_data = u.get_folders(name_training_dataset)

for td in training_data:
    seismic_train = u.get_cubes(name_training_dataset, td, 'seismic')
    fault_train = u.get_cubes(name_training_dataset, td, 'fault')

    print("\n Quantity marks of faults:", np.sum(fault_train))    

    # Train models
    for i in range(0, 300, 100):
        for j in range(0, 300, 100):
            # Train
            model.fit(x= np.reshape(u.rescale_volume(seismic_train, low= 5, high= 95)[i:i+100, j:j+100, :], (1, 100, 100, 1259, 1)),
                      y= np.reshape(fault_train.astype('float32')[i:i+100, j:j+100, :], (1, 100, 100, 1259, 1)),
                      batch_size= 256,
                      epochs= 200,
                      callbacks= [model_checkpoint, tensorboard, early_stopping, csv_logger],
                      verbose= 0)

            models = [m for m in os.listdir('model/') if 'best_unet' in m]
            try:
                filtered_models = [
                    m for m in models 
                    if float(re.findall(r'\d+\.\d+', m)[0]) < .99 and float(re.findall(r'\d+\.\d+', m)[1]) <= .95
                ]
                filtered_models.sort(key=lambda i: (-float(re.findall(r'\d.\d+', i)[1]), float(re.findall(r'\d.\d+', i)[0])))
                
                model = tf.keras.models.load_model(os.path.join('model/', filtered_models[0]))
                print(filtered_models[0])
            except:
                print("Default model")                

    gc.collect()