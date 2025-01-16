import os
import warnings

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger

# Setting directory
os.chdir('/workspace/darkside_test')

# Suppressed warnings
warnings.filterwarnings('ignore')

# Names datasets
name_training_dataset = "dark-side-train-data-part3"
name_test_dataset = "dark-side-test-data-part-3"

# Levels of clip
low_clip = 5
high_clip = 95

# Levels of loss and metric
iou_level = .95
dice_level = .99

# Callbacks variables
tensorboard_logs_name = './logs_' + name_training_dataset
csv_logger_name = 'training_log_unet_' + name_training_dataset + '.csv'

# Callbacks
model_checkpoint = ModelCheckpoint('./model/best_unet_model.{loss:.2f}.h5.keras')
tensorboard = TensorBoard(log_dir= tensorboard_logs_name)
reduce_on_plateau = ReduceLROnPlateau(monitor= 'loss', patience= 5)
early_stopping = EarlyStopping(monitor= "loss", patience= 50, start_from_epoch= 50, min_delta= .01, verbose= 1)
csv_logger = CSVLogger(csv_logger_name, separator= ",", append= True)
