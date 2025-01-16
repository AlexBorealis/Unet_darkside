import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import dice
from tensorflow.keras.metrics import IoU

# Function for a creation network
def get_model(img_size, num_classes):
    inputs = tf.keras.layers.Input(shape=img_size + (1,), name='unet_model')

    x = tf.keras.layers.Conv3D(16, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("leaky_relu")(x)

    x1 = tf.keras.layers.Conv3D(32, 3, padding="same")(x)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation("leaky_relu")(x1)
    
    x2 = tf.keras.layers.Conv3D(64, 3, padding="same")(x1)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.MaxPooling3D(2, padding="same")(x2)
    
    residual1 = tf.keras.layers.Conv3D(64, 3, padding="same", strides=2)(x)
    residual1 = tf.keras.layers.BatchNormalization()(residual1)
    residual1 = tf.keras.layers.Activation("leaky_relu")(residual1)
    x2 = tf.keras.layers.add([x2, residual1])

    x3 = tf.keras.layers.Conv3D(128, 3, padding="same")(x2)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.MaxPooling3D(2, padding="same")(x3)

    residual2 = tf.keras.layers.Conv3D(128, 3, padding="same", strides=2)(x2)
    residual2 = tf.keras.layers.BatchNormalization()(residual2)
    residual2 = tf.keras.layers.Activation("leaky_relu")(residual2)
    x3 = tf.keras.layers.add([x3, residual2])

    x4 = tf.keras.layers.Conv3D(256, 3, padding="same")(x3)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.MaxPooling3D(2, padding="same")(x4)

    residual3 = tf.keras.layers.Conv3D(256, 3, padding="same", strides=2)(x3)
    residual3 = tf.keras.layers.BatchNormalization()(residual3)
    residual3 = tf.keras.layers.Activation("leaky_relu")(residual3)
    x4 = tf.keras.layers.add([x4, residual3])

    x5 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x4)
    x5 = tf.keras.layers.Conv3D(256, 3, padding="same")(x5)
    x5 = tf.keras.layers.BatchNormalization()(x5)
    x5 = tf.keras.layers.Activation("leaky_relu")(x5)

    x6 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x5)
    x6 = tf.keras.layers.Conv3D(128, 3, padding="same")(x6)
    x6 = tf.keras.layers.BatchNormalization()(x6)
    x6 = tf.keras.layers.Activation("leaky_relu")(x6)

    x7 = tf.keras.layers.UpSampling3D(size=(2, 2, 2))(x6)
    x7 = tf.keras.layers.Conv3D(64, 3, padding="same")(x7)
    x7 = tf.keras.layers.BatchNormalization()(x7)
    x7 = tf.keras.layers.Activation("leaky_relu")(x7)

    x8 = tf.keras.layers.Conv3D(32, 3, padding="same")(x7)
    x8 = tf.keras.layers.BatchNormalization()(x8)
    x8 = tf.keras.layers.Activation("leaky_relu")(x8)

    x9 = tf.keras.layers.Conv3D(16, 3, padding="same")(x8)
    x9 = tf.keras.layers.BatchNormalization()(x9)
    x9 = tf.keras.layers.Activation("leaky_relu")(x9)

    x10 = tf.keras.layers.Cropping3D(cropping=((2, 2), (2, 2), (2, 3)))(x9)

    outputs = tf.keras.layers.Conv3D(num_classes, 3, activation="sigmoid", padding="same")(x10)

    model = tf.keras.Model(inputs, outputs, name='unet_model')
    return model

# Creation "model/" directory
if not os.path.exists('model/'):
    os.mkdir('model/')

# Testing if model's exists
if os.path.exists('model/') and len(os.listdir('model/')) > 0:
    models = [m for m in os.listdir('model/') if 'best_unet' in m]
    try:
        filtered_models = [
                    m for m in models 
                    if float(re.findall(r'\d+\.\d+', m)[0]) < dice_level and float(re.findall(r'\d+\.\d+', m)[1]) <= iou_level
                ]
        filtered_models.sort(key=lambda i: (-float(re.findall(r'\d.\d+', i)[1]), float(re.findall(r'\d.\d+', i)[0])))
        model = tf.keras.models.load_model(os.path.join('model/', filtered_models[0]))
        print(filtered_models[0])
    except:
        print("No suitable models found.")
        
else:
    model = get_model((100, 100, 1259), 1)
    
    # Model compilation
    model.compile(optimizer= SGD(nesterov= True, momentum= .1, learning_rate= .005), 
                  loss= dice,
                  metrics= [tf.metrics.IoU(num_classes= 2, target_class_ids= [0, 1])])
    
# Creation model summary  
model.summary(show_trainable= True, expand_nested= True)
