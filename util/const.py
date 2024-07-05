import tensorflow as tf


DATASET_IMAGES_PATH = './dataset/ISIC_Data_Images_Descriptions/Data/Images'
DATASET_SPECIFICATION_PATH = './dataset/data.csv'

# Model Constants
NUM_CLASSES = 3
CONV_LAYER_PARAMS = [
    {'filters': 16, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'filters': 16, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'filters': 16, 'kernel_size': (3, 3), 'activation': 'reslu'}
]
DENSE_LAYER_SIZE = 256
ACTIVATION_OUTPUT = 'sigmoid'
LOSS_FUNCTION = tf.keras.losses.BinaryCrossentropy()
OPTIMIZER = 'adam'
METRICS = ['accuracy']
LOG_DIR = 'logs'
EPOCHS = 20

IMAGE_SHAPE = (333, 500, 3)
BATCH_SIZE =32