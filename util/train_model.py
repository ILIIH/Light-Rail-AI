import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

from util.const import ACTIVATION_OUTPUT, BATCH_SIZE, CONV_LAYER_PARAMS, DENSE_LAYER_SIZE, EPOCHS, IMAGE_SHAPE, LOG_DIR, LOSS_FUNCTION, METRICS, NUM_CLASSES, OPTIMIZER
from util.save_model import get_weight_callback



def train_model(X_train, Y_train):
    # Data preprocessing
    X = np.stack(X_train, axis=0).astype(np.float32)
    Y = np.array(Y_train, dtype=np.float32)
    
    # Model architecture
    model = Sequential()
    model.add(Conv2D(CONV_LAYER_PARAMS[0]['filters'], CONV_LAYER_PARAMS[0]['kernel_size'], 
                     activation=CONV_LAYER_PARAMS[0]['activation'], input_shape=IMAGE_SHAPE))
    model.add(MaxPooling2D())

    for params in CONV_LAYER_PARAMS[1:]:
        model.add(Conv2D(params['filters'], params['kernel_size'], activation=params['activation']))
        model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(DENSE_LAYER_SIZE, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation=ACTIVATION_OUTPUT))

    # Compile the model
    model.compile(optimizer=OPTIMIZER, loss=LOSS_FUNCTION, metrics=METRICS)

    # Callbacks
    tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
    checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
    weight_callback = get_weight_callback(X.size, BATCH_SIZE)

    # Model training
    model.fit(x=X, y=Y, batch_size=BATCH_SIZE, epochs=EPOCHS, 
              callbacks=[tensorboard_callback, checkpoint_callback, weight_callback], 
              validation_split=0.2)

    return model
