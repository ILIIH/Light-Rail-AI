 #!/usr/bin/env python3
 # -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from util.const import DATASET_IMAGES_PATH, DATASET_SPECIFICATION_PATH
from util.prepare_data import read_images_dataset
from util.save_model import save_model
from util.train_model import train_model

def test(X_train, Y_train):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1000,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=10, batch_size=32)
    return model

def train(dataset): 
    end_of_train_dataset = int( dataset.size - dataset.size/6 ) 

    train_set = dataset.iloc[:end_of_train_dataset]  
    validation_set = dataset.iloc[end_of_train_dataset:] 

    X_train = train_set.pop('Data')
    Y_train = train_set

    model = train_model(X_train,Y_train)
    save_model(model)


def main():
    image_byte_arrays =  read_images_dataset(DATASET_IMAGES_PATH, DATASET_SPECIFICATION_PATH)
    train(image_byte_arrays)

if __name__ == "__main__":
    main()


