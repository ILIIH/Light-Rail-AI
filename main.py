#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from util.const import DATASET_IMAGES_PATH, DATASET_SPECIFICATION_PATH
from util.prepare_data import read_images_dataset
from util.save_model import save_model
from util.train_model import train_model



def train(dataset):
    end_of_train_dataset = int(dataset.size - dataset.size / 6)

    train_set = dataset.iloc[:end_of_train_dataset]
    validation_set = dataset.iloc[end_of_train_dataset:]

    X_train = train_set.pop('Data').values
    Y_train = train_set.values  

    model = train_model(X_train, Y_train)    
    print("MODEL FULLY TRAINED")
    model.save(CHECKPOINT_DIR, save_format='tf')
    save_model(model)

def main():
    image_byte_arrays = read_images_dataset(DATASET_IMAGES_PATH, DATASET_SPECIFICATION_PATH)
    train(image_byte_arrays)

if __name__ == "__main__":
    main()
