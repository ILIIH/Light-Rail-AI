import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from MolesRecognision.const import DATASET_IMAGES_PATH
from MolesRecognision.prepare_data import read_images_dataset


def main():
    image_byte_arrays =  read_images_dataset(DATASET_IMAGES_PATH)
    print(image_byte_arrays[0])

if __name__ == "__main__":
    main()
