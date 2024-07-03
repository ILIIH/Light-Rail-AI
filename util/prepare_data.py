#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.image import resize_with_crop_or_pad
from util.const import IMAGE_SHAPE

# Constants
NEW_WIDTH = 500

def scale_image_data(img):
    """Scales the image data to the required dimensions and converts it to an array."""
    new_height = img.height * NEW_WIDTH // img.width
    img = img.resize((NEW_WIDTH, new_height), Image.Resampling.LANCZOS)
    img = img.convert('RGB')
    pixel_data = np.array(img)
    img_resized = resize_with_crop_or_pad(pixel_data, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
    array_img = img_to_array(img_resized)
    return array_img

def read_images_dataset(folder_path, data_specification_path):
    """Reads images and their specifications, returning a DataFrame with image data and labels."""
    columns = ['Data', 'Is Melanoma', 'Is not Melanoma', 'Is not Mole']
    image_byte_arrays = pd.DataFrame(columns=columns)
    data_specification = pd.read_csv(data_specification_path, usecols=['file', 'b|m'])

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path)
                byte_data = scale_image_data(img)

                filename_no_ext = os.path.splitext(filename)[0]
                file_specification = data_specification[data_specification['file'] == filename_no_ext]

                if not file_specification.empty:
                    is_melanoma = int(file_specification['b|m'].values[0] == 'malignant')
                    is_not_melanoma = int(file_specification['b|m'].values[0] == 'benign')
                    is_not_mole = True  # Assuming this is always True based on the given logic

                    new_row = pd.DataFrame([{
                        'Data': byte_data,
                        'Is Melanoma': is_melanoma,
                        'Is not Melanoma': is_not_melanoma,
                        'Is not Mole': is_not_mole
                    }])
                    image_byte_arrays = pd.concat([image_byte_arrays, new_row], ignore_index=True)

            except IOError:
                print(f"Unable to open image file {filename}")

    return image_byte_arrays

