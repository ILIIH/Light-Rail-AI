import io
import os
from PIL import Image

def read_image(folder_path): 
    image_byte_arrays = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                byte_data = img_byte_arr.getvalue()
                image_byte_arrays.append(byte_data)
            except IOError:
                print(f"Unable to open image file {filename}")

    print(f"Total images read: {len(image_byte_arrays)}")