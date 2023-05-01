# code retrieved from chatGPT
from PIL import Image
import numpy as np
import os

def is_binary_png(image_path):
    # Open image
    img = Image.open(image_path).convert('L')  # convert to grayscale

    # Convert image to numpy array
    img_array = np.array(img)

    # Check if image is binary
    unique_values = np.unique(img_array)
    if len(unique_values) <= 2:
        return True
    else:
        return False
    
#is_binary = is_binary_png("augmented_data/new_data_111_1111_512/test/mask/IMG_0163_0.png")
#print(is_binary)


def binarize_image(image_path, threshold):
    # Open image and convert to grayscale
    img = Image.open(image_path).convert('L')

    # Binarize image
    bin_img = img.point(lambda x: 0 if x < threshold else 255, '1')

   

    bin_img.save(image_path)

    return bin_img

def binarize_folder(mask_folder, threshold):
    # Iterate over all images in the folder
    for image_name in os.listdir(mask_folder):
        image_path = mask_folder + "/" + image_name
        binarize_image(image_path, threshold)

def all_images_binary(mask_folder):
    # Iterate over all images in the folder
    for image_name in os.listdir(mask_folder):
        image_path = mask_folder + "/" + image_name
        if not is_binary_png(image_path):
            return False
    return True



