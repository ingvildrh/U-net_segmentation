import os
from convert_pixel_values import *
import shutil


def split_masks_and_images(folder):
    if not os.path.exists('body_masks/'):
        os.makedirs('body_masks/')
    for image_name in os.listdir(folder):
            if 'mask' in image_name:
                image_path = folder + "/" + image_name
                shutil.move(image_path, 'body_masks/')
                


split_masks_and_images('C:/Users/ingvilrh/master_data/fish_bodies/')