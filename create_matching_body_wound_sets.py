import os
import random
import shutil
from config_body_detection import *
import cv2

'''
This script aims to create matching datasets for the body detection dataset and the wound dataset.
This is to ensure that either of the two models have never seen the test images before.
'''

''' Set image and mask paths for the body detection dataset '''
image_folder = 'C:/Users/ingvilrh/master_data/fish_bodies'
mask_folder = 'remove_background/body_masks/'

''' Set the paths for the train, val and test data directories  for the wound dataset '''
#TODO: make all paths work together and make sure you can set them only one place, preferably in the config file
data_path ='C:/Users/ingvilrh/master_data/dataset_111_1111/train_images'
mask_path = 'C:/Users/ingvilrh/master_data/dataset_111_1111/train_masks'
val_path = 'C:/Users/ingvilrh/master_data/dataset_111_1111/val_images'
val_truth = 'C:/Users/ingvilrh/master_data/dataset_111_1111/val_masks'
test_path = 'C:/Users/ingvilrh/master_data/dataset_111_1111/test_images'
test_truth = 'C:/Users/ingvilrh/master_data/dataset_111_1111/test_masks'


''' Set the paths for the train, val and test data directories for the body detection dataset '''
train_image_folder = 'remove_background/train_images'
train_mask_folder = 'remove_background/train_masks'
val_image_folder = 'remove_background/val_images'
val_mask_folder = 'remove_background/val_masks'
test_image_folder = 'remove_background/test_images'
test_mask_folder = 'remove_background/test_masks'

''' Ensure the paths exist '''
create_dir(train_image_folder)
create_dir(train_mask_folder)
create_dir(val_image_folder)
create_dir(val_mask_folder)
create_dir(test_image_folder)
create_dir(test_mask_folder)

''' 
Move the images and masks to the correct folders 
INPUT:
    body_images : path to the folder containing all original images of fish
    wound_dataset : train, test or validation dataset which you are trying to find matches for
    destination_body_dataset : train, test or validation dataset which you are moving the images to

'''
def move_matching_images(wound_dataset, body_images, destination_body_dataset):
    for image_name in os.listdir(wound_dataset):
        print(image_name.split('.')[0])
        img_ID = image_name.split('.')[0]
        for image in os.listdir(body_images):
            if img_ID in image:
                print(image, 'is a match')
                shutil.copy(os.path.join(body_images, image), os.path.join(destination_body_dataset, image))
        
''' 
Move the images and masks to the correct folders 
INPUT:
    wound_dataset : train, test or validation dataset which you are trying to find matches for
    body_masks : path to the folder containing all original masks of fish body
    destination_body_dataset : train, test or validation dataset which you are moving the images to

'''

def move_matching_masks(wound_dataset, body_masks, destination_body_dataset):
    for image_name in os.listdir(wound_dataset):
        print(image_name.split('.')[0])
        img_ID = image_name.split('.')[0]
        for image in os.listdir(body_masks):
            if img_ID in image:
                print(image, 'is a match')
                shutil.copy(os.path.join(body_masks, image), os.path.join(destination_body_dataset, image))

# for name in os.listdir(train_image_folder):
#     if "mask" in name:
#         print(name)
#         os.remove(os.path.join(train_image_folder, name))


#move_matching_images(test_path, image_folder, test_image_folder)
#move_matching_masks(test_truth, mask_folder, test_mask_folder)
    