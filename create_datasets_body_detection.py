import os
import random
import shutil
from config_body_detection import *
import cv2

''' Set image and mask paths '''
image_folder = 'C:/Users/ingvilrh/master_data/fish_bodies'
mask_folder = 'remove_background/body_masks/'

''' Set the paths for the train, val and test data directories '''
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

''' Create a list of image and mask pairs '''
image_files = os.listdir(image_folder)
mask_files = os.listdir(mask_folder)
image_mask_pairs = list(zip(image_files, mask_files))

''' Shuffle the list of image and mask pairs'''
random.shuffle(image_mask_pairs)

''' Get the number of images '''
num_images = len(image_mask_pairs)

''' 
Function to find the number of images in each set 
INPUT: 
    train_percent : percentage of images in train set
    val_percent : percentage of images in val set
    test_percent : percentage of images in test set
    num_images : total number of images available
Output: 
    num_train : number of images in train set
    num_val : number of images in val set
    num_test : number of images in test set
'''
def find_set_sizes(train_percent, val_percent, test_percent, num_images):
    # train_percent = 0.7
    # val_percent = 0.2
    # test_percent = 0.1
    num_train = int(train_percent * num_images)
    num_val = int(val_percent * num_images)
    num_test = int(test_percent * num_images)
    num_test = num_test + (num_images - (num_train + num_val + num_test))
    return num_train, num_val, num_test
    
''' 
Function to check if each image has a mask
INPUT: 
    image_files : path to folder with images
    mask_files : path to folder with masks
OUTPUT: 
    True if each image has a mask, False if not
'''
def check_if_mask_exists(image_files, mask_files):
    for image_name in image_files:
        if image_name not in mask_files:
            print('Image {} has no mask'.format(image_name))
            return False
        return True

''' 
Function to create the train set
INPUT: 
    num_train : number of images in train set
'''

def create_train_set(num_train):
    train_pairs = image_mask_pairs[:num_train]
    for pair in train_pairs:
        image_name, mask_name = pair
        shutil.copy((image_folder + "/" + image_name), (train_image_folder + "/" + image_name))
        shutil.copy((mask_folder + "/" + mask_name), (train_mask_folder + "/" + mask_name))

''' 
Function to create the validation set
INPUT: 
    num_train : number of images in train set
    num_val : number of images in val set
'''
def create_val_set(num_train, num_val):
    val_pairs = image_mask_pairs[num_train:num_train + num_val]
    for pair in val_pairs:
        image_name, mask_name = pair
        shutil.copy((image_folder + "/" + image_name), (val_image_folder + "/" + image_name))
        shutil.copy((mask_folder + "/" + mask_name), (val_mask_folder + "/" + mask_name))

''' 
Function to create the test set 
INPUT: 
    num_train : number of images in train set
    num_val : number of images in val set
    num_test : number of images in test set
'''
def create_test_set(num_train, num_val, num_test):
    test_pairs = image_mask_pairs[num_train + num_val:]
    for pair in test_pairs:
        image_name, mask_name = pair
        shutil.copy((image_folder + "/" + image_name), (test_image_folder + "/" + image_name))
        shutil.copy((mask_folder + "/" + mask_name), (test_mask_folder + "/" + mask_name))

''' 
Function to empty the folders, must be done if wanted to shuffle the data over again or change the split
'''
def empty_folders():
    for folder in [train_image_folder, train_mask_folder, val_image_folder, val_mask_folder, test_image_folder, test_mask_folder]:
        for file in os.listdir(folder):
            os.remove(folder + "/" + file)

'''
Function to check for duplicates in the sets
INPUT: 
    train_image_folder : path to folder with train images
    val_image_folder : path to folder with val images
    test_image_folder : path to folder with test images
OUTPUT: 
    True if no duplicates, False if duplicates
'''
def check_duplicates_in_sets(train_image_folder, val_image_folder, test_image_folder):
    train_image_files = os.listdir(train_image_folder)
    
    val_image_files = os.listdir(val_image_folder)
   
    test_image_files = os.listdir(test_image_folder)
    
    for image_name in train_image_files:
        if image_name in val_image_files or image_name in test_image_files:
            print('Image {} is in more than one set'.format(image_name))
            return False
            

num_train, num_val, num_test = find_set_sizes(0.7, 0.2, 0.1, num_images)

if __name__ == '__main__':
    print("Number of images available for datasets: {}".format(num_images))
    empty_folders()
    print("Train set size: {} - Validation set size: {} - Test set size: {}".format(num_train, num_val, num_test))
    check_if_mask_exists(image_files, mask_files)
    create_train_set(num_train)
    create_val_set(num_train, num_val)
    create_test_set(num_train, num_val, num_test)
    check_duplicates_in_sets(train_image_folder, val_image_folder, test_image_folder)