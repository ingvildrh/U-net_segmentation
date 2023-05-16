import os
import shutil
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from imutils import paths
from create_datasets_body_detection import *
from config_body_detection import *
'''
This program takes paths to 
train images: 
ground truth masks for training:

validation images:
ground truth masks for validation:

test images:
ground truth masks for test:

and augment the training images and masks with 3 methods: horizontal flip, vertical flip and rotation.

The images is saved to a new folder with the name of the dataset and the size of the images that was created. 
'''
BODY_DETECTION_DATASET = 'body_detection_dataset'

''' 
Create a directory 
INPUT:
    path : path to the directory you want to create
'''
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


'''
Loads data from paths 
INPUT: 
    path_img : paths to the training images
    path_mask : path to the corresponding training masks
    path_val_img : paths to the validation images
    path_val_mask : path to the corresponding validation masks
    path_test_img : paths to the test images
    path_test_mask : path to the corresponding test masks
OUTPUT: 
    (train_x, train_y) : tuple of paths to the training images and masks
    (val_x, val_y) : tuple of paths to the validation images and masks
    (test_x, test_y) : tuple of paths to the test images and masks
'''
def load_data(path_img, path_mask, path_val_img, path_val_mask, path_test_img, path_test_mask):
    train_x = sorted(list(paths.list_images(path_img)))
    train_y = sorted(list(paths.list_images(path_mask)))

    val_x = sorted(list(paths.list_images(path_val_img)))
    val_y = sorted(list(paths.list_images(path_val_mask)))

    test_x = sorted(list(paths.list_images(path_test_img)))
    test_y = sorted(list(paths.list_images(path_test_mask)))

    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


'''
Augment the images data and the corresponding mask label data with 3 methods and save them to a different folders for training.
INPUT:
    images : paths to the images to augment
    masks : paths to the corresponding masks to augment
    save_path : path to save the augmented images and masks
    augment : boolean value to augment or not
'''
def augment_data(images, masks, save_path, augment=True):
    size = (H, W)

    for idx, (x,y) in tqdm(enumerate(zip(images, masks)), total = len(images)):
        """ Extracting the name """
        name = os.path.basename(x)
        name = name.split(".")[0]
        
        """ reading image and mask"""
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        
        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit = 45, p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X,Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            tmp_image_name = name + "_" + str(index) + ".png"
            tmp_mask_name = name + "_" + str(index) + ".png"

            image_path = save_path + "image/" + tmp_image_name
            mask_path = save_path + "mask/" + tmp_mask_name


            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)
            

            index += 1

'''
Empty the augmented data folder
'''
def empty_augmented_data():
    if os.path.exists(BODY_DETECTION_DATASET + "/train/"):
        shutil.rmtree(BODY_DETECTION_DATASET + "/train/")
    if os.path.exists(BODY_DETECTION_DATASET + "/val/"):
        shutil.rmtree(BODY_DETECTION_DATASET + "/val/")
    if os.path.exists(BODY_DETECTION_DATASET + "/test/"):
        shutil.rmtree(BODY_DETECTION_DATASET + "/test/")
   
def main():
    empty_augmented_data()

    """ Seeding """
    np.random.seed(42)

    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(train_image_folder, train_mask_folder, val_image_folder, val_mask_folder, test_image_folder, test_mask_folder)

    print("Train: ")
    print(len(train_x), len(train_y))
    print("Validation: ")
    print(len(val_x), len(val_y))
    print("Test: ")
    print(len(test_x), len(test_y))

    ''' Create directories to save the augmented data '''
    # create_dir(train_images)
    # create_dir(train_masks)
    # create_dir(val_images)
    # create_dir(val_masks)
    # create_dir(test_images)
    # create_dir(test_masks)

    create_dir(BODY_DETECTION_DATASET + "/train/image")
    create_dir(BODY_DETECTION_DATASET + "/train/mask")
    create_dir(BODY_DETECTION_DATASET + "/test/image")
    create_dir(BODY_DETECTION_DATASET + "/test/mask")
    create_dir(BODY_DETECTION_DATASET + "/val/image")
    create_dir(BODY_DETECTION_DATASET + "/val/mask")

    """ Data augmentation"""
    augment_data(train_x, train_y, BODY_DETECTION_DATASET + "/train/", augment=True)
    augment_data(val_x, val_y, BODY_DETECTION_DATASET  + "/val/", augment=False)
    augment_data(test_x, test_y, BODY_DETECTION_DATASET  + "/test/", augment=False)

if __name__ == "__main__":
    main()