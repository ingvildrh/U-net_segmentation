import os
import shutil
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from imutils import paths

'''
This program loads mask labels and images from a train directoy and a test directory and annotate them
'''
DATASET = '113_0329'

''' Set the data paths '''
data_path = "C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/DATA_SETS/dataset_" + DATASET +  "/train_images"
mask_path = "C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/DATA_SETS/dataset_" + DATASET +  "/train_masks"
test_path = "C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/DATA_SETS/dataset_" + DATASET +  "/val_images"
ground_truth = "C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/DATA_SETS/dataset_" + DATASET +  "/val_masks"



''' Set the paths for the augmented data '''
AUGMENTED_DATA_BASE_PATH = 'new_data_' + DATASET + "/"

train_images =  AUGMENTED_DATA_BASE_PATH + 'train/image/'
train_masks = AUGMENTED_DATA_BASE_PATH + "train/mask/"
test_images = AUGMENTED_DATA_BASE_PATH + "test/image/"
test_masks = AUGMENTED_DATA_BASE_PATH + "test/mask/"

''' 
Create a directory 
'''
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


'''
Loads data from paths 
Input: paths to images, masks, test images and tests ground truths
Output: lits of file paths for all train_x, train_y, test_x, test_y
'''
def load_data(path_img, path_mask, path_test_img, path_ground_truth):
    train_x = sorted(list(paths.list_images(path_img)))
    train_y = sorted(list(paths.list_images(path_mask)))

    test_x = sorted(list(paths.list_images(path_test_img)))
    test_y = sorted(list(paths.list_images(path_ground_truth)))

    return (train_x, train_y), (test_x, test_y)


'''
Augment the images data and the corresponding mask label data with 3 methods and save them to a different folders for training.
Test data is not annotated. 
Input: images to annotate, corresponding masks to annotate, path for saving of annotations, augment=True
'''
def augment_data(images, masks, save_path, augment=True):
    size = (512, 512)

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

   
   
        
def main():

    """ Seeding """
    np.random.seed(42)
    
    (train_x, train_y), (test_x, test_y) = load_data(data_path, mask_path, test_path, ground_truth)

    print("Train: ")
    print(len(train_x), len(train_y))
    print("Test: ")
    print(len(test_x), len(test_y))

    ''' Create directories to save the augmented data '''
    create_dir(train_images)
    create_dir(train_masks)
    create_dir(test_images)
    create_dir(test_masks)

    """ Data augmentation"""
    augment_data(train_x, train_y, AUGMENTED_DATA_BASE_PATH + "/train/", augment=True)
    print("her")
    augment_data(test_x, test_y, AUGMENTED_DATA_BASE_PATH + "/test/", augment=False)

if __name__ == "__main__":
    main()