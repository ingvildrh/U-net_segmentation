import os

'''
Set directories for training the network 
'''
device_ids = [0, 1, 2, 3]


DATASET = '111_1111'

''' Set the hyperparameters for the model you want to train or test '''
H = 512
W = 512
size = (H, W)
batch_size = 4
num_epochs = 1
lr = 1e-4

''' Set the model to train/test'''
MODEL_NAME = 'model'

''' Set the data paths, these must be changes if you want to run the code on your own computer '''
data_path ='C:/Users/ingvilrh/master_data/dataset_111_1111/train_images'
mask_path = 'C:/Users/ingvilrh/master_data/dataset_111_1111/train_masks'
val_path = 'C:/Users/ingvilrh/master_data/dataset_111_1111/val_images'
val_truth = 'C:/Users/ingvilrh/master_data/dataset_111_1111/val_masks'
test_path = 'C:/Users/ingvilrh/master_data/dataset_111_1111/test_images'
test_truth = 'C:/Users/ingvilrh/master_data/dataset_111_1111/test_masks'


''' Set the paths for the augmented data directories '''
AUGMENTED_DATA_BASE_PATH = 'augmented_data/new_data_' + DATASET + "_"+  str(H) +  "/"

train_images =  AUGMENTED_DATA_BASE_PATH + 'train/image/'
train_masks = AUGMENTED_DATA_BASE_PATH + "train/mask/"
val_images = AUGMENTED_DATA_BASE_PATH + "val/image/"
val_masks = AUGMENTED_DATA_BASE_PATH + "val/mask/"
test_images = AUGMENTED_DATA_BASE_PATH + "test/image/"
test_masks = AUGMENTED_DATA_BASE_PATH + "test/mask/"

''' Set the path for predicted segmentation directory '''
predicted_segmentation_path = 'predicted_segmentations/predicted_segmentation_' + DATASET + "_BS_" + str(batch_size) + "_E_" + str(num_epochs) + "_LR_" + str(lr) + "/"

''' Set the path for the predicted binary masks results '''
results = "predicted_binary_segmentations/results_" + DATASET + "_BS_" + str(batch_size) + "_E_" + str(num_epochs) + "_LR_" + str(lr) + "/"


''' Set the path for the json file to save metrics '''
json_path = "metrics_results_/"

''' 
Create a directory 
'''
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    create_dir(train_images)
    create_dir(train_masks)
    create_dir(val_images)
    create_dir(val_masks)
    create_dir(predicted_segmentation_path)
    create_dir(results)
    create_dir(json_path)

  