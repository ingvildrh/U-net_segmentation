import os


'''
Set directories for training the network 
'''

DATASET = '999_9999'

''' Set the hyperparameters for the model you want to train or test '''
H = 512
W = 512
size = (H, W)
batch_size = 1
num_epochs = 2
lr = 1e-4

''' Set the data paths '''
data_path = "C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/DATA_SETS/dataset_" + DATASET +  "/train_images"
mask_path = "C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/DATA_SETS/dataset_" + DATASET +  "/train_masks"
test_path = "C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/DATA_SETS/dataset_" + DATASET +  "/val_images"
ground_truth = "C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/DATA_SETS/dataset_" + DATASET +  "/val_masks"

''' Set the paths for the augmented data directories '''
AUGMENTED_DATA_BASE_PATH = 'new_data_' + DATASET + "/"

train_images =  AUGMENTED_DATA_BASE_PATH + 'train/image/'
train_masks = AUGMENTED_DATA_BASE_PATH + "train/mask/"
test_images = AUGMENTED_DATA_BASE_PATH + "test/image/"
test_masks = AUGMENTED_DATA_BASE_PATH + "test/mask/"

''' Set the path for predicted segmentation directory '''
predicted_segmentation_path = 'predicted_segmentation_' + DATASET + "_BS" + str(batch_size) + "E" + str(num_epochs) + "LR" + str(lr) + "/"

''' Set the path for the predicted binary masks results '''
results = "results_" + DATASET + "_BS" + str(batch_size) + "E" + str(num_epochs) + "LR" + str(lr) + "/"


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
    create_dir(test_images)
    create_dir(test_masks)
    create_dir(predicted_segmentation_path)
    create_dir(results)
    create_dir(json_path)

    for filename in os.listdir('C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/U-NET/UNET_trying/predicted_segmentations_999_9999'):
        
        if 'BS32E30.png' in filename:
            print('yes', filename)
            os.remove('C:/Users/ingvilrh/OneDrive - NTNU/MASTER_CODE23/U-NET/UNET_trying/predicted_segmentations_999_9999/' + filename)
    
    