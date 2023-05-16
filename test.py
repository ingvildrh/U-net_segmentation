import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, precision_recall_curve, auc, roc_curve, balanced_accuracy_score
import matplotlib.pyplot as plt
import json
from model import build_unet
from utils import create_dir, seeding
from imutils import paths
from config import * 
import torch.nn as nn
from create_datasets import num_test, num_train, num_val
from train import EARLY_STOPPING

''' Set the model to be used for testing '''
if MODEL_NAME == 'model2':
    from model2 import build_unet
# elif MODEL_NAME == 'model3':
#     from model3 import build_unet
# elif MODEL_NAME == 'model4':
#     from model4 import build_unet
else:
    from model import build_unet

''' Initialize a dictionary for counting of pixels '''
ulcer_pixels_dict = {}

'''
Generate confucion matrix for each image from the the test set
INPUT:
    y_true : ground truth
    y_pred : prediction
OUTPUT:
    confusion_matrix: 2x2 matrix
'''
def generate_confusion_matrix(y_true, y_pred):
    ''' Ground truth '''
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    ''' Prediction '''
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    confusion_matrix = np.zeros((2, 2))

    for i in range(len(y_true)):
        confusion_matrix[y_true[i]][y_pred[i]] += 1

    return confusion_matrix

'''
Plot the confusion matrix
INPUT:
    cf : confusion matrix
'''
def plot_confusion_matrix(cf):
    # Plot confusion matrix
    plt.imshow(cf, interpolation='nearest', cmap=plt.cm.Reds)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Add numbers to the plot
    width, height = cf.shape
    for x in range(width):
        for y in range(height):
            plt.annotate(str(cf[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    # Show plot
    plt.show()

'''
Calculate the metrics for the test set
INPUT:
    y_true : ground truth
    y_pred : prediction
OUTPUT:
    score_jaccard : jaccard score
    score_f1 : f1 score
    score_recall : recall score
    score_precision : precision score
    score_acc : accuracy score
    score_balanced_acc : balanced accuracy score
'''
def calculate_metrics(y_true, y_pred):
    ''' Ground truth '''
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    ''' Prediction '''
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_balanced_acc]

'''
Write the metrics to a .JSON file
INPUT:
    metrics : list of metrics
    len_test_x : length of the test set
    json_path : path to the .JSON file
'''
def write_metrics_to_file(metrics, len_test_x, json_path):
    metrics_dict = {
        "batch_size": batch_size,
        "epochs": num_epochs,
        "learning_rate": lr,
        "len_train_x": num_train,
        "len_test_x": num_test,
        "len_val_x": num_val,
        "jaccard": metrics[0] / len_test_x,
        "f1": metrics[1] / len_test_x,
        "recall": metrics[2] / len_test_x,
        "precision": metrics[3] / len_test_x,
        "accuracy": metrics[4] / len_test_x,
        "balanced_accuracy": metrics[5] / len_test_x,
        "H": H,
        "W": W,
        "model_name": MODEL_NAME,
        "early_stopping": EARLY_STOPPING,
    }
    out_file = open(json_path, "w")

    json.dump(metrics_dict, out_file, indent=4)

    out_file.close()

'''
Convert a grayscale mask to a RGB mask
INPUT:
    mask : grayscale mask
OUTPUT:
    mask : RGB mask
'''
def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

'''
Count the number of ulcer pixels in a mask
INPUT:
    mask_image : mask image
OUTPUT:
    pxls : number of ulcer pixels
'''
def count_ulcer_pixels(mask_prediction):
    return np.count_nonzero(mask_prediction), np.size(mask_prediction)

'''
Write body pixels to a .JSON file
INPUT:
    name : name of the image
    count : number of ulcer pixels
'''
def write_body_pixels_to_json(name, count, total_pixels):
    ulcer_pixels_dict[name] = [count, total_pixels]
    out_file = open("ulcer_pixels.json", "w")
    json.dump(ulcer_pixels_dict, out_file, indent=4)
    out_file.close()

'''
Count ulcer pixels and write to a .JSON file
INPUT:
    mask_image : mask image
    name : name of the image
'''
def create_ulcer_model_output(mask_image, name):
    pxls, total = count_ulcer_pixels(mask_image)
    print(pxls, total)
    write_body_pixels_to_json(name, pxls, total)

'''
Save the predicted images to a folder in the format: original image, ground truth mask segment on image, predicted mask segment on image
INPUT:
    y_true : ground truth mask
    y_pred : predicted mask
    image : original image
    size : size of the image
    name : name of the image
'''
def save_predicted_segmentation_images(y_true, y_pred, image, size, name):
    y_true = y_true[0].cpu().numpy()        ## (1, 512, 512)
    y_true = np.squeeze(y_true, axis=0)     ## (512, 512)     
    y_true = np.array(y_true, dtype=np.uint8)
    
    y_pred = 1 - y_pred
    y_true = 1 - y_true

    white1 = np.ones((H, W, 3), dtype = np.uint8) 
    white2 = np.ones((H, W, 3), dtype = np.uint8) 
    white1[y_pred == 0] = [1,255,1]
    white2[y_true == 0] = [1,255,1]

    
    overlayed_image = cv2.addWeighted(image, 1, white1, 0.5, 0.0) #predicted one
    overlayed_image2 = cv2.addWeighted(image, 1, white2, 0.5, 0.0) #true one

    line = np.ones((size[1], 10, 3)) * 128
    cat_images = np.concatenate([image, line, overlayed_image2, line, overlayed_image], axis=1)
    
    cv2.imwrite(predicted_segmentation_path + name + ".png", cat_images)


def main():
    print('Performing test on dataset', DATASET)

    ''' Seeding '''
    seeding(42)

    ''' Load dataset '''
    test_x = sorted(list(paths.list_images(AUGMENTED_DATA_BASE_PATH + 'test/image/')))
    #test_x = sorted(list(paths.list_images('body_model_output/')))
    test_y = sorted(list(paths.list_images(AUGMENTED_DATA_BASE_PATH + 'test/mask/')))

    checkpoint_path = "files/checkpoint_" + DATASET + "_BS_" + str(batch_size) + "_E_" + str(num_epochs) + "_LR_" + str(lr) + "_" + str(H) +  ".pth"
    

    ''' Load the checkpoint '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #this could be made global
    model = build_unet()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    ''' Initialize the metrics list '''
    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        ''' Extract the name '''
        name = os.path.basename(x)
        name = name.split(".")[0]
        
        ''' Reading image '''
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        ''' Reading mask '''
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        ## mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        
    
        with torch.no_grad():
            ''' Prediction and Calculating FPS '''
            start_time = time.time()

            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)

            total_time = time.time() - start_time
            time_taken.append(total_time)

            ''' Calculate metrics '''
            print("Y", y.unique())
            print("Pred range", pred_y.max(), pred_y.min())
            
            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            
           
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)

            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

            save_predicted_segmentation_images(y, pred_y, image, size, name)
           

        ''' Saving masks '''
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        print(np.size(pred_y))
        create_ulcer_model_output(pred_y*255, name)

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        ''' Writing the results to folder in the format: image, original mask, predicted mask'''
        cv2.imwrite(results + name + ".png", cat_images)

    ''' Calculating the mean of the metrics '''
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    balanced_accuracy_score = metrics_score[5]/len(test_x)

    write_metrics_to_file(metrics_score, len(test_x), json_path + DATASET + "_BS" + str(batch_size) + "E" + str(num_epochs) + "LR" + str(lr) + ".json")

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - Balanced Acc: {balanced_accuracy_score:1.4f}")

    fps = 1/np.mean(time_taken)
    print("FPS (frames per second): ", fps)


if __name__ == "__main__":
    main()
