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
from PIL import Image

import torch.nn as nn
from create_datasets import num_test, num_train, num_val

''' Create folders to save the predictions '''
binary_predictions_folder = 'binary_predictions_body_detections/'
segment_predictions_folder = 'segment_predictions_body_detections/'

''' Crerate folders for binary mask output in this image which are going to be added with the test image before sending into the secong model for testing '''
output_predicted_fish_body_mask_folder = 'model_output_predicted_fish_body_mask/'


MODEL_NAME = 'model'

if MODEL_NAME == 'model2':
    from model2 import build_unet
else:
    from model import build_unet


BODY_DETECTION_DATASET = 'body_detection_dataset/'


''' Set the hyperparameters for the model you want to train or test '''
H = 512
W = 512
size = (H, W)
batch_size = 4 #limit is 18 for 255x255 images and 4 for 512x512 images
num_epochs = 20
lr = 1e-4

body_pixels_dict = {}


def iou(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    x = (intersection + 1e-15) / (union + 1e-15)
    return x

def generate_confusion_matrix(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    confusion_matrix = np.zeros((2, 2))

    for i in range(len(y_true)):
        confusion_matrix[y_true[i]][y_pred[i]] += 1

    return confusion_matrix

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

#def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_iou = iou(y_true, y_pred)
    score_balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_iou, score_balanced_acc]

#def write_metrics_to_file(metrics, len_test_x, json_path):
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
        "iou": metrics[5] / len_test_x,
        "balanced_accuracy": metrics[6] / len_test_x,
        "H": H,
        "W": W,
        "model_name": MODEL_NAME,
    }
    out_file = open(json_path, "w")

    json.dump(metrics_dict, out_file, indent=4)

    out_file.close()


#def evaluation(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    print('ROC', auc(fpr, tpr))
        
    print('PR AUC', auc(recall, precision))
    

#def calculate_precicion_recall(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)

    return score_recall, score_precision

def save_output_body_model(prediction, name):
    prediction = prediction * 255
    cv2.imwrite(output_predicted_fish_body_mask_folder + name + ".png", prediction)

def remove_background(original_image, mask_image, name):
    output = cv2.bitwise_and(original_image, mask_image)
    
    cv2.imwrite('body_model_output/' + name + ".png", output)

def count_body_pixels(mask_image):
    return np.count_nonzero(mask_image), np.size(mask_image)

def write_body_pixels_to_json(name, count, total_pixels):
    body_pixels_dict[name] = [count, total_pixels]

    out_file = open("body_pixels.json", "w")

    json.dump(body_pixels_dict, out_file, indent=4)

    out_file.close()


def create_body_model_output(original_image, mask_image, name):
    remove_background(original_image, mask_image, name)
    pxls, total = count_body_pixels(mask_image)
    write_body_pixels_to_json(name, pxls, total)


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

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
    
    cv2.imwrite(segment_predictions_folder + name + ".png", cat_images)


def main():
    DATASET = 'body_detection_dataset'
    print('Performing test on dataset', DATASET)

    """ Seeding """
    seeding(42)

    """ Folders """
    #create_dir("results")

    """ Load dataset """
    #DETTE MÅ ENDRES FPR TESTING -----------------> MÅ HA RIKTIGE PATHS FOR TEST DATA
    test_x = sorted(list(paths.list_images(BODY_DETECTION_DATASET + 'test/image/')))
    test_y = sorted(list(paths.list_images(BODY_DETECTION_DATASET + 'test/mask/')))

    checkpoint_path = "files/checkpoint_" + DATASET + "_BS_" + str(batch_size) + "_E_" + str(num_epochs) + "_LR_" + str(lr) + "_" + str(H) +  ".pth"
    
    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #this could be made global
    model = build_unet()
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = os.path.basename(x)
        name = name.split(".")[0]
        
        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        ## mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)
    
        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()

            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)

            total_time = time.time() - start_time
            time_taken.append(total_time)

            """ Calculate metrics """ 
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)

            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

            save_predicted_segmentation_images(y, pred_y, image, size, name)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        save_output_body_model(pred_y, name)
        print(np.size(pred_y))
        create_body_model_output(image,pred_y*255, name)
       
        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        """ Writing the results to folder in the format: image, original mask, predicted mask"""
        cv2.imwrite(binary_predictions_folder + name + ".png", cat_images)

    # """ Calculating the mean of the metrics """
    # jaccard = metrics_score[0]/len(test_x)
    # f1 = metrics_score[1]/len(test_x)
    # recall = metrics_score[2]/len(test_x)
    # precision = metrics_score[3]/len(test_x)
    # acc = metrics_score[4]/len(test_x)
    # iou_mean = metrics_score[5]/len(test_x)
    # balanced_accuracy_score = metrics_score[6]/len(test_x)

    #write_metrics_to_file(metrics_score, len(test_x), json_path + DATASET + "_BS" + str(batch_size) + "E" + str(num_epochs) + "LR" + str(lr) + ".json")

    #print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - Balanced Acc: {balanced_accuracy_score:1.4f}")

    # fps = 1/np.mean(time_taken)
    # print("FPS (frames per second): ", fps)


if __name__ == "__main__":
    main()
