import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
from data import DriveDataset
from imutils import paths
import matplotlib.pyplot as plt
from config import * 

if MODEL_NAME == 'model2':
    from model2 import build_unet
else:
    from model import build_unet

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
'''
To prepare data for training, the image paths need to be modified in this file and in data_aug.py
To augmentate the images correctly, please run data_aug.py first 
'''


os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:256"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]= "backend:native"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]= "garbage_collection_threshold:0.5"

loss_dict = {"train_loss": [], "validation_loss": []}

def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
        #x = x.to(device_ids[0], dtype=torch.float32)
        #y = y.to(device_ids[0], dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            #x = x.to(device_ids[0], dtype=torch.float32)
            #y = y.to(device_ids[0], dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files") #prøv å droppe dette

    """ Load dataset """
    train_x = sorted(list(paths.list_images(AUGMENTED_DATA_BASE_PATH + 'train/image/')))
    train_y = sorted(list(paths.list_images(AUGMENTED_DATA_BASE_PATH + "train/mask/")))

    valid_x = sorted(list(paths.list_images(AUGMENTED_DATA_BASE_PATH + "val/image/")))
    valid_y = sorted(list(paths.list_images(AUGMENTED_DATA_BASE_PATH + "val/mask/")))


    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

   
    checkpoint_path = "files/checkpoint_" + DATASET + "_BS_" + str(batch_size) + "_E_" + str(num_epochs) + "_LR_" + str(lr) + ".pth"

    """ Dataset and loader """
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
    print("Running on:", device)
    model = build_unet()
    model = nn.DataParallel(model)

    model = model.to(device)
    #model = nn.DataParallel(model, device_ids=device_ids)
    #model = model.to(device_ids[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    """ Training the model """
    best_valid_loss = float("inf")
    print("Training the model for the dataset", DATASET)
    print("Batch size:", batch_size, "Epochs:", num_epochs, "Learning rate:", lr, "H:", H, "W:", W)
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)
        loss_dict["train_loss"].append(train_loss)
        loss_dict["validation_loss"].append(valid_loss)


        """ Plot the training and the validation loss """
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(loss_dict["train_loss"], label="train_loss")
        plt.plot(loss_dict["validation_loss"], label="validation_loss")
        plt.title("Training Loss on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend(loc="lower left")
        plt.savefig(os.path.sep.join(["plots", "loss_plot" + DATASET+ "_BS_" + str(batch_size) + "_E_" + str(num_epochs) + "_LR_" + str(lr) + " .png"]))

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} / {num_epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)