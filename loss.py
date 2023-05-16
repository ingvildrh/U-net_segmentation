import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Dice Loss
'''
class DiceLoss(nn.Module):
    '''
    Init function to initialize the DiceLoss object
    '''
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    '''
    Forward function to calculate the loss
    INPUT:
        inputs : predicted output
        targets : ground truth
        smooth : to avoid division by zero
    OUTPUT:
        dice : dice loss
    '''
    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

'''
Dice BCE Loss
'''
class DiceBCELoss(nn.Module):
    '''
    Init function to initialize the DiceBCELoss object
    '''
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    '''
    Forward function to calculate the loss
    INPUT:
        inputs : predicted output
        targets : ground truth
        smooth : to avoid division by zero
    OUTPUT:
        Dice_BCE : dice + binary cross entropy loss
    '''
    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE