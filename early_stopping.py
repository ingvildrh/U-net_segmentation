''' https://www.youtube.com/watch?v=7Fboe7_aTtY '''

import os
import copy

class EarlyStopping():
    ''' patience: now improvement for 5 epochs, we done. min_delta = the amount it have to change to be countes as iomprovements '''
    def __init__(self, patience = 5, min_delta = 0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = " "

    def __call__(self, model , val_loss):
        #set the first val_loss as the best_loss
        if self.best_loss == None:
            self.best_loss = val_loss
            #make a copy of the model
            self.best_model = copy.deepcopy(model)
        #if the val_loss is better than the best_loss, set the val_loss as the best_loss
        elif self.best_loss -val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.epoch}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False