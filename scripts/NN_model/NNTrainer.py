import torch
import torch.nn as nn
import numpy as np
from scripts.Constants import alpha, weightdecayparm, numepochs, early_stop_threshold
import matplotlib.pyplot as plt

class NNTrainer:
    def __init__(self, model):
        self.model = model
    
    def compute_accuracy(self,logits, labels):
        """
        Compute multiclass accuracy
        logits: [N, num_classes] (raw outputs from model)
        labels: [N] (integers 0..num_classes-1)
        """
        preds = torch.argmax(logits, dim=1)
        correct = (preds == labels).sum().item()
        return correct / labels.size(0)
        
    def TrainModel(self,train_loader,val_loader):
        model=self.model
        # loss function
        lossfun = nn.CrossEntropyLoss()
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(),lr=alpha,weight_decay=weightdecayparm)
        
        # intilaize:
        trainlosses = []
        trainacc    = []
        vallosses   = []
        valacc      = []
        
        # --- training loop ---
        for epochi in range(numepochs):
            model.train()
            batchLoss = []
            
            # loop over batch
            for X,y in train_loader:
                # forward pass and loss
                yHat = model(X)
                loss = lossfun(yHat,y)
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # loss from this batch
                batchLoss.append(loss.item())    
        
            # now that we've trained through the batches, get their average training loss - per epoch:
            trainlosses.append(np.mean(batchLoss))
            # compute training accuracy
            with torch.no_grad(): # deactivates autograd
                trainlogits = model(train_loader.dataset.tensors[0])
            trainacc.append(self.compute_accuracy(trainlogits, train_loader.dataset.tensors[1]))

    
            model.eval()
            # compute validation accuracy & loss
            with torch.no_grad(): # deactivates autograd
                vallogits = model(val_loader.dataset.tensors[0])
            
            valacc.append(self.compute_accuracy(vallogits, val_loader.dataset.tensors[1]))
            vallosses.append(lossfun(vallogits, val_loader.dataset.tensors[1]).item())
            """
            if abs(vallosses[epochi] - trainlosses[epochi]) > early_stop_threshold:
                print(f"Stopping early at epoch {epochi} – validation loss too high.")
                break
      """
        return trainlosses,trainacc,vallosses,valacc
    
    
    def Plot_acc_loss(self, trainlosses,trainacc,vallosses,valacc):
        # plot acc and losses:
        epochs = np.arange(len(trainlosses))

        plt.figure(figsize=(8,5))
        plt.plot(epochs, trainlosses, label="Train Loss")
        plt.plot(epochs, vallosses,   label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss (Scatter)")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8,5))
        plt.plot(epochs, trainacc, label="Train Accuracy")
        plt.plot(epochs, valacc, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training & Validation Loss (Scatter)")
        plt.legend()
        plt.grid(True)
        plt.show()

        
        

