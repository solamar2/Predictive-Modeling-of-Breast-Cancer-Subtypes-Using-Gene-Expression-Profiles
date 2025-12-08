# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 15:49:31 2025

@author: solam
"""

class NNTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        pass  # כאן תוסיף את לולאת האימון

    def evaluate(self, X_test, y_test):
        pass  # כאן תחשב accuracy או metrics אחרים

    def save_model(self, path):
        import torch
        torch.save(self.model.state_dict(), path)
        
def TrainModel(model,train_loader,val_loader):
    learningRate=0.0005
    numepochs=1000
    early_stop_threshold=0.1
    # loss function
    lossfun = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(),lr=learningRate,weight_decay=1e-3)
    
    # intilaize:
    trainlosses = []
    trainacc    = []
    vallosses   = []
    valacc      = []
    # loop over epochs
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
    
        # now that we've trained through the batches, get their average training accuracy - per epoch:
        # and get average losses across the batches
        trainlosses.append(np.mean(batchLoss))
        
        # compute the predictions and report train accuracy per epoch
        with torch.no_grad(): # deactivates autograd
            trainlogits = model(train_loader.dataset.tensors[0])
        trainacc.append(binary_accuracy(trainlogits, train_loader.dataset.tensors[1]))

           
        model.eval()
        # compute validation accuracy & loss
        with torch.no_grad(): # deactivates autograd
            vallogits = model(val_loader.dataset.tensors[0])
        
        valacc.append(binary_accuracy(vallogits, val_loader.dataset.tensors[1]))
        vallosses.append(lossfun(vallogits, val_loader.dataset.tensors[1]).item())
        
        if abs(vallosses[epochi] - trainlosses[epochi]) > early_stop_threshold:
            print(f"Stopping early at epoch {epochi} – validation loss too high.")
            break
  
    return trainlosses,trainacc,vallosses,valacc