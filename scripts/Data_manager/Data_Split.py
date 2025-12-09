import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split
from scripts.Constants import batchsize,valtestsize,testsize

class DataSplit:
    def __init__(self):
        """
        Initialize split parameters from constants.
        batchsize:     batch size for training
        valtestsize:   fraction for (validation + test)
        testsize:      fraction for test out of the temp set
        """
        self.batchsize = batchsize
        self.valtestsize=valtestsize
        self.testsize=testsize


    def split(self, data,labels):
        """
        Perform full train/validation/test split and wrap each set
        in a TensorDataset and DataLoader.

        Args:
            data   (pd.DataFrame): gene expression matrix [N_samples, N_genes]
            labels (array-like):   binary labels or continuous values [N_samples]

        Returns:
            train_loader, val_loader, test_loader
        """
        # Step 1: convert  from DataFrame to tensor
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"Expected data to be DataFrame or numpy array, got {type(data)}")
        data = torch.tensor(data).float()
        
        if isinstance(labels, pd.Series):
            labels = labels.to_numpy()
        labels = torch.tensor(labels).long()
        
    
        # Step 2: first split: train + temp (val+test)
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data, labels, test_size=self.valtestsize, shuffle=True)
        
        # Step 3: second split: validation + test
        val_data, test_data, val_labels, test_labels = train_test_split(
            temp_data, temp_labels, test_size=self.testsize, shuffle=True)
        
        # Step 4: create TensorDatasets
        train_ds = TensorDataset(train_data, train_labels)
        val_ds   = TensorDataset(val_data,   val_labels)
        test_ds  = TensorDataset(test_data,  test_labels)
       
        # Step 5: dataloaders
        train_loader = DataLoader(train_ds, batch_size=batchsize, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds,   batch_size=val_ds.tensors[0].shape[0])
        test_loader  = DataLoader(test_ds,  batch_size=test_ds.tensors[0].shape[0])
       
        return train_loader,val_loader,test_loader