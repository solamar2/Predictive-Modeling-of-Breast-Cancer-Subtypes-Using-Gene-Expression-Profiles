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
