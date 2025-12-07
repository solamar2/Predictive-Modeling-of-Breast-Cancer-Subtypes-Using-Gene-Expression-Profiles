# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 15:49:48 2025

@author: solam
"""

from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.utils import to_categorical

class Preprocessor:
    def __init__(self):
        self.scaler = None
        self.label_encoder = None

    def scale_X(self, X):
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X)

    def encode_y(self, y, onehot=True):
        self.label_encoder = LabelEncoder()
        y_int = self.label_encoder.fit_transform(y)
        if onehot:
            y_encoded = to_categorical(y_int)
            return y_encoded
        return y_int
