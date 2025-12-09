import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scripts.Constants import zscore, pca_components

class Preprocessing:
    def __init__(self):
        """
        zscore: apply Z-score normalization
        pca_components: number of PCA dimensions to keep (None = no PCA)
        """
        self.zscore = zscore
        self.pca_components = pca_components

        self.scaler = None      # StandardScaler model
        self.pca = None         # PCA model
        self.le = None          # LabelEncoder (if needed)
        self.y_numeric = False  # Flag to indicate if y is numeric


    def validate_data(self, X, y,fit_labels=False):
        """
        Cleans the data:
        - Converts all features to numeric (non-numeric → NaN)
        - Drops rows containing any NaN (missing OR non-numeric)
        - Drops the same rows from both X and y to keep alignment
        - Converts y to numeric (float) or integer labels
        - fit_labels: if True, fit a new LabelEncoder; else use existing one

        """
        df = pd.DataFrame(X)

        # Convert all values to numeric;
        df = df.apply(pd.to_numeric, errors='coerce')

        # Boolean mask: rows that contain NO NaN values
        valid_mask = ~df.isna().any(axis=1)

        # Filter X
        X_clean = df.loc[valid_mask].values

        # Filter y using the same mask
        y_clean = np.array(y)[valid_mask]
        
        # Convert y to numeric
        if fit_labels:
            # Try converting to float first
            try:
                y_clean = y_clean.astype(float)
                self.y_numeric = True
            except ValueError:
                # If y is strings/classes, use LabelEncoder
                self.le = LabelEncoder()
                y_clean = self.le.fit_transform(y_clean)
                self.y_numeric = False
        else:
            # Use the fitted LabelEncoder if it exists
            if not self.y_numeric and self.le is not None:
                y_clean = self.le.transform(y_clean)
            else:
                y_clean = y_clean.astype(float)
        
        return X_clean, y_clean


    def fit_transform(self, X, y):
        """
        Full preprocessing for training data:
        1. Clean X and y
        2. Fit normalizer + transform
        3. Fit PCA + transform
        """
        # Step 1: clean data
        X_proc, y_proc = self.validate_data(X, y,fit_labels=True)

        # Step 2: Z-score normalization
        if self.zscore:
            self.scaler = StandardScaler()
            X_proc = self.scaler.fit_transform(X_proc)

        # Step 3: PCA
        if self.pca_components is not None:
            if self.pca_components < X_proc.shape[1]:
                self.pca = PCA(n_components=self.pca_components)
                X_proc = self.pca.fit_transform(X_proc)

        return X_proc, y_proc

    
    def transform(self, X, y=None):
        """
        Applies the SAME preprocessing learned in fit_transform:
        - Clean X (and y if provided)
        - Apply stored scaler
        - Apply stored PCA
        """
        if y is not None:
            X_proc, y_proc = self.validate_data(X, y, fit_labels=False)
        else:
            # No labels provided (e.g. inference stage)
            X_proc, _ = self.validate_data(X, np.zeros(len(X)), fit_labels=False)
            y_proc = None

        # Z-score transform using fitted scaler
        if self.zscore and self.scaler is not None:
            X_proc = self.scaler.transform(X_proc)

        # PCA transform using fitted PCA
        if self.pca is not None:
            X_proc = self.pca.transform(X_proc)

        return (X_proc, y_proc) if y is not None else X_proc

  