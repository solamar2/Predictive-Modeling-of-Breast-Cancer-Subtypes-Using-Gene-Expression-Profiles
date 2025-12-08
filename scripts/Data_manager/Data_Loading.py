import pandas as pd
<<<<<<< HEAD
from scripts.Constants import expr_path, clinical_path, relevant_subtypes

class DataLoading:
    
    def __init__(self):
        self.expr_path = expr_path
        self.clinical_path = clinical_path
        self.relevant_subtypes = relevant_subtypes
=======

class DataLoading:
    def __init__(self, expr_path, clinical_path, relevant_subtypes):
        self.expr_path = expr_path
        self.clinical_path = clinical_path
        self.relevant_subtypes = relevant_subtypes
        
        # Initialize
        self.expr = None
        self.clinical = None
        self.GenesData = None
        self.labels = None
>>>>>>> 58782fb9c1e65a678d3c1f4d6d6498eefe511b78

    def load_data(self):
        """
        Load expression and clinical data, filter by relevant subtypes, 
        and return GenesData and labels as numpy arrays.
        """
        # Load gene expression data, transpose so rows are samples(patients)
<<<<<<< HEAD
        expr = pd.read_csv(self.expr_path, sep='\t', index_col=0).T
      
        # Load clinical data
        clinical = pd.read_csv(self.clinical_path, sep='\t', index_col=0)
        
        # Filter clinical data to include only relevant subtypes
        clinical = clinical[
            clinical['Pam50 + Claudin-low subtype'].isin(self.relevant_subtypes)]
        
        # Keep only patients that exist in both expression and clinical data
        expr = expr.loc[expr.index.intersection(clinical.index)]
        clinical = clinical.loc[expr.index]

        # Check that the indices match between expression and clinical data
        assert all(expr.index == clinical.index), "Mismatch between expression and clinical data!"

        # Convert to numpy arrays for model input
        GenesData = expr.values
        labels = clinical['Pam50 + Claudin-low subtype'].values
        
        return GenesData, labels
=======
        self.expr = pd.read_csv(self.expr_path, sep='\t', index_col=0).T
      
        # Load clinical data
        self.clinical = pd.read_csv(self.clinical_path, sep='\t', index_col=0)
        
        # Filter clinical data to include only relevant subtypes
        self.clinical = self.clinical[
            self.clinical['Pam50 + Claudin-low subtype'].isin(self.relevant_subtypes)]
        
        # Keep only patients that exist in both expression and clinical data
        self.expr = self.expr.loc[self.expr.index.intersection(self.clinical.index)]
        self.clinical = self.clinical.loc[self.expr.index]

        # Check that the indices match between expression and clinical data
        assert all(self.expr.index == self.clinical.index), "Mismatch between expression and clinical data!"

        # Convert to numpy arrays for model input
        self.GenesData = self.expr.values
        self.labels = self.clinical['Pam50 + Claudin-low subtype'].values
        
        return self.GenesData, self.labels
>>>>>>> 58782fb9c1e65a678d3c1f4d6d6498eefe511b78
            
        
