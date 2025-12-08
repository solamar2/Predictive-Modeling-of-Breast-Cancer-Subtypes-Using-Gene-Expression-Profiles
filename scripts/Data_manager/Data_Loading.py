import pandas as pd
from scripts.Constants import expr_path, clinical_path, relevant_subtypes

class DataLoading:
    
    def __init__(self):
        self.expr_path = expr_path
        self.clinical_path = clinical_path
        self.relevant_subtypes = relevant_subtypes

    def load_data(self):
        """
        Load expression and clinical data, filter by relevant subtypes, 
        and return GenesData and labels as numpy arrays.
        """
        # Load gene expression data, transpose so rows are samples(patients)
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
            
        
