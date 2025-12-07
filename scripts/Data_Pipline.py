import scripts.Data_manager as DM

class Pipline():
    
    
    def main():
        expr_path="data/raw/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt",
        clinical_path="data/raw/clinical_patient_data.txt",
        relevant_subtypes = ['LumA', 'LumB', 'Her2', 'Basal', 'Normal']
        
        # load data:
        GenesData, labels= DM.load_data(expr_path, clinical_path, relevant_subtypes)
        
        
        # Pre-process:
        pre = DM.Preprocessing(zscore=True, pca_components=200)
        GenesData_proc, labels_proc = pre.fit_transform(GenesData, labels)

#X_test_proc, y_test_proc = pre.transform(X_test, y_test)

   if __name__ == "__main__":
    main() 
    





