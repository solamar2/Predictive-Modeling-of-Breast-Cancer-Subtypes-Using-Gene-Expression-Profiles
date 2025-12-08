<<<<<<< HEAD
from scripts.Data_manager.Data_Loading import DataLoading
from scripts.Data_manager.Data_Preprocessing import Preprocessing
from scripts.Data_manager.Data_Split import DataSplit

class Pipline:
    def __init__(self):
        self.loader = DataLoading()
        self.preprocess=Preprocessing()
        self.datasplit=DataSplit()

    def main(self):
        # load data:
        GenesData, labels= self.loader.load_data()
        print("=== Stage 1: Data Loading ===")
        print("Data loaded successfully.")
        print(f"Patients (N): {GenesData.shape[0]}")
        print(f"Genes per patient (D): {GenesData.shape[1]}")
        print(f"Labels shape: {len(labels)}")
        print("----------------------------------")

        # Pre-process:
        GenesData_proc, labels_proc = self.preprocess.fit_transform(GenesData, labels)
        print("=== Stage 2: Preprocessing ===")
        print("Preprocessing completed successfully.")
        print(f"Processed data shape: {GenesData_proc.shape}")
        print(f"Processed labels shape: {labels_proc.shape}")
        print("----------------------------------")
        
        # Split the data
        train_loader,val_loader,test_loader=self.datasplit.split(GenesData_proc,labels_proc)
        print("=== Stage 3: Data Splitting ===")
        print("Data split completed successfully.")
        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Validation set size: {len(val_loader.dataset)}")
        print(f"Test set size: {len(test_loader.dataset)}")
        print("----------------------------------")

#X_test_proc, y_test_proc = pre.transform(X_test, y_test)

if __name__ == "__main__":
    pipeline = Pipline()
    pipeline.main()
=======
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
    

>>>>>>> 58782fb9c1e65a678d3c1f4d6d6498eefe511b78




