from scripts.Data_manager.Data_Loading import DataLoading
from scripts.Data_manager.Data_Preprocessing import Preprocessing
from scripts.Data_manager.Data_Split import DataSplit
from scripts.NN_model.NN_model import createANNmodel
from scripts.NN_model.NNTrainer import NNTrainer
from scripts.NN_model.Evaluate_Model import EvaluateModel

import numpy as np

class Pipline:
    def __init__(self):
        self.loader = DataLoading()
        self.preprocess=Preprocessing()
        self.datasplit=DataSplit()
       

    def main(self):
        # load data:
        print("=== Stage 1: Data Loading ===")
        GenesData, labels= self.loader.load_data()
        print("Data loaded successfully.")
        print(f"Patients (N): {GenesData.shape[0]}")
        print(f"Genes per patient (D): {GenesData.shape[1]}")
        print(f"Labels shape: {len(labels)}")
        print("----------------------------------")

        # Pre-process:
        print("=== Stage 2: Preprocessing ===")
        GenesData_proc, labels_proc = self.preprocess.fit_transform(GenesData, labels)
        print("Preprocessing completed successfully.")
        print(f"Processed data shape: {GenesData_proc.shape}")
        print(f"Processed labels shape: {labels_proc.shape}")
        print("----------------------------------")
        
        
        # Count number of samples per class
        unique_classes, counts = np.unique(labels_proc, return_counts=True)
        
        for cls, count in zip(unique_classes, counts):
            print(f"Class {cls}: {count} samples")
            
        
        # Split the data
        print("=== Stage 3: Data Splitting ===")
        train_loader,val_loader,test_loader=self.datasplit.split(GenesData_proc,labels_proc)
        print("Data split completed successfully.")
        print(f"Train set size: {len(train_loader.dataset)}")
        print(f"Validation set size: {len(val_loader.dataset)}")
        print(f"Test set size: {len(test_loader.dataset)}")
        print("----------------------------------")
        
        # NN model
        print("=== Stage 4: Model Creation ===")
        self.model=createANNmodel(numofgenes=GenesData.shape[1])
        print("Neural network model created successfully.")
        print(f"Model architecture:\n{self.model}")
        print(f"Number of input features: {GenesData.shape[1]}")
        print("----------------------------------")
        
        # Train the model
        print("=== Stage 5: Model Training ===")
        print("Training initialized...")
        Model_NNtrain=NNTrainer(self.model)
        trainlosses,trainacc,vallosses,valacc=Model_NNtrain.TrainModel(train_loader,val_loader)
        print("Training completed successfully.")
        print(f"Final Training Loss:  {trainlosses[-1]:.4f}")
        print(f"Final Training Acc:   {trainacc[-1]:.4f}")
        print(f"Final Validation Loss:{vallosses[-1]:.4f}")
        print(f"Final Validation Acc: {valacc[-1]:.4f}")
        print("----------------------------------")
        Model_NNtrain.Plot_acc_loss(trainlosses,trainacc,vallosses,valacc)
        # Test evaluation
        print("=== Stage 6: Test Evaluation ===")
        
        evaluator=EvaluateModel( self.model,test_loader)
        evaluator.plot_confusion_matrix()
        evaluator.plot_accuracy_per_class()
        evaluator.plot_loss_per_class()



if __name__ == "__main__":
    pipeline = Pipline()
    pipeline.main()



