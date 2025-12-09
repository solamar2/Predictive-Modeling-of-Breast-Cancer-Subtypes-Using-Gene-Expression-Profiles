import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class EvaluateModel:
    def __init__(self, model, dataloader, class_names=None):
        self.model = model
        """
        Evaluate model and plot:
        - Confusion matrix
        - Accuracy per class
        - Loss per class
        """
        
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')  # keep loss per sample
    
        # Get all data from dataloader (single batch)
        with torch.no_grad():
            logits = model(dataloader.dataset.tensors[0])
            losses = loss_fn(logits, dataloader.dataset.tensors[1])
            preds = torch.argmax(logits, dim=1)
    
        # Convert to numpy
        self.all_preds = preds.numpy()
        self.all_labels = dataloader.dataset.tensors[1].numpy()
        self.all_losses = losses.numpy()
    
        if class_names is None:
            self.class_names = [str(i) for i in range(len(np.unique(self.all_labels)))]
        else:
            self.class_names=class_names
    
        self.num_classes = len(self.class_names)
        
         # --- Print overall metrics ---
        overall_acc = (self.all_preds == self.all_labels).mean()
        overall_loss = self.all_losses.mean()
        print("=== Overall Metrics ===")
        print(f"Overall Accuracy: {overall_acc:.4f}")
        print(f"Overall Loss: {overall_loss:.4f}")
        print("------------------------")
        
        
    
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.all_labels, self.all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title("Confusion Matrix")
        plt.show()
        return cm

    
    def plot_accuracy_per_class(self):
        acc_per_class = []
        for cls in range(self.num_classes):
            cls_mask = (self.all_labels == cls)
            acc = (self.all_preds[cls_mask] == self.all_labels[cls_mask]).mean()
            acc_per_class.append(acc)

        plt.figure(figsize=(8,4))
        plt.bar(self.class_names, acc_per_class, color='skyblue')
        plt.ylabel("Accuracy")
        plt.title("Accuracy per Class")
        plt.ylim(0,1)
        plt.show()
        return acc_per_class


    def plot_loss_per_class(self):
        loss_per_class = []
        for cls in range(self.num_classes):
            cls_mask = (self.all_labels == cls)
            cls_loss = self.all_losses[cls_mask].mean()
            loss_per_class.append(cls_loss)

        plt.figure(figsize=(8,4))
        plt.bar(self.class_names, loss_per_class, color='salmon')
        plt.ylabel("Loss")
        plt.title("Loss per Class")
        plt.show()
        return loss_per_class





