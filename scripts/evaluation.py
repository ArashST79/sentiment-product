# scripts/evaluation.py
from transformers import pipeline, AutoModelForSequenceClassification, DistilBertTokenizer, AdamW
from sklearn import metrics
import joblib
from preprocessing import Preprocessor  
import torch
from data_loader import ProductReviewDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self,model,data_loader,num_labels) -> None:
        self.model = model
        self.num_labels = num_labels
        self.all_actual_labels = []
        self.all_predicted_labels = []
        self.mean_difference = None
        self.loss = None
        self.data_loader = data_loader
    def evaluate_model(self):
        self.model.eval()
        
        all_val_losses = []

        with torch.no_grad():
            for val_batch in tqdm(self.data_loader):
                val_outputs = self.model(**val_batch)
                val_loss = val_outputs.loss
                all_val_losses.append(val_loss.item())

                probabilities = torch.nn.functional.softmax(val_outputs.logits, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()

                self.all_actual_labels.extend(val_batch['labels'].cpu().numpy())
                self.all_predicted_labels.extend(predicted_labels)


        average_difference = np.mean(np.abs(np.array(self.all_actual_labels) - np.array(self.all_predicted_labels)))
        self.mean_difference = average_difference
        self.loss = sum(all_val_losses) / len(all_val_losses)
        return sum(all_val_losses) / len(all_val_losses), average_difference
    
    def plot_confusion_matrix(self, ax = None):
        class_names = [f"Class {i}" for i in range(self.num_labels)] 
        cm = confusion_matrix(self.all_actual_labels, self.all_predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    
        # You can customize the color map as per your preference
        cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
        
        disp.plot(ax = ax,cmap=cmap, values_format="d")

    def display_sample_table(self, sample_indices, review_texts_test, ax=None):
        sample_data = [(review_texts_test[i], self.all_actual_labels[i], self.all_predicted_labels[i]) for i in sample_indices]

        # Create a table
        columns = ['Sample Text', 'Actual Label', 'Predicted Label']
        rows = [f"Sample {i+1}" for i in range(len(sample_indices))]

        # Adjust the size of the plot for better visibility
        if ax is None:
            plt.figure(figsize=(12, 8))
        else:
            plt.sca(ax)

        cell_text = [[str(data[0])[:200], str(data[1]), str(data[2])] for data in sample_data]

        # Increase the font size for better visibility
        table = plt.table(cellText=cell_text, colLabels=columns, rowLabels=rows, loc='center', cellLoc='center', fontsize=15)

        # Adjust the table size to fit the content
        table.auto_set_font_size(False)
        table.set_fontsize(15)
        table.auto_set_column_width([0, 1, 2])

        # Customize cell size
        cell_height = 0.05
        cell_width = 0.2
        table.scale(1, 1.5)  # Adjust the scale factor as needed

     

        # Adjust the table size to fit the content
        plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.85)

        plt.axis('off')


