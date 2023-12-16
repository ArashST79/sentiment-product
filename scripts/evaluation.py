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
        table = plt.table(cellText=cell_text, colLabels=columns, rowLabels=rows, loc='center', cellLoc='center', fontsize=6)

        # Adjust the table size to fit the content
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.auto_set_column_width([0, 1, 2])

        # Customize cell size
        cell_height = 0.05
        cell_width = 0.2
        table.scale(1, 1.5)  # Adjust the scale factor as needed

     

        # Adjust the table size to fit the content
        plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.85)

        plt.axis('off')



def main():
    import os
    import pandas as pd
    from data_handler import get_data
    from preprocessing import Preprocessor
    from transformers import AutoTokenizer,DistilBertForSequenceClassification
    from sklearn.model_selection import train_test_split
    import random

    cache_dir_d_drive = "D:/models"
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=cache_dir_d_drive)
    model = DistilBertForSequenceClassification.from_pretrained('fine_tuned_distilbert', cache_dir=cache_dir_d_drive, num_labels = 5)
    model_bert = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment' ,cache_dir=cache_dir_d_drive,  num_labels = 5)
    tokenizer_bert = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment' ,cache_dir=cache_dir_d_drive)
    review_texts, labels = get_data()
    review_texts_train, review_texts_test, labels_train, labels_test = train_test_split(review_texts, labels, test_size=0.2, random_state=42)
    
    train_loader, val_loader = ProductReviewDataset.getDataLoader(review_texts_train, review_texts_test, labels_train, labels_test,tokenizer)
    train_loader_bert, val_loader_bert = ProductReviewDataset.getDataLoader(review_texts_train, review_texts_test, labels_train, labels_test,tokenizer_bert)
    evaluator = Evaluator(model,val_loader,num_labels = 5)
    evaluator.evaluate_model()
    loss = evaluator.loss
    mean_difference = evaluator.mean_difference

    evaluator_bert = Evaluator(model_bert,val_loader_bert,num_labels = 5)
    evaluator_bert.evaluate_model()
    loss_bert = evaluator_bert.loss
    mean_difference_bert = evaluator_bert.mean_difference
    print(f"for DistilBert Mode : loss = {loss}", f"mean difference = {mean_difference}")
    print(f"for Bert Mode : loss = {loss_bert}", f"mean difference = {mean_difference_bert}")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    evaluator.plot_confusion_matrix(ax = axes[0])
    axes[0].set_title("DistilBert Confusion Matrix")
    evaluator_bert.plot_confusion_matrix(ax = axes[1])
    axes[1].set_title("Bert Confusion Matrix")

    # Adjust layout for better spacing
    plt.tight_layout()

    fig2, axes2 = plt.subplots(nrows=2, ncols=1, figsize=(20, 12))
    num_samples = 15
    sample_indices = random.sample(range(len(evaluator.all_actual_labels)), num_samples)
    
    evaluator.display_sample_table(sample_indices, review_texts_test, ax=axes2[0])
    axes2[0].set_title("DistilBert Sample Table")
    evaluator_bert.display_sample_table(sample_indices, review_texts_test, ax=axes2[1])
    axes2[1].set_title("Bert Sample Table")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the combined figure
    plt.show()

if __name__ == "__main__":
    main()
