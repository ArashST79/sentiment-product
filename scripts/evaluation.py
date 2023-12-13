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
    def __init__(self,model,num_labels) -> None:
        self.model = model
        self.num_labels = num_labels
        self.all_actual_labels = []
        self.all_predicted_labels = []
        self.mean_difference = None
        self.loss = None
    def evaluate_model(self,data_loader):
        self.model.eval()
        
        all_val_losses = []

        with torch.no_grad():
            for val_batch in tqdm(data_loader):
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
    
    def plot_confusion_matrix(self):
        class_names = [f"Class {i}" for i in range(self.num_labels)] 
        cm = confusion_matrix(self.all_actual_labels, self.all_predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        # Adjust the size of the plot for better visibility
        plt.figure(figsize=(8, 8))
        
        # You can customize the color map as per your preference
        cmap = sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True)
        
        disp.plot(cmap=cmap, values_format="d")

        plt.title("Confusion Matrix")
        plt.show()


def main():
    import os
    import pandas as pd
    from data_handler import get_data
    from preprocessing import Preprocessor
    from transformers import AutoTokenizer,DistilBertForSequenceClassification
    from sklearn.model_selection import train_test_split

    cache_dir_d_drive = "D:/models"
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=cache_dir_d_drive)
    model = DistilBertForSequenceClassification.from_pretrained('fine_tuned_distilbert', cache_dir=cache_dir_d_drive, num_labels = 5)
    model_bert = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment' ,cache_dir=cache_dir_d_drive,  num_labels = 5)
    tokenizer_bert = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment' ,cache_dir=cache_dir_d_drive)
    review_texts, labels = get_data()
    review_texts_train, review_texts_test, labels_train, labels_test = train_test_split(review_texts, labels, test_size=0.2, random_state=42)
    
    train_loader, val_loader = ProductReviewDataset.getDataLoader(review_texts_train, review_texts_test, labels_train, labels_test,tokenizer)
    train_loader_bert, val_loader_bert = ProductReviewDataset.getDataLoader(review_texts_train, review_texts_test, labels_train, labels_test,tokenizer_bert)
    evaluator = Evaluator(model,num_labels = 5)
    evaluator.evaluate_model(val_loader)
    loss = evaluator.loss
    mean_difference = evaluator.mean_difference

    evaluator_bert = Evaluator(model_bert,num_labels = 5)
    evaluator_bert.evaluate_model(val_loader_bert)
    loss_bert = evaluator_bert.loss
    mean_difference_bert = evaluator_bert.mean_difference
    print(f"for DistilBert Mode : loss = {loss}", f"mean difference = {mean_difference}")
    print(f"for Bert Mode : loss = {loss_bert}", f"mean difference = {mean_difference_bert}")
    evaluator.plot_confusion_matrix()
    evaluator_bert.plot_confusion_matrix()

if __name__ == "__main__":
    main()
