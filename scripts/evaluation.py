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

def evaluate_model(data_loader, model):
    model.eval()
    all_actual_labels = []
    all_predicted_labels = []
    all_val_losses = []

    with torch.no_grad():
        for val_batch in tqdm(data_loader):
            val_outputs = model(**val_batch)
            val_loss = val_outputs.loss
            all_val_losses.append(val_loss.item())

            probabilities = torch.nn.functional.softmax(val_outputs.logits, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()

            all_actual_labels.extend(val_batch['labels'].cpu().numpy())
            all_predicted_labels.extend(predicted_labels)


    average_difference = np.mean(np.abs(np.array(all_actual_labels) - np.array(all_predicted_labels)))

    return sum(all_val_losses) / len(all_val_losses), average_difference

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

    review_texts, labels = get_data()
    review_texts_train, review_texts_test, labels_train, labels_test = train_test_split(review_texts, labels, test_size=0.2, random_state=42)
    
    train_loader, val_loader = ProductReviewDataset.getDataLoader(review_texts_train, review_texts_test, labels_train, labels_test,tokenizer)

    loss, mean_difference = evaluate_model(val_loader, model)
    print(f"loss = {loss}", f"mean difference = {mean_difference}")

if __name__ == "__main__":
    main()
