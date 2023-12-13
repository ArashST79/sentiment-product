# scripts/model_training.py
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

def train_and_evaluate_model(train_loader,val_loader, model):

    all_actual_labels = []
    all_predicted_labels = []

    optimizer = AdamW(model.parameters(), lr=1e-5)

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        epoch_actual_labels = []
        epoch_predicted_labels = []

        with torch.no_grad():
            for val_batch in val_loader:
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss
                val_losses.append(val_loss.item())

                probabilities = torch.nn.functional.softmax(val_outputs.logits, dim=1)
                predicted_labels = torch.argmax(probabilities, dim=1).cpu().numpy()
                
                epoch_actual_labels.extend(val_batch['labels'].cpu().numpy())
                epoch_predicted_labels.extend(predicted_labels)

        average_difference = np.mean(np.abs(np.array(epoch_actual_labels) - np.array(epoch_predicted_labels)))

        print(f'Epoch {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {sum(val_losses)/len(val_losses)}, Mean Difference: {average_difference}')

        all_actual_labels.append(epoch_actual_labels)
        all_predicted_labels.append(epoch_predicted_labels)

    model.save_pretrained('fine_tuned_distilbert')

def main():
    import os
    import pandas as pd
    from data_handler import get_data
    from preprocessing import Preprocessor
    from transformers import AutoTokenizer,DistilBertForSequenceClassification
    from sklearn.model_selection import train_test_split

    cache_dir_d_drive = "D:/models"
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', cache_dir=cache_dir_d_drive)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', cache_dir=cache_dir_d_drive, num_labels = 5)

    review_texts, labels = get_data()
    review_texts_train, review_texts_test, labels_train, labels_test = train_test_split(review_texts, labels, test_size=0.2, random_state=42)
    
    train_loader, val_loader = ProductReviewDataset.getDataLoader(review_texts_train, review_texts_test, labels_train, labels_test,tokenizer)

    train_and_evaluate_model(train_loader, val_loader, model)
    # train_and_evaluate_model(preprocessor.X_train_preprocessed, preprocessor.y_train_preprocessed, preprocessor.X_test_preprocessed, preprocessor.y_test_preprocessed)

if __name__ == "__main__":
    main()
