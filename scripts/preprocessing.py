# scripts/preprocessing.py
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split



class Preprocessor:
    def __init__(self,data, tokenizer) :
        self.tokenizer = tokenizer
        self.nlp = spacy.load("en_core_web_sm")
        self.data = data
        self.X_train_preprocessed = None
        self.X_test_preprocessed = None
        self.y_train = None
        self.y_test = None
        self.train_data = None
        self.test_data = None
    
    def preprocess_text(self,text):
        doc = self.nlp(text)
        tokens = [token.lemma_ for token in doc if not token.is_stop]

        preprocessed_text = ' '.join(tokens)

        return preprocessed_text
    def preprocess_data(self):
        df = self.data
        X = df['Review Text']
        y = df['Sentiment']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.y_train = y_train
        self.y_test = y_test
        self.X_train_preprocessed = X_train.apply(self.preprocess_text)
        self.X_test_preprocessed = X_test.apply(self.preprocess_text)
        from transformers import AutoTokenizer



        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)

        self.train_data = {"text": self.X_train_preprocessed.tolist(), "score": self.y_train.tolist()}
        self.test_data = {"text": self.X_test_preprocessed.tolist(), "score": self.y_test.tolist()}
        self.train_data['label'] = [int(round((score + 1) / 2 * 4)) for score in self.train_data['score']]


        self.test_data['label'] = [int(round((score + 1) / 2 * 4)) for score in self.test_data['score']]



        from datasets import Dataset
        self.train_data = Dataset.from_dict(self.train_data)
        self.test_data = Dataset.from_dict(self.test_data)
        self.train_data = self.train_data.map(tokenize_function, batched=True)
        self.test_data = self.test_data.map(tokenize_function, batched=True)
        print("Training set shape:", self.X_train_preprocessed.shape)
        print("Test set shape:", self.X_test_preprocessed.shape)

