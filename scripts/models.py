import torch
import torch.nn as nn


class ModelOutputWithLoss:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss

# Define your sequence classification head
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, num_labels):
        super(SentimentClassifier, self).__init__()
        self.fc = nn.Linear(input_size, num_labels)

    def forward(self, x):
        return self.fc(x)

# Modify the DistilRoBERTa model to include the classification head
class DistilRobertaForSentimentClassification(nn.Module):
    def __init__(self, base_model, num_labels):
        super(DistilRobertaForSentimentClassification, self).__init__()
        classifier = SentimentClassifier(input_size=base_model.config.hidden_size, num_labels=num_labels)
        self.base_model = base_model
        self.classifier = classifier
        self.num_labels = num_labels
    def forward(self, input_ids, attention_mask, labels=None):
        # Get output from DistilRoBERTa base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the pooled output for classification
        pooled_output = outputs.pooler_output

        # Forward through the classification head
        logits = self.classifier(pooled_output)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return ModelOutputWithLoss(logits=logits, loss=loss)


