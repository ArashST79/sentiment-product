# Sentiment Analysis of Customer Reviews

## Overview

This project focuses on sentiment analysis of customer reviews using machine learning techniques. The goal is to fine-tune various models on a customer review dataset, compare their performance, and evaluate pre-built models for sentiment analysis. The project is designed to be executed on CPU, utilizing low-size models for efficiency.

## Dataset

The dataset used for this project consists of approximately 20,000 customer comments. The labels assigned to each comment range from -0.9 to 0.9, reflecting a sentiment score. The sentiment scores are mapped to a classification scale resembling a star rating, ranging from 1 star to 5 stars.

## Model Fine-Tuning

### Updated Model Description and Evaluation Process

The project features three distinct models, each serving a specific purpose in sentiment analysis. The models include two fine-tuned models, namely 'distilbert-base-uncased' and 'distilroberta-base,' both trained on project-specific data. Additionally, a pre-built classification model, 'nlptown/bert-base-multilingual-uncased-sentiment' from Hugging Face, fine-tuned on external data, is incorporated for comparison.

#### Fine-Tuned Models

1. **DistilBERT (Uncased):**
   - The 'distilbert-base-uncased' model has been fine-tuned on project-specific data. Its selection is based on its compact size, making it suitable for training and testing on CPU.

2. **DistilRoBERTa (Base):**
   - A second fine-tuned model, 'distilroberta-base,' has been created through the training process outlined in the `trainer.ipynb` notebook. This model complements the DistilBERT variant and provides an alternative for evaluation.

#### Pre-Built Model

3. **Pre-Built BERT Model:**
   - The project incorporates a pre-built classification model, 'nlptown/bert-base-multilingual-uncased-sentiment,' obtained from Hugging Face. This model has been fine-tuned on external data specifically for customer reviews.

#### Running Evaluations

To assess and compare the performance of these models, follow these steps:

1. **Training Fine-Tuned Models:**
   - Navigate to the `notebooks` folder and execute the cells in `trainer.ipynb` to train both fine-tuned DistilBERT and DistilRoBERTa models.

2. **Evaluating Models:**
   - Access the `evaluator.ipynb` notebook for detailed evaluations and comparisons.
   - The notebook includes two distinct comparisons:
     - Evaluation between fine-tuned DistilBERT and the pre-built BERT model ('nlptown/bert-base-multilingual-uncased-sentiment').
     - Evaluation between fine-tuned DistilBERT and fine-tuned DistilRoBERTa.
   - Each comparison provides mean loss and mean difference metrics for each model, accompanied by confusion matrices and random samples with true and predicted labels for enhanced understanding.

3. **Adjusting Data Samples:**
   - Customize the number of data samples for training or evaluation by accessing the `scripts` folder and modifying the `datahandler.py` file.

## Additional Notes

- Ensure that the required dependencies are installed. Refer to the `requirements.txt` file for details.
- The project is optimized for execution on CPU and utilizes smaller-sized models for efficiency.
- Customize hyperparameters, model architecture, or training settings as needed for your specific requirements.

Feel free to explore and experiment with different models and datasets to enhance the project's performance.

---
