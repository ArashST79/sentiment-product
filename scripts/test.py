from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, DistilBertTokenizer
from data_handler import get_data
from tqdm import tqdm
model_name = 'fine_tuned_distilbert'
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)  # Adjust num_labels based on your task

sentiment_classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)


text_list,labels = get_data()

results = []
for text in tqdm(text_list, desc="Sentiment Analysis"):
    result = sentiment_classifier(text)
    results.append(result)

for text, result in zip(text_list, results):
    print(f'Text: {text}')
    print(f'Sentiment: {result[0]["label"]}')
    print(f'Confidence: {result[0]["score"]}')
    print('\n---\n')
