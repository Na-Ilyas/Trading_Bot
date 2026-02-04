from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT for specific financial sentiment
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment_score(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # Return average probability of positive sentiment
    return predictions[:, 0].mean().item()