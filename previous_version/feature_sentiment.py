from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

def get_sentiment_score(text_list):
    """
    Returns the average probability of POSITIVE sentiment.
    ProsusAI/finbert labels: 0: positive, 1: negative, 2: neutral.
    """
    if not text_list:
        return 0.5 # Default neutral if no news
        
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Index 0 is strictly 'positive' for ProsusAI/finbert
        positive_probs = probs[:, 0]
        
    return positive_probs.mean().item()