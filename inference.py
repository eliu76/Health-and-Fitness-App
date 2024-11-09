from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("./activity-recognition-model")
tokenizer = BertTokenizer.from_pretrained("./activity-recognition-tokenizer")

# Function to make predictions
def predict_activity(data):
    # Preprocess data (e.g., scaling based on the scaler used during training)
    scaler = StandardScaler()
    data_scaled = scaler.transform(data)
    
    inputs = tokenizer(data_scaled, return_tensors="pt", padding="max_length", truncation=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    return predictions

# Example usage with test data
test_data = pd.read_csv('../data/X_test.txt', delim_whitespace=True, header=None)
predictions = predict_activity(test_data)
print(predictions)