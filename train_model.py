import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import Dataset
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
import torch

# Load the dataset
data = pd.read_csv('../data/Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv')

# Assuming the dataset has 'text' as input and 'label' as the target column
X = data['text']
y = data['label']

# Preprocess the data (you might want to modify this based on the actual structure of your dataset)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values.reshape(-1, 1))  # Adjust depending on the input format
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(pd.DataFrame(X_train).assign(label=y_train.values))
val_dataset = Dataset.from_pandas(pd.DataFrame(X_val).assign(label=y_val.values))

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(y.unique()))  # Adjust the num_labels based on unique labels in y

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
model.save_pretrained("./activity-recognition-model")
tokenizer.save_pretrained("./activity-recognition-tokenizer")