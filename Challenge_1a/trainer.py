import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Config
CSV_PATH = './output/training_data.csv'
MODEL_NAME = 'prajjwal1/bert-tiny'
OUTPUT_DIR = './output/fine_tuned_heading_model'
BATCH_SIZE = 16
EPOCHS = 3
MAX_LENGTH = 64

# 1. Load and preprocess data
print('Loading data...')
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['text', 'label'])
texts = df['text'].astype(str).tolist()
labels = df['label'].astype(str).tolist()

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
label2id = {label: int(idx) for idx, label in enumerate(label_encoder.classes_)}
id2label = {int(idx): label for idx, label in enumerate(label_encoder.classes_)}

# Train/val split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels_encoded, test_size=0.1, random_state=42, stratify=labels_encoded)

# 2. Tokenization
print('Tokenizing...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_batch(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors='pt')

class HeadingDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenize_batch(texts)
        self.labels = torch.tensor(labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = HeadingDataset(train_texts, train_labels)
val_dataset = HeadingDataset(val_texts, val_labels)

# 3. Model
print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id)

# 4. Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
    logging_steps=10,
    report_to='none',
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    acc = (preds == labels).mean()
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print('Training...')
trainer.train()

print('Saving model to', OUTPUT_DIR)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print('Done!')
print('Label mapping:', id2label) 