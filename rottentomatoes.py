# -*- coding: utf-8 -*-
"""rottenTomatoes.ipynb

Original file is located at
    https://colab.research.google.com/drive/1o_120nSvwRttJGBtbvMAefvu5PNeKO42
"""

# !pip install datasets
# !pip install accelerate -U

from datasets import load_dataset

"""# Load Movie Review Dataset"""

df = load_dataset('rotten_tomatoes')

df

df['train'][0]

df['train'].features

df['train']['text'][40:50]

df['train']['label'][40:50]

"""# Tokenize text"""

from transformers import AutoTokenizer

myTokenizer = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(myTokenizer)

def tokenize(inp):
  return tokenizer(inp['text'], padding='max_length', truncation=True)

tk_df = df.map(tokenize, batched=True)

tk_df

"""# Convert to Pytorch datasets

"""

# Convert datasets to PyTorch datasets
train_dataset = tk_df['train']
val_dataset = tk_df['validation']
test_dataset = tk_df['test']

train_dataset

"""# Init model for classification"""

import torch
from transformers import AutoModelForSequenceClassification

myModel = 'distilbert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(myModel, num_labels=2)

"""# Define Training Args"""

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir='./rottenResults',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./rottenLogs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,

)

"""# Init Trainer"""

from transformers import Trainer, AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Define compute_metrics function
def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, pred, average='binary')
    acc = accuracy_score(labels, pred)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define Optimizer to use
optimizer = AdamW(model.parameters(), lr=5e-5)

# Instantiate Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    optimizers=(optimizer, None),
    compute_metrics=compute_metrics
)

train_result = trainer.train(); train_result

"""# Eval model on Val dataset"""

eval_result = trainer.evaluate()
# print(eval_result)

print("Eval Accuracy:", eval_result['eval_accuracy'])
print("Eval F1-score:", eval_result['eval_f1'])
print("Eval Precision:", eval_result['eval_precision'])
print("Eval Recall:", eval_result['eval_recall'])

"""# Eval model on Test dataset"""

test_results = trainer.evaluate(test_dataset)
# print(trest_results)

print("Test Accuracy:", test_results['eval_accuracy'])
print("Test F1-score:", test_results['eval_f1'])
print("Test Precision:", test_results['eval_precision'])
print("Test Recall:", test_results['eval_recall'])
