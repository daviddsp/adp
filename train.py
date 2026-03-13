import pandas as pd
import numpy as np
import torch
import transformers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import Dataset
import os

# Suppress visual warnings
transformers.logging.set_verbosity_error()

def load_and_split_data(data_path="data/available_conversations.csv"):
    df = pd.read_csv(data_path)
    
    # 1. Separate Train (70%) and Temp (30%)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df['message'], df['topic_id'], test_size=0.3, random_state=42, stratify=df['topic_id']
    )
    
    # 2. Separate Temp into Validation (15%) and Test (15%)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels

def main():
    print("Starting training pipeline...")
    
    # 1. Load data
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_split_data()
    
    # 2. Tokenization
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    train_ds = Dataset.from_dict({"text": train_texts.tolist(), "label": train_labels.tolist()})
    val_ds = Dataset.from_dict({"text": val_texts.tolist(), "label": val_labels.tolist()})
    test_ds = Dataset.from_dict({"text": test_texts.tolist(), "label": test_labels.tolist()})
    
    tokenized_train = train_ds.map(tokenize_function, batched=True)
    tokenized_val = val_ds.map(tokenize_function, batched=True)
    tokenized_test = test_ds.map(tokenize_function, batched=True)
    
    # 3. Model Configuration
    model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=8)
    
    training_args = TrainingArguments(
        output_dir="./model_output",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=10, 
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_pin_memory=False
    )
    
    # 4. Training using VALIDATION for evaluation, not TEST
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val, 
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("Training model...")
    trainer.train()
    
    # 5. Final evaluation on TEST SET (totally blind to the model until now)
    print("\nEvaluating on Test Set (Isolated)...")
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=-1)
    
    print("\nFinal Report (Test Set):")
    print(classification_report(test_labels, preds))
    
    # 6. Save final model and tokenizer
    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("Model saved successfully in './saved_model'")

if __name__ == "__main__":
    main()