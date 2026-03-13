import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class TopicPredictor:
    def __init__(self, model_dir="./saved_model", topics_path="data/available_topics.csv"):
        # Load topic mapping
        self.topics_df = pd.read_csv(topics_path)
        self.id2label = dict(zip(self.topics_df['topic_id'], self.topics_df['topic_name']))
        
        # Load model and tokenizer
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model not found at {model_dir}. Run train.py first.")
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval() # Evaluation mode
        
    def predict(self, text, threshold=0.60):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        max_prob, predicted_idx = torch.max(probs, dim=1)
        
        confidence = max_prob.item()
        topic_id = predicted_idx.item()
        
        if confidence < threshold:
            return {
                "status": "unsupported",
                "topic": "Operation not supported",
                "confidence": confidence
            }
            
        return {
            "status": "success",
            "topic": self.id2label[topic_id],
            "confidence": confidence
        }
