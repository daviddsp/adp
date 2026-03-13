# HR Topic Classification Project

This project focuses on classifying Human Resources related messages into eight distinct categories using a fine-tuned **DistilBERT** model. The categories include: `employee_benefits`, `employee_training`, `payroll`, `performance_management`, `talent_acquisition`, `tax_services`, `time_and_attendance`, and `other`.

## Key Features
- **Transformer-based Classification**: Uses `distilbert-base-uncased` for efficient and accurate NLP.
- **Data Leakage Correction**: Implements a strict train/validation/test split (70/15/15) to ensure reliable evaluation.
- **Confidence Thresholding**: Predictions below a 60% confidence threshold are flagged as "unsupported", preventing low-quality routing.
- **Ready for Production**: Organized structure with dedicated scripts for training, inference, and testing.

## Project Structure
- `data/`: Contains the datasets (`available_conversations.csv` and `available_topics.csv`).
- `notebooks/`: Exploratory Data Analysis (EDA) and baseline experiments.
- `model_output/`: Stores checkpoints during training.
- `saved_model/`: Contains the final fine-tuned model and tokenizer.
- `imgs/`: Performance visualizations (confusion matrix, metrics comparison, etc.).
- `train.py`: Script to fine-tune the model.
- `predict.py`: Contains the `TopicPredictor` class for making inferences.
- `test_model.py`: Unit tests to verify model behavior and requirements.
- `reporte_adp.md` / `reporte-adp.pdf`: Comprehensive technical report and analysis.
- `requirements.txt`: Project dependencies.

## Installation

### 1. Set up a Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To re-train the model or start from scratch:
```bash
python train.py
```
This script will split the data, fine-tune the model, evaluate it on a blind test set, and save the result in `./saved_model`.

### Running Predictions
You can use the `TopicPredictor` class from `predict.py` in your own scripts:

```python
from predict import TopicPredictor

predictor = TopicPredictor(model_dir="./saved_model")
result = predictor.predict("How can I check my payroll deductions?")
print(result)
# Output: {'status': 'success', 'topic': 'payroll', 'confidence': 0.98}
```

### Running Tests
To verify the model meets the technical requirements (single routing, confidence threshold, accuracy on key domains):
```bash
python test_model.py
```

## Results
The fine-tuned model significantly outperforms the Logistic Regression baseline, achieving high precision in core HR domains like `payroll` and `tax_services`. Visualizations of the performance can be found in the `imgs/` directory.

For a detailed analysis of the methodology, challenges (imbalanced classes), and future improvements, please refer to [reporte_adp.md](reporte_adp.md).
