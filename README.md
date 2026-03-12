# HR Topic Classification Project

This project uses a fine-tuned **DistilBERT** model to classify Human Resources related messages into eight categories: `employee_benefits`, `employee_training`, `payroll`, `performance_management`, `talent_acquisition`, `tax_services`, `time_and_attendance`, and `other`.

## Project Structure
- `ADP/`: Main directory containing data, results, and source files.
  - `available_conversations.csv`: Training and evaluation data.
  - `available_topics.csv`: Mapping of topic IDs to labels.
  - `notebook.ipynb`: Development and inference notebook.
  - `results/`: Contains model checkpoints (e.g., `checkpoint-360`).
  - `reporte_adp.md`: Technical project report.

## Setup Instructions

To get this project running on your local machine, follow these steps:

### 1. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies. From the `ADP/` directory, run:

```bash
cd ADP
python3 -m venv .venv
```

### 2. Activate the Virtual Environment
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```
- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

### 3. Install Dependencies
Install the required libraries. **Important:** When using `zsh`, you must wrap `transformers[torch]` in quotes to prevent shell errors.

```bash
pip install "transformers[torch]" pandas datasets ipykernel matplotlib seaborn
```

### 4. Setup Jupyter Kernel (Optional)
If you want to use the virtual environment within the `notebook.ipynb` file:

```bash
python -m ipykernel install --user --name=adp-project --display-name "Python (ADP Project)"
```

## Usage

### Running Inference
You can use the `ADP/notebook.ipynb` file to run predictions. The notebook is pre-configured to load the `checkpoint-360` model and run sample classifications.

### Generating the Report
If you need to regenerate the technical report in PDF format (requires `pandoc` and a LaTeX engine like `xelatex`):

```bash
cd ADP
pandoc reporte_adp.md -o reporte-adp.pdf --pdf-engine=xelatex
```

## Results
The fine-tuned model achieved an accuracy of **0.65** on the test set, significantly outperforming the initial Logistic Regression baseline. Full details can be found in `ADP/reporte_adp.md`.
