# stack_pop

A collection of four fine-tuned transformer models (BERT, RoBERTa, ALBERT, DistilBERT) trained on 60,000 Stack Overflow questions from Kaggle. This repository provides preprocessed data, model training and evaluation notebooks, and guidance for reproducibility and further research.

---

## 📌 Overview

**stack_pop** aims to provide robust, fine-tuned transformer models for question understanding and related NLP tasks in the programming Q&A domain. The models are trained and evaluated on a large, real-world dataset of Stack Overflow questions, making them suitable for academic, research, and practical applications.

---

## 📋 Contents

- **df_preprocessed**: Preprocessed version of the original Stack Overflow dataset, ready for model training and easier retraining.
- **finetuning.ipynb**: Jupyter notebook for preprocessing, model loading, fine-tuning, and saving models to Hugging Face.
- **model_eval.ipynb**: Jupyter notebook for evaluating all fine-tuned models and comparing their performance.

---

## 🧠 Models

The following transformer models are fine-tuned and included:

- **BERT-base**
- **RoBERTa**
- **ALBERT**
- **DistilBERT**

All models are fine-tuned on the same dataset for consistent benchmarking and comparison.

---

## 📚Dataset

- **Source**: [60,000 Stack Overflow questions from Kaggle](https://www.kaggle.com/datasets/imoore/60k-stack-overflow-questions-with-quality-rate)
- **Preprocessing**: Cleaning, normalization, and formatting performed in `finetuning.ipynb`
- **Saved File**: Preprocessed data is stored as `df_preprocessed` for reproducibility and faster retraining

---

## 🛠️ Getting Started

**Requirements:**

- Python 3.8+
- Jupyter Notebook
- PyTorch
- `transformers`, `datasets`, `pandas`, `scikit-learn`, and other standard ML/NLP libraries

**Setup:**

1. Clone the repository.
2. Install dependencies:
```bash
pip install requirements.txt
```

4. Open `finetuning.ipynb` to preprocess data and fine-tune models.
5. Use `model_eval.ipynb` to evaluate and compare model performance.

---

## ⚙️ Usage

- **Retraining**: Use `df_preprocessed` as your starting dataset for further fine-tuning or experimentation.
- **Model Evaluation**: Run `model_eval.ipynb` to assess model accuracy, F1, and other relevant metrics.
- **Hugging Face Integration**: Models can be uploaded and shared via Hugging Face Model Hub.

---

## 📂 File Structure

| File/Folder         | Description                                                      |
|---------------------|------------------------------------------------------------------|
| df_preprocessed     | Preprocessed Stack Overflow dataset (post-cleaning)              |
| finetuning.ipynb    | Data preprocessing, model loading, fine-tuning, and saving       |
| model_eval.ipynb    | Evaluation and comparison of all fine-tuned models               |

---
## 📊 Results

- Comparative evaluation of all four models is available in `model_eval.ipynb`, including accuracy, F1-score, and other relevant metrics.

