# Writer Prediction Using BERT-Based Embeddings and Stacking Ensemble

## Overview

This project focuses on **author identification (authorship attribution)** using Natural Language Processing (NLP) and Machine Learning. The objective is to predict the author of a given blog or text passage based on their unique writing style.

Unlike traditional approaches that rely solely on word frequency, this project utilizes **BERT (Bidirectional Encoder Representations from Transformers)** to generate contextual text embeddings. Multiple embedding representations are combined using a **stacking ensemble** to improve prediction performance and generalization.

---

## Features

- Author prediction using contextual BERT embeddings
- Sentence-based text chunking
- Multiple BERT feature extraction techniques
- Stacking ensemble for improved accuracy
- Interactive Streamlit web application
- Model persistence using Pickle (.pkl)

---

## Dataset

The dataset consists of blog posts written by **11 different authors**.

To increase the number of training samples, a **sentence-based chunking strategy** was applied:

- Blogs are split into sentences.
- Every three consecutive sentences are grouped into a single sample.
- The final dataset contains approximately **2,337 text samples**.

---

## Methodology

### 1. Data Preprocessing

- Text cleaning
- Sentence tokenization
- Sentence-based chunking
- Label encoding

---

### 2. BERT Embedding Generation

Each text sample is converted into contextual embeddings using the pre-trained **bert-base-uncased** model.

Three different feature representations are extracted:

- Mean Pooling
- CLS Token Representation
- Max Pooling

Each representation captures different characteristics of the text.

---

### 3. Base Models

Three independent models are trained using different BERT representations:

| Embedding | Model |
|-----------|-------|
| Mean Pooling | MLP Classifier |
| CLS Token | Support Vector Machine (SVM) |
| Max Pooling | MLP Classifier |

---

### 4. Stacking Ensemble

Instead of relying on a single classifier, predictions from all base models are combined.

Pipeline:

```
Input Text
      │
      ▼
BERT Embeddings
      │
 ┌────┼────┐
 ▼    ▼    ▼
Mean CLS  Max
 │     │     │
MLP   SVM   MLP
 └────┼────┘
      ▼
Concatenated Probabilities
      ▼
Logistic Regression
      ▼
Final Author Prediction
```

The Logistic Regression model serves as the **meta-classifier**, producing the final prediction.

---

## Model Evaluation

Evaluation was performed using **5-Fold Stratified Cross Validation** to preserve class distribution across folds.

Performance summary:

| Method | Accuracy |
|---------|----------|
| TF-IDF | ~96% |
| Individual BERT Models | ~90% |
| Stacking Ensemble | ~95% |

Although TF-IDF achieved slightly higher accuracy, BERT embeddings provide superior contextual understanding and better generalization on unseen text.

---

## Project Structure

```
writer-prediction-bert/
│
├── dataset/
│
├── models/
│   ├── mean_pooling_mlp.pkl
│   ├── cls_svm.pkl
│   ├── max_pooling_mlp.pkl
│   └── meta_model.pkl
│
├── phase1/
├── phase2/
├── phase3/
├── phase4/
├── phase4clustering/
├── phase5/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technologies Used

- Python
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- Joblib / Pickle

---

## Installation

Clone the repository.

```bash
git clone https://github.com/your-username/writer-prediction-bert.git
```

Navigate into the project.

```bash
cd writer-prediction-bert
```

Install the required dependencies.

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit application.

```bash
streamlit run app.py
```

The application allows users to:

- Enter blog text
- Generate BERT embeddings
- Predict the most likely author in real time

---

## Future Improvements

- Fine-tune BERT on the dataset instead of using frozen embeddings.
- Expand the dataset with additional authors.
- Experiment with RoBERTa, DeBERTa, and DistilBERT.
- Deploy the application on Streamlit Cloud or Hugging Face Spaces.
- Improve inference speed using optimized embedding generation.

---

## Contributors

- **S. S. Nikil Raagav**
- **Aneesh Sagar Reddy**
- **T. Poojya Sousheel**

---

## License

This project is intended for educational and research purposes.
