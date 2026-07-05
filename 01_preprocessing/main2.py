import pandas as pd
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Embedders
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Models
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# NEW
from gensim.models import Word2Vec, FastText

# BERT
from transformers import BertTokenizer, BertModel
import torch

# =========================
# LOAD + PREPROCESS
# =========================
df = pd.read_excel("Dataset.xlsx", sheet_name="Sheet2")
df = df[['Author', 'Blog']].dropna()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

texts = df['Blog'].apply(clean_text).tolist()
labels = df['Author'].tolist()

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
labels = le.fit_transform(labels)

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# =========================
# EMBEDDERS
# =========================
def get_tfidf():
    return TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')

def get_bow():
    return CountVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')

# -------- BERT --------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def bert_encode(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=128
        )
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)

# -------- Word2Vec --------
def word2vec_encode(train_texts, test_texts):
    tokenized_train = [text.split() for text in train_texts]
    tokenized_test = [text.split() for text in test_texts]

    model = Word2Vec(sentences=tokenized_train, vector_size=100, window=5, min_count=1)

    def get_vector(tokens):
        vectors = [model.wv[word] for word in tokens if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    X_train_vec = np.array([get_vector(tokens) for tokens in tokenized_train])
    X_test_vec = np.array([get_vector(tokens) for tokens in tokenized_test])

    return X_train_vec, X_test_vec

# -------- FastText --------
def fasttext_encode(train_texts, test_texts):
    tokenized_train = [text.split() for text in train_texts]
    tokenized_test = [text.split() for text in test_texts]

    model = FastText(sentences=tokenized_train, vector_size=100, window=5, min_count=1)

    def get_vector(tokens):
        vectors = [model.wv[word] for word in tokens]
        return np.mean(vectors, axis=0) if vectors else np.zeros(100)

    X_train_vec = np.array([get_vector(tokens) for tokens in tokenized_train])
    X_test_vec = np.array([get_vector(tokens) for tokens in tokenized_test])

    return X_train_vec, X_test_vec

# =========================
# MODELS
# =========================
models = {
    "SVM": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "NaiveBayes": MultinomialNB(),
    "MLP": MLPClassifier(max_iter=300),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# =========================
# EMBEDDERS DICTIONARY
# =========================
embedders = {
    "TF-IDF": get_tfidf(),
    "BoW": get_bow(),
    "BERT": "bert",
    "Word2Vec": "w2v",
    "FastText": "ft"
}

# =========================
# EXPERIMENT LOOP
# =========================
results = []

for emb_name, embedder in embedders.items():
    print(f"\nRunning for embedder: {emb_name}")

    # ===== EMBEDDING =====
    if emb_name in ["TF-IDF", "BoW"]:
        X_train_vec = embedder.fit_transform(X_train)
        X_test_vec = embedder.transform(X_test)

    elif emb_name == "BERT":
        X_train_vec = bert_encode(X_train)
        X_test_vec = bert_encode(X_test)

    elif emb_name == "Word2Vec":
        X_train_vec, X_test_vec = word2vec_encode(X_train, X_test)

    elif emb_name == "FastText":
        X_train_vec, X_test_vec = fasttext_encode(X_train, X_test)

    # ===== MODELS =====
    for model_name, model in models.items():
        try:
            print(f"  Model: {model_name}")

            # Skip invalid combos
            if model_name == "NaiveBayes" and emb_name in ["BERT", "Word2Vec", "FastText"]:
                continue

            # Train
            model.fit(X_train_vec, y_train)

            # Predictions
            train_preds = model.predict(X_train_vec)
            test_preds = model.predict(X_test_vec)

            # Metrics
            train_acc = accuracy_score(y_train, train_preds)
            test_acc = accuracy_score(y_test, test_preds)

            precision = precision_score(y_test, test_preds, average='weighted', zero_division=0)
            recall = recall_score(y_test, test_preds, average='weighted', zero_division=0)
            f1 = f1_score(y_test, test_preds, average='weighted', zero_division=0)

            results.append({
                "Embedder": emb_name,
                "Model": model_name,
                "Train Accuracy": train_acc,
                "Test Accuracy": test_acc,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            })

        except Exception as e:
            print(f"Error with {model_name} + {emb_name}: {e}")

# =========================
# RESULTS TABLE
# =========================
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Test Accuracy", "F1 Score", "Precision"],
    ascending=False
).reset_index(drop=True)

print("\nFinal Results:\n")
print(results_df)