import pandas as pd
import re
import numpy as np
import copy
import os
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from gensim.models import Word2Vec, FastText

from transformers import BertTokenizer, BertModel
import torch

# ==========================================
# CREATE SAVE DIRECTORY
# ==========================================
os.makedirs("saved_models", exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_excel("sentence_chunks_dataset.xlsx")

# ==========================================
# CLEAN TEXT
# ==========================================
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()

df['Chunk'] = df['Chunk'].apply(clean_text)

# ==========================================
# LENGTH FEATURE
# ==========================================
df['length'] = df['Chunk'].apply(
    lambda x: len(x.split())
)

median_len = df['length'].median()

# ==========================================
# LABEL ENCODING
# ==========================================
le = LabelEncoder()

df['AuthorEncoded'] = le.fit_transform(df['Author'])

joblib.dump(
    le,
    "saved_models/label_encoder.pkl"
)

# ==========================================
# EMBEDDERS
# ==========================================
def get_tfidf(use_stopwords):

    return TfidfVectorizer(
        max_features=5000,
        stop_words='english' if use_stopwords else None
    )

def get_bow(use_stopwords):

    return CountVectorizer(
        max_features=5000,
        stop_words='english' if use_stopwords else None
    )

# ==========================================
# BERT
# ==========================================
tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased'
)

bert_model = BertModel.from_pretrained(
    'bert-base-uncased'
)

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

        cls_embedding = (
            outputs.last_hidden_state[:, 0, :]
            .squeeze()
            .numpy()
        )

        embeddings.append(cls_embedding)

    return np.array(embeddings)

# ==========================================
# WORD2VEC
# ==========================================
def word2vec_encode(train, test):

    train_tokens = [t.split() for t in train]
    test_tokens = [t.split() for t in test]

    w2v = Word2Vec(
        train_tokens,
        vector_size=100,
        min_count=1
    )

    def vec(tokens):

        valid = [
            w2v.wv[w]
            for w in tokens
            if w in w2v.wv
        ]

        if len(valid) == 0:
            return np.zeros(100)

        return np.mean(valid, axis=0)

    Xtr = np.array([
        vec(t)
        for t in train_tokens
    ])

    Xte = np.array([
        vec(t)
        for t in test_tokens
    ])

    return Xtr, Xte

# ==========================================
# FASTTEXT
# ==========================================
def fasttext_encode(train, test):

    train_tokens = [t.split() for t in train]
    test_tokens = [t.split() for t in test]

    ft = FastText(
        train_tokens,
        vector_size=100,
        min_count=1
    )

    def vec(tokens):

        valid = [ft.wv[w] for w in tokens]

        if len(valid) == 0:
            return np.zeros(100)

        return np.mean(valid, axis=0)

    Xtr = np.array([
        vec(t)
        for t in train_tokens
    ])

    Xte = np.array([
        vec(t)
        for t in test_tokens
    ])

    return Xtr, Xte

# ==========================================
# MODELS
# ==========================================
models = {

    "LinearSVM": LinearSVC(),

    "DecisionTree": DecisionTreeClassifier(),

    "RandomForest": RandomForestClassifier(),

    "AdaBoost": AdaBoostClassifier(),

    "NaiveBayes": MultinomialNB(),

    "MLP": MLPClassifier(
        max_iter=300,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss'
    ),

    "CatBoost": CatBoostClassifier(
        verbose=0
    )
}

embedders = [
    "TF-IDF",
    "BoW",
    "BERT",
    "Word2Vec",
    "FastText"
]

# ==========================================
# CROSS VALIDATION
# ==========================================
kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

# ==========================================
# TRAINING LOOP
# ==========================================
for use_sw in [True, False]:

    for emb in embedders:

        print(f"\nEmbedder: {emb} | Stopwords: {use_sw}")

        for model_name, base_model in models.items():

            # NB can't handle negatives/dense embeddings
            if model_name == "NaiveBayes" and emb in [
                "BERT",
                "Word2Vec",
                "FastText"
            ]:
                continue

            fold_acc = []
            short_acc = []
            long_acc = []

            for train_idx, test_idx in kf.split(
                df['Chunk'],
                df['AuthorEncoded']
            ):

                model = copy.deepcopy(base_model)

                X_train = df.iloc[train_idx]
                X_test = df.iloc[test_idx]

                y_train = X_train['AuthorEncoded'].values
                y_test = X_test['AuthorEncoded'].values

                # ======================================
                # EMBEDDINGS
                # ======================================
                if emb == "TF-IDF":

                    vec = get_tfidf(use_sw)

                    Xtr = vec.fit_transform(
                        X_train['Chunk']
                    )

                    Xte = vec.transform(
                        X_test['Chunk']
                    )

                elif emb == "BoW":

                    vec = get_bow(use_sw)

                    Xtr = vec.fit_transform(
                        X_train['Chunk']
                    )

                    Xte = vec.transform(
                        X_test['Chunk']
                    )

                elif emb == "BERT":

                    Xtr = bert_encode(
                        X_train['Chunk'].tolist()
                    )

                    Xte = bert_encode(
                        X_test['Chunk'].tolist()
                    )

                elif emb == "Word2Vec":

                    Xtr, Xte = word2vec_encode(
                        X_train['Chunk'],
                        X_test['Chunk']
                    )

                elif emb == "FastText":

                    Xtr, Xte = fasttext_encode(
                        X_train['Chunk'],
                        X_test['Chunk']
                    )

                # ======================================
                # TRAIN
                # ======================================
                model.fit(Xtr, y_train)

                preds = model.predict(Xte)

                acc = accuracy_score(
                    y_test,
                    preds
                )

                fold_acc.append(acc)

                # ======================================
                # SHORT / LONG
                # ======================================
                short_mask = (
                    X_test['length'] <= median_len
                )

                long_mask = (
                    X_test['length'] > median_len
                )

                if short_mask.sum() > 0:

                    short_acc.append(
                        accuracy_score(
                            y_test[short_mask],
                            preds[short_mask]
                        )
                    )

                if long_mask.sum() > 0:

                    long_acc.append(
                        accuracy_score(
                            y_test[long_mask],
                            preds[long_mask]
                        )
                    )

            avg_acc = np.mean(fold_acc)

            result = {
                "Embedder": emb,
                "Model": model_name,
                "Stopwords": use_sw,
                "CV Accuracy": avg_acc,
                "Short Accuracy": np.mean(short_acc),
                "Long Accuracy": np.mean(long_acc)
            }

            results.append(result)

            print(result)

# ==========================================
# RESULTS
# ==========================================
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="CV Accuracy",
    ascending=False
)

results_df.to_excel(
    "final_results.xlsx",
    index=False
)

print("\nSaved results to final_results.xlsx")

# ==========================================
# SAVE TOP 3 MODELS
# ==========================================
top3 = results_df.head(3)

print("\nTraining Top 3 Models on FULL dataset...")

for idx, row in top3.iterrows():

    emb = row['Embedder']
    model_name = row['Model']
    use_sw = row['Stopwords']

    model = copy.deepcopy(
        models[model_name]
    )

    # ======================================
    # FINAL TRAINING
    # ======================================
    if emb == "TF-IDF":

        vec = get_tfidf(use_sw)

        X = vec.fit_transform(df['Chunk'])

        model.fit(X, df['AuthorEncoded'])

        saved_obj = {
            "vectorizer": vec,
            "model": model
        }

    elif emb == "BoW":

        vec = get_bow(use_sw)

        X = vec.fit_transform(df['Chunk'])

        model.fit(X, df['AuthorEncoded'])

        saved_obj = {
            "vectorizer": vec,
            "model": model
        }

    elif emb == "BERT":

        X = bert_encode(df['Chunk'].tolist())

        model.fit(X, df['AuthorEncoded'])

        saved_obj = {
            "model": model,
            "embedder": "BERT"
        }

    elif emb == "Word2Vec":

        X_tokens = [
            t.split()
            for t in df['Chunk']
        ]

        w2v = Word2Vec(
            X_tokens,
            vector_size=100,
            min_count=1
        )

        def vec_fn(tokens):

            valid = [
                w2v.wv[w]
                for w in tokens
                if w in w2v.wv
            ]

            if len(valid) == 0:
                return np.zeros(100)

            return np.mean(valid, axis=0)

        X = np.array([
            vec_fn(t)
            for t in X_tokens
        ])

        model.fit(X, df['AuthorEncoded'])

        saved_obj = {
            "model": model,
            "w2v": w2v
        }

    elif emb == "FastText":

        X_tokens = [
            t.split()
            for t in df['Chunk']
        ]

        ft = FastText(
            X_tokens,
            vector_size=100,
            min_count=1
        )

        def vec_fn(tokens):

            valid = [ft.wv[w] for w in tokens]

            return np.mean(valid, axis=0)

        X = np.array([
            vec_fn(t)
            for t in X_tokens
        ])

        model.fit(X, df['AuthorEncoded'])

        saved_obj = {
            "model": model,
            "fasttext": ft
        }

    save_path = (
        f"saved_models/"
        f"{model_name}_{emb}.pkl"
    )

    joblib.dump(
        saved_obj,
        save_path
    )

    print(f"Saved: {save_path}")

print("\nDONE")