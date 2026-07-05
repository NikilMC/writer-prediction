import pandas as pd
import re
import numpy as np
import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from gensim.models import Word2Vec, FastText

from transformers import BertTokenizer, BertModel
import torch

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# =========================
# DISPLAY SETTINGS
# =========================
pd.set_option('display.max_rows', None)

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("Dataset.xlsx", sheet_name="Sheet2")
df = df[['Author', 'Blog']].dropna()

# Save original author names
df['AuthorName'] = df['Author']

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['Blog'] = df['Blog'].apply(clean_text)

# Blog length
df['length'] = df['Blog'].apply(lambda x: len(x.split()))
median_len = df['length'].median()

# Encode labels
le = LabelEncoder()
df['Author'] = le.fit_transform(df['Author'])

# =========================
# EMBEDDERS
# =========================
def get_tfidf(use_stopwords):
    return TfidfVectorizer(max_features=5000, stop_words='english' if use_stopwords else None)

def get_bow(use_stopwords):
    return CountVectorizer(max_features=5000, stop_words='english' if use_stopwords else None)

# BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def bert_encode(texts):
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().numpy())
    return np.array(embeddings)

# Word2Vec
def word2vec_encode(train, test):
    train_tokens = [t.split() for t in train]
    test_tokens = [t.split() for t in test]

    model = Word2Vec(train_tokens, vector_size=100, min_count=1)

    def vec(tokens):
        valid = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(valid, axis=0) if len(valid) > 0 else np.zeros(100)

    return np.array([vec(t) for t in train_tokens]), np.array([vec(t) for t in test_tokens])

# FastText
def fasttext_encode(train, test):
    train_tokens = [t.split() for t in train]
    test_tokens = [t.split() for t in test]

    model = FastText(train_tokens, vector_size=100, min_count=1)

    def vec(tokens):
        valid = [model.wv[w] for w in tokens]
        return np.mean(valid, axis=0) if len(valid) > 0 else np.zeros(100)

    return np.array([vec(t) for t in train_tokens]), np.array([vec(t) for t in test_tokens])

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

embedders = ["TF-IDF", "BoW", "BERT", "Word2Vec", "FastText"]

# =========================
# CROSS VALIDATION
# =========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for use_sw in [True, False]:

    for emb in embedders:
        print(f"\nEmbedder: {emb} | Stopwords: {use_sw}")

        for model_name, base_model in models.items():

            if model_name == "NaiveBayes" and emb in ["BERT", "Word2Vec", "FastText"]:
                continue

            fold_acc = []
            short_acc = []
            long_acc = []

            for train_idx, test_idx in kf.split(df['Blog'], df['Author']):

                model = copy.deepcopy(base_model)

                X_train = df.iloc[train_idx]
                X_test = df.iloc[test_idx]

                y_train = X_train['Author'].values
                y_test = X_test['Author'].values

                # Embedding
                if emb == "TF-IDF":
                    vec = get_tfidf(use_sw)
                    Xtr = vec.fit_transform(X_train['Blog'])
                    Xte = vec.transform(X_test['Blog'])

                elif emb == "BoW":
                    vec = get_bow(use_sw)
                    Xtr = vec.fit_transform(X_train['Blog'])
                    Xte = vec.transform(X_test['Blog'])

                elif emb == "BERT":
                    Xtr = bert_encode(X_train['Blog'].tolist())
                    Xte = bert_encode(X_test['Blog'].tolist())

                elif emb == "Word2Vec":
                    Xtr, Xte = word2vec_encode(X_train['Blog'], X_test['Blog'])

                elif emb == "FastText":
                    Xtr, Xte = fasttext_encode(X_train['Blog'], X_test['Blog'])

                model.fit(Xtr, y_train)
                preds = model.predict(Xte)

                acc = accuracy_score(y_test, preds)
                fold_acc.append(acc)

                short_mask = X_test['length'] <= median_len
                long_mask = X_test['length'] > median_len

                if short_mask.sum() > 0:
                    short_acc.append(accuracy_score(y_test[short_mask], preds[short_mask]))

                if long_mask.sum() > 0:
                    long_acc.append(accuracy_score(y_test[long_mask], preds[long_mask]))

            results.append({
                "Embedder": emb,
                "Model": model_name,
                "Stopwords": "Yes" if use_sw else "No",
                "CV Accuracy": np.mean(fold_acc),
                "Short Blog Acc": np.mean(short_acc) if short_acc else None,
                "Long Blog Acc": np.mean(long_acc) if long_acc else None
            })

# =========================
# RESULTS
# =========================
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="CV Accuracy", ascending=False).reset_index(drop=True)

print("\nFINAL RESULTS:\n")
print(results_df)

# Save results
results_df.to_excel("final_results.xlsx", index=False)

# =========================
# WORD CLOUD
# =========================
all_text = " ".join(df['Blog'])

wc = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud - All Blogs")
plt.show()

# Per author (using real names)
authors = df['AuthorName'].unique()

for author in authors:
    text = " ".join(df[df['AuthorName'] == author]['Blog'])
    
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud - {author}")
    plt.show()