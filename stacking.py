import pandas as pd
import numpy as np
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from transformers import BertTokenizer, BertModel
import torch

df = pd.read_excel("sentence_chunks_dataset.xlsx")
df = df[['Author', 'Chunk']].dropna()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

df['Chunk'] = df['Chunk'].apply(clean_text)

le = LabelEncoder()
df['Author'] = le.fit_transform(df['Author'])
num_classes = len(le.classes_)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def encode_mean(texts):
    emb = []
    for t in texts:
        inputs = tokenizer(t, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            out = bert_model(**inputs)
        vec = out.last_hidden_state.mean(dim=1).squeeze().numpy()
        emb.append(vec)
    return np.array(emb)

def encode_cls(texts):
    emb = []
    for t in texts:
        inputs = tokenizer(t, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            out = bert_model(**inputs)
        vec = out.last_hidden_state[:,0,:].squeeze().numpy()
        emb.append(vec)
    return np.array(emb)

def encode_max(texts):
    emb = []
    for t in texts:
        inputs = tokenizer(t, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            out = bert_model(**inputs)
        vec = out.last_hidden_state.max(dim=1).values.squeeze().numpy()
        emb.append(vec)
    return np.array(emb)

model1 = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42)
model2 = SVC(kernel='linear', probability=True)
model3 = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=42)

meta_model = LogisticRegression(max_iter=1000)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

final_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df['Chunk'], df['Author'])):

    print(f"\nFold {fold+1}")

    X_train = df.iloc[train_idx]['Chunk'].tolist()
    X_test = df.iloc[test_idx]['Chunk'].tolist()

    y_train = df.iloc[train_idx]['Author'].values
    y_test = df.iloc[test_idx]['Author'].values

    Xtr_mean = encode_mean(X_train)
    Xte_mean = encode_mean(X_test)

    Xtr_cls = encode_cls(X_train)
    Xte_cls = encode_cls(X_test)

    Xtr_max = encode_max(X_train)
    Xte_max = encode_max(X_test)

    model1.fit(Xtr_mean, y_train)
    model2.fit(Xtr_cls, y_train)
    model3.fit(Xtr_max, y_train)

    p1 = model1.predict_proba(Xte_mean)
    p2 = model2.predict_proba(Xte_cls)
    p3 = model3.predict_proba(Xte_max)

    meta_X = np.hstack([p1, p2, p3])

    meta_model.fit(meta_X, y_test)

    preds = meta_model.predict(meta_X)

    acc = accuracy_score(y_test, preds)
    final_scores.append(acc)

    print(f"Accuracy: {acc:.4f}")

print("\nFinal Stacking Accuracy:", np.mean(final_scores))