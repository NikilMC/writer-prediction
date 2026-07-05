import pandas as pd
import re
import numpy as np
import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

from transformers import BertTokenizer, BertModel
import torch

# =========================
# LOAD DATA
# =========================
df = pd.read_excel("sentence_chunks_dataset.xlsx")
df = df[['Author', 'Chunk']].dropna()

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text

df['Chunk'] = df['Chunk'].apply(clean_text)

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
df['Author'] = le.fit_transform(df['Author'])

# =========================
# LOAD BERT
# =========================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# =========================
# BERT ENCODING (MEAN POOLING)
# =========================
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

        # ✅ MEAN POOLING (IMPORTANT CHANGE)
        vec = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(vec)

    return np.array(embeddings)

# =========================
# MODEL
# =========================
model = MLPClassifier(max_iter=300, random_state=42)

# =========================
# CROSS VALIDATION
# =========================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for fold, (train_idx, test_idx) in enumerate(kf.split(df['Chunk'], df['Author'])):

    print(f"\nFold {fold+1}")

    X_train = df.iloc[train_idx]['Chunk'].tolist()
    X_test = df.iloc[test_idx]['Chunk'].tolist()

    y_train = df.iloc[train_idx]['Author'].values
    y_test = df.iloc[test_idx]['Author'].values

    # Encode using BERT
    Xtr = bert_encode(X_train)
    Xte = bert_encode(X_test)

    m = copy.deepcopy(model)
    m.fit(Xtr, y_train)

    preds = m.predict(Xte)

    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)

    print(f"Accuracy: {acc:.4f}")

# =========================
# FINAL RESULT
# =========================
print("\nFinal CV Accuracy:", np.mean(accuracies))