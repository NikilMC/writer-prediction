import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from lightgbm import LGBMClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("chunked_dataset.csv")

# =========================
# LABEL ENCODING
# =========================
le = LabelEncoder()
df['Author'] = le.fit_transform(df['Author'])
num_classes = len(le.classes_)

# =========================
# EMBEDDING
# =========================
vectorizer = CountVectorizer(
    max_features=8000,
    ngram_range=(1,2),
    stop_words='english'
)

X = vectorizer.fit_transform(df['Chunk'])
y = df['Author'].values
groups = df['BlogID']

n_samples = X.shape[0]

# =========================
# BASE MODELS
# =========================
base_models = [
    MultinomialNB(),
    MLPClassifier(max_iter=300, random_state=42),
    SVC(probability=True, kernel='linear')
]

# =========================
# OOF META FEATURE CREATION
# =========================
gkf = GroupKFold(n_splits=5)

meta_train = np.zeros((n_samples, len(base_models) * num_classes))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):

    print(f"Base Fold {fold+1}")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train = y[train_idx]

    for i, model in enumerate(base_models):

        m = copy.deepcopy(model)
        m.fit(X_train, y_train)

        preds = m.predict_proba(X_val)

        start = i * num_classes
        end = (i + 1) * num_classes

        meta_train[val_idx, start:end] = preds

# =========================
# META MODEL CV (IMPORTANT FIX)
# =========================
meta_scores = []

for fold, (train_idx, test_idx) in enumerate(gkf.split(meta_train, y, groups)):

    print(f"Meta Fold {fold+1}")

    X_train, X_test = meta_train[train_idx], meta_train[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    meta_model = LGBMClassifier(random_state=42)
    meta_model.fit(X_train, y_train)

    preds = meta_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    meta_scores.append(acc)

# =========================
# FINAL RESULT
# =========================
print("\nFinal Stacking Accuracy:", np.mean(meta_scores))