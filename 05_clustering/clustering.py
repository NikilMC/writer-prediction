import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['Chunk'] = df['Chunk'].apply(clean_text)

# =========================
# TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X = vectorizer.fit_transform(df['Chunk'])

# =========================
# AGGREGATE PER AUTHOR
# =========================
authors = df['Author'].unique()

author_vectors = []
author_names = []

for author in authors:
    indices = df[df['Author'] == author].index
    avg_vec = X[indices].mean(axis=0)
    
    author_vectors.append(np.asarray(avg_vec).flatten())
    author_names.append(author)

author_matrix = np.vstack(author_vectors)

# =========================
# KMEANS CLUSTERING
# =========================
k = min(5, len(authors))  # safe choice

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(author_matrix)

# =========================
# SHOW CLUSTERS
# =========================
cluster_map = {}

for i, author in enumerate(author_names):
    cluster = clusters[i]
    cluster_map.setdefault(cluster, []).append(author)

print("\nAuthor Clusters:\n")
for cluster, auths in cluster_map.items():
    print(f"Cluster {cluster}: {auths}")

# =========================
# PCA VISUALIZATION
# =========================
pca = PCA(n_components=2)
reduced = pca.fit_transform(author_matrix)

plt.figure(figsize=(8,6))

for i, author in enumerate(author_names):
    plt.scatter(reduced[i, 0], reduced[i, 1])
    plt.text(reduced[i, 0], reduced[i, 1], author)

plt.title("Author Clustering (PCA Visualization)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.grid()
plt.show()