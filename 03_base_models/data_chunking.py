import pandas as pd
import re

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Original_Dataset.csv")

# Keep only needed columns
df = df[['Author', 'URL', 'Blog']]

# =========================
# REMOVE EMPTY ROWS
# =========================
df = df.dropna(subset=['Author', 'Blog'])

# Remove rows where Blog is just empty/whitespace
df = df[df['Blog'].str.strip() != ""]

# Reset index
df = df.reset_index(drop=True)

print("Cleaned rows:", len(df))

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['Blog'] = df['Blog'].apply(clean_text)

# =========================
# CREATE BLOG ID
# =========================
df['BlogID'] = df['URL']

# =========================
# CHUNK FUNCTION (WITH OVERLAP 🔥)
# =========================
def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = words[i:i+chunk_size]

        if len(chunk) >= 30:   # avoid tiny chunks
            chunks.append(" ".join(chunk))

    return chunks

# =========================
# CREATE FINAL CHUNKED DATASET
# =========================
rows = []

for _, row in df.iterrows():
    chunks = chunk_text(row['Blog'])

    for chunk in chunks:
        rows.append({
            'Author': row['Author'],
            'BlogID': row['BlogID'],
            'Chunk': chunk
        })

chunked_df = pd.DataFrame(rows)

print("Total chunks:", len(chunked_df))

# =========================
# SAVE FINAL DATASET
# =========================
chunked_df.to_csv("chunked_dataset.csv", index=False)

print("✅ Chunked dataset saved!")