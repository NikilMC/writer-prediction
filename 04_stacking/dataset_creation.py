import pandas as pd
import re

# =========================
# LOAD ORIGINAL DATASET
# =========================
df = pd.read_csv("Original_Dataset.csv")   # original dataset

# Keep required columns
df = df[['Author', 'Blog']].dropna()

# =========================
# CLEAN TEXT
# =========================
def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text

df['Blog'] = df['Blog'].apply(clean_text)

# =========================
# SPLIT INTO SENTENCES
# =========================
def split_into_sentences(text):
    # Split using full stop
    sentences = text.split('.')
    
    # Clean sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    return sentences

# =========================
# CREATE 3-SENTENCE CHUNKS
# =========================
rows = []

for _, row in df.iterrows():
    author = row['Author']
    blog = row['Blog']
    
    sentences = split_into_sentences(blog)
    
    # Group every 3 sentences
    for i in range(0, len(sentences), 3):
        chunk = sentences[i:i+3]
        
        if len(chunk) == 3:   # only full 3-sentence chunks
            chunk_text = ". ".join(chunk) + "."
            
            rows.append({
                'Author': author,
                'Chunk': chunk_text
            })

# =========================
# FINAL DATASET
# =========================
chunked_df = pd.DataFrame(rows)

print("Total samples:", len(chunked_df))

# =========================
# SAVE DATASET
# =========================
chunked_df.to_csv("sentence_chunks_dataset.csv", index=False)

print("✅ Dataset created successfully!")