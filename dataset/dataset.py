import pandas as pd

# Load dataset
df = pd.read_csv("Book1.csv")

# Keep needed columns
df = df[['Author', 'Blog']].dropna()

# Function to split text into chunks
def split_blog(text, chunk_size=120):
    words = text.split()
    
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        
        # Ignore very tiny chunks
        if len(chunk.split()) >= 40:
            chunks.append(chunk)
    
    return chunks

# Create expanded dataset
expanded_rows = []

for _, row in df.iterrows():
    author = row['Author']
    blog = str(row['Blog'])
    
    chunks = split_blog(blog, chunk_size=120)
    
    for chunk in chunks:
        expanded_rows.append({
            'Author': author,
            'Blog': chunk
        })

# New dataframe
expanded_df = pd.DataFrame(expanded_rows)

# Save expanded dataset
expanded_df.to_csv("expanded_writer_dataset.csv", index=False)

print("Original rows:", len(df))
print("Expanded rows:", len(expanded_df))