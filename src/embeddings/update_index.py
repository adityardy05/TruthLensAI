import pandas as pd
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

def update_faiss_index(new_data_csv, index_path, metadata_path):
    """
    Loads new cleaned news articles, generates embeddings, and safely 
    appends them to the existing FAISS index and metadata file without 
    rebuilding the entire database from scratch.
    """
    # 1. Verify the existing index and metadata exist
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("Error: Existing FAISS index or metadata not found.")
        print("Please run generate_index.py first to build the base database.")
        return

    # 2. Load the NEW data
    print(f"Loading new data from {new_data_csv}...")
    df_new = pd.read_csv(new_data_csv)
    df_new = df_new.dropna(subset=['cleaned_text'])
    
    if len(df_new) == 0:
        print("No valid new articles to add.")
        return

    new_texts = df_new['cleaned_text'].tolist()

    # 3. Load the Sentence Transformer model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 4. Generate embeddings for ONLY the new articles
    print(f"Generating embeddings for {len(new_texts)} new articles...")
    new_embeddings = model.encode(new_texts, show_progress_bar=True)
    new_embeddings = np.array(new_embeddings).astype('float32')

    # 5. Load the EXISTING FAISS index
    print(f"Loading existing FAISS index from {index_path}...")
    index = faiss.read_index(index_path)
    
    # Check dimensionality to prevent crashes
    if index.d != new_embeddings.shape[1]:
        print("Error: Dimension mismatch between new embeddings and existing index.")
        return

    # 6. Append the new embeddings to the existing FAISS index
    # This is much faster than rebuilding the entire database
    index.add(new_embeddings)
    print(f"New total vectors in FAISS index: {index.ntotal}")

    # 7. Overwrite the FAISS index with the updated version
    faiss.write_index(index, index_path)
    print(f"Updated FAISS index saved to {index_path}")

    # 8. Load the EXISTING metadata
    print("Updating metadata...")
    with open(metadata_path, 'rb') as f:
        existing_metadata = pickle.load(f)

    # 9. Append the NEW metadata
    new_metadata = df_new.to_dict(orient='records')
    existing_metadata.extend(new_metadata)

    # 10. Overwrite the metadata file to keep it synchronized with FAISS
    # Because FAISS appends sequentially (e.g. index IDs 100, 101, 102), 
    # appending to a Python list keeps everything perfectly mapped.
    with open(metadata_path, 'wb') as f:
        pickle.dump(existing_metadata, f)
        
    print(f"Metadata synchronized. New total records: {len(existing_metadata)}")

if __name__ == "__main__":
    # Example paths
    # Assume we scraped new data and cleaned it, saving to a 'batch_2' file
    NEW_DATA_PATH = "../../data/processed/cleaned_news_batch_2.csv"
    
    FAISS_INDEX_PATH = "../../index/fake_news.faiss"
    METADATA_PATH = "../../index/metadata.pkl"
    
    update_faiss_index(NEW_DATA_PATH, FAISS_INDEX_PATH, METADATA_PATH)
