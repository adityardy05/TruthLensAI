import pandas as pd
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

def build_faiss_index(input_csv, index_output_path, metadata_output_path):
    """
    Loads cleaned news articles, generates dense embeddings, builds a FAISS 
    vector database, and saves the index and metadata for future retrieval.
    """
    
    # 1. Load the cleaned dataset using pandas
    print(f"Loading cleaned data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 2. Drop any rows where 'cleaned_text' is missing (NaN), which would crash the embedder
    df = df.dropna(subset=['cleaned_text'])
    
    # 3. Extract the text column into a Python list so we can pass it to the model
    texts = df['cleaned_text'].tolist()
    
    # 4. Load the Sentence Transformer model
    # 'all-MiniLM-L6-v2' is a fast, highly effective model for mapping sentences into a 384-dimensional vector space
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 5. Generate embeddings for the entire list of texts
    # show_progress_bar=True helps track progress since this step can take some time
    print("Generating embeddings... (This may take a while depending on dataset size)")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # 6. Convert the embeddings to a NumPy array of type float32 
    # FAISS requires input vectors to be strictly 32-bit floats
    embeddings = np.array(embeddings).astype('float32')
    
    # 7. Get the dimension size of the vectors (for 'all-MiniLM-L6-v2', this is 384)
    dimension = embeddings.shape[1]
    
    # 8. Initialize the FAISS index using L2 distance (Euclidean distance)
    # IndexFlatL2 performs exact, exhaustive search (perfect for smaller datasets)
    print(f"Initializing FAISS index with dimension: {dimension}")
    index = faiss.IndexFlatL2(dimension)
    
    # 9. Add all of our generated embeddings into the FAISS index
    index.add(embeddings)
    print(f"Total vectors added to FAISS index: {index.ntotal}")
    
    # 10. Save the compiled FAISS index to the disk so we don't have to recompute it every time
    faiss.write_index(index, index_output_path)
    print(f"FAISS index saved to {index_output_path}")
    
    # 11. Prepare metadata to save alongside the index. 
    # FAISS only returns numerical IDs (e.g., ID 45). We need to map those IDs back to the original text, title, and URL.
    # We convert the DataFrame into a list of dictionaries (one dictionary per row)
    metadata = df.to_dict(orient='records')
    
    # 12. Save the metadata to a binary file using Python's built-in 'pickle' module
    with open(metadata_output_path, 'wb') as f:
        pickle.dump(metadata, f)
        
    print(f"Metadata saved to {metadata_output_path}")

if __name__ == "__main__":
    # 13. Define file paths corresponding to the VS Code project structure
    # Input comes from our 'data/processed' folder
    INPUT_PATH = "data/processed/cleaned_news.csv"
    
    # Outputs go into our 'index/' folder
    FAISS_INDEX_PATH = "index/fake_news.faiss"
    METADATA_PATH = "index/metadata.pkl"
    
    # 14. Execute the pipeline
    build_faiss_index(INPUT_PATH, FAISS_INDEX_PATH, METADATA_PATH)
