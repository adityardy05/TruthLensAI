import pandas as pd
import re
import spacy
import string
import nltk
from nltk.corpus import stopwords

# 1. Download the NLTK stop words list if not already downloaded
nltk.download('stopwords', quiet=True)

# 2. Load the spaCy English model for lemmatization
# You may need to run `python -m spacy download en_core_web_sm` in your terminal first
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Warning: spaCy model not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None

def clean_text(text):
    """
    Cleans a single string of text by removing URLs, HTML, emojis, numbers, 
    punctuation, and stop words, followed by lemmatization.
    """
    # 3. Handle empty or NaN values safely
    if not isinstance(text, str):
        return ""
        
    # 4. Convert all text to lowercase to ensure uniformity
    text = text.lower()
    
    # 5. Remove HTML tags using a regular expression (anything between < and >)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 6. Remove URLs (http/https links)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 7. Remove emojis and special non-ASCII characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # 8. Remove numbers (digits)
    text = re.sub(r'\d+', '', text)
    
    # 9. Remove punctuation using the built-in string.punctuation list
    # string.punctuation is a string of characters like !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 10. Tokenize (split the text into words) and remove stop words (e.g., 'the', 'is', 'in')
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # 11. Rejoin the words into a single string for spaCy to process
    cleaned_string = ' '.join(words)
    
    # 12. Perform Lemmatization using spaCy (converting words to their base form, e.g., 'running' -> 'run')
    if nlp:
        doc = nlp(cleaned_string)
        # We extract the lemma_ (base word) for each token
        lemmatized_words = [token.lemma_ for token in doc]
        # Rejoin into the final cleaned string
        return ' '.join(lemmatized_words)
    
    # Fallback if spaCy is not installed
    return cleaned_string

def process_datasets(input_csv_list, output_csv, text_column="statement"):
    """
    Loads multiple datasets, concatenates them, applies the text cleaning pipeline to a specific column, 
    and saves the combined result.
    """
    df_list = []
    
    # LIAR dataset columns
    liar_columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'job_title', 
        'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts', 
        'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
    ]

    for csv_file in input_csv_list:
        if os.path.exists(csv_file):
            print(f"Loading dataset from {csv_file}...")
            # Use tab separator and assign columns since LIAR TSV has no header
            df = pd.read_csv(csv_file, sep='\t', header=None, names=liar_columns)
            df_list.append(df)
        else:
            print(f"Warning: File {csv_file} not found, skipping.")
            
    if not df_list:
        raise FileNotFoundError("None of the specified input CSV files were found.")
        
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined total rows: {len(combined_df)}")
    
    # 14. Apply the clean_text function to every row in the specified text column
    print("Cleaning text (this may take a moment)...")
    combined_df['cleaned_text'] = combined_df[text_column].apply(clean_text)
    
    # 15. Drop empty rows that might have been created if an article was only junk/links
    combined_df = combined_df[combined_df['cleaned_text'].str.strip() != ""]
    
    # 16. Save the cleaned DataFrame to a new CSV file
    combined_df.to_csv(output_csv, index=False)
    print(f"Cleaned merged dataset saved to {output_csv} (Total cleaned articles: {len(combined_df)})")

if __name__ == "__main__":
    import os
    # 17. Define paths to LIAR datasets in the data/raw folder
    INPUT_FILES = [
        "data/raw/train.tsv",
        "data/raw/valid.tsv",
        "data/raw/test.tsv"
    ]
    OUTPUT_PATH = "data/processed/cleaned_news.csv"
    
    # 18. Run the pipeline
    process_datasets(INPUT_FILES, OUTPUT_PATH, text_column="statement")
