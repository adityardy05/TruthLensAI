import os
import sys
from flask import Flask, request, jsonify

# Add the 'src' directory to the Python path so we can import our RetrievalAgent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.embeddings.retriever import RetrievalAgent

app = Flask(__name__)

# Define paths to our index and metadata
# We use relative paths based on the location of this app.py file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE_DIR, "index", "fake_news.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "index", "metadata.pkl")

# Initialize the agent globally so it stays in memory while the server runs
try:
    print("Starting Flask API. Initializing Retrieval Agent...")
    agent = RetrievalAgent(INDEX_PATH, METADATA_PATH)
except FileNotFoundError:
    print("Warning: FAISS index or metadata not found. Run generate_index.py first.")
    agent = None

@app.route('/health', methods=['GET'])
def health_check():
    """Simple endpoint to verify the API is running."""
    if agent:
        return jsonify({"status": "healthy", "message": "Retrieval API is active."}), 200
    else:
        return jsonify({"status": "error", "message": "Index not loaded."}), 503

@app.route('/retrieve', methods=['POST'])
def retrieve():
    """
    POST endpoint that accepts a JSON query and returns Top-K retrieved articles.
    Expected JSON body: {"query": "The claim text here", "top_k": 5}
    """
    if not agent:
        return jsonify({"error": "Retrieval Agent is not initialized."}), 500

    # 1. Check if the request contains valid JSON
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON."}), 400
        
    data = request.get_json()
    
    # 2. Extract the 'query' from the JSON body
    query = data.get("query")
    if not query:
        return jsonify({"error": "'query' field is required in the JSON body."}), 400
        
    # 3. Extract the 'top_k' parameter (default to 5 if not provided)
    top_k = data.get("top_k", 5)
    
    try:
        # Ensure top_k is an integer
        top_k = int(top_k)
    except ValueError:
        return jsonify({"error": "'top_k' must be an integer."}), 400

    try:
        # 4. Search the FAISS database using our agent
        results = agent.search(query, top_k=top_k)
        
        # 5. Return the retrieved articles and their similarity scores
        return jsonify({
            "query": query,
            "top_k_requested": top_k,
            "retrieved_articles": results
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"An error occurred during retrieval: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the Flask app on localhost (127.0.0.1) port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
