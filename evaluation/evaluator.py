import numpy as np

class RetrievalEvaluator:
    """
    A specialized class to evaluate the accuracy of the Retrieval Module using 
    standard Information Retrieval metrics.
    """
    
    @staticmethod
    def precision_at_k(retrieved_ids, relevant_ids, k=5):
        """
        Precision@K: Out of the Top-K articles the system retrieved, 
        how many were actually relevant?
        """
        top_k_retrieved = retrieved_ids[:k]
        # Count how many of the retrieved IDs are in the relevant_ids set
        hits = len(set(top_k_retrieved).intersection(set(relevant_ids)))
        return hits / k

    @staticmethod
    def recall_at_k(retrieved_ids, relevant_ids, k=5):
        """
        Recall@K: Out of all the truly relevant articles that exist in the database, 
        what percentage did our system successfully find in the Top-K?
        """
        if not relevant_ids:
            return 0.0
        top_k_retrieved = retrieved_ids[:k]
        hits = len(set(top_k_retrieved).intersection(set(relevant_ids)))
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids_list, relevant_ids_list):
        """
        Mean Reciprocal Rank (MRR): Evaluates where the *first* relevant article appeared.
        If the first relevant article is at rank 1, score = 1.
        If it is at rank 2, score = 1/2. Rank 3 = 1/3, etc.
        """
        reciprocal_ranks = []
        for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
            rank = 0
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant:
                    rank = i + 1
                    break
            
            if rank > 0:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
                
        return np.mean(reciprocal_ranks)

    @staticmethod
    def map_score(retrieved_ids_list, relevant_ids_list):
        """
        Mean Average Precision (MAP): The average of the Precision scores calculated 
        every time a relevant document is found in the ranked list. Rewards systems
        that place all relevant documents at the very top.
        """
        average_precisions = []
        
        for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
            hits = 0
            sum_precisions = 0.0
            
            for i, doc_id in enumerate(retrieved):
                if doc_id in relevant:
                    hits += 1
                    # Calculate precision exactly at this rank position
                    precision_at_this_point = hits / (i + 1.0)
                    sum_precisions += precision_at_this_point
                    
            if not relevant:
                average_precisions.append(0.0)
            else:
                average_precisions.append(sum_precisions / len(relevant))
                
        return np.mean(average_precisions)


if __name__ == "__main__":
    # --- SAMPLE TESTING SCENARIO ---
    print("Running Retrieval Evaluation Simulation...\n")
    
    # Imagine we ran 3 different user queries through our Retrieval Agent
    # For each query, FAISS returned a list of 5 Document IDs
    retrieved_results = [
        [101, 102, 103, 104, 105], # Query 1 Results
        [201, 202, 203, 204, 205], # Query 2 Results
        [301, 302, 303, 304, 305]  # Query 3 Results
    ]
    
    # These are the actual "Ground Truth" IDs (The articles that perfectly answer the queries)
    ground_truth = [
        [101, 103],      # For Query 1: The system got rank 1 and 3 correct.
        [205, 999],      # For Query 2: The system only found one relevant article, and it was dead last.
        [301, 302, 303]  # For Query 3: Perfect retrieval! The top 3 are all correct.
    ]
    
    # 1. Calculate P@5 and R@5 for each query
    print("--- Individual Query Metrics ---")
    for i in range(len(retrieved_results)):
        p5 = RetrievalEvaluator.precision_at_k(retrieved_results[i], ground_truth[i], k=5)
        r5 = RetrievalEvaluator.recall_at_k(retrieved_results[i], ground_truth[i], k=5)
        print(f"Query {i+1} | Precision@5: {p5:.2f} | Recall@5: {r5:.2f}")

    # 2. Calculate global MRR and MAP across all queries
    mrr_score = RetrievalEvaluator.mrr(retrieved_results, ground_truth)
    map_final = RetrievalEvaluator.map_score(retrieved_results, ground_truth)
    
    print("\n--- Global System Performance ---")
    print(f"Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
    print(f"Mean Average Precision (MAP): {map_final:.4f}")
