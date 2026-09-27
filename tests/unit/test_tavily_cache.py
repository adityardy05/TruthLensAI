from types import SimpleNamespace
from unittest.mock import patch

from backend.rag.retriever import TeamARetrievalPipeline, tavily_results_to_docs


class FakeIndex:
    def __init__(self):
        self.ntotal = 1

    def add(self, vectors):
        self.ntotal += len(vectors)


def make_pipeline():
    pipeline = TeamARetrievalPipeline.__new__(TeamARetrievalPipeline)
    pipeline.faiss_index = FakeIndex()
    pipeline.metadata = [{"url": "http://liar-dataset.internal/1", "cleaned_text": "existing"}]
    pipeline.embedding_model = SimpleNamespace(
        encode=lambda texts, convert_to_numpy=True: __import__("numpy").ones(
            (len(texts), 2), dtype="float32"
        )
    )
    pipeline.tfidf_svd_model = None
    pipeline.index_path = "index.faiss"
    pipeline.metadata_path = "metadata.pkl"
    pipeline._index_lock = __import__("threading").RLock()
    pipeline._ingested_hashes = {
        pipeline._content_hash(pipeline.metadata[0]["url"]),
        pipeline._content_hash(pipeline.metadata[0]["cleaned_text"]),
    }
    return pipeline


def test_tavily_results_to_docs_filters_invalid_content():
    docs = tavily_results_to_docs([
        {"url": "https://example.org/a", "content": "article", "score": 0.8},
        {"url": "https://example.org/b", "content": ""},
        {"url": "", "content": "missing URL"},
    ])

    assert len(docs) == 1
    assert docs[0]["origin"] == "tavily_cache"
    assert docs[0]["tavily_score"] == 0.8
    assert docs[0]["rd_score"] is None


@patch("backend.rag.retriever.faiss", SimpleNamespace(normalize_L2=lambda vectors: None))
@patch.object(TeamARetrievalPipeline, "_persist_cache")
def test_cache_deduplicates_by_url_and_content(mock_persist):
    pipeline = make_pipeline()
    summary = pipeline.cache_tavily_results([
        {"source_url": "https://example.org/a", "text_snippet": "new article", "retrieval_score": 0.7},
        {"source_url": "https://example.org/a", "text_snippet": "different text", "retrieval_score": 0.6},
        {"source_url": "https://example.org/b", "text_snippet": "new article", "retrieval_score": 0.5},
    ])

    assert summary == {"cached": 1, "skipped": 2, "failed": 0}
    assert len(pipeline.metadata) == 2
    assert pipeline.metadata[-1]["origin"] == "tavily_cache"
    mock_persist.assert_called_once()