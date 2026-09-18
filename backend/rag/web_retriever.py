import os
import sys
from typing import List, Dict, Any, Optional

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    trafilatura = None
    HAS_TRAFILATURA = False

try:
    # tavily-python >= 0.2.x uses TavilyClient; v0.1.x uses Client
    try:
        from tavily import TavilyClient
    except ImportError:
        from tavily import Client as TavilyClient
    HAS_TAVILY = True
except ImportError:
    TavilyClient = None
    HAS_TAVILY = False

class HybridScraper:
    """
    Hybrid Web Scraper that uses Tavily Search API for evidence retrieval,
    with a smart fallback to Trafilatura to extract full web article content
    whenever the content returned by Tavily is small/brief.
    """

    def __init__(self, api_key: Optional[str] = None, min_content_length: int = 400):
        """
        Initialize the scraper.
        
        :param api_key: Tavily API Key. If None, reads from TAVILY_API_KEY env variable.
        :param min_content_length: Minimum character length threshold for Tavily content.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.min_content_length = min_content_length

    def search_and_extract(
        self, 
        query: str, 
        max_results: int = 3, 
        search_depth: str = "advanced"
    ) -> List[Dict[str, Any]]:
        """
        Searches the web for evidence related to the given query.
        
        - If Tavily content >= min_content_length: uses Tavily content directly.
        - If Tavily content < min_content_length: uses Trafilatura to scrape the full web page.
        
        :param query: The claim or search text.
        :param max_results: Maximum number of search results to return.
        :param search_depth: Tavily search depth ('basic' or 'advanced').
        :return: List of evidence dictionaries formatted for RAG pipeline.
        """
        if not HAS_TAVILY:
            print("Warning: tavily package is not installed. Web retrieval disabled.")
            return []

        if not self.api_key:
            print("Warning: Tavily API Key is not set. Web retrieval disabled.")
            return []

        client = TavilyClient(api_key=self.api_key)
        print(f"Executing Tavily search for query: '{query}'...")
        
        # 1. Perform Tavily Search
        response = client.search(
            query=query, 
            max_results=max_results, 
            search_depth=search_depth
        )
        
        raw_results = response.get("results", [])
        evidence_list = []

        # 2. Process each result
        for idx, item in enumerate(raw_results, start=1):
            url = item.get("url", "")
            title = item.get("title", "")
            tavily_content = item.get("content", "") or item.get("raw_content", "") or ""
            score = item.get("score", 0.0)
            published_date = item.get("published_date", None)

            content_len = len(tavily_content.strip())

            # Rule: If content is sufficient, use Tavily directly.
            if content_len >= self.min_content_length:
                final_text = tavily_content.strip()
                method = "tavily"
                print(f"[{idx}/{len(raw_results)}] Content length ({content_len} chars) >= threshold ({self.min_content_length}). Using Tavily content.")
            else:
                # Rule: If content is small, use Trafilatura to scrape full webpage from URL.
                print(f"[{idx}/{len(raw_results)}] Content length ({content_len} chars) < threshold ({self.min_content_length}). Scraping URL with Trafilatura...")
                scraped_text = self._scrape_with_trafilatura(url)
                
                if scraped_text and len(scraped_text.strip()) > 0:
                    final_text = scraped_text.strip()
                    method = "trafilatura"
                    print(f"  -> Trafilatura successfully extracted {len(final_text)} chars from {url}")
                else:
                    final_text = tavily_content.strip()
                    method = "tavily_fallback"
                    print(f"  -> Trafilatura extraction failed/empty. Falling back to Tavily content ({len(final_text)} chars).")

            evidence_list.append({
                "source_id": f"web_{idx}",
                "source_url": url,
                "title": title,
                "source_type": "web",
                "text_snippet": final_text,
                "retrieval_score": score,
                "published_date": published_date,
                "extraction_method": method
            })

        return evidence_list

    def _scrape_with_trafilatura(self, url: str) -> Optional[str]:
        """
        Helper method to fetch and extract clean article text from a URL using Trafilatura.
        """
        if not HAS_TRAFILATURA:
            return None
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                return trafilatura.extract(downloaded, include_formatting=False, include_links=False)
        except Exception as e:
            print(f"Warning: Failed to scrape {url} with Trafilatura: {e}")
        return None

if __name__ == "__main__":
    # Test script usage
    test_query = "Hillary Clinton health claims 2016"
    test_key = os.getenv("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY_HERE")
    
    if test_key == "YOUR_TAVILY_API_KEY_HERE":
        print("Usage instructions:")
        print("1. Set your environment variable: $env:TAVILY_API_KEY='tvly-xxx'")
        print("2. Or pass api_key directly: HybridScraper(api_key='tvly-xxx')")
    else:
        scraper = HybridScraper(api_key=test_key, min_content_length=400)
        results = scraper.search_and_extract(test_query, max_results=2)
        import json
        print(json.dumps(results, indent=2))
