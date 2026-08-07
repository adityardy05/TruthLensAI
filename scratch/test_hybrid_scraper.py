import os
import sys

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.scraping.tavily_scraper import HybridScraper

def test_scraper_unit():
    print("Testing Trafilatura standalone extraction unit test...")
    scraper = HybridScraper(api_key="mock_key_for_unit_test", min_content_length=400)
    
    # Test URL extraction using trafilatura directly
    test_url = "https://www.example.com"
    extracted = scraper._scrape_with_trafilatura(test_url)
    print(f"Extracted content from {test_url}:")
    print(repr(extracted))
    print("Unit test finished successfully!")

if __name__ == "__main__":
    test_scraper_unit()
