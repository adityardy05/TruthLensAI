import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

def scrape_reuters_rss():
    """
    Scrapes the Reuters World News RSS feed to get the latest reliable news.
    Returns a pandas DataFrame.
    """
    # 1. Define the RSS feed URL for a reliable source (e.g., Reuters)
    rss_url = "http://feeds.reuters.com/Reuters/worldNews"
    
    # 2. Send an HTTP GET request to the RSS feed
    response = requests.get(rss_url)
    
    # 3. Check if the request was successful
    if response.status_code != 200:
        print("Failed to retrieve the RSS feed.")
        return pd.DataFrame()
        
    # 4. Parse the XML content using BeautifulSoup
    soup = BeautifulSoup(response.content, features="xml")
    
    # 5. Find all <item> tags, which represent individual news articles
    articles = soup.findAll("item")
    
    # 6. Initialize an empty list to store the scraped data
    scraped_data = []
    
    # 7. Loop through each article found in the RSS feed
    for a in articles:
        # 8. Extract the title, link, description, and publish date (using .text to get the string inside the tag)
        title = a.find("title").text if a.find("title") else ""
        link = a.find("link").text if a.find("link") else ""
        description = a.find("description").text if a.find("description") else ""
        pub_date = a.find("pubDate").text if a.find("pubDate") else ""
        
        # 9. Append the extracted details as a dictionary into our list
        scraped_data.append({
            "title": title,
            "text": description,
            "url": link,
            "publish_date": pub_date,
            "source": "Reuters"
        })
        
    # 10. Convert the list of dictionaries into a Pandas DataFrame
    df = pd.DataFrame(scraped_data)
    
    # 11. Return the DataFrame
    return df

if __name__ == "__main__":
    print("Scraping reliable news...")
    # 12. Call the scraping function
    df_news = scrape_reuters_rss()
    
    # 13. Save the DataFrame to a CSV file in our raw data folder
    output_path = "../../data/raw/scraped_news.csv"
    df_news.to_csv(output_path, index=False)
    print(f"Successfully scraped {len(df_news)} articles and saved to {output_path}")
