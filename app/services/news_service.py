import logging
import httpx
import asyncio
import feedparser
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime

from app.utils.config import NEWS_SOURCES
from app.utils.helpers import run_in_thread

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def fetch_rss_feed(url: str) -> str:
    """Fetch RSS feed content from the given URL"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching RSS feed {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error fetching RSS feed {url}: {e}")
        raise

async def get_news(symbol: str = None, limit: int = 3) -> list:
    """
    Get financial news from various sources with optional symbol filtering
    """
    news_items = []
    tasks = []
    
    # Create tasks for fetching each news source
    for source_name, url in NEWS_SOURCES.items():
        tasks.append(fetch_news_from_source(source_name, url, symbol))
    
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Error fetching news: {result}")
            continue
            
        if result:
            news_items.extend(result)
    
    # Sort news by published date (newest first) and limit the results
    sorted_news = sorted(news_items, key=lambda x: x.get('published_date', datetime.now()), reverse=True)
    return sorted_news[:limit]

async def fetch_news_from_source(source_name: str, url: str, symbol: str = None) -> list:
    """Fetch news from a specific source and filter by symbol if provided"""
    try:
        rss_content = await fetch_rss_feed(url)
        news_items = await run_in_thread(parse_rss_content, rss_content)
        
        # Filter news if symbol is provided
        if symbol:
            filtered_news = []
            for item in news_items:
                # Check if symbol is mentioned in title or summary
                title = item.get('title', '').lower()
                summary = item.get('summary', '').lower()
                if symbol.lower() in title or symbol.lower() in summary:
                    item['source'] = source_name
                    filtered_news.append(item)
            return filtered_news
        else:
            # Add source name to each item
            for item in news_items:
                item['source'] = source_name
            return news_items
            
    except Exception as e:
        logger.error(f"Error fetching news from {source_name}: {e}")
        return []

def parse_rss_content(rss_text: str):
    """Parse RSS content into structured news items"""
    try:
        # Use feedparser to parse RSS content
        feed = feedparser.parse(rss_text)
        
        news_items = []
        for entry in feed.entries:
            # Extract key information from each entry
            title = entry.get('title', '')
            link = entry.get('link', '')
            summary = entry.get('summary', '')
            
            # Clean summary (remove HTML tags)
            if summary:
                soup = BeautifulSoup(summary, 'html.parser')
                summary = soup.get_text(separator=' ', strip=True)
            
            # Parse published date
            published_date = None
            if 'published_parsed' in entry:
                try:
                    published_tuple = entry.published_parsed
                    published_date = datetime(*published_tuple[:6])
                except (TypeError, ValueError):
                    published_date = datetime.now()
            else:
                published_date = datetime.now()
            
            # Create news item
            news_item = {
                'title': title,
                'link': link,
                'summary': summary[:200] + '...' if len(summary) > 200 else summary,
                'published_date': published_date,
                'published_str': published_date.strftime('%Y-%m-%d %H:%M')
            }
            
            news_items.append(news_item)
            
        return news_items
    except Exception as e:
        logger.error(f"Error parsing RSS content: {e}")
        return [] 