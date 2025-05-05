#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module thu thập tin tức chứng khoán từ nhiều nguồn RSS
"""

import logging
import asyncio
import aiohttp
import feedparser
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pytz import timezone
import re
from typing import Dict, List, Any, Optional
from cachetools import TTLCache
import os
import yaml
from rapidfuzz import fuzz  # pip install rapidfuzz
import concurrent.futures
import requests

# Thiết lập logging
logger = logging.getLogger("vnai.news")

# Thiết lập múi giờ
TZ = timezone('Asia/Ho_Chi_Minh')

SOURCE_PRIORITY = {
    "cafef.vn": 3,
    "vnexpress.net": 2,
    "vietnamnet.vn": 2,
    "vietstock.vn": 2,
    "news.google.com": 1,
    "nytimes.com": 1,
    "wsj.com": 1,
}

def is_relevant(content, keywords, symbol=None):
    # Regex match symbol (nếu có)
    if symbol:
        pattern = re.compile(rf'\b{re.escape(symbol)}\b', re.IGNORECASE)
        if pattern.search(content):
            return True
        # Fuzzy match (nâng cao)
        for kw in keywords:
            if fuzz.partial_ratio(kw.lower(), content.lower()) > 80:
                return True
    # Fallback: keyword in content
    return any(kw.lower() in content.lower() for kw in keywords)

def get_source_priority(source):
    for k, v in SOURCE_PRIORITY.items():
        if k in source:
            return v
    return 0

class NewsLoader:
    """
    Lớp thu thập và xử lý tin tức từ các nguồn RSS khác nhau.
    """
    
    def __init__(self, max_connections: int = 5, cache_ttl: int = 21600, cache_maxsize: int = 1000, rss_config_path: str = 'rss_sources.yaml'):
        """
        Khởi tạo NewsLoader với các nguồn RSS, session pool, semaphore và TTLCache.
        cache_ttl: thời gian sống của cache (giây), mặc định 6 giờ
        cache_maxsize: số lượng cache tối đa
        rss_config_path: đường dẫn file cấu hình nguồn RSS (YAML)
        """
        # Nếu có file cấu hình, đọc danh sách nguồn từ file, nếu không thì dùng mặc định
        if os.path.exists(rss_config_path):
            try:
                with open(rss_config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.rss_urls = config.get('rss_urls', [])
                if not self.rss_urls:
                    raise ValueError('rss_urls trong file cấu hình rỗng!')
            except Exception as e:
                logger.warning(f"Không thể đọc file cấu hình RSS: {e}. Sử dụng danh sách mặc định.")
                self.rss_urls = self._get_default_rss_urls()
        else:
            self.rss_urls = self._get_default_rss_urls()
        
        # Từ khóa thị trường chung
        self.market_keywords = [
            # Tiếng Việt
            "thị trường", "chứng khoán", "cổ phiếu", "vn-index", "vnindex", "vn index", 
            "hose", "hnx", "upcom", "trái phiếu", "bluechip", "đầu tư", "tài chính",
            # Tiếng Anh
            "market", "stock", "index", "shares", "trading", "finance", "investment"
        ]
        
        # Từ điển sentiment
        self.sentiment_dict = {
            'positive': [
                # Tiếng Việt
                'tăng', 'tích cực', 'lãi', 'thắng', 'vượt', 'tốt', 'hưởng lợi', 'kỳ vọng',
                'triển vọng', 'phục hồi', 'cải thiện', 'tăng trưởng', 'khởi sắc', 'thành công',
                'mạnh mẽ', 'đột phá', 'bứt phá', 'khuyến nghị', 'mua', 'lạc quan', 'tốt đẹp',
                # Tiếng Anh
                'up', 'rise', 'gain', 'profit', 'positive', 'bullish', 'advance', 'grow',
                'increase', 'improve', 'good', 'strong', 'success', 'opportunity', 'outperform'
            ],
            'negative': [
                # Tiếng Việt
                'giảm', 'tiêu cực', 'lỗ', 'thua', 'thấp', 'đi xuống', 'ảnh hưởng', 'bất lợi',
                'rủi ro', 'khó khăn', 'suy giảm', 'suy thoái', 'nợ', 'đóng cửa', 'thất bại',
                'giảm sút', 'mất giá', 'bán', 'bi quan', 'e ngại', 'nguy cơ', 'áp lực',
                # Tiếng Anh
                'down', 'fall', 'loss', 'negative', 'bearish', 'decline', 'decrease',
                'worsen', 'bad', 'weak', 'fail', 'risk', 'debt', 'underperform'
            ]
        }
        # Tối ưu session pool và giới hạn đồng thời
        self.session = None
        self.max_connections = max_connections
        self.cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
    
    def _get_default_rss_urls(self):
        return [
            # Nguồn tiếng Việt
            "https://cafef.vn/thi-truong-chung-khoan.rss",
            "https://cafef.vn/smart-money.rss",
            "https://cafef.vn/tai-chinh-ngan-hang.rss",
            "https://cafef.vn/doanh-nghiep.rss",
            "https://vnexpress.net/rss/kinh-doanh.rss",
            "https://vietnamnet.vn/rss/kinh-doanh.rss",     
            # Nguồn tiếng Anh
            "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
            "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
            "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
            # Google News RSS (thị trường chung)
            "https://news.google.com/rss/search?q=chứng+khoán+Việt+Nam",
            # Google News RSS (vĩ mô, kinh tế Việt Nam)
            "https://news.google.com/rss/search?q=kinh+tế+Việt+Nam",
            "https://news.google.com/rss/search?q=vĩ+mô+Việt+Nam",
            "https://news.google.com/rss/search?q=lạm+phát+Việt+Nam",
            "https://news.google.com/rss/search?q=GDP+Việt+Nam",
            "https://news.google.com/rss/search?q=ngân+hàng+nhà+nước+Việt+Nam",
            "https://news.google.com/rss/search?q=chính+sách+tiền+tệ+Việt+Nam",
            "https://news.google.com/rss/search?q=thị+trường+thế+giới+kinh+tế",
            "https://news.google.com/rss/search?q=global+economy",
            "https://news.google.com/rss/search?q=macroeconomics+Vietnam",
        ]
    
    def _get_session(self):
        """Tạo session đồng bộ (requests.Session)"""
        if not hasattr(self, '_sync_session') or self._sync_session is None:
            self._sync_session = requests.Session()
        return self._sync_session

    def close(self):
        """Đóng session đồng bộ"""
        if hasattr(self, '_sync_session') and self._sync_session is not None:
            self._sync_session.close()
            self._sync_session = None

    def _get_news(self, symbol: str = None, limit: int = 10, days: int = 7) -> list:
        """
        Thu thập tin tức liên quan đến chứng khoán từ nhiều nguồn RSS (sync)
        """
        # Tạo khóa cache dựa trên symbol và ngày hiện tại
        cache_key = f"news_{symbol}_{limit}_{datetime.now(TZ).strftime('%Y%m%d')}" if symbol else f"news_market_{limit}_{datetime.now(TZ).strftime('%Y%m%d')}"
        
        # Kiểm tra cache
        if cache_key in self.cache:
            logger.debug(f"Sử dụng tin tức từ cache cho {'mã ' + symbol if symbol else 'thị trường chung'}")
            return self.cache[cache_key]
        
        # Từ khóa cho mã cụ thể
        symbol_keywords = []
        if symbol:
            symbol_lower = symbol.lower()
            # Tạo các biến thể của mã để tăng khả năng tìm thấy
            symbol_keywords = [
                symbol_lower,
                f"{symbol_lower} ",
                f" {symbol_lower}",
                f" {symbol_lower} ",
                f"mã {symbol_lower}",
                f"cổ phiếu {symbol_lower}",
                f"{symbol_lower} stock",
                f"{symbol.upper()}"
            ]
        
        # Tạo danh sách để lưu tin tức
        news_list = []
        
        try:
            # Tải và xử lý đồng thời các nguồn RSS
            session = self._get_session()
            tasks = []
            # Nếu tìm kiếm theo symbol, thêm nguồn Google News RSS theo symbol
            rss_urls = list(self.rss_urls)
            if symbol:
                google_news_url = f"https://news.google.com/rss/search?q={symbol}+chứng+khoán+Việt+Nam"
                rss_urls.append(google_news_url)
            for url in rss_urls:
                if symbol and "vietstock.vn" in url:
                    url = f"https://vietstock.vn/rss.aspx?Keyword={symbol.upper()}"
                tasks.append(self.fetch_rss_sync(url, symbol_keywords if symbol else self.market_keywords, symbol is not None, session))
            
            # Kết hợp kết quả
            for result in tasks:
                if isinstance(result, list):
                    news_list.extend(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Error fetching RSS: {result}")
            
            # Loại bỏ các tin trùng lặp dựa trên URL, title, summary
            unique_keys = set()
            unique_news = []
            for news in news_list:
                key = (news.get('link', ''), news.get('title', '').strip().lower(), news.get('summary', '').strip().lower())
                if key not in unique_keys:
                    unique_keys.add(key)
                    unique_news.append(news)
            
            # Sắp xếp unique_news theo ưu tiên nguồn và ngày (mới nhất, nguồn ưu tiên trước)
            sorted_news = sorted(unique_news, 
                                key=lambda x: (get_source_priority(x.get('source', '')), 
                                              self._parse_date(x)), 
                                reverse=True)
            
            # Chỉ lấy tin trong số ngày đã chỉ định
            cutoff_date = datetime.now(TZ) - timedelta(days=days)
            filtered_news = []
            for news in sorted_news:
                try:
                    pub_date = self._parse_date(news)
                    if pub_date >= cutoff_date:
                        news['published'] = pub_date.strftime('%Y-%m-%d %H:%M')
                        filtered_news.append(news)
                except Exception as e:
                    logger.warning(f"Lỗi khi xử lý ngày xuất bản: {e} (tin: {news.get('title', '')[:50]})")
                    continue
            
            # Sắp xếp lại theo ngày mới nhất
            filtered_news = sorted(filtered_news, key=lambda x: x.get("published", ""), reverse=True)
            limited_news = filtered_news[:limit]
            
            result = limited_news if limited_news else [{"title": "⚠️ Không có tin tức", "link": "#", "summary": "", "published": datetime.now(TZ).strftime('%Y-%m-%d %H:%M')}]
            
            # Lưu vào cache
            self.cache[cache_key] = result
            logger.info(f"Thu thập {len(result)} tin tức cho {'mã ' + symbol if symbol else 'thị trường chung'} (limit={limit}, days={days})")
            return result
        except Exception as e:
            logger.error(f"Lỗi tổng thể khi thu thập tin tức: {str(e)}")
            return [{"title": f"⚠️ Lỗi khi thu thập tin tức: {str(e)[:100]}", "link": "#", "summary": "", "published": datetime.now(TZ).strftime('%Y-%m-%d %H:%M')}]

    def get_news(self, symbol: str = None, days: int = 7) -> list:
        """
        Phương thức đồng bộ cho ChatbotAI, thu thập tin tức liên quan (sync)
        """
        return self._get_news(symbol, 10, days)

    def get_market_news(self, days: int = 3) -> list:
        """
        Phương thức đồng bộ để lấy tin tức thị trường chung
        
        Args:
            days: Số ngày gần đây để lấy tin
            
        Returns:
            Danh sách các tin tức thị trường
        """
        return self._get_news(None, 10, days)
        
    def analyze_sentiment(self, news_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Phân tích sentiment từ danh sách tin tức
        
        Args:
            news_list: Danh sách các tin tức
            
        Returns:
            Dictionary với kết quả phân tích sentiment
        """
        if not news_list:
            return {
                "overall": "Trung tính",
                "positive": 50.0,
                "negative": 50.0,
                "details": []
            }
        
        sentiment_scores = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        mixed_count = 0
        
        for news in news_list:
            title = news.get('title', '').lower()
            summary = news.get('summary', '').lower()
            # Tính điểm sentiment riêng cho title và summary
            pos_title = sum(1 for word in self.sentiment_dict['positive'] if word.lower() in title)
            neg_title = sum(1 for word in self.sentiment_dict['negative'] if word.lower() in title)
            pos_summary = sum(1 for word in self.sentiment_dict['positive'] if word.lower() in summary)
            neg_summary = sum(1 for word in self.sentiment_dict['negative'] if word.lower() in summary)
            # Áp dụng trọng số: title 70%, summary 30%
            pos_score = pos_title * 0.7 + pos_summary * 0.3
            neg_score = neg_title * 0.7 + neg_summary * 0.3
            # Xác định sentiment tổng thể
            if pos_score > 0 and neg_score > 0:
                sentiment = "Mixed"
                mixed_count += 1
            elif pos_score > neg_score:
                sentiment = "Tích cực"
                positive_count += 1
            elif neg_score > pos_score:
                sentiment = "Tiêu cực"
                negative_count += 1
            else:
                sentiment = "Trung tính"
                neutral_count += 1
            # Lưu kết quả
            sentiment_scores.append({
                'title': news.get('title', ''),
                'source': news.get('source', ''),
                'sentiment': sentiment,
                'pos_score': pos_score,
                'neg_score': neg_score
            })
        # Tính phần trăm
        total_articles = len(news_list)
        positive_pct = (positive_count / total_articles) * 100 if total_articles > 0 else 0
        negative_pct = (negative_count / total_articles) * 100 if total_articles > 0 else 0
        neutral_pct = (neutral_count / total_articles) * 100 if total_articles > 0 else 0
        mixed_pct = (mixed_count / total_articles) * 100 if total_articles > 0 else 0
        # Xác định sentiment tổng thể
        if positive_pct > negative_pct + 10 and positive_pct > mixed_pct + 10:
            overall = "Tích cực"
        elif negative_pct > positive_pct + 10 and negative_pct > mixed_pct + 10:
            overall = "Tiêu cực"
        elif mixed_pct > max(positive_pct, negative_pct) + 10:
            overall = "Mixed"
        else:
            overall = "Trung tính"
        return {
            "overall": overall,
            "positive": positive_pct,
            "negative": negative_pct,
            "mixed": mixed_pct,
            "neutral": neutral_pct,
            "details": sentiment_scores
        }
    
    def _parse_date(self, news):
        """Parse and standardize date from news item"""
        try:
            date_str = news.get('published', '')
            # Try different date formats
            formats = ['%Y-%m-%d %H:%M', '%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%dT%H:%M:%S%z']
            
            dt = None
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            # If all formats failed, use current time
            if dt is None:
                return datetime.now(TZ)
                
            # Ensure timezone is set
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TZ)
            else:
                dt = dt.astimezone(TZ)
                
            return dt
        except Exception:
            return datetime.now(TZ)

    def fetch_rss_sync(self, url: str, keywords: list, is_symbol_search: bool = False, session=None) -> list:
        """
        Tải và phân tích nguồn RSS cụ thể (sync)
        """
        news_items = []
        if session is None:
            session = requests.Session()
        try:
            response = session.get(url, timeout=10)
            if response.status_code != 200:
                logger.warning(f"Không thể tải RSS từ {url}, mã trạng thái: {response.status_code}")
                return []
            text = response.text
            feed = feedparser.parse(text)
            if not feed.entries:
                logger.debug(f"Không có mục nào trong feed từ {url}")
                return []
            entries_to_process = feed.entries[:10]
            for entry in entries_to_process:
                try:
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "")
                    summary = entry.get("summary", entry.get("description", "")).strip()
                    # Rút gọn summary
                    try:
                        summary_clean = BeautifulSoup(summary, "html.parser").get_text()
                    except Exception as e:
                        logger.warning(f"Không thể phân tích HTML trong tóm tắt: {str(e)}")
                        summary_clean = summary
                    if len(summary_clean) > 200:
                        summary_clean = summary_clean[:197] + "..."
                    published = entry.get("published", entry.get("pubDate", datetime.now(TZ).isoformat()))
                    
                    # Format ngày
                    pub_date = published
                    try:
                        if isinstance(published, str):
                            for fmt in ['%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d %H:%M:%S', '%a, %d %b %y %H:%M:%S %z']:
                                try:
                                    pub_date_dt = datetime.strptime(published, fmt)
                                    if pub_date_dt.tzinfo is None:
                                        pub_date_dt = pub_date_dt.replace(tzinfo=TZ)
                                    else:
                                        pub_date_dt = pub_date_dt.astimezone(TZ)
                                    pub_date = pub_date_dt
                                    break
                                except ValueError:
                                    continue
                        if isinstance(pub_date, datetime):
                            published_fmt = pub_date.strftime('%Y-%m-%d %H:%M')
                        else:
                            published_fmt = str(pub_date)
                    except:
                        published_fmt = str(published)
                    
                    content = f"{title} {summary_clean}".lower()
                    if is_relevant(content, keywords, symbol=title if is_symbol_search else None):
                        news_items.append({
                            "title": title,
                            "link": link,
                            "summary": summary_clean,
                            "published": published_fmt,
                            "source": url.split('/')[2] if '/' in url else url
                        })
                except Exception as e:
                    logger.warning(f"Error processing RSS entry from {url}: {str(e)}")
                    continue
            return news_items
        except Exception as e:
            logger.error(f"Lỗi khi tải RSS từ {url}: {str(e)}")
            return []

# Tạo một instance của NewsLoader để sử dụng trực tiếp
news_loader = NewsLoader()

# Hàm tiện ích để tương thích với code cũ
async def get_news(symbol: str = None, limit: int = 3) -> list:
    """
    Hàm trợ giúp để duy trì tương thích với code cũ.
    """
    # Tạo một session mới cho mỗi lần gọi để tránh xung đột
    loader = NewsLoader()
    try:
        result = loader._get_news(symbol, limit)
        return result
    finally:
        loader.close()