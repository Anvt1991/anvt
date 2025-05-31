import logging
import os
import asyncio
# Nhóm các import thư viện bên ngoài
import feedparser
import httpx
import redis.asyncio as aioredis
import google.generativeai as genai
# Nhóm import telegram
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes, MessageHandler, filters
# Nhóm các import khác
import re
from urllib.parse import urlparse
import unicodedata
import datetime
import pytz
from typing import Dict, List, Any, Optional
import pickle
from functools import wraps
# Import cho việc phát hiện tin trùng lặp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Import cho sentiment analysis tiếng Việt
import numpy as np
import requests
import json
# Import thêm signal để xử lý tín hiệu đóng/khởi động lại
import signal
import sys
import hashlib
# Thêm import cho phát hiện ngôn ngữ và dịch
from langdetect import detect
from googletrans import Translator

# --- 1. Config & setup ---
class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")
    DB_URL = os.getenv("DATABASE_URL")
    ADMIN_ID = int(os.getenv("ADMIN_ID", "1225226589"))
    GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")
    OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "deepseek/deepseek-chat-v3-0324:free")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")
    FEED_URLS = [
        # Google News theo từ khóa
        "https://news.google.com/rss/search?q=kinh+t%E1%BA%BF&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=ch%E1%BB%A9ng+kho%C3%A1n&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=v%C4%A9+m%C3%B4&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=chi%E1%BA%BFn+tranh&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=l%C3%A3i+su%E1%BA%A5t&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=fed&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=tin+n%C3%B3ng&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=%C4%91%E1%BA%A7u+t%C6%B0&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=doanh+nghi%E1%BB%87p&hl=vi&gl=VN&ceid=VN:vi",
        # Chính trị thế giới, quan hệ quốc tế
        "https://news.google.com/rss/search?q=ch%C3%ADnh+tr%E1%BB%8B+th%E1%BA%BF+gi%E1%BB%9Bi&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=geopolitics&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=world+politics&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=international+relations&hl=vi&gl=VN&ceid=VN:vi",
        # Quốc tế (Google News search các nguồn quốc tế)
        "https://news.google.com/rss/search?q=site:bloomberg.com+stock+OR+market+OR+finance&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=site:ft.com+stock+OR+market+OR+finance&hl=vi&gl=VN&ceid=VN:vi",
        # Bổ sung các nguồn Google News RSS tối ưu
        # Chứng khoán Việt Nam từ các báo lớn
        "https://news.google.com/rss/search?q=chứng+khoán+site:cafef.vn&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=chứng+khoán+site:vnexpress.net&hl=vi&gl=VN&ceid=VN:vi",
        # Chính sách vĩ mô
        "https://news.google.com/rss/search?q=chính+sách+vĩ+mô&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=lãi+suất+site:sbv.gov.vn&hl=vi&gl=VN&ceid=VN:vi",
        # Biến động thế giới, tài chính quốc tế
        "https://news.google.com/rss/search?q=market+crash&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=site:reuters.com+economy+OR+policy&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=site:theguardian.com+world+OR+politics&hl=vi&gl=VN&ceid=VN:vi",
        # --- Thêm nguồn PiQ và CNWire ---
        # PiQ Twitter/X (dạng RSS qua Nitter hoặc dịch vụ RSS Twitter nếu có, ví dụ nitter.net)
        "https://nitter.net/piqsuite/rss",  # Nếu không dùng được, cần thay bằng dịch vụ RSS Twitter khác
        # CNWire (Canada Newswire RSS)
        "https://www.newswire.ca/news-releases/news-releases-rss-feed-en.xml",
    ]
    REDIS_TTL = int(os.getenv("REDIS_TTL", "60000"))  # 6h
    NEWS_JOB_INTERVAL = int(os.getenv("NEWS_JOB_INTERVAL", "800"))
    HOURLY_JOB_INTERVAL = int(os.getenv("HOURLY_JOB_INTERVAL", "500"))  # ... phút/lần
    FETCH_LIMIT_DAYS = int(os.getenv("FETCH_LIMIT_DAYS", "1"))  # Chỉ lấy tin 1 ngày gần nhất 
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # Số lần thử lại khi feed lỗi
    MAX_NEWS_PER_CYCLE = int(os.getenv("MAX_NEWS_PER_CYCLE", "1"))  # Tối đa 1 tin mỗi lần
    TIMEZONE = pytz.timezone('Asia/Ho_Chi_Minh')  # Timezone chuẩn cho Việt Nam
    
    # Cấu hình phát hiện tin trùng lặp - nâng cấp
    DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.65"))  # Điều chỉnh ngưỡng phát hiện trùng lặp nội dung
    TITLE_SIMILARITY_THRESHOLD = float(os.getenv("TITLE_SIMILARITY_THRESHOLD", "0.92"))  # Ngưỡng phát hiện tiêu đề tương tự
    LOG_SIMILARITY_DETAILS = os.getenv("LOG_SIMILARITY_DETAILS", "False").lower() == "true"  # Bật/tắt log chi tiết về similarity
    
    RECENT_NEWS_DAYS = int(os.getenv("RECENT_NEWS_DAYS", "2"))  # Số ngày để lấy tin gần đây để so sánh
    
    # Cấu hình phát hiện tin nóng
    HOT_NEWS_KEYWORDS = [
        "khẩn cấp", "tin nóng", "breaking", "khủng hoảng", "crash", "sập", "bùng nổ", "tin nhanh chứng khoán", "trước giờ giao dịch", 
        "shock", "ảnh hưởng lớn", "thảm khốc", "thảm họa", "market crash", "sell off", "VNINDEX", "vnindex", "Trump", "fed", "FED",
        "rơi mạnh", "tăng mạnh", "giảm mạnh", "sụp đổ", "bất thường", "emergency", 
        "urgent", "alert", "cảnh báo", "đột biến", "lịch sử", "kỷ lục", "cao nhất"
    ]
    HOT_NEWS_IMPACT_PHRASES = [
        "tác động mạnh", "ảnh hưởng nghiêm trọng", "thay đổi lớn", "biến động mạnh",
        "trọng điểm", "quan trọng", "đáng chú ý", "đáng lo ngại", "cần lưu ý"
    ]
    
    # Danh sách từ khóa lọc tin tức liên quan (mở rộng)
    RELEVANT_KEYWORDS = [
        # Chính trị, vĩ mô, doanh nghiệp, chứng khoán, chiến tranh 
        "chính trị", "vĩ mô", "doanh nghiệp", "chứng khoán", "chiến tranh", "chính sách", "lãi suất", "fed",
        "phe", "đảng", "chính phủ", "quốc hội", "nhà nước", "bộ trưởng", "thủ tướng", "chủ tịch",
        # Nhóm ngành, bluechip, midcap, thị trường
        "bluechip", "midcap", "ngân hàng", "bất động sản", "thép", "dầu khí", "công nghệ", "bán lẻ",
        "xuất khẩu", "điện", "xây dựng", "thủy sản", "dược phẩm", "logistics", "vận tải", 
        # Các mã chứng khoán, chỉ số
        "vn30", "hnx", "upcom", "vnindex", "cổ phiếu", "thị trường", "tài chính", "kinh tế", 
        "gdp", "lạm phát", "tín dụng", "trái phiếu", "phái sinh", "quỹ etf", 
        # Các mã bluechip VN30
        "fpt", "vnm", "vcb", "ssi", "msn", "mwg", "vic", "vhm", "hpg", "ctg", "bid", "mbb", "stb",
        "hdb", "bvh", "vpb", "nvl", "pdr", "tcb", "tpb", "bcm", "pnj", "acb", "vib", "plx",
        # Các mã midcap, các chỉ báo kinh tế
        "vnm", "cpi", "pmi", "m2", "đầu tư", "gdp", "xuất khẩu", "nhập khẩu", "dự trữ", "dự báo",
        # Từ khóa tài chính quốc tế
        "fed", "ecb", "boj", "pboc", "imf", "world bank", "nasdaq", "dow jones", "s&p", "nikkei",
        "treasury", "usd", "eur", "jpy", "cny", "bitcoin", "crypto", "commodities", "wti", "brent",
        # Mã cổ phiếu nổi bật
        "vnd", "ssi", "hpg", "vic", "vhm", "vnm", "mwg", "ctg", "bid", "tcb", "acb", "vib", "stb", "mbb", "shb",
        # Tên công ty lớn
        "vinamilk", "vietcombank", "vietinbank", "masan", "fpt", "hoa phat", "vietjet", "petro vietnam",
        # Sự kiện kinh tế
        "lạm phát", "tăng trưởng", "giảm phát", "GDP", "CPI", "PMI", "xuất khẩu", "nhập khẩu", "tín dụng", "trái phiếu",
        # Thuật ngữ thị trường
        "bull", "bear", "breakout", "margin", "room ngoại", "ETF", "IPO", "niêm yết", "phát hành", "cổ tức", "chia thưởng",
        # Sự kiện quốc tế
        "fed", "ecb", "boj", "nasdaq", "dow jones", "s&p", "nikkei", "usd", "eur", "jpy", "bitcoin", "crypto",
    ]

# Danh sách từ khóa bổ sung
additional_keywords = []

# Danh sách từ khóa để loại trừ tin không liên quan
EXCLUDE_KEYWORDS = [
    "khuyến mãi", "giảm giá", "mua ngay", "tuyển dụng", "sự kiện", "giải trí", "thể thao", "lifestyle", 
    "du lịch", "ẩm thực", "hot deal", "sale off", "quảng cáo", "đặt hàng", "shoppe", "tiki", "lazada"
]

# --- Kiểm tra biến môi trường bắt buộc ---
REQUIRED_ENV_VARS = ["BOT_TOKEN", "OPENROUTER_API_KEY"]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Redis ---
redis_client = None

# Lưu trữ danh sách user đã được duyệt
approved_users = set()

# --- Flag để kiểm soát việc tắt bot ---
shutdown_flag = False

# --- Cleanup function ---
async def cleanup_resources():
    """Dọn dẹp tài nguyên khi bot tắt"""
    global redis_client, application
    
    logger.info("Đang dọn dẹp tài nguyên trước khi tắt...")
    
    # Dừng job queue nếu đang chạy
    if application and application.job_queue:
        logger.info("Dừng job queue...")
        await application.job_queue.stop()
    
    # Đóng kết nối Redis
    if redis_client:
        logger.info("Đóng kết nối Redis...")
        await redis_client.close()
        redis_client = None
    
    logger.info("Đã dọn dẹp tất cả tài nguyên.")

# --- Signal handlers ---
def signal_handler(sig, frame):
    """Xử lý tín hiệu tắt từ hệ thống"""
    global shutdown_flag
    
    if shutdown_flag:
        logger.warning("Nhận tín hiệu tắt lần hai, tắt ngay lập tức!")
        sys.exit(1)
    
    logger.info(f"Nhận tín hiệu {sig}, chuẩn bị tắt bot...")
    shutdown_flag = True
    
    # Chạy cleanup trong asyncio event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(shutdown())
        else:
            loop.run_until_complete(shutdown())
    except Exception as e:
        logger.error(f"Lỗi khi dọn dẹp tài nguyên: {e}")
        sys.exit(1)

async def shutdown():
    """Xử lý shutdown một cách đồng bộ"""
    global application
    
    logger.info("Bắt đầu quy trình shutdown...")
    
    # Dọn dẹp tài nguyên
    await cleanup_resources()
    
    # Dừng bot nếu đang chạy
    if application:
        logger.info("Dừng bot...")
        if hasattr(application, 'stop'):
            await application.stop()
        application = None
    
    logger.info("Bot đã tắt hoàn toàn.")
    sys.exit(0)

async def is_sent(entry_id):
    return await redis_client.sismember("sent_news", entry_id)

async def mark_sent(entry_id):
    await redis_client.sadd("sent_news", entry_id)
    await redis_client.expire("sent_news", Config.REDIS_TTL)

# --- AI Analysis (Gemini) ---
GEMINI_MODEL = Config.GEMINI_MODEL
OPENROUTER_FALLBACK_MODEL = Config.OPENROUTER_FALLBACK_MODEL
GOOGLE_GEMINI_API_KEY = Config.GOOGLE_GEMINI_API_KEY

async def call_groq_api(prompt, model=None):
    """Gọi Groq API để lấy kết quả AI phân tích"""
    api_key = Config.GROQ_API_KEY
    model = model or Config.GROQ_MODEL
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
            )
            result = response.json()
            if "choices" not in result:
                logging.error(f"Groq API error (model={model}): {result}")
                raise RuntimeError(f"Groq API error: {result}")
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Groq API lỗi: {e}")
        raise e

async def analyze_news(prompt, model=None):
    try:
        # Gọi Google Gemini API chính thức
        genai.configure(api_key=GOOGLE_GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        response = await asyncio.to_thread(model.generate_content, prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API lỗi: {e}, fallback sang OpenRouter {OPENROUTER_FALLBACK_MODEL}")
        # Fallback sang OpenRouter
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",},
                    json={
                        "model": OPENROUTER_FALLBACK_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7
                    }
                )
                result = response.json()
                if "choices" not in result:
                    logging.error(f"OpenRouter API error (model={OPENROUTER_FALLBACK_MODEL}): {result}")
                    raise RuntimeError(f"OpenRouter API error: {result}")
                return result["choices"][0]["message"]["content"]
        except Exception as e2:
            logging.error(f"OpenRouter fallback cũng lỗi: {e2}, thử tiếp Groq...")
            # Fallback sang Groq
            try:
                return await call_groq_api(prompt)
            except Exception as e3:
                logging.error(f"Groq fallback cũng lỗi: {e3}")
                raise e3

# --- Extract sentiment from AI result ---
def extract_sentiment_rule_based(ai_summary):
    """Extract sentiment from AI summary using rule-based approach"""
    sentiment = "Trung lập"  # Default
    try:
        # Tìm kiếm dòng có chứa 'cảm xúc'
        for line in ai_summary.splitlines():
            line_lower = line.lower()
            if "cảm xúc:" in line_lower or "sentiment:" in line_lower:
                sentiment_text = line.split(":")[-1].strip().lower()
                if "tích cực" in sentiment_text or "positive" in sentiment_text:
                    return "Tích cực"
                elif "tiêu cực" in sentiment_text or "negative" in sentiment_text:
                    return "Tiêu cực"
                else:
                    return "Trung lập"
        # Nếu không tìm thấy dòng cảm xúc, dùng regex tìm toàn văn bản
        text = ai_summary.lower()
        if re.search(r"(tích cực|positive|lạc quan|upbeat|bullish)", text):
            return "Tích cực"
        elif re.search(r"(tiêu cực|negative|bi quan|bearish|lo ngại|lo lắng)", text):
            return "Tiêu cực"
    except Exception as e:
        logging.warning(f"Lỗi khi parse sentiment rule-based: {e}")
    return sentiment

async def extract_sentiment(ai_summary):
    """Extract sentiment from AI summary, chỉ sử dụng rule-based"""
    return extract_sentiment_rule_based(ai_summary)

def is_hot_news(entry, ai_summary, sentiment):
    """Phát hiện tin nóng dựa trên phân tích nội dung, từ khóa và cảm xúc"""
    try:
        title = getattr(entry, 'title', '').lower()
        summary = getattr(entry, 'summary', '').lower()
        content_text = f"{title} {summary}".lower()
        
        # 1. Kiểm tra từ khóa tin nóng trong tiêu đề hoặc nội dung
        for keyword in Config.HOT_NEWS_KEYWORDS:
            if keyword.lower() in content_text:
                logging.info(f"Hot news phát hiện bởi từ khóa '{keyword}': {title}")
                return True
                
        # 2. Kiểm tra các cụm từ ảnh hưởng trong AI summary
        ai_text = ai_summary.lower()
        for phrase in Config.HOT_NEWS_IMPACT_PHRASES:
            if phrase.lower() in ai_text:
                logging.info(f"Hot news phát hiện bởi cụm từ ảnh hưởng '{phrase}': {title}")
                return True
        
        # 3. Phân tích dựa trên cảm xúc và mức độ ảnh hưởng
        if sentiment != "Trung lập":
            # Nếu có cảm xúc và các từ chỉ mức độ cao trong phân tích AI
            intensity_words = ["rất", "mạnh", "nghiêm trọng", "đáng kể", "lớn", "quan trọng"]
            for word in intensity_words:
                if word in ai_text and (
                    "thị trường" in ai_text or "nhà đầu tư" in ai_text or "ảnh hưởng" in ai_text
                ):
                    logging.info(f"Hot news phát hiện bởi cảm xúc và mức độ ảnh hưởng: {title}")
                    return True
        
        return False
    except Exception as e:
        logging.warning(f"Lỗi khi phát hiện tin nóng: {e}")
        return False

# --- Parse RSS Feed & News Processing ---
def normalize_text(text):
    if not text:
        return ""
    # Loại bỏ dấu tiếng Việt
    text = unicodedata.normalize('NFD', text)
    text = ''.join([c for c in text if unicodedata.category(c) != 'Mn'])
    # Loại bỏ ký tự đặc biệt, chỉ giữ lại chữ và số
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    # Viết thường, loại bỏ khoảng trắng thừa
    text = text.lower().strip()
    return text

def is_recent_news(entry, days=Config.FETCH_LIMIT_DAYS):
    """
    Chỉ lấy tin có ngày đăng là hôm nay (theo timezone Việt Nam).
    Nếu không parse được ngày thì bỏ qua (không lấy).
    """
    published = getattr(entry, 'published', None) or getattr(entry, 'updated', None)
    if not published:
        return False
    try:
        # Thử parse nhiều định dạng ngày tháng
        try:
            dt = datetime.datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
        except ValueError:
            try:
                dt = datetime.datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %z')
            except ValueError:
                # Nếu không parse được, bỏ qua tin này
                return False
        # Đảm bảo có timezone
        if not dt.tzinfo:
            dt = Config.TIMEZONE.localize(dt)
        now = get_now_with_tz()
        # So sánh ngày (không so sánh giờ)
        return dt.date() == now.date()
    except Exception as e:
        logger.warning(f"Lỗi khi kiểm tra thời gian tin: {e}")
        # Nếu có lỗi, bỏ qua tin này
        return False

def is_relevant_news_smart(entry):
    """
    Lọc tin thông minh: nhiều từ khóa liên quan, loại trừ spam/PR.
    """
    title = normalize_text(getattr(entry, 'title', ''))
    summary = normalize_text(getattr(entry, 'summary', ''))
    content_text = f"{title} {summary}"

    # Loại trừ tin spam/PR
    for ex_kw in EXCLUDE_KEYWORDS:
        if normalize_text(ex_kw) in content_text:
            return False

    # Đếm số từ khóa liên quan xuất hiện
    all_keywords = [normalize_text(k) for k in Config.RELEVANT_KEYWORDS] + [normalize_text(k) for k in additional_keywords]
    match_count = sum(1 for kw in all_keywords if kw and kw in content_text)
    
    # Phải có ít nhất 1 từ khóa
    return match_count >= 1

def is_hot_news_simple(entry):
    """
    Phát hiện tin nóng mà không dùng AI, chỉ dựa trên từ khóa.
    """
    title = getattr(entry, 'title', '').lower()
    summary = getattr(entry, 'summary', '').lower()
    content_text = f"{title} {summary}".lower()
    
    for keyword in Config.HOT_NEWS_KEYWORDS:
        if keyword.lower() in content_text:
            logger.info(f"Hot news đơn giản phát hiện bởi từ khóa '{keyword}': {title}")
            return True
    
    return False

def is_relevant_news(entry):
    """
    Kiểm tra xem tin tức có liên quan đến các chủ đề quan tâm không dựa trên từ khóa (chuẩn hóa)
    """
    # Lấy nội dung từ tiêu đề và tóm tắt, chuẩn hóa
    title = normalize_text(getattr(entry, 'title', ''))
    summary = normalize_text(getattr(entry, 'summary', ''))
    content_text = f"{title} {summary}"

    # Chuẩn hóa từ khóa mặc định và bổ sung
    all_keywords = [normalize_text(k) for k in Config.RELEVANT_KEYWORDS] + [normalize_text(k) for k in additional_keywords]

    # So khớp từ khóa
    for keyword in all_keywords:
        if keyword and keyword in content_text:
            return True
    return False

async def parse_feed(url):
    try:
        feed_data = await asyncio.to_thread(feedparser.parse, url)
        if not feed_data.entries:
            logger.warning(f"Không tìm thấy tin tức từ feed: {url}")
            return []
        return feed_data.entries
    except Exception as e:
        logger.error(f"Lỗi khi parse RSS feed {url}: {e}")
        return []

def extract_image_url(entry):
    """Extract image URL from entry if available"""
    try:
        if 'media_content' in entry and entry.media_content:
            for media in entry.media_content:
                if 'url' in media:
                    return media['url']
        
        # Try finding image in content
        if 'content' in entry and entry.content:
            for content in entry.content:
                if 'value' in content:
                    match = re.search(r'<img[^>]+src="([^">]+)"', content['value'])
                    if match:
                        return match.group(1)
        
        # Try finding image in summary
        if hasattr(entry, 'summary'):
            match = re.search(r'<img[^>]+src="([^">]+)"', entry.summary)
            if match:
                return match.group(1)
    except Exception as e:
        logger.warning(f"Lỗi khi extract ảnh: {e}")
    
    return None
    
# --- Command Handler Functions ---

# Admin only decorator
def admin_only(func):
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        if user_id != Config.ADMIN_ID:
            await update.message.reply_text("❌ Bạn không có quyền sử dụng lệnh này.")
            return
        return await func(update, context, *args, **kwargs)
    return wrapped

# --- Redis-only user approval ---
async def is_user_approved(user_id):
    # Luôn coi ADMIN_ID là approved, không cần lưu vào Redis
    return user_id == Config.ADMIN_ID or await redis_client.sismember("approved_users", user_id)

# --- User approval logic using Redis set ---
async def approve_user_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    action, user_id = query.data.split("_")
    user_id = int(user_id)
    if action == "approve":
        if user_id != Config.ADMIN_ID:
            await redis_client.sadd("approved_users", user_id)
        await query.edit_message_text(f"✅ User {user_id} đã được phê duyệt.")
        await context.bot.send_message(chat_id=user_id, text="✅ Bạn đã được phê duyệt để sử dụng Bot News! Gõ /help để xem hướng dẫn.")
    else:
        await query.edit_message_text(f"❌ Đã từ chối yêu cầu từ user {user_id}.")
        await context.bot.send_message(chat_id=user_id, text="❌ Yêu cầu sử dụng bot của bạn đã bị từ chối.")

# --- Lưu và lấy tiêu đề/tin gần đây bằng Redis list (giới hạn 200) ---
async def add_recent_title(normalized_title, limit=200):
    await redis_client.lpush("recent_titles", normalized_title)
    await redis_client.ltrim("recent_titles", 0, limit-1)

async def get_recent_titles(limit=200):
    return [title.decode() if isinstance(title, bytes) else title for title in await redis_client.lrange("recent_titles", 0, limit-1)]

async def add_recent_news_text(news_text, limit=200):
    await redis_client.lpush("recent_news_texts", news_text)
    await redis_client.ltrim("recent_news_texts", 0, limit-1)

async def get_recent_news_texts(limit=200):
    return [txt.decode() if isinstance(txt, bytes) else txt for txt in await redis_client.lrange("recent_news_texts", 0, limit-1)]

# --- Registration commands
async def register_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    
    # Check if already approved
    if await is_user_approved(user_id):
        await update.message.reply_text("✅ Bạn đã được đăng ký sử dụng bot rồi!")
        return
    
    # Notify admin about registration request
    keyboard = [
        [
            InlineKeyboardButton("Approve ✅", callback_data=f"approve_{user_id}"),
            InlineKeyboardButton("Deny ❌", callback_data=f"deny_{user_id}")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await context.bot.send_message(
        chat_id=Config.ADMIN_ID,
        text=(
            f"🔔 Yêu cầu đăng ký mới:\n"
            f"User ID: {user_id}\n"
            f"Name: {user.first_name} {user.last_name or ''}\n"
            f"Username: @{user.username or 'N/A'}"
        ),
        reply_markup=reply_markup
    )
    
    await update.message.reply_text(
        "📝 Yêu cầu đăng ký của bạn đã được gửi tới admin. "
        "Bạn sẽ được thông báo khi yêu cầu được xử lý."
    )

# --- Timezone Helper Functions ---
def get_now_with_tz():
    """Return current datetime with timezone"""
    return datetime.datetime.now(Config.TIMEZONE)

def format_datetime(dt, format='%Y-%m-%d %H:%M:%S'):
    """Format datetime with timezone if needed"""
    if dt is None:
        return get_now_with_tz().strftime(format)
    
    # Check if datetime has timezone info
    if not dt.tzinfo:
        # Add timezone info if missing
        dt = Config.TIMEZONE.localize(dt)
    
    return dt.strftime(format)

def ensure_timezone_aware(dt):
    """Đảm bảo datetime object có timezone trước khi đưa vào DB"""
    if dt is None:
        return get_now_with_tz()
    # Nếu đã có tzinfo và offset, trả về nguyên bản
    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
        return dt
    # Nếu chưa có timezone, thêm vào
    try:
        return Config.TIMEZONE.localize(dt)
    except (ValueError, AttributeError):
        # Xử lý các trường hợp ngoại lệ
        logger.warning(f"Không thể thêm timezone cho datetime: {dt}")
        # Tạo mới datetime với timezone
        return datetime.datetime(
            dt.year, dt.month, dt.day, 
            dt.hour, dt.minute, dt.second, 
            dt.microsecond, Config.TIMEZONE
        )

async def normalize_title(title):
    """Chuẩn hóa tiêu đề tin tức để so sánh"""
    if not title:
        return ""
    
    # Remove HTML tags if any
    title = re.sub(r'<[^>]+>', '', title)
    
    # Normalize unicode
    title = unicodedata.normalize('NFD', title)
    title = ''.join([c for c in title if unicodedata.category(c) != 'Mn'])
    
    # Convert to lowercase and remove extra spaces
    title = re.sub(r'\s+', ' ', title.lower().strip())
    
    # Remove special characters
    title = re.sub(r'[^\w\s]', '', title)
    
    return title

async def is_title_sent(normalized_title):
    """Kiểm tra xem tiêu đề đã được gửi chưa (dựa trên tiêu đề chuẩn hóa)"""
    return await redis_client.sismember("sent_titles", normalized_title)

async def mark_title_sent(normalized_title):
    """Đánh dấu tiêu đề đã được gửi"""
    await redis_client.sadd("sent_titles", normalized_title)
    await redis_client.expire("sent_titles", Config.REDIS_TTL)

# --- News Deduplication by Hash ---
async def is_hash_sent(content_hash):
    return await redis_client.sismember("sent_hashes", content_hash)

async def mark_hash_sent(content_hash):
    await redis_client.sadd("sent_hashes", content_hash)
    await redis_client.expire("sent_hashes", Config.REDIS_TTL)

def is_similar_title(new_title, recent_titles, threshold=None):
    """So sánh tiêu đề mới với các tiêu đề cũ, nếu similarity > threshold thì coi là trùng"""
    if not recent_titles:
        return False
        
    # Sử dụng ngưỡng từ Config nếu không được chỉ định
    if threshold is None:
        threshold = Config.TITLE_SIMILARITY_THRESHOLD
        
    try:
        titles = [new_title] + recent_titles
        vectorizer = TfidfVectorizer().fit_transform(titles)
        vectors = vectorizer.toarray()
        sim_scores = cosine_similarity([vectors[0]], vectors[1:])[0]
        max_sim = max(sim_scores) if len(sim_scores) > 0 else 0
        
        # Ghi log chi tiết nếu được cấu hình
        if Config.LOG_SIMILARITY_DETAILS and max_sim > 0.6:
            max_idx = sim_scores.argmax() if len(sim_scores) > 0 else -1
            similar_title = recent_titles[max_idx] if max_idx >= 0 else "N/A"
            logger.info(f"Similarity tiêu đề: {max_sim:.2f} (ngưỡng: {threshold:.2f})")
            logger.debug(f"Tiêu đề mới: {new_title[:30]}... | Tiêu đề tương tự: {similar_title[:30]}...")
            
        return max_sim > threshold
    except Exception as e:
        logger.error(f"Lỗi khi so sánh tiêu đề tương tự: {e}")
        return False

# --- News Duplication Detection ---
def is_duplicate_by_content(new_text, recent_texts, threshold=Config.DUPLICATE_THRESHOLD):
    """
    Phát hiện tin trùng lặp bằng TF-IDF và Cosine Similarity
    Nâng cấp:
    - Sử dụng ngram từ 1-3 để bắt cụm từ dài hơn
    - Loại bỏ stopwords tiếng Việt
    - Tối ưu vector hóa
    - Phát hiện chính xác hơn các tin có cùng nội dung nhưng khác nguồn
    """
    if not recent_texts:
        return False
    
    try:
        # Danh sách stopwords tiếng Việt cơ bản
        vn_stopwords = {
            "và", "là", "của", "có", "được", "trong", "cho", "không", "đã", "với", "được", "này",
            "đến", "từ", "khi", "như", "người", "những", "sẽ", "vào", "về", "còn", "bị", "theo",
            "để", "tại", "nhưng", "ra", "nên", "một", "các", "cũng", "đang", "tới", "trên", "tôi",
            "bạn", "chúng", "rằng", "thì", "đó", "làm", "nếu", "nói", "bởi", "lên", "khác", "họ"
        }
        
        # Thêm tin mới vào đầu danh sách để vector hóa
        texts = [new_text] + recent_texts
        
        # Tiền xử lý để loại bỏ nhiễu và giữ lại nội dung quan trọng
        # Đây là bước quan trọng để phát hiện tin cùng nội dung từ các nguồn khác nhau
        processed_texts = []
        for text in texts:
            # Chuẩn hóa tin, chỉ giữ từ khóa chính
            words = text.lower().split()
            words = [w for w in words if w not in vn_stopwords and len(w) > 1]
            processed_texts.append(' '.join(words))
        
        # Tính vector TF-IDF, dùng ngram từ 1-3 để bắt được cụm từ có ý nghĩa
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english", # Vẫn giữ cho các từ tiếng Anh
            ngram_range=(1, 3),   # Nâng lên (1, 3) để bắt cụm từ dài hơn
            min_df=1,
            max_features=10000    # Giới hạn số lượng đặc trưng để tăng hiệu suất
        ).fit_transform(processed_texts)
        
        # Chuyển sang mảng để so sánh
        vectors = vectorizer.toarray()
        
        # Tính cosine similarity giữa tin mới và các tin cũ
        sim_scores = cosine_similarity([vectors[0]], vectors[1:])[0]
        
        # Kiểm tra có trùng lặp không (similarity > threshold)
        max_similarity = max(sim_scores) if len(sim_scores) > 0 else 0
        is_duplicate = max_similarity > threshold
        
        if is_duplicate:
            max_idx = sim_scores.argmax() if len(sim_scores) > 0 else -1
            similar_text = recent_texts[max_idx] if max_idx >= 0 else "N/A"
            logger.info(f"Phát hiện tin trùng lặp! Similarity: {max_similarity:.2f}, Threshold: {threshold}")
            logger.debug(f"Tin mới: {new_text[:50]}... | Tin trùng: {similar_text[:50]}...")
        
        return is_duplicate
    except Exception as e:
        logger.error(f"Lỗi khi phát hiện tin trùng lặp: {e}")
        return False  # Nếu lỗi, coi như không trùng để xử lý tin

async def init_redis():
    """Initialize Redis connection"""
    global redis_client
    try:
        redis_client = aioredis.from_url(Config.REDIS_URL)
        
        # Load additional keywords from Redis if they exist
        global additional_keywords
        keywords_data = await redis_client.get("additional_keywords")
        if keywords_data:
            additional_keywords = pickle.loads(keywords_data)
            logger.info(f"Loaded {len(additional_keywords)} additional keywords from Redis")
        
        # Load duplicate detection settings
        dup_threshold = await redis_client.get("duplicate_threshold")
        if dup_threshold:
            try:
                Config.DUPLICATE_THRESHOLD = float(dup_threshold.decode())
                logger.info(f"Loaded duplicate threshold: {Config.DUPLICATE_THRESHOLD}")
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to load duplicate threshold: {e}")
                
        title_threshold = await redis_client.get("title_threshold")
        if title_threshold:
            try:
                Config.TITLE_SIMILARITY_THRESHOLD = float(title_threshold.decode())
                logger.info(f"Loaded title similarity threshold: {Config.TITLE_SIMILARITY_THRESHOLD}")
            except (ValueError, AttributeError) as e:
                logger.warning(f"Failed to load title threshold: {e}")
                
        log_similarity = await redis_client.get("log_similarity_details")
        if log_similarity:
            try:
                Config.LOG_SIMILARITY_DETAILS = log_similarity.decode().lower() == "true"
                logger.info(f"Loaded similarity logging setting: {Config.LOG_SIMILARITY_DETAILS}")
            except AttributeError as e:
                logger.warning(f"Failed to load similarity logging setting: {e}")
        
        logger.info("Redis initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing Redis: {e}")
        return False

async def clear_news_queues():
    """Xóa tất cả tin tức trong queue khi khởi động lại bot để tránh gửi lại tin cũ"""
    try:
        # Xóa các queue tin tức
        await redis_client.delete("hot_news_queue")
        await redis_client.delete("news_queue")
        # Giữ lại ids đã gửi để tránh trùng lặp
        logger.info("Đã xóa tất cả tin trong queue để chuẩn bị cho quét mới")
    except Exception as e:
        logger.error(f"Lỗi khi xóa queue tin tức: {e}")

# --- Helper for rotating RSS feed index ---
async def get_current_feed_index():
    idx = await redis_client.get("current_feed_index")
    if idx is not None:
        return int(idx)
    return 0

async def set_current_feed_index(idx):
    await redis_client.set("current_feed_index", str(idx))

# --- Main Function ---

# Global application variable to store our bot instance
application = None

async def send_message_to_user(user_id, message, entry=None, is_hot_news=False):
    """Send a news message to a user"""
    try:
        # Chuẩn bị nội dung tin nhắn
        title = getattr(entry, 'title', 'Không có tiêu đề')
        link = getattr(entry, 'link', '#')
        # Lấy published date với xử lý timezone
        published = getattr(entry, 'published', None)
        # Nếu published là string, convert sang datetime
        if isinstance(published, str):
            try:
                published = datetime.datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %Z')
                # Đảm bảo published có timezone
                published = ensure_timezone_aware(published)
            except ValueError:
                try:
                    # Thử với format khác (RSS feeds có thể khác nhau)
                    published = datetime.datetime.strptime(published, '%a, %d %b %Y %H:%M:%S %z')
                except ValueError:
                    # Fallback nếu parse thất bại
                    published = None
        # Format date
        date = format_datetime(published) if published else format_datetime(None)
        # Extract domain from link
        domain = urlparse(link).netloc
        # Create message with emoji based on news type
        prefix = "🔥 TIN NÓNG: " if is_hot_news else "📰 TIN MỚI: "
        # Format message
        formatted_message = (
            f"{prefix}<b>{title}</b>\n\n"
            f"<pre>{message}</pre>\n\n"
            f"<i>Nguồn: {domain} • {date}</i>\n"
            f"<a href='{link}'>Đọc chi tiết</a>"
        )
        # Tạo nút đọc chi tiết
        keyboard = [[InlineKeyboardButton("Đọc chi tiết", url=link)]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        # Get the global application's bot
        global application
        if application and application.bot:
            bot = application.bot
        else:
            from telegram import Bot
            bot = Bot(token=Config.BOT_TOKEN)
        # Luôn gửi tin nhắn text, không gửi ảnh
        await bot.send_message(
            chat_id=user_id,
            text=formatted_message,
            reply_markup=reply_markup,
            parse_mode='HTML',
            disable_web_page_preview=False
        )
    except Exception as e:
        logger.error(f"Lỗi khi gửi tin tức cho user {user_id}: {e}")

# --- Các job định kỳ mới ---

async def process_and_send_news(news_data):
    """
    Xử lý (AI, dịch, đánh dấu đã gửi) và gửi tin cho tất cả user đã duyệt. Dùng cho cả gửi định kỳ và gửi ngay lập tức.
    """
    try:
        # Lấy danh sách người dùng đã được phê duyệt
        approved_users_list = [int(uid) for uid in await redis_client.smembers("approved_users")]
        if Config.ADMIN_ID not in approved_users_list:
            approved_users_list.append(Config.ADMIN_ID)
        if not approved_users_list:
            logger.warning("Không có người dùng nào được phê duyệt để gửi tin.")
            return

        # Phân tích tin tức bằng AI
        domain = urlparse(news_data['link']).netloc if 'link' in news_data else 'N/A'
        prompt = f"""
        Tóm tắt và phân tích tin tức sau cho nhà đầu tư chứng khoán Việt Nam.
        \nTiêu đề: {news_data.get('title', 'Không có tiêu đề')}
        Tóm tắt: {news_data.get('summary', 'Không có tóm tắt')}
        Nguồn: {domain}
        \n1. Tóm tắt ngắn gọn nội dung 
        2. Đánh giá tác động ( 2-3 câu ). Cảm xúc (Tích cực/Tiêu cực/Trung lập)
        3. Mã/ngành liên quan
        """
        try:
            ai_summary = await analyze_news(prompt)
        except Exception as e:
            logger.error(f"Lỗi khi phân tích tin bằng AI: {e}")
            ai_summary = news_data.get('summary', 'Không có phân tích nào.')

        # Nếu là tiếng Anh thì dịch sang tiếng Việt
        lang = detect_language(news_data.get('title', '') + ' ' + news_data.get('summary', ''))
        if lang == 'en':
            ai_summary = await translate_to_vietnamese(ai_summary)

        # Tạo đối tượng entry từ news_data để truyền vào hàm gửi
        class EntryObject:
            pass
        entry = EntryObject()
        for key, value in news_data.items():
            setattr(entry, key, value)

        # Xử lý sentiment nếu cần
        is_hot = news_data.get('is_hot', False)
        sentiment = await extract_sentiment(ai_summary) if is_hot else 'Trung lập'

        # Đánh dấu tin đã được gửi
        await mark_sent(news_data.get('id', '') or news_data.get('link', ''))

        # Gửi tin cho tất cả người dùng
        sent_count = 0
        for user_id in approved_users_list:
            try:
                await send_message_to_user(user_id, ai_summary, entry, news_data.get('is_hot', False))
                sent_count += 1
            except Exception as e:
                logger.error(f"Lỗi khi gửi tin cho user {user_id}: {e}")
        logger.info(f"Đã gửi tin '{news_data.get('title', '')[:30]}...' cho {sent_count}/{len(approved_users_list)} người dùng.")
    except Exception as e:
        logger.error(f"Lỗi trong process_and_send_news: {e}")

async def fetch_and_cache_news(context: ContextTypes.DEFAULT_TYPE):
    """
    Job chạy mỗi 10 phút để quét 1 RSS, lọc tin và đẩy vào queue (luân phiên từng nguồn).
    Nâng cấp: chỉ lấy tin trong ngày, lọc trùng tiêu đề đã gửi, chuẩn hóa nội dung khi so sánh trùng lặp, 
    lọc hash nội dung, lọc tiêu đề tương tự, lọc nội dung tương tự từ các nguồn khác nhau.
    """
    try:
        logger.info("Đang quét 1 RSS và cache tin mới...")
        recent_news_texts_raw = await get_recent_news_texts()
        recent_news_texts = [normalize_text(txt) for txt in recent_news_texts_raw]
        # Lấy danh sách tiêu đề chuẩn hóa gần đây (giới hạn 200)
        recent_titles = await get_recent_titles(limit=200)
        queued_count = 0
        skipped_count = 0
        duplicate_content_count = 0
        hot_news_count = 0
        feeds = Config.FEED_URLS
        feed_idx = await get_current_feed_index()
        feed_url = feeds[feed_idx % len(feeds)]
        logger.info(f"Quét RSS: {feed_url}")
        entries = await parse_feed(feed_url)
        for entry in entries:
            try:
                if not is_recent_news(entry, days=Config.FETCH_LIMIT_DAYS):
                    continue
                if not is_relevant_news_smart(entry):
                    continue
                entry_id = getattr(entry, 'id', '') or getattr(entry, 'link', '')
                normalized_title = await normalize_title(getattr(entry, 'title', ''))
                # 1. Lọc hash nội dung
                entry_text = f"{getattr(entry, 'title', '')} {getattr(entry, 'summary', '')}"
                entry_text_norm = normalize_text(entry_text)
                content_hash = hashlib.sha256(entry_text_norm.encode('utf-8')).hexdigest()
                if await is_hash_sent(content_hash):
                    skipped_count += 1
                    continue
                # 2. Lọc tiêu đề tương tự
                if is_similar_title(normalized_title, recent_titles, threshold=Config.TITLE_SIMILARITY_THRESHOLD):
                    skipped_count += 1
                    continue
                # 3. Nâng cấp: Lọc tin có nội dung tương tự từ các nguồn khác nhau
                prepared_content = normalize_text(f"{getattr(entry, 'title', '')} {getattr(entry, 'summary', '')}")
                if is_duplicate_by_content(prepared_content, recent_news_texts, threshold=Config.DUPLICATE_THRESHOLD):
                    logger.info(f"Phát hiện tin trùng lặp nội dung từ nguồn khác nhau: {getattr(entry, 'title', '')[:50]}...")
                    duplicate_content_count += 1
                    continue
                recent_titles.append(normalized_title)
                # 4. Lọc tiêu đề đã gửi tuyệt đối
                if await is_title_sent(normalized_title):
                    skipped_count += 1
                    continue
                if await redis_client.sismember("news_queue_ids", entry_id):
                    skipped_count += 1
                    continue
                if await is_sent(entry_id):
                    skipped_count += 1
                    continue
                # Thêm vào danh sách các nội dung tin gần đây để so sánh cho tin sau
                recent_news_texts.append(entry_text_norm)
                # Xác định loại tin & lưu vào queue
                is_hot = is_hot_news_simple(entry)
                news_data = {
                    "id": entry_id,
                    "title": getattr(entry, "title", ""),
                    "link": getattr(entry, "link", ""),
                    "summary": getattr(entry, "summary", ""),
                    "published": getattr(entry, "published", ""),
                    "is_hot": is_hot,
                }
                if is_hot:
                    await redis_client.rpush("hot_news_queue", json.dumps(news_data))
                    hot_news_count += 1
                    # Gửi ngay lập tức tin nóng
                    await process_and_send_news(news_data)
                else:
                    await redis_client.rpush("news_queue", json.dumps(news_data))
                await redis_client.sadd("news_queue_ids", entry_id)
                await redis_client.expire("news_queue_ids", Config.REDIS_TTL)
                await mark_title_sent(normalized_title)
                await mark_hash_sent(content_hash)
                queued_count += 1
                await add_recent_title(normalized_title)
                await add_recent_news_text(entry_text_norm)
            except Exception as e:
                logger.warning(f"Lỗi khi xử lý tin từ feed {feed_url}: {e}")
        hot_queue_len = await redis_client.llen("hot_news_queue")
        normal_queue_len = await redis_client.llen("news_queue")
        logger.info(f"Quét RSS hoàn tất: Đã cache {queued_count} tin mới ({hot_news_count} tin nóng), "
                   f"bỏ qua {skipped_count} tin trùng lặp thông thường, {duplicate_content_count} tin trùng nội dung. "
                   f"Số tin trong queue: {hot_queue_len} tin nóng, {normal_queue_len} tin thường.")
        if queued_count == 0:
            logger.info("Không có tin mới, chuyển sang feed tiếp theo và sẽ thử sau 1 phút.")
            await set_current_feed_index((feed_idx + 1) % len(feeds))
            context.job_queue.run_once(fetch_and_cache_news, 60)
        else:
            await set_current_feed_index((feed_idx + 1) % len(feeds))
    except Exception as e:
        logger.error(f"Lỗi trong job fetch_and_cache_news: {e}")

async def send_news_from_queue(context: ContextTypes.DEFAULT_TYPE):
    """
    Job chạy mỗi 800s để lấy 1 tin từ queue và gửi cho user.
    Ưu tiên tin nóng trước.
    """
    try:
        # Ưu tiên lấy tin nóng trước
        news_json = await redis_client.lpop("hot_news_queue")
        if not news_json:
            # Nếu không có tin nóng, lấy tin thường
            news_json = await redis_client.lpop("news_queue")
        if not news_json:
            logger.info("Không còn tin trong cả hai queue. Đợi chu kỳ fetch tiếp theo.")
            return
        # Parse JSON thành dict
        news_data = json.loads(news_json)
        await process_and_send_news(news_data)
    except Exception as e:
        logger.error(f"Lỗi trong job send_news_from_queue: {e}")

# Hàm phát hiện ngôn ngữ
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"

# Hàm dịch sang tiếng Việt
async def translate_to_vietnamese(text):
    try:
        translator = Translator()
        # googletrans không async, nên dùng to_thread
        result = await asyncio.to_thread(translator.translate, text, dest='vi')
        return result.text
    except Exception as e:
        logging.error(f"Lỗi khi dịch sang tiếng Việt: {e}")
        return text

# Function to handle callback queries
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    # Handle different callback data
    if query.data == "view_keywords":
        # Reuse view_keywords_command logic
        user_id = update.effective_user.id
        
        if not await is_user_approved(user_id):
            await query.message.reply_text(
                "❌ Bạn chưa được phê duyệt để sử dụng bot. Gõ /register để đăng ký."
            )
            return
        
        global additional_keywords
        default_keywords = Config.RELEVANT_KEYWORDS
        
        message = (
            f"📋 *Danh sách từ khóa hiện tại*\n\n"
            f"*Từ khóa mặc định ({len(default_keywords)})*: Bao gồm các từ khóa về chứng khoán, kinh tế, tài chính...\n\n"
        )
        
        if additional_keywords:
            message += f"*Từ khóa bổ sung ({len(additional_keywords)})*:\n{', '.join(additional_keywords)}\n\n"
        else:
            message += "*Từ khóa bổ sung*: Chưa có\n\n"
        
        message += (
            "Sử dụng /set_keywords để thêm từ khóa bổ sung.\n"
            "Sử dụng /clear_keywords để xóa tất cả từ khóa bổ sung."
        )
        
        await query.message.reply_text(message, parse_mode='Markdown')
    
    elif query.data == "help":
        # Show help message
        await help_command(update, context)
    
    # Handle other callbacks
    elif query.data.startswith("approve_") or query.data.startswith("deny_"):
        await approve_user_callback(update, context)

async def job_ping(context: ContextTypes.DEFAULT_TYPE):
    try:
        url = "https://anvt.onrender.com/health"  # Nếu endpoint này không có, đổi thành "/"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.get(url)
        logger.info(f"Ping giữ bot awake tới {url}")
    except Exception as e:
        logger.error(f"Lỗi khi ping giữ awake: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    await update.message.reply_text(
        f"👋 Xin chào {user.first_name or 'bạn'}! Đây là bot tổng hợp tin tức chứng khoán, kinh tế, tài chính.\n"
        "Gõ /help để xem hướng dẫn sử dụng."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    is_admin = user_id == Config.ADMIN_ID
    
    basic_commands = (
        "Các lệnh hỗ trợ:\n"
        "/start - Bắt đầu\n"
        "/help - Hướng dẫn\n"
        "/register - Đăng ký sử dụng bot\n"
        "/keywords - Xem từ khóa lọc tin\n"
        "/set_keywords - Thêm từ khóa lọc tin\n"
        "/clear_keywords - Xóa từ khóa bổ sung"
    )
    
    if is_admin:
        admin_commands = (
            "\n\nLệnh dành riêng cho admin:\n"
            "/check_dup_settings - Kiểm tra cài đặt lọc tin trùng\n"
            "/set_dup_threshold - Điều chỉnh ngưỡng phát hiện nội dung trùng\n"
            "/set_title_threshold - Điều chỉnh ngưỡng phát hiện tiêu đề tương tự\n"
            "/toggle_sim_log - Bật/tắt log chi tiết về similarity"
        )
        await update.message.reply_text(basic_commands + admin_commands)
    else:
        await update.message.reply_text(basic_commands)

async def view_keywords_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not await is_user_approved(user_id):
        await update.message.reply_text(
            "❌ Bạn chưa được phê duyệt để sử dụng bot. Gõ /register để đăng ký."
        )
        return

    global additional_keywords
    default_keywords = Config.RELEVANT_KEYWORDS

    message = (
        f"📋 *Danh sách từ khóa hiện tại*\n\n"
        f"*Từ khóa mặc định ({len(default_keywords)})*: Bao gồm các từ khóa về chứng khoán, kinh tế, tài chính...\n\n"
    )

    if additional_keywords:
        message += f"*Từ khóa bổ sung ({len(additional_keywords)})*:\n{', '.join(additional_keywords)}\n\n"
    else:
        message += "*Từ khóa bổ sung*: Chưa có\n\n"

    message += (
        "Sử dụng /set_keywords để thêm từ khóa bổ sung.\n"
        "Sử dụng /clear_keywords để xóa tất cả từ khóa bổ sung."
    )

    await update.message.reply_text(message, parse_mode='Markdown')

async def set_keywords_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not await is_user_approved(user_id):
        await update.message.reply_text(
            "❌ Bạn chưa được phê duyệt để sử dụng bot. Gõ /register để đăng ký."
        )
        return

    global additional_keywords
    if not context.args:
        await update.message.reply_text(
            "✏️ Vui lòng nhập từ khóa bạn muốn thêm. Ví dụ: `/set_keywords bitcoin, eth`"
        )
        return

    new_keywords = [normalize_text(kw.strip()) for kw in ' '.join(context.args).split(',') if kw.strip()]
    
    if not new_keywords:
        await update.message.reply_text("⚠️ Không có từ khóa hợp lệ nào được cung cấp.")
        return

    added_count = 0
    for kw in new_keywords:
        if kw not in additional_keywords:
            additional_keywords.append(kw)
            added_count += 1
    
    if added_count > 0:
        # Save to Redis
        await redis_client.set("additional_keywords", pickle.dumps(additional_keywords))
        await update.message.reply_text(
            f"✅ Đã thêm {added_count} từ khóa mới. Hiện có {len(additional_keywords)} từ khóa bổ sung.\n"
            "Gõ /keywords để xem danh sách đầy đủ."
        )
    else:
        await update.message.reply_text(
            "ℹ️ Các từ khóa bạn nhập đã có sẵn hoặc không hợp lệ."
        )

async def clear_keywords_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not await is_user_approved(user_id):
        await update.message.reply_text(
            "❌ Bạn chưa được phê duyệt để sử dụng bot. Gõ /register để đăng ký."
        )
        return

    global additional_keywords
    if not additional_keywords:
        await update.message.reply_text("ℹ️ Hiện không có từ khóa bổ sung nào để xóa.")
        return
        
    additional_keywords.clear()
    # Save to Redis
    await redis_client.delete("additional_keywords")
    await update.message.reply_text(
        "🗑️ Đã xóa tất cả từ khóa bổ sung. Bot sẽ chỉ sử dụng danh sách từ khóa mặc định.\n"
        "Gõ /keywords để xem lại."
    )

@admin_only
async def set_duplicate_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin command to set the duplicate detection threshold without restarting the bot
    Usage: /set_dup_threshold 0.65
    """
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(
            "❌ Cú pháp sai. Sử dụng: `/set_dup_threshold 0.65`\n"
            "Giá trị từ 0.0 đến 1.0, càng thấp càng nhạy với việc phát hiện trùng lặp."
        )
        return
        
    try:
        threshold = float(context.args[0])
        if threshold < 0.0 or threshold > 1.0:
            await update.message.reply_text("❌ Giá trị phải từ 0.0 đến 1.0")
            return
            
        # Cập nhật giá trị trong Config
        Config.DUPLICATE_THRESHOLD = threshold
        
        # Lưu vào Redis để giữ giá trị khi khởi động lại
        await redis_client.set("duplicate_threshold", str(threshold))
        
        await update.message.reply_text(
            f"✅ Đã cập nhật ngưỡng phát hiện tin trùng lặp nội dung: {threshold}\n"
            f"Áp dụng ngay cho các lần quét RSS tiếp theo."
        )
    except ValueError:
        await update.message.reply_text("❌ Giá trị không hợp lệ. Vui lòng nhập số từ 0.0 đến 1.0")

@admin_only
async def toggle_similarity_logging(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin command to toggle detailed similarity logging
    Usage: /toggle_sim_log
    """
    # Toggle giá trị
    Config.LOG_SIMILARITY_DETAILS = not Config.LOG_SIMILARITY_DETAILS
    
    # Lưu vào Redis
    await redis_client.set("log_similarity_details", str(Config.LOG_SIMILARITY_DETAILS).lower())
    
    status = "bật" if Config.LOG_SIMILARITY_DETAILS else "tắt"
    await update.message.reply_text(
        f"✅ Đã {status} ghi log chi tiết về phát hiện tin trùng lặp.\n"
        "Xem log hệ thống để theo dõi thông tin chi tiết về similarity."
    )

@admin_only
async def check_dup_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin command to check current duplicate detection settings
    Usage: /check_dup_settings
    """
    settings = (
        f"📊 *Cài đặt phát hiện tin trùng lặp*\n\n"
        f"• Ngưỡng tin trùng nội dung (DUPLICATE_THRESHOLD): {Config.DUPLICATE_THRESHOLD}\n"
        f"• Ngưỡng tiêu đề tương tự (TITLE_SIMILARITY_THRESHOLD): {Config.TITLE_SIMILARITY_THRESHOLD}\n"
        f"• Ghi log chi tiết: {'Bật' if Config.LOG_SIMILARITY_DETAILS else 'Tắt'}\n\n"
        f"Lệnh điều chỉnh:\n"
        f"• /set_dup_threshold [0.0-1.0] - Đặt ngưỡng phát hiện nội dung trùng lặp\n"
        f"• /set_title_threshold [0.0-1.0] - Đặt ngưỡng phát hiện tiêu đề tương tự\n"
        f"• /toggle_sim_log - Bật/tắt log chi tiết về similarity"
    )
    await update.message.reply_text(settings, parse_mode='Markdown')

@admin_only
async def set_title_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Admin command to set the title similarity threshold
    Usage: /set_title_threshold 0.92
    """
    if not context.args or len(context.args) != 1:
        await update.message.reply_text(
            "❌ Cú pháp sai. Sử dụng: `/set_title_threshold 0.92`\n"
            "Giá trị từ 0.0 đến 1.0, càng cao càng yêu cầu tiêu đề giống nhau mới coi là trùng."
        )
        return
        
    try:
        threshold = float(context.args[0])
        if threshold < 0.0 or threshold > 1.0:
            await update.message.reply_text("❌ Giá trị phải từ 0.0 đến 1.0")
            return
            
        # Cập nhật giá trị trong Config
        Config.TITLE_SIMILARITY_THRESHOLD = threshold
        
        # Lưu vào Redis để giữ giá trị khi khởi động lại
        await redis_client.set("title_threshold", str(threshold))
        
        await update.message.reply_text(
            f"✅ Đã cập nhật ngưỡng phát hiện tiêu đề tương tự: {threshold}\n"
            f"Áp dụng ngay cho các lần quét RSS tiếp theo."
        )
    except ValueError:
        await update.message.reply_text("❌ Giá trị không hợp lệ. Vui lòng nhập số từ 0.0 đến 1.0")

def main():
    global application, shutdown_flag
    
    # Reset shutdown flag
    shutdown_flag = False
    
    # Thiết lập signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        application = Application.builder().token(Config.BOT_TOKEN).build()

        # Initialize database and Redis
        loop = asyncio.get_event_loop()
        redis_ok = loop.run_until_complete(init_redis())
        
        # Tải cài đặt ngưỡng từ Redis nếu có
        if redis_ok:
            dup_threshold = loop.run_until_complete(redis_client.get("duplicate_threshold"))
            if dup_threshold:
                try:
                    Config.DUPLICATE_THRESHOLD = float(dup_threshold.decode())
                except (ValueError, AttributeError):
                    pass
                    
            title_threshold = loop.run_until_complete(redis_client.get("title_threshold"))
            if title_threshold:
                try:
                    Config.TITLE_SIMILARITY_THRESHOLD = float(title_threshold.decode())
                except (ValueError, AttributeError):
                    pass
                    
            log_similarity = loop.run_until_complete(redis_client.get("log_similarity_details"))
            if log_similarity:
                try:
                    Config.LOG_SIMILARITY_DETAILS = log_similarity.decode().lower() == "true"
                except (AttributeError):
                    pass
                    
            logger.info(f"Cài đặt lọc tin trùng: DUPLICATE_THRESHOLD={Config.DUPLICATE_THRESHOLD}, " 
                      f"TITLE_SIMILARITY_THRESHOLD={Config.TITLE_SIMILARITY_THRESHOLD}, "
                      f"LOG_SIMILARITY_DETAILS={Config.LOG_SIMILARITY_DETAILS}")

        if not redis_ok:
            logger.error("Failed to initialize Redis. Exiting.")
            return

        # Xóa queue cũ khi khởi động lại
        loop.run_until_complete(clear_news_queues())

        # Add command handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("register", register_user))
        application.add_handler(CommandHandler("keywords", view_keywords_command))
        application.add_handler(CommandHandler("set_keywords", set_keywords_command))
        application.add_handler(CommandHandler("clear_keywords", clear_keywords_command))
        
        # Thêm lệnh quản lý cài đặt lọc tin trùng
        application.add_handler(CommandHandler("check_dup_settings", check_dup_settings))
        application.add_handler(CommandHandler("set_dup_threshold", set_duplicate_threshold))
        application.add_handler(CommandHandler("set_title_threshold", set_title_threshold))
        application.add_handler(CommandHandler("toggle_sim_log", toggle_similarity_logging))

        # Add callback query handler
        application.add_handler(CallbackQueryHandler(button_callback))

        # Set up the job queue
        job_queue = application.job_queue
        
        # Cấu hình các job định kỳ mới:
        # 1. Job quét RSS mỗi giờ và cache tin
        job_queue.run_repeating(fetch_and_cache_news, interval=Config.HOURLY_JOB_INTERVAL, first=10)
        
        # 2. Job gửi tin từ queue mỗi 800s
        job_queue.run_repeating(send_news_from_queue, interval=Config.NEWS_JOB_INTERVAL, first=30)
        
        # 3. Job ping giữ awake mỗi 5 phút
        job_queue.run_repeating(job_ping, interval=300, first=60)
        
        # In thông tin job
        logger.info(f"Đã thiết lập 3 job định kỳ:\n"
                    f"- Quét RSS & cache: {Config.HOURLY_JOB_INTERVAL}s/lần\n"
                    f"- Gửi tin từ queue: {Config.NEWS_JOB_INTERVAL}s/lần\n"
                    f"- Ping giữ awake: 300s/lần")

        if Config.WEBHOOK_URL:
            webhook_port = int(os.environ.get("PORT", 8443))
            logger.info(f"Starting webhook on port {webhook_port} with URL: {Config.WEBHOOK_URL}")
            application.run_webhook(
                listen="0.0.0.0",
                port=webhook_port,
                url_path=Config.BOT_TOKEN,
                webhook_url=f"{Config.WEBHOOK_URL}/{Config.BOT_TOKEN}"
            )
        else:
            logger.info("Starting bot in polling mode")
            application.run_polling(allowed_updates=Update.ALL_TYPES)
            
    except Exception as e:
        logger.error(f"Lỗi không xử lý được trong hàm main: {e}")
        # Dọn dẹp tài nguyên khi có lỗi không xử lý được
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(cleanup_resources())
        else:
            loop.run_until_complete(cleanup_resources())
        sys.exit(1)

if __name__ == "__main__":
    main()
