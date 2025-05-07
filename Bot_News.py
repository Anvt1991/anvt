import logging
import os
import asyncio
# Nhóm các import thư viện bên ngoài
import feedparser
import httpx
import asyncpg
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

# --- 1. Config & setup ---
class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")
    DB_URL = os.getenv("DATABASE_URL")
    ADMIN_ID = int(os.getenv("ADMIN_ID", "1225226589"))
    GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
    OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "deepseek/deepseek-chat-v3-0324:free")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
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
    ]
    REDIS_TTL = int(os.getenv("REDIS_TTL", "21600"))  # 6h
    NEWS_JOB_INTERVAL = int(os.getenv("NEWS_JOB_INTERVAL", "900"))
    HOURLY_JOB_INTERVAL = int(os.getenv("HOURLY_JOB_INTERVAL", "600"))  # 10 phút/lần
    FETCH_LIMIT_DAYS = int(os.getenv("FETCH_LIMIT_DAYS", "2"))  # Chỉ lấy tin 2 ngày gần nhất 
    DELETE_OLD_NEWS_DAYS = int(os.getenv("DELETE_OLD_NEWS_DAYS", "2"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # Số lần thử lại khi feed lỗi
    MAX_NEWS_PER_CYCLE = int(os.getenv("MAX_NEWS_PER_CYCLE", "1"))  # Tối đa 1 tin mỗi lần
    TIMEZONE = pytz.timezone('Asia/Ho_Chi_Minh')  # Timezone chuẩn cho Việt Nam
    DUPLICATE_THRESHOLD = float(os.getenv("DUPLICATE_THRESHOLD", "0.85"))  # Ngưỡng để xác định tin trùng lặp
    RECENT_NEWS_DAYS = int(os.getenv("RECENT_NEWS_DAYS", "3"))  # Số ngày để lấy tin gần đây để so sánh
    
    # Cấu hình phát hiện tin nóng
    HOT_NEWS_KEYWORDS = [
        "khẩn cấp", "tin nóng", "breaking", "khủng hoảng", "crash", "sập", "bùng nổ", 
        "shock", "ảnh hưởng lớn", "thảm khốc", "thảm họa", "market crash", "sell off", 
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

# --- PostgreSQL ---
pool = None

# Lưu trữ danh sách user đã được duyệt
approved_users = set()

async def is_sent(entry_id):
    return await redis_client.sismember("sent_news", entry_id)

async def mark_sent(entry_id):
    await redis_client.sadd("sent_news", entry_id)
    await redis_client.expire("sent_news", Config.REDIS_TTL)

async def save_news(entry, ai_summary, sentiment, is_hot_news=False):
    try:
        # Lấy thời gian hiện tại với timezone, luôn đảm bảo timezone
        now = get_now_with_tz()
        now = ensure_timezone_aware(now)  # Đảm bảo có timezone

        # Log debug để trợ giúp phát hiện lỗi timezone
        logger.debug(f"Lưu tin vào DB: timezone={now.tzinfo}, value={now}")

        async with pool.acquire() as conn:
            # Dùng AT TIME ZONE trong PostgreSQL để đảm bảo nhất quán
            await conn.execute("""
                INSERT INTO news_insights (title, link, summary, sentiment, ai_opinion, is_hot_news, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7::timestamptz)
                ON CONFLICT (link) DO NOTHING
            """, entry.title, entry.link, entry.summary, sentiment, ai_summary, is_hot_news, now)
    except Exception as e:
        logging.warning(f"Lỗi khi lưu tin tức vào DB (link={entry.link}): {e}")
        logging.debug(f"Debug datetime: type={type(now)}, tzinfo={now.tzinfo}, value={now}")

async def is_in_db(entry):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT 1 FROM news_insights WHERE link=$1", entry.link)
        return row is not None

# Hàm xóa tin cũ hơn n ngày
async def delete_old_news(days=Config.DELETE_OLD_NEWS_DAYS):
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM news_insights WHERE created_at < NOW() - INTERVAL '{days} days'"
            )
            count = await conn.fetchval("SELECT count(*) FROM news_insights")
            logger.info(f"Đã xóa tin cũ hơn {days} ngày. Còn lại {count} tin trong DB.")
    except Exception as e:
        logging.error(f"Lỗi khi xóa tin cũ: {e}")

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
        model = genai.GenerativeModel("gemini-1.5-pro")
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
    Kiểm tra tin có nằm trong khung thời gian days gần nhất không.
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
                # Fallback: nếu không parse được, coi như là tin mới
                return True
        
        # Đảm bảo có timezone
        if not dt.tzinfo:
            dt = Config.TIMEZONE.localize(dt)
        now = get_now_with_tz()
        delta = now - dt
        return delta.days < days
    except Exception as e:
        logger.warning(f"Lỗi khi kiểm tra thời gian tin: {e}")
        # Nếu có lỗi, coi như là tin mới để an toàn
        return True

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

# Registration system - Only approved users can use the bot
async def is_user_approved(user_id):
    """Kiểm tra xem user đã được duyệt chưa"""
    global approved_users
    return user_id == Config.ADMIN_ID or user_id in approved_users

# Registration commands
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

# Callback handler for approve/deny requests
async def approve_user_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    action, user_id = query.data.split("_")
    user_id = int(user_id)
    
    if action == "approve":
        # Add to approved users
        global approved_users
        approved_users.add(user_id)
        
        # Save to database
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO approved_users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                str(user_id)
            )
        
        await query.edit_message_text(f"✅ User {user_id} đã được phê duyệt.")
        await context.bot.send_message(
            chat_id=user_id,
            text="✅ Bạn đã được phê duyệt để sử dụng Bot News! Gõ /help để xem hướng dẫn."
        )
    else:
        await query.edit_message_text(f"❌ Đã từ chối yêu cầu từ user {user_id}.")
        await context.bot.send_message(
            chat_id=user_id,
            text="❌ Yêu cầu sử dụng bot của bạn đã bị từ chối."
        )

# Start command
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    user_id = user.id
    
    if not await is_user_approved(user_id):
        await update.message.reply_text(
            "👋 Chào mừng bạn đến với Bot News Chứng Khoán!"
            "\n\nĐể sử dụng bot, bạn cần đăng ký và được phê duyệt."
            "\nGõ /register để gửi yêu cầu đăng ký."
        )
        return
    
    welcome_message = (
        f"👋 Chào mừng {user.first_name} đến với Bot News Chứng Khoán!\n\n"
        f"Bot này giúp bạn nhận tin tức chứng khoán, kinh tế và tài chính quan trọng, "
        f"kèm phân tích AI giúp đánh giá tác động.\n\n"
        f"🔍 Tin tức sẽ được lọc theo từ khóa quan trọng và gửi tự động khi có tin mới.\n"
        f"🔥 Tin nóng sẽ được gắn thẻ ưu tiên cao hơn.\n\n"
        f"Gõ /help để xem toàn bộ lệnh và hướng dẫn sử dụng."
    )
    
    keyboard = [
        [InlineKeyboardButton("🔑 Xem từ khóa hiện tại", callback_data="view_keywords")],
        [InlineKeyboardButton("❓ Hỗ trợ", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

# Help command
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    # Xác định xem người dùng có phải admin không để hiển thị lệnh nâng cao
    is_admin = (user_id == Config.ADMIN_ID)
    
    if not await is_user_approved(user_id) and not is_admin:
        await update.message.reply_text(
            "👋 Chào mừng bạn đến với Bot News Chứng Khoán!"
            "\n\nĐể sử dụng bot, bạn cần đăng ký và được phê duyệt."
            "\nGõ /register để gửi yêu cầu đăng ký."
        )
        return
    
    help_text = (
        "📚 *HƯỚNG DẪN SỬ DỤNG BOT NEWS*\n\n"
        "*Lệnh cơ bản:*\n"
        "/start - Khởi động bot\n"
        "/help - Hiển thị hướng dẫn này\n"
        "/register - Đăng ký sử dụng bot\n\n"
        
        "*Quản lý từ khóa:*\n"
        "/keywords - Xem danh sách từ khóa theo dõi\n"
        "/set_keywords <từ khóa> - Thêm từ khóa (cách nhau bởi dấu phẩy)\n"
        "/clear_keywords - Xóa tất cả từ khóa bổ sung\n\n"
        
        "*Lưu ý:*\n"
        "• Bot sẽ tự động gửi tin tức quan trọng khi phát hiện\n"
        "• Tin nóng sẽ được đánh dấu đặc biệt\n"
        "• Mỗi tin được phân tích bởi AI để đánh giá tác động\n"
    )
    
    # Thêm lệnh admin nếu là admin
    if is_admin:
        admin_help = (
            "\n*Lệnh dành cho Admin:*\n"
            "• Người dùng mới sẽ gửi request và admin nhận thông báo\n"
            "• Admin có thể phê duyệt/từ chối qua nút bấm\n"
        )
        help_text += admin_help
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

# Keyword management commands
async def set_keywords_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not await is_user_approved(user_id):
        await update.message.reply_text(
            "❌ Bạn chưa được phê duyệt để sử dụng bot. Gõ /register để đăng ký."
        )
        return
    
    # Lấy từ khóa từ arguments
    if not context.args or not context.args[0]:
        await update.message.reply_text(
            "❌ Vui lòng nhập các từ khóa, cách nhau bởi dấu phẩy.\n"
            "Ví dụ: /set_keywords bitcoin, AI, tesla, vàng"
        )
        return
    
    # Xử lý từ khóa
    text = ' '.join(context.args)
    global additional_keywords
    new_keywords = [kw.strip() for kw in text.split(',') if kw.strip()]
    
    if not new_keywords:
        await update.message.reply_text("❌ Không tìm thấy từ khóa hợp lệ.")
        return
    
    # Cập nhật từ khóa
    additional_keywords = new_keywords
    
    # Lưu vào Redis để ghi nhớ
    try:
        await redis_client.set("additional_keywords", pickle.dumps(additional_keywords), ex=86400*30)  # 30 ngày
    except Exception as e:
        logger.error(f"Lỗi khi lưu từ khóa vào Redis: {e}")
    
    await update.message.reply_text(
        f"✅ Đã cập nhật {len(new_keywords)} từ khóa bổ sung:\n"
        f"{', '.join(new_keywords)}"
    )

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

async def clear_keywords_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not await is_user_approved(user_id):
        await update.message.reply_text(
            "❌ Bạn chưa được phê duyệt để sử dụng bot. Gõ /register để đăng ký."
        )
        return
    
    global additional_keywords
    additional_keywords = []
    
    # Xóa khỏi Redis
    try:
        await redis_client.delete("additional_keywords")
    except Exception as e:
        logger.error(f"Lỗi khi xóa từ khóa từ Redis: {e}")
    
    await update.message.reply_text("✅ Đã xóa tất cả từ khóa bổ sung.")

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

# --- News Duplication Detection ---
async def get_recent_news_texts(days=Config.RECENT_NEWS_DAYS):
    """Lấy danh sách tin gần đây từ DB để so sánh tìm trùng lặp"""
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT title, summary FROM news_insights WHERE created_at > NOW() - INTERVAL '{days} days'"
            )
            # Ghép tiêu đề và tóm tắt để so sánh
            return [f"{row['title']} {row['summary']}" for row in rows]
    except Exception as e:
        logger.error(f"Lỗi khi lấy tin gần đây từ DB: {e}")
        return []

def is_duplicate_by_content(new_text, recent_texts, threshold=Config.DUPLICATE_THRESHOLD):
    """Phát hiện tin trùng lặp bằng TF-IDF và Cosine Similarity"""
    if not recent_texts:
        return False
    
    try:
        # Thêm tin mới vào đầu danh sách để vector hóa
        texts = [new_text] + recent_texts
        
        # Tính vector TF-IDF
        vectorizer = TfidfVectorizer(
            lowercase=True, 
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        ).fit_transform(texts)
        
        # Chuyển sang mảng để so sánh
        vectors = vectorizer.toarray()
        
        # Tính cosine similarity giữa tin mới và các tin cũ
        sim_scores = cosine_similarity([vectors[0]], vectors[1:])[0]
        
        # Kiểm tra có trùng lặp không (similarity > threshold)
        max_similarity = max(sim_scores) if len(sim_scores) > 0 else 0
        is_duplicate = max_similarity > threshold
        
        if is_duplicate:
            logger.info(f"Phát hiện tin trùng lặp! Similarity: {max_similarity:.2f}, Threshold: {threshold}")
        
        return is_duplicate
    except Exception as e:
        logger.error(f"Lỗi khi phát hiện tin trùng lặp: {e}")
        return False  # Nếu lỗi, coi như không trùng để xử lý tin

# --- Function to update database to ensure all timestamps use timezone ---
async def ensure_db_timezone_columns():
    """Đảm bảo tất cả các cột datetime trong DB đều lưu timezone"""
    try:
        async with pool.acquire() as conn:
            # Kiểm tra các bảng hiện có
            tables = await conn.fetch("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            for table in tables:
                table_name = table['table_name']
                # Kiểm tra các cột timestamp
                columns = await conn.fetch("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = $1 AND data_type LIKE '%timestamp%'
                """, table_name)
                
                for column in columns:
                    col_name = column['column_name']
                    data_type = column['data_type']
                    
                    # Nếu cột timestamp không có timezone, alter để thêm
                    if data_type == 'timestamp without time zone':
                        logger.info(f"Cập nhật cột {table_name}.{col_name} sang timestamptz")
                        await conn.execute(f"""
                            ALTER TABLE {table_name} 
                            ALTER COLUMN {col_name} TYPE timestamp with time zone 
                            USING {col_name} AT TIME ZONE 'Asia/Ho_Chi_Minh'
                        """)
            
            logger.info("Đã kiểm tra và cập nhật các cột timestamp trong DB")
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật DB timestamp columns: {e}")

async def init_db():
    """Initialize the database with necessary tables"""
    global pool
    try:
        # Connect to the database
        pool = await asyncpg.create_pool(Config.DB_URL)
        
        # Create news_insights table if it doesn't exist
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS news_insights (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    link TEXT UNIQUE NOT NULL,
                    summary TEXT,
                    sentiment TEXT,
                    ai_opinion TEXT,
                    is_hot_news BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Create approved_users table if it doesn't exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS approved_users (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT UNIQUE NOT NULL,
                    approved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """)
            
            # Đảm bảo admin luôn có trong bảng approved_users
            await conn.execute(
                "INSERT INTO approved_users (user_id) VALUES ($1) ON CONFLICT (user_id) DO NOTHING",
                str(Config.ADMIN_ID)
            )
            
            # Load approved users from the database
            global approved_users
            rows = await conn.fetch("SELECT user_id FROM approved_users")
            approved_users = set(int(row['user_id']) for row in rows)
            # Đảm bảo admin luôn trong set
            approved_users.add(Config.ADMIN_ID)
            
        # Đảm bảo tất cả các cột timestamp đều lưu timezone
        await ensure_db_timezone_columns()
            
        logger.info(f"Database initialized successfully. Loaded {len(approved_users)} approved users.")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

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

async def fetch_and_cache_news(context: ContextTypes.DEFAULT_TYPE):
    """
    Job chạy mỗi 10 phút để quét 1 RSS, lọc tin và đẩy vào queue (luân phiên từng nguồn).
    """
    try:
        logger.info("Đang quét 1 RSS và cache tin mới...")
        # Xóa tin cũ khỏi DB
        await delete_old_news()
        # Lấy tin gần đây từ DB để kiểm tra trùng lặp nội dung
        recent_news_texts = await get_recent_news_texts()
        queued_count = 0
        skipped_count = 0
        hot_news_count = 0
        feeds = Config.FEED_URLS
        # Lấy index nguồn hiện tại từ Redis
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
                if await redis_client.sismember("news_queue_ids", entry_id):
                    skipped_count += 1
                    continue
                if await is_sent(entry_id) or await is_in_db(entry):
                    await redis_client.sadd("news_queue_ids", entry_id)
                    skipped_count += 1
                    continue
                entry_text = f"{getattr(entry, 'title', '')} {getattr(entry, 'summary', '')}"
                if is_duplicate_by_content(entry_text, recent_news_texts):
                    skipped_count += 1
                    continue
                recent_news_texts.append(entry_text)
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
                else:
                    await redis_client.rpush("news_queue", json.dumps(news_data))
                await redis_client.sadd("news_queue_ids", entry_id)
                await redis_client.expire("news_queue_ids", Config.REDIS_TTL)
                queued_count += 1
            except Exception as e:
                logger.warning(f"Lỗi khi xử lý tin từ feed {feed_url}: {e}")
        hot_queue_len = await redis_client.llen("hot_news_queue")
        normal_queue_len = await redis_client.llen("news_queue")
        logger.info(f"Quét RSS hoàn tất: Đã cache {queued_count} tin mới ({hot_news_count} tin nóng), "
                   f"bỏ qua {skipped_count} tin trùng lặp. "
                   f"Số tin trong queue: {hot_queue_len} tin nóng, {normal_queue_len} tin thường.")
        # Tăng index lên 1 cho lần sau
        await set_current_feed_index((feed_idx + 1) % len(feeds))
    except Exception as e:
        logger.error(f"Lỗi trong job fetch_and_cache_news: {e}")

async def send_news_from_queue(context: ContextTypes.DEFAULT_TYPE):
    """
    Job chạy mỗi 800s để lấy 1 tin từ queue và gửi cho user.
    Ưu tiên tin nóng trước.
    """
    try:
        # Lấy danh sách người dùng đã được phê duyệt
        approved_users_list = []
        try:
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT user_id FROM approved_users")
                approved_users_list = [int(row['user_id']) for row in rows]
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách approved users: {e}")
            return
            
        if not approved_users_list:
            logger.warning("Không có người dùng nào được phê duyệt để gửi tin.")
            return
            
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
        
        # Phân tích tin tức bằng AI
        domain = urlparse(news_data['link']).netloc if 'link' in news_data else 'N/A'
        prompt = f"""
        Tóm tắt và phân tích tin tức sau cho nhà đầu tư chứng khoán Việt Nam.
        \nTiêu đề: {news_data.get('title', 'Không có tiêu đề')}
        Tóm tắt: {news_data.get('summary', 'Không có tóm tắt')}
        Nguồn: {domain}
        \n1. Tóm tắt ngắn gọn nội dung (2-3 câu)
        2. Phân tích, đánh giá tác động ( 3-5 câu ). Cảm xúc (Tích cực/Tiêu cực/Trung lập)
        3. Lời khuyên cho nhà đầu tư (1 câu)
        """
        
        try:
            ai_summary = await analyze_news(prompt)
        except Exception as e:
            logger.error(f"Lỗi khi phân tích tin bằng AI: {e}")
            # Nếu không phân tích được, dùng summary gốc
            ai_summary = news_data.get('summary', 'Không có phân tích nào.')
            
        # Tạo đối tượng entry từ news_data để truyền vào hàm gửi
        class EntryObject:
            pass
            
        entry = EntryObject()
        for key, value in news_data.items():
            setattr(entry, key, value)
            
        # Lưu tin vào DB
        try:
            is_hot = news_data.get('is_hot', False)
            sentiment = await extract_sentiment(ai_summary) if is_hot else 'Trung lập'
            await save_news(entry, ai_summary, sentiment, is_hot)
            
            # Đánh dấu tin đã được gửi
            await mark_sent(news_data.get('id', '') or news_data.get('link', ''))
        except Exception as e:
            logger.error(f"Lỗi khi lưu tin vào DB: {e}")
            
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
        logger.error(f"Lỗi trong job send_news_from_queue: {e}")

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

def main():
    global application
    application = Application.builder().token(Config.BOT_TOKEN).build()

    # Initialize database and Redis
    loop = asyncio.get_event_loop()
    db_ok = loop.run_until_complete(init_db())
    redis_ok = loop.run_until_complete(init_redis())

    if not db_ok or not redis_ok:
        logger.error("Failed to initialize database or Redis. Exiting.")
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

    # Add callback query handler
    application.add_handler(CallbackQueryHandler(button_callback))

    # Set up the job queue
    job_queue = application.job_queue
    
    # Cấu hình các job định kỳ mới:
    # 1. Job quét RSS mỗi giờ và cache tin
    job_queue.run_repeating(fetch_and_cache_news, interval=Config.HOURLY_JOB_INTERVAL, first=10)
    
    # 2. Job gửi tin từ queue mỗi 800s
    job_queue.run_repeating(send_news_from_queue, interval=Config.NEWS_JOB_INTERVAL, first=30)
    
    # In thông tin job
    logger.info(f"Đã thiết lập 2 job định kỳ:\n"
                f"- Quét RSS & cache: {Config.HOURLY_JOB_INTERVAL}s/lần\n"
                f"- Gửi tin từ queue: {Config.NEWS_JOB_INTERVAL}s/lần")

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

if __name__ == "__main__":
    main()
