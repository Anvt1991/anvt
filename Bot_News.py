import logging
import os
import asyncio
# Nhóm các import thư viện bên ngoài
import feedparser
import httpx
import asyncpg
import redis.asyncio as aioredis
import google.generativeai as genai
# Nhóm các import aiogram 
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
# Nhóm các import khác
from aiohttp import web
import re
from urllib.parse import urlparse
import unicodedata
import datetime

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
    FEED_URLS = [
        "https://news.google.com/rss/search?q=kinh+t%E1%BA%BF&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=ch%E1%BB%A9ng+kho%C3%A1n&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=v%C4%A9+m%C3%B4&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=chi%E1%BA%BFn+tranh&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=l%C3%A3i+su%E1%BA%A5t&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=fed&hl=vi&gl=VN&ceid=VN:vi",
        "https://news.google.com/rss/search?q=tin+n%C3%B3ng&hl=vi&gl=VN&ceid=VN:vi",  # Tin nóng
        "https://news.google.com/rss/search?q=%C4%91%E1%BA%A7u+t%C6%B0&hl=vi&gl=VN&ceid=VN:vi",  # Tin đầu tư
        "https://news.google.com/rss/search?q=doanh+nghi%E1%BB%87p&hl=vi&gl=VN&ceid=VN:vi",  # Tin doanh nghiệp
    ]
    REDIS_TTL = int(os.getenv("REDIS_TTL", "21600"))  # 6h
    NEWS_JOB_INTERVAL = int(os.getenv("NEWS_JOB_INTERVAL", "600"))  # 10 phút (giây)
    DELETE_OLD_NEWS_DAYS = int(os.getenv("DELETE_OLD_NEWS_DAYS", "7"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # Số lần thử lại khi feed lỗi
    MAX_NEWS_PER_CYCLE = int(os.getenv("MAX_NEWS_PER_CYCLE", "1"))  # Tối đa 1 tin mỗi lần
    
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

# --- Kiểm tra biến môi trường bắt buộc ---
REQUIRED_ENV_VARS = ["BOT_TOKEN", "OPENROUTER_API_KEY"]
for var in REQUIRED_ENV_VARS:
    if not os.getenv(var):
        raise RuntimeError(f"Missing required environment variable: {var}")

# --- Logging ---
logging.basicConfig(level=logging.INFO)
bot = Bot(token=Config.BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# --- Redis ---
redis = None

async def is_sent(entry_id):
    return await redis.sismember("sent_news", entry_id)

async def mark_sent(entry_id):
    await redis.sadd("sent_news", entry_id)
    await redis.expire("sent_news", Config.REDIS_TTL)

# --- PostgreSQL ---
pool = None

async def save_news(entry, ai_summary, sentiment, is_hot_news=False):
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO news_insights (title, link, summary, sentiment, ai_opinion, is_hot_news)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (link) DO NOTHING
            """, entry.title, entry.link, entry.summary, sentiment, ai_summary, is_hot_news)
    except Exception as e:
        logging.warning(f"Lỗi khi lưu tin tức vào DB (link={entry.link}): {e}")

async def is_in_db(entry):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT 1 FROM news_insights WHERE link=$1", entry.link)
        return row is not None

# Hàm xóa tin cũ hơn n ngày
async def delete_old_news(days=Config.DELETE_OLD_NEWS_DAYS):
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM news_insights WHERE created_at < NOW() - INTERVAL '{days} days';"
            )
    except Exception as e:
        logging.error(f"Lỗi khi xóa tin cũ: {e}")

# --- AI Analysis (Gemini) ---
GEMINI_MODEL = Config.GEMINI_MODEL
OPENROUTER_FALLBACK_MODEL = Config.OPENROUTER_FALLBACK_MODEL
GOOGLE_GEMINI_API_KEY = Config.GOOGLE_GEMINI_API_KEY

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
            logging.error(f"OpenRouter fallback cũng lỗi: {e2}")
            raise e2

# --- Extract sentiment from AI result ---
def extract_sentiment(ai_summary):
    """Extract sentiment from AI summary"""
    sentiment = "Trung lập"  # Default
    try:
        for line in ai_summary.splitlines():
            if "Cảm xúc:" in line:
                sentiment_text = line.split(":")[-1].strip().lower()
                if "tích cực" in sentiment_text:
                    return "Tích cực"
                elif "tiêu cực" in sentiment_text:
                    return "Tiêu cực"
                else:
                    return "Trung lập"
    except Exception as e:
        logging.warning(f"Lỗi khi parse sentiment: {e}")
    return sentiment

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
async def parse_feed(url):
    """Parse RSS feed with error handling and retries"""
    for attempt in range(Config.MAX_RETRIES):
        try:
            feed = feedparser.parse(url)
            if not feed.entries and not hasattr(feed, 'status'):
                raise Exception("Empty feed without status")
            return feed
        except Exception as e:
            logging.warning(f"Error parsing feed {url}, attempt {attempt+1}/{Config.MAX_RETRIES}: {e}")
            if attempt < Config.MAX_RETRIES - 1:
                await asyncio.sleep(1)  # Short delay before retry
            else:
                logging.error(f"Failed to parse feed after {Config.MAX_RETRIES} attempts: {url}")
                return feedparser.FeedParserDict(entries=[])  # Return empty feed

def extract_image_url(entry):
    if hasattr(entry, 'media_content') and entry.media_content:
        return entry.media_content[0].get('url')
    if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
        return entry.media_thumbnail[0].get('url')
    match = re.search(r'<img[^>]+src=["\"]([^"\"]+)["\"]', getattr(entry, 'summary', ''))
    if match:
        return match.group(1)
    return None
    
# --- Lệnh đăng ký user ---
@dp.message(Command("register"))
async def register_user(msg: types.Message):
    user_id = msg.from_user.id
    username = msg.from_user.username or ""
    async with pool.acquire() as conn:
        user = await conn.fetchrow("SELECT * FROM subscribed_users WHERE user_id=$1", user_id)
        if user:
            if user["is_approved"]:
                await msg.answer("Bạn đã được duyệt và sẽ nhận tin tức!")
            else:
                await msg.answer("Bạn đã đăng ký, vui lòng chờ admin duyệt!")
            return
            # Nếu là admin, tự động duyệt luôn
            if user_id == Config.ADMIN_ID:
                await conn.execute(
                    "INSERT INTO subscribed_users (user_id, username, is_approved) VALUES ($1, $2, TRUE) ON CONFLICT (user_id) DO UPDATE SET is_approved=TRUE",
                    user_id, username
                )
                await msg.answer("Bạn là admin, đã được duyệt và sẽ nhận tin tức!")
                return
            await conn.execute(
                "INSERT INTO subscribed_users (user_id, username, is_approved) VALUES ($1, $2, FALSE) ON CONFLICT (user_id) DO NOTHING",
                user_id, username
            )
        # Gửi thông báo cho admin
        kb = InlineKeyboardMarkup(
            inline_keyboard=[[InlineKeyboardButton(text="Duyệt user này", callback_data=f"approve_{user_id}")]]
        )
        await bot.send_message(
            Config.ADMIN_ID,
            f"Yêu cầu duyệt user mới: @{username} (ID: {user_id})",
            reply_markup=kb
        )
        await msg.answer("Đã gửi yêu cầu đăng ký, vui lòng chờ admin duyệt!")

# --- Xử lý callback admin duyệt user ---
@dp.callback_query(F.data.startswith("approve_"))
async def approve_user_callback(cb: CallbackQuery):
    if cb.from_user.id != Config.ADMIN_ID:
        await cb.answer("Chỉ admin mới được duyệt!", show_alert=True)
        return
    user_id = int(cb.data.split("_")[1])
    async with pool.acquire() as conn:
        await conn.execute("UPDATE subscribed_users SET is_approved=TRUE WHERE user_id=$1", user_id)
        user = await conn.fetchrow("SELECT username FROM subscribed_users WHERE user_id=$1", user_id)
    await bot.send_message(user_id, "Bạn đã được admin duyệt, sẽ nhận tin tức từ bot!")
    await cb.answer(f"Đã duyệt user @{user['username']} ({user_id})")

# Biến toàn cục để theo dõi trạng thái của news_job task
news_job_running = False

async def news_job():
    """
    Hàm chạy định kỳ để kiểm tra, phân tích và gửi tin tức mới.
    Được thiết kế để tránh chạy nhiều instance cùng lúc.
    """
    global news_job_running
    
    # Nếu đã có một instance của news_job đang chạy, trở về
    if news_job_running:
        logging.info("Phát hiện news_job đã đang chạy, bỏ qua việc khởi tạo task mới")
        return
    
    # Đánh dấu task đang chạy
    news_job_running = True
    logging.info("News job bắt đầu chạy")
    
    try:
        while True:
            try:
                # Xóa tin cũ khỏi DB định kỳ
                await delete_old_news()
                
                # Lấy danh sách URL nguồn tin
                feed_urls = Config.FEED_URLS
                all_entries = []
                all_normalized_titles = {}  # Lưu trữ tiêu đề đã chuẩn hóa
                
                # Lấy tin từ tất cả các nguồn
                for url in feed_urls:
                    feed = await parse_feed(url)
                    for entry in feed.entries:
                        # Tạo ID duy nhất nếu không có
                        if not hasattr(entry, 'id'):
                            entry.id = entry.link
                        
                        # Chuẩn hóa tiêu đề để tránh trùng lặp
                        normalized_title = normalize_title(entry.title)
                        # Lưu mapping giữa id và tiêu đề chuẩn hóa
                        all_normalized_titles[entry.id] = normalized_title
                        
                        # Kiểm tra xem tin đã được gửi hoặc đã lưu trong DB chưa
                        sent = await is_sent(entry.id) or await is_title_sent(normalized_title)
                        in_db = await is_in_db(entry)
                        
                        if not sent and not in_db:
                            all_entries.append(entry)
                
                # Sắp xếp tin mới theo thời gian nếu có thông tin published
                all_entries.sort(
                    key=lambda e: getattr(e, 'published_parsed', 0) or 0,
                    reverse=True
                )
                
                # Giới hạn số lượng tin phân tích mỗi chu kỳ
                new_entries = all_entries[:Config.MAX_NEWS_PER_CYCLE]
                
                if not new_entries:
                    logging.info("Không có tin mới trong chu kỳ này")
                else:
                    logging.info(f"Phát hiện {len(new_entries)} tin mới cần phân tích")
                
                # Phân tích AI cho tất cả tin mới
                ai_results = {}
                cached_results = {}
                
                for entry in new_entries:
                    try:
                        # Kiểm tra cache Redis trước
                        cached_summary = await redis.get(f"ai_summary:{entry.id}")
                        if cached_summary:
                            cached_results[entry.id] = cached_summary.decode('utf-8')
                            logging.info(f"Đã tìm thấy kết quả AI từ cache cho {entry.title}")
                            continue
                        
                        # Chuẩn bị prompt cho AI
                        prompt = f"""Phân tích tin tức sau và đưa ra nhận định với góc nhìn của nhà đầu tư:
Tiêu đề: {entry.title}
Tóm tắt: {getattr(entry, 'summary', 'Không có tóm tắt')}
URL: {entry.link}

Yêu cầu:
1. Tóm tắt ngắn gọn nội dung chính trong 1-2 câu
2. Giải thích ý nghĩa với thị trường tài chính/chứng khoán
3. Phân tích tác động ngắn và dài hạn có thể có
4. Cảm xúc: Tích cực/Tiêu cực/Trung lập (đặt ở cuối phân tích)

Viết ngắn gọn, súc tích trong 4-6 dòng."""
                        
                        # Phân tích bằng AI
                        ai_summary = await analyze_news(prompt)
                        ai_results[entry.id] = ai_summary
                        
                        # Lưu kết quả vào Redis cache
                        await redis.set(f"ai_summary:{entry.id}", ai_summary.encode('utf-8'), ex=Config.REDIS_TTL)
                        
                        # Đánh dấu đã gửi ngay sau khi phân tích
                        await mark_sent(entry.id)
                        await mark_title_sent(all_normalized_titles[entry.id])
                        
                        logging.info(f"Đã phân tích tin: {entry.title}")
                        # Đợi một chút giữa các lần gọi API để tránh rate limit
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logging.error(f"Lỗi phân tích tin {entry.title}: {str(e)}")
                
                # Lấy danh sách người dùng được duyệt để gửi tin
                async with pool.acquire() as conn:
                    rows = await conn.fetch("SELECT user_id FROM subscribed_users WHERE is_approved=TRUE")
                    users_to_notify = {row["user_id"] for row in rows}
                    users_to_notify.add(Config.ADMIN_ID)  # Luôn thêm admin vào danh sách nhận tin
                
                for entry in new_entries:
                    if entry.id in cached_results:
                        ai_summary = cached_results[entry.id]
                    elif entry.id in ai_results:
                        ai_summary = ai_results[entry.id]
                    else:
                        continue  # Không có kết quả AI

                    sentiment = extract_sentiment(ai_summary)
                    is_hot = is_hot_news(entry, ai_summary, sentiment)
                    await save_news(entry, ai_summary, sentiment, is_hot)
                    
                    # Lấy nguồn từ link (domain)
                    domain = urlparse(entry.link).netloc.replace('www.', '') if hasattr(entry, 'link') else ''
                    message = f"📰 *{entry.title}*\nNguồn: {domain}\n\n🤖 *Gemini AI phân tích:*\n{ai_summary}"
                    
                    # Phát hiện và gửi thông báo đặc biệt cho tin nóng
                    if is_hot:
                        hot_message = f"🔥🔥 *TIN NÓNG - QUAN TRỌNG!* 🔥🔥\n\n{message}\n\n⚠️ *Tin này có thể ảnh hưởng lớn đến thị trường*"
                        sending_tasks = []
                        for user_id in users_to_notify:
                            sending_tasks.append(send_message_to_user(user_id, hot_message, entry=entry, is_hot_news=True))
                        if sending_tasks:
                            await asyncio.gather(*sending_tasks, return_exceptions=True)
                    else:
                        sending_tasks = []
                        for user_id in users_to_notify:
                            sending_tasks.append(send_message_to_user(user_id, message, entry=entry))
                        if sending_tasks:
                            await asyncio.gather(*sending_tasks, return_exceptions=True)
                    
            except Exception as e:
                logging.error(f"Lỗi trong chu kỳ news_job: {e}")
                
            await asyncio.sleep(Config.NEWS_JOB_INTERVAL)
    except asyncio.CancelledError:
        logging.info("News job bị hủy")
    except Exception as e:
        logging.error(f"Lỗi nghiêm trọng trong news_job: {e}")
    finally:
        # Đánh dấu task đã kết thúc
        news_job_running = False
        logging.info("News job đã kết thúc")

async def send_message_to_user(user_id, message, entry=None, is_hot_news=False):
    """Send message to user với error handling, kèm ảnh nếu có (chỉ gửi qua bot chính)"""
    try:
        image_url = extract_image_url(entry) if entry else None
        
        # Nếu là hot news, thêm các tùy chọn đặc biệt
        if is_hot_news:
            # Gửi với disable_notification=False để đảm bảo thông báo push được gửi
            if image_url:
                await bot.send_photo(
                    user_id, 
                    image_url, 
                    caption=message, 
                    parse_mode="Markdown",
                    disable_notification=False
                )
            else:
                await bot.send_message(
                    user_id, 
                    message, 
                    parse_mode="Markdown",
                    disable_notification=False
                )
        else:
            # Cho tin thông thường
            if image_url:
                await bot.send_photo(user_id, image_url, caption=message, parse_mode="Markdown")
            else:
                await bot.send_message(user_id, message, parse_mode="Markdown")
    except Exception as e:
        logging.warning(f"Không gửi được tin cho user {user_id}: {e}")

# --- Webhook setup ---
# (Không đăng ký bất kỳ handler nào cho lệnh từ user)
async def init_db():
    async with pool.acquire() as conn:
        # Tạo bảng news_insights nếu chưa có
        await conn.execute('''
        CREATE TABLE IF NOT EXISTS news_insights (
            id SERIAL PRIMARY KEY,
            title TEXT,
            link TEXT UNIQUE,
            summary TEXT,
            sentiment TEXT,
            ai_opinion TEXT,
            is_hot_news BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT NOW()
        );
        ''')
        # Đảm bảo cột created_at tồn tại (nếu migrate từ bản cũ)
        await conn.execute('''
        ALTER TABLE news_insights ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
        ''')
        # Đảm bảo cột is_hot_news tồn tại (nếu migrate từ bản cũ)
        await conn.execute('''
        ALTER TABLE news_insights ADD COLUMN IF NOT EXISTS is_hot_news BOOLEAN DEFAULT FALSE;
        ''')
        # Tạo bảng subscribed_users nếu chưa có
        await conn.execute('''
        CREATE TABLE IF NOT EXISTS subscribed_users (
            user_id BIGINT PRIMARY KEY,
            username TEXT,
            is_approved BOOLEAN DEFAULT FALSE
        );
        ''')

# --- 8. Webhook & main ---
async def on_startup(app):
    try:
        global redis, pool, news_job_running
        
        # Reset trạng thái và hủy task cũ nếu có
        news_job_running = False
        await cancel_running_tasks()
        
        logging.info("Bot khởi động, thiết lập kết nối Redis...")
        redis = await aioredis.from_url(Config.REDIS_URL)
        logging.info("Thiết lập kết nối PostgreSQL...")
        pool = await asyncpg.create_pool(dsn=Config.DB_URL)
        logging.info("Khởi tạo database...")
        await init_db()
        logging.info(f"Thiết lập webhook: {Config.WEBHOOK_URL}")
        await bot.delete_webhook() # Xóa webhook cũ nếu có
        result = await bot.set_webhook(Config.WEBHOOK_URL)
        logging.info(f"Kết quả thiết lập webhook: {result}")
        
        # Kiểm tra webhook đã set đúng chưa
        webhook_info = await bot.get_webhook_info()
        logging.info(f"WebhookInfo: URL={webhook_info.url}, pending_updates={webhook_info.pending_update_count}")
        
        logging.info("Khởi động task gửi tin...")
        task = asyncio.create_task(news_job())
        task.set_name("news_job")
        logging.info("Bot đã sẵn sàng hoạt động!")
    except Exception as e:
        logging.error(f"Lỗi trong on_startup: {e}")
        raise e

async def on_shutdown(app):
    logging.info("Bot đang tắt...")
    await bot.delete_webhook()
    if pool:
        await pool.close()
    if redis:
        await redis.close()
    logging.info("Bot đã tắt hoàn toàn.")

# Route cho healthcheck
async def healthcheck(request):
    return web.Response(text="Bot đang hoạt động!", status=200)

# Route cho ping để giữ bot hoạt động
async def ping_bot(request):
    logging.info("Nhận yêu cầu ping")
    return web.Response(text="pong", status=200)

async def cancel_running_tasks():
    """Hủy các task đang chạy để tránh bị treo"""
    try:
        for task in asyncio.all_tasks():
            if task.get_name() == "news_job" and not task.done():
                logging.info("Đang hủy task news_job cũ...")
                task.cancel()
                try:
                    # Chờ task kết thúc nếu đang bị hủy
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.TimeoutError:
                    logging.warning("Hủy task news_job cũ bị timeout")
                except asyncio.CancelledError:
                    logging.info("Đã hủy thành công task news_job cũ")
    except Exception as e:
        logging.error(f"Lỗi khi hủy tasks: {e}")

# Route cho restart từ bên ngoài (có thể sử dụng cronjob để gọi định kỳ)
async def restart_bot(request):
    try:
        logging.info("Yêu cầu khởi động lại bot từ endpoint /restart")
        
        global redis, pool, news_job_running
        
        # Reset trạng thái news_job và hủy task cũ
        news_job_running = False
        await cancel_running_tasks()
        
        # Đóng kết nối cũ
        if redis:
            try:
                await redis.close()
                logging.info("Đã đóng kết nối Redis")
            except Exception as e:
                logging.warning(f"Không thể đóng kết nối Redis: {e}")
        
        if pool:
            try:
                await pool.close()
                logging.info("Đã đóng kết nối PostgreSQL")
            except Exception as e:
                logging.warning(f"Không thể đóng kết nối PostgreSQL: {e}")
        
        # Khởi tạo lại kết nối
        logging.info("Khởi tạo lại kết nối Redis...")
        redis = await aioredis.from_url(Config.REDIS_URL)
        
        logging.info("Khởi tạo lại kết nối PostgreSQL...")
        pool = await asyncpg.create_pool(dsn=Config.DB_URL)
        
        logging.info("Khởi tạo lại database...")
        await init_db()
        
        # Thiết lập lại webhook
        logging.info(f"Thiết lập lại webhook: {Config.WEBHOOK_URL}")
        await bot.delete_webhook()
        result = await bot.set_webhook(Config.WEBHOOK_URL)
        logging.info(f"Kết quả thiết lập lại webhook: {result}")
        
        # Khởi động lại task tin tức
        task = asyncio.create_task(news_job())
        task.set_name("news_job")
            
        return web.Response(text="Bot đã được khởi động lại thành công!", status=200)
    except Exception as e:
        error_msg = f"Lỗi khi khởi động lại bot: {str(e)}"
        logging.error(error_msg)
        return web.Response(text=error_msg, status=500)

@dp.message(Command("start"))
async def start_command(msg: types.Message):
    try:
        # Thử khởi động lại bot và kết nối các dịch vụ
        logging.info("Lệnh /start được gọi - đang khởi động lại các dịch vụ...")
        
        global redis, pool, news_job_running
        
        # Reset trạng thái news_job và hủy task cũ
        news_job_running = False
        await cancel_running_tasks()
        
        # Đóng kết nối cũ nếu có
        if redis:
            try:
                await redis.close()
                logging.info("Đã đóng kết nối Redis cũ")
            except Exception as e:
                logging.warning(f"Không thể đóng kết nối Redis cũ: {e}")
        
        if pool:
            try:
                await pool.close()
                logging.info("Đã đóng kết nối PostgreSQL cũ")
            except Exception as e:
                logging.warning(f"Không thể đóng kết nối PostgreSQL cũ: {e}")
        
        # Khởi tạo lại kết nối Redis
        logging.info("Khởi tạo lại kết nối Redis...")
        redis = await aioredis.from_url(Config.REDIS_URL)
        
        # Khởi tạo lại kết nối PostgreSQL
        logging.info("Khởi tạo lại kết nối PostgreSQL...")
        pool = await asyncpg.create_pool(dsn=Config.DB_URL)
        
        # Khởi tạo lại database
        logging.info("Khởi tạo lại database...")
        await init_db()
        
        # Thiết lập lại webhook
        logging.info(f"Thiết lập lại webhook: {Config.WEBHOOK_URL}")
        await bot.delete_webhook()
        result = await bot.set_webhook(Config.WEBHOOK_URL)
        logging.info(f"Kết quả thiết lập lại webhook: {result}")
        
        # Kiểm tra webhook đã set đúng chưa
        webhook_info = await bot.get_webhook_info()
        logging.info(f"WebhookInfo: URL={webhook_info.url}, pending_updates={webhook_info.pending_update_count}")
        
        # Khởi động lại task xử lý tin tức
        task = asyncio.create_task(news_job())
        task.set_name("news_job")
        
        logging.info("Bot đã được khởi động lại thành công!")
        
        # Gửi thông báo thành công
        await msg.answer(
            "✅ Bot đã được khởi động lại thành công!\n\n"
            "Chào mừng bạn đến với bot tin tức tài chính!\n"
            "- Gõ /register để đăng ký nhận tin tức.\n"
            "- Sau khi được admin duyệt, bạn sẽ nhận tin tức tự động.\n"
            "- Tin nóng sẽ được đánh dấu đặc biệt 🔥 và gửi thông báo.\n"
            "- Gõ /help để xem hướng dẫn."
        )
    except Exception as e:
        logging.error(f"Lỗi khi thử khởi động lại bot từ lệnh /start: {e}")
        await msg.answer(
            f"❌ Không thể khởi động lại bot: {str(e)}\n\n"
            "Chào mừng bạn đến với bot tin tức tài chính!\n"
            "- Gõ /register để đăng ký nhận tin tức.\n"
            "- Sau khi được admin duyệt, bạn sẽ nhận tin tức tự động.\n"
            "- Tin nóng sẽ được đánh dấu đặc biệt 🔥 và gửi thông báo.\n"
            "- Gõ /help để xem hướng dẫn."
        )

@dp.message(Command("help"))
async def help_command(msg: types.Message):
    await msg.answer(
        "📚 *Hướng dẫn sử dụng Bot Tin Tức Tài Chính*\n\n"
        "- Bot sẽ tự động gửi tin tức tài chính mới.\n"
        "- Mỗi tin đều được phân tích bởi AI (Gemini).\n"
        "- 🔥 *Tin nóng (Hot News)*: Những tin quan trọng, có ảnh hưởng lớn đến thị trường sẽ được đánh dấu đặc biệt.\n\n"
        "*Các lệnh:*\n"
        "/start - Khởi động bot\n"
        "/register - Đăng ký nhận tin tức\n"
        "/help - Xem hướng dẫn\n\n"
        "Tin tức được cập nhật mỗi 10 phút."
    )

def normalize_title(title):
    """Chuẩn hóa tiêu đề: viết thường, loại bỏ dấu, ký tự đặc biệt, khoảng trắng thừa"""
    if not title:
        return ""
    # Loại bỏ dấu tiếng Việt
    title = unicodedata.normalize('NFD', title)
    title = ''.join([c for c in title if unicodedata.category(c) != 'Mn'])
    # Viết thường, loại bỏ ký tự đặc biệt, khoảng trắng thừa
    title = re.sub(r'[^a-zA-Z0-9 ]', '', title)
    title = title.lower().strip()
    return title

async def is_title_sent(normalized_title):
    """Kiểm tra tiêu đề đã chuẩn hóa đã gửi chưa (Redis set)"""
    return await redis.sismember("sent_titles", normalized_title)

async def mark_title_sent(normalized_title):
    await redis.sadd("sent_titles", normalized_title)
    await redis.expire("sent_titles", Config.REDIS_TTL)

app = web.Application()
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)
app.router.add_get("/", healthcheck)  # Thêm route / để kiểm tra bot sống
app.router.add_get("/ping", ping_bot)  # Thêm route /ping để giữ bot hoạt động
app.router.add_get("/restart", restart_bot)  # Thêm route /restart để khởi động lại bot
SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path="/webhook")
setup_application(app, dp, bot=bot)

if __name__ == "__main__":
    logging.info("Khởi động web server...")
    web.run_app(app, host="0.0.0.0", port=8000)
