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
        "https://news.google.com/rss?hl=vi&gl=VN&ceid=VN:vi",
    ]
    REDIS_TTL = int(os.getenv("REDIS_TTL", "21600"))  # 6h
    NEWS_JOB_INTERVAL = int(os.getenv("NEWS_JOB_INTERVAL", "600"))  # 10 phút (giây)
    DELETE_OLD_NEWS_DAYS = int(os.getenv("DELETE_OLD_NEWS_DAYS", "7"))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # Số lần thử lại khi feed lỗi
    MAX_NEWS_PER_CYCLE = int(os.getenv("MAX_NEWS_PER_CYCLE", "3"))  # Tối đa 3 tin mỗi lần

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

async def save_news(entry, ai_summary, sentiment):
    try:
        async with pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO news_insights (title, link, summary, sentiment, ai_opinion)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (link) DO NOTHING
            """, entry.title, entry.link, entry.summary, sentiment, ai_summary)
    except Exception as e:
        logging.warning(f"Lỗi khi lưu tin tức vào DB (link={entry.link}): {e}")

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

# --- Parse RSS Feed ---
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

# --- Tin tức theo chu kỳ ---
async def is_in_db(entry):
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT 1 FROM news_insights WHERE link=$1", entry.link)
        return row is not None

async def news_job():
    while True:
        await delete_old_news(days=Config.DELETE_OLD_NEWS_DAYS)
        new_entries = []
        cached_results = {}
        entries_to_analyze = []

        try:
            # Collect entries from feeds
            for url in Config.FEED_URLS:
                feed = await parse_feed(url)
                for entry in feed.entries:
                    if await is_sent(entry.id) or await is_in_db(entry):
                        continue
                    await mark_sent(entry.id)
                    cache_key = f"ai_summary:{entry.id}"
                    cached = await redis.get(cache_key)
                    if cached:
                        cached_results[entry.id] = cached.decode()
                    else:
                        entries_to_analyze.append(entry)
                    new_entries.append(entry)
                    if len(new_entries) >= Config.MAX_NEWS_PER_CYCLE:
                        break
                if len(new_entries) >= Config.MAX_NEWS_PER_CYCLE:
                    break

            # Gọi Gemini cho các entry chưa có cache
            ai_results = {}
            if entries_to_analyze:
                prompt = "Đây là các tin tức tài chính:\n"
                for idx, entry in enumerate(entries_to_analyze, 1):
                    prompt += f"\n---\nTin số {idx}:\n{entry.title}\n{entry.summary}\n"
                prompt += '''
Hãy với mỗi tin:
1. Tóm tắt ngắn gọn (dưới 2 câu)
2. Đưa ra nhận định thị trường ngắn gọn (dưới 2 câu)
3. Phân tích cảm xúc: tích cực / tiêu cực / trung lập.
Trả về kết quả cho từng tin theo định dạng:
- Tin số X:
  - Tóm tắt:
  - Nhận định:
  - Cảm xúc:
'''
                ai_result = await analyze_news(prompt)
                
                try:
                    # Parse kết quả từ Gemini
                    results = re.split(r"- Tin số \d+:", ai_result)[1:]  # Sửa regex
                    
                    # Check if we have enough results for all entries
                    if len(results) >= len(entries_to_analyze):
                        for entry, idx, result in zip(entries_to_analyze, range(1, len(entries_to_analyze)+1), results):
                            ai_summary = f"- Tin số {idx}:{result.strip()}"
                            ai_results[entry.id] = ai_summary
                            await redis.set(f"ai_summary:{entry.id}", ai_summary, ex=Config.REDIS_TTL)
                    else:
                        logging.warning(f"Gemini trả về không đủ kết quả: {len(results)} results for {len(entries_to_analyze)} entries")
                        # Backup: tạo kết quả trống cho mọi entry chưa phân tích
                        for entry in entries_to_analyze:
                            if entry.id not in ai_results:
                                ai_results[entry.id] = f"- Tin số 0:\n  - Tóm tắt: {entry.title}\n  - Nhận định: Không có đủ thông tin\n  - Cảm xúc: Trung lập"
                except Exception as e:
                    logging.error(f"Lỗi khi parse kết quả Gemini: {e}, kết quả: {ai_result}")
                    # Tạo kết quả trống cho mọi entry chưa phân tích
                    for entry in entries_to_analyze:
                        ai_results[entry.id] = f"- Tin số 0:\n  - Tóm tắt: {entry.title}\n  - Nhận định: Lỗi phân tích\n  - Cảm xúc: Trung lập"

            # Gửi và lưu DB cho tất cả entry
            users_to_notify = []
            async with pool.acquire() as conn:
                rows = await conn.fetch("SELECT user_id FROM subscribed_users WHERE is_approved=TRUE")
                users_to_notify = [row["user_id"] for row in rows]
            
            for entry in new_entries:
                if entry.id in cached_results:
                    ai_summary = cached_results[entry.id]
                elif entry.id in ai_results:
                    ai_summary = ai_results[entry.id]
                else:
                    continue  # Không có kết quả AI

                sentiment = extract_sentiment(ai_summary)
                await save_news(entry, ai_summary, sentiment)
                
                message = f"📰 *{entry.title}*\n{entry.link}\n\n🤖 *Gemini AI phân tích:*\n{ai_summary}"
                
                # Send to all users (in parallel using gather)
                sending_tasks = []
                for user_id in users_to_notify:
                    sending_tasks.append(send_message_to_user(user_id, message, entry=entry))
                if sending_tasks:
                    await asyncio.gather(*sending_tasks, return_exceptions=True)
                
        except Exception as e:
            logging.error(f"Lỗi trong chu kỳ news_job: {e}")
            
        await asyncio.sleep(Config.NEWS_JOB_INTERVAL)

async def send_message_to_user(user_id, message, entry=None):
    """Send message to user with error handling, kèm ảnh nếu có"""
    try:
        image_url = extract_image_url(entry) if entry else None
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
            created_at TIMESTAMP DEFAULT NOW()
        );
        ''')
        # Đảm bảo cột created_at tồn tại (nếu migrate từ bản cũ)
        await conn.execute('''
        ALTER TABLE news_insights ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();
        ''')
        # Tạo bảng subscribed_users nếu chưa có
        await conn.execute('''
        CREATE TABLE IF NOT EXISTS subscribed_users (
            user_id BIGINT PRIMARY KEY,
            username TEXT,
            is_approved BOOLEAN DEFAULT FALSE
        );
        ''')

# Hàm xóa tin cũ hơn n ngày
async def delete_old_news(days=Config.DELETE_OLD_NEWS_DAYS):
    try:
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM news_insights WHERE created_at < NOW() - INTERVAL '{days} days';"
            )
    except Exception as e:
        logging.error(f"Lỗi khi xóa tin cũ: {e}")

# --- 8. Webhook & main ---
async def on_startup(app):
    try:
        global redis, pool
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
        asyncio.create_task(news_job())
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

app = web.Application()
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)
app.router.add_get("/", healthcheck)  # Thêm route / để kiểm tra bot sống
SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path="/webhook")
setup_application(app, dp, bot=bot)

if __name__ == "__main__":
    logging.info("Khởi động web server...")
    web.run_app(app, host="0.0.0.0", port=8000)

def extract_image_url(entry):
    # 1. RSS chuẩn có thể có media_content
    if hasattr(entry, 'media_content') and entry.media_content:
        return entry.media_content[0].get('url')
    # 2. RSS có thể có media_thumbnail
    if hasattr(entry, 'media_thumbnail') and entry.media_thumbnail:
        return entry.media_thumbnail[0].get('url')
    # 3. Tìm ảnh trong summary (nếu có thẻ <img>)
    match = re.search(r'<img[^>]+src=["\"]([^"\"]+)["\"]', getattr(entry, 'summary', ''))
    if match:
        return match.group(1)
    return None

@dp.message(Command("start"))
async def start_command(msg: types.Message):
    await msg.answer(
        "Chào mừng bạn đến với bot tin tức tài chính!\n" \
        "- Gõ /register để đăng ký nhận tin tức.\n" \
        "- Sau khi được admin duyệt, bạn sẽ nhận tin tức tự động.\n" \
        "- Gõ /help để xem hướng dẫn."
    )
