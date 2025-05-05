import logging
import os
import feedparser
import httpx
import asyncio
import asyncpg
import redis.asyncio as aioredis
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
import google.generativeai as genai

# --- Config ---
class Config:
    BOT_TOKEN = os.getenv("BOT_TOKEN")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")
    DB_URL = os.getenv("DATABASE_URL")
    ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
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

# --- Kiểm tra biến môi trường bắt buộc ---
REQUIRED_ENV_VARS = ["BOT_TOKEN", "OPENROUTER_API_KEY"]  # Không còn CHANNEL_ID
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
    await redis.expire("sent_news", 21600)  # 6h

# --- PostgreSQL ---
pool = None

async def save_news(entry, ai_summary, sentiment):
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO news_insights (title, link, summary, sentiment, ai_opinion)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (link) DO NOTHING
        """, entry.title, entry.link, entry.summary, sentiment, ai_summary)

# --- AI Analysis (Gemini) ---
GEMINI_MODEL = Config.GEMINI_MODEL
OPENROUTER_FALLBACK_MODEL = Config.OPENROUTER_FALLBACK_MODEL
GOOGLE_GEMINI_API_KEY = Config.GOOGLE_GEMINI_API_KEY

async def analyze_news(title, summary, model=None):
    prompt = f"""
    Đây là một tin tức tài chính:
    ---
    {title}
    {summary}
    ---
    Hãy:
    1. Tóm tắt ngắn gọn (dưới 2 câu)
    2. Đưa ra nhận định thị trường ngắn gọn (dưới 2 câu)
    3. Phân tích cảm xúc: tích cực / tiêu cực / trung lập.
    Trả về kết quả theo định dạng:
    - Tóm tắt:
    - Nhận định:
    - Cảm xúc:
    """
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

async def analyze_news_cached(entry_id, title, summary):
    cache_key = f"ai_summary:{entry_id}"
    cached = await redis.get(cache_key)
    if cached:
        return cached.decode()
    result = await analyze_news(title, summary)
    await redis.set(cache_key, result, ex=21600)  # TTL 6h
    return result

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
async def news_job():
    while True:
        await delete_old_news(days=7)  # Xóa tin cũ hơn 7 ngày
        for url in Config.FEED_URLS:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if await is_sent(entry.id):
                    continue
                await mark_sent(entry.id)

                title = entry.title
                summary = entry.summary

                # Luôn phân tích AI, không dịch nữa
                ai_summary = await analyze_news_cached(entry.id, title, summary)

                sentiment = "Trung lập"
                for line in ai_summary.splitlines():
                    if "Cảm xúc" in line:
                        sentiment = line.split(":")[-1].strip()
                        break

                await save_news(entry, ai_summary, sentiment)

                # Lấy danh sách user đã duyệt từ DB và gửi tin
                async with pool.acquire() as conn:
                    rows = await conn.fetch("SELECT user_id FROM subscribed_users WHERE is_approved=TRUE")
                for row in rows:
                    await bot.send_message(row["user_id"], f"📰 *{title}*\n{entry.link}\n\n🤖 *Gemini AI phân tích:*\n{ai_summary}", parse_mode="Markdown")
                break  # Chỉ gửi 1 tin mới đầu tiên mỗi nguồn

        await asyncio.sleep(14 * 60)  # 14 phút

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

# Hàm xóa tin cũ hơn 7 ngày
async def delete_old_news(days=7):
    async with pool.acquire() as conn:
        await conn.execute(
            f"DELETE FROM news_insights WHERE created_at < NOW() - INTERVAL '{days} days';"
        )

async def on_startup(app):
    global redis, pool
    redis = await aioredis.from_url(Config.REDIS_URL)
    pool = await asyncpg.create_pool(dsn=Config.DB_URL)
    await init_db()  # Tự động tạo bảng nếu chưa có
    await bot.set_webhook(Config.WEBHOOK_URL)
    asyncio.create_task(news_job())

async def on_shutdown(app):
    await bot.delete_webhook()
    await pool.close()
    await redis.close()

app = web.Application()
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)
SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path="/webhook")
setup_application(app, dp, bot=bot)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8000)
