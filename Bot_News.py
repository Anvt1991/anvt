import logging
import os
import feedparser
import httpx
import asyncio
import asyncpg
import aioredis
from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from aiogram.filters import Command
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery

# --- Config ---
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost")
DB_URL = os.getenv("DATABASE_URL")
ADMIN_ID = int(os.getenv("ADMIN_ID", "0"))
CHANNEL_ID = os.getenv("CHANNEL_ID")  # Chat id của channel Telegram
# Danh sách nguồn tin Google News RSS theo chủ đề nóng (tiếng Việt và tiếng Anh)
FEED_URLS = [
    # --- Tiếng Việt ---
    # Kinh tế
    "https://news.google.com/rss/search?q=kinh+t%E1%BA%BF&hl=vi&gl=VN&ceid=VN:vi",
    # Chứng khoán
    "https://news.google.com/rss/search?q=ch%E1%BB%A9ng+kho%C3%A1n&hl=vi&gl=VN&ceid=VN:vi",
    # Vĩ mô
    "https://news.google.com/rss/search?q=v%C4%A9+m%C3%B4&hl=vi&gl=VN&ceid=VN:vi",
    # Chiến tranh
    "https://news.google.com/rss/search?q=chi%E1%BA%BFn+tranh&hl=vi&gl=VN&ceid=VN:vi",
    # Lãi suất
    "https://news.google.com/rss/search?q=l%C3%A3i+su%E1%BA%A5t&hl=vi&gl=VN&ceid=VN:vi",
    # Fed
    "https://news.google.com/rss/search?q=fed&hl=vi&gl=VN&ceid=VN:vi",
    # Tin nóng
    "https://news.google.com/rss?hl=vi&gl=VN&ceid=VN:vi",
    # --- Tiếng Anh ---
    # Stock market
    "https://news.google.com/rss/search?q=stock+market&hl=en&gl=US&ceid=US:en",
    # Economic policy
    "https://news.google.com/rss/search?q=economic+policy&hl=en&gl=US&ceid=US:en",
    # Macro economics
    "https://news.google.com/rss/search?q=macroeconomics&hl=en&gl=US&ceid=US:en",
    # Federal Reserve (Fed)
    "https://news.google.com/rss/search?q=federal+reserve+OR+fed&hl=en&gl=US&ceid=US:en",
    # Interest rates
    "https://news.google.com/rss/search?q=interest+rates&hl=en&gl=US&ceid=US:en",
    # War (vĩ mô, địa chính trị)
    "https://news.google.com/rss/search?q=war&hl=en&gl=US&ceid=US:en",
    # Breaking news (kinh tế, tài chính)
    "https://news.google.com/rss/search?q=breaking+news+economy+finance&hl=en&gl=US&ceid=US:en",
]

# --- Logging ---
logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "google/gemini-pro")
OPENROUTER_FALLBACK_MODEL = os.getenv("OPENROUTER_FALLBACK_MODEL", "openai/gpt-3.5-turbo")

async def analyze_news(title, summary, model=None):
    if model is None:
        model = GEMINI_MODEL
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
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7
                }
            )
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        if model == GEMINI_MODEL:
            print(f"Gemini lỗi: {e}, fallback sang {OPENROUTER_FALLBACK_MODEL}")
            return await analyze_news(title, summary, model=OPENROUTER_FALLBACK_MODEL)
        else:
            raise e

async def analyze_news_cached(entry_id, title, summary):
    cache_key = f"ai_summary:{entry_id}"
    cached = await redis.get(cache_key)
    if cached:
        return cached.decode()
    result = await analyze_news(title, summary)
    await redis.set(cache_key, result, ex=21600)  # TTL 6h
    return result

# --- Dịch tự động với LibreTranslate ---
async def translate_text(text, source_lang="en", target_lang="vi"):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://libretranslate.de/translate",
            data={
                "q": text,
                "source": source_lang,
                "target": target_lang,
                "format": "text"
            }
        )
        return response.json()["translatedText"]

# --- Tin tức theo chu kỳ ---
async def news_job():
    while True:
        for url in FEED_URLS:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if await is_sent(entry.id):
                    continue
                await mark_sent(entry.id)

                title = entry.title
                summary = entry.summary

                is_english = any(ord(c) < 128 for c in title)

                if is_english:
                    ai_summary_en = await analyze_news_cached(entry.id, title, summary)
                    ai_summary = await translate_text(ai_summary_en, source_lang="en", target_lang="vi")
                else:
                    ai_summary = await analyze_news_cached(entry.id, title, summary)

                sentiment = "Trung lập"
                for line in ai_summary.splitlines():
                    if "Cảm xúc" in line:
                        sentiment = line.split(":")[-1].strip()
                        break

                await save_news(entry, ai_summary, sentiment)

                # Gửi tin vào channel duy nhất
                await bot.send_message(CHANNEL_ID, f"📰 *{title}*\n{entry.link}\n\n🤖 *Gemini AI phân tích:*\n{ai_summary}", parse_mode="Markdown")

        await asyncio.sleep(14 * 60)  # 14 phút

# --- Webhook setup ---
# (Không đăng ký bất kỳ handler nào cho lệnh từ user)
async def on_startup(app):
    global redis, pool
    redis = await aioredis.from_url(REDIS_URL)
    pool = await asyncpg.create_pool(dsn=DB_URL)
    await bot.set_webhook(WEBHOOK_URL)
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
