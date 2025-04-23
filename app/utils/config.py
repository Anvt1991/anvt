import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost/stock_bot")

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_CACHE_EXPIRY = int(os.getenv("REDIS_CACHE_EXPIRY", 86400))  # 24 hours default

# Telegram Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
ADMIN_USER_IDS = os.getenv("ADMIN_USER_IDS", "").split(",")

# AI API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Default Values
DEFAULT_CANDLES = int(os.getenv("DEFAULT_CANDLES", 365))
DEFAULT_TIMEFRAMES = ["1D", "1W", "1M"]

# Data Source Configuration
DEFAULT_DATA_SOURCE = os.getenv("DEFAULT_DATA_SOURCE", "vnstock")

# Common Vietnam market symbols
VN_INDICES = ["VNINDEX", "VN30", "HNX", "UPCOM"]
DEFAULT_MARKET_SYMBOLS = ["FPT", "VCB", "VHM", "VIC", "VNM", "HPG", "MWG", "MSN", "TCB"]

# Telegram webhook configuration
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")
WEBHOOK_PORT = int(os.getenv("WEBHOOK_PORT", 8443))
WEBHOOK_LISTEN = os.getenv("WEBHOOK_LISTEN", "0.0.0.0")

# News sources
NEWS_SOURCES = {
    "vn_economy": "https://vneconomy.vn/rss/trang-chu.rss",
    "cafef": "https://cafef.vn/rss/thoi-su-kinh-doanh.rss",
    "vietstock": "https://vietstock.vn/feed",
    "ndh": "https://ndh.vn/rss/ngan-hang.rss"
}

# Technical analysis settings
TECHNICAL_INDICATOR_PERIODS = [14, 20, 50, 100, 200] 