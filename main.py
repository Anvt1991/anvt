#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot Chứng Khoán Toàn Diện Phiên Bản V18.7 (Tối ưu hóa):
- Tải dữ liệu chứng khoán qua VNStock/Yahoo Finance với cache Redis (async) + caching thông minh.
- Hợp nhất, chuẩn hóa dữ liệu, bổ sung khung thời gian (1w, 1mo), phát hiện bất thường.
- Phân tích kỹ thuật đa khung thời gian (SMA, RSI, MACD, ...) với tối ưu tính toán.
- Thu thập tin tức từ nhiều nguồn RSS (async + parallel fetching).
- Phân tích cơ bản nâng cao, phân biệt cổ phiếu/chỉ số.
- Dự báo giá (Prophet) và tín hiệu giao dịch (XGBoost) với cải tiến hiệu suất model.
- Báo cáo phân tích bằng Gemini AI, lưu lịch sử vào PostgreSQL (async + connection pooling).
- Tích hợp Telegram với các lệnh: /start, /analyze, /getid, /approve.
- Auto training mô hình theo lịch định kỳ (mỗi ngày 2h sáng) + incremental training.
- Tối ưu deploy trên Render với webhook, xử lý async hoàn toàn.
- Nâng cấp: Tối ưu hiệu suất, cache thông minh, kiểm soát dữ liệu, cải thiện mô hình ML, báo cáo đẹp hơn.
- Cải tiến: Tách bạch raw data ↔ cleaned data ↔ indicator data, pipeline xử lý dữ liệu, cache versioning, timezone consistency, unit tests mở rộng.
- Tối ưu: Kết nối bất đồng bộ cho DB và Redis với connection pooling, retry thông minh, quản lý bộ nhớ.
- Hiệu suất: Cải thiện xử lý dữ liệu lớn, giảm độ trễ, mô hình ML nhẹ hơn & nhanh hơn.
- Nâng cấp mới: Validate & normalizing data, phát hiện outlier nâng cao, lọc ngày lễ Việt Nam, tách pipeline xử lý dữ liệu.
"""

import os
import sys
import io
import logging
import pickle
import time
import warnings
from functools import lru_cache, wraps
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import json
import inspect
from inspect import isawaitable

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import yfinance as yf
from vnstock import Vnstock
import google.generativeai as genai

from ta import trend, momentum, volatility
from ta.volume import MFIIndicator
import feedparser

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, select, LargeBinary, Index, func

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import LocalOutlierFactor
from prophet import Prophet
import holidays
import html
from aiohttp import web
import unittest

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tenacity import retry, stop_after_attempt, wait_exponential

# Suppress warnings
warnings.filterwarnings('ignore')

# ---------- CẤU HÌNH & LOGGING ----------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")          # ví dụ: postgres://user:pass@hostname:port/dbname
REDIS_URL = os.getenv("REDIS_URL")                # ví dụ: redis://:pass@hostname:port/0
ADMIN_ID = os.getenv("ADMIN_ID", "1225226589")
PORT = int(os.environ.get("PORT", 10000))
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")
RENDER_SERVICE_NAME = os.getenv("RENDER_SERVICE_NAME", "")

# Thêm cache version để quản lý phiên bản cache
CACHE_VERSION = "v1.0"

# Tối ưu cấu hình logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler("stock_bot.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Ensure system can handle Unicode properly
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace') if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace') if hasattr(sys.stderr, 'reconfigure') else None

# Cache expiration settings
CACHE_EXPIRE_SHORT = 1800         # 30 phút cho dữ liệu ngắn hạn
CACHE_EXPIRE_MEDIUM = 3600        # 1 giờ cho dữ liệu trung hạn
CACHE_EXPIRE_LONG = 86400         # 1 ngày cho dữ liệu dài hạn
NEWS_CACHE_EXPIRE = 900           # 15 phút cho tin tức
DEFAULT_CANDLES = 100
DEFAULT_TIMEFRAME = '1D'

# Tăng số lượng worker cho thread pool
thread_executor = ThreadPoolExecutor(max_workers=10)
# Thêm process pool cho các tác vụ nặng về CPU
process_executor = ProcessPoolExecutor(max_workers=4)

# Decorator để đo thời gian thực thi của hàm
def measure_execution_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_executor, lambda: func(*args, **kwargs))

async def run_in_process(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(process_executor, lambda: func(*args, **kwargs))

# Hàm tạo cache key với version
def make_cache_key(prefix, params):
    """Tạo cache key với version để quản lý phiên bản cache"""
    if isinstance(params, dict):
        # Sort keys để đảm bảo tính nhất quán
        params_str = json.dumps(params, sort_keys=True)
    else:
        params_str = str(params)
    
    hash_key = hashlib.md5(params_str.encode()).hexdigest()
    return f"{prefix}:{CACHE_VERSION}:{hash_key}"

# ---------- KẾT NỐI REDIS (Async) ----------
class RedisManager:
    def __init__(self):
        try:
            # Ensure Redis URL has the proper scheme
            redis_url = REDIS_URL
            
            if not redis_url:
                # Use a default local Redis URL if none is provided
                redis_url = "redis://localhost:6379/0"
                logger.info(f"Using default Redis URL: {redis_url}")
            elif not redis_url.startswith(('redis://', 'rediss://', 'unix://')):
                redis_url = f"redis://{redis_url}"
                logger.info(f"Added 'redis://' prefix to Redis URL")
                
            self.redis_client = redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=False,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30,
                encoding_errors="replace"  # Handle encoding errors gracefully
            )
            logger.info("Kết nối Redis thành công.")
        except Exception as e:
            logger.error(f"Lỗi kết nối Redis: {str(e)}")
            self.redis_client = None
            logger.warning("Continuing without Redis. Some features may not work correctly.")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def set(self, key, value, expire):
        try:
            if self.redis_client is None:
                logger.debug(f"Redis not available, skipping set for key: {key}")
                return False
                
            serialized_value = pickle.dumps(value)
            await self.redis_client.set(key, serialized_value, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis set: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get(self, key):
        try:
            if self.redis_client is None:
                logger.debug(f"Redis not available, skipping get for key: {key}")
                return None
                
            data = await self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Lỗi Redis get: {str(e)}")
            return None
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def delete(self, key):
        """Xóa một key khỏi cache"""
        try:
            if self.redis_client is None:
                logger.debug(f"Redis not available, skipping delete for key: {key}")
                return False
                
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis delete: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def invalidate_by_pattern(self, pattern):
        """Xóa tất cả các keys khớp với pattern"""
        try:
            if self.redis_client is None:
                logger.debug(f"Redis not available, skipping invalidate for pattern: {pattern}")
                return False
                
            cursor = b'0'
            while cursor:
                cursor, keys = await self.redis_client.scan(cursor=cursor, match=pattern, count=100)
                if keys:
                    await self.redis_client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis invalidate pattern: {str(e)}")
            return False

redis_manager = RedisManager()

# ---------- KẾT NỐI POSTGRESQL (Async) ----------
Base = declarative_base()

class ApprovedUser(Base):
    __tablename__ = 'approved_users'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, nullable=False)
    approved_at = Column(DateTime, default=datetime.now)

class ReportHistory(Base):
    __tablename__ = 'report_history'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(String, nullable=False)
    report = Column(Text, nullable=False)
    close_today = Column(Float, nullable=False)
    close_yesterday = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    
    # Thêm indexes cho tìm kiếm nhanh
    __table_args__ = (
        Index('idx_report_history_symbol', 'symbol'),
        Index('idx_report_history_date', 'date'),
        Index('idx_report_history_symbol_date', 'symbol', 'date'),
    )

class TrainedModel(Base):
    __tablename__ = 'trained_models'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)   # 'prophet' hoặc 'xgboost'
    model_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    performance = Column(Float, nullable=True)    # Thêm để lưu hiệu suất mô hình
    
    # Thêm indexes cho tìm kiếm nhanh
    __table_args__ = (
        Index('idx_trained_models_symbol', 'symbol'),
        Index('idx_trained_models_model_type', 'model_type'),
        Index('idx_trained_models_symbol_model_type', 'symbol', 'model_type'),
    )

# Tối ưu kết nối database với connection pooling
if not DATABASE_URL:
    DATABASE_URL = "sqlite+aiosqlite:///./stock_bot.db"
    logger.info(f"Using default SQLite database: {DATABASE_URL}")

engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,  # Recycle connections every 30 minutes
    pool_pre_ping=True  # Verify connections before using
)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        logger.warning("Continuing without database initialization. Some features may not work correctly.")

class DBManager:
    def __init__(self):
        self.Session = SessionLocal
        self.connection_ok = True
        
        # Test the database connection
        async def test_connection():
            try:
                async with self.Session() as session:
                    await session.execute(select(1))
                return True
            except Exception as e:
                logger.error(f"Database connection test failed: {str(e)}")
                return False
        
        # Create an event loop if not already running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to schedule the test
                asyncio.create_task(self._set_connection_status())
            else:
                # If loop is not running yet, run the test directly
                self.connection_ok = loop.run_until_complete(test_connection())
        except Exception as e:
            logger.error(f"Error testing database connection: {str(e)}")
            self.connection_ok = False
    
    async def _set_connection_status(self):
        try:
            async with self.Session() as session:
                await session.execute(select(1))
            self.connection_ok = True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            self.connection_ok = False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def is_user_approved(self, user_id) -> bool:
        # Always approve the admin
        if str(user_id) == ADMIN_ID:
            return True
            
        # If the database connection is not okay, we can't check
        if not self.connection_ok:
            logger.warning(f"Database connection not available, can't verify user {user_id}")
            # Default to not approved when DB is unavailable
            return False
            
        try:
            async with self.Session() as session:
                result = await session.execute(select(ApprovedUser).filter_by(user_id=str(user_id)))
                return result.scalar_one_or_none() is not None
        except Exception as e:
            logger.error(f"Lỗi kiểm tra người dùng: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def add_approved_user(self, user_id, approved_at=None) -> None:
        try:
            async with self.Session() as session:
                # Kiểm tra người dùng đã tồn tại chưa bằng cách sử dụng UPSERT
                stmt = select(ApprovedUser).filter_by(user_id=str(user_id))
                result = await session.execute(stmt)
                existing_user = result.scalar_one_or_none()
                
                if not existing_user and str(user_id) != ADMIN_ID:
                    new_user = ApprovedUser(user_id=str(user_id), approved_at=approved_at or datetime.now())
                    session.add(new_user)
                    await session.commit()
                    logger.info(f"Thêm người dùng được phê duyệt: {user_id}")
        except Exception as e:
            logger.error(f"Lỗi thêm người dùng: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_report_history(self, symbol: str) -> list:
        try:
            async with self.Session() as session:
                reports = await session.execute(
                    select(ReportHistory)
                    .filter_by(symbol=symbol)
                    .order_by(ReportHistory.date.desc())
                    .limit(10)  # Chỉ lấy 10 báo cáo gần nhất để tối ưu
                )
                reports = reports.scalars().all()
                return [
                    {
                        "id": report.id,
                        "symbol": report.symbol,
                        "date": report.date,
                        "report": report.report,
                        "close_today": report.close_today,
                        "close_yesterday": report.close_yesterday,
                        "timestamp": report.timestamp.isoformat()
                    }
                    for report in reports
                ]
        except Exception as e:
            logger.error(f"Lỗi tải lịch sử báo cáo: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def save_report_history(self, symbol: str, report: str, close_today: float, close_yesterday: float) -> None:
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Sanitize report text to ensure it's properly encoded
            sanitized_report = report
            if isinstance(report, str):
                # Handle any potential encoding issues with Vietnamese text
                sanitized_report = report.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            
            async with self.Session() as session:
                # Kiểm tra xem đã có báo cáo cho ngày hôm nay chưa
                existing_report = await session.execute(
                    select(ReportHistory)
                    .filter_by(symbol=symbol, date=today)
                )
                existing_report = existing_report.scalar_one_or_none()
                
                if existing_report:
                    # Cập nhật báo cáo hiện tại
                    existing_report.report = sanitized_report
                    existing_report.close_today = close_today
                    existing_report.close_yesterday = close_yesterday
                    existing_report.timestamp = datetime.now()
                else:
                    # Tạo báo cáo mới
                    new_report = ReportHistory(
                        symbol=symbol,
                        date=today,
                        report=sanitized_report,
                        close_today=close_today,
                        close_yesterday=close_yesterday
                    )
                    session.add(new_report)
                
                await session.commit()
                logger.info(f"Đã lưu báo cáo cho {symbol} ngày {today}")
                
                # Xóa cache để làm mới dữ liệu
                cache_key = make_cache_key(f"report_history:{symbol}", {})
                await redis_manager.delete(cache_key)
                
        except Exception as e:
            logger.error(f"Lỗi lưu lịch sử báo cáo: {str(e)}")
            raise

db = DBManager()

# ---------- QUẢN LÝ MÔ HÌNH (Prophet & XGBoost) ----------
class ModelDBManager:
    def __init__(self):
        self.Session = SessionLocal
        # Share the connection status with DBManager
        # since both use the same database connection
        self.connection_ok = db_manager.connection_ok if 'db_manager' in globals() else True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def store_trained_model(self, symbol: str, model_type: str, model, performance: float = None):
        try:
            # Serialize model
            model_bytes = await run_in_thread(pickle.dumps, model)
            
            async with self.Session() as session:
                # Check if model already exists
                existing_model = await session.execute(
                    select(TrainedModel)
                    .filter_by(symbol=symbol, model_type=model_type)
                )
                existing_model = existing_model.scalar_one_or_none()
                
                if existing_model:
                    # Update existing model
                    existing_model.model_blob = model_bytes
                    existing_model.performance = performance
                    existing_model.created_at = datetime.now()
                else:
                    # Create new model
                    new_model = TrainedModel(
                        symbol=symbol,
                        model_type=model_type,
                        model_blob=model_bytes,
                        performance=performance,
                        created_at=datetime.now()
                    )
                    session.add(new_model)
                
                await session.commit()
                
                # Invalidate cache
                cache_key = make_cache_key(f"model:{symbol}:{model_type}", {})
                await redis_manager.delete(cache_key)
                
                logger.info(f"Lưu mô hình {model_type} cho {symbol} thành công. Hiệu suất: {performance}")
                return True
        except Exception as e:
            logger.error(f"Lỗi lưu mô hình {model_type} cho {symbol}: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_trained_model(self, symbol: str, model_type: str):
        # Try cache first
        cache_key = make_cache_key(f"model:{symbol}:{model_type}", {})
        cached_model = await redis_manager.get(cache_key)
        if cached_model:
            logger.info(f"Tải mô hình {model_type} cho {symbol} từ cache")
            return cached_model
        
        try:
            async with self.Session() as session:
                # Get the latest model
                model_record = await session.execute(
                    select(TrainedModel)
                    .filter_by(symbol=symbol, model_type=model_type)
                    .order_by(TrainedModel.created_at.desc())
                    .limit(1)
                )
                model_record = model_record.scalar_one_or_none()
                
                if model_record:
                    model = await run_in_thread(pickle.loads, model_record.model_blob)
                    
                    # Cache the model
                    await redis_manager.set(cache_key, model, CACHE_EXPIRE_LONG)
                    
                    logger.info(f"Tải mô hình {model_type} cho {symbol} thành công. Hiệu suất: {model_record.performance}")
                    return model
                return None
        except Exception as e:
            logger.error(f"Lỗi tải mô hình {model_type} cho {symbol}: {str(e)}")
            return None

model_db_manager = ModelDBManager()

# ---------- HÀM HỖ TRỢ ----------
def is_index(symbol: str) -> bool:
    """Kiểm tra xem ký hiệu có phải là chỉ số"""
    return symbol.startswith('^') or symbol.upper() in ['VN30', 'VN100', 'VNINDEX', 'HNX', 'UPCOM']

async def is_user_approved(user_id) -> bool:
    """Wrapper cho db_manager.is_user_approved"""
    return await db.is_user_approved(user_id)

def standardize_data_for_db(data: dict) -> dict:
    """Chuẩn hóa dữ liệu trước khi lưu vào DB"""
    result = {}
    for key, value in data.items():
        # Chuyển đổi số thành float nếu có thể
        if isinstance(value, (int, float)) or (isinstance(value, str) and value.replace('.', '', 1).isdigit()):
            try:
                result[key] = float(value)
            except (ValueError, TypeError):
                result[key] = value
        # Chuyển đổi dict thành json string
        elif isinstance(value, dict):
            result[key] = json.dumps(value)
        # Giữ nguyên các loại khác
        else:
            result[key] = value
    return result

# ---------- DATA VALIDATION & NORMALIZATION ----------
class DataValidator:
    """Class xử lý validation và chuẩn hóa dữ liệu"""
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> (pd.DataFrame, dict):
        """
        Validate và chuẩn hóa dữ liệu đầu vào, trả về DataFrame đã xử lý và báo cáo lỗi
        """
        if df.empty:
            return df, {"status": "error", "message": "Dữ liệu trống"}
            
        # Tạo bản sao để tránh sửa đổi dữ liệu gốc
        df_clean = df.copy()
        validation_report = {"status": "success", "warnings": [], "fixes": []}
        
        # Kiểm tra và xử lý các cột bắt buộc
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]
        
        if missing_columns:
            validation_report["status"] = "error"
            validation_report["message"] = f"Thiếu các cột bắt buộc: {', '.join(missing_columns)}"
            return df_clean, validation_report
        
        # Đảm bảo index là datetime
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            if 'date' in df_clean.columns:
                try:
                    df_clean['date'] = pd.to_datetime(df_clean['date'])
                    df_clean.set_index('date', inplace=True)
                    validation_report["fixes"].append("Chuyển đổi cột 'date' thành index datetime")
                except Exception as e:
                    validation_report["status"] = "error"
                    validation_report["message"] = f"Không thể chuyển đổi cột 'date' thành datetime: {str(e)}"
                    return df_clean, validation_report
            else:
                try:
                    df_clean.index = pd.to_datetime(df_clean.index)
                    validation_report["fixes"].append("Chuyển đổi index thành datetime")
                except Exception as e:
                    validation_report["status"] = "error"
                    validation_report["message"] = f"Không thể chuyển đổi index thành datetime: {str(e)}"
                    return df_clean, validation_report
        
        # Kiểm tra giá trị âm trong các cột giá và volume
        for col in required_columns:
            if (df_clean[col] < 0).any():
                count_neg = (df_clean[col] < 0).sum()
                validation_report["warnings"].append(f"Tìm thấy {count_neg} giá trị âm trong cột {col}")
                # Thay thế giá trị âm bằng NaN
                df_clean.loc[df_clean[col] < 0, col] = np.nan
                validation_report["fixes"].append(f"Đã thay thế giá trị âm trong cột {col} bằng NaN")
        
        # Kiểm tra giá trị NaN
        na_count = df_clean[required_columns].isna().sum()
        if na_count.sum() > 0:
            for col in required_columns:
                if na_count[col] > 0:
                    validation_report["warnings"].append(f"Tìm thấy {na_count[col]} giá trị NaN trong cột {col}")
            
            # Điền giá trị NaN
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
            validation_report["fixes"].append("Đã điền các giá trị NaN bằng phương pháp forward fill và backward fill")
        
        # Kiểm tra logic giá OHLC
        invalid_candles = (
            (df_clean['high'] < df_clean['low']) | 
            (df_clean['open'] > df_clean['high']) | 
            (df_clean['open'] < df_clean['low']) | 
            (df_clean['close'] > df_clean['high']) | 
            (df_clean['close'] < df_clean['low'])
        )
        
        invalid_count = invalid_candles.sum()
        if invalid_count > 0:
            validation_report["warnings"].append(f"Tìm thấy {invalid_count} nến không hợp lệ (vi phạm logic OHLC)")
            # Sửa các nến không hợp lệ
            for idx in df_clean[invalid_candles].index:
                row = df_clean.loc[idx]
                # Điều chỉnh high và low
                new_high = max(row['open'], row['close'], row['high'])
                new_low = min(row['open'], row['close'], row['low'])
                df_clean.loc[idx, 'high'] = new_high
                df_clean.loc[idx, 'low'] = new_low
            validation_report["fixes"].append("Đã sửa các nến không hợp lệ để tuân thủ logic OHLC")
        
        # Sắp xếp theo thời gian
        df_clean = df_clean.sort_index()
        
        return df_clean, validation_report
    
    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> (pd.DataFrame, int):
        """Loại bỏ các bản ghi trùng lặp, trả về DataFrame đã xử lý và số lượng bản ghi đã xóa"""
        if df.empty:
            return df, 0
            
        # Đếm số lượng bản ghi trước khi xử lý
        initial_count = len(df)
        
        # Loại bỏ trùng lặp theo index
        df = df[~df.index.duplicated(keep='last')]
        
        # Tính số lượng bản ghi đã loại bỏ
        removed_count = initial_count - len(df)
        
        return df, removed_count
    
    @staticmethod
    def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa dữ liệu để sử dụng trong các mô hình ML"""
        if df.empty:
            return df
            
        df_norm = df.copy()
        
        # Chuẩn hóa khối lượng giao dịch
        if 'volume' in df_norm.columns:
            # Log transform để giảm ảnh hưởng của các giá trị cực lớn
            df_norm['volume_log'] = np.log1p(df_norm['volume'])
            
            # Min-Max scaling cho volume_log
            min_vol = df_norm['volume_log'].min()
            max_vol = df_norm['volume_log'].max()
            if max_vol > min_vol:  # Tránh chia cho 0
                df_norm['volume_norm'] = (df_norm['volume_log'] - min_vol) / (max_vol - min_vol)
            else:
                df_norm['volume_norm'] = 0
        
        # Chuẩn hóa giá theo phần trăm thay đổi
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df_norm.columns:
                df_norm[f'{col}_pct_change'] = df_norm[col].pct_change()
                df_norm[f'{col}_pct_change'].fillna(0, inplace=True)
        
        # Chuẩn hóa giá theo Z-score
        for col in price_cols:
            if col in df_norm.columns:
                mean = df_norm[col].mean()
                std = df_norm[col].std()
                if std > 0:  # Tránh chia cho 0
                    df_norm[f'{col}_zscore'] = (df_norm[col] - mean) / std
                else:
                    df_norm[f'{col}_zscore'] = 0
        
        return df_norm

# ---------- HÀM HỖ TRỢ: LỌC NGÀY GIAO DỊCH -----------
def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Lọc các ngày giao dịch, loại bỏ cuối tuần và ngày lễ Việt Nam"""
    if df.empty:
        return df
        
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
    df = df.copy()
    
    # Chuyển index thành datetime nếu chưa
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    
    # Lọc ra ngày trong tuần (1-5: Thứ Hai - Thứ Sáu)
    df = df[df.index.dayofweek < 5]
    
    # Lọc bỏ các ngày nghỉ lễ Việt Nam sử dụng hàm có sẵn
    years = df.index.year.unique().tolist()
    
    # Dùng thư viện holidays trực tiếp để tránh tạo thêm instance
    vietnam_holidays = holidays.VN(years=years)
    
    # Tạo mặt nạ để lọc các ngày không phải ngày lễ
    holiday_mask = ~df.index.isin(vietnam_holidays)
    df = df[holiday_mask]
    
    return df

# ---------- TẢI DỮ LIỆU (NÂNG CẤP) ----------
class DataLoader:
    def __init__(self, source: str = 'vnstock'):
        self.source = source
        self.vnstock = Vnstock() if source == 'vnstock' else None
        
        # Cache cho holiday calendar
        self._vietnam_holidays = None
        
    def _get_vietnam_holidays(self, years=None):
        """Cache holiday calendar để tối ưu hiệu suất"""
        if years is None:
            # Lấy năm hiện tại và năm tiếp theo
            current_year = datetime.now().year
            years = [current_year - 1, current_year, current_year + 1]
            
        if self._vietnam_holidays is None:
            self._vietnam_holidays = holidays.VN(years=years)
        return self._vietnam_holidays

    @measure_execution_time
    def detect_outliers(self, df: pd.DataFrame) -> (pd.DataFrame, str):
        """Phát hiện điểm bất thường trong dữ liệu giá đóng cửa bằng nhiều phương pháp"""
        if df.empty or len(df) < 2:
            return df, "Không đủ dữ liệu để phát hiện bất thường"
            
        # Tạo bản sao
        df_result = df.copy()
        outlier_methods = {}
        
        # 1. Phương pháp Z-score
        def detect_zscore_outliers(series, threshold=3.0):
            mean_val = series.mean()
            std_val = series.std()
            if std_val == 0:  # Tránh chia cho 0
                return pd.Series(False, index=series.index)
            z_scores = (series - mean_val) / std_val
            return np.abs(z_scores) > threshold
        
        # 2. Phương pháp IQR (Interquartile Range)
        def detect_iqr_outliers(series, k=1.5):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:  # Tránh chia cho 0
                return pd.Series(False, index=series.index)
            lower_bound = q1 - k * iqr
            upper_bound = q3 + k * iqr
            return (series < lower_bound) | (series > upper_bound)
        
        # 3. Phương pháp biến động giá bất thường
        def detect_price_change_outliers(series, threshold=0.1):
            pct_change = series.pct_change().abs()
            return pct_change > threshold
        
        # 4. Phương pháp Local Outlier Factor (LOF) với moving window
        def detect_lof_outliers(series, window=20, n_neighbors=5):
            result = pd.Series(False, index=series.index)
            if len(series) < window:
                return result
                
            from sklearn.neighbors import LocalOutlierFactor
            
            # Áp dụng LOF trên window di chuyển để phát hiện outlier cục bộ
            for i in range(window, len(series) + 1):
                window_data = series.iloc[i-window:i].values.reshape(-1, 1)
                try:
                    lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, window-1))
                    outlier_labels = lof.fit_predict(window_data)
                    # LOF returns -1 for outliers and 1 for inliers
                    outliers = outlier_labels == -1
                    if any(outliers):
                        result.iloc[i-window:i][outliers] = True
                except Exception:
                    # Bỏ qua nếu có lỗi (ví dụ: không đủ mẫu)
                    pass
            return result
        
        # Áp dụng các phương pháp
        price_series = df_result['close']
        
        # Tạo cột Z-score
        df_result['z_score'] = (price_series - price_series.mean()) / price_series.std() if price_series.std() > 0 else 0
        
        # Áp dụng từng phương pháp và lưu kết quả
        outlier_methods['z_score'] = detect_zscore_outliers(price_series)
        outlier_methods['iqr'] = detect_iqr_outliers(price_series)
        outlier_methods['price_change'] = detect_price_change_outliers(price_series)
        
        # Chỉ áp dụng LOF nếu có đủ dữ liệu
        if len(df_result) >= 20:
            try:
                outlier_methods['lof'] = detect_lof_outliers(price_series)
            except Exception as e:
                logger.warning(f"Không thể áp dụng LOF: {str(e)}")
                outlier_methods['lof'] = pd.Series(False, index=price_series.index)
        else:
            outlier_methods['lof'] = pd.Series(False, index=price_series.index)
        
        # Kết hợp kết quả từ tất cả các phương pháp (xem là outlier nếu ít nhất 2 phương pháp phát hiện)
        df_result['outlier_count'] = sum(outlier_methods.values())
        df_result['is_outlier'] = df_result['outlier_count'] >= 2
        
        # Thêm thông tin outlier từ từng phương pháp
        for method, outliers in outlier_methods.items():
            df_result[f'outlier_{method}'] = outliers
        
        # Tạo báo cáo chi tiết
        outliers = df_result[df_result['is_outlier']]
        if len(outliers) > 0:
            outlier_report = f"Phát hiện {len(outliers)} điểm bất thường:\n"
            for idx, row in outliers.iterrows():
                date_str = idx.strftime('%Y-%m-%d') if isinstance(idx, pd.Timestamp) else str(idx)
                methods_detected = []
                for method, outliers_series in outlier_methods.items():
                    if idx in outliers_series.index and outliers_series.loc[idx]:
                        methods_detected.append(method)
                outlier_report += (
                    f"- Ngày {date_str}: Giá {row['close']} (Z-score: {row['z_score']:.2f}, "
                    f"Phát hiện bởi: {', '.join(methods_detected)})\n"
                )
        else:
            outlier_report = "Không phát hiện điểm bất thường trong dữ liệu."
            
        return df_result, outlier_report
    
    @measure_execution_time
    async def load_data(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME, num_candles: int = DEFAULT_CANDLES) -> (pd.DataFrame, str):
        """Tải dữ liệu chứng khoán với cache thông minh"""
        # Standardize timeframe
        timeframe = timeframe.upper()
        
        # Tạo cache key
        cache_params = {
            'symbol': symbol,
            'timeframe': timeframe,
            'num_candles': num_candles
        }
        cache_key = make_cache_key('stock_data', cache_params)
        
        # Check cache
        cached_data = await redis_manager.get(cache_key)
        if cached_data is not None:
            logger.info(f"Tải dữ liệu {symbol} ({timeframe}) từ cache")
            df, outlier_report = cached_data
            return df, outlier_report
        
        outlier_report = ""
        try:
            # Load data from source
            if is_index(symbol) or self.source == 'yahoo':
                # Sử dụng Yahoo Finance cho chỉ số và khi source là yahoo
                period_mapping = {
                    '1D': '1d',
                    '1W': '1wk',
                    '1MO': '1mo'
                }
                period = period_mapping.get(timeframe, '1d')
                interval_mapping = {
                    '1D': '1d',
                    '1W': '1wk', 
                    '1MO': '1mo'
                }
                interval = interval_mapping.get(timeframe, '1d')
                
                # Điều chỉnh symbol cho Yahoo Finance
                yahoo_symbol = symbol
                if symbol.upper() == 'VNINDEX':
                    yahoo_symbol = '^VNINDEX'
                elif symbol.upper() == 'VN30':
                    yahoo_symbol = '^VN30'
                elif symbol.upper() == 'HNX':
                    yahoo_symbol = '^HNX'
                elif not symbol.startswith('^'):
                    yahoo_symbol = f"{symbol}.VN"
                
                df = await self._download_yahoo_data(yahoo_symbol, num_candles, period, interval)
            else:
                # Sử dụng VNStock cho cổ phiếu Việt Nam
                async def fetch_vnstock():
                    if timeframe == '1D':
                        # Import trực tiếp từ vnstock module
                        from vnstock import stock_historical_data
                        
                        return stock_historical_data(
                            symbol=symbol, 
                            start_date=(datetime.now() - timedelta(days=num_candles*2)).strftime('%Y-%m-%d'),
                            end_date=datetime.now().strftime('%Y-%m-%d')
                        )
                    else:
                        raise ValueError(f"VNStock không hỗ trợ khung thời gian {timeframe}")
                
                df = await run_in_thread(fetch_vnstock)
                
                # Preprocessing for VNStock data
                if not df.empty:
                    df.columns = df.columns.str.lower()
                    # Ensure correct column names
                    column_mapping = {
                        'time': 'date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume'
                    }
                    df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})
                    
                    # Ensure date column is datetime
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    
                    # Take only the last num_candles
                    df = df.iloc[-num_candles:]
            
            # Process the data
            if not df.empty:
                # Filter trading days
                df = filter_trading_days(df)
                
                # Sort by date
                df = df.sort_index()
                
                # Detect outliers
                df, outlier_report = self.detect_outliers(df)
                
                # Handle NaN values
                df = df.fillna(method='ffill').fillna(method='bfill')
                
                # Cache the result
                cache_expiry = CACHE_EXPIRE_SHORT if timeframe == '1D' else CACHE_EXPIRE_MEDIUM
                await redis_manager.set(cache_key, (df, outlier_report), cache_expiry)
                
                return df, outlier_report
            else:
                return pd.DataFrame(), "Không có dữ liệu"
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu {symbol} ({timeframe}): {str(e)}")
            return pd.DataFrame(), f"Lỗi: {str(e)}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
    async def _download_yahoo_data(self, symbol: str, num_candles: int, period: str, interval: str) -> pd.DataFrame:
        """Tải dữ liệu từ Yahoo Finance với xử lý lỗi tốt hơn"""
        try:
            # Tính toán khoảng thời gian dựa trên số lượng nến và period
            days_multiplier = {
                '1d': 1,
                '1wk': 7,
                '1mo': 30
            }
            
            # Thêm thời gian để đảm bảo có đủ dữ liệu sau khi lọc
            buffer_multiplier = 1.5
            days_to_fetch = int(num_candles * days_multiplier.get(period, 1) * buffer_multiplier)
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_to_fetch)
            
            # Format dates for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data from Yahoo Finance
            def fetch_yf():
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_str, end=end_str, interval=interval)
                return df
            
            df = await run_in_thread(fetch_yf)
            
            if df.empty:
                logger.warning(f"Yahoo Finance không trả về dữ liệu cho {symbol}")
                return pd.DataFrame()
            
            # Clean and standardize columns
            df.columns = df.columns.str.lower()
            # Keep only OHLCV columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Kiểm tra trường hợp chỉ số (có thể không có volume)
            if is_index(symbol) and 'volume' not in df.columns:
                # Đối với chỉ số, không bắt buộc phải có volume
                required_cols = [col for col in required_cols if col != 'volume']
                logger.info(f"Bỏ qua cột volume cho chỉ số {symbol}")
                # Thêm cột volume giả để đảm bảo tính nhất quán
                df['volume'] = 0
            
            df = df[[col for col in required_cols if col in df.columns]]
            
            # Take only the last num_candles
            df = df.tail(num_candles)
            
            return df
            
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu Yahoo Finance cho {symbol}: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def fetch_fundamental_data_vnstock(self, symbol: str) -> dict:
        """Tải dữ liệu cơ bản từ VNStock"""
        # Check cache
        cache_key = make_cache_key(f"fundamental:vnstock:{symbol}", {})
        cached_data = await redis_manager.get(cache_key)
        if cached_data is not None:
            logger.info(f"Tải dữ liệu cơ bản của {symbol} từ cache")
            return cached_data
        
        # Fetch data
        def fetch():
            result = {}
            try:
                if not is_index(symbol):
                    # Import các hàm trực tiếp từ vnstock
                    from vnstock import company_overview, financial_ratio, income_statement, balance_sheet, cash_flow
                    
                    # Company overview
                    try:
                        overview = company_overview(symbol=symbol)
                        # Sanitize data to ensure it's JSON-serializable and handle encoding issues
                        overview_sanitized = {}
                        for key, value in overview.items():
                            if isinstance(value, str):
                                # Handle any potential encoding issues with Vietnamese text
                                overview_sanitized[key] = value.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                            else:
                                overview_sanitized[key] = value
                        result['overview'] = overview_sanitized
                    except Exception as e:
                        logger.error(f"Lỗi lấy company_overview cho {symbol}: {str(e)}")
                        result['overview'] = {'symbol': symbol, 'error': 'Failed to retrieve company overview'}
                    
                    # Financial ratios
                    try:
                        ratios = financial_ratio(symbol=symbol, report_type='yearly', report_range=3)
                        result['ratios'] = ratios
                    except Exception as e:
                        logger.error(f"Lỗi lấy financial_ratio cho {symbol}: {str(e)}")
                        result['ratios'] = None
                    
                    # Financial reports
                    try:
                        result['income_statement'] = income_statement(symbol=symbol, report_type='yearly', report_range=3)
                    except Exception as e:
                        logger.error(f"Lỗi lấy income_statement cho {symbol}: {str(e)}")
                        result['income_statement'] = None
                        
                    try:
                        result['balance_sheet'] = balance_sheet(symbol=symbol, report_type='yearly', report_range=3)
                    except Exception as e:
                        logger.error(f"Lỗi lấy balance_sheet cho {symbol}: {str(e)}")
                        result['balance_sheet'] = None
                        
                    try:
                        result['cash_flow'] = cash_flow(symbol=symbol, report_type='yearly', report_range=3)
                    except Exception as e:
                        logger.error(f"Lỗi lấy cash_flow cho {symbol}: {str(e)}")
                        result['cash_flow'] = None
                else:
                    # For indices
                    result['overview'] = {'symbol': symbol, 'type': 'index'}
            except Exception as e:
                logger.error(f"Lỗi lấy dữ liệu cơ bản VNStock cho {symbol}: {str(e)}")
            
            return result
        
        fundamental_data = await run_in_thread(fetch)
        
        # Cache the result
        await redis_manager.set(cache_key, fundamental_data, CACHE_EXPIRE_LONG)
        
        return fundamental_data
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def fetch_fundamental_data_yahoo(self, symbol: str) -> dict:
        """Tải dữ liệu cơ bản từ Yahoo Finance"""
        # Check cache
        cache_key = make_cache_key(f"fundamental:yahoo:{symbol}", {})
        cached_data = await redis_manager.get(cache_key)
        if cached_data is not None:
            logger.info(f"Tải dữ liệu cơ bản của {symbol} từ cache")
            return cached_data
        
        def fetch():
            result = {}
            try:
                # Điều chỉnh symbol cho Yahoo Finance
                yahoo_symbol = symbol
                if symbol.upper() == 'VNINDEX':
                    yahoo_symbol = '^VNINDEX'
                elif symbol.upper() == 'VN30':
                    yahoo_symbol = '^VN30'
                elif symbol.upper() == 'HNX':
                    yahoo_symbol = '^HNX'
                elif not symbol.startswith('^'):
                    yahoo_symbol = f"{symbol}.VN"
                
                ticker = yf.Ticker(yahoo_symbol)
                
                if is_index(symbol):
                    result = {'info': {'shortName': symbol, 'type': 'index'}}
                else:
                    # Basic info
                    result['info'] = ticker.info
                    
                    # Financials
                    if 'financials' not in result:
                        result['financials'] = {}
                    
                    # Balance sheet
                    try:
                        result['financials']['balance_sheet'] = ticker.balance_sheet
                    except:
                        result['financials']['balance_sheet'] = None
                    
                    # Income statement
                    try:
                        result['financials']['income_stmt'] = ticker.income_stmt
                    except:
                        result['financials']['income_stmt'] = None
                    
                    # Cash flow
                    try:
                        result['financials']['cash_flow'] = ticker.cashflow
                    except:
                        result['financials']['cash_flow'] = None
            except Exception as e:
                logger.error(f"Lỗi lấy dữ liệu cơ bản Yahoo Finance cho {symbol}: {str(e)}")
                result = {'error': str(e)}
            
            return result
                
        fundamental_data = await run_in_thread(fetch)
        
        # Cache the result
        await redis_manager.set(cache_key, fundamental_data, CACHE_EXPIRE_LONG)
        
        return fundamental_data
    
    async def get_fundamental_data(self, symbol: str) -> dict:
        """Tải dữ liệu cơ bản của mã chứng khoán"""
        try:
            # Ưu tiên tải từ VNStock cho cổ phiếu Việt Nam
            if not is_index(symbol) and self.source == 'vnstock':
                data = await self.fetch_fundamental_data_vnstock(symbol)
            else:
                data = await self.fetch_fundamental_data_yahoo(symbol)
            
            return data
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu cơ bản cho {symbol}: {str(e)}")
            return {}

# ---------- PIPELINE XỬ LÝ DỮ LIỆU ----------
class DataPipeline:
    """Class quản lý quy trình xử lý dữ liệu hoàn chỉnh"""
    
    def __init__(self, source: str = 'vnstock'):
        self.data_loader = DataLoader(source)
        self.validator = DataValidator()
        self.outlier_reports = {}
        self.validation_reports = {}
    
    @measure_execution_time
    async def process_data(self, symbol: str, timeframe: str = DEFAULT_TIMEFRAME, num_candles: int = DEFAULT_CANDLES) -> (pd.DataFrame, dict):
        """
        Quy trình xử lý dữ liệu hoàn chỉnh: tải, validate, phát hiện outlier và chuẩn hóa
        Trả về DataFrame đã xử lý và báo cáo xử lý
        """
        # 1. Tải dữ liệu
        df, outlier_report = await self.data_loader.load_data(symbol, timeframe, num_candles)
        
        # Lưu báo cáo outlier
        self.outlier_reports[symbol] = outlier_report
        
        if df.empty:
            return df, {"status": "error", "message": "Không thể tải dữ liệu"}
        
        # 2. Validate và loại bỏ trùng lặp
        df_validated, validation_report = self.validator.validate_data(df)
        df_validated, removed_count = self.validator.remove_duplicates(df_validated)
        
        # Cập nhật báo cáo về số lượng bản ghi đã loại bỏ
        if removed_count > 0:
            validation_report["fixes"].append(f"Đã loại bỏ {removed_count} bản ghi trùng lặp")
        
        # Lưu báo cáo validation
        self.validation_reports[symbol] = validation_report
        
        if validation_report["status"] == "error":
            return df_validated, validation_report
        
        # 3. Lọc các ngày giao dịch
        df_validated = filter_trading_days(df_validated)
        
        # 4. Chuẩn hóa dữ liệu cho ML
        df_normalized = self.validator.normalize_data(df_validated)
        
        # 5. Đảm bảo sắp xếp theo thời gian
        df_normalized = df_normalized.sort_index()
        
        # Tổng hợp báo cáo xử lý
        processing_report = {
            "symbol": symbol,
            "timeframe": timeframe,
            "original_rows": len(df),
            "processed_rows": len(df_normalized),
            "validation": validation_report,
            "outliers": outlier_report,
            "status": "success"
        }
        
        return df_normalized, processing_report
    
    @measure_execution_time
    async def process_multi_timeframe(self, symbol: str, timeframes: list = None) -> (dict, dict):
        """
        Xử lý dữ liệu cho nhiều khung thời gian
        Trả về dict của DataFrames đã xử lý và báo cáo xử lý
        """
        if timeframes is None:
            timeframes = ['1D', '1W', '1MO']
        
        results = {}
        reports = {}
        
        # Xử lý từng khung thời gian
        for tf in timeframes:
            # Điều chỉnh số lượng nến tùy theo khung thời gian
            if tf == '1D':
                num_candles = DEFAULT_CANDLES
            elif tf == '1W':
                num_candles = DEFAULT_CANDLES // 5
            elif tf == '1MO':
                num_candles = DEFAULT_CANDLES // 20
            else:
                num_candles = DEFAULT_CANDLES
            
            df, report = await self.process_data(symbol, tf, num_candles)
            results[tf] = df
            reports[tf] = report
        
        return results, reports
    
    async def get_fundamental_and_technical_data(self, symbol: str) -> (dict, dict, dict):
        """
        Tải và xử lý cả dữ liệu kỹ thuật và cơ bản cho mã chứng khoán
        Trả về dữ liệu kỹ thuật đã xử lý, dữ liệu cơ bản và báo cáo xử lý
        """
        # Tải dữ liệu kỹ thuật đa khung thời gian
        dfs, reports = await self.process_multi_timeframe(symbol)
        
        # Tải dữ liệu cơ bản
        fundamental_data = await self.data_loader.get_fundamental_data(symbol)
        
        # Tạo báo cáo tổng hợp
        # Kiểm tra dữ liệu một cách an toàn
        has_data = False
        for df_key, df in dfs.items():
            # Nếu df là một coroutine, cần await nó trước
            if isawaitable(df):
                try:
                    df = await df
                    dfs[df_key] = df  # Cập nhật lại dictionary với giá trị đã await
                except Exception as e:
                    logger.error(f"Lỗi khi await coroutine cho {df_key}: {str(e)}")
                    continue
            
            # Kiểm tra DataFrame có dữ liệu
            if df is not None and hasattr(df, 'empty') and not df.empty:
                has_data = True
                break
                
        summary_report = {
            "symbol": symbol,
            "timeframes_processed": list(dfs.keys()),
            "fundamental_data_available": len(fundamental_data) > 0,
            "outlier_reports": self.outlier_reports.get(symbol, ""),
            "validation_reports": self.validation_reports.get(symbol, {}),
            "status": "success" if has_data else "error"
        }
        
        return dfs, fundamental_data, summary_report

# ---------- PHÂN TÍCH KỸ THUẬT ----------
class TechnicalAnalyzer:
    """Lớp phân tích kỹ thuật với cải tiến hiệu suất"""
    
    @staticmethod
    @lru_cache(maxsize=32)
    def _calculate_common_indicators(prices, volumes=None) -> dict:
        """Tính các chỉ báo kỹ thuật phổ biến với cache cho chuỗi giá"""
        result = {}
        
        # Mã hash để nhận dạng đầu vào
        price_hash = hashlib.md5(prices.tobytes()).hexdigest()
        cache_key = f"indicators:{price_hash}"
        
        # Moving Averages
        result['sma_20'] = trend.sma_indicator(prices, window=20, fillna=True)
        result['sma_50'] = trend.sma_indicator(prices, window=50, fillna=True)
        result['sma_200'] = trend.sma_indicator(prices, window=200, fillna=True)
        result['ema_12'] = trend.ema_indicator(prices, window=12, fillna=True)
        result['ema_26'] = trend.ema_indicator(prices, window=26, fillna=True)
        
        # MACD
        result['macd_line'] = trend.macd(prices, window_slow=26, window_fast=12, fillna=True)
        result['macd_signal'] = trend.macd_signal(prices, window_slow=26, window_fast=12, window_sign=9, fillna=True)
        result['macd_diff'] = trend.macd_diff(prices, window_slow=26, window_fast=12, window_sign=9, fillna=True)
        
        # RSI
        result['rsi'] = momentum.rsi(prices, window=14, fillna=True)
        
        # Bollinger Bands
        result['bb_high'] = volatility.bollinger_hband(prices, window=20, window_dev=2, fillna=True)
        result['bb_mid'] = volatility.bollinger_mavg(prices, window=20, fillna=True)
        result['bb_low'] = volatility.bollinger_lband(prices, window=20, window_dev=2, fillna=True)
        
        # Stochastic Oscillator
        result['stoch_k'] = momentum.stoch(prices, prices, prices, window=14, smooth_window=3, fillna=True)
        result['stoch_d'] = momentum.stoch_signal(prices, prices, prices, window=14, smooth_window=3, fillna=True)
        
        # ATR for volatility
        result['atr'] = volatility.average_true_range(prices, prices, prices, window=14, fillna=True)
        
        # Volume-based indicators if volume data is provided
        if volumes is not None:
            try:
                # MFI (Money Flow Index)
                mfi = MFIIndicator(prices, prices, prices, volumes, window=14, fillna=True)
                result['mfi'] = mfi.money_flow_index()
            except Exception as e:
                logger.warning(f"Lỗi tính MFI: {str(e)}")
        
        return result
    
    @measure_execution_time
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các chỉ báo kỹ thuật cho dataframe"""
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original dataframe
        result_df = df.copy()
        
        # Calculate technical indicators
        if 'close' in result_df.columns:
            indicators = self._calculate_common_indicators(
                result_df['close'].values, 
                result_df['volume'].values if 'volume' in result_df.columns else None
            )
            
            # Add indicators to dataframe
            for name, values in indicators.items():
                result_df[name] = values
            
            # Calculate additional derived signals
            result_df['golden_cross'] = (result_df['sma_50'] > result_df['sma_200']) & (result_df['sma_50'].shift(1) <= result_df['sma_200'].shift(1))
            result_df['death_cross'] = (result_df['sma_50'] < result_df['sma_200']) & (result_df['sma_50'].shift(1) >= result_df['sma_200'].shift(1))
            result_df['macd_cross_above'] = (result_df['macd_line'] > result_df['macd_signal']) & (result_df['macd_line'].shift(1) <= result_df['macd_signal'].shift(1))
            result_df['macd_cross_below'] = (result_df['macd_line'] < result_df['macd_signal']) & (result_df['macd_line'].shift(1) >= result_df['macd_signal'].shift(1))
            
            # RSI overbought/oversold signals
            result_df['rsi_overbought'] = result_df['rsi'] > 70
            result_df['rsi_oversold'] = result_df['rsi'] < 30
            
            # Bollinger Band signals
            result_df['bb_upper_breakout'] = result_df['close'] > result_df['bb_high']
            result_df['bb_lower_breakout'] = result_df['close'] < result_df['bb_low']
        
        return result_df
    
    @measure_execution_time
    def calculate_multi_timeframe_indicators(self, dfs: dict) -> dict:
        """Tính toán chỉ báo cho nhiều khung thời gian song song"""
        results = {}
        
        async def calculate_for_df(timeframe, df):
            if not df.empty:
                return timeframe, self.calculate_indicators(df)
            return timeframe, df
        
        # Sử dụng asyncio.gather để tính toán song song
        async def process_all():
            tasks = [calculate_for_df(timeframe, df) for timeframe, df in dfs.items()]
            result_list = await asyncio.gather(*tasks)
            return {timeframe: df for timeframe, df in result_list}
        
        # Chạy bất đồng bộ nhưng trong thread hiện tại
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(process_all())
        
        return results

# ---------- THU THẬP TIN TỨC (SỬA LỖI) ----------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def fetch_rss_feed(url: str) -> str:
    """Tải dữ liệu từ feed RSS cải tiến với async"""
    try:
        import aiohttp
        
        # Fast fail timeout
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Lỗi tải RSS từ {url}: HTTP {response.status}")
                    return ""
                return await response.text()
    except Exception as e:
        logger.error(f"Lỗi tải RSS từ {url}: {str(e)}")
        return ""

async def get_news(symbol: str = None, limit: int = 3) -> list:
    """Thu thập tin tức liên quan cải tiến với parallel fetching"""
    # Kiểm tra cache
    cache_key = make_cache_key(f"news:{symbol if symbol else 'general'}", {"limit": limit})
    cached_news = await redis_manager.get(cache_key)
    if cached_news:
        return cached_news[:limit]
    
    # Danh sách nguồn tin
    sources = [
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://vnexpress.net/rss/chung-khoan.rss",
        "https://vietnamnet.vn/rss/kinh-doanh.rss",
        "https://cafef.vn/feed/thi-truong-chung-khoan.rss",
        "https://cafef.vn/feed/doanh-nghiep.rss"
    ]
    
    # Thu thập tin tức song song từ các nguồn
    tasks = [fetch_rss_feed(source) for source in sources]
    feed_contents = await asyncio.gather(*tasks)
    
    # Lọc tin tức liên quan đến mã chứng khoán (nếu có)
    all_news = []
    for content in feed_contents:
        if content:
            try:
                news_items = parse_rss_content(content)
                # Nếu cần lọc theo mã chứng khoán
                if symbol:
                    news_items = [
                        item for item in news_items 
                        if symbol.upper() in item.get('title', '').upper() or 
                           symbol.upper() in item.get('description', '').upper()
                    ]
                all_news.extend(news_items)
            except Exception as e:
                logger.error(f"Lỗi xử lý nội dung RSS: {str(e)}")
    
    # Sắp xếp tin tức theo thời gian mới nhất
    all_news = sorted(all_news, key=lambda x: x.get('pubDate', ''), reverse=True)
    
    # Loại bỏ tin trùng lặp (dựa trên tiêu đề)
    unique_news = []
    seen_titles = set()
    for item in all_news:
        title = item.get('title', '')
        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_news.append(item)
    
    # Lưu cache
    await redis_manager.set(cache_key, unique_news, NEWS_CACHE_EXPIRE)
    
    # Trả về số lượng tin theo yêu cầu
    return unique_news[:limit]

def parse_rss_content(rss_text: str):
    """Phân tích nội dung RSS với xử lý lỗi cải tiến"""
    import html
    
    if not rss_text:
        return []
    
    try:
        feed = feedparser.parse(rss_text)
        
        if not feed or not feed.entries:
            return []
        
        items = []
        for entry in feed.entries:
            item = {
                'title': html.unescape(entry.get('title', '')),
                'link': entry.get('link', ''),
                'description': html.unescape(entry.get('description', '')),
                'pubDate': entry.get('published', entry.get('pubDate', ''))
            }
            
            # Xử lý mô tả HTML để lấy text
            from bs4 import BeautifulSoup
            if item['description']:
                try:
                    soup = BeautifulSoup(item['description'], 'html.parser')
                    item['description'] = soup.get_text().strip()
                except:
                    # Nếu không parse được, giữ nguyên
                    pass
            
            items.append(item)
        
        return items
    except Exception as e:
        logger.error(f"Lỗi xử lý RSS: {str(e)}")
        return []

# ---------- PHÂN TÍCH CƠ BẢN ----------
def deep_fundamental_analysis(fundamental_data: dict) -> str:
    """Phân tích sâu dữ liệu cơ bản được cải tiến"""
    if not fundamental_data or not isinstance(fundamental_data, dict):
        return "Không đủ dữ liệu để phân tích cơ bản."
    
    analysis = []
    
    # VNStock data analysis
    if 'overview' in fundamental_data:
        overview = fundamental_data.get('overview', {})
        if isinstance(overview, dict):
            company_name = overview.get('companyName', '')
            industry = overview.get('industryName', '')
            if company_name and industry:
                analysis.append(f"**Công ty:** {company_name} - **Ngành:** {industry}")
    
    # Extract key financial ratios
    all_ratios = {}
    
    # From VNStock
    if 'ratios' in fundamental_data and isinstance(fundamental_data['ratios'], pd.DataFrame):
        try:
            # Get the most recent year
            recent_ratios = fundamental_data['ratios'].iloc[-1].to_dict() if not fundamental_data['ratios'].empty else {}
            all_ratios.update(recent_ratios)
        except:
            pass
    
    # From Yahoo Finance
    if 'info' in fundamental_data and isinstance(fundamental_data['info'], dict):
        info = fundamental_data['info']
        
        # Add any key financial metrics from Yahoo
        yahoo_metrics = {
            'PE': info.get('trailingPE'),
            'ForwardPE': info.get('forwardPE'),
            'PB': info.get('priceToBook'),
            'ROE': info.get('returnOnEquity'),
            'ROA': info.get('returnOnAssets'),
            'DividendYield': info.get('dividendYield'),
            'EPS': info.get('trailingEps'),
            'ProfitMargin': info.get('profitMargins'),
            'DebtToEquity': info.get('debtToEquity')
        }
        
        # Only add non-None values
        all_ratios.update({k: v for k, v in yahoo_metrics.items() if v is not None})
    
    # Format key ratios for display
    if all_ratios:
        analysis.append("\n**Chỉ số tài chính chính:**")
        
        # Map metric names to display names
        metric_display = {
            'ROE': 'ROE',
            'ROA': 'ROA',
            'PE': 'P/E',
            'ForwardPE': 'Forward P/E',
            'PB': 'P/B',
            'EPS': 'EPS',
            'DividendYield': 'Tỷ suất cổ tức',
            'DebtToEquity': 'Nợ/Vốn chủ sở hữu',
            'ProfitMargin': 'Biên lợi nhuận'
        }
        
        for metric, display_name in metric_display.items():
            if metric in all_ratios and all_ratios[metric] is not None:
                value = all_ratios[metric]
                # Format percentages
                if metric in ['ROE', 'ROA', 'DividendYield', 'ProfitMargin']:
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value*100:.2f}%" if value < 1 else f"{value:.2f}%"
                    else:
                        formatted_value = str(value)
                else:
                    formatted_value = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                
                analysis.append(f"- **{display_name}:** {formatted_value}")
    
    if not analysis:
        return "Không tìm thấy dữ liệu tài chính cho phân tích cơ bản."
    
    return "\n".join(analysis)

# ---------- HUẤN LUYỆN VÀ LƯU MÔ HÌNH ----------
def prepare_data_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn bị dữ liệu cho Prophet cải tiến"""
    if df.empty:
        return pd.DataFrame()
    
    # Tạo bản sao để tránh SettingWithCopyWarning
    df_prophet = df.copy()
    
    # Đảm bảo df có cột close
    if 'close' not in df_prophet.columns:
        return pd.DataFrame()
    
    # Đặt lại index thành ngày nếu chưa
    if not isinstance(df_prophet.index, pd.DatetimeIndex):
        df_prophet.index = pd.to_datetime(df_prophet.index)
    
    # Tạo cột ds (ngày) và y (giá đóng cửa) cho Prophet
    df_prophet = df_prophet.reset_index()
    df_prophet.rename(columns={'index': 'ds', 'close': 'y'}, inplace=True)
    
    # Đảm bảo cột ds là datetime không có timezone
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
    
    # Sắp xếp theo ngày tăng dần
    df_prophet = df_prophet.sort_values('ds')
    
    return df_prophet[['ds', 'y']]

def get_vietnam_holidays(years) -> pd.DataFrame:
    """Lấy ngày nghỉ ở Việt Nam cho Prophet với cache"""
    if isinstance(years, int):
        years = [years]
    
    vn_holidays = holidays.Vietnam(years=years)
    
    # Convert to DataFrame for Prophet
    holiday_df = pd.DataFrame([
        {'holiday': name, 'ds': date, 'lower_window': 0, 'upper_window': 1}
        for date, name in vn_holidays.items()
    ])
    
    return holiday_df

@measure_execution_time
def forecast_with_prophet(df: pd.DataFrame, periods: int = 7) -> (pd.DataFrame, Prophet):
    """Dự báo giá với Prophet cải tiến hiệu suất"""
    if df.empty or len(df) < 30:  # Tối thiểu 30 điểm dữ liệu để dự báo
        return pd.DataFrame(), None
    
    # Chuẩn bị dữ liệu
    df_prophet = prepare_data_for_prophet(df)
    if df_prophet.empty:
        return pd.DataFrame(), None
    
    try:
        # Lấy danh sách năm từ dữ liệu
        years = pd.DatetimeIndex(df_prophet['ds']).year.unique().tolist()
        # Thêm năm hiện tại và năm tiếp theo cho dự báo
        current_year = datetime.now().year
        years.extend([current_year, current_year + 1])
        years = list(set(years))  # Loại bỏ trùng lặp
        
        # Lấy ngày nghỉ
        holidays_df = get_vietnam_holidays(years)
        
        # Khởi tạo và train model với các tham số tối ưu
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays_df,
            seasonality_mode='multiplicative',  # Tốt hơn cho dữ liệu chứng khoán
            interval_width=0.95,  # Khoảng tin cậy 95%
            changepoint_prior_scale=0.05,  # Mức độ linh hoạt của đường xu hướng
            changepoint_range=0.9  # Chỉ phát hiện điểm thay đổi trong 90% đầu tiên của dữ liệu
        )
        
        # Thêm seasonality mùa vụ theo quý
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
        
        # Fit model
        model.fit(df_prophet)
        
        # Dự báo
        future = model.make_future_dataframe(periods=periods, freq='D')
        # Lọc ra chỉ các ngày trong tuần (không lấy cuối tuần)
        future = future[future['ds'].dt.dayofweek < 5]
        
        # Dự báo với future dataframe đã lọc
        forecast = model.predict(future)
        
        return forecast, model
    except Exception as e:
        logger.error(f"Lỗi dự báo Prophet: {str(e)}")
        return pd.DataFrame(), None

def evaluate_prophet_performance(df: pd.DataFrame, forecast: pd.DataFrame) -> float:
    """Đánh giá hiệu suất model Prophet cải tiến"""
    if df.empty or forecast.empty:
        return 0.0
    
    # Chuẩn bị dữ liệu
    df_prophet = prepare_data_for_prophet(df)
    if df_prophet.empty:
        return 0.0
    
    # Lấy giá trị thực tế từ df_prophet
    actual = df_prophet.set_index('ds')['y']
    
    # Lấy giá trị dự báo từ forecast (chỉ lấy những ngày trùng với dữ liệu thực tế)
    predicted = forecast.set_index('ds')['yhat']
    predicted = predicted.loc[actual.index.intersection(predicted.index)]
    actual = actual.loc[predicted.index]
    
    if len(actual) == 0 or len(predicted) == 0:
        return 0.0
    
    # Tính MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Convert to accuracy (100 - MAPE)
    accuracy = max(0, 100 - mape)
    
    return accuracy / 100  # Return as a decimal between 0 and 1

@measure_execution_time
async def predict_xgboost_signal(df: pd.DataFrame, features: list) -> (int, float):
    """Dự đoán tín hiệu giao dịch bằng XGBoost"""
    try:
        # Kiểm tra model đã train
        model_db = ModelDBManager()
        symbol = "generic"  # Sử dụng model chung nếu không có symbol cụ thể
        
        if hasattr(df, 'symbol') and df.symbol:
            symbol = df.symbol
        
        # Tải model
        model = await model_db.load_trained_model(symbol, 'xgboost')
        
        if model is None:
            logger.warning(f"Không tìm thấy model XGBoost cho {symbol}, sử dụng dự đoán đơn giản")
            # Dự đoán đơn giản dựa trên xu hướng gần đây
            if len(df) >= 5:
                recent_trend = df['close'].iloc[-5:].pct_change().mean()
                signal = 1 if recent_trend > 0 else -1 if recent_trend < 0 else 0
                confidence = min(abs(recent_trend) * 20, 0.7)  # Scale confidence
                return signal, confidence
            return 0, 0.0
        
        # Chuẩn bị dữ liệu đầu vào
        if not all(feature in df.columns for feature in features):
            logger.error(f"Thiếu một số đặc trưng cần thiết cho XGBoost: {set(features) - set(df.columns)}")
            return 0, 0.0
        
        # Lấy dữ liệu mới nhất
        latest_data = df.iloc[-1][features].values.reshape(1, -1)
        
        # Dự đoán
        prediction = model.predict(latest_data)[0]
        probability = model.predict_proba(latest_data)[0]
        
        # Trả về tín hiệu (1 = mua, 0 = giữ, -1 = bán) và độ tin cậy
        confidence = probability[int(prediction)] if len(probability) > 1 else probability[0]
        
        return int(prediction), float(confidence)
    except Exception as e:
        logger.error(f"Lỗi dự đoán XGBoost: {str(e)}")
        return 0, 0.0

@measure_execution_time
async def train_prophet_model(df: pd.DataFrame) -> (Prophet, float):
    """Train model Prophet với xử lý bất đồng bộ"""
    if df.empty or len(df) < 90:  # Cần ít nhất 90 ngày dữ liệu
        logger.warning("Không đủ dữ liệu để train model Prophet")
        return None, 0.0
    
    try:
        # Chuẩn bị dữ liệu
        df_prophet = prepare_data_for_prophet(df)
        
        # Split data for evaluation
        train_size = int(len(df_prophet) * 0.8)
        train_df = df_prophet.iloc[:train_size]
        test_df = df_prophet.iloc[train_size:]
        
        if train_df.empty or test_df.empty:
            return None, 0.0
        
        # Train model trên bộ train
        forecast_df, model = await run_in_process(forecast_with_prophet, train_df, len(test_df))
        
        if model is None:
            return None, 0.0
        
        # Đánh giá hiệu suất
        performance = await run_in_thread(evaluate_prophet_performance, test_df, forecast_df)
        
        # Train lại trên toàn bộ dữ liệu
        final_forecast, final_model = await run_in_process(forecast_with_prophet, df_prophet, 7)
        
        return final_model, performance
    except Exception as e:
        logger.error(f"Lỗi train model Prophet: {str(e)}")
        return None, 0.0

@measure_execution_time
async def train_xgboost_model(df: pd.DataFrame, features: list) -> (xgb.XGBClassifier, float):
    """Train model XGBoost với cải tiến hiệu suất"""
    if df.empty or len(df) < 100:  # Cần ít nhất 100 điểm dữ liệu
        logger.warning("Không đủ dữ liệu để train model XGBoost")
        return None, 0.0
    
    try:
        # Tạo nhãn (1 = tăng, 0 = giảm) cho dữ liệu
        df = df.copy()
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Loại bỏ NaN
        df = df.dropna()
        
        if df.empty:
            return None, 0.0
        
        # Kiểm tra và lọc features
        available_features = [f for f in features if f in df.columns]
        if len(available_features) < len(features):
            logger.warning(f"Thiếu một số đặc trưng cho XGBoost: {set(features) - set(available_features)}")
        
        if not available_features:
            return None, 0.0
        
        # Chuẩn bị dữ liệu
        X = df[available_features]
        y = df['target']
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Tham số tối ưu cho XGBoost
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'n_estimators': 200,
            'random_state': 42
        }
        
        # Function để train model trong process khác để tối ưu memory
        def train_model(X, y, params, tscv):
            model = xgb.XGBClassifier(**params)
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                f1 = f1_score(y_test, y_pred, average='weighted')
                cv_scores.append(f1)
            
            # Final model training on full dataset
            final_model = xgb.XGBClassifier(**params)
            final_model.fit(X, y)
            
            # Average F1 score as performance metric
            avg_performance = np.mean(cv_scores)
            
            return final_model, avg_performance
        
        # Train model in a separate process
        model, performance = await run_in_process(train_model, X, y, params, tscv)
        
        return model, performance
    except Exception as e:
        logger.error(f"Lỗi train model XGBoost: {str(e)}")
        return None, 0.0

async def get_training_symbols() -> list:
    """Lấy danh sách các mã chứng khoán để train mô hình"""
    # Các mã chứng khoán lớn và được quan tâm
    default_symbols = ['VNM', 'VCB', 'VIC', 'VHM', 'HPG', 'MSN', 'VRE', 'FPT', 'MWG', 'TCB']
    
    try:
        # Thu thập từ VNStock
        def fetch_vn30():
            try:
                # Import trực tiếp từ vnstock module
                from vnstock import listing_companies
                
                # Sử dụng listing_companies thay vì ticker_industry
                return listing_companies()
            except Exception as e:
                logger.error(f"Lỗi khi lấy danh sách công ty: {str(e)}")
                return None
        
        industry_data = await run_in_thread(fetch_vn30)
        
        if industry_data is not None and not industry_data.empty:
            # Lấy top 30 mã có vốn hóa lớn nhất (nếu có thông tin vốn hóa)
            if 'markettCap' in industry_data.columns:
                top_symbols = industry_data.sort_values('markettCap', ascending=False).head(30)['ticker'].tolist()
                return list(set(top_symbols + default_symbols))
        
        return default_symbols
    except Exception as e:
        logger.error(f"Lỗi lấy danh sách mã train model: {str(e)}")
        return default_symbols

@measure_execution_time
async def train_models_for_symbol(symbol: str):
    """Huấn luyện các mô hình Prophet và XGBoost cho mã chứng khoán cụ thể"""
    try:
        # Sử dụng pipeline mới để tải và xử lý dữ liệu
        data_pipeline = DataPipeline()
        df, processing_report = await data_pipeline.process_data(symbol, timeframe='1D', num_candles=500)
        
        if df.empty or processing_report['status'] == 'error':
            logger.error(f"Không thể tải dữ liệu cho {symbol}: {processing_report.get('message', 'Unknown error')}")
            return False
        
        # Thêm các chỉ báo kỹ thuật
        analyzer = TechnicalAnalyzer()
        df = analyzer.calculate_indicators(df)
        
        # Tạo các đặc trưng cho mô hình XGBoost
        features = [col for col in df.columns if col.startswith(('rsi', 'macd', 'ema', 'ma', 'bb_'))]
        
        model_db = ModelDBManager()
        
        # Huấn luyện mô hình Prophet
        prophet_model, prophet_performance = await train_prophet_model(df)
        if prophet_model:
            await model_db.store_trained_model(symbol, 'prophet', prophet_model, prophet_performance)
            logger.info(f"Huấn luyện Prophet cho {symbol} thành công, hiệu suất: {prophet_performance:.2f}")
        
        # Huấn luyện mô hình XGBoost
        xgb_model, xgb_performance = await train_xgboost_model(df, features)
        if xgb_model:
            await model_db.store_trained_model(symbol, 'xgboost', xgb_model, xgb_performance)
            logger.info(f"Huấn luyện XGBoost cho {symbol} thành công, hiệu suất: {xgb_performance:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"Lỗi huấn luyện mô hình cho {symbol}: {str(e)}")
        return False

async def auto_train_models():
    """Tự động train tất cả các mô hình theo lịch"""
    logger.info("Bắt đầu quá trình train mô hình tự động")
    
    try:
        # Lấy danh sách mã chứng khoán cần train
        symbols = await get_training_symbols()
        
        # Khởi tạo các tác vụ train
        tasks = [train_models_for_symbol(symbol) for symbol in symbols]
        
        # Chạy song song nhưng giới hạn số lượng tác vụ đồng thời để tránh quá tải
        semaphore = asyncio.Semaphore(4)  # Chạy tối đa 4 tác vụ song song
        
        async def train_with_semaphore(task):
            async with semaphore:
                return await task
        
        # Bọc các tác vụ với semaphore
        semaphore_tasks = [train_with_semaphore(task) for task in tasks]
        
        # Chạy và đợi hoàn thành
        await asyncio.gather(*semaphore_tasks)
        
        logger.info(f"Hoàn thành train mô hình cho {len(symbols)} mã chứng khoán")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình auto train: {str(e)}")

# ---------- AI VÀ BÁO CÁO ----------
class AIAnalyzer:
    """Lớp phân tích AI với Gemini cải tiến"""
    
    def __init__(self):
        self.db_manager = DBManager()
        
        # Check if Gemini API key is set
        self.gemini_available = bool(GEMINI_API_KEY) and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY"
        
        if self.gemini_available:
            self.model = genai.GenerativeModel('gemini-pro')
            self.safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        else:
            logger.warning("AIAnalyzer initialized without Gemini API. Will use basic analysis only.")
            self.model = None
            self.safety_settings = None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_content(self, prompt):
        """Tạo nội dung AI với Gemini API"""
        if not self.gemini_available:
            return "AI analysis not available (Gemini API key not set)"
            
        try:
            response = await run_in_thread(
                lambda: self.model.generate_content(
                    prompt,
                    safety_settings=self.safety_settings,
                    generation_config={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "max_output_tokens": 2048,
                    }
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Lỗi tạo nội dung Gemini: {str(e)}")
            return "Không thể tạo báo cáo do lỗi AI. Vui lòng thử lại sau."
    
    async def load_report_history(self, symbol: str) -> list:
        """Wrapper cho db_manager.load_report_history"""
        return await self.db_manager.load_report_history(symbol)
    
    async def save_report_history(self, symbol: str, report: str, close_today: float, close_yesterday: float) -> None:
        """Wrapper cho db_manager.save_report_history"""
        await self.db_manager.save_report_history(symbol, report, close_today, close_yesterday)
    
    @measure_execution_time
    def analyze_price_action(self, df: pd.DataFrame) -> str:
        """Phân tích hành động giá với nhiều mẫu hình giá cải tiến"""
        if df.empty or len(df) < 10:
            return "Không đủ dữ liệu để phân tích hành động giá."
        
        # Tạo bản sao để tránh SettingWithCopyWarning
        df = df.copy()
        
        # Đảm bảo có đủ các cột cần thiết
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return "Thiếu dữ liệu OHLC cần thiết để phân tích hành động giá."
        
        # Kết quả phân tích
        analysis = []
        
        # 1. Phân tích xu hướng gần đây
        recent_days = min(10, len(df))
        recent_df = df.iloc[-recent_days:]
        
        # Tính % thay đổi
        close_change_pct = (recent_df['close'].iloc[-1] / recent_df['close'].iloc[0] - 1) * 100
        
        if close_change_pct > 5:
            trend = "tăng mạnh"
        elif close_change_pct > 2:
            trend = "tăng"
        elif close_change_pct < -5:
            trend = "giảm mạnh"
        elif close_change_pct < -2:
            trend = "giảm"
        else:
            trend = "đi ngang"
        
        analysis.append(f"Xu hướng {recent_days} phiên gần nhất: {trend} ({close_change_pct:.2f}%)")
        
        # 2. Phân tích mẫu hình nến
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2] if len(df) > 1 else None
        
        # 2.1. Phân tích nến hiện tại
        body_size = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        upper_shadow = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_shadow = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        # Doji (thân rất nhỏ)
        if body_size < 0.1 * total_range:
            analysis.append("Nến Doji: Phản ánh sự do dự, không chắc chắn của thị trường.")
        
        # Hammer/Inverted Hammer (bóng dưới/trên dài)
        elif lower_shadow > 2 * body_size and upper_shadow < 0.2 * body_size:
            analysis.append("Nến Hammer: Tín hiệu đảo chiều tăng tiềm năng.")
        elif upper_shadow > 2 * body_size and lower_shadow < 0.2 * body_size:
            analysis.append("Nến Inverted Hammer: Có thể là tín hiệu đảo chiều.")
        
        # Bullish/Bearish Engulfing
        if prev_candle is not None:
            prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
            
            if (last_candle['close'] > last_candle['open'] and  # Nến tăng hiện tại
                prev_candle['close'] < prev_candle['open'] and  # Nến giảm trước đó
                last_candle['open'] < prev_candle['close'] and  # Mở cửa thấp hơn đóng cửa trước
                last_candle['close'] > prev_candle['open']):   # Đóng cửa cao hơn mở cửa trước
                analysis.append("Mẫu hình Bullish Engulfing: Tín hiệu đảo chiều tăng mạnh.")
            
            elif (last_candle['close'] < last_candle['open'] and  # Nến giảm hiện tại
                  prev_candle['close'] > prev_candle['open'] and  # Nến tăng trước đó
                  last_candle['open'] > prev_candle['close'] and  # Mở cửa cao hơn đóng cửa trước
                  last_candle['close'] < prev_candle['open']):   # Đóng cửa thấp hơn mở cửa trước
                analysis.append("Mẫu hình Bearish Engulfing: Tín hiệu đảo chiều giảm mạnh.")
        
        # 3. Phân tích hỗ trợ/kháng cự
        price_levels = list(df['close']) + list(df['high']) + list(df['low'])
        
        # Sử dụng KMeans để phân cụm giá
        from sklearn.cluster import KMeans
        
        # Reshape để phù hợp với KMeans
        price_array = np.array(price_levels).reshape(-1, 1)
        
        # Xác định số lượng cụm (3-5 là hợp lý)
        n_clusters = min(5, len(df) // 10 + 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(price_array)
        
        # Lấy các tâm cụm và sắp xếp
        support_resistance_levels = sorted(kmeans.cluster_centers_.flatten())
        
        # Giá hiện tại
        current_price = df['close'].iloc[-1]
        
        # Tìm mức hỗ trợ/kháng cự gần nhất
        closest_support = max([level for level in support_resistance_levels if level < current_price], default=None)
        closest_resistance = min([level for level in support_resistance_levels if level > current_price], default=None)
        
        if closest_support:
            support_distance = (current_price - closest_support) / current_price * 100
            analysis.append(f"Mức hỗ trợ gần nhất: {closest_support:.2f} (cách {support_distance:.2f}%)")
        
        if closest_resistance:
            resistance_distance = (closest_resistance - current_price) / current_price * 100
            analysis.append(f"Mức kháng cự gần nhất: {closest_resistance:.2f} (cách {resistance_distance:.2f}%)")
        
        # Kết hợp kết quả
        return "\n".join(analysis)
    
    @measure_execution_time
    async def generate_report(self, dfs: dict, symbol: str, fundamental_data: dict, outlier_reports: dict) -> str:
        """Tạo báo cáo tổng hợp với phân tích kỹ thuật, cơ bản và AI"""
        try:
            # Lấy DataFrame chính
            df_daily = dfs.get('1D', pd.DataFrame())
            if df_daily.empty:
                return f"Không có dữ liệu cho {symbol}"
            
            # Khởi tạo bộ nhớ đệm cho báo cáo
            report_parts = []
            
            # Thông tin cơ bản
            last_price = df_daily['close'].iloc[-1] if 'close' in df_daily.columns else None
            prev_price = df_daily['close'].iloc[-2] if 'close' in df_daily.columns and len(df_daily) > 1 else None
            
            price_change = 0
            price_change_pct = 0
            if last_price is not None and prev_price is not None:
                price_change = last_price - prev_price
                price_change_pct = (price_change / prev_price) * 100
            
            price_change_symbol = "🔴" if price_change < 0 else "🟢" if price_change > 0 else "⚪"
            
            # Tiêu đề báo cáo
            report_title = f"📊 *PHÂN TÍCH {symbol}*\n"
            report_title += f"Giá: {last_price:,.2f} ({price_change_symbol}{price_change:+,.2f} | {price_change_pct:+,.2f}%)\n"
            report_title += f"Ngày: {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
            report_parts.append(report_title)
            
            # Thêm cảnh báo nếu phát hiện dữ liệu bất thường
            outlier_report = outlier_reports.get(symbol, "")
            if outlier_report and "Phát hiện" in outlier_report:
                # Sanitize outlier report text
                outlier_report = outlier_report.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                report_parts.append(f"⚠️ *CẢNH BÁO DỮ LIỆU BẤT THƯỜNG*\n{outlier_report}\n")
            
            # Phân tích giá
            price_analysis = self.analyze_price_action(df_daily)
            # Sanitize price analysis text
            price_analysis = price_analysis.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            report_parts.append(f"*PHÂN TÍCH GIÁ*\n{price_analysis}\n")
            
            # Phân tích cơ bản
            if fundamental_data:
                fundamental_analysis = deep_fundamental_analysis(fundamental_data)
                # Sanitize fundamental analysis text
                fundamental_analysis = fundamental_analysis.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                report_parts.append(f"*PHÂN TÍCH CƠ BẢN*\n{fundamental_analysis}\n")
            
            # Tin tức
            try:
                news_items = await get_news(symbol, limit=3)
                if news_items:
                    news_section = "*TIN TỨC MỚI NHẤT*\n"
                    for item in news_items:
                        # Sanitize news title
                        title = item['title'].encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                        news_section += f"• [{title}]({item['link']})\n"
                    report_parts.append(f"{news_section}\n")
            except Exception as e:
                logger.error(f"Lỗi lấy tin tức: {str(e)}")
            
            # Dự báo
            try:
                # Prophet
                df_prophet = prepare_data_for_prophet(df_daily)
                forecast, _ = forecast_with_prophet(df_prophet, periods=7)
                
                if not forecast.empty:
                    last_date = df_daily.index[-1]
                    forecast_filtered = forecast[forecast['ds'] > pd.Timestamp(last_date)]
                    
                    if not forecast_filtered.empty:
                        forecast_section = "*DỰ BÁO GIÁ (7 NGÀY TỚI)*\n"
                        for _, row in forecast_filtered.iterrows():
                            date_str = row['ds'].strftime('%d/%m/%Y')
                            forecast_price = row['yhat']
                            lower_bound = row['yhat_lower']
                            upper_bound = row['yhat_upper']
                            
                            # Xác định xu hướng
                            if forecast_filtered.index[0] == _:  # Ngày đầu tiên
                                change = forecast_price - last_price
                                change_pct = (change / last_price) * 100
                            else:
                                prev_forecast = forecast_filtered.iloc[forecast_filtered.index.get_loc(_) - 1]['yhat']
                                change = forecast_price - prev_forecast
                                change_pct = (change / prev_forecast) * 100
                            
                            trend_symbol = "🔴" if change < 0 else "🟢" if change > 0 else "⚪"
                            
                            forecast_section += f"{trend_symbol} {date_str}: {forecast_price:,.2f} [{lower_bound:,.2f} - {upper_bound:,.2f}] ({change_pct:+,.2f}%)\n"
                        
                        report_parts.append(f"{forecast_section}\n")
                
                # XGBoost Signal
                features = [col for col in df_daily.columns if col.startswith(('rsi', 'macd', 'ema', 'ma', 'bb_'))]
                if features:
                    signal, confidence = await predict_xgboost_signal(df_daily, features)
                    
                    signal_text = "MUA" if signal == 1 else "BÁN" if signal == -1 else "ĐỨNG NGOÀI"
                    signal_symbol = "🟢" if signal == 1 else "🔴" if signal == -1 else "⚪"
                    
                    report_parts.append(f"*TÍN HIỆU GIAO DỊCH*\n{signal_symbol} {signal_text} (Độ tin cậy: {confidence*100:.1f}%)\n")
            except Exception as e:
                logger.error(f"Lỗi tạo dự báo: {str(e)}")
            
            # Kết hợp tất cả phần báo cáo và đảm bảo xử lý Unicode
            final_report = '\n'.join(report_parts)
            # Final sanitization of the entire report
            final_report = final_report.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            
            # Lưu báo cáo vào lịch sử
            if last_price is not None and prev_price is not None:
                await self.save_report_history(symbol, final_report, last_price, prev_price)
            
            return final_report
        except Exception as e:
            logger.error(f"Lỗi tạo báo cáo: {str(e)}")
            return f"Đã xảy ra lỗi khi tạo báo cáo: {str(e)}"

# ---------- TELEGRAM COMMANDS ----------
async def notify_admin_new_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Check if update is None (called from main function)
    if update is None:
        # Just log that we're starting the bot
        logger.info("Bot started successfully")
        return
        
    user = update.message.from_user
    user_id = user.id
    if not await is_user_approved(user_id):
        message = f"🔔 Người dùng mới:\nID: {user_id}\nUsername: {user.username}\nTên: {user.full_name}\nDuyệt: /approve {user_id}"
        await context.bot.send_message(chat_id=ADMIN_ID, text=message)
        await update.message.reply_text("⏳ Chờ admin duyệt!")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    logger.info(f"Start called: user_id={user_id}, ADMIN_ID={ADMIN_ID}")

    if str(user_id) == ADMIN_ID and not await db.is_user_approved(user_id):
        await db.add_approved_user(user_id)
        logger.info(f"Admin {user_id} tự động duyệt.")

    if not await is_user_approved(user_id):
        await notify_admin_new_user(update, context)
        return

    await update.message.reply_text(
        "🚀 **V18.7 - THUA GIA CÁT LƯỢNG MỖI CÁI QUẠT!**\n"
        "📊 **Lệnh**:\n"
        "- /analyze [Mã] [Số nến] - Phân tích đa khung.\n"
        "- /getid - Lấy ID.\n"
        "- /approve [user_id] - Duyệt người dùng (admin).\n"
        "💡 **Bắt đầu nào!**"
    )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /analyze"""
    # Kiểm tra quyền
    user_id = str(update.effective_user.id)
    if not await is_user_approved(user_id):
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Bạn không có quyền sử dụng lệnh này. Vui lòng liên hệ admin để được cấp quyền."
        )
        return
    
    # Lấy tham số
    args = context.args
    if not args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Vui lòng cung cấp mã chứng khoán. Ví dụ: /analyze VNM"
        )
        return
    
    symbol = args[0].upper()
    
    # Gửi tin nhắn đang phân tích
    message = await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"⏳ Đang phân tích {symbol}..."
    )
    
    try:
        # Khởi tạo data pipeline
        data_pipeline = DataPipeline()
        analyzer = TechnicalAnalyzer()
        ai_analyzer = AIAnalyzer()
        
        # Tải và xử lý dữ liệu
        dfs, fundamental_data, processing_report = await data_pipeline.get_fundamental_and_technical_data(symbol)
        
        if not dfs or all(df.empty for df in dfs.values()):
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=message.message_id,
                text=f"Không thể tải dữ liệu cho {symbol}. Vui lòng kiểm tra mã chứng khoán."
            )
            return
            
        # Tính toán chỉ báo kỹ thuật
        dfs_with_indicators = analyzer.calculate_multi_timeframe_indicators(dfs)
        
        # Tạo báo cáo
        report = await ai_analyzer.generate_report(
            dfs_with_indicators, 
            symbol, 
            fundamental_data,
            data_pipeline.outlier_reports
        )
        
        # Gửi báo cáo
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=message.message_id,
            text=report,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        logger.error(f"Lỗi xử lý lệnh analyze: {str(e)}")
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=message.message_id,
            text=f"Đã xảy ra lỗi: {str(e)}"
        )

async def get_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    await update.message.reply_text(f"ID của bạn: {user_id}")

async def approve_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.message.from_user.id) != ADMIN_ID:
        await update.message.reply_text("❌ Chỉ admin dùng được lệnh này!")
        return
    if len(context.args) != 1:
        await update.message.reply_text("❌ Nhập user_id: /approve 123456789")
        return
    user_id = context.args[0]
    if not await db.is_user_approved(user_id):
        await db.add_approved_user(user_id)
        await update.message.reply_text(f"✅ Đã duyệt {user_id}")
    else:
        await update.message.reply_text(f"ℹ️ {user_id} đã được duyệt")

# ---------- MAIN & DEPLOY ----------
async def main():
    """Hàm chính để khởi động bot"""
    try:
        # Khởi tạo cơ sở dữ liệu
        await init_db()
        
        # Cài đặt logging
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )
        
        # Check if Telegram token is set correctly
        if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN":
            logger.error("Telegram token not set. Please set the TELEGRAM_TOKEN environment variable.")
            logger.warning("Running in limited mode without Telegram bot integration.")
            
            # Instead of returning, you could add code here to run in a "headless" mode
            # where the bot doesn't connect to Telegram but performs other tasks
            
            # For now, we'll just show that the bot is operational but limited
            logger.info("Bot started successfully in limited mode (no Telegram integration)")
            return
        
        # Mode bot: webhook hoặc polling
        use_webhook = os.getenv('USE_WEBHOOK', 'True').lower() == 'true'
        mode = os.getenv('MODE', 'production').lower()
        port = int(os.getenv('PORT', 8443))
        
        # Test mode
        if mode == 'test':
            # Log test bắt đầu
            logger.info("--- BẮT ĐẦU CHẾ ĐỘ KIỂM THỬ ---")
            
            # Tạo dữ liệu kiểm thử
            symbol = 'VNM'  # Mã cổ phiếu kiểm thử
            
            # Kiểm thử data pipeline
            logger.info("Đang kiểm thử data pipeline...")
            data_pipeline = DataPipeline()
            df, processing_report = await data_pipeline.process_data(symbol)
            
            logger.info(f"Đã tải và xử lý dữ liệu: {len(df)} bản ghi")
            logger.info(f"Báo cáo xử lý: {processing_report['status']}")
            
            if 'outliers' in processing_report and 'Phát hiện' in processing_report['outliers']:
                logger.info(f"Phát hiện outliers: {processing_report['outliers']}")
            
            # Kiểm thử phân tích kỹ thuật
            logger.info("Đang kiểm thử phân tích kỹ thuật...")
            analyzer = TechnicalAnalyzer()
            df_with_indicators = analyzer.calculate_indicators(df)
            logger.info(f"Đã tính toán {len(df_with_indicators.columns) - len(df.columns)} chỉ báo kỹ thuật")
            
            # Kiểm thử mô hình dự báo
            logger.info("Đang kiểm thử mô hình dự báo...")
            model_success = await train_models_for_symbol(symbol)
            if model_success:
                logger.info("Huấn luyện mô hình thành công")
            
            # Kiểm thử sinh báo cáo AI
            logger.info("Đang kiểm thử sinh báo cáo AI...")
            dfs, fundamental_data, _ = await data_pipeline.get_fundamental_and_technical_data(symbol)
            dfs_with_indicators = analyzer.calculate_multi_timeframe_indicators(dfs)
            
            ai_analyzer = AIAnalyzer()
            report = await ai_analyzer.generate_report(
                dfs_with_indicators, 
                symbol, 
                fundamental_data,
                data_pipeline.outlier_reports
            )
            
            logger.info(f"Đã tạo báo cáo AI: {len(report)} ký tự")
            logger.info("--- KẾT THÚC CHẾ ĐỘ KIỂM THỬ ---")
            
            # Huấn luyện mô hình tự động
            if os.getenv('AUTO_TRAIN', 'False').lower() == 'true':
                logger.info("Bắt đầu huấn luyện mô hình tự động...")
                await auto_train_models()
            
            return
        
        # Khởi tạo bot Telegram
        application = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Thêm handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("analyze", analyze_command))
        application.add_handler(CommandHandler("getid", get_id))
        application.add_handler(CommandHandler("approve", approve_user))
        
        # Ghi log khởi động bot (passing None as update since this is initialization, not a user update)
        await notify_admin_new_user(None, application)
        
        # Chọn chế độ chạy - luôn sử dụng webhook
        # Webhook mode
        if RENDER_EXTERNAL_URL:
            webhook_path = '/telegram/webhook'
            webhook_url = f"{RENDER_EXTERNAL_URL}{webhook_path}"
        else:
            webhook_path = '/telegram/webhook'
            webhook_url = os.getenv('WEBHOOK_URL', f'https://example.com:{port}{webhook_path}')
        webhook_listen = os.getenv('WEBHOOK_LISTEN', '0.0.0.0')
        webhook_port = port
        
        logger.info(f"Starting webhook on {webhook_url}")
        await application.bot.set_webhook(url=webhook_url, secret_token=TELEGRAM_TOKEN)
        
        # Setup aiohttp webapp
        app = web.Application()
        
        # Telegram webhook handler
        async def telegram_webhook_handler(request):
            try:
                # Verify the Telegram secret token for security
                secret_header = request.headers.get('X-Telegram-Bot-Api-Secret-Token')
                if secret_header != TELEGRAM_TOKEN:
                    logger.warning(f"Unauthorized webhook request from {request.remote}")
                    return web.Response(status=403)
                    
                request_body_json = await request.json()
                update = Update.de_json(request_body_json, application.bot)
                # Initialize the application before processing updates
                await application.initialize()
                await application.process_update(update)
                return web.Response()
            except Exception as e:
                logger.error(f"Error in webhook handler: {str(e)}")
                return web.Response(status=500)
        
        # Health check endpoint
        async def health_check_handler(request):
            # Kiểm tra Redis và DB để đảm bảo các thành phần hoạt động
            try:
                # Kiểm tra Redis availability
                redis_health = False
                if redis_manager.redis_client is not None:
                    redis_key = "health_check"
                    await redis_manager.set(redis_key, True, 10)
                    redis_value = await redis_manager.get(redis_key)
                    redis_health = redis_value is True
                
                # Kiểm tra DB
                db_health = await db_manager.is_user_approved("admin")
                
                return web.json_response({
                    "status": "healthy" if redis_health and db_health is not None else "degraded",
                    "redis": "ok" if redis_health else "disabled" if redis_manager.redis_client is None else "error",
                    "database": "ok" if db_health is not None else "error",
                    "uptime": time.time() - START_TIME
                })
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                return web.json_response({
                    "status": "error",
                    "message": str(e)
                }, status=500)
        
        # Middleware để log các request
        @web.middleware
        async def logging_middleware(request, handler):
            start_time = time.time()
            try:
                response = await handler(request)
                end_time = time.time()
                logger.info(f"Request {request.method} {request.path} completed in {end_time - start_time:.3f}s with status {response.status}")
                return response
            except Exception as e:
                end_time = time.time()
                logger.error(f"Request {request.method} {request.path} failed in {end_time - start_time:.3f}s: {str(e)}")
                raise
        
        # Áp dụng middleware
        app.middlewares.append(logging_middleware)
        
        # Đăng ký các route - use the webhook_path variable
        app.router.add_post(webhook_path, telegram_webhook_handler)
        app.router.add_get('/health', health_check_handler)
        
        # Bắt đầu server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, webhook_listen, webhook_port)
        await site.start()
        
        # Giữ ứng dụng chạy
        while True:
            # Mỗi 3 giờ huấn luyện mô hình tự động nếu được bật
            if os.getenv('AUTO_TRAIN', 'False').lower() == 'true':
                await auto_train_models()
                
            await asyncio.sleep(3 * 60 * 60)  # 3 giờ
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

# Check if being run directly as script
if __name__ == "__main__":
    # Khởi tạo các managers
    redis_manager = RedisManager()
    db_manager = DBManager()
    model_db = ModelDBManager()
    START_TIME = time.time()
    
    try:
        # Khởi tạo Gemini API if API key is available
        if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
            logger.warning("Gemini API key not set. AI-powered features will not be available.")
        else:
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("Gemini API initialized successfully")
        
        # Khởi tạo constants
        TOKEN = TELEGRAM_TOKEN
        
        # Check if we're running in test mode
        if len(sys.argv) > 1 and sys.argv[1] == "-test":
            run_tests()
        else:
            # Chạy bot bình thường
            import nest_asyncio
            nest_asyncio.apply()
            asyncio.run(main())
    except Exception as e:
        logger.critical(f"Lỗi khởi động bot: {str(e)}")
        # Tạo thông báo lỗi chi tiết
        error_message = f"CRITICAL ERROR: {str(e)}\n"
        error_message += f"Traceback: {sys.exc_info()}"
        logger.critical(error_message)
        sys.exit(1)

# Add standalone test function
def run_tests():
    """Run unit tests for the new features"""
    class TestDataValidator(unittest.TestCase):
        def setUp(self):
            dates = pd.date_range(start="2023-01-01", periods=7, freq='D')
            self.df = pd.DataFrame({
                'open': [100, 101, 102, 103, 500, 104, 105],
                'high': [110, 111, 112, 113, 550, 114, 115],
                'low': [95, 96, 97, 98, 480, 99, 100],
                'close': [105, 106, 107, 108, 520, 109, 110],
                'volume': [1000, 1100, 1200, 1300, 15000, 1400, 1500]
            }, index=dates)
            
            # Add invalid candle
            self.df_invalid = self.df.copy()
            self.df_invalid.loc[dates[2], 'high'] = 90  # Invalid high (lower than low)
            
        def test_validate_data(self):
            validator = DataValidator()
            df_valid, report = validator.validate_data(self.df)
            self.assertEqual(report['status'], 'success')
            self.assertEqual(len(df_valid), 7)
            
            # Test invalid data
            df_fixed, report = validator.validate_data(self.df_invalid)
            self.assertEqual(report['status'], 'success')
            self.assertTrue(any("nến không hợp lệ" in warning for warning in report['warnings']))
            self.assertEqual(df_fixed.loc[self.df_invalid.index[2], 'high'], 
                            max(self.df_invalid.loc[self.df_invalid.index[2], ['open', 'close', 'high']]))
        
        def test_normalize_data(self):
            validator = DataValidator()
            df_norm = validator.normalize_data(self.df)
            self.assertIn('volume_norm', df_norm.columns)
            self.assertIn('close_zscore', df_norm.columns)
            self.assertIn('close_pct_change', df_norm.columns)
            
        def test_remove_duplicates(self):
            # Create DataFrame with duplicates
            df_with_dups = pd.concat([self.df, self.df.iloc[-2:-1]])
            validator = DataValidator()
            df_cleaned, removed = validator.remove_duplicates(df_with_dups)
            self.assertEqual(removed, 1)
            self.assertEqual(len(df_cleaned), 7)
    
    class TestOutlierDetection(unittest.TestCase):
        def setUp(self):
            dates = pd.date_range(start="2023-01-01", periods=7, freq='D')
            self.df = pd.DataFrame({
                'close': [100, 101, 102, 103, 500, 104, 105]
            }, index=dates)
        
        def test_detect_outliers(self):
            loader = DataLoader()
            df_with_outliers, report = loader.detect_outliers(self.df)
            self.assertIn('is_outlier', df_with_outliers.columns)
            self.assertEqual(df_with_outliers['is_outlier'].sum(), 1)
            self.assertIn('500', report)
            # Check multiple outlier detection methods
            self.assertIn('outlier_z_score', df_with_outliers.columns)
            self.assertIn('outlier_iqr', df_with_outliers.columns)
            self.assertIn('outlier_price_change', df_with_outliers.columns)
    
    class TestFilterTradingDays(unittest.TestCase):
        def setUp(self):
            # Create DataFrame with weekend days
            dates = pd.date_range(start="2023-01-01", periods=9, freq='D')  # Jan 1, 2023 is Sunday
            self.df = pd.DataFrame({
                'close': list(range(9))
            }, index=dates)
        
        def test_filter_trading_days(self):
            filtered_df = filter_trading_days(self.df)
            # Should exclude weekends (Sunday and Saturday)
            self.assertLess(len(filtered_df), len(self.df))
            # Check that no weekends are in the filtered data
            self.assertTrue(all(idx.dayofweek < 5 for idx in filtered_df.index))
    
    # Run tests
    unittest.main(argv=['first-arg-is-ignored'])