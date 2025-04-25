#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot Chứng Khoán Toàn Diện Phiên Bản V18.9 (Nâng cấp):
- Tích hợp AI OpenRouter cho phân tích mẫu hình, sóng, và nến nhật.
- Sử dụng mô hình deepseek/deepseek-chat-v3-0324:free.
- Nâng cấp bộ xác thực và chuẩn hóa dữ liệu đầu vào (DataValidator).
- Cải tiến hệ thống tải dữ liệu với hỗ trợ đa khung thời gian (5m, 15m, 30m, 1h, 4h, 1D, 1W, 1M).
- Bổ sung xử lý dữ liệu thông minh: phát hiện và xử lý outlier, chuẩn hóa DataFrame.
- Cấu hình webhook độc quyền cho Render, không sử dụng fallback polling.
- Sửa lỗi cảnh báo datetime với isin và lỗi event loop.
- Tăng cường JSON response formatting và xử lý lỗi với OpenRouter API.
- Đảm bảo các chức năng và công nghệ hiện có không bị ảnh hưởng.
"""

import os
import sys
import io
import logging
import pickle
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from telegram import Update
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
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, select, LargeBinary

import xgboost as xgb
from sklearn.metrics import accuracy_score
from prophet import Prophet

import matplotlib.pyplot as plt
import holidays
import html

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tenacity import retry, stop_after_attempt, wait_exponential

import aiohttp
import json
import traceback
import re

# ---------- CẤU HÌNH & LOGGING ----------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")
ADMIN_ID = os.getenv("ADMIN_ID", "1225226589")
PORT = int(os.environ.get("PORT", 10000))
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")
RENDER_SERVICE_NAME = os.getenv("RENDER_SERVICE_NAME", "")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_EXPIRE_SHORT = 1800
CACHE_EXPIRE_MEDIUM = 3600
CACHE_EXPIRE_LONG = 86400
NEWS_CACHE_EXPIRE = 900
DEFAULT_CANDLES = 100
DEFAULT_TIMEFRAME = '1D'

executor = ThreadPoolExecutor(max_workers=5)

async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

# ---------- KẾT NỐI REDIS (Async) ----------
class RedisManager:
    def __init__(self):
        try:
            self.redis_client = redis.from_url(REDIS_URL)
            logger.info("Kết nối Redis thành công.")
        except Exception as e:
            logger.error(f"Lỗi kết nối Redis: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def set(self, key, value, expire):
        try:
            serialized_value = pickle.dumps(value)
            await self.redis_client.set(key, serialized_value, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis set: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get(self, key):
        try:
            data = await self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Lỗi Redis get: {str(e)}")
            return None

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

class TrainedModel(Base):
    __tablename__ = 'trained_models'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    performance = Column(Float, nullable=True)

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

class DBManager:
    def __init__(self):
        self.Session = SessionLocal

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def is_user_approved(self, user_id) -> bool:
        try:
            async with self.Session() as session:
                result = await session.execute(select(ApprovedUser).filter_by(user_id=str(user_id)))
                return result.scalar_one_or_none() is not None or str(user_id) == ADMIN_ID
        except Exception as e:
            logger.error(f"Lỗi kiểm tra người dùng: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def add_approved_user(self, user_id, approved_at=None) -> None:
        try:
            async with self.Session() as session:
                if not await self.is_user_approved(user_id) and str(user_id) != ADMIN_ID:
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
                reports = await session.execute(select(ReportHistory).filter_by(symbol=symbol).order_by(ReportHistory.id.asc()))
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
            async with self.Session() as session:
                date_str = datetime.now().strftime('%Y-%m-%d')
                standardized_data = standardize_data_for_db({
                    'symbol': symbol,
                    'date': date_str,
                    'report': report,
                    'close_today': close_today,
                    'close_yesterday': close_yesterday
                })
                new_report = ReportHistory(**standardized_data)
                session.add(new_report)
                await session.commit()
                logger.info(f"Lưu báo cáo mới cho {symbol}")
        except Exception as e:
            logger.error(f"Lỗi lưu báo cáo: {str(e)}")
            raise

db = DBManager()

# ---------- QUẢN LÝ MÔ HÌNH (Prophet & XGBoost) ----------
class ModelDBManager:
    def __init__(self):
        self.Session = SessionLocal

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def store_trained_model(self, symbol: str, model_type: str, model, performance: float = None):
        try:
            model_blob = pickle.dumps(model)
            async with self.Session() as session:
                result = await session.execute(select(TrainedModel).filter_by(symbol=symbol, model_type=model_type))
                existing = result.scalar_one_or_none()
                if existing:
                    existing.model_blob = model_blob
                    existing.created_at = datetime.now()
                    existing.performance = performance
                else:
                    new_model = TrainedModel(symbol=symbol, model_type=model_type, model_blob=model_blob, performance=performance)
                    session.add(new_model)
                await session.commit()
            logger.info(f"Lưu mô hình {model_type} cho {symbol} thành công với hiệu suất: {performance}")
        except Exception as e:
            logger.error(f"Lỗi lưu mô hình {model_type} cho {symbol}: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_trained_model(self, symbol: str, model_type: str):
        try:
            async with self.Session() as session:
                result = await session.execute(select(TrainedModel).filter_by(symbol=symbol, model_type=model_type))
                model_record = result.scalar_one_or_none()
                if model_record:
                    model = pickle.loads(model_record.model_blob)
                    return model, model_record.performance
            return None, None
        except Exception as e:
            logger.error(f"Lỗi tải mô hình {model_type} cho {symbol}: {str(e)}")
            return None, None

model_db_manager = ModelDBManager()

# ---------- HÀM HỖ TRỢ ----------
def is_index(symbol: str) -> bool:
    indices = ['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX', 'UPCOM']
    return symbol.upper() in indices

async def is_user_approved(user_id) -> bool:
    return await db.is_user_approved(user_id)

def standardize_data_for_db(data: dict) -> dict:
    standardized_data = {}
    for key, value in data.items():
        if isinstance(value, np.float64):
            standardized_data[key] = float(value)
        elif isinstance(value, np.int64):
            standardized_data[key] = int(value)
        elif isinstance(value, pd.Timestamp):
            standardized_data[key] = value.to_pydatetime()
        else:
            standardized_data[key] = value
    return standardized_data

# ---------- HÀM HỖ TRỢ: LỌC NGÀY GIAO DỊCH -----------
def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[df.index.weekday < 5]
    years = df.index.year.unique()
    vn_holidays = holidays.Vietnam(years=years)
    holiday_dates = set(vn_holidays.keys())
    
    # Sửa lỗi deprecation: chuyển đổi rõ ràng để tránh cảnh báo isin với datetime64
    date_indexes = pd.to_datetime(df.index.date)
    holiday_dates_dt = pd.to_datetime(list(holiday_dates))
    
    # Phương pháp an toàn hơn: sử dụng list comprehension thay vì isin
    df = df[[d not in holiday_dates for d in df.index.date]]
    
    return df

# ---------- XÁC THỰC VÀ CHUẨN HÓA DỮ LIỆU ----------
class DataValidator:
    """Xác thực và chuẩn hóa dữ liệu đầu vào cho các phân tích chứng khoán."""
    
    VALID_TIMEFRAMES = {
        "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", 
        "1d": "1D", "1D": "1D", "1w": "1W", "1W": "1W", "1mo": "1M", "1M": "1M"
    }
    
    VALID_SYMBOLS = {
        "indices": ["VNINDEX", "VN30", "HNX30", "HNXINDEX", "UPCOM"],
        "min_length": 3,
        "max_length": 10
    }
    
    @classmethod
    def validate_symbol(cls, symbol: str) -> str:
        """Xác thực và chuẩn hóa mã chứng khoán."""
        if not symbol:
            raise ValueError("Mã chứng khoán không được để trống")
        
        normalized = symbol.strip().upper()
        
        if not 2 <= len(normalized) <= cls.VALID_SYMBOLS["max_length"]:
            raise ValueError(f"Mã chứng khoán phải có từ 2 đến {cls.VALID_SYMBOLS['max_length']} ký tự")
            
        if not normalized.isalnum():
            raise ValueError("Mã chứng khoán chỉ được chứa chữ cái và số")
            
        return normalized
    
    @classmethod
    def validate_timeframe(cls, timeframe: str) -> str:
        """Xác thực và chuẩn hóa khung thời gian."""
        if not timeframe:
            return DEFAULT_TIMEFRAME
            
        normalized = timeframe.lower().strip()
        
        if normalized not in cls.VALID_TIMEFRAMES:
            valid_options = ", ".join(sorted(set(cls.VALID_TIMEFRAMES.keys())))
            raise ValueError(f"Khung thời gian không hợp lệ. Các lựa chọn: {valid_options}")
            
        return cls.VALID_TIMEFRAMES[normalized]
    
    @classmethod
    def validate_candles(cls, candles: str, default: int = DEFAULT_CANDLES) -> int:
        """Xác thực và chuẩn hóa số lượng nến."""
        if not candles:
            return default
            
        try:
            num_candles = int(candles)
        except ValueError:
            raise ValueError("Số lượng nến phải là số nguyên")
            
        if num_candles < 20:
            raise ValueError("Số lượng nến phải ít nhất là 20 để tính toán chỉ báo")
            
        if num_candles > 500:
            raise ValueError("Số lượng nến không được vượt quá 500")
            
        return num_candles
    
    @classmethod
    def validate_date_range(cls, start_date: str = None, end_date: str = None) -> tuple:
        """Xác thực và chuẩn hóa khoảng thời gian."""
        today = datetime.now()
        
        if not end_date:
            end_date = today.strftime('%Y-%m-%d')
        else:
            try:
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                if end_date_obj > today:
                    end_date = today.strftime('%Y-%m-%d')
            except ValueError:
                raise ValueError("Định dạng ngày không hợp lệ. Sử dụng YYYY-MM-DD")
                
        if not start_date:
            start_date = (today - timedelta(days=365)).strftime('%Y-%m-%d')
        else:
            try:
                start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
                end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
                
                if start_date_obj > end_date_obj:
                    raise ValueError("Ngày bắt đầu phải trước ngày kết thúc")
                
                date_diff = (end_date_obj - start_date_obj).days
                if date_diff > 365 * 5:
                    raise ValueError("Khoảng thời gian không được vượt quá 5 năm")
            except ValueError as e:
                if "Định dạng ngày không hợp lệ" not in str(e):
                    raise ValueError("Định dạng ngày không hợp lệ. Sử dụng YYYY-MM-DD")
                else:
                    raise e
                    
        return start_date, end_date
    
    @classmethod
    def validate_inputs(cls, symbol: str, timeframe: str = None, candles: str = None, 
                       start_date: str = None, end_date: str = None) -> dict:
        """Xác thực và chuẩn hóa tất cả đầu vào."""
        validated = {
            "symbol": cls.validate_symbol(symbol),
            "timeframe": cls.validate_timeframe(timeframe or DEFAULT_TIMEFRAME),
            "num_candles": cls.validate_candles(candles),
        }
        
        start, end = cls.validate_date_range(start_date, end_date)
        validated["start_date"] = start
        validated["end_date"] = end
        
        return validated

# ---------- TẢI DỮ LIỆU (NÂNG CẤP) ----------
class DataLoader:
    def __init__(self, source: str = 'vnstock'):
        self.source = source
        
    async def load_raw_data(self, symbol: str, timeframe: str, num_candles: int, 
                          start_date: str = None, end_date: str = None) -> (pd.DataFrame, str):
        """Tải dữ liệu thô từ nguồn chính, với fallback nếu cần."""
        # Xác thực đầu vào
        try:
            inputs = DataValidator.validate_inputs(symbol, timeframe, str(num_candles), start_date, end_date)
            symbol = inputs["symbol"]
            timeframe = inputs["timeframe"]
            num_candles = inputs["num_candles"]
            start_date = inputs["start_date"]
            end_date = inputs["end_date"]
        except ValueError as e:
            logger.error(f"Đầu vào không hợp lệ: {str(e)}")
            raise
            
        # Kiểm tra cache
        cache_key = f"raw_data_{self.source}_{symbol}_{timeframe}_{num_candles}_{start_date}_{end_date}"
        cached_data = await redis_manager.get(cache_key)
        if cached_data is not None:
            logger.info(f"Lấy dữ liệu từ cache cho {symbol} [{timeframe}]")
            return cached_data, "Dữ liệu từ cache"
            
        # Xác định thời gian hết hạn cache dựa trên khung thời gian
        expire_map = {
            "5m": 300,      # 5 phút
            "15m": 900,     # 15 phút
            "30m": 1800,    # 30 phút
            "1h": 3600,     # 1 giờ
            "4h": 14400,    # 4 giờ
            "1D": CACHE_EXPIRE_SHORT,     # 30 phút
            "1W": CACHE_EXPIRE_MEDIUM,    # 1 giờ
            "1M": CACHE_EXPIRE_LONG       # 1 ngày
        }
        expire = expire_map.get(timeframe, CACHE_EXPIRE_SHORT)
        
        # Thử tải từ nguồn chính (VNStock)
        try:
            df = await self._load_from_vnstock(symbol, timeframe, num_candles, start_date, end_date)
            if df is not None and not df.empty and len(df) >= 10:
                await redis_manager.set(cache_key, df, expire=expire)
                return df, "Dữ liệu từ VNStock"
        except Exception as e:
            logger.warning(f"Không thể tải dữ liệu từ VNStock cho {symbol}: {str(e)}")
        
        # Fallback sang Yahoo Finance
        try:
            df = await self._load_from_yahoo(symbol, timeframe, num_candles, start_date, end_date)
            if df is not None and not df.empty and len(df) >= 10:
                await redis_manager.set(cache_key, df, expire=expire)
                return df, "Dữ liệu từ Yahoo Finance"
            else:
                raise ValueError(f"Không đủ dữ liệu cho {symbol} từ Yahoo Finance")
        except Exception as e:
            logger.error(f"Không thể tải dữ liệu từ Yahoo Finance cho {symbol}: {str(e)}")
            raise ValueError(f"Không thể tải dữ liệu cho {symbol}: {str(e)}")
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _load_from_vnstock(self, symbol: str, timeframe: str, num_candles: int, 
                               start_date: str, end_date: str) -> pd.DataFrame:
        """Tải dữ liệu từ VNStock."""
        def fetch_vnstock():
            try:
                stock = Vnstock().stock(symbol=symbol, source='TCBS')
                # VNStock chỉ hỗ trợ khung thời gian cụ thể
                supported_timeframes = {'1D', '1W', '1M'}
                
                if timeframe not in supported_timeframes:
                    logger.warning(f"VNStock không hỗ trợ khung thời gian {timeframe}, sử dụng Yahoo Finance thay thế")
                    return None
                    
                df = stock.quote.history(start=start_date, end=end_date, interval=timeframe)
                
                if df is None or df.empty:
                    logger.warning(f"VNStock trả về dữ liệu rỗng cho {symbol}")
                    return None
                    
                df = df.rename(columns={'time': 'date', 'open': 'open', 'high': 'high',
                                      'low': 'low', 'close': 'close', 'volume': 'volume'})
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df.index = df.index.tz_localize('Asia/Bangkok')
                df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
                
                # Kiểm tra tính hợp lệ của dữ liệu
                if not (df['high'] >= df['low']).all() or not ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all():
                    logger.warning(f"Dữ liệu không hợp lệ cho {symbol} từ VNStock")
                    return None
                    
                return df.tail(num_candles)
            except Exception as e:
                logger.error(f"Lỗi khi tải dữ liệu từ VNStock: {str(e)}")
                return None
                
        return await run_in_thread(fetch_vnstock)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _load_from_yahoo(self, symbol: str, timeframe: str, num_candles: int, 
                             start_date: str, end_date: str) -> pd.DataFrame:
        """Tải dữ liệu từ Yahoo Finance."""
        # Chuyển đổi khung thời gian sang định dạng Yahoo Finance
        yahoo_timeframe_map = {
            '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '4h',
            '1D': '1d', '1W': '1wk', '1M': '1mo'
        }
        
        if timeframe not in yahoo_timeframe_map:
            raise ValueError(f"Khung thời gian {timeframe} không được hỗ trợ bởi Yahoo Finance")
            
        yahoo_interval = yahoo_timeframe_map[timeframe]
        
        # Xác định symbol cho Yahoo Finance
        yahoo_symbol = f"{symbol}.VN" if not is_index(symbol) else symbol
        
        try:
            async with aiohttp.ClientSession() as session:
                start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
                end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
                
                url = (f"https://query1.finance.yahoo.com/v7/finance/download/{yahoo_symbol}"
                     f"?period1={start_ts}&period2={end_ts}&interval={yahoo_interval}&events=history")
                     
                async with asyncio.wait_for(session.get(url), timeout=15) as response:
                    if response.status != 200:
                        raise ValueError(f"Không thể tải dữ liệu từ Yahoo, HTTP {response.status}")
                        
                    text = await response.text()
                    df = pd.read_csv(io.StringIO(text))
                    
                    if df.empty:
                        raise ValueError("Dữ liệu Yahoo rỗng")
                        
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                          'Close': 'close', 'Volume': 'volume'})
                    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
                    
                    # Kiểm tra tính hợp lệ của dữ liệu
                    if not (df['high'] >= df['low']).all() or not ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all():
                        logger.warning(f"Dữ liệu không hợp lệ cho {symbol} từ Yahoo Finance")
                        raise ValueError(f"Dữ liệu không hợp lệ cho {symbol}")
                        
                    return df.tail(num_candles)
        except asyncio.TimeoutError:
            logger.error("Timeout khi tải dữ liệu từ Yahoo Finance.")
            raise
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu Yahoo: {str(e)}")
            raise
            
    async def detect_and_handle_outliers(self, df: pd.DataFrame, method: str = 'z-score', 
                                       threshold: float = 3.0, handle: str = 'flag') -> (pd.DataFrame, str):
        """Phát hiện và xử lý dữ liệu ngoại lai."""
        if df is None or df.empty or 'close' not in df.columns:
            return df, "Không đủ dữ liệu để phát hiện outlier"
            
        df = df.copy()
        outlier_report = "Không có giá trị bất thường"
        
        if method == 'z-score':
            z_scores = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
            df['is_outlier'] = z_scores > threshold
        elif method == 'iqr':
            q1 = df['close'].quantile(0.25)
            q3 = df['close'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            df['is_outlier'] = (df['close'] < lower_bound) | (df['close'] > upper_bound)
        else:
            raise ValueError(f"Phương pháp không hỗ trợ: {method}")
            
        outliers = df[df['is_outlier']]
        
        if not outliers.empty:
            outlier_report = f"Phát hiện {len(outliers)} giá trị bất thường trong dữ liệu:\n"
            for idx, row in outliers.iterrows():
                outlier_report += f"- {idx.strftime('%Y-%m-%d')}: {row['close']:.2f}\n"
                
            # Xử lý ngoại lai theo yêu cầu
            if handle == 'remove':
                df = df[~df['is_outlier']]
                outlier_report += "Đã loại bỏ các giá trị ngoại lai.\n"
            elif handle == 'interpolate':
                # Lưu chỉ số ngoại lai
                outlier_indices = df[df['is_outlier']].index
                
                # Nội suy giá trị
                for col in ['open', 'high', 'low', 'close']:
                    df.loc[outlier_indices, col] = df[col].interpolate(method='time')
                
                df.loc[outlier_indices, 'is_outlier'] = False
                outlier_report += "Đã nội suy các giá trị ngoại lai.\n"
            elif handle == 'flag':
                # Chỉ đánh dấu, không thay đổi
                outlier_report += "Các giá trị ngoại lai đã được đánh dấu nhưng không thay đổi.\n"
                
        return df, outlier_report
        
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa DataFrame để đảm bảo định dạng nhất quán."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Đảm bảo tên cột nhất quán
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Kiểm tra xem có tất cả cột cần thiết không
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Thiếu các cột: {', '.join(missing_columns)}")
            
        # Loại bỏ hàng có giá trị NaN
        df = df[required_columns].dropna()
        
        # Kiểm tra tính hợp lệ của dữ liệu giá
        if not (df['high'] >= df['low']).all():
            invalid_rows = df[~(df['high'] >= df['low'])]
            logger.warning(f"Có {len(invalid_rows)} hàng với giá cao thấp hơn giá thấp")
            # Sửa lỗi bằng cách hoán đổi giá trị
            df.loc[~(df['high'] >= df['low']), ['high', 'low']] = df.loc[~(df['high'] >= df['low']), ['low', 'high']].values
            
        # Kiểm tra giá đóng cửa nằm trong khoảng cao-thấp
        invalid_close = ~((df['close'] >= df['low']) & (df['close'] <= df['high']))
        if invalid_close.any():
            logger.warning(f"Có {invalid_close.sum()} hàng với giá đóng cửa nằm ngoài khoảng cao-thấp")
            # Sửa giá đóng cửa để nằm trong khoảng
            df.loc[invalid_close, 'close'] = df.loc[invalid_close].apply(
                lambda row: min(row['high'], max(row['low'], row['close'])), axis=1
            )
            
        # Đảm bảo khối lượng không âm
        if (df['volume'] < 0).any():
            logger.warning("Phát hiện khối lượng âm, đang chuyển thành giá trị tuyệt đối")
            df['volume'] = df['volume'].abs()
            
        return df
        
    def align_timestamps(self, df: pd.DataFrame, timezone: str = 'Asia/Bangkok') -> pd.DataFrame:
        """Căn chỉnh timestamp để đảm bảo nhất quán múi giờ."""
        if df is None or df.empty:
            return df
            
        df = df.copy()
        
        # Xử lý index là timestamp
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index không phải là DatetimeIndex, đang chuyển đổi")
            df.index = pd.to_datetime(df.index)
            
        # Đảm bảo múi giờ nhất quán
        if df.index.tz is None:
            df.index = df.index.tz_localize(timezone)
        elif str(df.index.tz) != timezone:
            logger.info(f"Chuyển đổi múi giờ từ {df.index.tz} sang {timezone}")
            df.index = df.index.tz_convert(timezone)
            
        # Sắp xếp theo thời gian tăng dần
        df = df.sort_index()
        
        return df
        
    def calculate_trading_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lọc ngày giao dịch thực tế."""
        return filter_trading_days(df)
        
    async def clean_data(self, df: pd.DataFrame, detect_outliers: bool = True, 
                        filter_days: bool = True) -> (pd.DataFrame, str):
        """Làm sạch dữ liệu bằng cách áp dụng tất cả các bước tiền xử lý."""
        if df is None or df.empty:
            return df, "Dữ liệu rỗng"
            
        # Chuẩn hóa DataFrame
        try:
            df = self.normalize_dataframe(df)
        except ValueError as e:
            logger.error(f"Lỗi chuẩn hóa DataFrame: {str(e)}")
            return df, f"Lỗi chuẩn hóa dữ liệu: {str(e)}"
            
        # Căn chỉnh timestamp
        df = self.align_timestamps(df)
        
        # Lọc ngày giao dịch nếu cần
        if filter_days:
            df = self.calculate_trading_days(df)
            
        # Phát hiện và xử lý outlier nếu cần
        if detect_outliers:
            df, outlier_report = await self.detect_and_handle_outliers(df, handle='flag')
        else:
            outlier_report = "Bỏ qua phát hiện outlier theo yêu cầu"
            
        return df, outlier_report
        
    async def load_data(self, symbol: str, timeframe: str, num_candles: int) -> (pd.DataFrame, str):
        """Phương thức load_data nâng cấp, sử dụng luồng mới."""
        # Tải dữ liệu thô
        raw_df, source_info = await self.load_raw_data(
            symbol, 
            timeframe, 
            num_candles,
            (datetime.now() - timedelta(days=num_candles * 3)).strftime('%Y-%m-%d'),
            datetime.now().strftime('%Y-%m-%d')
        )
        
        # Làm sạch dữ liệu
        clean_df, outlier_report = await self.clean_data(raw_df)
        
        return clean_df, outlier_report

    def detect_outliers(self, df: pd.DataFrame) -> (pd.DataFrame, str):
        """Phương thức cũ để tương thích, sử dụng phương thức mới."""
        loop = asyncio.get_event_loop()
        df_result, report = loop.run_until_complete(self.detect_and_handle_outliers(df))
        return df_result, report

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def fetch_fundamental_data_vnstock(self, symbol: str) -> dict:
        cache_key = f"fundamental_vnstock_{symbol}"
        cached_data = await redis_manager.get(cache_key)
        if cached_data is not None:
            return cached_data

        def fetch():
            stock = Vnstock().stock(symbol=symbol, source='TCBS')
            fundamental_data = {}
            ratios = stock.finance.ratio()
            if ratios is not None and not ratios.empty:
                fundamental_data.update(ratios.iloc[-1].to_dict())
            if hasattr(stock.finance, 'valuation'):
                valuation = stock.finance.valuation()
                if valuation is not None and not valuation.empty:
                    fundamental_data.update(valuation.iloc[-1].to_dict())
            if not fundamental_data:
                raise ValueError("Không có dữ liệu cơ bản từ VNStock")
            return fundamental_data

        try:
            fundamental_data = await run_in_thread(fetch)
            await redis_manager.set(cache_key, fundamental_data, expire=86400)
            return fundamental_data
        except Exception as e:
            logger.error(f"Lỗi lấy dữ liệu cơ bản từ VNStock: {str(e)}")
            return {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def fetch_fundamental_data_yahoo(self, symbol: str) -> dict:
        cache_key = f"fundamental_yahoo_{symbol}"
        cached_data = await redis_manager.get(cache_key)
        if cached_data is not None:
            return cached_data

        def fetch():
            stock = yf.Ticker(f"{symbol}.VN")
            info = stock.info
            fundamental_data = {
                'EPS': info.get('trailingEps', None),
                'P/E': info.get('trailingPE', None),
                'P/B': info.get('priceToBook', None),
                'ROE': info.get('returnOnEquity', None),
                'Dividend Yield': info.get('dividendYield', None),
                'Market Cap': info.get('marketCap', None),
                'BVPS': info.get('bookValue', None)
            }
            if all(v is None for v in fundamental_data.values()):
                raise ValueError("Không có dữ liệu cơ bản từ Yahoo Finance")
            return fundamental_data

        try:
            fundamental_data = await run_in_thread(fetch)
            await redis_manager.set(cache_key, fundamental_data, expire=86400)
            return fundamental_data
        except Exception as e:
            logger.error(f"Lỗi lấy dữ liệu cơ bản từ Yahoo: {str(e)}")
            return {}

    async def get_fundamental_data(self, symbol: str) -> dict:
        if is_index(symbol):
            return {"error": f"{symbol} là chỉ số, không có dữ liệu cơ bản"}
        fundamental_data = await self.fetch_fundamental_data_vnstock(symbol)
        if fundamental_data and any(v is not None for v in fundamental_data.values()):
            return fundamental_data
        fundamental_data = await self.fetch_fundamental_data_yahoo(symbol)
        if fundamental_data and any(v is not None for v in fundamental_data.values()):
            return fundamental_data
        return {"error": f"Không có dữ liệu cơ bản cho {symbol}"}

# ---------- PHÂN TÍCH KỸ THUẬT ----------
class TechnicalAnalyzer:
    @staticmethod
    def _calculate_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("DataFrame rỗng, không thể tính toán chỉ báo")
        if 'close' not in df.columns:
            raise ValueError("Dữ liệu không có cột 'close' cần thiết để tính toán chỉ báo")
        if len(df) < 20:
            raise ValueError("Không đủ dữ liệu để tính toán SMA20 (cần ít nhất 20 nến)")
        try:
            df['sma20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['sma50'] = trend.SMAIndicator(df['close'], window=50).sma_indicator()
            df['sma200'] = trend.SMAIndicator(df['close'], window=200).sma_indicator() if len(df) >= 200 else np.nan
            df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi() if len(df) >= 14 else np.nan
            macd = trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['signal'] = macd.macd_signal()
            bb = volatility.BollingerBands(df['close'])
            df['bb_high'] = bb.bollinger_hband()
            df['bb_low'] = bb.bollinger_lband()
            ichimoku = trend.IchimokuIndicator(df['high'], df['low'])
            df['ichimoku_a'] = ichimoku.ichimoku_a()
            df['ichimoku_b'] = ichimoku.ichimoku_b()
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=14).money_flow_index()
            ichimoku_short = trend.IchimokuIndicator(df['high'], df['low'], window1=9, window2=26)
            df['tenkan_sen'] = ichimoku_short.ichimoku_a()
            ichimoku_medium = trend.IchimokuIndicator(df['high'], df['low'], window1=26, window2=52)
            df['kijun_sen'] = ichimoku_medium.ichimoku_b()
            df['chikou_span'] = df['close'].shift(-26) if len(df) > 26 else np.nan
            high_price = df['high'].max()
            low_price = df['low'].min()
            diff = high_price - low_price
            df['fib_0.0'] = high_price
            df['fib_23.6'] = high_price - 0.236 * diff
            df['fib_38.2'] = high_price - 0.382 * diff
            df['fib_50.0'] = high_price - 0.5 * diff
            df['fib_61.8'] = high_price - 0.618 * diff
            df['fib_100.0'] = low_price
            return df
        except Exception as e:
            logger.error(f"Lỗi tính toán chỉ số kỹ thuật: {str(e)}")
            raise

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._calculate_common_indicators(df)

    def calculate_multi_timeframe_indicators(self, dfs: dict) -> dict:
        indicators = {}
        for timeframe, df in dfs.items():
            try:
                df_processed = self._calculate_common_indicators(df)
                indicators[timeframe] = df_processed.tail(1).to_dict(orient='records')[0]
            except Exception as e:
                logger.error(f"Lỗi tính toán cho khung {timeframe}: {str(e)}")
                indicators[timeframe] = {}
        return indicators

# ---------- THU THẬP TIN TỨC (SỬA LỖI) ----------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def fetch_rss_feed(url: str) -> str:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    raise ValueError(f"HTTP {response.status} khi lấy RSS từ {url}")
    except asyncio.TimeoutError:
        logger.error(f"Timeout khi lấy RSS từ {url}")
    except Exception as e:
        logger.error(f"Lỗi lấy RSS từ {url}: {str(e)}")
    return None

async def get_news(symbol: str = None, limit: int = 3) -> list:
    cache_key = f"news_{symbol}_{limit}" if symbol else f"news_market_{limit}"
    cached_news = await redis_manager.get(cache_key)
    if cached_news is not None:
        return cached_news

    rss_urls = [
        "https://cafef.vn/thi-truong-chung-khoan.rss",
        "https://cafef.vn/smart-money.rss",
        "https://cafef.vn/tai-chinh-ngan-hang.rss",
        "https://cafef.vn/doanh-nghiep.rss",
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"
    ]

    market_keywords = ["market", "stock", "index", "economy", "thị trường", "chứng khoán", "lãi suất", "vnindex"]
    symbol_keywords = [symbol.lower(), f"{symbol.lower()} ", f"mã {symbol.lower()}"] if symbol else []

    news_list = []
    tasks = [fetch_rss_feed(url) for url in rss_urls]
    rss_results = await asyncio.gather(*tasks)
    for rss_text in rss_results:
        if rss_text is None:
            continue
        feed = parse_rss_content(rss_text)
        if not feed or not feed.entries:
            continue
        for entry in feed.entries[:2]:
            title = entry.get("title", "").strip()
            link = entry.get("link", "")
            summary = entry.get("summary", entry.get("description", "")).strip()
            pub_date = entry.get("published", datetime.now().isoformat())
            content = f"{title} {summary}".lower()
            match = False
            if symbol and any(kw in content for kw in symbol_keywords):
                match = True
            elif not symbol and any(kw in content for kw in market_keywords):
                match = True
            if match:
                news_list.append({
                    "title": title,
                    "link": link,
                    "summary": summary[:200] + "..." if len(summary) > 200 else summary,
                    "published": pub_date
                })
    unique_news = sorted(news_list, key=lambda x: x.get("published", ""), reverse=True)[:limit]
    result = unique_news if unique_news else [{"title": "⚠️ Không có tin tức", "link": "#", "summary": ""}]
    await redis_manager.set(cache_key, result, expire=NEWS_CACHE_EXPIRE)
    return result

def parse_rss_content(rss_text: str):
    try:
        return feedparser.parse(rss_text) if rss_text else None
    except Exception as e:
        logger.error(f"Lỗi phân tích RSS: {str(e)}")
        return None

# ---------- PHÂN TÍCH CƠ BẢN ----------
def deep_fundamental_analysis(fundamental_data: dict) -> str:
    report = "📊 **Phân tích cơ bản**:\n"
    if not fundamental_data or 'error' in fundamental_data:
        return report + f"❌ {fundamental_data.get('error', 'Không có dữ liệu')}\n"

    keys_mapping = {
        'EPS': ['EPS', 'eps', 'Earning Per Share'],
        'P/E': ['P/E', 'PE', 'pe', 'Price to Earning', 'trailingPE'],
        'P/B': ['P/B', 'PB', 'pb', 'Price to Book', 'priceToBook'],
        'ROE': ['ROE', 'roe', 'Return on Equity', 'returnOnEquity'],
        'Dividend Yield': ['Dividend Yield', 'DY', 'dividend_yield', 'dividendYield'],
        'BVPS': ['BVPS', 'Book Value Per Share', 'bookValue'],
        'Market Cap': ['Market Cap', 'market_cap', 'marketCap']
    }

    extracted = {}
    for standard_key, possible_keys in keys_mapping.items():
        for key in possible_keys:
            if key in fundamental_data and fundamental_data[key] is not None:
                extracted[standard_key] = fundamental_data[key]
                break

    for key, value in extracted.items():
        report += f"- **{key}**: {value:.2f}\n" if isinstance(value, (int, float)) else f"- **{key}**: {value}\n"

    if 'P/E' in extracted and isinstance(extracted['P/E'], (int, float)):
        pe = extracted['P/E']
        report += "- **P/E**: Cổ phiếu có thể định giá thấp\n" if pe < 10 else "- **P/E**: Định giá cao\n" if pe > 20 else "- **P/E**: Định giá hợp lý\n"

    if 'ROE' in extracted and isinstance(extracted['ROE'], (int, float)):
        report += "- **ROE**: Hiệu quả sử dụng vốn tốt\n" if extracted['ROE'] > 15 else "- **ROE**: Cần cải thiện\n"

    if 'Dividend Yield' in extracted and isinstance(extracted['Dividend Yield'], (int, float)):
        report += "- **Dividend**: Hấp dẫn\n" if extracted['Dividend Yield'] > 5 else "- **Dividend**: Trung bình\n"

    return report

# ---------- HUẤN LUYỆN VÀ LƯU MÔ HÌNH ----------
def prepare_data_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame rỗng, không thể dự báo")
    df_reset = df.reset_index()
    df_reset['date'] = df_reset['date'].dt.tz_localize(None)
    df_reset = df_reset.rename(columns={'date': 'ds', 'close': 'y'})
    return df_reset[['ds', 'y']]

def get_vietnam_holidays(years) -> pd.DataFrame:
    holiday_list = []
    for year in years:
        vn_holidays = holidays.Vietnam(years=year)
        for date, name in vn_holidays.items():
            holiday_list.append({'ds': pd.to_datetime(date), 'holiday': name})
    holiday_df = pd.DataFrame(holiday_list)
    holiday_df['ds'] = holiday_df['ds'].dt.tz_localize(None)
    return holiday_df

def forecast_with_prophet(df: pd.DataFrame, periods: int = 7) -> (pd.DataFrame, Prophet):
    try:
        data = prepare_data_for_prophet(df)
        if data.empty:
            raise ValueError("Không đủ dữ liệu để dự báo Prophet.")
        current_year = datetime.now().year
        holiday_df = get_vietnam_holidays(range(current_year-1, current_year+2))
        model = Prophet(holidays=holiday_df)
        model.fit(data)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        return forecast, model
    except Exception as e:
        logger.error(f"Lỗi dự báo Prophet: {str(e)}")
        raise

def evaluate_prophet_performance(df: pd.DataFrame, forecast: pd.DataFrame) -> float:
    actual = df['close'].iloc[-len(forecast):].values
    predicted = forecast['yhat'].values[:len(actual)]
    if len(actual) < 1 or len(predicted) < 1:
        return 0.0
    mse = np.mean((actual - predicted) ** 2)
    return 1 / (1 + mse)

def predict_xgboost_signal(df: pd.DataFrame, features: list) -> (int, float):
    if df is None or df.empty:
        return "Dữ liệu rỗng", 0.0
    df = df.copy()
    df['target'] = (df['close'] > df['close'].shift(1)).astype(int)
    X = df[features].shift(1)
    y = df['target']
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    if len(X) < 100:
        return "Không đủ dữ liệu", 0.0
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    X_pred = X.iloc[-1:]
    y_test = y.iloc[-1:]
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_pred)[0]
    accuracy = accuracy_score(y_test, [pred])
    return pred, accuracy

def train_prophet_model(df: pd.DataFrame) -> (Prophet, float):
    data = prepare_data_for_prophet(df)
    if data.empty:
        raise ValueError("Không đủ dữ liệu để huấn luyện Prophet")
    current_year = datetime.now().year
    holiday_df = get_vietnam_holidays(range(current_year-1, current_year+2))
    model = Prophet(holidays=holiday_df)
    model.fit(data)
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    performance = evaluate_prophet_performance(df, forecast)
    return model, performance

def train_xgboost_model(df: pd.DataFrame, features: list) -> (xgb.XGBClassifier, float):
    if df is None or df.empty:
        raise ValueError("DataFrame rỗng, không thể huấn luyện XGBoost")
    df = df.copy()
    df['target'] = (df['close'] > df['close'].shift(1)).astype(int)
    X = df[features].shift(1)
    y = df['target']
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    if len(X) < 100:
        raise ValueError("Không đủ dữ liệu để huấn luyện XGBoost")
    X_train = X.iloc[:-1]
    y_train = y.iloc[:-1]
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    performance = accuracy_score(y_train, y_pred)
    return model, performance

# ---------- AUTO TRAINING ----------
async def get_training_symbols() -> list:
    try:
        async with SessionLocal() as session:
            result = await session.execute(select(ReportHistory.symbol).distinct())
            symbols = [row[0] for row in result.fetchall()]
            logger.info(f"Các mã được lấy từ lịch sử báo cáo: {symbols}")
            return symbols
    except Exception as e:
        logger.error(f"Lỗi truy vấn symbols từ ReportHistory: {str(e)}")
        return []

async def train_models_for_symbol(symbol: str):
    try:
        logger.info(f"Bắt đầu auto training cho mã: {symbol}")
        loader = DataLoader()
        tech_analyzer = TechnicalAnalyzer()
        df, _ = await loader.load_data(symbol, '1D', 500)
        df = tech_analyzer.calculate_indicators(df)
        features = ['sma20', 'sma50', 'sma200', 'rsi', 'macd', 'signal',
                    'bb_high', 'bb_low', 'ichimoku_a', 'ichimoku_b', 'vwap', 'mfi']
        task_prophet = asyncio.to_thread(train_prophet_model, df)
        task_xgb = asyncio.to_thread(train_xgboost_model, df, features)
        (prophet_model, prophet_perf), (xgb_model, xgb_perf) = await asyncio.gather(task_prophet, task_xgb)
        await model_db_manager.store_trained_model(symbol, 'prophet', prophet_model, prophet_perf)
        await model_db_manager.store_trained_model(symbol, 'xgboost', xgb_model, xgb_perf)
        logger.info(f"Auto training cho {symbol} hoàn tất.")
    except Exception as e:
        logger.error(f"Lỗi auto training cho {symbol}: {str(e)}")

async def auto_train_models():
    try:
        symbols = await get_training_symbols()
        if not symbols:
            logger.info("Không có mã nào trong ReportHistory, bỏ qua auto training.")
            return
        tasks = [train_models_for_symbol(symbol) for symbol in symbols]
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Lỗi auto training: {str(e)}")

# ---------- AI VÀ BÁO CÁO ----------
class AIAnalyzer:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_content(self, prompt):
        return await self.model.generate_content_async(prompt)

    async def load_report_history(self, symbol: str) -> list:
        return await db.load_report_history(symbol)

    async def save_report_history(self, symbol: str, report: str, close_today: float, close_yesterday: float) -> None:
        await db.save_report_history(symbol, report, close_today, close_yesterday)

    def analyze_price_action(self, df: pd.DataFrame) -> str:
        if len(df) < 6:
            return "Không đủ dữ liệu để phân tích."
        last_5_days = df['close'].tail(5)
        changes = last_5_days.pct_change().dropna()
        trend_summary = []
        for i, change in enumerate(changes):
            date = last_5_days.index[i+1].strftime('%Y-%m-%d')
            if df.loc[last_5_days.index[i+1], 'is_outlier']:
                outlier_note = " (⚠️ outlier)"
            else:
                outlier_note = ""
            if change > 0:
                trend_summary.append(f"{date}: Tăng {change*100:.2f}%{outlier_note}")
            elif change < 0:
                trend_summary.append(f"{date}: Giảm {-change*100:.2f}%{outlier_note}")
            else:
                trend_summary.append(f"{date}: Không đổi{outlier_note}")
        consecutive_up = consecutive_down = 0
        for change in changes[::-1]:
            if change > 0:
                consecutive_up += 1
                consecutive_down = 0
            elif change < 0:
                consecutive_down += 1
                consecutive_up = 0
            else:
                break
        if consecutive_up >= 3:
            summary = f"✅ Giá tăng {consecutive_up} phiên liên tiếp.\n"
        elif consecutive_down >= 3:
            summary = f"⚠️ Giá giảm {consecutive_down} phiên liên tiếp.\n"
        else:
            summary = "🔍 Xu hướng chưa rõ.\n"
        summary += "\n".join(trend_summary)
        return summary

    async def analyze_with_openrouter(self, technical_data):
        if not OPENROUTER_API_KEY:
            raise Exception("Chưa có OPENROUTER_API_KEY")

        prompt = (
            "Bạn là chuyên gia phân tích kỹ thuật chứng khoán. "
            "Nhiệm vụ của bạn là phân tích dữ liệu và trả về JSON hợp lệ theo định dạng cụ thể.\n\n"
            "Dựa trên dữ liệu dưới đây, hãy nhận diện các mẫu hình nến như Doji, Hammer, Shooting Star, Engulfing, "
            "sóng Elliott, mô hình Wyckoff, và các vùng hỗ trợ/kháng cự.\n\n"
            "⚠️ QUAN TRỌNG: Chỉ trả về kết quả ở dạng JSON nghiêm ngặt như mẫu dưới đây, KHÔNG thêm bất kỳ văn bản nào khác trước hoặc sau JSON:\n"
            "```json\n"
            "{\n"
            "  \"support_levels\": [giá1, giá2, ...],\n"
            "  \"resistance_levels\": [giá1, giá2, ...],\n"
            "  \"patterns\": [\n"
            "    {\"name\": \"tên mẫu hình\", \"description\": \"giải thích ngắn\"},\n"
            "    ...\n"
            "  ]\n"
            "}\n"
            "```\n\n"
            "Đảm bảo:\n"
            "1. Tất cả giá trị trong 'support_levels' và 'resistance_levels' là số (không đặt trong dấu ngoặc kép)\n"
            "2. Trường 'patterns' phải luôn là một mảng, ngay cả khi trống ([])\n"
            "3. Mỗi phần tử trong 'patterns' phải có đủ 'name' và 'description'\n"
            "4. Đảm bảo không có dấu phẩy thừa ở cuối mảng hoặc đối tượng\n"
            "5. KHÔNG có văn bản nào ở bên ngoài JSON\n\n"
            f"Dữ liệu phân tích:\n{json.dumps(technical_data, ensure_ascii=False, indent=2)}"
        )

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourapp.com",
            "X-Title": "Stock Analysis Bot"
        }
        payload = {
            "model": "deepseek/deepseek-chat-v3-0324:free",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.2,
            "response_format": {"type": "json_object"}
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30) as resp:
                    if resp.status != 200:
                        logger.error(f"Lỗi OpenRouter API, mã trạng thái: {resp.status}")
                        return {"support_levels": [], "resistance_levels": [], "patterns": []}
                        
                    text = await resp.text()
                    if not text or text.isspace():
                        logger.error("OpenRouter trả về phản hồi trống")
                        return {"support_levels": [], "resistance_levels": [], "patterns": []}
                        
                    try:
                        result = json.loads(text)
                        if 'choices' not in result or not result['choices'] or 'message' not in result['choices'][0]:
                            logger.error(f"Cấu trúc phản hồi từ OpenRouter không hợp lệ: {text}")
                            return {"support_levels": [], "resistance_levels": [], "patterns": []}
                            
                        content = result['choices'][0]['message']['content']
                        if not content or content.isspace():
                            logger.error("Nội dung phản hồi OpenRouter trống")
                            return {"support_levels": [], "resistance_levels": [], "patterns": []}
                            
                        # Xử lý các trường hợp nội dung có thể chứa markdown hoặc không phải JSON thuần túy
                        content = self._extract_json_from_content(content)
                            
                        try:
                            parsed_content = json.loads(content)
                            # Đảm bảo các khóa dự kiến tồn tại
                            if not isinstance(parsed_content, dict):
                                raise json.JSONDecodeError("Phản hồi không phải là đối tượng JSON", content, 0)
                                
                            result_dict = {
                                "support_levels": self._ensure_list_of_numbers(parsed_content.get("support_levels", [])),
                                "resistance_levels": self._ensure_list_of_numbers(parsed_content.get("resistance_levels", [])),
                                "patterns": self._validate_patterns_format(parsed_content.get("patterns", []))
                            }
                            return result_dict
                        except json.JSONDecodeError:
                            logger.error(f"Nội dung không thể phân tích thành JSON: {content}")
                            return {"support_levels": [], "resistance_levels": [], "patterns": []}
                    except json.JSONDecodeError:
                        logger.error(f"Phản hồi không hợp lệ từ OpenRouter: {text}")
                        return {"support_levels": [], "resistance_levels": [], "patterns": []}
                    except KeyError as e:
                        logger.error(f"Phản hồi thiếu trường cần thiết: {e}")
                        return {"support_levels": [], "resistance_levels": [], "patterns": []}
        except Exception as e:
            logger.error(f"Lỗi kết nối OpenRouter: {str(e)}")
            return {"support_levels": [], "resistance_levels": [], "patterns": []}
            
    def _extract_json_from_content(self, content: str) -> str:
        """Trích xuất JSON từ nội dung có thể chứa markdown hoặc text thừa."""
        # Tìm JSON giữa ```json và ```
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            return json_match.group(1).strip()
            
        # Tìm bất kỳ JSON nào trong nội dung
        json_match = re.search(r'\{\s*"[^"]+"\s*:', content)
        if json_match:
            # Tìm đoạn văn bản từ vị trí bắt đầu của JSON đến hết
            start_idx = json_match.start()
            json_content = content[start_idx:]
            # Đếm số lượng dấu { và }
            open_braces = 0
            close_braces = 0
            end_idx = len(json_content)
            
            for i, char in enumerate(json_content):
                if char == '{':
                    open_braces += 1
                elif char == '}':
                    close_braces += 1
                    if open_braces == close_braces:
                        end_idx = i + 1
                        break
                        
            return json_content[:end_idx]
            
        # Trả về nguyên văn nếu không tìm thấy mẫu JSON
        return content
        
    def _ensure_list_of_numbers(self, values) -> list:
        """Đảm bảo giá trị là danh sách các số."""
        if not isinstance(values, list):
            return []
            
        result = []
        for val in values:
            try:
                if isinstance(val, (int, float)):
                    result.append(val)
                elif isinstance(val, str):
                    # Cố gắng chuyển đổi chuỗi thành số
                    result.append(float(val.replace(',', '.')))
            except (ValueError, TypeError):
                # Bỏ qua các giá trị không thể chuyển thành số
                pass
                
        return result
        
    def _validate_patterns_format(self, patterns) -> list:
        """Xác thực và chuẩn hóa định dạng của mảng patterns."""
        if not isinstance(patterns, list):
            return []
            
        result = []
        for pattern in patterns:
            if isinstance(pattern, dict) and 'name' in pattern:
                # Đảm bảo có mô tả
                if 'description' not in pattern:
                    pattern['description'] = ""
                    
                # Chỉ giữ lại các trường cần thiết
                result.append({
                    'name': str(pattern['name']),
                    'description': str(pattern['description'])
                })
                
        return result
        
    async def generate_report(self, dfs: dict, symbol: str, fundamental_data: dict, outlier_reports: dict) -> str:
        try:
            tech_analyzer = TechnicalAnalyzer()
            indicators = tech_analyzer.calculate_multi_timeframe_indicators(dfs)
            news = await get_news(symbol=symbol)
            news_text = "\n".join([f"📰 **{n['title']}**\n🔗 {n['link']}\n📝 {n['summary']}" for n in news])
            df_1d = dfs.get('1D')
            close_today = df_1d['close'].iloc[-1]
            close_yesterday = df_1d['close'].iloc[-2]
            price_action = self.analyze_price_action(df_1d)
            history = await self.load_report_history(symbol)
            past_report = ""
            if history:
                last = history[-1]
                past_result = "đúng" if (close_today > last["close_today"] and "mua" in last["report"].lower()) else "sai"
                past_report = f"📜 **Báo cáo trước** ({last['date']}): {last['close_today']} → {close_today} ({past_result})\n"
            fundamental_report = deep_fundamental_analysis(fundamental_data)

            # Phân tích với OpenRouter
            technical_data = {
                "candlestick_data": df_1d.tail(50).to_dict(orient="records"),
                "technical_indicators": indicators['1D']
            }
            openrouter_result = await self.analyze_with_openrouter(technical_data)
            support_levels = openrouter_result.get('support_levels', [])
            resistance_levels = openrouter_result.get('resistance_levels', [])
            patterns = openrouter_result.get('patterns', [])

            forecast, prophet_model = forecast_with_prophet(df_1d, periods=7)
            prophet_perf = evaluate_prophet_performance(df_1d, forecast)
            future_forecast = forecast[forecast['ds'] > df_1d.index[-1].tz_localize(None)]
            if not future_forecast.empty:
                next_day_pred = future_forecast.iloc[0]
                day7_pred = future_forecast.iloc[6] if len(future_forecast) >= 7 else future_forecast.iloc[-1]
            else:
                next_day_pred = forecast.iloc[-1]
                day7_pred = forecast.iloc[-1]

            forecast_summary = f"**Dự báo giá (Prophet)** (Hiệu suất: {prophet_perf:.2f}):\n"
            forecast_summary += f"- Ngày tiếp theo ({next_day_pred['ds'].strftime('%d/%m/%Y')}): {next_day_pred['yhat']:.2f}\n"
            forecast_summary += f"- Sau 7 ngày ({day7_pred['ds'].strftime('%d/%m/%Y')}): {day7_pred['yhat']:.2f}\n"

            features = ['sma20', 'sma50', 'sma200', 'rsi', 'macd', 'signal', 'bb_high', 'bb_low', 'ichimoku_a', 'ichimoku_b', 'vwap', 'mfi']
            xgb_signal, xgb_perf = predict_xgboost_signal(df_1d.copy(), features)
            if isinstance(xgb_signal, int):
                xgb_text = "Tăng" if xgb_signal == 1 else "Giảm"
            else:
                xgb_text = xgb_signal
            xgb_summary = f"**XGBoost dự đoán tín hiệu giao dịch** (Hiệu suất: {xgb_perf:.2f}): {xgb_text}\n"

            outlier_text = "\n".join([f"**{tf}**: {report}" for tf, report in outlier_reports.items()])

            prompt = f"""
Bạn là chuyên gia phân tích kỹ thuật và cơ bản, trader chuyên nghiệp, chuyên gia bắt đáy 30 năm kinh nghiệm ở chứng khoán Việt Nam. Hãy viết báo cáo chi tiết cho {symbol}:

**Thông tin cơ bản:**
- Ngày: {datetime.now().strftime('%d/%m/%Y')}
- Giá hôm qua: {close_yesterday:.2f}
- Giá hôm nay: {close_today:.2f}

**Hành động giá:**
{price_action}

**Lịch sử dự đoán:**
{past_report}

**Chất lượng dữ liệu:**
{outlier_text}

**Chỉ số kỹ thuật:**
"""
            for tf, ind in indicators.items():
                prompt += f"\n--- {tf} ---\n"
                prompt += f"- Close: {ind.get('close', 0):.2f}\n"
                prompt += f"- SMA20: {ind.get('sma20', 0):.2f}, SMA50: {ind.get('sma50', 0):.2f}, SMA200: {ind.get('sma200', 0):.2f}\n"
                prompt += f"- RSI: {ind.get('rsi', 0):.2f}\n"
                prompt += f"- MACD: {ind.get('macd', 0):.2f} (Signal: {ind.get('signal', 0):.2f})\n"
                prompt += f"- Bollinger: {ind.get('bb_low', 0):.2f} - {ind.get('bb_high', 0):.2f}\n"
                prompt += f"- Ichimoku: A: {ind.get('ichimoku_a', 0):.2f}, B: {ind.get('ichimoku_b', 0):.2f}\n"
                prompt += f"- Fibonacci: 0.0: {ind.get('fib_0.0', 0):.2f}, 61.8: {ind.get('fib_61.8', 0):.2f}\n"
            prompt += f"\n**Cơ bản:**\n{fundamental_report}\n"
            prompt += f"\n**Tin tức:**\n{news_text}\n"
            prompt += f"\n**Phân tích từ OpenRouter:**\n"
            prompt += f"- Hỗ trợ: {', '.join(map(str, support_levels))}\n"
            prompt += f"- Kháng cự: {', '.join(map(str, resistance_levels))}\n"
            prompt += f"- Mẫu hình: {', '.join([p['name'] for p in patterns])}\n"
            prompt += f"\n{xgb_summary}\n"
            prompt += f"{forecast_summary}\n"
            prompt += """
**Yêu cầu:**
1. So sánh giá/ chỉ số phiên hiện tại và phiên trước đó.
2. Phân tích đa khung thời gian, xu hướng ngắn hạn, trung hạn, dài hạn.
3. Đánh giá các chỉ số kỹ thuật, động lực thị trường.
4. Xác định hỗ trợ/kháng cự từ OpenRouter. Đưa ra kịch bản và xác suất % (tăng, giảm, sideway).
5. Đề xuất MUA/BÁN/NẮM GIỮ với % tin cậy, điểm vào, cắt lỗ, chốt lời. Phương án đi vốn, phân bổ tỷ trọng cụ thể.
6. Đánh giá rủi ro và tỷ lệ risk/reward.
7. Kết hợp tin tức, phân tích kỹ thuật, cơ bản và kết quả từ OpenRouter để đưa ra nhận định.
8. Không cần theo form cố định, trình bày logic, súc tích nhưng đủ thông tin để hành động và sáng tạo với emoji.

**Hướng dẫn bổ sung:**
- Dựa vào hành động giá gần đây để xác định quán tính (momentum) hiện tại.
- Sử dụng dữ liệu, số liệu được cung cấp, KHÔNG tự suy diễn thêm.
- Chú ý: VNINDEX, VN30 là chỉ số, không phải cổ phiếu.
"""
            response = await self.generate_content(prompt)
            report = response.text
            await self.save_report_history(symbol, report, close_today, close_yesterday)
            return report
        except Exception as e:
            logger.error(f"Lỗi tạo báo cáo: {str(e)}")
            return f"❌ Lỗi tạo báo cáo: {str(e)}"

# ---------- TELEGRAM COMMANDS ----------
async def notify_admin_new_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

    # Hiển thị các khung thời gian được hỗ trợ
    valid_timeframes = ", ".join(sorted([key for key in DataValidator.VALID_TIMEFRAMES.keys()]))

    await update.message.reply_text(
        "🚀 **V18.9 - PHÂN TÍCH CHỨNG KHOÁN TOÀN DIỆN!**\n"
        "📊 **Lệnh**:\n"
        "- /analyze [Mã] [Số nến] [Khung thời gian] - Phân tích đa khung.\n"
        f"  (Khung thời gian hỗ trợ: {valid_timeframes})\n"
        "- /getid - Lấy ID.\n"
        "- /approve [user_id] - Duyệt người dùng (admin).\n"
        "💡 **Bắt đầu nào!**"
    )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not await is_user_approved(user_id):
        await notify_admin_new_user(update, context)
        return
    try:
        args = context.args
        if not args:
            raise ValueError("Nhập mã chứng khoán (e.g., VNINDEX, SSI).")
            
        # Xác thực đầu vào
        try:
            symbol = DataValidator.validate_symbol(args[0])
            num_candles = DataValidator.validate_candles(args[1] if len(args) > 1 else str(DEFAULT_CANDLES))
            timeframe = DataValidator.validate_timeframe(args[2] if len(args) > 2 else DEFAULT_TIMEFRAME)
        except ValueError as e:
            await update.message.reply_text(f"❌ Lỗi xác thực: {str(e)}")
            return
            
        loader = DataLoader()
        tech_analyzer = TechnicalAnalyzer()
        ai_analyzer = AIAnalyzer()
        
        # Tải dữ liệu cho các khung thời gian khác nhau - luôn sử dụng các giá trị chuẩn hóa
        standard_timeframes = ['1D', '1W', '1M']  # Đảm bảo các khung thời gian này tồn tại trong DataValidator.VALID_TIMEFRAMES
        dfs = {}
        outlier_reports = {}
        
        await update.message.reply_text(f"⏳ Đang tải và phân tích dữ liệu cho {symbol}...")
        
        # Tải dữ liệu song song
        async def load_timeframe_data(tf):
            try:
                # Đảm bảo khung thời gian đã được xác thực
                validated_tf = DataValidator.validate_timeframe(tf)
                df, outlier_report = await loader.load_data(symbol, validated_tf, num_candles)
                processed_df = tech_analyzer.calculate_indicators(df)
                return validated_tf, processed_df, outlier_report
            except Exception as e:
                logger.error(f"Lỗi xử lý khung thời gian {tf}: {str(e)}")
                raise e
            
        tasks = [load_timeframe_data(tf) for tf in standard_timeframes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Lỗi tải dữ liệu: {str(result)}")
                continue
                
            tf, df, report = result
            dfs[tf] = df
            outlier_reports[tf] = report
            
        if not dfs:
            raise ValueError(f"Không thể tải dữ liệu cho {symbol} ở bất kỳ khung thời gian nào.")
            
        # Tải dữ liệu cơ bản
        fundamental_data = await loader.get_fundamental_data(symbol)
        
        # Tạo báo cáo
        report = await ai_analyzer.generate_report(dfs, symbol, fundamental_data, outlier_reports)
        await redis_manager.set(f"report_{symbol}_{num_candles}", report, expire=CACHE_EXPIRE_SHORT)

        formatted_report = f"<b>📈 Báo cáo phân tích cho {symbol}</b>\n\n"
        formatted_report += f"<pre>{html.escape(report)}</pre>"
        await update.message.reply_text(formatted_report, parse_mode='HTML')
    except ValueError as e:
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi trong analyze_command: {str(e)}")
        await update.message.reply_text(f"❌ Lỗi không xác định: {str(e)}")

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
    try:
        await init_db()

        scheduler = AsyncIOScheduler()
        scheduler.add_job(auto_train_models, 'cron', hour=2, minute=0)
        scheduler.start()
        logger.info("Auto training scheduler đã khởi động.")

        app = Application.builder().token(TELEGRAM_TOKEN).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(CommandHandler("analyze", analyze_command))
        app.add_handler(CommandHandler("getid", get_id))
        app.add_handler(CommandHandler("approve", approve_user))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, notify_admin_new_user))
        logger.info("🤖 Bot khởi động!")

        # Kiểm tra xem có đang chạy trên Render hay ở môi trường local
        if RENDER_EXTERNAL_URL:
            # Chế độ Render - chỉ sử dụng webhook
            BASE_URL = RENDER_EXTERNAL_URL
            WEBHOOK_URL = f"{BASE_URL}/{TELEGRAM_TOKEN}"
            logger.info(f"Chạy trên Render với webhook URL: {WEBHOOK_URL}")

            # Đảm bảo tất cả các dependencies cần thiết cho webhook được cài đặt
            try:
                import sys, subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "python-telegram-bot[webhooks]"])
                logger.info("Đã cài đặt/kiểm tra python-telegram-bot[webhooks]")
            except Exception as e:
                logger.error(f"Lỗi cài đặt dependencies webhook: {str(e)}")
                raise

            # Sử dụng run_webhook một cách an toàn
            webhook_server = await app.initialize()
            await webhook_server.start_webhook(
                listen="0.0.0.0",
                port=PORT,
                url_path=TELEGRAM_TOKEN,
                webhook_url=WEBHOOK_URL
            )
            
            # Sử dụng signal handler để hỗ trợ tắt bot một cách an toàn
            import signal
            
            def signal_handler(sig, frame):
                logger.info("Đang dừng ứng dụng...")
                asyncio.create_task(app.shutdown())
                
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Giữ cho ứng dụng tiếp tục chạy
            await asyncio.Event().wait()
        else:
            # Chế độ local development - sử dụng polling
            logger.info("Khởi động bot ở chế độ polling (local development)...")
            await app.run_polling()
    except Exception as e:
        logger.critical(f"Lỗi nghiêm trọng trong main(): {str(e)}")
        logger.critical(f"Traceback: {traceback.format_exc()}")
        # Đảm bảo tất cả tài nguyên được giải phóng
        try:
            if 'scheduler' in locals() and scheduler.running:
                scheduler.shutdown()
            if 'app' in locals():
                await app.shutdown()
        except Exception as cleanup_error:
            logger.error(f"Lỗi khi dọn dẹp tài nguyên: {str(cleanup_error)}")
        raise

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import unittest

        class TestTechnicalAnalysis(unittest.TestCase):
            def setUp(self):
                dates = pd.date_range(start="2023-01-01", periods=50, freq='D')
                data = {
                    'open': np.random.random(50) * 100,
                    'high': np.random.random(50) * 100,
                    'low': np.random.random(50) * 100,
                    'close': np.linspace(100, 150, 50),
                    'volume': np.random.randint(1000, 5000, 50)
                }
                self.df = pd.DataFrame(data, index=dates)

            def test_calculate_indicators(self):
                analyzer = TechnicalAnalyzer()
                df_processed = analyzer.calculate_indicators(self.df)
                self.assertIn('sma20', df_processed.columns)
                self.assertIn('rsi', df_processed.columns)

        class TestForecastProphet(unittest.TestCase):
            def setUp(self):
                dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
                self.df = pd.DataFrame({
                    'close': np.linspace(100, 200, 100),
                    'open': np.linspace(100, 200, 100),
                    'high': np.linspace(100, 200, 100),
                    'low': np.linspace(100, 200, 100),
                    'volume': np.random.randint(1000, 5000, 100)
                }, index=dates)

            def test_forecast_with_prophet(self):
                forecast, model = forecast_with_prophet(self.df, periods=7)
                self.assertFalse(forecast.empty)
                self.assertIn('yhat', forecast.columns)

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

        unittest.main(argv=[sys.argv[0]])
    else:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())