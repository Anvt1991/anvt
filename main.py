#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot Chứng Khoán Toàn Diện Phiên Bản V18.11 (Nâng cấp):
- Tích hợp AI OpenRouter cho phân tích mẫu hình, sóng, và nến nhật.
- Sử dụng mô hình deepseek/deepseek-chat-v3-0324:free.
- Chuẩn hóa dữ liệu và pipeline xử lý.
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
import gc
import time

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
from pydantic import BaseModel, validator
import cProfile
import pstats
from io import StringIO

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

# ---------- CHUẨN HÓA DỮ LIỆU ----------
class DataNormalizer:
    """
    Lớp chuẩn hóa và xử lý dữ liệu với các phương pháp nâng cao
    để xử lý các trường hợp ngoại lệ và bất thường.
    """
    
    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa dataframe theo nhiều phương pháp khác nhau
        """
        if df.empty:
            return df
            
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        normalized_df = df.copy()
        
        # Đảm bảo các cột số đúng định dạng
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in normalized_df.columns:
                # Chuyển đổi sang kiểu số
                normalized_df[col] = pd.to_numeric(normalized_df[col], errors='coerce')
        
        # Xử lý các cột ngày tháng
        date_columns = ['date', 'time', 'datetime']
        for col in date_columns:
            if col in normalized_df.columns:
                try:
                    normalized_df[col] = pd.to_datetime(normalized_df[col], errors='coerce')
                except:
                    pass
                    
        # Xóa các hàng trùng lặp
        normalized_df = normalized_df.drop_duplicates()
        
        # Sắp xếp theo ngày nếu có cột date
        if 'date' in normalized_df.columns:
            normalized_df = normalized_df.sort_values('date')
            
        return normalized_df
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> (bool, str):
        """
        Kiểm tra tính hợp lệ của dữ liệu
        """
        if df.empty:
            return False, "DataFrame rỗng"
            
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Thiếu các cột: {', '.join(missing_columns)}"
            
        # Kiểm tra dữ liệu hợp lệ
        if df['high'].min() < df['low'].min():
            return False, "Có giá trị high nhỏ hơn giá trị low"
            
        # Kiểm tra giá âm
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns and (df[col] < 0).any():
                return False, f"Có giá trị âm trong cột {col}"
                
        # Kiểm tra giá đóng cửa nằm ngoài phạm vi high-low
        invalid_close = ((df['close'] > df['high']) | (df['close'] < df['low'])).sum()
        if invalid_close > 0:
            return False, f"Có {invalid_close} giá đóng cửa nằm ngoài phạm vi high-low"
            
        return True, "Dữ liệu hợp lệ"
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns=['open', 'high', 'low', 'close'], 
                         method='zscore', threshold=3) -> (pd.DataFrame, str):
        """
        Phát hiện giá trị ngoại lai trong dữ liệu sử dụng nhiều phương pháp
        """
        if df.empty:
            return df, "DataFrame rỗng"
            
        outlier_report = {}
        outlier_indices = set()
        
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        df_copy = df.copy()
        
        for col in columns:
            if col not in df_copy.columns:
                continue
                
            # Phát hiện giá trị ngoại lai bằng phương pháp Z-Score
            if method == 'zscore':
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                outliers = z_scores > threshold
                col_outliers = df_copy.index[outliers].tolist()
                
                if len(col_outliers) > 0:
                    outlier_report[col] = {
                        'count': len(col_outliers),
                        'indices': col_outliers[:10],  # Chỉ lấy 10 vị trí đầu tiên để tránh quá dài
                        'method': 'Z-Score'
                    }
                    outlier_indices.update(col_outliers)
            
            # Phát hiện giá trị ngoại lai bằng IQR (Interquartile Range)
            elif method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                
                outliers = (df_copy[col] < (Q1 - 1.5 * IQR)) | (df_copy[col] > (Q3 + 1.5 * IQR))
                col_outliers = df_copy.index[outliers].tolist()
                
                if len(col_outliers) > 0:
                    outlier_report[col] = {
                        'count': len(col_outliers),
                        'indices': col_outliers[:10],
                        'method': 'IQR'
                    }
                    outlier_indices.update(col_outliers)
            
            # Phương pháp Modified Z-Score (Robust Z-Score)
            elif method == 'modified_zscore':
                median = df_copy[col].median()
                # Sử dụng MAD (Median Absolute Deviation) thay vì độ lệch chuẩn
                mad = np.median(np.abs(df_copy[col] - median))
                
                if mad == 0:  # Tránh chia cho 0
                    continue
                    
                modified_z_scores = 0.6745 * np.abs(df_copy[col] - median) / mad
                outliers = modified_z_scores > threshold
                col_outliers = df_copy.index[outliers].tolist()
                
                if len(col_outliers) > 0:
                    outlier_report[col] = {
                        'count': len(col_outliers),
                        'indices': col_outliers[:10],
                        'method': 'Modified Z-Score'
                    }
                    outlier_indices.update(col_outliers)
        
        # Tạo báo cáo tóm tắt
        report_text = ""
        total_outliers = len(outlier_indices)
        
        if total_outliers > 0:
            report_text = f"Phát hiện {total_outliers} giá trị ngoại lai trong {len(columns)} cột.\n"
            for col, details in outlier_report.items():
                report_text += f"- Cột {col}: {details['count']} giá trị ngoại lai (phương pháp {details['method']})\n"
        else:
            report_text = "Không phát hiện giá trị ngoại lai."
            
        return df_copy, report_text
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Điền các giá trị bị thiếu bằng nhiều phương pháp nâng cao
        """
        if df.empty:
            return df
            
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        df_filled = df.copy()
        
        # Danh sách các cột số
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col not in df_filled.columns:
                continue
                
            # Kiểm tra giá trị NaN
            if df_filled[col].isna().sum() > 0:
                # Nếu ít hơn 10% giá trị bị thiếu, sử dụng nội suy tuyến tính
                if df_filled[col].isna().mean() < 0.1:
                    df_filled[col] = df_filled[col].interpolate(method='linear')
                # Nếu lớn hơn 10% nhưng nhỏ hơn 30%, sử dụng phương pháp nội suy spline
                elif df_filled[col].isna().mean() < 0.3:
                    df_filled[col] = df_filled[col].interpolate(method='spline', order=3)
                # Nếu quá nhiều giá trị bị thiếu, sử dụng giá trị trung bình
                else:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mean())
        
        # Đối với các cột phi số
        for col in df_filled.columns:
            if col not in numeric_columns and df_filled[col].isna().any():
                # Sử dụng phương pháp ffill (forward fill) cho dữ liệu thời gian
                if col in ['date', 'time', 'datetime']:
                    df_filled[col] = df_filled[col].fillna(method='ffill')
                # Đối với các cột khác, sử dụng chế độ (giá trị phổ biến nhất)
                else:
                    df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
        
        return df_filled
    
    @staticmethod
    def standardize_for_db(data: dict) -> dict:
        """
        Chuẩn hóa dữ liệu cho cơ sở dữ liệu
        """
        if not data:
            return {}
            
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        standardized = data.copy()
        
        # Xử lý các kiểu dữ liệu phức tạp
        for key, value in standardized.items():
            # Chuyển đổi datetime thành chuỗi ISO format
            if isinstance(value, datetime):
                standardized[key] = value.isoformat()
            # Chuyển đổi pandas Timestamp thành chuỗi ISO format
            elif hasattr(value, 'timestamp') and callable(getattr(value, 'timestamp')):
                standardized[key] = value.isoformat()
            # Chuyển đổi numpy int/float thành Python int/float
            elif hasattr(value, 'item') and callable(getattr(value, 'item')):
                standardized[key] = value.item()
            # Chuyển đổi NaN thành None
            elif pd.isna(value):
                standardized[key] = None
                
        return standardized

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
            
    async def optimize_cache(self):
        """Tối ưu bộ nhớ Redis bằng cách xóa cache cũ và không sử dụng"""
        try:
            # Lấy tất cả các key từ Redis
            all_keys = await self.redis_client.keys("*")
            current_time = datetime.now()
            deleted_count = 0
            
            # Ưu tiên xóa các loại cache khác nhau
            for key in all_keys:
                try:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    
                    # Không xóa dữ liệu người dùng và báo cáo
                    if key_str.startswith(("user_", "report_history_")):
                        continue
                        
                    # Xóa cache dữ liệu cũ (>1 ngày) và tin tức
                    if (key_str.startswith("data_") and "1D" not in key_str) or key_str.startswith("news_"):
                        await self.redis_client.delete(key)
                        deleted_count += 1
                    
                    # Chỉ giữ lại cache cho các cổ phiếu VN30 và các chỉ số
                    elif key_str.startswith("data_") and not any(index in key_str for index in ["VNINDEX", "VN30", "HNX30"]):
                        # Kiểm tra TTL, nếu còn trên 1 giờ thì giữ lại
                        ttl = await self.redis_client.ttl(key)
                        if ttl < 3600:  # Dưới 1 giờ
                            await self.redis_client.delete(key)
                            deleted_count += 1
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý key {key}: {str(e)}")
                    continue
                    
            # Chạy garbage collector sau khi dọn dẹp
            gc.collect()
            
            logger.info(f"Đã tối ưu Redis cache: xóa {deleted_count}/{len(all_keys)} key")
            return deleted_count
        except Exception as e:
            logger.error(f"Lỗi tối ưu Redis cache: {str(e)}")
            return 0

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
                standardized_data = DataNormalizer.standardize_for_db({
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
    return DataNormalizer.standardize_for_db(data)

# ---------- HÀM HỖ TRỢ: LỌC NGÀY GIAO DỊCH -----------
def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df[df.index.weekday < 5]
    years = df.index.year.unique()
    vn_holidays = holidays.Vietnam(years=years)
    holiday_dates = set(vn_holidays.keys())
    df = df[~pd.to_datetime(df.index.date).isin(holiday_dates)]
    return df

# ---------- TẢI DỮ LIỆU (NÂNG CẤP) ----------
class DataLoader:
    def __init__(self, source: str = 'vnstock'):
        self.source = source

    async def load_data(self, symbol: str, timeframe: str, num_candles: int) -> (pd.DataFrame, str):
        timeframe_map = {'1d': '1D', '1w': '1W', '1mo': '1M'}
        timeframe = timeframe_map.get(timeframe.lower(), timeframe).upper()
        
        # Tối ưu cache theo loại dữ liệu
        is_popular = symbol.upper() in ['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX', 'UPCOM']
        is_intraday = timeframe not in ['1D', '1W', '1M']
        
        if is_popular:
            expire = CACHE_EXPIRE_SHORT if is_intraday else CACHE_EXPIRE_MEDIUM
        else:
            expire = CACHE_EXPIRE_SHORT // 2 if is_intraday else CACHE_EXPIRE_SHORT
        
        # Giới hạn số nến để tránh quá tải
        effective_num_candles = min(num_candles, 300)  # Giới hạn tối đa 300 nến
        
        cache_key = f"data_{self.source}_{symbol}_{timeframe}_{effective_num_candles}"
        cached_data = await redis_manager.get(cache_key)
        if cached_data is not None:
            return cached_data, "Dữ liệu từ cache, không kiểm tra outlier"

        try:
            if self.source == 'vnstock':
                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
                def fetch_vnstock():
                    stock = Vnstock().stock(symbol=symbol, source='TCBS')
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=(num_candles + 1) * 3)).strftime('%Y-%m-%d')
                    df = stock.quote.history(start=start_date, end=end_date, interval=timeframe)
                    if df is None or df.empty or len(df) < 20:
                        raise ValueError(f"Không đủ dữ liệu cho {'chỉ số' if is_index(symbol) else 'mã'} {symbol}")
                    
                    # Chuẩn hóa dữ liệu với DataNormalizer
                    df = df.rename(columns={'time': 'date', 'open': 'open', 'high': 'high',
                                            'low': 'low', 'close': 'close', 'volume': 'volume'})
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    df = DataNormalizer.normalize_dataframe(df)
                    df.index = df.index.tz_localize('Asia/Bangkok')
                    
                    # Xác thực dữ liệu
                    is_valid, error_msg = DataNormalizer.validate_data(df)
                    if not is_valid:
                        logger.warning(f"Dữ liệu không hợp lệ cho {symbol}: {error_msg}")
                    
                    # Điền giá trị thiếu
                    df = DataNormalizer.fill_missing_values(df)
                    
                    if len(df) < 200:
                        logger.warning(f"Dữ liệu cho {symbol} dưới 200 nến, SMA200 có thể không chính xác")
                    
                    return df.tail(num_candles + 1)
                df = await run_in_thread(fetch_vnstock)
            elif self.source == 'yahoo':
                period_map = {'1D': 'd', '1W': 'wk', '1M': 'mo'}
                df = await self._download_yahoo_data(symbol, num_candles + 1, period_map.get(timeframe, 'd'))
                if df is None or df.empty or len(df) < 20:
                    raise ValueError(f"Không đủ dữ liệu cho {symbol} từ Yahoo Finance")
                
                # Chuẩn hóa dữ liệu với DataNormalizer
                df = DataNormalizer.normalize_dataframe(df)
                df = DataNormalizer.fill_missing_values(df)
                df.index = df.index.tz_localize('Asia/Bangkok')
                
                # Xác thực dữ liệu
                is_valid, error_msg = DataNormalizer.validate_data(df)
                if not is_valid:
                    logger.warning(f"Dữ liệu không hợp lệ cho {symbol}: {error_msg}")
                
                if len(df) < 200:
                    logger.warning(f"Dữ liệu cho {symbol} dưới 200 nến, SMA200 có thể không chính xác")
            else:
                raise ValueError("Nguồn dữ liệu không hợp lệ")

            trading_df = filter_trading_days(df)
            trading_df, outlier_report = DataNormalizer.detect_outliers(trading_df)
            await redis_manager.set(cache_key, trading_df, expire=expire)
            return trading_df, outlier_report
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu cho {symbol}: {str(e)}")
            raise ValueError(f"Không thể tải dữ liệu: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
    async def _download_yahoo_data(self, symbol: str, num_candles: int, period: str) -> pd.DataFrame:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                start_ts = int((datetime.now() - timedelta(days=num_candles * 3)).timestamp())
                end_ts = int(datetime.now().timestamp())
                url = (f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
                       f"?period1={start_ts}&period2={end_ts}&interval=1{period}&events=history")
                async with asyncio.wait_for(session.get(url), timeout=15) as response:
                    if response.status != 200:
                        raise ValueError(f"Không thể tải dữ liệu từ Yahoo, HTTP {response.status}")
                    text = await response.text()
                    df = pd.read_csv(io.StringIO(text))
                    if df.empty:
                        raise ValueError("Dữ liệu Yahoo rỗng")
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                    return df.tail(num_candles)
        except asyncio.TimeoutError:
            logger.error("Timeout khi tải dữ liệu từ Yahoo Finance.")
            raise
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu Yahoo: {str(e)}")
            raise

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
            return DataNormalizer.standardize_for_db(fundamental_data)

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
            return DataNormalizer.standardize_for_db(fundamental_data)

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
    """
    Lớp phân tích kỹ thuật với các phương pháp tính toán đã được tối ưu hóa
    và áp dụng lưu trữ đệm (caching)
    """
    def __init__(self):
        # Cache để lưu kết quả tính toán
        self._cache = {}
    
    @staticmethod
    def _calculate_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Tính toán các chỉ báo kỹ thuật phổ biến"""
        if df.empty:
            return df
            
        result_df = df.copy()
        
        # Chuyển đổi sang dataframe TA-Lib
        try:
            # RSI (Relative Strength Index)
            result_df['rsi_14'] = ta.momentum.RSIIndicator(
                close=result_df['close'], window=14
            ).rsi()
            
            # MACD (Moving Average Convergence Divergence)
            macd = ta.trend.MACD(
                close=result_df['close'], window_slow=26, window_fast=12, window_sign=9
            )
            result_df['macd'] = macd.macd()
            result_df['macd_signal'] = macd.macd_signal()
            result_df['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(
                close=result_df['close'], window=20, window_dev=2
            )
            result_df['bb_upper'] = bollinger.bollinger_hband()
            result_df['bb_lower'] = bollinger.bollinger_lband()
            result_df['bb_mavg'] = bollinger.bollinger_mavg()
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                high=result_df['high'], low=result_df['low'], close=result_df['close'], window=14, smooth_window=3
            )
            result_df['stoch_k'] = stoch.stoch()
            result_df['stoch_d'] = stoch.stoch_signal()
            
            # Moving Averages
            result_df['sma_20'] = ta.trend.SMAIndicator(close=result_df['close'], window=20).sma_indicator()
            result_df['sma_50'] = ta.trend.SMAIndicator(close=result_df['close'], window=50).sma_indicator()
            result_df['sma_200'] = ta.trend.SMAIndicator(close=result_df['close'], window=200).sma_indicator()
            result_df['ema_9'] = ta.trend.EMAIndicator(close=result_df['close'], window=9).ema_indicator()
            
            # ATR (Average True Range)
            result_df['atr_14'] = ta.volatility.AverageTrueRange(
                high=result_df['high'], low=result_df['low'], close=result_df['close'], window=14
            ).average_true_range()
            
            # ADX (Average Directional Index)
            adx = ta.trend.ADXIndicator(
                high=result_df['high'], low=result_df['low'], close=result_df['close'], window=14
            )
            result_df['adx_14'] = adx.adx()
            result_df['adx_pos'] = adx.adx_pos()
            result_df['adx_neg'] = adx.adx_neg()
            
        except Exception as e:
            logger.error(f"Lỗi khi tính toán chỉ báo kỹ thuật: {str(e)}")
        
        return result_df
    
    @lru_cache(maxsize=32)
    def _cached_calculate_indicators(self, df_key, columns_hash):
        """
        Phiên bản cached của calculate_indicators
        df_key: khóa đại diện cho dataframe
        columns_hash: hash của các cột cần thiết để tránh cache sai
        """
        # Khôi phục dataframe từ cache, hoặc trả về None nếu không tìm thấy
        if df_key in self._cache:
            return self._cache[df_key]['indicators']
        return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán chỉ báo kỹ thuật với cache để tối ưu hiệu suất
        """
        if df is None or df.empty:
            return df
        
        # Tạo key cho dataframe dựa trên nội dung
        required_cols = ['open', 'high', 'low', 'close', 'volume'] if 'volume' in df.columns else ['open', 'high', 'low', 'close']
        cols_available = all(col in df.columns for col in required_cols)
        
        if not cols_available:
            return df
        
        # Tạo hash cho dataframe để dùng làm key
        df_hash = hash(tuple(map(tuple, df[required_cols].tail(5).values.tolist())))
        columns_hash = hash(tuple(required_cols))
        
        # Kiểm tra cache
        cached_result = self._cached_calculate_indicators(df_hash, columns_hash)
        if cached_result is not None:
            return cached_result
        
        # Nếu không có trong cache, tính toán và lưu vào cache
        result = self._calculate_common_indicators(df)
        self._cache[df_hash] = {
            'indicators': result,
            'timestamp': datetime.now()
        }
        
        # Giới hạn kích thước cache
        if len(self._cache) > 50:  # Giữ tối đa 50 kết quả
            # Xóa các mục cũ nhất
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]['timestamp'])
            del self._cache[oldest_key]
        
        return result
    
    def calculate_multi_timeframe_indicators(self, dfs: dict) -> dict:
        """
        Tính toán chỉ báo kỹ thuật cho nhiều khung thời gian
        dfs: dict với key là timeframe, value là dataframe
        """
        result = {}
        for timeframe, df in dfs.items():
            if df is not None and not df.empty:
                result[timeframe] = self.calculate_indicators(df)
            else:
                result[timeframe] = df
        return result

# ---------- CHUẨN HÓA PIPELINE DỮ LIỆU ----------
class DataPipeline:
    """
    Lớp chuẩn hóa pipeline xử lý dữ liệu:
    - Tải dữ liệu
    - Chuẩn hóa và kiểm tra chất lượng
    - Tính toán chỉ báo kỹ thuật
    - Chuẩn bị dữ liệu để phân tích
    """
    def __init__(self):
        self.data_loader = DataLoader()
        self.tech_analyzer = TechnicalAnalyzer()
    
    async def prepare_symbol_data(self, symbol: str, timeframes: list = None, num_candles: int = DEFAULT_CANDLES) -> dict:
        """
        Chuẩn bị dữ liệu đầy đủ cho một mã chứng khoán bao gồm:
        - Dữ liệu đa khung thời gian 
        - Chỉ báo kỹ thuật
        - Phát hiện outlier
        - Dữ liệu cơ bản (nếu có)
        """
        if timeframes is None:
            timeframes = ['1D', '1W', '1M']
        
        result = {
            'symbol': symbol,
            'dataframes': {},
            'indicators': {},
            'outlier_reports': {},
            'fundamental_data': {},
            'errors': []
        }
        
        # Tải dữ liệu đa khung thời gian
        for tf in timeframes:
            try:
                df, outlier_report = await self.data_loader.load_data(symbol, tf, num_candles)
                result['dataframes'][tf] = df
                result['outlier_reports'][tf] = outlier_report
            except Exception as e:
                error_msg = f"Lỗi tải dữ liệu {tf} cho {symbol}: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
        
        # Nếu không có dữ liệu nào được tải thành công
        if not result['dataframes']:
            return result
        
        # Tính toán chỉ báo kỹ thuật
        try:
            result['indicators'] = self.tech_analyzer.calculate_multi_timeframe_indicators(result['dataframes'])
        except Exception as e:
            error_msg = f"Lỗi tính toán chỉ báo kỹ thuật cho {symbol}: {str(e)}"
            logger.error(error_msg)
            result['errors'].append(error_msg)
        
        # Lấy dữ liệu cơ bản
        if not is_index(symbol):
            try:
                result['fundamental_data'] = await self.data_loader.get_fundamental_data(symbol)
            except Exception as e:
                error_msg = f"Lỗi lấy dữ liệu cơ bản cho {symbol}: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
        
        return result
    
    async def prepare_market_data(self, market_symbols: list = None) -> dict:
        """Chuẩn bị dữ liệu tổng quan thị trường"""
        if market_symbols is None:
            market_symbols = ['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX']
        
        result = {
            'market_data': {},
            'market_indicators': {},
            'errors': []
        }
        
        for symbol in market_symbols:
            try:
                symbol_data = await self.prepare_symbol_data(symbol, timeframes=['1D'], num_candles=100)
                result['market_data'][symbol] = symbol_data['dataframes'].get('1D')
                result['market_indicators'][symbol] = symbol_data['indicators'].get('1D', {})
            except Exception as e:
                error_msg = f"Lỗi chuẩn bị dữ liệu thị trường cho {symbol}: {str(e)}"
                logger.error(error_msg)
                result['errors'].append(error_msg)
        
        return result
    
    @staticmethod
    def extract_last_candle_info(df: pd.DataFrame) -> dict:
        """Trích xuất thông tin nến gần nhất"""
        if df is None or df.empty:
            return {}
        
        last_candle = df.iloc[-1].to_dict()
        previous_candle = df.iloc[-2].to_dict() if len(df) > 1 else {}
        
        if previous_candle:
            last_candle['change'] = last_candle.get('close', 0) - previous_candle.get('close', 0)
            last_candle['change_pct'] = (last_candle['change'] / previous_candle.get('close', 1)) * 100 if previous_candle.get('close') else 0
        
        return last_candle
    
    @staticmethod
    def extract_patterns(dfs: dict) -> dict:
        """Trích xuất các mẫu hình nến và đặc điểm kỹ thuật quan trọng"""
        patterns = {}
        
        for tf, df in dfs.items():
            if df is None or df.empty or len(df) < 5:
                continue
                
            df_tail = df.tail(5)
            
            # Kiểm tra xu hướng
            close_prices = df_tail['close']
            trend = 'uptrend' if close_prices.iloc[-1] > close_prices.iloc[0] else 'downtrend'
            
            # Kiểm tra việc cắt qua SMA
            if 'sma20' in df.columns and 'sma50' in df.columns:
                last_row = df.iloc[-1]
                cross_sma20 = close_prices.iloc[-2] < df['sma20'].iloc[-2] and close_prices.iloc[-1] > df['sma20'].iloc[-1]
                cross_sma50 = close_prices.iloc[-2] < df['sma50'].iloc[-2] and close_prices.iloc[-1] > df['sma50'].iloc[-1]
                
                if cross_sma20:
                    patterns[f'{tf}_cross_sma20'] = 'bullish'
                if cross_sma50:
                    patterns[f'{tf}_cross_sma50'] = 'bullish'
            
            # Thêm các mẫu hình khác khi cần
            
            # Lưu xu hướng
            patterns[f'{tf}_trend'] = trend
            
        return patterns

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
        # Khai báo bộ đếm API calls để theo dõi
        self.api_calls_count = 0
        self.last_reset_time = datetime.now()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_content(self, prompt):
        # Sử dụng semaphore để giới hạn số lượng API call đồng thời
        async with api_semaphore:
            # Theo dõi số lượng API calls
            current_time = datetime.now()
            if (current_time - self.last_reset_time).total_seconds() > 60:
                # Reset bộ đếm mỗi phút
                self.api_calls_count = 0
                self.last_reset_time = current_time
                
            self.api_calls_count += 1
            if self.api_calls_count > 20:  # Giới hạn 20 calls/phút
                # Chờ đợi nếu vượt quá giới hạn
                wait_time = 60 - (current_time - self.last_reset_time).total_seconds()
                if wait_time > 0:
                    logger.info(f"Đã đạt giới hạn API calls, chờ {wait_time:.1f}s trước khi tiếp tục")
                    await asyncio.sleep(wait_time)
                self.api_calls_count = 1
                self.last_reset_time = datetime.now()
                
            logger.info(f"Thực hiện API call ({self.api_calls_count}/20 trong phút này)")
            return await self.model.generate_content_async(prompt)
    
    async def analyze_with_openrouter(self, technical_data):
        if not OPENROUTER_API_KEY:
            raise Exception("Chưa có OPENROUTER_API_KEY")

        # Tính toán mức hỗ trợ/kháng cự từ dữ liệu candlestick
        df = pd.DataFrame(technical_data["candlestick_data"])
        calculated_levels = self.calculate_support_resistance_levels(df)
        
        # Tối ưu prompt để giảm token
        prompt = (
            "Bạn là chuyên gia phân tích kỹ thuật chứng khoán. Nhận diện mẫu hình nến, "
            "sóng Elliott, Wyckoff, và các vùng hỗ trợ/kháng cự từ dữ liệu sau:"
            f"\n\nGiá hiện tại: {df['close'].iloc[-1]:.2f}"
            "\n\nChỉ trả về kết quả dạng JSON, không thêm giải thích:\n"
            "{\n"
            "  \"support_levels\": [giá1, giá2, ...],\n"
            "  \"resistance_levels\": [giá1, giá2, ...],\n"
            "  \"patterns\": [\n"
            "    {\"name\": \"tên mẫu hình\", \"description\": \"giải thích ngắn\"},\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            f"Dữ liệu:\n{json.dumps(technical_data, ensure_ascii=False)}"
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
            "temperature": 0.2
        }

        # Sử dụng semaphore để giới hạn API calls
        async with api_semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as resp:
                    text = await resp.text()
                    try:
                        result = json.loads(text)
                        logger.info(f"OpenRouter response keys: {result.keys()}")
                        content = result['choices'][0]['message']['content']
                        
                        # Xử lý khi nội dung được bọc trong ```json ... ```
                        if content.startswith('```json') and content.endswith('```'):
                            content = content[7:-3]  # Cắt bỏ ```json và ```
                        
                        # Xử lý khi nội dung là plain JSON
                        try:
                            openrouter_response = json.loads(content)
                            
                            # Kiểm tra và lọc mức hỗ trợ/kháng cự từ OpenRouter
                            current_price = df['close'].iloc[-1]
                            
                            # Lọc mức hỗ trợ (phải thấp hơn giá hiện tại và là số hợp lệ)
                            filtered_support = []
                            openrouter_support = openrouter_response.get('support_levels', [])
                            for level in openrouter_support:
                                try:
                                    level_value = float(level)
                                    if level_value < current_price and level_value > 0:
                                        filtered_support.append(round(level_value, 2))
                                except (ValueError, TypeError):
                                    continue
                            
                            # Lọc mức kháng cự (phải cao hơn giá hiện tại và là số hợp lệ)
                            filtered_resistance = []
                            openrouter_resistance = openrouter_response.get('resistance_levels', [])
                            for level in openrouter_resistance:
                                try:
                                    level_value = float(level)
                                    if level_value > current_price and level_value > 0:
                                        filtered_resistance.append(round(level_value, 2))
                                except (ValueError, TypeError):
                                    continue
                            
                            # Nếu không có mức hỗ trợ/kháng cự hoặc không hợp lệ, sử dụng kết quả từ phương pháp tính toán
                            if not filtered_support and calculated_levels['support_levels']:
                                filtered_support = calculated_levels['support_levels']
                            if not filtered_resistance and calculated_levels['resistance_levels']:
                                filtered_resistance = calculated_levels['resistance_levels']
                            
                            # Trả về kết quả đã lọc
                            return {
                                "support_levels": filtered_support,
                                "resistance_levels": filtered_resistance,
                                "patterns": openrouter_response.get('patterns', [])
                            }
                        except json.JSONDecodeError:
                            logger.error(f"Lỗi parse JSON từ nội dung: {content}")
                            return calculated_levels
                    except json.JSONDecodeError:
                        logger.error(f"Phản hồi không hợp lệ từ OpenRouter: {text}")
                        return calculated_levels
                    except KeyError as e:
                        logger.error(f"Phản hồi thiếu trường cần thiết: {e}")
                        return calculated_levels

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
                if is_index(symbol):
                    past_result = "đúng" if ((close_today > last["close_today"] and "tăng" in last["report"].lower()) or
                                           (close_today < last["close_today"] and "giảm" in last["report"].lower())) else "sai"
                else:
                    past_result = "đúng" if (close_today > last["close_today"] and "mua" in last["report"].lower()) else "sai"
                past_report = f"📜 **Báo cáo trước** ({last['date']}): {last['close_today']} → {close_today} ({past_result})\n"
            
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
            
            if is_index(symbol):
                xgb_summary = f"**XGBoost dự đoán xu hướng tiếp theo** (Hiệu suất: {xgb_perf:.2f}): {xgb_text}\n"
            else:
                xgb_summary = f"**XGBoost dự đoán tín hiệu giao dịch** (Hiệu suất: {xgb_perf:.2f}): {xgb_text}\n"

            outlier_text = "\n".join([f"**{tf}**: {report}" for tf, report in outlier_reports.items()])

            # Tự tính toán thêm mức hỗ trợ/kháng cự để đối chiếu
            calculated_levels = self.calculate_support_resistance_levels(df_1d)
            calc_support_str = ", ".join([f"{level:.2f}" for level in calculated_levels['support_levels']])
            calc_resistance_str = ", ".join([f"{level:.2f}" for level in calculated_levels['resistance_levels']])

            # Tạo prompt khác nhau cho chỉ số và cổ phiếu
            if is_index(symbol):
                # Phân tích cho chỉ số
                fundamental_report = f"📊 **{symbol} là chỉ số, không phải cổ phiếu**\n"
                
                prompt = f"""
Bạn là chuyên gia phân tích kỹ thuật, phân tích thị trường chứng khoán Việt Nam với 30 năm kinh nghiệm. 
Hãy viết báo cáo chi tiết cho CHỈ SỐ {symbol} (LƯU Ý: ĐÂY LÀ CHỈ SỐ, KHÔNG PHẢI CỔ PHIẾU):

**Thông tin cơ bản:**
- Ngày: {datetime.now().strftime('%d/%m/%Y')}
- Giá hôm qua: {close_yesterday:.2f}
- Giá hôm nay: {close_today:.2f} ({((close_today-close_yesterday)/close_yesterday*100):.2f}%)

**Diễn biến chỉ số:**
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

                prompt += f"\n**Tin tức thị trường:**\n{news_text}\n"
                prompt += f"\n**Phân tích mức hỗ trợ/kháng cự của chỉ số:**\n"
                prompt += f"- Mức hỗ trợ: {', '.join(map(str, support_levels))}\n"
                prompt += f"- Mức kháng cự: {', '.join(map(str, resistance_levels))}\n"
                prompt += f"- Mức hỗ trợ từ phân tích đồ thị: {calc_support_str}\n"  
                prompt += f"- Mức kháng cự từ phân tích đồ thị: {calc_resistance_str}\n"
                prompt += f"- Mẫu hình đồ thị: {', '.join([p.get('name', 'Unknown') for p in patterns])}\n"
                prompt += f"\n{xgb_summary}\n"
                prompt += f"{forecast_summary}\n"
                prompt += """
**Yêu cầu:**
1. Đánh giá tổng quan thị trường. So sánh chỉ số phiên hiện tại và phiên trước đó.
2. Phân tích đa khung thời gian, xu hướng ngắn hạn, trung hạn, dài hạn của CHỈ SỐ.
3. Đánh giá các mô hình, mẫu hình, sóng (nếu có) chỉ số kỹ thuật, động lực thị trường.
4. Xác định hỗ trợ/kháng cự cho CHỈ SỐ. Đưa ra kịch bản và xác suất % (tăng, giảm, sideway).
5. Đề xuất chiến lược cho nhà đầu tư: nên theo xu hướng thị trường hay đi ngược, mức độ thận trọng.
6. Đánh giá rủi ro thị trường hiện tại.
7. Đưa ra nhận định tổng thể về xu hướng thị trường.
8. Không cần theo form cố định, trình bày logic, súc tích nhưng đủ thông tin để hành động và sáng tạo với emoji.

**Hướng dẫn bổ sung:**
- QUAN TRỌNG: Đây là phân tích cho CHỈ SỐ, KHÔNG PHẢI CỔ PHIẾU. Không đưa ra khuyến nghị mua/bán chỉ số.
- Dựa vào hành động giá gần đây để xác định quán tính (momentum) hiện tại.
- Sử dụng dữ liệu, số liệu được cung cấp, KHÔNG tự suy diễn thêm.
"""
            else:
                # Phân tích cho cổ phiếu
                fundamental_report = deep_fundamental_analysis(fundamental_data)
                
                prompt = f"""
Bạn là chuyên gia phân tích kỹ thuật và cơ bản, trader chuyên nghiệp, chuyên gia bắt đáy 30 năm kinh nghiệm ở chứng khoán Việt Nam. Hãy viết báo cáo chi tiết cho cổ phiếu {symbol}:

**Thông tin cơ bản:**
- Ngày: {datetime.now().strftime('%d/%m/%Y')}
- Giá hôm qua: {close_yesterday:.2f}
- Giá hôm nay: {close_today:.2f} ({((close_today-close_yesterday)/close_yesterday*100):.2f}%)

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
                prompt += f"\n**Phân tích mức hỗ trợ/kháng cự:**\n"
                prompt += f"- Mức hỗ trợ: {', '.join(map(str, support_levels))}\n"
                prompt += f"- Mức kháng cự: {', '.join(map(str, resistance_levels))}\n"
                prompt += f"- Mức hỗ trợ từ phân tích đồ thị: {calc_support_str}\n"  
                prompt += f"- Mức kháng cự từ phân tích đồ thị: {calc_resistance_str}\n"
                prompt += f"- Mẫu hình nến: {', '.join([p.get('name', 'Unknown') for p in patterns])}\n"
                prompt += f"\n{xgb_summary}\n"
                prompt += f"{forecast_summary}\n"
                prompt += """
**Yêu cầu:**
1. Đánh giá tổng quan. So sánh giá/chỉ số phiên hiện tại và phiên trước đó.
2. Phân tích đa khung thời gian, xu hướng ngắn hạn, trung hạn, dài hạn.
3. Đánh giá các mô hình, mẫu hình, sóng (nếu có), chỉ số kỹ thuật, động lực thị trường.
4. Xác định hỗ trợ/kháng cự. Đưa ra kịch bản và xác suất % (tăng, giảm, sideway).
5. Đề xuất các chiến lược giao dịch phù hợp, với % tin cậy.
6. Đánh giá rủi ro và tỷ lệ risk/reward.
7. Đưa ra nhận định.
8. Không cần theo form cố định, trình bày logic, súc tích nhưng đủ thông tin để hành động và sáng tạo với emoji.

**Hướng dẫn bổ sung:**
- Dựa vào hành động giá gần đây để xác định quán tính (momentum) hiện tại.
- Sử dụng dữ liệu, số liệu được cung cấp, KHÔNG tự suy diễn thêm.
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

    await update.message.reply_text(
        "🚀 **V18.11 - THUA GIA CÁT LƯỢNG MỖI CÁI QUẠT!**\n"
        "📊 **Lệnh**:\n"
        "- /analyze [Mã] [Số nến] - Phân tích đa khung.\n"
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
        symbol = args[0].upper()
        num_candles = int(args[1]) if len(args) > 1 else DEFAULT_CANDLES
        
        # Giới hạn số nến để tránh quá tải
        if num_candles < 20:
            raise ValueError("Số nến phải lớn hơn hoặc bằng 20 để tính toán chỉ báo!")
        if num_candles > 300:
            num_candles = 300
            await update.message.reply_text("⚠️ Số nến đã được giới hạn tối đa 300 để tránh quá tải.")
        
        # Kiểm tra cache trước
        cache_key = f"report_{symbol}_{num_candles}"
        cached_report = await redis_manager.get(cache_key)
        if cached_report:
            await update.message.reply_text(f"📊 Báo cáo từ cache cho {symbol}:")
            formatted_report = f"<b>📈 Báo cáo phân tích cho {symbol}</b>\n\n"
            formatted_report += f"<pre>{html.escape(cached_report)}</pre>"
            await update.message.reply_text(formatted_report, parse_mode='HTML')
            
            # Tạo một task để làm mới cache trong nền nếu báo cáo đã cũ (>30 phút)
            if isinstance(cached_report, dict) and cached_report.get('timestamp'):
                cache_time = datetime.fromisoformat(cached_report.get('timestamp'))
                if (datetime.now() - cache_time).total_seconds() > 1800:  # 30 phút
                    asyncio.create_task(refresh_report_cache(symbol, num_candles))
            return
            
        # Sử dụng pipeline chuẩn hóa
        data_pipeline = DataPipeline()
        ai_analyzer = AIAnalyzer()
        
        # Gửi thông báo trạng thái tải dữ liệu
        status_message = await update.message.reply_text(f"⏳ Đang chuẩn bị dữ liệu cho {symbol}...")
        
        # Sử dụng timeframe phù hợp cho mỗi loại phân tích
        timeframes = ['1D']
        if not is_index(symbol):  # Chỉ tải nhiều khung thời gian cho cổ phiếu
            timeframes = ['1D', '1W']  # Giảm bớt khung thời gian (không tải 1M)
            
        # Chuẩn bị dữ liệu với pipeline
        pipeline_result = await data_pipeline.prepare_symbol_data(symbol, timeframes=timeframes, num_candles=num_candles)
        
        # Cập nhật trạng thái
        await status_message.edit_text(f"⏳ Dữ liệu đã sẵn sàng. Đang phân tích {symbol}...")
        
        if pipeline_result['errors']:
            error_message = f"⚠️ Một số lỗi xảy ra trong quá trình chuẩn bị dữ liệu:\n"
            error_message += "\n".join(pipeline_result['errors'])
            await update.message.reply_text(error_message)
        
        if not pipeline_result['dataframes']:
            raise ValueError(f"Không thể tải dữ liệu cho {symbol}")
        
        # Tạo báo cáo với AI
        report = await ai_analyzer.generate_report(
            pipeline_result['dataframes'], 
            symbol, 
            pipeline_result['fundamental_data'], 
            pipeline_result['outlier_reports']
        )
        
        # Thêm timestamp vào cache để biết thời gian tạo báo cáo
        report_cache = {
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
        # Lưu vào cache với thời gian ngắn hơn
        cache_expire = CACHE_EXPIRE_SHORT // 2 if not is_index(symbol) else CACHE_EXPIRE_SHORT
        await redis_manager.set(f"report_{symbol}_{num_candles}", report_cache, expire=cache_expire)

        # Cập nhật trạng thái hoàn thành
        await status_message.delete()
        
        formatted_report = f"<b>📈 Báo cáo phân tích cho {symbol}</b>\n\n"
        formatted_report += f"<pre>{html.escape(report)}</pre>"
        await update.message.reply_text(formatted_report, parse_mode='HTML')
        
        # Giải phóng bộ nhớ sau khi hoàn thành
        gc.collect()
        
    except ValueError as e:
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi trong analyze_command: {str(e)}")
        await update.message.reply_text(f"❌ Lỗi không xác định: {str(e)}")

async def refresh_report_cache(symbol: str, num_candles: int):
    """Làm mới cache báo cáo trong nền để người dùng tiếp theo có dữ liệu mới"""
    try:
        logger.info(f"Đang làm mới cache báo cáo cho {symbol} trong nền")
        
        # Tạo mới dữ liệu
        data_pipeline = DataPipeline()
        ai_analyzer = AIAnalyzer()
        
        # Sử dụng timeframe phù hợp
        timeframes = ['1D']
        if not is_index(symbol):
            timeframes = ['1D', '1W']
            
        # Chuẩn bị dữ liệu với pipeline
        pipeline_result = await data_pipeline.prepare_symbol_data(symbol, timeframes=timeframes, num_candles=num_candles)
        
        if not pipeline_result['dataframes']:
            logger.error(f"Không thể làm mới cache cho {symbol}: không có dữ liệu")
            return
        
        # Tạo báo cáo mới
        report = await ai_analyzer.generate_report(
            pipeline_result['dataframes'], 
            symbol, 
            pipeline_result['fundamental_data'], 
            pipeline_result['outlier_reports']
        )
        
        # Thêm timestamp vào cache
        report_cache = {
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
        # Lưu vào cache
        cache_expire = CACHE_EXPIRE_SHORT // 2 if not is_index(symbol) else CACHE_EXPIRE_SHORT
        await redis_manager.set(f"report_{symbol}_{num_candles}", report_cache, expire=cache_expire)
        
        logger.info(f"Đã làm mới cache báo cáo cho {symbol} thành công")
        
        # Giải phóng bộ nhớ
        gc.collect()
        
    except Exception as e:
        logger.error(f"Lỗi làm mới cache báo cáo cho {symbol}: {str(e)}")

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
async def keep_alive():
    """Giữ cho ứng dụng không bị ngủ trên Render"""
    app_url = os.getenv("RENDER_EXTERNAL_URL", "")
    if not app_url:
        return
        
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(app_url, timeout=10) as response:
                if response.status == 200:
                    logger.info("Keep alive ping thành công")
                else:
                    logger.warning(f"Keep alive ping trả về mã lỗi: {response.status}")
    except Exception as e:
        logger.error(f"Keep alive ping thất bại: {str(e)}")

async def send_telegram_document(file_path, caption):
    """Gửi file qua Telegram API"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        
        # Kiểm tra file tồn tại
        if not os.path.exists(file_path):
            logger.error(f"File không tồn tại: {file_path}")
            return False
            
        # Chuẩn bị form data
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field('chat_id', ADMIN_ID)
            form.add_field('caption', caption)
            
            # Thêm file
            with open(file_path, 'rb') as file:
                form.add_field('document', file, 
                               filename=os.path.basename(file_path),
                               content_type='application/json')
            
            # Gửi request
            async with session.post(url, data=form) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('ok'):
                        logger.info(f"Đã gửi file {file_path} đến Telegram thành công")
                        return True
                    else:
                        logger.error(f"Lỗi API Telegram: {result}")
                else:
                    logger.error(f"Lỗi HTTP khi gửi file: {response.status}")
                    
        return False
    except Exception as e:
        logger.error(f"Lỗi gửi file qua Telegram: {str(e)}")
        return False

async def backup_database():
    """Sao lưu dữ liệu quan trọng và gửi qua Telegram thay vì lưu cục bộ"""
    try:
        # Tạo thư mục tạm để lưu file backup
        temp_dir = "temp_backups"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Sao lưu danh sách người dùng
        async with SessionLocal() as session:
            users_query = await session.execute(select(ApprovedUser))
            users_data = [
                {
                    "user_id": user.user_id, 
                    "approved_at": user.approved_at.isoformat()
                } 
                for user in users_query.scalars().all()
            ]
            
            # Sao lưu báo cáo gần nhất
            reports_query = await session.execute(
                select(ReportHistory).order_by(ReportHistory.id.desc()).limit(20)
            )
            reports_data = [
                {
                    "id": report.id,
                    "symbol": report.symbol,
                    "date": report.date,
                    "close_today": report.close_today,
                    "close_yesterday": report.close_yesterday,
                    "timestamp": report.timestamp.isoformat()
                }
                for report in reports_query.scalars().all()
            ]
            
            # Sao lưu thông tin model đã train
            models_query = await session.execute(
                select(TrainedModel).order_by(TrainedModel.id.desc())
            )
            models_data = [
                {
                    "id": model.id,
                    "symbol": model.symbol,
                    "model_type": model.model_type,
                    "created_at": model.created_at.isoformat(),
                    "performance": model.performance
                }
                for model in models_query.scalars().all()
            ]
        
        # Lưu vào tệp tạm thời
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_data = {
            "users": users_data,
            "reports": reports_data,
            "models": models_data,
            "timestamp": datetime.now().isoformat(),
            "app_version": "18.9"
        }
        
        backup_file = os.path.join(temp_dir, f"backup_{current_time}.json")
        with open(backup_file, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
        
        # Gửi file qua Telegram
        caption = f"🔄 Backup dữ liệu {datetime.now().strftime('%d/%m/%Y %H:%M')}\n"
        caption += f"👥 Users: {len(users_data)}\n"
        caption += f"📊 Reports: {len(reports_data)}\n"
        caption += f"🤖 Models: {len(models_data)}"
        
        sent = await send_telegram_document(backup_file, caption)
        
        if sent:
            logger.info(f"Đã sao lưu dữ liệu và gửi qua Telegram thành công")
        else:
            logger.error("Không thể gửi backup qua Telegram")
        
        # Xóa file tạm sau khi gửi
        try:
            os.remove(backup_file)
            logger.info(f"Đã xóa file tạm: {backup_file}")
        except Exception as e:
            logger.warning(f"Không thể xóa file tạm {backup_file}: {str(e)}")
        
        return sent
    except Exception as e:
        logger.error(f"Lỗi sao lưu dữ liệu: {str(e)}")
        return False

async def main():
    # Khởi tạo DB 
    await init_db()

    # Thiết lập semaphore cho API call
    global api_semaphore
    api_semaphore = asyncio.Semaphore(3)  # Giới hạn tối đa 3 API call đồng thời

    # Khởi tạo scheduler với các tác vụ định kỳ
    scheduler = AsyncIOScheduler()
    
    # Tác vụ định kỳ
    scheduler.add_job(auto_train_models, 'cron', hour=2, minute=0)
    scheduler.add_job(keep_alive, 'interval', minutes=14)  # Ping trước khi Render sleep (15 phút)
    scheduler.add_job(backup_database, 'cron', hour=1, minute=0)  # Sao lưu hàng ngày lúc 1:00
    scheduler.add_job(redis_manager.optimize_cache, 'interval', hours=6)  # Tối ưu Redis cache mỗi 6 giờ
    
    scheduler.start()
    logger.info("Scheduler đã khởi động với các tác vụ định kỳ.")

    # Cài đặt bot
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("getid", get_id))
    app.add_handler(CommandHandler("approve", approve_user))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, notify_admin_new_user))
    logger.info("🤖 Bot khởi động!")

    # Cài đặt webhook với cơ chế tự phục hồi
    BASE_URL = os.getenv("RENDER_EXTERNAL_URL", f"https://{os.getenv('RENDER_SERVICE_NAME')}.onrender.com")
    WEBHOOK_URL = f"{BASE_URL}/{TELEGRAM_TOKEN}"
    
    async def setup_webhook():
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                webhook_info = await app.bot.get_webhook_info()
                if webhook_info.url != WEBHOOK_URL:
                    await app.bot.set_webhook(url=WEBHOOK_URL)
                    logger.info(f"Webhook đã được thiết lập thành công: {WEBHOOK_URL}")
                else:
                    logger.info(f"Webhook đã được thiết lập trước đó: {WEBHOOK_URL}")
                return
            except Exception as e:
                retry_count += 1
                logger.error(f"Lỗi thiết lập webhook (thử lần {retry_count}): {str(e)}")
                await asyncio.sleep(5)
    
    # Khởi tạo webhook
    await setup_webhook()
    
    # Khởi động webhook server
    await app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        webhook_url=WEBHOOK_URL,
        url_path=TELEGRAM_TOKEN
    )

class StockData(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: int
    date: datetime

    @validator('high')
    def high_must_be_greater_than_low(cls, v, values):
        if 'low' in values and v < values['low']:
            raise ValueError('High must be greater than low')
        return v

    @validator('close')
    def close_must_be_within_range(cls, v, values):
        if 'low' in values and 'high' in values and not (values['low'] <= v <= values['high']):
            raise ValueError('Close must be within the range of low and high')
        return v

    @validator('volume')
    def volume_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError('Volume must be non-negative')
        return v

# Example usage
# stock_data = StockData(open=100.0, high=105.0, low=99.0, close=102.0, volume=1000, date=datetime.now())

def profile_function(func):
    """
    Decorator để profile một hàm và ghi ra thông tin về hiệu suất
    """
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        
        # Tạo báo cáo
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(20)  # In ra 20 hàm tiêu tốn thời gian nhất
        
        logger.debug(f"Profiling results for {func.__name__}:\n{s.getvalue()}")
        return result
    return wrapper

# Áp dụng cache để lưu trữ kết quả tính toán các chỉ báo kỹ thuật
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_technical_calculation(df_key, indicator_name):
    """
    Hàm này sẽ lưu trữ kết quả tính toán các chỉ báo kỹ thuật
    df_key: một key duy nhất đại diện cho dataframe (ví dụ: symbol+timeframe+hash)
    indicator_name: tên của chỉ báo kỹ thuật
    """
    # Thực hiện tính toán chỉ báo dựa trên key
    # Đây chỉ là hàm giúp cache kết quả
    return None

# Tối ưu hóa việc huấn luyện mô hình
async def optimized_train_models_for_symbol(symbol: str):
    """
    Phiên bản tối ưu của train_models_for_symbol với khả năng phân tích hiệu suất
    và các tối ưu hóa về bộ nhớ và CPU
    """
    try:
        logger.info(f"Bắt đầu huấn luyện mô hình cho {symbol}")
        
        # Tải dữ liệu
        data_loader = DataLoader()
        df, status = await data_loader.load_data(symbol, 'daily', 1000)
        
        if df is None or df.empty:
            logger.error(f"Không thể tải dữ liệu cho {symbol}")
            return
        
        # Tiền xử lý dữ liệu tối ưu hóa
        df = optimize_dataframe_memory(df)
        
        # Huấn luyện các mô hình và đo hiệu suất
        model_db_manager = ModelDBManager()
        
        # Huấn luyện mô hình Prophet
        start_time = time.time()
        prophet_model, prophet_performance = train_prophet_model(df.copy())
        prophet_time = time.time() - start_time
        logger.info(f"Huấn luyện mô hình Prophet cho {symbol} hoàn tất trong {prophet_time:.2f}s, hiệu suất: {prophet_performance:.4f}")
        
        # Lưu mô hình
        if prophet_model:
            prophet_model_binary = pickle.dumps(prophet_model)
            await model_db_manager.store_trained_model(symbol, "prophet", prophet_model_binary, prophet_performance)
        
        # Chuẩn bị dữ liệu cho XGBoost 
        features = prepare_features_for_xgboost(df)
        
        # Huấn luyện mô hình XGBoost
        start_time = time.time()
        xgb_model, xgb_performance = train_xgboost_model(df.copy(), features)
        xgb_time = time.time() - start_time
        logger.info(f"Huấn luyện mô hình XGBoost cho {symbol} hoàn tất trong {xgb_time:.2f}s, hiệu suất: {xgb_performance:.4f}")
        
        # Lưu mô hình
        if xgb_model:
            xgb_model_binary = pickle.dumps(xgb_model)
            await model_db_manager.store_trained_model(symbol, "xgboost", xgb_model_binary, xgb_performance)
        
        # Giải phóng bộ nhớ
        del df, prophet_model, xgb_model
        gc.collect()
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình cho {symbol}: {str(e)}")
        return False

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tối ưu hóa bộ nhớ sử dụng cho DataFrame bằng cách chuyển đổi kiểu dữ liệu
    """
    # Tạo một bản sao để tránh ảnh hưởng đến dữ liệu gốc
    result = df.copy()
    
    # Tối ưu kiểu dữ liệu số nguyên
    for col in result.select_dtypes(include=['int']):
        # Chuyển đổi sang các kiểu int nhỏ hơn nếu có thể
        c_min = result[col].min()
        c_max = result[col].max()
        
        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
            result[col] = result[col].astype(np.int8)
        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
            result[col] = result[col].astype(np.int16)
        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
            result[col] = result[col].astype(np.int32)
    
    # Tối ưu kiểu dữ liệu số thực
    for col in result.select_dtypes(include=['float']):
        # Chuyển đổi sang các kiểu float nhỏ hơn nếu có thể
        result[col] = result[col].astype(np.float32)
    
    # Tối ưu kiểu dữ liệu object
    for col in result.select_dtypes(include=['object']):
        # Nếu là cột chứa danh mục có số lượng giá trị nhỏ, chuyển sang categorical
        if result[col].nunique() < 50:  # Ngưỡng 50 danh mục
            result[col] = result[col].astype('category')
    
    return result

def prepare_features_for_xgboost(df: pd.DataFrame) -> list:
    """
    Chuẩn bị và chọn lọc đặc trưng cho mô hình XGBoost
    """
    # Tính toán các chỉ báo kỹ thuật
    analyzer = TechnicalAnalyzer()
    df_with_indicators = analyzer.calculate_indicators(df)
    
    # Chọn lọc đặc trưng quan trọng để giảm kích thước mô hình và tăng tốc độ
    # (Có thể sử dụng SelectKBest, PCA, hoặc các phương pháp khác)
    
    # Danh sách các đặc trưng quan trọng (ví dụ)
    important_features = [
        'rsi_14', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 
        'ema_9', 'sma_20', 'sma_50', 'atr_14', 'adx_14'
    ]
    
    # Chỉ lấy các đặc trưng có trong DataFrame
    available_features = [f for f in important_features if f in df_with_indicators.columns]
    
    return available_features

# Tối ưu hóa quá trình dự đoán
def optimized_predict_xgboost_signal(df: pd.DataFrame, features: list, model) -> (int, float):
    """
    Phiên bản tối ưu của hàm dự đoán tín hiệu XGBoost
    """
    # Lấy dữ liệu mới nhất
    X = df[features].iloc[-1:].values
    
    # Kiểm tra NaN và thay thế
    if np.isnan(X).any():
        # Thay thế NaN bằng giá trị trung bình của cột
        col_means = np.nanmean(df[features].values, axis=0)
        for i in range(X.shape[1]):
            if np.isnan(X[0, i]):
                X[0, i] = col_means[i]
    
    # Dự đoán
    prediction = model.predict_proba(X)[0]
    signal = 1 if prediction[1] > 0.5 else (-1 if prediction[1] < 0.3 else 0)
    confidence = prediction[1] if signal == 1 else (1 - prediction[1] if signal == -1 else 0.5)
    
    return signal, confidence

# Thay thế hàm train_models_for_symbol gốc bằng phiên bản tối ưu
async def train_models_for_symbol(symbol: str):
    return await optimized_train_models_for_symbol(symbol)

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
                df_with_outliers, report = DataNormalizer.detect_outliers(self.df)
                self.assertIn('is_outlier', df_with_outliers.columns)
                self.assertEqual(df_with_outliers['is_outlier'].sum(), 1)
                self.assertIn('500', report)
                
        class TestDataNormalizer(unittest.TestCase):
            def setUp(self):
                dates = pd.date_range(start="2023-01-01", periods=10, freq='D')
                data = {
                    'open': [100, 101, 102, np.nan, 104, 105, 106, 107, 108, 109],
                    'high': [110, 111, 112, np.nan, 114, 115, 116, 117, 118, 119],
                    'low': [90, 91, 92, np.nan, 94, 95, 96, 97, 98, 99],
                    'close': [105, 106, 107, np.nan, 109, 110, 111, 112, 113, 114],
                    'volume': [1000, 1100, 1200, np.nan, 1400, 1500, 1600, 1700, 1800, 1900]
                }
                self.df = pd.DataFrame(data, index=dates)
                
            def test_fill_missing_values(self):
                df_filled = DataNormalizer.fill_missing_values(self.df)
                self.assertFalse(df_filled.isna().any().any())
                # Kiểm tra giá trị được điền đúng
                self.assertEqual(df_filled['close'][3], 107)  # Giá trị close trước đó
                
            def test_normalize_dataframe(self):
                # Tạo DataFrame với tên cột khác
                df_diff_cols = self.df.copy()
                df_diff_cols.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                df_normalized = DataNormalizer.normalize_dataframe(df_diff_cols)
                self.assertListEqual(list(df_normalized.columns), ['open', 'high', 'low', 'close', 'volume'])

        unittest.main(argv=[sys.argv[0]])
    else:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())