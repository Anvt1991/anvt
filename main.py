#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot Chứng Khoán Toàn Diện Phiên Bản Local (Nâng cấp từ V18.9):
- Tải dữ liệu chứng khoán qua VNStock/Yahoo Finance với cache bằng cachetools.
- Hợp nhất, chuẩn hóa dữ liệu, bổ sung khung thời gian (1w, 1mo), phát hiện bất thường.
- Phân tích kỹ thuật đa khung thời gian (SMA, RSI, MACD, ...).
- Thu thập tin tức từ nhiều nguồn RSS (async).
- Phân tích cơ bản nâng cao, phân biệt cổ phiếu/chỉ số.
- Dự báo giá (Prophet) và tín hiệu giao dịch (XGBoost) làm đầu vào cho Gemini AI.
- Báo cáo phân tích bằng Gemini AI (fallback sang Groq), lưu lịch sử vào SQLite (async).
- Tích hợp Telegram với các lệnh: /start, /analyze, /getid, /approve.
- Auto training mô hình theo lịch định kỳ (mỗi ngày 2h sáng).
- Tối ưu chạy local với polling, xử lý async hoàn toàn.
- Nâng cấp từ 18.9 lên 19.2:
  - Thêm lớp Validate/Normalize toàn diện với xác thực dữ liệu, căn chỉnh chuỗi thời gian
  - Mở rộng hỗ trợ đa khung thời gian (5m, 15m, 30m, 1h, 4h, và legacy) với đồng bộ hóa
  - Nâng cấp từ SQLite đến aiosqlite với migrations, theo dõi version và lịch sử mô hình
  - Cải thiện xử lý lỗi và logging với truy xuất nguồn gốc và stack chi tiết
  - Tối ưu hóa hiệu suất phân tích đa khung thời gian và dự báo
"""

# Standard library imports
import asyncio
import io
import inspect
import json
import logging
import os
import pickle
import re
import sqlite3
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-party imports
# Data processing and analysis
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Environment setup
import pytz
from dotenv import load_dotenv

# Async and scheduling
import aiohttp
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tenacity import retry, stop_after_attempt, wait_exponential

# Database
import redis.asyncio as redis
from sqlalchemy import (Column, DateTime, Float, Integer, LargeBinary, String,
                       Text, func, select, text)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Telegram
from telegram import Update
from telegram.ext import (Application, CommandHandler, ContextTypes,
                         MessageHandler, filters)

# Data analysis and ML
import feedparser
import google.generativeai as genai
import holidays
import xgboost as xgb
import yfinance as yf
from prophet import Prophet
from ta import momentum, trend, volatility
from ta.volume import MFIIndicator
from vnstock import Vnstock

# In-memory cache dictionary
cache = {}

# ---------- CẤU HÌNH & LOGGING ----------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "deepseek-r1-distill-llama-70b"  # Model ổn định từ bot 1.5
ADMIN_ID = os.getenv("ADMIN_ID", "1225226589")
# Database URLs
DATABASE_URL = os.getenv("DATABASE_URL")          # ví dụ: postgres://user:pass@hostname:port/dbname
REDIS_URL = os.getenv("REDIS_URL")                # ví dụ: redis://:pass@hostname:port/0
# Render deployment variables
PORT = int(os.environ.get("PORT", 10000))
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "")
RENDER_SERVICE_NAME = os.getenv("RENDER_SERVICE_NAME", "")

# Thiết lập logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Các hằng số và cấu hình
CACHE_EXPIRE_SHORT = 1800   # 30 phút
CACHE_EXPIRE_MEDIUM = 3600  # 1 giờ
CACHE_EXPIRE_LONG = 86400   # 1 ngày
NEWS_CACHE_EXPIRE = 900     # 15 phút
DEFAULT_CANDLES = 100
DEFAULT_TIMEFRAME = '1D'
TZ = pytz.timezone('Asia/Bangkok')
VERSION = "V19.2"           # Phiên bản ứng dụng

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
    async def set(self, key: str, value: Any, expire: int) -> bool:
        try:
            serialized_value = pickle.dumps(value)
            await self.redis_client.set(key, serialized_value, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis set: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get(self, key: str) -> Any:
        try:
            data = await self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Lỗi Redis get: {str(e)}")
            return None

redis_manager = RedisManager()

executor = ThreadPoolExecutor(max_workers=5)

async def run_in_thread(func: Callable, *args, **kwargs) -> Any:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: func(*args, **kwargs))

# ---------- KẾT NỐI POSTGRESQL (Async) ----------
Base = declarative_base()

class ApprovedUser(Base):
    __tablename__ = 'approved_users'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, nullable=False)
    approved_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

class ReportHistory(Base):
    __tablename__ = 'report_history'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False, default='1D')
    date = Column(String, nullable=False)
    report = Column(Text, nullable=False)
    close_today = Column(Float, nullable=False)
    close_yesterday = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

class TrainedModel(Base):
    __tablename__ = 'trained_models'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)   # 'prophet' hoặc 'xgboost'
    model_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    performance = Column(Float, nullable=True)
    version = Column(String, nullable=False, default="1.0")
    params = Column(Text, nullable=True)  # JSON string của các tham số mô hình

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ---------- LỚP VALIDATE/NORMALIZE TOÀN DIỆN ----------
class DataValidator:
    """
    Lớp xác thực và chuẩn hóa dữ liệu toàn diện.
    Cung cấp các phương thức để xác thực, chuẩn hóa và căn chỉnh dữ liệu chứng khoán.
    """
    # Danh sách khung thời gian hợp lệ
    VALID_TIMEFRAMES = [
        '5m', '15m', '30m', '1h', '4h',  # Khung thời gian mới
        '1D', '1W', '1M'                 # Khung thời gian legacy
    ]
    
    # Danh sách các mã chỉ số
    INDICES = ['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX', 'UPCOM']
    
    # Định dạng mã hợp lệ
    TICKER_PATTERN = r'^[A-Z0-9]{3,6}$'
    
    @staticmethod
    def is_valid_timeframe(timeframe: str) -> bool:
        """Kiểm tra tính hợp lệ của khung thời gian"""
        return timeframe.upper() in [tf.upper() for tf in DataValidator.VALID_TIMEFRAMES]
    
    @staticmethod
    def normalize_timeframe(timeframe: str) -> str:
        """Chuẩn hóa khung thời gian"""
        mapping = {
            '1d': '1D', '1day': '1D', 'daily': '1D', 'd': '1D', 'day': '1D',
            '1w': '1W', '1week': '1W', 'weekly': '1W', 'w': '1W', 'week': '1W',
            '1m': '1M', '1month': '1M', 'monthly': '1M', 'm': '1M', 'month': '1M',
            '5m': '5m', '5min': '5m', '5': '5m',
            '15m': '15m', '15min': '15m', '15': '15m',
            '30m': '30m', '30min': '30m', '30': '30m',
            '1h': '1h', '1hour': '1h', 'h': '1h', 'hour': '1h',
            '4h': '4h', '4hour': '4h'
        }
        normalized = mapping.get(timeframe.lower(), timeframe)
        
        # Kiểm tra xem có phải định dạng timeframe hợp lệ không (thêm logic mở rộng)
        if DataValidator.is_valid_timeframe(normalized):
            return normalized
            
        # Nếu không phải timeframe đã biết, kiểm tra xem có định dạng phổ biến không
        # Ví dụ "2h", "8h" vẫn có thể được coi là hợp lệ nếu theo quy ước
        import re
        tf_pattern = r'^(\d+)([mhdwM])$'  # Patterns như "2h", "6h", "8h", "45m"
        match = re.match(tf_pattern, normalized)
        if match:
            # Giữ nguyên định dạng nhưng đăng ký vào log để sau này có thể bổ sung
            logger.info(f"Khung thời gian mới phát hiện: {normalized}, sẽ sử dụng nguyên bản.")
            return normalized
            
        # Nếu không khớp với bất kỳ định dạng nào, mới báo lỗi
        raise ValueError(f"Khung thời gian không hợp lệ: {timeframe}")
    
    @staticmethod
    def validate_ticker(ticker: str) -> str:
        """Xác thực mã chứng khoán"""
        import re
        ticker = ticker.upper().strip()
        # Kiểm tra xem ticker có phải là chỉ số không
        if ticker in DataValidator.INDICES:
            return ticker
        # Nếu không phải chỉ số, kiểm tra định dạng
        if re.match(DataValidator.TICKER_PATTERN, ticker):
            return ticker
        raise ValueError(f"Mã chứng khoán không hợp lệ: {ticker}")
    
    @staticmethod
    def is_index(ticker: str) -> bool:
        """Kiểm tra xem mã có phải là chỉ số hay không"""
        return ticker.upper() in DataValidator.INDICES
    
    @staticmethod
    def validate_candles(num_candles: int) -> int:
        """Xác thực số lượng nến"""
        if 20 <= num_candles <= 1000:
            return num_candles
        raise ValueError(f"Số lượng nến phải từ 20 đến 1000, nhận được: {num_candles}")
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
        """Xác thực khoảng thời gian"""
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if start >= end:
                raise ValueError("Ngày bắt đầu phải trước ngày kết thúc")
            if end > datetime.now(TZ):
                end = datetime.now(TZ)
            return start, end
        except Exception as e:
            raise ValueError(f"Khoảng thời gian không hợp lệ: {str(e)}")
    
    @staticmethod
    def align_timestamps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Căn chỉnh dấu thời gian theo khung thời gian chính xác"""
        if df.empty:
            return df
        
        df = df.copy()
        # Đảm bảo index là datetime và có múi giờ
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Nếu không có múi giờ, giả định là múi giờ Bangkok
        if df.index.tz is None:
            df.index = df.index.tz_localize(TZ)
        
        # Căn chỉnh timestamp dựa trên khung thời gian
        if timeframe == '1D':
            # Đối với dữ liệu ngày, lấy giá mỗi ngày lúc EOD (15:00)
            df.index = df.index.normalize() + pd.Timedelta(hours=15)
        elif timeframe == '1W':
            # Đối với dữ liệu tuần, lấy giá vào cuối tuần (Thứ 6, 15:00)
            df.index = df.index.to_period('W').to_timestamp() + pd.Timedelta(days=4, hours=15)
        elif timeframe == '1M':
            # Đối với dữ liệu tháng, lấy giá vào cuối tháng
            df.index = df.index.to_period('M').to_timestamp() + pd.Timedelta(hours=15)
        elif timeframe == '5m':
            # Căn chỉnh theo mỗi 5 phút
            df.index = df.index.floor('5min')
        elif timeframe == '15m':
            # Căn chỉnh theo mỗi 15 phút
            df.index = df.index.floor('15min')
        elif timeframe == '30m':
            # Căn chỉnh theo mỗi 30 phút
            df.index = df.index.floor('30min')
        elif timeframe == '1h':
            # Căn chỉnh theo mỗi giờ
            df.index = df.index.floor('H')
        elif timeframe == '4h':
            # Căn chỉnh theo mỗi 4 giờ
            df.index = df.index.floor('4H')
        
        # Sắp xếp theo thời gian tăng dần
        df = df.sort_index()
        
        return df
    
    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa DataFrame với xử lý giá trị thiếu, outlier và định dạng cột"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Chuẩn hóa tên cột
        column_mapping = {
            'time': 'date', 'Time': 'date', 'DATE': 'date', 'Date': 'date',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'Adj Close': 'adj_close', 'Adjusted_close': 'adj_close'
        }
        df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})
        
        # Đảm bảo các cột bắt buộc
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame thiếu cột bắt buộc: {col}")
        
        # Xử lý giá trị null: lấp đầy giá trị thiếu trong chuỗi thời gian
        if any(df.isnull().sum()):
            # Phương pháp điền: forward fill cho giá mở/đóng/cao/thấp, không điền cho khối lượng
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].fillna(method='ffill')
            
            # Các giá trị vẫn còn thiếu (ở đầu chuỗi) sẽ được điền bằng backward fill
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].fillna(method='bfill')
            
            # Khối lượng thiếu được điền 0
            df['volume'] = df['volume'].fillna(0)
        
        # Kiểm tra tính nhất quán của dữ liệu
        # - Giá cao >= giá đóng, mở, thấp
        # - Giá thấp <= giá đóng, mở, cao
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        # Chuyển đổi kiểu dữ liệu
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Loại bỏ các hàng có giá trị âm trong giá
        df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
        
        # Thêm cột ngày giao dịch nếu đang dùng cột date là index
        if 'date' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df['date'] = df.index
        
        return df
    
    @staticmethod
    def detect_and_handle_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> Tuple[pd.DataFrame, str]:
        """
        Phát hiện và xử lý giá trị ngoại lệ (outlier) trong dữ liệu
        
        Args:
            df: DataFrame cần xử lý
            method: Phương pháp phát hiện ('iqr' hoặc 'zscore')
            threshold: Ngưỡng phát hiện
            
        Returns:
            DataFrame đã xử lý và báo cáo về outliers
        """
        if df.empty:
            return df, "DataFrame rỗng, không thể phát hiện outlier"
        
        df = df.copy()
        outlier_report = "Phát hiện outlier:\n"
        outliers_found = False
        
        # Cột cần kiểm tra outlier
        cols_to_check = ['open', 'high', 'low', 'close', 'volume']
        cols_to_check = [col for col in cols_to_check if col in df.columns]
        
        if method == 'iqr':
            for col in cols_to_check:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if not outliers.empty:
                    outliers_found = True
                    outlier_report += f"\n{col}: {len(outliers)} giá trị ngoại lệ\n"
                    for idx, row in outliers.iterrows():
                        date_str = idx.strftime('%Y-%m-%d %H:%M') if isinstance(idx, pd.Timestamp) else str(idx)
                        outlier_report += f"  - {date_str}: {row[col]:.2f}\n"
                    
                    # Xử lý outlier bằng cách clip giới hạn
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        elif method == 'zscore':
            from scipy import stats
            for col in cols_to_check:
                z_scores = stats.zscore(df[col])
                abs_z_scores = np.abs(z_scores)
                outliers_mask = abs_z_scores > threshold
                outliers = df[outliers_mask]
                
                if not outliers.empty:
                    outliers_found = True
                    outlier_report += f"\n{col}: {len(outliers)} giá trị ngoại lệ\n"
                    for idx, row in outliers.iterrows():
                        date_str = idx.strftime('%Y-%m-%d %H:%M') if isinstance(idx, pd.Timestamp) else str(idx)
                        z = z_scores[df.index.get_loc(idx)]
                        outlier_report += f"  - {date_str}: {row[col]:.2f} (z={z:.2f})\n"
                    
                    # Xử lý outlier: thay thế bằng giá trị trung bình
                    mean_val = df[~outliers_mask][col].mean()
                    df.loc[outliers_mask, col] = mean_val
        
        if not outliers_found:
            outlier_report = "Không phát hiện giá trị ngoại lệ"
        
        return df, outlier_report
    
    @staticmethod
    def validate_fundamental_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Xác thực dữ liệu cơ bản của cổ phiếu"""
        if not data:
            return {"error": "Không có dữ liệu cơ bản"}
        
        valid_data = {}
        # Danh sách các chỉ số cơ bản và phạm vi hợp lệ
        key_ranges = {
            'EPS': (-1000, 100000),
            'P/E': (0, 1000),
            'P/B': (0, 100),
            'ROE': (-100, 100),
            'ROA': (-100, 100),
            'Dividend Yield': (0, 100),
            'Market Cap': (0, 1e12),
            'Revenue': (0, 1e12),
            'Profit': (-1e12, 1e12)
        }
        
        # Chuẩn hóa tên key
        key_mapping = {
            'earningPerShare': 'EPS',
            'eps': 'EPS',
            'priceToEarning': 'P/E',
            'pe': 'P/E',
            'priceToBook': 'P/B',
            'pb': 'P/B',
            'returnOnEquity': 'ROE',
            'roe': 'ROE',
            'returnOnAsset': 'ROA',
            'roa': 'ROA',
            'dividendYield': 'Dividend Yield',
            'dividend': 'Dividend Yield',
            'marketCap': 'Market Cap',
            'market_cap': 'Market Cap',
            'revenue': 'Revenue',
            'profit': 'Profit'
        }
        
        # Chuẩn hóa và xác thực dữ liệu
        for key, value in data.items():
            normalized_key = key_mapping.get(key.lower(), key)
            
            # Nếu là key quan trọng, kiểm tra phạm vi
            if normalized_key in key_ranges:
                min_val, max_val = key_ranges[normalized_key]
                try:
                    # Chuyển đổi sang float
                    numeric_value = float(value)
                    if min_val <= numeric_value <= max_val:
                        valid_data[normalized_key] = numeric_value
                    else:
                        logger.warning(f"Dữ liệu '{normalized_key}' = {numeric_value} ngoài phạm vi hợp lệ ({min_val}, {max_val})")
                except (ValueError, TypeError):
                    logger.warning(f"Không thể chuyển đổi '{normalized_key}' = {value} sang số")
            else:
                # Các key khác giữ nguyên
                valid_data[normalized_key] = value
        
        return valid_data
    
    @staticmethod
    def calculate_trading_days(start_date: datetime, end_date: datetime, country: str = 'VN') -> pd.DatetimeIndex:
        """Tính toán các ngày giao dịch trong khoảng thời gian, loại bỏ ngày nghỉ và cuối tuần"""
        if start_date >= end_date:
            raise ValueError("Ngày bắt đầu phải trước ngày kết thúc")
        
        # Lấy danh sách ngày nghỉ lễ
        country_holidays = holidays.country_holidays(country, years=range(start_date.year, end_date.year + 1))
        
        # Tạo DatetimeIndex với tần suất ngày làm việc (loại bỏ cuối tuần)
        business_days = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Loại bỏ ngày nghỉ lễ
        trading_days = [day for day in business_days if day not in country_holidays]
        
        return pd.DatetimeIndex(trading_days)
    
    @staticmethod
    def validate_api_response(response: Dict[str, Any], expected_keys: List[str]) -> bool:
        """Xác thực phản hồi API có đầy đủ các trường dữ liệu cần thiết hay không"""
        if not response:
            return False
        return all(key in response for key in expected_keys)

# Sử dụng PostgreSQL cho database thay vì SQLite
# async_db = AsyncDBManager()

# DB Manager để tương tác với PostgreSQL
class DBManager:
    """
    Quản lý cơ sở dữ liệu PostgreSQL thông qua SQLAlchemy
    Cung cấp các phương thức để tương tác với cơ sở dữ liệu theo mô hình bất đồng bộ
    """
    def __init__(self):
        self.Session = SessionLocal

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def is_user_approved(self, user_id: str) -> bool:
        try:
            async with self.Session() as session:
                # Chỉ truy vấn cột user_id thay vì tất cả các cột
                query = select(ApprovedUser.user_id).filter_by(user_id=str(user_id))
                result = await session.execute(query)
                return result.scalar_one_or_none() is not None or str(user_id) == ADMIN_ID
        except Exception as e:
            logger.error(f"Lỗi kiểm tra người dùng: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def add_approved_user(self, user_id: str, approved_at: Optional[datetime] = None, notes: Optional[str] = None) -> None:
        try:
            # Kiểm tra xem người dùng đã được phê duyệt chưa
            is_approved = await self.is_user_approved(user_id)
            if not is_approved and str(user_id) != ADMIN_ID:
                async with self.Session() as session:
                    try:
                        # Thử sử dụng phương pháp an toàn hơn bằng SQL trực tiếp
                        insert_stmt = text("""
                            INSERT INTO approved_users (user_id, approved_at, notes)
                            VALUES (:user_id, :approved_at, :notes)
                        """)
                        await session.execute(
                            insert_stmt,
                            {
                                "user_id": str(user_id),
                                "approved_at": approved_at or datetime.now(),
                                "notes": notes
                            }
                        )
                        await session.commit()
                        logger.info(f"Thêm người dùng được phê duyệt: {user_id}")
                    except Exception as inner_e:
                        logger.error(f"Lỗi khi thêm người dùng bằng SQL trực tiếp: {str(inner_e)}")
                        # Nếu không thành công với SQL trực tiếp, thử cách ORM
                        await session.rollback()
                        try:
                            new_user = ApprovedUser(
                                user_id=str(user_id),
                                approved_at=approved_at or datetime.now(),
                                notes=notes
                            )
                            session.add(new_user)
                            await session.commit()
                            logger.info(f"Thêm người dùng được phê duyệt (qua ORM): {user_id}")
                        except Exception as orm_e:
                            logger.error(f"Lỗi khi thêm người dùng qua ORM: {str(orm_e)}")
                            await session.rollback()
                            raise
        except Exception as e:
            logger.error(f"Lỗi thêm người dùng: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def update_user_last_active(self, user_id: str) -> None:
        try:
            async with self.Session() as session:
                # Kiểm tra người dùng tồn tại
                query = select(ApprovedUser.id).filter_by(user_id=str(user_id))
                result = await session.execute(query)
                user_id_exists = result.scalar_one_or_none()
                
                if user_id_exists:
                    try:
                        # Cập nhật last_active sử dụng SQL trực tiếp để tránh lỗi ORM
                        update_stmt = text("""
                            UPDATE approved_users 
                            SET last_active = :now 
                            WHERE user_id = :user_id
                        """)
                        await session.execute(
                            update_stmt, 
                            {"now": datetime.now(), "user_id": str(user_id)}
                        )
                        await session.commit()
                    except Exception as e:
                        # Nếu có lỗi khi cập nhật last_active, có thể là do cột không tồn tại
                        # Ghi log và tiếp tục, không cần dừng chương trình
                        logger.warning(f"Không thể cập nhật last_active (có thể cột chưa tồn tại): {str(e)}")
                        await session.rollback()
        except Exception as e:
            logger.error(f"Lỗi cập nhật trạng thái người dùng: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_report_history(self, symbol: str, timeframe: str = '1D', limit: int = 10) -> List[Dict[str, Any]]:
        try:
            async with self.Session() as session:
                query = select(ReportHistory).filter_by(symbol=symbol, timeframe=timeframe).order_by(ReportHistory.id.desc()).limit(limit)
                reports = await session.execute(query)
                reports = reports.scalars().all()
                return [
                    {
                        "id": report.id,
                        "symbol": report.symbol,
                        "timeframe": report.timeframe,
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
    async def save_report_history(self, symbol: str, report: str, close_today: float, close_yesterday: float, timeframe: str = '1D') -> None:
        try:
            async with self.Session() as session:
                date_str = datetime.now().strftime('%Y-%m-%d')
                new_report = ReportHistory(
                    symbol=symbol,
                    timeframe=timeframe,
                    date=date_str,
                    report=report,
                    close_today=close_today,
                    close_yesterday=close_yesterday
                )
                session.add(new_report)
                await session.commit()
                logger.info(f"Lưu báo cáo mới cho {symbol} ({timeframe})")
        except Exception as e:
            logger.error(f"Lỗi lưu báo cáo: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def store_trained_model(self, symbol: str, model_type: str, model: Any, 
                                  performance: Optional[float] = None, 
                                  version: str = "1.0", 
                                  params: Optional[Dict[str, Any]] = None) -> None:
        try:
            model_blob = pickle.dumps(model)
            params_json = json.dumps(params) if params else None
            
            async with self.Session() as session:
                # Kiểm tra xem đã có mô hình chưa (với cùng symbol, model_type, version)
                result = await session.execute(
                    select(TrainedModel).filter_by(
                        symbol=symbol, 
                        model_type=model_type, 
                        version=version
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Cập nhật mô hình hiện có
                    existing.model_blob = model_blob
                    existing.created_at = datetime.now()
                    existing.performance = performance
                    existing.params = params_json
                else:
                    # Tạo mô hình mới
                    new_model = TrainedModel(
                        symbol=symbol, 
                        model_type=model_type, 
                        model_blob=model_blob, 
                        performance=performance,
                        version=version,
                        params=params_json
                    )
                    session.add(new_model)
                
                await session.commit()
                logger.info(f"Lưu mô hình {model_type} v{version} cho {symbol} thành công với hiệu suất: {performance}")
                
        except Exception as e:
            logger.error(f"Lỗi lưu mô hình {model_type} cho {symbol}: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_trained_model(self, symbol: str, model_type: str, version: Optional[str] = None) -> Tuple[Any, Optional[float], Optional[str], Optional[datetime], Optional[Dict[str, Any]]]:
        try:
            async with self.Session() as session:
                if version:
                    # Nếu chỉ định version, lấy mô hình với version đó
                    query = select(TrainedModel).filter_by(symbol=symbol, model_type=model_type, version=version)
                else:
                    # Nếu không, lấy mô hình mới nhất
                    subquery = select(
                        TrainedModel.symbol,
                        TrainedModel.model_type,
                        func.max(TrainedModel.created_at).label('max_date')
                    ).filter_by(symbol=symbol, model_type=model_type).group_by(TrainedModel.symbol, TrainedModel.model_type).subquery()
                    
                    query = select(TrainedModel).join(
                        subquery,
                        (TrainedModel.symbol == subquery.c.symbol) &
                        (TrainedModel.model_type == subquery.c.model_type) &
                        (TrainedModel.created_at == subquery.c.max_date)
                    )
                
                result = await session.execute(query)
                model_record = result.scalar_one_or_none()
                
                if model_record:
                    logger.info(f"Tải mô hình {model_type} v{model_record.version} cho {symbol} thành công")
                    return (
                        pickle.loads(model_record.model_blob),
                        model_record.performance,
                        model_record.version,
                        model_record.created_at,
                        json.loads(model_record.params) if model_record.params else None
                    )
                return None, None, None, None, None
                
        except Exception as e:
            logger.error(f"Lỗi tải mô hình {model_type} cho {symbol}: {str(e)}")
            return None, None, None, None, None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get_model_versions(self, symbol: str, model_type: str) -> List[Dict[str, Any]]:
        try:
            async with self.Session() as session:
                query = select(
                    TrainedModel.version,
                    TrainedModel.created_at,
                    TrainedModel.performance
                ).filter_by(symbol=symbol, model_type=model_type).order_by(TrainedModel.created_at.desc())
                
                result = await session.execute(query)
                versions = result.all()
                
                return [
                    {
                        'version': v.version,
                        'created_at': v.created_at.isoformat(),
                        'performance': v.performance
                    }
                    for v in versions
                ]
        except Exception as e:
            logger.error(f"Lỗi lấy versions mô hình cho {symbol}: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get_training_symbols(self) -> List[str]:
        try:
            async with self.Session() as session:
                query = select(TrainedModel.symbol).distinct()
                result = await session.execute(query)
                symbols = result.scalars().all()
                return list(symbols)
        except Exception as e:
            logger.error(f"Lỗi lấy danh sách symbol đã train: {str(e)}")
            return []

db = DBManager()

# ---------- HÀM HỖ TRỢ ----------
def is_index(symbol: str) -> bool:
    """Kiểm tra xem một mã có phải là chỉ số hay không (tương thích ngược)"""
    return DataValidator.is_index(symbol)

async def is_user_approved(user_id: str) -> bool:
    """Wrapper function để kiểm tra người dùng đã được phê duyệt chưa"""
    return await db.is_user_approved(str(user_id))

def get_vn_trading_days(start_date: datetime, end_date: datetime) -> pd.DatetimeIndex:
    """Hàm giữ lại để tương thích với code cũ"""
    return DataValidator.calculate_trading_days(start_date, end_date)

# ---------- TẢI DỮ LIỆU (NÂNG CẤP) ----------
class DataLoader:
    def __init__(self, source: str = 'vnstock'):
        self.source = source
        self.validator = DataValidator

    def detect_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """Hàm giữ lại để tương thích với code cũ"""
        return self.validator.detect_and_handle_outliers(df, method='iqr')

    async def clean_data(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, str]:
        """Làm sạch dữ liệu với xác thực validator"""
        # Chuẩn hóa DataFrame
        df = self.validator.normalize_dataframe(df)
        
        # Phát hiện và xử lý outlier
        df_cleaned, outlier_report = self.validator.detect_and_handle_outliers(df)
        
        # Đảm bảo đủ ngày giao dịch
        if len(df_cleaned) > 5:  # Chỉ thực hiện nếu có đủ dữ liệu
            try:
                trading_days = self.validator.calculate_trading_days(
                    df_cleaned.index.min(), 
                    df_cleaned.index.max()
                )
                df_cleaned = df_cleaned[df_cleaned.index.isin(trading_days)]
            except Exception as e:
                logger.warning(f"Không thể lọc theo ngày giao dịch: {str(e)}")
        
        return df_cleaned, outlier_report

    async def load_raw_data(self, symbol: str, timeframe: str, num_candles: int) -> pd.DataFrame:
        """Tải dữ liệu thô với validator và hỗ trợ đa khung thời gian"""
        # Chuẩn hóa tham số đầu vào
        symbol = self.validator.validate_ticker(symbol)
        timeframe = self.validator.normalize_timeframe(timeframe)
        num_candles = self.validator.validate_candles(num_candles)
        
        # Ánh xạ khung thời gian cho các nguồn dữ liệu khác nhau
        timeframe_map = {'1d': '1D', '1w': '1W', '1mo': '1M'}
        vnstock_tf = timeframe_map.get(timeframe.lower(), timeframe).upper()
        
        # Tạo khóa cache
        cache_key = f"raw_data_{self.source}_{symbol}_{timeframe}_{num_candles}"
        
        # Kiểm tra cache thường
        if cache_key in cache:
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                return cached_data
                
        # Kiểm tra cache Redis
        try:
            cached_data = await redis_manager.get(cache_key)
            if cached_data is not None:
                # Lưu vào cache thường để truy cập nhanh hơn lần sau
                cache[cache_key] = cached_data
                return cached_data
        except Exception as e:
            logger.warning(f"Không thể lấy dữ liệu từ Redis cache: {str(e)}")

        try:
            if self.source == 'vnstock':
                # Xử lý khung thời gian intraday (mới)
                if timeframe in ['5m', '15m', '30m', '1h', '4h']:
                    stock = Vnstock().stock(symbol=symbol, source='TCBS')
                    end_date = datetime.now(TZ).strftime('%Y-%m-%d')
                    # Tính số ngày cần tải dựa trên khung thời gian
                    if timeframe == '5m':
                        days_back = 5  # 5 phút chỉ có dữ liệu 5 ngày gần nhất
                    elif timeframe in ['15m', '30m']:
                        days_back = 10  # 15-30 phút có 10 ngày
                    elif timeframe == '1h':
                        days_back = 15  # 1h có 15 ngày
                    else:  # 4h
                        days_back = 30  # 4h có 30 ngày
                    
                    start_date = (datetime.now(TZ) - timedelta(days=days_back)).strftime('%Y-%m-%d')
                    
                    # Chuyển đổi timeframe sang định dạng phù hợp với API
                    api_tf_map = {
                        '5m': '5', '15m': '15', '30m': '30', 
                        '1h': '60', '4h': '240'
                    }
                    
                    # Gọi API intraday
                    df = stock.quote.intraday(start=start_date, end=end_date, 
                                              interval=api_tf_map[timeframe])
                    
                    if df is None or df.empty:
                        raise ValueError(f"Không có dữ liệu intraday cho {symbol} với tf={timeframe}")
                    
                    # Chuẩn hóa cột
                    df = df.rename(columns={'time': 'date', 'open': 'open', 'high': 'high',
                                          'low': 'low', 'close': 'close', 'volume': 'volume'})
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(TZ)
                    df = df.set_index('date')
                    # Căn chỉnh thời gian
                    df = self.validator.align_timestamps(df, timeframe)
                    
                else:
                    # Dữ liệu OHLC thông thường (ngày, tuần, tháng)
                    stock = Vnstock().stock(symbol=symbol, source='TCBS')
                    end_date = datetime.now(TZ).strftime('%Y-%m-%d')
                    start_date = (datetime.now(TZ) - timedelta(days=(num_candles + 1) * 3)).strftime('%Y-%m-%d')
                    df = stock.quote.history(start=start_date, end=end_date, interval=vnstock_tf)
                    if df is None or df.empty or len(df) < 20:
                        raise ValueError(f"Không đủ dữ liệu từ VNStock cho {symbol}")
                    df = df.rename(columns={'time': 'date', 'open': 'open', 'high': 'high',
                                          'low': 'low', 'close': 'close', 'volume': 'volume'})
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(TZ)
                    df = df.set_index('date')
                
                # Chuẩn hóa dữ liệu
                df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
                df = df.tail(num_candles + 1)
                
            elif self.source == 'yahoo':
                # Ánh xạ khung thời gian cho Yahoo Finance
                yahoo_period_map = {
                    '1D': 'd', '1W': 'wk', '1M': 'mo',
                    '5m': '5m', '15m': '15m', '30m': '30m',
                    '1h': '60m', '4h': '4h'
                }
                df = await self._download_yahoo_data(
                    symbol, 
                    num_candles + 1, 
                    yahoo_period_map.get(timeframe, 'd')
                )
                
                if df is None or df.empty or len(df) < 20:
                    raise ValueError(f"Không đủ dữ liệu cho {symbol} từ Yahoo Finance")
                df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                      'Close': 'close', 'Volume': 'volume'})
                df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
                df.index = df.index.tz_localize(TZ)
                
                # Căn chỉnh thời gian
                df = self.validator.align_timestamps(df, timeframe)
            else:
                raise ValueError(f"Nguồn dữ liệu không hỗ trợ: {self.source}")
            
            # Thêm vào cache
            cache[cache_key] = df
            
            # Lưu vào Redis cache nếu có thể
            try:
                await redis_manager.set(cache_key, df, CACHE_EXPIRE_MEDIUM)
            except Exception as e:
                logger.warning(f"Không thể lưu dữ liệu vào Redis cache: {str(e)}")
                
            return df
        except Exception as e:
            if self.source == 'vnstock':
                logger.warning(f"Không thể lấy dữ liệu từ VNStock cho {symbol}: {str(e)}. Thử Yahoo Finance...")
                self.source = 'yahoo'
                return await self.load_raw_data(symbol, timeframe, num_candles)
            else:
                raise Exception(f"Không thể tải dữ liệu: {str(e)}")

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60))
    async def _download_yahoo_data(self, symbol: str, num_candles: int, period: str) -> pd.DataFrame:
        """Tải dữ liệu từ Yahoo Finance với retry logic"""
        async with aiohttp.ClientSession() as session:
            # Đối với dữ liệu intraday, cần tính toán khoảng thời gian phù hợp
            is_intraday = period in ['5m', '15m', '30m', '60m', '90m', '1h', '4h']
            
            if is_intraday:
                # Yahoo chỉ cho phép tải dữ liệu intraday trong khoảng thời gian giới hạn
                # Ví dụ: 5m chỉ có thể tải tối đa 7 ngày gần nhất
                days_back = {
                    '5m': 7, '15m': 7, '30m': 7,
                    '60m': 14, '90m': 14, '1h': 14, '4h': 30
                }.get(period, 7)
                
                start_ts = int((datetime.now(TZ) - timedelta(days=days_back)).timestamp())
            else:
                # Dữ liệu ngày, tuần, tháng có thể tải trong khoảng thời gian dài hơn
                start_ts = int((datetime.now(TZ) - timedelta(days=num_candles * 3)).timestamp())
            
            end_ts = int(datetime.now(TZ).timestamp())
            
            # Xây dựng URL dựa trên khung thời gian
            if is_intraday:
                url = (f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
                      f"?period1={start_ts}&period2={end_ts}&interval={period}&events=history")
            else:
                url = (f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
                      f"?period1={start_ts}&period2={end_ts}&interval=1{period}&events=history")
            
            async with session.get(url) as response:
                if response.status == 429:
                    logger.warning(f"HTTP 429 từ Yahoo Finance cho {symbol}. Thử lại sau...")
                    raise Exception("HTTP 429: Too Many Requests")
                if response.status != 200:
                    raise ValueError(f"Không thể tải dữ liệu cho {symbol} từ Yahoo, HTTP {response.status}")
                text = await response.text()
                df = pd.read_csv(io.StringIO(text))
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                return df.tail(num_candles)

    async def load_data(self, symbol: str, timeframe: str, num_candles: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """Tải và xử lý dữ liệu toàn diện với validator"""
        raw_df = await self.load_raw_data(symbol, timeframe, num_candles)
        cleaned_df, outlier_report = await self.clean_data(raw_df, symbol)
        return raw_df, cleaned_df, outlier_report

    async def get_fundamental_data(self, symbol: str) -> dict:
        """Lấy dữ liệu cơ bản với validator"""
        if self.validator.is_index(symbol):
            return {"error": f"{symbol} là chỉ số, không có dữ liệu cơ bản"}
            
        cache_key = f"fundamental_{symbol}_{datetime.now(TZ).strftime('%Y%m%d')}"
        
        # Kiểm tra cache thường
        if cache_key in cache:
            cached_data = cache.get(cache_key)
            if cached_data is not None:
                return cached_data
                
        # Kiểm tra Redis cache
        try:
            cached_data = await redis_manager.get(cache_key)
            if cached_data is not None:
                # Lưu vào cache thường để truy cập nhanh hơn lần sau
                cache[cache_key] = cached_data
                return cached_data
        except Exception as e:
            logger.warning(f"Không thể lấy dữ liệu cơ bản từ Redis cache: {str(e)}")
            
        fundamental_data = await self.fetch_fundamental_data_vnstock(symbol)
        if fundamental_data and any(v is not None for v in fundamental_data.values()):
            # Xác thực dữ liệu trước khi lưu cache
            valid_data = self.validator.validate_fundamental_data(fundamental_data)
            cache[cache_key] = valid_data
            
            # Lưu vào Redis cache nếu có thể
            try:
                await redis_manager.set(cache_key, valid_data, CACHE_EXPIRE_LONG)
            except Exception as e:
                logger.warning(f"Không thể lưu dữ liệu cơ bản vào Redis cache: {str(e)}")
                
            return valid_data
            
        fundamental_data = await self.fetch_fundamental_data_yahoo(symbol)
        if fundamental_data and any(v is not None for v in fundamental_data.values()):
            # Xác thực dữ liệu trước khi lưu cache
            valid_data = self.validator.validate_fundamental_data(fundamental_data)
            cache[cache_key] = valid_data
            
            # Lưu vào Redis cache nếu có thể
            try:
                await redis_manager.set(cache_key, valid_data, CACHE_EXPIRE_LONG)
            except Exception as e:
                logger.warning(f"Không thể lưu dữ liệu cơ bản vào Redis cache: {str(e)}")
                
            return valid_data
            
        return {"error": f"Không có dữ liệu cơ bản cho {symbol}"}

    async def fetch_fundamental_data_vnstock(self, symbol: str) -> dict:
        try:
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
        except Exception as e:
            logger.error(f"Lỗi lấy dữ liệu cơ bản từ VNStock: {str(e)}")
            return {}

    async def fetch_fundamental_data_yahoo(self, symbol: str) -> dict:
        try:
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
        except Exception as e:
            logger.error(f"Lỗi lấy dữ liệu cơ bản từ Yahoo: {str(e)}")
            return {}

# ---------- PHÂN TÍCH KỸ THUẬT ----------
class TechnicalAnalyzer:
    def calculate_common_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or len(df) < 20:
            raise ValueError("Không đủ dữ liệu để tính toán chỉ báo")
        df = df.copy()
        df['sma20'] = trend.SMAIndicator(df['close'], window=20).sma_indicator()
        df['sma50'] = trend.SMAIndicator(df['close'], window=50).sma_indicator()
        df['sma200'] = trend.SMAIndicator(df['close'], window=min(200, len(df)-1)).sma_indicator()
        df['rsi'] = momentum.RSIIndicator(df['close'], window=14).rsi()
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
        df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).money_flow_index()
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

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.calculate_common_indicators(df)

    def calculate_multi_timeframe_indicators(self, dfs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]) -> Dict[str, Dict[str, float]]:
        indicators = {}
        for timeframe, (_, cleaned_df) in dfs.items():
            df_processed = self.calculate_common_indicators(cleaned_df)
            if df_processed.empty:
                continue
            indicators[timeframe] = df_processed.tail(1).to_dict(orient='records')[0]
        return indicators

# ---------- DỰ BÁO GIÁ ----------
class EnhancedPredictor:
    """
    Lớp dự báo nâng cao sử dụng nhiều mô hình để dự đoán giá và tín hiệu giao dịch.
    Hỗ trợ mô hình lai, đánh giá hiệu suất và sanity check.
    """
    def hybrid_predict(self, df: pd.DataFrame, days: int = 5) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        """
        Dự báo giá bằng mô hình lai kết hợp nhiều phương pháp
        
        Args:
            df: DataFrame chứa dữ liệu giá
            days: Số ngày cần dự báo
            
        Returns:
            Tuple gồm mảng dự báo và dictionary các kịch bản
        """
        if df.empty or len(df) < 50:
            raise ValueError("Không đủ dữ liệu để dự báo (cần ít nhất 50 điểm dữ liệu)")
        df = df.copy()
        if getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_convert(None)
            df.index = df.index.tz_localize(None)
        
        # Fix for "cannot insert date, already exists" error
        if 'date' in df.columns:
            # If 'date' is already a column, use it directly
            df_prophet = df.copy()
            df_prophet.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)
        else:
            # Otherwise reset the index as before
            df_prophet = df.reset_index().rename(columns={'date': 'ds', 'close': 'y'})
            
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds']).dt.tz_localize(None)
        
        # Cải thiện mô hình Prophet với các tham số điều chỉnh
        model = Prophet(
            changepoint_prior_scale=0.05,  # Giảm để giới hạn sự biến động
            seasonality_prior_scale=10,    # Tăng trọng số cho yếu tố mùa vụ
            seasonality_mode='multiplicative'  # Phù hợp hơn với dữ liệu tài chính
        )
        
        # Thêm mùa vụ tuần và tháng phù hợp với thị trường chứng khoán
        model.add_seasonality(name='weekly', period=5, fourier_order=5)
        model.add_seasonality(name='monthly', period=21, fourier_order=5)
        
        # Đặt giới hạn tăng trưởng cho mô hình
        last_value = df['close'].iloc[-1]
        growth_cap = last_value * 1.5  # Không dự báo tăng quá 50%
        growth_floor = last_value * 0.5  # Không dự báo giảm quá 50%
        df_prophet['cap'] = growth_cap
        df_prophet['floor'] = growth_floor
        
        model.fit(df_prophet[['ds', 'y', 'cap', 'floor']])
        future = model.make_future_dataframe(periods=days)
        future['cap'] = growth_cap
        future['floor'] = growth_floor
        forecast = model.predict(future)
        
        # Lấy kết quả dự báo
        raw_predictions = forecast.tail(days)['yhat'].values
        
        # Áp dụng bộ lọc sanity check để giới hạn dự báo phi lý
        predictions = self._apply_sanity_checks(raw_predictions, last_value)
        
        # Tính toán xác suất các kịch bản dựa trên dự báo đã điều chỉnh
        last_close = df['close'].iloc[-1]
        volatility = df['close'].pct_change().std() * 100
        
        # Điều chỉnh xác suất dựa trên độ biến động thực tế và giá trị dự báo
        trend_strength = (predictions[-1] - last_close) / last_close
        
        # Xác định xác suất các kịch bản
        if abs(trend_strength) < 0.02:  # Dự báo sideway
            sideway_prob = 60
            breakout_prob = 20 if trend_strength > 0 else 10
            breakdown_prob = 20 if trend_strength < 0 else 10
        elif trend_strength > 0:  # Dự báo tăng
            breakout_prob = min(70, 40 + abs(trend_strength) * 100)
            breakdown_prob = max(10, 30 - abs(trend_strength) * 50)
            sideway_prob = 100 - breakout_prob - breakdown_prob
        else:  # Dự báo giảm
            breakdown_prob = min(70, 40 + abs(trend_strength) * 100)
            breakout_prob = max(10, 30 - abs(trend_strength) * 50)
            sideway_prob = 100 - breakout_prob - breakdown_prob
        
        # Điều chỉnh dựa trên độ biến động
        volatility_factor = min(1.5, max(0.5, volatility / 2))
        sideway_prob = sideway_prob * (1 / volatility_factor)
        
        # Đảm bảo tổng xác suất là 100%
        total = breakout_prob + breakdown_prob + sideway_prob
        breakout_prob = breakout_prob / total * 100
        breakdown_prob = breakdown_prob / total * 100
        sideway_prob = 100 - breakout_prob - breakdown_prob

        return predictions, {
            "Breakout": {"prob": breakout_prob, "target": last_close * 1.05},
            "Breakdown": {"prob": breakdown_prob, "target": last_close * 0.95},
            "Sideway": {"prob": sideway_prob, "target": last_close}
        }
    
    def _apply_sanity_checks(self, predictions: np.ndarray, last_value: float) -> np.ndarray:
        """
        Áp dụng kiểm tra hợp lý cho dự báo để loại bỏ giá trị phi lý.
        
        Args:
            predictions: Mảng các giá trị dự báo
            last_value: Giá trị gần nhất
            
        Returns:
            Mảng các giá trị dự báo đã điều chỉnh
        """
        adjusted_predictions = np.array(predictions)
        
        # Lọc 1: Giới hạn biến động tối đa mỗi bước (không quá 5%)
        max_change_per_step = 0.05
        for i in range(1, len(adjusted_predictions)):
            max_change = last_value * max_change_per_step
            if abs(adjusted_predictions[i] - adjusted_predictions[i-1]) > max_change:
                direction = 1 if adjusted_predictions[i] > adjusted_predictions[i-1] else -1
                adjusted_predictions[i] = adjusted_predictions[i-1] + direction * max_change
        
        # Lọc 2: Giới hạn biến động tổng thể so với giá hiện tại
        max_total_change_pct = 0.15  # Giới hạn 15% tổng biến động
        for i in range(len(adjusted_predictions)):
            max_total_change = last_value * max_total_change_pct
            if abs(adjusted_predictions[i] - last_value) > max_total_change:
                direction = 1 if adjusted_predictions[i] > last_value else -1
                adjusted_predictions[i] = last_value + direction * max_total_change
        
        # Lọc 3: Không cho phép giá âm
        adjusted_predictions = np.maximum(adjusted_predictions, 0)
        
        return adjusted_predictions

    def evaluate_prophet_performance(self, df: pd.DataFrame, forecast: pd.DataFrame) -> float:
        """
        Đánh giá hiệu suất của mô hình Prophet
        
        Args:
            df: DataFrame chứa dữ liệu thực tế
            forecast: DataFrame chứa dự báo từ Prophet
            
        Returns:
            Điểm hiệu suất từ 0 đến 1
        """
        if df.empty or forecast.empty:
            logger.warning("DataFrame trống, không thể đánh giá hiệu suất")
            return 0.0
        
        try:
            # Đảm bảo đồng bộ dữ liệu giữa dữ liệu thực tế và dự báo
            df_dates = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            forecast_dates = pd.to_datetime(forecast['ds']).dt.strftime('%Y-%m-%d')
            
            actual = []
            predicted = []
            
            # Tìm các ngày trùng khớp
            common_date_strs = set(df_dates).intersection(set(forecast_dates))
            
            if not common_date_strs:
                logger.error("Không thể đồng bộ ngày giữa df và forecast: không có ngày trùng khớp")
                return 0.0
            
            # Lấy giá trị cho các ngày trùng khớp
            for date_str in common_date_strs:
                # Tìm index trong df có cùng ngày
                df_idx = df_dates[df_dates == date_str].index
                # Tìm index trong forecast có cùng ngày
                forecast_idx = forecast_dates[forecast_dates == date_str].index
                
                if len(df_idx) > 0 and len(forecast_idx) > 0:
                    actual.append(df.iloc[df_idx[0]]['Close'])
                    predicted.append(forecast.iloc[forecast_idx[0]]['yhat'])
            
            if len(actual) < 5:
                logger.warning(f"Không đủ dữ liệu để đánh giá (chỉ có {len(actual)} ngày trùng khớp)")
                return 0.1
            
            actual = np.array(actual)
            predicted = np.array(predicted)
            
            # Đánh giá bằng nhiều chỉ số
            mse = np.mean((actual - predicted) ** 2)
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            
            # Kiểm tra dự báo phi lý
            is_unrealistic = False
            avg_price = np.mean(actual)
            max_deviation = np.max(np.abs(predicted - actual) / actual)
            
            # Tính biên độ thông thường của cổ phiếu
            avg_daily_change = np.mean(np.abs(np.diff(actual) / actual[:-1])) * 100
            volatility = np.std(np.diff(actual) / actual[:-1]) * 100
            
            # Nếu có dự báo lệch quá 50% hoặc lớn hơn 3 lần biên độ biến động thông thường
            if max_deviation > 0.5 or max_deviation > (3 * avg_daily_change / 100):
                is_unrealistic = True
                logger.warning(f"Phát hiện dự báo phi lý: độ lệch tối đa {max_deviation:.2f}, biên độ TB: {avg_daily_change:.2f}%, volatility: {volatility:.2f}%")
            
            # Phát hiện xu hướng không thực tế (ngược với dữ liệu gần đây)
            recent_trend = np.polyfit(range(min(5, len(actual))), actual[-min(5, len(actual)):], 1)[0]
            forecast_trend = np.polyfit(range(min(5, len(predicted))), predicted[-min(5, len(predicted)):], 1)[0]
            
            # Nếu xu hướng ngược nhau và độ dốc lớn
            if (recent_trend * forecast_trend < 0) and (abs(forecast_trend) > 2 * abs(recent_trend)):
                is_unrealistic = True
                logger.warning(f"Phát hiện xu hướng dự báo không thực tế: {forecast_trend:.4f} vs {recent_trend:.4f}")
            
            # Nếu phát hiện dự báo phi lý, giảm điểm hiệu suất
            if is_unrealistic:
                logger.warning("Phát hiện dự báo Prophet phi lý, giảm điểm hiệu suất")
                return 0.1  # Giảm mạnh điểm hiệu suất
                
            # Đánh giá hiệu suất dựa trên MSE và MAPE
            # 1/(1+MSE) cho giá trị từ 0-1, càng gần 1 càng tốt
            mse_score = 1 / (1 + mse)
            # MAPE < 10% là tốt, >30% là kém
            mape_score = max(0, 1 - (mape / 30))
            
            # Kết hợp hai điểm số (70% MSE, 30% MAPE)
            performance = 0.7 * mse_score + 0.3 * mape_score
            
            return performance
            
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá hiệu suất Prophet: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    def forecast_with_prophet(self, df: pd.DataFrame, periods: int = 7) -> Tuple[pd.DataFrame, Prophet, float]:
        """
        Dự báo giá bằng mô hình Prophet với các cải tiến
        
        Args:
            df: DataFrame chứa dữ liệu giá
            periods: Số kỳ cần dự báo
            
        Returns:
            Tuple gồm forecast, model và hiệu suất
        """
        if df.empty or len(df) < 50:
            raise ValueError("Không đủ dữ liệu để dự báo Prophet (cần ít nhất 50 điểm dữ liệu)")
        df = df.copy()
        if getattr(df.index, 'tz', None) is not None:
            df.index = df.index.tz_convert(None)
            df.index = df.index.tz_localize(None)
            
        # Fix for "cannot insert date, already exists" error
        if 'date' in df.columns:
            # If 'date' is already a column, use it directly
            df_reset = df.copy()
            df_reset.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)
        else:
            # Otherwise reset the index as before
            df_reset = df.reset_index().rename(columns={'date': 'ds', 'close': 'y'})
            
        df_reset['ds'] = pd.to_datetime(df_reset['ds']).dt.tz_localize(None)
        
        # Cải thiện mô hình Prophet với các tham số điều chỉnh
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative'
        )
        
        # Thêm các tham số mùa vụ phù hợp với thị trường chứng khoán
        model.add_seasonality(name='weekly', period=5, fourier_order=5)
        model.add_seasonality(name='monthly', period=21, fourier_order=5)
        
        # Thêm giới hạn tăng trưởng
        last_value = df['close'].iloc[-1]
        growth_cap = last_value * 1.5
        growth_floor = last_value * 0.5
        df_reset['cap'] = growth_cap
        df_reset['floor'] = growth_floor
        
        # Huấn luyện mô hình với các ràng buộc
        model.fit(df_reset[['ds', 'y', 'cap', 'floor']])
        
        # Tạo dữ liệu dự báo
        future = model.make_future_dataframe(periods=periods)
        future['cap'] = growth_cap
        future['floor'] = growth_floor
        
        # Dự báo
        forecast = model.predict(future)
        
        # Đánh giá hiệu suất
        performance = self.evaluate_prophet_performance(df, forecast)
        
        # Kiểm tra và điều chỉnh dự báo phi lý
        if performance < 0.3:  # Hiệu suất kém
            logger.warning("Hiệu suất Prophet thấp, áp dụng lọc dự báo phi lý")
            # Điều chỉnh dự báo: Sử dụng phương pháp hồi quy tuyến tính đơn giản thay thế
            last_n = min(30, len(df))
            X = np.arange(last_n).reshape(-1, 1)
            y = df['close'].tail(last_n).values
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(X, y)
            future_idx = np.arange(last_n, last_n + periods).reshape(-1, 1)
            linear_pred = reg.predict(future_idx)
            
            # Giới hạn dự báo trong phạm vi hợp lý
            for i in range(len(linear_pred)):
                day = i + 1
                max_change = last_value * (0.02 * day)  # Tối đa 2% mỗi ngày
                if abs(linear_pred[i] - last_value) > max_change:
                    direction = 1 if linear_pred[i] > last_value else -1
                    linear_pred[i] = last_value + direction * max_change
            
            # Ghi đè giá trị dự báo
            future_indices = forecast.index[-periods:]
            forecast.loc[future_indices, 'yhat'] = linear_pred
            forecast.loc[future_indices, 'yhat_lower'] = linear_pred * 0.95
            forecast.loc[future_indices, 'yhat_upper'] = linear_pred * 1.05
            
            performance = 0.5  # Đặt hiệu suất mặc định cho mô hình hồi quy tuyến tính
        
        return forecast, model, performance

    def predict_xgboost_signal(self, df: pd.DataFrame, features: List[str]) -> Tuple[str, float]:
        """
        Dự báo tín hiệu giao dịch bằng XGBoost
        
        Args:
            df: DataFrame chứa dữ liệu đã tính toán chỉ báo kỹ thuật
            features: Danh sách các đặc trưng sử dụng để dự báo
            
        Returns:
            Tuple của tín hiệu dự báo và độ chính xác
        """
        if df.empty or len(df) < 50:
            raise ValueError("Không đủ dữ liệu để dự báo XGBoost (cần ít nhất 50 điểm dữ liệu)")
        df = df.copy()
        df['target'] = (df['close'] > df['close'].shift(1)).astype(int)
        X = df[features].shift(1)
        y = df['target']
        valid_idx = X.notna().all(axis=1) & y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        if len(X) < 50:
            return "Không đủ dữ liệu để dự báo XGBoost", 0.0
        X_train = X.iloc[:-1]
        y_train = y.iloc[:-1]
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        pred = model.predict(X.iloc[-1:])[0]
        actual = y.iloc[-1]
        accuracy = 1 if pred == actual else 0
        return "Tăng" if pred == 1 else "Giảm", accuracy

    def buy_and_sell_signal(self, df: pd.DataFrame, forecast: pd.DataFrame, days_ahead: int = 3) -> Dict[str, List[Any]]:
        """
        Tạo tín hiệu mua/bán dựa trên dự báo
        
        Args:
            df: DataFrame chứa dữ liệu lịch sử
            forecast: DataFrame chứa dự báo
            days_ahead: Số ngày nhìn về phía trước để dự đoán xu hướng
            
        Returns:
            Dictionary chứa tín hiệu mua/bán và ngày tương ứng
        """
        try:
            if df.empty or forecast.empty:
                logger.warning("DataFrame trống, không thể tạo tín hiệu mua/bán")
                return {"buy_dates": [], "sell_dates": []}
                
            # Chỉ lấy dự báo tương lai (ngày sau ngày cuối cùng trong df)
            if 'Date' in df.columns:
                last_actual_date = pd.to_datetime(df['Date'].iloc[-1])
            else:
                last_actual_date = pd.to_datetime(df.index[-1])
                
            future_forecast = forecast[pd.to_datetime(forecast['ds']) > last_actual_date]
            
            if future_forecast.empty:
                logger.warning("Không có dự báo tương lai sau ngày cuối cùng trong dữ liệu lịch sử")
                return {"buy_dates": [], "sell_dates": []}
                
            # Tính toán tín hiệu mua/bán dựa trên xu hướng dự báo
            buy_dates = []
            sell_dates = []
            
            # Lấy giá trị dự báo
            dates = future_forecast['ds'].values
            predictions = future_forecast['yhat'].values
            upper_bounds = future_forecast['yhat_upper'].values if 'yhat_upper' in future_forecast.columns else None
            lower_bounds = future_forecast['yhat_lower'].values if 'yhat_lower' in future_forecast.columns else None
            
            # Lấy giá cuối cùng từ dữ liệu thực tế
            if 'Close' in df.columns:
                last_price = df['Close'].iloc[-1]
            elif 'close' in df.columns:
                last_price = df['close'].iloc[-1]
            else:
                logger.warning("Không tìm thấy cột 'Close' hoặc 'close', dùng giá trị mặc định")
                last_price = predictions[0] if len(predictions) > 0 else 0
                
            # Tính toán volatility từ dữ liệu lịch sử để làm ngưỡng
            if 'Close' in df.columns:
                price_col = 'Close'
            elif 'close' in df.columns:
                price_col = 'close'
            else:
                logger.warning("Không tìm thấy cột giá, không thể tính biến động")
                return {"buy_dates": [], "sell_dates": []}
                
            # Tính biến động
            recent_df = df.tail(30)  # Dùng 30 ngày gần đây
            recent_df['pct_change'] = recent_df[price_col].pct_change()
            volatility = recent_df['pct_change'].std()
            avg_daily_change = recent_df['pct_change'].abs().mean()
            
            # Ngưỡng % thay đổi giá để tạo tín hiệu mua/bán (điều chỉnh theo biến động)
            threshold = max(0.015, 1.5 * avg_daily_change)  # Tối thiểu 1.5%, hoặc 1.5 lần biến động trung bình
            
            # Cải tiến: xem xét cả xu hướng ngắn hạn và dài hạn
            for i in range(min(days_ahead, len(dates) - 1)):
                # Tính % thay đổi từ giá hiện tại đến giá dự báo
                if i == 0:
                    # So sánh với giá cuối cùng trong dữ liệu thực tế
                    pct_change = (predictions[i] - last_price) / last_price
                else:
                    # So sánh với dự báo trước đó
                    pct_change = (predictions[i] - predictions[i-1]) / predictions[i-1]
                
                # Tính xu hướng trong 3 ngày tới (nếu có đủ dữ liệu)
                future_window = min(3, len(predictions) - i)
                if future_window > 1:
                    future_trend = (predictions[i + future_window - 1] - predictions[i]) / predictions[i]
                else:
                    future_trend = 0
                
                # Tạo tín hiệu mua khi giá tăng vượt ngưỡng
                if pct_change > threshold and future_trend > 0:
                    # Kiểm tra thêm confidence interval nếu có
                    if upper_bounds is not None and lower_bounds is not None:
                        # Chỉ mua nếu cả khoảng tin cậy dưới cũng tăng
                        lower_pct_change = (lower_bounds[i] - last_price) / last_price if i == 0 else (lower_bounds[i] - lower_bounds[i-1]) / lower_bounds[i-1]
                        
                        if lower_pct_change > 0:  # Khoảng tin cậy dưới cũng tăng
                            buy_dates.append(dates[i])
                    else:
                        buy_dates.append(dates[i])
                
                # Tạo tín hiệu bán khi giá giảm vượt ngưỡng
                if pct_change < -threshold and future_trend < 0:
                    # Kiểm tra thêm confidence interval nếu có
                    if upper_bounds is not None and lower_bounds is not None:
                        # Chỉ bán nếu cả khoảng tin cậy trên cũng giảm
                        upper_pct_change = (upper_bounds[i] - last_price) / last_price if i == 0 else (upper_bounds[i] - upper_bounds[i-1]) / upper_bounds[i-1]
                        
                        if upper_pct_change < 0:  # Khoảng tin cậy trên cũng giảm
                            sell_dates.append(dates[i])
                    else:
                        sell_dates.append(dates[i])
            
            # Chuẩn hóa định dạng ngày
            buy_dates = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in buy_dates]
            sell_dates = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in sell_dates]
            
            logger.info(f"Tạo tín hiệu mua: {len(buy_dates)} ngày, bán: {len(sell_dates)} ngày")
            
            return {
                "buy_dates": buy_dates,
                "sell_dates": sell_dates
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo tín hiệu mua/bán: {str(e)}")
            logger.error(traceback.format_exc())
            return {"buy_dates": [], "sell_dates": []}

# ---------- AI & BÁO CÁO ----------
class AIAnalyzer:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        self.tech_analyzer = TechnicalAnalyzer()
        self.predictor = EnhancedPredictor()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_content(self, prompt: str) -> Any:
        return await self.gemini_model.generate_content_async(prompt)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_with_groq(self, technical_data: Dict[str, Any]) -> Dict[str, Any]:
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY không được thiết lập")
            return {}
        
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.strftime('%Y-%m-%d')
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        # Chuẩn bị dữ liệu kỹ thuật
        prompt = """
        Phân tích dữ liệu kỹ thuật sau đây để nhận diện mô hình giá (pattern) theo Elliott Wave, Wyckoff, và các mô hình nến Nhật.
        Hãy xác định chính xác các mô hình đang xuất hiện trên biểu đồ dựa trên dữ liệu được cung cấp.
        
        Dữ liệu:
        {}
        
        Trả về kết quả dưới dạng JSON với định dạng:
        ```json
        {{
          "patterns": [
            {{"name": "Tên Mô Hình 1", "description": "Mô tả ngắn gọn"}},
            {{"name": "Tên Mô Hình 2", "description": "Mô tả ngắn gọn"}}
          ]
        }}
        ```
        
        Hãy giữ mô tả ngắn gọn, tối đa 100 ký tự. Chỉ trả lại JSON, không giải thích thêm.
        """.format(json.dumps(technical_data, ensure_ascii=False, cls=CustomJSONEncoder))
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1200  # Tăng max_tokens để tránh bị cắt giữa chừng
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    error_content = await resp.text()
                    logger.error(f"Groq API trả về lỗi {resp.status} khi phân tích dữ liệu kỹ thuật. Nội dung lỗi: {error_content}")
                    if resp.status == 429:
                        logger.warning("Gặp lỗi 429 - Quá nhiều yêu cầu, thử lại...")
                    return {}
                result_text = await resp.text()
                try:
                    result = json.loads(result_text)
                    # Extract the content from the message
                    message_content = result['choices'][0]['message']['content']
                    finish_reason = result['choices'][0]['finish_reason']
                    
                    # Kiểm tra nếu phản hồi bị cắt do độ dài
                    if finish_reason == "length":
                        logger.warning("Phản hồi từ Groq bị cắt do giới hạn độ dài. Cố gắng phục hồi JSON...")
                        # Trả về mẫu JSON đơn giản nếu không thể phục hồi
                        return {
                            "patterns": [
                                {"name": "Phản hồi bị cắt", "description": "Không thể phân tích đầy đủ do phản hồi từ Groq bị cắt"},
                                {"name": "Dữ liệu một phần", "description": "Chỉ trả về thông tin một phần, vui lòng thử lại"}
                            ]
                        }
                    
                    # Extract JSON portion from the content (between triple backticks)
                    import re
                    
                    # First, check if there's a <think> tag and extract content after it
                    think_match = re.search(r'<think>[\s\S]*?</think>([\s\S]*)', message_content)
                    if think_match:
                        # Use the content after </think> tag
                        message_content = think_match.group(1).strip()
                    
                    # Try to find JSON in code blocks
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', message_content)
                    if json_match:
                        json_text = json_match.group(1).strip()
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Lỗi khi parse JSON trong code block: {str(e)}")
                            # Cố gắng sửa JSON không hoàn chỉnh
                            fixed_json = self._fix_incomplete_json(json_text)
                            if fixed_json:
                                return fixed_json
                    
                    # Try to find JSON in any code blocks
                    json_match = re.search(r'```\s*([\s\S]*?)\s*```', message_content)
                    if json_match:
                        json_text = json_match.group(1).strip()
                        if json_text.startswith('{') and ('}' in json_text):
                            try:
                                return json.loads(json_text)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Lỗi khi parse JSON trong code block khác: {str(e)}")
                                # Cố gắng sửa JSON không hoàn chỉnh
                                fixed_json = self._fix_incomplete_json(json_text)
                                if fixed_json:
                                    return fixed_json
                    
                    # Try to find JSON directly in the content
                    json_match = re.search(r'({[\s\S]*})', message_content)
                    if json_match:
                        json_text = json_match.group(1).strip()
                        try:
                            return json.loads(json_text)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Lỗi khi parse JSON trực tiếp: {str(e)}")
                            # Cố gắng sửa JSON không hoàn chỉnh
                            fixed_json = self._fix_incomplete_json(json_text)
                            if fixed_json:
                                return fixed_json
                        
                    # If message is already JSON formatted
                    if message_content.strip().startswith('{') and message_content.strip().endswith('}'):
                        try:
                            return json.loads(message_content)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Lỗi khi parse message_content dạng JSON: {str(e)}")
                            # Cố gắng sửa JSON không hoàn chỉnh
                            fixed_json = self._fix_incomplete_json(message_content)
                            if fixed_json:
                                return fixed_json
                    
                    logger.error(f"Không thể tìm thấy phần JSON trong nội dung phản hồi: {message_content}")
                    return {"patterns": [{"name": "JSON không đúng định dạng", "description": "Không thể tìm thấy JSON trong nội dung phản hồi"}]}
                except (KeyError, json.JSONDecodeError) as e:
                    logger.error(f"Lỗi parsing phản hồi từ Groq: {str(e)}. Nội dung phản hồi: {result_text}")
                    # Cố gắng tìm kiếm và sửa JSON trong văn bản gốc
                    fixed_json = self._extract_and_fix_json(result_text)
                    if fixed_json:
                        return fixed_json
                    return {"patterns": [{"name": "Lỗi parsing", "description": "Không thể phân tích phản hồi từ Groq"}]}
    
    def _fix_incomplete_json(self, json_text: str) -> Optional[Dict[str, Any]]:
        """Cố gắng sửa JSON không hoàn chỉnh"""
        try:
            # Kiểm tra xem JSON có dấu hiệu của patterns array không
            if '"patterns"' in json_text and '"name"' in json_text:
                # Try to extract patterns array
                import re
                patterns_match = re.search(r'"patterns"\s*:\s*\[\s*(.*?)\s*\]', json_text, re.DOTALL)
                if patterns_match:
                    patterns_content = patterns_match.group(1)
                    # Split by objects
                    pattern_objects = re.findall(r'{(.*?)}', patterns_content, re.DOTALL)
                    fixed_patterns = []
                    for obj in pattern_objects:
                        name_match = re.search(r'"name"\s*:\s*"([^"]*)"', obj)
                        desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', obj)
                        
                        if name_match and desc_match:
                            fixed_patterns.append({
                                "name": name_match.group(1),
                                "description": desc_match.group(1)
                            })
                    
                    if fixed_patterns:
                        return {"patterns": fixed_patterns}
            
            # If it's obviously not patterns format, try general purpose JSON repair
            # Remove all newlines and extra spaces
            cleaned = re.sub(r'\s+', ' ', json_text).strip()
            # Fix missing quotes around property names
            cleaned = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', cleaned)
            # Fix trailing commas
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # If still failing, return None to indicate failure
                return None
                
        except Exception as e:
            logger.error(f"Lỗi khi cố gắng sửa JSON: {str(e)}")
            return None
    
    def _extract_and_fix_json(self, result_text: str) -> Optional[Dict[str, Any]]:
        """Cố gắng trích xuất và sửa JSON từ phản hồi API gốc"""
        try:
            import re
            
            # Look for JSON pattern in the entire response
            json_pattern = re.search(r'({[\s\S]*?})(?=[\s,.]|$)', result_text)
            if not json_pattern:
                # If no pattern found, look for any {...} blocks
                json_pattern = re.search(r'{[\s\S]*?}', result_text)
            
            if json_pattern:
                potential_json = json_pattern.group(0)
                try:
                    return json.loads(potential_json)
                except json.JSONDecodeError:
                    # Try to fix and parse again
                    fixed_json = self._fix_incomplete_json(potential_json)
                    if fixed_json:
                        return fixed_json
            
            # If no JSON-like patterns found, look for patterns array content
            patterns_content = re.search(r'"patterns"\s*:\s*\[\s*(.*?)\s*\]', result_text, re.DOTALL)
            if patterns_content:
                pattern_matches = re.findall(r'{(.*?)}', patterns_content.group(1), re.DOTALL)
                fixed_patterns = []
                
                for pattern_match in pattern_matches:
                    name_match = re.search(r'"name"\s*:\s*"([^"]*)"', pattern_match)
                    desc_match = re.search(r'"description"\s*:\s*"([^"]*)"', pattern_match)
                    
                    if name_match and desc_match:
                        fixed_patterns.append({
                            "name": name_match.group(1),
                            "description": desc_match.group(1)
                        })
                
                if fixed_patterns:
                    return {"patterns": fixed_patterns}
            
            # If everything fails, return default
            return {
                "patterns": [
                    {"name": "JSON extraction failed", "description": "Could not extract valid JSON from response"}
                ]
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất JSON: {str(e)}")
            return {
                "patterns": [
                    {"name": "Error", "description": f"Exception when extracting JSON: {str(e)}"}
                ]
            }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_report_with_groq(self, prompt: str) -> str:
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY không được thiết lập")
            return "Error: GROQ API key không được thiết lập"
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2500,
            "temperature": 0.7
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    error_content = await resp.text()
                    logger.error(f"Groq API error {resp.status}: {error_content}")
                    if resp.status == 429:
                        logger.warning("Rate limit hit, retrying...")
                        raise Exception("Rate limit hit")
                    return f"Error: Groq API trả về lỗi {resp.status}"
                
                result_text = await resp.text()
                try:
                    result = json.loads(result_text)
                    return result['choices'][0]['message']['content']
                except Exception as e:
                    logger.error(f"Lỗi parsing phản hồi từ Groq: {str(e)}")
                    return f"Error: Không thể parse phản hồi từ Groq API"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def try_gemini_then_groq(self, prompt: str) -> str:
        """Thử phân tích với Gemini Pro, nếu thất bại thì chuyển sang Groq"""
        try:
            if GEMINI_API_KEY:
                response = await self.generate_content(prompt)
                return response.text
            else:
                logger.warning("GEMINI_API_KEY không được thiết lập, chuyển sang Groq")
                return await self.generate_report_with_groq(prompt)
        except Exception as e:
            logger.error(f"Lỗi với Gemini API: {str(e)}, chuyển sang Groq")
            return await self.generate_report_with_groq(prompt)
    
    def analyze_price_action(self, df: pd.DataFrame) -> str:
        """Phân tích hành động giá với các mẫu nến Nhật"""
        if df.empty or len(df) < 5:
            return "Không đủ dữ liệu để phân tích hành động giá"
        
        recent = df.tail(5)
        last_close = recent['close'].iloc[-1]
        prev_close = recent['close'].iloc[-2]
        
        trend = "tăng" if last_close > prev_close else "giảm" if last_close < prev_close else "đi ngang"
        pct_change = ((last_close - prev_close) / prev_close) * 100
        
        # Kiểm tra một số mẫu nến phổ biến
        patterns = []
        
        # Doji (thân nến rất nhỏ)
        last_candle = recent.iloc[-1]
        if abs(last_candle['close'] - last_candle['open']) / (last_candle['high'] - last_candle['low']) < 0.1:
            patterns.append("Doji")
        
        # Hammer (nến búa)
        if (last_candle['high'] - max(last_candle['open'], last_candle['close'])) / (max(last_candle['open'], last_candle['close']) - last_candle['low']) < 0.3:
            if (max(last_candle['open'], last_candle['close']) - last_candle['low']) / (last_candle['high'] - last_candle['low']) > 0.6:
                patterns.append("Hammer")
                
        # Engulfing (nến bao phủ)
        last_candle = recent.iloc[-1]
        prev_candle = recent.iloc[-2]
        if (last_candle['open'] < prev_candle['close'] and last_candle['close'] > prev_candle['open']):
            patterns.append("Bullish Engulfing")
        elif (last_candle['open'] > prev_candle['close'] and last_candle['close'] < prev_candle['open']):
            patterns.append("Bearish Engulfing")
            
        patterns_str = ", ".join(patterns) if patterns else "không có mẫu nến đặc biệt"
        
        return f"Giá {trend} {abs(pct_change):.2f}% với mẫu nến: {patterns_str}"
    
    async def generate_report(self, dfs: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], symbol: str, fundamental_data: Dict[str, Any], outlier_reports: Dict[str, str]) -> str:
        """Tạo báo cáo phân tích toàn diện"""
        try:
            # Kiểm tra và sắp xếp dữ liệu
            if not dfs or not any(dfs.values()):
                return "Không có đủ dữ liệu để phân tích"
            
            # Lấy khung thời gian chính (1D)
            if '1D' in dfs and dfs['1D'][1] is not None and not dfs['1D'][1].empty:
                df_main = dfs['1D'][1]
                # Phân tích kỹ thuật
                df_with_indicators = self.tech_analyzer.calculate_indicators(df_main)
                
                # Phân tích đa khung thời gian
                timeframe_indicators = self.tech_analyzer.calculate_multi_timeframe_indicators(dfs)
                
                # Dự báo giá
                if len(df_main) >= 50:
                    try:
                        predictions, scenarios = self.predictor.hybrid_predict(df_main)
                        forecast_text = f"\n- Dự báo giá: {predictions[-1]:.2f} (sau 5 ngày)"
                        scenario_text = "\n- Kịch bản:\n"
                        for name, data in scenarios.items():
                            scenario_text += f"  + {name}: {data['prob']:.1f}% (giá mục tiêu: {data['target']:.2f})\n"
                    except Exception as e:
                        logger.error(f"Lỗi khi dự báo giá: {str(e)}")
                        forecast_text = "\n- Dự báo giá: Không thể dự báo do lỗi mô hình"
                        scenario_text = ""
                else:
                    forecast_text = "\n- Dự báo giá: Không đủ dữ liệu để dự báo"
                    scenario_text = ""
                
                # Phân tích hành động giá gần đây
                price_action = self.analyze_price_action(df_main)
                
                # Thu thập tin tức
                news_items = await get_news(symbol, limit=3)
                news_text = "\n\n### Tin tức gần đây:\n"
                if news_items:
                    for idx, news in enumerate(news_items, 1):
                        news_text += f"{idx}. {news['title']}\n"
                else:
                    news_text += "Không tìm thấy tin tức liên quan gần đây."
                
                # Tóm tắt thông tin cơ bản
                is_index = DataValidator.is_index(symbol)
                
                if is_index:
                    entity_type = "chỉ số"
                    fundamental_summary = f"{symbol} là chỉ số chứng khoán, không có dữ liệu cơ bản như P/E, EPS, v.v."
                else:
                    entity_type = "cổ phiếu"
                    if fundamental_data and 'error' not in fundamental_data:
                        fundamental_summary = deep_fundamental_analysis(fundamental_data)
                    else:
                        fundamental_summary = "Không có thông tin cơ bản cho mã này."
                
                # Kiểm tra outliers
                outlier_summary = ""
                for tf, report in outlier_reports.items():
                    if "Không phát hiện giá trị ngoại lệ" not in report:
                        outlier_summary += f"\n- Khung thời gian {tf}: Phát hiện outlier"
                
                if not outlier_summary:
                    outlier_summary = "\nKhông phát hiện dữ liệu bất thường"
                
                # Tạo prompt cho AI
                prompt = f"""
                Hãy phân tích chi tiết {entity_type} {symbol} dựa trên các thông tin sau đây. Văn phong rõ ràng, chuyên nghiệp, phù hợp với một báo cáo phân tích tài chính.
                
                ### Thông tin cơ bản:
                {fundamental_summary}
                
                ### Phân tích kỹ thuật:
                - RSI (14): {df_with_indicators['rsi'].iloc[-1]:.2f}
                - MACD: {df_with_indicators['macd'].iloc[-1]:.4f}
                - Tín hiệu MACD: {df_with_indicators['signal'].iloc[-1]:.4f}
                - SMA20: {df_with_indicators['sma20'].iloc[-1]:.2f}
                - SMA50: {df_with_indicators['sma50'].iloc[-1]:.2f}
                - Khoảng Bollinger: {df_with_indicators['bb_low'].iloc[-1]:.2f} - {df_with_indicators['bb_high'].iloc[-1]:.2f}

                ### Phân tích đa khung thời gian:
                {json.dumps(timeframe_indicators, indent=2, ensure_ascii=False)}
                
                ### Hành động giá gần đây:
                {price_action}
                {forecast_text}
                {scenario_text}
                
                ### Dữ liệu bất thường:
                {outlier_summary}
                
                {news_text}
                
                Hãy đưa ra nhận định tổng quan về {entity_type} này, phân tích xu hướng và đưa ra khuyến nghị cho nhà đầu tư. Tập trung vào:
                1. Tóm tắt tình hình hiện tại
                2. Phân tích xu hướng kỹ thuật ngắn, trung và dài hạn
                3. Nhận định về dữ liệu cơ bản (nếu có)
                4. Khuyến nghị: Nên mua, bán hay giữ, với mức giá mục tiêu và dừng lỗ nếu có thể
                
                Hãy phân tích theo format sau:
                
                ## PHÂN TÍCH {symbol}
                
                ### Tổng quan
                (tóm tắt ngắn gọn tình hình hiện tại với thông tin cơ bản)
                
                ### Phân tích kỹ thuật
                (phân tích chỉ báo và hành động giá)
                
                ### Xu hướng
                (nhận định xu hướng ngắn, trung và dài hạn)
                
                ### Khuyến nghị
                (đề xuất hành động cụ thể)
                """
                
                # Gọi API Gemini Pro (fall back to Groq nếu thất bại)
                report = await self.try_gemini_then_groq(prompt)
                return report
            else:
                return "Không có dữ liệu khung thời gian ngày (1D) để phân tích"
        except Exception as e:
            error_info = traceback.format_exc()
            logger.error(f"Lỗi khi tạo báo cáo: {str(e)}\n{error_info}")
            return f"Lỗi khi tạo báo cáo phân tích: {str(e)}"

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

# ---------- THU THẬP TIN TỨC ----------
async def get_news(symbol: str = None, limit: int = 3) -> list:
    """
    Thu thập tin tức liên quan đến chứng khoán từ nhiều nguồn RSS.
    
    Args:
        symbol: Mã chứng khoán cần thu thập tin tức (None để lấy tin thị trường chung)
        limit: Số lượng tin tức tối đa cần lấy
        
    Returns:
        Danh sách các tin tức đã thu thập
    """
    # Tạo khóa cache dựa trên symbol và ngày hiện tại
    cache_key = f"news_{symbol}_{limit}_{datetime.now(TZ).strftime('%Y%m%d')}" if symbol else f"news_market_{limit}_{datetime.now(TZ).strftime('%Y%m%d')}"
    
    # Kiểm tra cache Redis
    cached_news = await redis_manager.get(cache_key)
    if cached_news is not None:
        logger.debug(f"Sử dụng tin tức từ cache cho {'mã ' + symbol if symbol else 'thị trường chung'}")
        return cached_news
    
    # Danh sách các nguồn RSS
    rss_urls = [
        # Nguồn tiếng Việt
        "https://cafef.vn/thi-truong-chung-khoan.rss",
        "https://cafef.vn/smart-money.rss",
        "https://cafef.vn/tai-chinh-ngan-hang.rss",
        "https://cafef.vn/doanh-nghiep.rss",
        "https://vnexpress.net/rss/kinh-doanh.rss",
        "https://vietnamnet.vn/rss/kinh-doanh.rss",     
        # Nguồn tiếng Anh
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    ]
    
    # Từ khóa tìm kiếm
    market_keywords = [
        # Tiếng Việt
        "thị trường", "chứng khoán", "cổ phiếu", "vn-index", "vnindex", "vn index", 
        "hose", "hnx", "upcom", "trái phiếu", "bluechip", "đầu tư", "tài chính",
        # Tiếng Anh
        "market", "stock", "index", "shares", "trading", "finance", "investment"
    ]
    
    # Từ khóa cho mã cụ thể
    symbol_keywords = []
    if symbol:
        symbol_lower = symbol.lower()
        # Tạo các biến thể của mã để tăng khả năng tìm thấy
        symbol_keywords = [
            symbol_lower,
            f"{symbol_lower} ",
            f" {symbol_lower}",
            f" {symbol_lower} ",
            f"mã {symbol_lower}",
            f"cổ phiếu {symbol_lower}",
            f"{symbol_lower} stock",
            f"{symbol.upper()}"
        ]
    
    # Tạo danh sách để lưu tin tức
    news_list = []
    
    # Tải và xử lý đồng thời các nguồn RSS
    tasks = []
    for url in rss_urls:
        if symbol and "vietstock.vn" in url:
            # Thêm mã vào URL của Vietstock
            url = f"https://vietstock.vn/rss.aspx?Keyword={symbol.upper()}"
        tasks.append(fetch_rss(url, symbol_keywords if symbol else market_keywords, symbol is not None))
    
    # Thực hiện tất cả các yêu cầu một cách đồng thời
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Kết hợp kết quả
    for result in results:
        if isinstance(result, list):
            news_list.extend(result)
    
    # Loại bỏ các tin trùng lặp dựa trên URL
    unique_urls = set()
    unique_news = []
    for news in news_list:
        if news['link'] not in unique_urls:
            unique_urls.add(news['link'])
            unique_news.append(news)
    
    # Sắp xếp theo thời gian (mới nhất trước)
    sorted_news = sorted(unique_news, key=lambda x: x.get("published", ""), reverse=True)
    
    # Giới hạn số lượng tin tức
    limited_news = sorted_news[:limit]
    
    # Nếu không có tin tức, trả về thông báo
    result = limited_news if limited_news else [{"title": "⚠️ Không có tin tức", "link": "#", "summary": ""}]
    
    # Lưu vào cache Redis
    await redis_manager.set(cache_key, result, NEWS_CACHE_EXPIRE)
    
    return result

async def fetch_rss(url: str, keywords: list, is_symbol_search: bool = False) -> list:
    """
    Tải và phân tích nguồn RSS cụ thể
    
    Args:
        url: URL của nguồn RSS
        keywords: Danh sách từ khóa cần tìm kiếm
        is_symbol_search: True nếu đang tìm kiếm cho một mã cụ thể
        
    Returns:
        Danh sách các tin tức phù hợp
    """
    news_items = []
    try:
        # Thiết lập timeout để tránh chờ quá lâu
        timeout = aiohttp.ClientTimeout(total=10)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Không thể tải RSS từ {url}, mã trạng thái: {response.status}")
                    return []
                
                text = await response.text()
                feed = feedparser.parse(text)
                
                if not feed.entries:
                    logger.debug(f"Không có mục nào trong feed từ {url}")
                    return []
                
                # Giới hạn số lượng mục để xử lý (để tránh quá tải)
                entries_to_process = feed.entries[:10]
                
                for entry in entries_to_process:
                    title = entry.get("title", "").strip()
                    link = entry.get("link", "")
                    
                    # Lấy tóm tắt hoặc mô tả
                    summary = entry.get("summary", entry.get("description", "")).strip()
                    
                    # Cắt ngắn tóm tắt nếu quá dài
                    if len(summary) > 200:
                        summary = summary[:197] + "..."
                    
                    # Xử lý thời gian xuất bản
                    published = entry.get("published", entry.get("pubDate", datetime.now(TZ).isoformat()))
                    
                    # Kết hợp nội dung để tìm kiếm từ khóa
                    content = f"{title} {summary}".lower()
                    
                    # Kiểm tra khớp với từ khóa
                    match = False
                    
                    if is_symbol_search:
                        # Khi tìm kiếm mã cụ thể, cần khớp với ít nhất một từ khóa
                        if any(kw in content for kw in keywords):
                            match = True
                    else:
                        # Khi tìm kiếm tin thị trường, cần khớp với ít nhất một từ khóa
                        if any(kw in content for kw in keywords):
                            match = True
                    
                    if match:
                        # Làm sạch HTML trong tóm tắt
                        if "<" in summary and ">" in summary:
                            from bs4 import BeautifulSoup
                            try:
                                summary = BeautifulSoup(summary, "html.parser").get_text()
                            except Exception as e:
                                logger.warning(f"Không thể phân tích HTML trong tóm tắt: {str(e)}")
                        
                        news_items.append({
                            "title": title,
                            "link": link,
                            "summary": summary,
                            "published": published,
                            "source": url.split('/')[2]  # Lấy domain làm nguồn
                        })
        
        return news_items
    except Exception as e:
        logger.error(f"Lỗi khi tải RSS từ {url}: {str(e)}")
        return []

# ---------- AUTO TRAINING ----------
async def auto_train_models():
    """
    Tự động huấn luyện mô hình cho các mã đã có trong lịch sử báo cáo.
    Sử dụng PostgreSQL database.
    """
    # Lấy danh sách mã cần training
    symbols = await db.get_training_symbols()
    if not symbols:
        logger.info("Không có mã nào trong ReportHistory, bỏ qua auto training.")
        return
        
    for symbol in symbols:
        try:
            logger.info(f"Bắt đầu auto training cho mã: {symbol}")
            loader = DataLoader()
            tech_analyzer = TechnicalAnalyzer()
            raw_df, cleaned_df, _ = await loader.load_data(symbol, '1D', 500)
            df = tech_analyzer.calculate_indicators(cleaned_df)
            features = ['sma20', 'sma50', 'sma200', 'rsi', 'macd', 'signal', 'bb_high', 'bb_low', 'ichimoku_a', 'ichimoku_b', 'vwap', 'mfi']
            
            # Huấn luyện mô hình
            prophet_model, prophet_perf = await run_in_thread(train_prophet_model, df)
            xgb_model, xgb_perf = await run_in_thread(train_xgboost_model, df, features)
            
            # Chuẩn bị tham số
            prophet_params = {
                'changepoint_prior_scale': 0.05,
                'seasonality_prior_scale': 10,
                'seasonality_mode': 'multiplicative'
            }
            xgb_params = {
                'features': features,
                'data_points': len(df)
            }
            
            # Lưu mô hình với thông tin version và params
            await db.store_trained_model(
                symbol, 'prophet', prophet_model, prophet_perf, 
                version="1.1", params=prophet_params
            )
            await db.store_trained_model(
                symbol, 'xgboost', xgb_model, xgb_perf,
                version="1.1", params=xgb_params
            )
            
            logger.info(f"Auto training cho {symbol} hoàn tất.")
        except Exception as e:
            logger.error(f"Lỗi auto training cho {symbol}: {str(e)}")
            logger.error(traceback.format_exc())  # Thêm stack trace đầy đủ

async def migrate_database():
    """
    Di chuyển dữ liệu từ cơ sở dữ liệu cũ sang mới, sử dụng trong quá trình cập nhật.
    Cập nhật: Sử dụng SQLAlchemy trực tiếp thay vì AsyncDBManager đã bị loại bỏ.
    """
    old_db_file = 'bot_sieucap_v18.db'
    
    # Kiểm tra xem file cơ sở dữ liệu cũ có tồn tại không
    if os.path.exists(old_db_file):
        logger.info(f"Phát hiện cơ sở dữ liệu cũ ({old_db_file}), bắt đầu di chuyển dữ liệu...")
        try:
            # Sử dụng SQLite trực tiếp để đọc dữ liệu cũ
            import sqlite3
            old_conn = sqlite3.connect(old_db_file)
            old_conn.row_factory = sqlite3.Row
            
            # Khởi tạo PostgreSQL trước
            await init_db()
            
            # Di chuyển approved_users
            cur = old_conn.execute("SELECT user_id, approved_at FROM approved_users")
            users = cur.fetchall()
            for user in users:
                user_id = user['user_id']
                approved_at = user['approved_at'] or datetime.now().isoformat()
                notes = "Migrated from legacy database"
                
                # Dùng DBManager hiện tại thay vì async_db đã bị comment
                await db.add_approved_user(user_id, approved_at, notes)
            
            # Di chuyển report_history
            try:
                cur = old_conn.execute("SELECT symbol, date, report, close_today, close_yesterday, timestamp FROM report_history")
                reports = cur.fetchall()
                for report in reports:
                    symbol = report['symbol']
                    date = report['date']
                    report_text = report['report']
                    close_today = report['close_today']
                    close_yesterday = report['close_yesterday']
                    timeframe = '1D'  # Mặc định cho dữ liệu cũ
                    
                    # Lưu báo cáo vào PostgreSQL
                    await db.save_report_history(
                        symbol, report_text, close_today, close_yesterday, timeframe
                    )
            except sqlite3.OperationalError:
                logger.warning("Bảng report_history không tồn tại trong cơ sở dữ liệu cũ, bỏ qua.")
            
            # Di chuyển trained_models
            try:
                cur = old_conn.execute("SELECT symbol, model_type, model_blob, created_at, performance FROM trained_models")
                models = cur.fetchall()
                for model in models:
                    symbol = model['symbol']
                    model_type = model['model_type']
                    model_blob = model['model_blob']
                    created_at = model['created_at'] or datetime.now().isoformat()
                    performance = model['performance']
                    version = "1.0"  # Mặc định cho dữ liệu cũ
                    
                    # Giải mã và lưu mô hình
                    try:
                        model_obj = pickle.loads(model_blob)
                        # Lưu mô hình vào PostgreSQL
                        await db.store_trained_model(
                            symbol, model_type, model_obj, performance, version
                        )
                    except Exception as e:
                        logger.error(f"Lỗi khi giải mã mô hình {symbol}/{model_type}: {str(e)}")
            except sqlite3.OperationalError:
                logger.warning("Bảng trained_models không tồn tại trong cơ sở dữ liệu cũ, bỏ qua.")
            
            # Đóng kết nối SQLite
            old_conn.close()
            
            # Đổi tên file cũ để tránh di chuyển lại
            backup_file = f"{old_db_file}.bak"
            os.rename(old_db_file, backup_file)
            logger.info(f"Di chuyển dữ liệu thành công, đã sao lưu cơ sở dữ liệu cũ tại {backup_file}")
        except Exception as e:
            logger.error(f"Lỗi khi di chuyển dữ liệu: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        logger.info("Không tìm thấy cơ sở dữ liệu cũ, bỏ qua quá trình di chuyển.")
    
    # Kiểm tra và thêm cột last_active nếu chưa tồn tại
    try:
        from sqlalchemy import inspect, text
        
        # Sử dụng engine từ db.Session
        async with db.Session() as session:
            # Lấy connection
            connection = await session.connection()
            
            # Kiểm tra bảng approved_users
            try:
                # Kiểm tra xem bảng approved_users có tồn tại không và có cột last_active không
                has_last_active = False
                
                # Sử dụng raw SQL để kiểm tra cột trong bảng approved_users
                check_column_sql = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'approved_users' AND column_name = 'last_active'
                """)
                result = await session.execute(check_column_sql)
                columns = result.fetchall()
                
                if not columns:
                    logger.info("Cột last_active chưa tồn tại, đang thêm vào...")
                    # Thêm cột last_active vào bảng
                    add_column_sql = text("""
                        ALTER TABLE approved_users 
                        ADD COLUMN last_active TIMESTAMP NULL
                    """)
                    await session.execute(add_column_sql)
                    await session.commit()
                    logger.info("Đã thêm cột last_active vào bảng approved_users")
                else:
                    logger.info("Cột last_active đã tồn tại")
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra/thêm cột last_active: {str(e)}")
                logger.error(traceback.format_exc())
                
            # Kiểm tra bảng report_history
            try:
                # Kiểm tra xem bảng report_history có tồn tại không và có cột timeframe không
                check_column_sql = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'report_history' AND column_name = 'timeframe'
                """)
                result = await session.execute(check_column_sql)
                columns = result.fetchall()
                
                if not columns:
                    logger.info("Cột timeframe chưa tồn tại trong bảng report_history, đang thêm vào...")
                    # Thêm cột timeframe vào bảng
                    add_timeframe_sql = text("""
                        ALTER TABLE report_history 
                        ADD COLUMN timeframe VARCHAR NOT NULL DEFAULT '1D'
                    """)
                    await session.execute(add_timeframe_sql)
                    await session.commit()
                    logger.info("Đã thêm cột timeframe vào bảng report_history")
                else:
                    logger.info("Cột timeframe đã tồn tại trong bảng report_history")
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra/thêm cột timeframe: {str(e)}")
                logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra/thêm cột trong database: {str(e)}")
        logger.error(traceback.format_exc())

def train_prophet_model(df: pd.DataFrame) -> tuple[Prophet, float]:
    """
    Huấn luyện mô hình Prophet cho dự báo giá
    
    Args:
        df: DataFrame chứa dữ liệu giá
        
    Returns:
        Tuple gồm model Prophet và điểm hiệu suất
    """
    if df.empty or len(df) < 50:
        raise ValueError("Không đủ dữ liệu để huấn luyện Prophet (cần ít nhất 50 điểm dữ liệu)")
    df = df.copy()
    if getattr(df.index, 'tz', None) is not None:
        df.index = df.index.tz_convert(None)
        df.index = df.index.tz_localize(None)
    
    # Fix for "cannot insert date, already exists" error
    if 'date' in df.columns:
        # If 'date' is already a column, use it directly
        df_reset = df.copy()
        df_reset.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)
    else:
        # Otherwise reset the index as before
        df_reset = df.reset_index().rename(columns={'date': 'ds', 'close': 'y'})
    
    df_reset['ds'] = pd.to_datetime(df_reset['ds']).dt.tz_localize(None)
    
    # Thêm các tham số điều chỉnh Prophet cho quá trình training
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        seasonality_mode='multiplicative'
    )
    
    # Thêm các tham số mùa vụ phù hợp với thị trường chứng khoán
    model.add_seasonality(name='weekly', period=5, fourier_order=5)
    model.add_seasonality(name='monthly', period=21, fourier_order=5)
    
    # Thêm các ràng buộc tăng trưởng để tránh dự báo phi lý
    last_value = df['close'].iloc[-1]
    growth_cap = last_value * 1.5
    growth_floor = last_value * 0.5
    df_reset['cap'] = growth_cap
    df_reset['floor'] = growth_floor
    
    model.fit(df_reset[['ds', 'y', 'cap', 'floor']])
    future = model.make_future_dataframe(periods=0)
    future['cap'] = growth_cap
    future['floor'] = growth_floor
    forecast = model.predict(future)
    
    # Đánh giá hiệu suất với nhiều chỉ số
    actual = df['close'].values
    predicted = forecast['yhat'].values
    if len(actual) != len(predicted):
        logger.error(f"Kích thước không khớp trong train_prophet_model: actual ({len(actual)}), predicted ({len(predicted)})")
        return model, 0.0
        
    # Đánh giá bằng nhiều chỉ số
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Kiểm tra dự báo phi lý
    is_unrealistic = False
    avg_price = np.mean(actual)
    max_deviation = np.max(np.abs(predicted - actual) / actual)
    if max_deviation > 0.5:  # Nếu có dự báo lệch quá 50%
        is_unrealistic = True
        logger.warning(f"Phát hiện dự báo phi lý trong quá trình training: độ lệch tối đa {max_deviation:.2f}")
    
    # Tính toán điểm hiệu suất tổng hợp
    mse_score = 1 / (1 + mse)
    mape_score = max(0, 1 - (mape / 30))
    performance = 0.7 * mse_score + 0.3 * mape_score
    
    # Giảm điểm nếu có dự báo phi lý
    if is_unrealistic:
        performance *= 0.5
    
    return model, performance

def train_xgboost_model(df: pd.DataFrame, features: list) -> tuple[xgb.XGBClassifier, float]:
    """
    Huấn luyện mô hình XGBoost cho dự báo tín hiệu giao dịch
    
    Args:
        df: DataFrame chứa dữ liệu đã có chỉ báo kỹ thuật
        features: Danh sách các đặc trưng sử dụng để huấn luyện
        
    Returns:
        Tuple gồm model XGBoost và điểm hiệu suất
    """
    if df.empty or len(df) < 100:
        raise ValueError("Không đủ dữ liệu để huấn luyện XGBoost (cần ít nhất 100 điểm dữ liệu)")
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

# ---------- TELEGRAM COMMAND HANDLERS ----------
async def notify_admin_new_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Thông báo cho admin khi có người dùng mới"""
    user = update.message.from_user
    user_id = str(user.id)
    username = user.username or "Không có username"
    name = f"{user.first_name} {user.last_name if user.last_name else ''}".strip()
    
    try:
        # Chỉ thông báo cho admin nếu người dùng chưa được phê duyệt
        if not await db.is_user_approved(user_id):
            admin_message = (
                f"👤 Người dùng mới yêu cầu truy cập:\n"
                f"ID: {user_id}\n"
                f"Username: @{username}\n"
                f"Tên: {name}\n\n"
                f"Để phê duyệt, sử dụng lệnh:\n"
                f"/approve {user_id}"
            )
            # Gửi thông báo cho admin
            try:
                await context.bot.send_message(chat_id=ADMIN_ID, text=admin_message)
                logger.info(f"Đã thông báo admin về người dùng mới: {user_id}")
            except Exception as e:
                logger.error(f"Không thể gửi thông báo đến admin: {str(e)}")
            
            # Thông báo cho người dùng rằng yêu cầu đã được gửi
            await update.message.reply_text(
                "🔒 Bạn cần được phê duyệt để sử dụng bot.\n"
                "Yêu cầu của bạn đã được gửi đến admin. Vui lòng chờ xác nhận."
            )
            return False
        return True
    except Exception as e:
        logger.error(f"Lỗi trong notify_admin_new_user: {str(e)}")
        logger.error(traceback.format_exc())
        return False

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý tin nhắn văn bản không phải lệnh"""
    user_id = str(update.message.from_user.id)
    
    # Kiểm tra xem người dùng đã được phê duyệt chưa
    if not await db.is_user_approved(user_id):
        # Thông báo admin về người dùng mới
        await notify_admin_new_user(update, context)
        return
    
    # Cập nhật thời gian hoạt động
    try:
        await db.update_user_last_active(user_id)
    except Exception as e:
        logger.error(f"Lỗi cập nhật thời gian hoạt động: {str(e)}")
    
    # Hướng dẫn người dùng
    await update.message.reply_text(
        "Vui lòng sử dụng lệnh /analyze để phân tích cổ phiếu.\n"
        "Ví dụ: /analyze VNM 1D 100\n\n"
        "Hoặc /help để xem hướng dẫn sử dụng."
    )

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /analyze để phân tích chứng khoán"""
    user_id = str(update.message.from_user.id)
    
    # Kiểm tra người dùng đã được phê duyệt chưa
    if not await db.is_user_approved(user_id):
        # Thông báo admin về người dùng mới và thông báo cho người dùng chờ phê duyệt
        await notify_admin_new_user(update, context)
        return
    
    # Cập nhật thời gian hoạt động
    try:
        await db.update_user_last_active(user_id)
    except Exception as e:
        logger.error(f"Lỗi cập nhật thời gian hoạt động: {str(e)}")
        # Tiếp tục xử lý, không dừng lại vì lỗi này
    
    try:
        # Phân tích tham số đầu vào
        args = context.args
        if not args:
            raise ValueError("Nhập mã chứng khoán (e.g., /analyze VNINDEX hoặc /analyze SSI 1D 100)")
        
        # Phân tích tham số
        symbol = args[0].upper()
        timeframe = args[1].upper() if len(args) > 1 else DEFAULT_TIMEFRAME
        num_candles = int(args[2]) if len(args) > 2 else DEFAULT_CANDLES
        
        # Xác thực đầu vào
        symbol = DataValidator.validate_ticker(symbol)
        timeframe = DataValidator.normalize_timeframe(timeframe)
        num_candles = DataValidator.validate_candles(num_candles)
        
        # Thông báo đang xử lý
        await update.message.reply_text(f"⏳ Đang phân tích {symbol} [{timeframe}] với {num_candles} nến...")
        
        # Tải dữ liệu và phân tích
        loader = DataLoader()
        ai_analyzer = AIAnalyzer()
        
        # Danh sách các khung thời gian cần phân tích
        # Luôn bao gồm khung thời gian chính và các khung thời gian liên quan
        timeframes = []
        
        # Nếu là intraday, thêm khung thời gian intraday và daily
        if timeframe in ['5m', '15m', '30m']:
            timeframes = [timeframe, '1h', '1D']
        elif timeframe == '1h':
            timeframes = [timeframe, '4h', '1D']
        elif timeframe == '4h':
            timeframes = [timeframe, '1D', '1W']
        elif timeframe == '1D':
            timeframes = ['1D', '1W', '1M']
        elif timeframe == '1W':
            timeframes = ['1W', '1D', '1M']
        elif timeframe == '1M':
            timeframes = ['1M', '1W', '1D']
        else:
            timeframes = ['1D', '1W', '1M']  # Mặc định
        
        # Đảm bảo khung thời gian chính là đầu tiên
        if timeframe not in timeframes:
            timeframes.insert(0, timeframe)
            
        # Loại bỏ trùng lặp
        timeframes = list(dict.fromkeys(timeframes))
        
        # Tải dữ liệu cho mỗi khung thời gian
        dfs = {}
        outlier_reports = {}
        
        for tf in timeframes:
            # Điều chỉnh số nến dựa vào khung thời gian
            tf_candles = num_candles
            if tf == '1W' and num_candles > 52:
                tf_candles = 52  # Tối đa 1 năm cho tuần
            elif tf == '1M' and num_candles > 60:
                tf_candles = 60  # Tối đa 5 năm cho tháng
                
            raw_df, cleaned_df, outlier_report = await loader.load_data(symbol, tf, tf_candles)
            dfs[tf] = (raw_df, cleaned_df)
            outlier_reports[tf] = outlier_report
            
        # Tải dữ liệu cơ bản
        fundamental_data = await loader.get_fundamental_data(symbol)
        
        # Tải lịch sử báo cáo
        report_history = await db.load_report_history(symbol, timeframe, limit=3)
        
        # Tạo báo cáo phân tích
        report = await ai_analyzer.generate_report(dfs, symbol, fundamental_data, outlier_reports)
        
        # Lưu báo cáo vào lịch sử
        if len(dfs) > 0 and timeframe in dfs:
            raw_df, _ = dfs[timeframe]
            if len(raw_df) >= 2:
                close_today = raw_df['close'].iloc[-1]
                close_yesterday = raw_df['close'].iloc[-2]
                await db.save_report_history(
                    symbol, report, close_today, close_yesterday, timeframe
                )
        
        # Định dạng và gửi báo cáo
        formatted_report = f"<b>📈 Báo cáo {symbol} [{timeframe}] - {datetime.now(TZ).strftime('%d-%m-%Y %H:%M')}</b>\n\n<pre>{html.escape(report)}</pre>"
        await update.message.reply_text(formatted_report, parse_mode='HTML')
        
    except ValueError as e:
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")
    except Exception as e:
        logger.error(f"Lỗi trong analyze_command: {str(e)}")
        logger.error(traceback.format_exc())  # In stack trace chi tiết
        error_msg = "❌ Không thể phân tích mã chứng khoán. Vui lòng thử lại sau hoặc kiểm tra mã chứng khoán."
        if "HTTP 429" in str(e):
            error_msg = "❌ Quá nhiều yêu cầu đến Yahoo Finance. Vui lòng thử lại sau vài phút."
        await update.message.reply_text(error_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý lệnh /help để hiển thị hướng dẫn sử dụng"""
    user_id = str(update.message.from_user.id)
    
    # Kiểm tra người dùng đã được phê duyệt chưa
    if not await db.is_user_approved(user_id):
        await notify_admin_new_user(update, context)
        return
    
    # Cập nhật thời gian hoạt động
    try:
        await db.update_user_last_active(user_id)
    except Exception as e:
        logger.error(f"Lỗi cập nhật thời gian hoạt động: {str(e)}")
    
    help_text = (
        "📚 <b>HƯỚNG DẪN SỬ DỤNG BOT</b>\n\n"
        "<b>Các lệnh có sẵn:</b>\n"
        "• /analyze [Mã] [Khung TG] [Số nến] - Phân tích đa khung thời gian\n"
        "  Ví dụ: /analyze VNM 1D 100\n\n"
        "<b>Khung thời gian hỗ trợ:</b>\n"
        "• Intraday: 5m, 15m, 30m, 1h, 4h\n"
        "• Daily/Weekly/Monthly: 1D, 1W, 1M\n\n"
        "<b>Các lệnh khác:</b>\n"
        "• /getid - Lấy ID người dùng\n"
        "• /help - Hiển thị hướng dẫn này\n"
    )
    
    if user_id == ADMIN_ID:
        help_text += (
            "\n<b>Lệnh dành cho Admin:</b>\n"
            "• /approve [user_id] - Phê duyệt người dùng mới\n"
        )
    
    await update.message.reply_text(help_text, parse_mode='HTML')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý command /start"""
    user_id = str(update.message.from_user.id)
    if user_id == ADMIN_ID and not await db.is_user_approved(user_id):
        await db.add_approved_user(user_id)
        logger.info(f"Admin {user_id} tự động duyệt.")
    if not await db.is_user_approved(user_id):
        await notify_admin_new_user(update, context)
        return
    
    # Cập nhật thời gian hoạt động
    try:
        await db.update_user_last_active(user_id)
    except Exception as e:
        logger.error(f"Lỗi cập nhật thời gian hoạt động trong start: {str(e)}")
        # Tiếp tục xử lý, không dừng lại vì lỗi này
    
    # Tạo phiên bản và thời gian
    version = "V19.2"
    current_time = datetime.now(TZ).strftime("%d/%m/%Y %H:%M:%S")
    
    await update.message.reply_text(
        f"🚀 **{version} - {current_time}**\n"
        "📊 **Lệnh**:\n"
        "- /analyze [Mã] [Khung TG] [Số nến] - Phân tích đa khung.\n"
        "  Ví dụ: /analyze VNM 1D 100\n"
        "  Khung TG: 5m, 15m, 30m, 1h, 4h, 1D, 1W, 1M\n"
        "- /getid - Lấy ID người dùng.\n"
        "- /help - Xem hướng dẫn chi tiết.\n"
        "💡 **Bắt đầu nào!**"
    )

async def get_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý command /getid"""
    user_id = str(update.message.from_user.id)
    await update.message.reply_text(f"ID của bạn: {user_id}")

async def approve_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý command /approve - chỉ admin được dùng"""
    if str(update.message.from_user.id) != ADMIN_ID:
        await update.message.reply_text("❌ Chỉ admin dùng được lệnh này!")
        return
    
    if len(context.args) != 1:
        await update.message.reply_text("❌ Nhập user_id: /approve 123456789")
        return
    
    user_id = context.args[0]
    if not await db.is_user_approved(user_id):
        await db.add_approved_user(user_id, approved_at=datetime.now(TZ), 
                                notes="Approved by admin via command")
        await update.message.reply_text(f"✅ Đã duyệt {user_id}")
    else:
        await update.message.reply_text(f"ℹ️ {user_id} đã được duyệt")

# ---------- MAIN ----------
def main():
    """
    Hàm chính khởi chạy ứng dụng Telegram bot
    """
    # Tạo và khởi tạo application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Thêm các handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("getid", get_id))
    application.add_handler(CommandHandler("approve", approve_user))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))

    # Thiết lập event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Tạo event loop mới nếu không có
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Khởi tạo cơ sở dữ liệu PostgreSQL
    loop.run_until_complete(init_db())
    
    # Thực hiện di chuyển dữ liệu từ cơ sở dữ liệu cũ (nếu có)
    loop.run_until_complete(migrate_database())
    
    # Thiết lập scheduler cho auto training
    scheduler = AsyncIOScheduler(event_loop=loop)
    # Chạy training vào 2 giờ sáng hàng ngày
    scheduler.add_job(auto_train_models, 'cron', hour=2, minute=0, 
                     misfire_grace_time=3600, coalesce=True, max_instances=1)
    scheduler.start()
    logger.info("Auto training scheduler đã khởi động.")

    # Xác định chế độ chạy (webhook trên Render hoặc polling cho local)
    is_render = RENDER_EXTERNAL_URL and RENDER_SERVICE_NAME
    
    if is_render:
        # Thiết lập webhook cho Render
        webhook_url = f"{RENDER_EXTERNAL_URL}/webhook"
        application.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path="webhook",
            webhook_url=webhook_url
        )
        logger.info(f"Bot V19.2 đã khởi chạy trên Render với webhook: {webhook_url}")
    else:
        # Chạy mode polling cho môi trường local
        logger.info("Bot V19.2 đã khởi chạy (chế độ local).")
        application.run_polling()

if __name__ == "__main__":
    # Đảm bảo event loop được khởi tạo đúng cách
    try:
        main()
    except Exception as e:
        logger.error(f"Lỗi khi khởi động bot: {str(e)}")
        # Nếu lỗi event loop, thử tạo mới và khởi động lại
        if "no current event loop" in str(e).lower():
            logger.info("Thử tạo event loop mới và khởi động lại...")
            asyncio.set_event_loop(asyncio.new_event_loop())
            main()