#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot Chứng Khoán Toàn Diện Phiên Bản V18.8.1T (Nâng cấp tải dữ liệu):
- Tối ưu hóa tải và xử lý dữ liệu, tự động làm sạch và sửa lỗi dữ liệu
- Hệ thống kiểm soát chất lượng dữ liệu tự động với nhiều tiêu chí
- Cập nhật dữ liệu gia tăng giảm tải hệ thống và băng thông
- Tự động phát hiện và xử lý ngoại lai, dữ liệu bị thiếu
- Hệ thống tạo đặc trưng phái sinh tự động cho phân tích kỹ thuật
- Sử dụng mô hình deepseek/deepseek-chat-v3-0324:free
- Đảm bảo các chức năng và công nghệ hiện có không bị ảnh hưởng
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
from timestamp_aligner import TimestampAligner

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
    df = df[~pd.to_datetime(df.index.date).isin(holiday_dates)]
    return df

# ---------- TẢI DỮ LIỆU (NÂNG CẤP V18.8.1T) ----------
class DataLoader:
    def __init__(self, primary_source: str = 'vnstock', backup_sources: list = None):
        self.primary_source = primary_source
        self.backup_sources = backup_sources or ['yahoo']
        self.data_quality_metrics = {}
        self.source_reliability = {
            'vnstock': 1.0,
            'yahoo': 0.8
        }
        # Khởi tạo bộ căn chỉnh timestamp
        self.timestamp_aligner = TimestampAligner(exchange_timezone='Asia/Bangkok')
        
    def _get_data_source_priorities(self):
        """Trả về danh sách các nguồn dữ liệu theo thứ tự ưu tiên."""
        sources = [self.primary_source] + [s for s in self.backup_sources if s != self.primary_source]
        return sources
        
    def detect_outliers(self, df: pd.DataFrame, method: str = 'zscore', threshold: float = 3.0) -> (pd.DataFrame, str):
        if 'close' not in df.columns:
            return df, "Không có cột 'close' để phát hiện outlier"
            
        if method == 'zscore':
            z_scores = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
            df['is_outlier'] = z_scores > threshold
            outliers = df[df['is_outlier']]
            
            # Ghi lại báo cáo chi tiết
            outlier_report = f"Phát hiện {len(outliers)} giá trị bất thường trong dữ liệu:\n"
            for idx, row in outliers.iterrows():
                outlier_report += f"- {idx.strftime('%Y-%m-%d')}: {row['close']:.2f}\n"
                
            return df, outlier_report if not outliers.empty else "Không có giá trị bất thường"
        
        elif method == 'iqr':
            # Phương pháp phát hiện ngoại lai dựa trên IQR
            Q1 = df['close'].quantile(0.25)
            Q3 = df['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df['is_outlier'] = (df['close'] < lower_bound) | (df['close'] > upper_bound)
            outliers = df[df['is_outlier']]
            
            outlier_report = f"Phát hiện {len(outliers)} giá trị bất thường (IQR) trong dữ liệu:\n"
            for idx, row in outliers.iterrows():
                outlier_report += f"- {idx.strftime('%Y-%m-%d')}: {row['close']:.2f}\n"
                
            return df, outlier_report if not outliers.empty else "Không có giá trị bất thường (IQR)"
        
        return df, "Phương pháp phát hiện ngoại lai không được hỗ trợ"

    def handle_missing_values(self, df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
        """Xử lý các giá trị còn thiếu trong dữ liệu chuỗi thời gian."""
        if df.empty:
            return df
            
        # Kiểm tra giá trị còn thiếu
        missing_count = df.isna().sum().sum()
        if missing_count == 0:
            return df
            
        # Thêm cờ đánh dấu dữ liệu đã được điền
        df['is_imputed'] = False
        
        # Xử lý từng cột
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns and df[col].isna().any():
                missing_indices = df[col].isna()
                
                if method == 'linear':
                    df.loc[missing_indices, col] = df[col].interpolate(method='linear')
                elif method == 'ffill':
                    df.loc[missing_indices, col] = df[col].ffill()
                elif method == 'bfill':
                    df.loc[missing_indices, col] = df[col].bfill()
                elif method == 'mean':
                    df.loc[missing_indices, col] = df[col].fillna(df[col].mean())
                    
                # Đánh dấu các dòng đã được điền
                df.loc[missing_indices, 'is_imputed'] = True
                
        # Ghi log kết quả xử lý
        if missing_count > 0:
            logger.info(f"Đã xử lý {missing_count} giá trị còn thiếu bằng phương pháp {method}")
            
        return df
    
    def standardize_dataframe(self, df: pd.DataFrame, required_columns: list = None) -> pd.DataFrame:
        """Chuẩn hóa DataFrame đảm bảo cấu trúc nhất quán."""
        required_columns = required_columns or ['open', 'high', 'low', 'close', 'volume']
        
        # Chuẩn hóa tên cột
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'time': 'date', 'Time': 'date', 'Date': 'date', 'Datetime': 'date'
        }
        
        df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})
        
        # Đảm bảo có đủ cột cần thiết
        for col in required_columns:
            if col not in df.columns:
                if col == 'volume':
                    df[col] = 0  # Giá trị mặc định cho volume
                else:
                    raise ValueError(f"Dữ liệu thiếu cột bắt buộc: {col}")
        
        # Chuyển đổi kiểu dữ liệu nếu cần
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = df[col].astype('float32')
                
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype('float32')
            
        # Chuẩn hóa index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
        # Sắp xếp theo thời gian
        df = df.sort_index()
        
        return df
    
    def validate_price_data(self, df: pd.DataFrame) -> (bool, str):
        """Kiểm tra tính hợp lệ của dữ liệu giá."""
        if df.empty:
            return False, "DataFrame rỗng"
            
        validation_errors = []
        
        # Kiểm tra giá high >= giá low
        if not (df['high'] >= df['low']).all():
            invalid_rows = df[df['high'] < df['low']]
            validation_errors.append(f"Phát hiện {len(invalid_rows)} dòng có giá high < giá low")
            
        # Kiểm tra giá close nằm trong khoảng high-low
        if not ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all():
            invalid_rows = df[~((df['close'] >= df['low']) & (df['close'] <= df['high']))]
            validation_errors.append(f"Phát hiện {len(invalid_rows)} dòng có giá close nằm ngoài khoảng high-low")
            
        # Kiểm tra volume không âm
        if 'volume' in df.columns and (df['volume'] < 0).any():
            invalid_rows = df[df['volume'] < 0]
            validation_errors.append(f"Phát hiện {len(invalid_rows)} dòng có volume âm")
            
        if validation_errors:
            return False, "\n".join(validation_errors)
            
        return True, "Dữ liệu giá hợp lệ"

    async def load_data(self, symbol: str, timeframe: str, num_candles: int) -> (pd.DataFrame, str):
        """Tải dữ liệu từ nguồn chính, nếu thất bại sẽ dùng nguồn dự phòng."""
        timeframe_map = {'1d': '1D', '1w': '1W', '1mo': '1M'}
        timeframe = timeframe_map.get(timeframe.lower(), timeframe).upper()
        
        expire = CACHE_EXPIRE_SHORT if timeframe == '1D' else CACHE_EXPIRE_MEDIUM if timeframe == '1W' else CACHE_EXPIRE_LONG
        
        # Kiểm tra cache
        cache_key = f"data_{self.primary_source}_{symbol}_{timeframe}_{num_candles}"
        cached_data = await redis_manager.get(cache_key)
        if cached_data is not None:
            return cached_data, "Dữ liệu từ cache, không kiểm tra outlier"

        # Thử tải dữ liệu lần lượt từ các nguồn theo thứ tự ưu tiên
        sources = self._get_data_source_priorities()
        last_error = None
        
        for source in sources:
            try:
                logger.info(f"Đang tải dữ liệu cho {symbol} từ nguồn {source}...")
                
                if source == 'vnstock':
                    df = await self._load_from_vnstock(symbol, timeframe, num_candles)
                elif source == 'yahoo':
                    df = await self._load_from_yahoo(symbol, timeframe, num_candles)
                else:
                    logger.warning(f"Nguồn dữ liệu không được hỗ trợ: {source}")
                    continue
                
                # Chuẩn hóa dữ liệu
                df = self.standardize_dataframe(df)
                
                # Căn chỉnh timestamp chính xác
                df = self.timestamp_aligner.fix_timestamp_issues(df)
                df = self.timestamp_aligner.standardize_timeframe(df, freq=timeframe)
                
                # Kiểm tra tính hợp lệ
                is_valid, validation_msg = self.validate_price_data(df)
                if not is_valid:
                    logger.warning(f"Dữ liệu từ {source} không hợp lệ: {validation_msg}")
                    continue
                
                # Xử lý giá trị thiếu 
                df = self.handle_missing_values(df)
                
                # Lọc ngày giao dịch 
                df = self.timestamp_aligner.filter_trading_days(df)
                
                # Phát hiện ngoại lai
                df, outlier_report = self.detect_outliers(df)
                
                # Lưu vào cache
                await redis_manager.set(cache_key, df, expire=expire)
                
                # Tối ưu bộ nhớ bằng cách chuyển đổi kiểu dữ liệu
                for col in df.select_dtypes(include=['float64']).columns:
                    df[col] = df[col].astype('float32')
                
                # Cập nhật độ tin cậy của nguồn
                self.source_reliability[source] = min(1.0, self.source_reliability.get(source, 0.5) + 0.1)
                
                return df, outlier_report
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Lỗi tải dữ liệu từ {source} cho {symbol}: {last_error}")
                # Giảm độ tin cậy của nguồn này 
                self.source_reliability[source] = max(0.1, self.source_reliability.get(source, 0.5) - 0.1)
        
        # Nếu tất cả các nguồn đều thất bại
        raise ValueError(f"Không thể tải dữ liệu cho {symbol} từ bất kỳ nguồn nào: {last_error}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def _load_from_vnstock(self, symbol: str, timeframe: str, num_candles: int) -> pd.DataFrame:
        """Tải dữ liệu từ VNStock."""
        def fetch_vnstock():
            stock = Vnstock().stock(symbol=symbol, source='TCBS')
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=(num_candles + 1) * 3)).strftime('%Y-%m-%d')
            df = stock.quote.history(start=start_date, end=end_date, interval=timeframe)
            
            if df is None or df.empty or len(df) < 20:
                raise ValueError(f"Không đủ dữ liệu cho {'chỉ số' if is_index(symbol) else 'mã'} {symbol}")
                
            df = df.rename(columns={'time': 'date', 'open': 'open', 'high': 'high',
                                     'low': 'low', 'close': 'close', 'volume': 'volume'})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            
            # Thêm múi giờ cho index nếu chưa có
            if df.index.tz is None:
                df.index = df.index.tz_localize('Asia/Bangkok')
                
            df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
            
            return df.tail(num_candles + 1)
            
        return await run_in_thread(fetch_vnstock)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
    async def _load_from_yahoo(self, symbol: str, timeframe: str, num_candles: int) -> pd.DataFrame:
        """Tải dữ liệu từ Yahoo Finance."""
        period_map = {'1D': 'd', '1W': 'wk', '1M': 'mo'}
        period = period_map.get(timeframe, 'd')
        
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
                    
                    # Chuẩn hóa tên cột
                    df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low',
                                            'Close': 'close', 'Volume': 'volume'})
                    
                    # Thêm múi giờ
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('Asia/Bangkok')
                    
                    return df.tail(num_candles)
        except asyncio.TimeoutError:
            logger.error("Timeout khi tải dữ liệu từ Yahoo Finance.")
            raise
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu Yahoo: {str(e)}")
            raise

    async def get_incremental_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Chỉ tải dữ liệu mới từ lần cập nhật cuối."""
        cache_key = f"last_update_{symbol}_{timeframe}"
        last_update = await redis_manager.get(cache_key)
        
        if not last_update:
            # Tải toàn bộ dữ liệu nếu chưa có
            df, _ = await self.load_data(symbol, timeframe, DEFAULT_CANDLES)
            await redis_manager.set(cache_key, datetime.now(), expire=CACHE_EXPIRE_LONG)
            return df
            
        # Tính toán khoảng thời gian cần tải
        from_date = last_update + timedelta(days=1)
        to_date = datetime.now()
        
        # Không cần tải nếu thời gian chưa đủ 1 ngày
        if (to_date - from_date).days < 1:
            df_old, _ = await self.load_data(symbol, timeframe, DEFAULT_CANDLES)
            return df_old
            
        try:
            # Tải dữ liệu mới
            if self.primary_source == 'vnstock':
                def fetch_incremental():
                    stock = Vnstock().stock(symbol=symbol, source='TCBS')
                    df = stock.quote.history(start=from_date.strftime('%Y-%m-%d'), 
                                          end=to_date.strftime('%Y-%m-%d'), 
                                          interval=timeframe)
                    if df is not None and not df.empty:
                        df = df.rename(columns={'time': 'date', 'open': 'open', 'high': 'high',
                                              'low': 'low', 'close': 'close', 'volume': 'volume'})
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        if df.index.tz is None:
                            df.index = df.index.tz_localize('Asia/Bangkok')
                        return df[['open', 'high', 'low', 'close', 'volume']]
                    return pd.DataFrame()
                
                df_new = await run_in_thread(fetch_incremental)
                
                if df_new is None or df_new.empty:
                    logger.info(f"Không có dữ liệu mới cho {symbol} từ {from_date} đến {to_date}")
                    df_old, _ = await self.load_data(symbol, timeframe, DEFAULT_CANDLES)
                    return df_old
                    
                # Merge với dữ liệu cũ
                df_old, _ = await self.load_data(symbol, timeframe, DEFAULT_CANDLES)
                df = pd.concat([df_old, df_new]).drop_duplicates()
                
                # Chuẩn hóa, xử lý và lưu
                df = self.standardize_dataframe(df)
                df = self.handle_missing_values(df)
                df = filter_trading_days(df)
                
                # Cập nhật cache
                cache_key_data = f"data_{self.primary_source}_{symbol}_{timeframe}_{DEFAULT_CANDLES}"
                await redis_manager.set(cache_key_data, df, expire=CACHE_EXPIRE_MEDIUM)
                await redis_manager.set(cache_key, datetime.now(), expire=CACHE_EXPIRE_LONG)
                
                return df
                
            else:
                # Fallback to full load for other sources
                return await self.load_data(symbol, timeframe, DEFAULT_CANDLES)[0]
                
        except Exception as e:
            logger.error(f"Lỗi cập nhật gia tăng cho {symbol}: {str(e)}")
            # Fallback to cached data
            return await self.load_data(symbol, timeframe, DEFAULT_CANDLES)[0]

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

    async def merge_data_sources(self, symbol: str, timeframe: str, num_candles: int) -> pd.DataFrame:
        """
        Tải dữ liệu từ nhiều nguồn và hợp nhất lại với căn chỉnh timestamp.
        
        Args:
            symbol: Mã chứng khoán cần tải
            timeframe: Khung thời gian ('1D', '1W', '1M')
            num_candles: Số nến cần tải
            
        Returns:
            DataFrame hợp nhất từ nhiều nguồn
        """
        dataframes = []
        sources = self._get_data_source_priorities()
        
        # Tải dữ liệu từ các nguồn
        for source in sources:
            try:
                if source == 'vnstock':
                    df = await self._load_from_vnstock(symbol, timeframe, num_candles)
                elif source == 'yahoo':
                    df = await self._load_from_yahoo(symbol, timeframe, num_candles)
                else:
                    continue
                    
                # Chuẩn hóa dữ liệu
                df = self.standardize_dataframe(df)
                
                # Tạo cột để đánh dấu nguồn dữ liệu
                df['data_source'] = source
                
                dataframes.append(df)
            except Exception as e:
                logger.warning(f"Không thể tải dữ liệu từ nguồn {source}: {str(e)}")
        
        if not dataframes:
            raise ValueError(f"Không thể tải dữ liệu cho {symbol} từ bất kỳ nguồn nào")
            
        # Sử dụng TimestampAligner để hợp nhất các DataFrame
        merged_df = self.timestamp_aligner.merge_dataframes_with_alignment(dataframes, freq=timeframe)
        
        # Xử lý trùng lặp và lọc dữ liệu
        merged_df = self.handle_missing_values(merged_df)
        merged_df = self.timestamp_aligner.filter_trading_days(merged_df)
        
        return merged_df
        
    async def get_precise_timestamp_data(self, symbol: str, timeframe: str, num_candles: int) -> pd.DataFrame:
        """
        Tải dữ liệu với timestamp được căn chỉnh chính xác.
        
        Args:
            symbol: Mã chứng khoán cần tải
            timeframe: Khung thời gian ('1D', '1W', '1M')
            num_candles: Số nến cần tải
            
        Returns:
            DataFrame với timestamp đã được căn chỉnh chính xác
        """
        cache_key = f"precise_ts_{symbol}_{timeframe}_{num_candles}"
        cached_data = await redis_manager.get(cache_key)
        
        if cached_data is not None:
            return cached_data
            
        try:
            # Tải dữ liệu từ nguồn chính
            df, _ = await self.load_data(symbol, timeframe, num_candles)
            
            # Căn chỉnh timestamp
            fixed_df = self.timestamp_aligner.fix_timestamp_issues(df)
            aligned_df = self.timestamp_aligner.standardize_timeframe(fixed_df, freq=timeframe)
            
            # Thêm các đặc trưng timestamp
            enhanced_df = self.timestamp_aligner.extract_timestamp_features(aligned_df)
            
            # Lưu vào cache
            await redis_manager.set(cache_key, enhanced_df, expire=CACHE_EXPIRE_MEDIUM)
            
            return enhanced_df
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu timestamp chính xác cho {symbol}: {str(e)}")
            raise

# ---------- QUẢN LÝ CHẤT LƯỢNG DỮ LIỆU ----------
class DataQualityControl:
    def __init__(self, db_manager=None):
        self.quality_metrics = {}
        self.db_manager = db_manager
        self.quality_threshold = 0.7
        
    def evaluate_data_quality(self, df: pd.DataFrame, symbol: str) -> dict:
        """Đánh giá chất lượng dữ liệu theo nhiều tiêu chí."""
        if df is None or df.empty:
            return {
                "symbol": symbol,
                "completeness": 0.0,
                "consistency": 0.0,
                "timeliness": 0.0,
                "validity": 0.0,
                "accuracy": 0.0,
                "overall_score": 0.0,
                "recommendation": "Không có dữ liệu để đánh giá",
                "timestamp": datetime.now().isoformat()
            }
            
        metrics = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat()
        }
        
        # 1. Tính toán điểm đầy đủ (completeness)
        missing_values = df.isnull().mean().mean()
        metrics["completeness"] = float(1.0 - missing_values)
        
        # 2. Tính toán điểm nhất quán (consistency)
        # Kiểm tra các ràng buộc giữa các cột
        if 'high' in df.columns and 'low' in df.columns:
            valid_hl = (df['high'] >= df['low']).mean()
            metrics["consistency"] = float(valid_hl)
        else:
            metrics["consistency"] = 0.5  # Nếu không có dữ liệu để kiểm tra
            
        # 3. Tính toán điểm kịp thời (timeliness)
        # Kiểm tra dữ liệu có cập nhật mới không
        if isinstance(df.index, pd.DatetimeIndex):
            latest_date = df.index.max()
            days_since_update = (datetime.now() - latest_date.to_pydatetime()).days
            metrics["timeliness"] = float(max(0, 1.0 - days_since_update/30.0))  # Giảm 1/30 mỗi ngày không cập nhật
        else:
            metrics["timeliness"] = 0.0
            
        # 4. Tính toán điểm hợp lệ (validity)
        # Kiểm tra các giá trị có nằm trong khoảng hợp lệ không
        if 'close' in df.columns and 'low' in df.columns and 'high' in df.columns:
            valid_close = ((df['close'] >= df['low']) & (df['close'] <= df['high'])).mean()
            metrics["validity"] = float(valid_close)
        else:
            metrics["validity"] = 0.5
            
        # 5. Ước lượng độ chính xác (accuracy)
        # Phát hiện outliers bằng Z-score
        if 'close' in df.columns:
            z_scores = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
            outlier_ratio = (z_scores > 3).mean()
            metrics["accuracy"] = float(1.0 - outlier_ratio)
        else:
            metrics["accuracy"] = 0.5
            
        # Tính điểm tổng hợp
        weights = {
            "completeness": 0.25,
            "consistency": 0.2,
            "timeliness": 0.2,
            "validity": 0.2,
            "accuracy": 0.15
        }
        
        weighted_scores = [metrics[key] * weights[key] for key in weights.keys()]
        metrics["overall_score"] = float(sum(weighted_scores))
        
        # Xác định khuyến nghị dựa trên chất lượng
        if metrics["overall_score"] < 0.5:
            metrics["recommendation"] = "Dữ liệu chất lượng thấp, nên thu thập lại"
        elif metrics["overall_score"] < 0.7:
            metrics["recommendation"] = "Dữ liệu cần được làm sạch thêm"
        elif metrics["overall_score"] < 0.9:
            metrics["recommendation"] = "Dữ liệu có chất lượng khá tốt"
        else:
            metrics["recommendation"] = "Dữ liệu có chất lượng rất tốt"
            
        # Lưu kết quả đánh giá
        self.quality_metrics[symbol] = metrics
        
        return metrics
        
    async def save_quality_metrics(self, metrics: dict):
        """Lưu trữ các chỉ số chất lượng vào DB nếu có."""
        if self.db_manager:
            # Implementation would depend on your database schema
            pass
            
    def is_data_usable(self, metrics: dict) -> bool:
        """Kiểm tra dữ liệu có đủ chất lượng để sử dụng không."""
        return metrics["overall_score"] >= self.quality_threshold
        
    async def generate_quality_report(self, symbol: str, timeframe: str) -> str:
        """Tạo báo cáo về chất lượng dữ liệu."""
        if symbol not in self.quality_metrics:
            return f"Chưa có đánh giá chất lượng dữ liệu cho {symbol}"
            
        metrics = self.quality_metrics[symbol]
        
        report = f"📊 BÁO CÁO CHẤT LƯỢNG DỮ LIỆU: {symbol} ({timeframe})\n\n"
        report += f"⏱️ Thời điểm đánh giá: {metrics['timestamp']}\n"
        report += f"✅ Điểm tổng hợp: {metrics['overall_score']:.2f}/1.0\n\n"
        report += "CHI TIẾT:\n"
        report += f"- Đầy đủ: {metrics['completeness']:.2f}/1.0\n"
        report += f"- Nhất quán: {metrics['consistency']:.2f}/1.0\n"
        report += f"- Kịp thời: {metrics['timeliness']:.2f}/1.0\n"
        report += f"- Hợp lệ: {metrics['validity']:.2f}/1.0\n"
        report += f"- Chính xác: {metrics['accuracy']:.2f}/1.0\n\n"
        report += f"📌 Khuyến nghị: {metrics['recommendation']}"
        
        return report

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
            "Bạn là chuyên gia phân tích kỹ thuật chứng khoán."
            " Dựa trên dữ liệu dưới đây, hãy nhận diện các mẫu hình nến như Doji, Hammer, Shooting Star, Engulfing,"
            " sóng Elliott, mô hình Wyckoff, và các vùng hỗ trợ/kháng cự."
            "\n\nChỉ trả về kết quả ở dạng JSON như sau, không thêm giải thích nào khác:\n"
            "{\n"
            "  \"support_levels\": [giá1, giá2, ...],\n"
            "  \"resistance_levels\": [giá1, giá2, ...],\n"
            "  \"patterns\": [\n"
            "    {\"name\": \"tên mẫu hình\", \"description\": \"giải thích ngắn\"},\n"
            "    ...\n"
            "  ]\n"
            "}\n\n"
            f"Dữ liệu:\n{json.dumps(technical_data, ensure_ascii=False, indent=2)}"
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

        async with aiohttp.ClientSession() as session:
            async with session.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as resp:
                text = await resp.text()
                try:
                    result = json.loads(text)
                    content = result['choices'][0]['message']['content']
                    return json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Phản hồi không hợp lệ từ OpenRouter: {text}")
                    return {}
                except KeyError:
                    logger.error(f"Phản hồi thiếu trường cần thiết: {text}")
                    return {}

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

    await update.message.reply_text(
        "🚀 **V18.8.1T - Nâng cấp tải và xử lý dữ liệu!**\n"
        "📊 **Lệnh**:\n"
        "- /analyze [Mã] [Số nến] - Phân tích đa khung.\n"
        "- /refresh [Mã] - Làm mới dữ liệu cho mã.\n"
        "- /getid - Lấy ID.\n"
        "- /approve [user_id] - Duyệt người dùng (admin).\n"
        "- /datastats - Xem thống kê dữ liệu (admin).\n"
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
        if num_candles < 20:
            raise ValueError("Số nến phải lớn hơn hoặc bằng 20 để tính toán chỉ báo!")
        if num_candles > 500:
            raise ValueError("Tối đa 500 nến!")
            
        # Khởi tạo các lớp xử lý dữ liệu nâng cao (nếu chưa tồn tại)
        data_loader = DataLoader(primary_source='vnstock', backup_sources=['yahoo'])
        data_quality_control = DataQualityControl()
        data_processor = AdvancedDataProcessor()
        tech_analyzer = TechnicalAnalyzer()
        ai_analyzer = AIAnalyzer()
        
        # Thông báo cho người dùng
        processing_msg = await update.message.reply_text("⏳ Đang xử lý dữ liệu và phân tích... Vui lòng đợi.")
        
        # Tải dữ liệu đa khung thời gian
        timeframes = ['1D', '1W', '1M']
        dfs = {}
        outlier_reports = {}
        quality_reports = {}
        
        for tf in timeframes:
            # Tải dữ liệu
            df, outlier_report = await data_loader.load_data(symbol, tf, num_candles)
            
            # Đánh giá chất lượng dữ liệu
            quality_metrics = data_quality_control.evaluate_data_quality(df, symbol)
            quality_report = f"Chất lượng: {quality_metrics['overall_score']:.2f}/1.0"
            
            # Xử lý dữ liệu nâng cao
            if data_quality_control.is_data_usable(quality_metrics):
                df = data_processor.preprocess_data(df)
                
            # Tính toán các chỉ báo kỹ thuật
            df = tech_analyzer.calculate_indicators(df)
            
            # Lưu kết quả
            dfs[tf] = df
            outlier_reports[tf] = outlier_report
            quality_reports[tf] = quality_report
            
        # Lấy dữ liệu cơ bản
        fundamental_data = await data_loader.get_fundamental_data(symbol)
        
        # Tạo báo cáo
        report = await ai_analyzer.generate_report(dfs, symbol, fundamental_data, outlier_reports)
        
        # Lưu vào cache
        await redis_manager.set(f"report_{symbol}_{num_candles}", report, expire=CACHE_EXPIRE_SHORT)
        
        # Thêm thông tin chất lượng dữ liệu vào báo cáo
        quality_info = "\n".join([f"🔍 {tf}: {report}" for tf, report in quality_reports.items()])
        formatted_report = f"<b>📈 Báo cáo phân tích cho {symbol}</b>\n<i>{quality_info}</i>\n\n<pre>{html.escape(report)}</pre>"
        
        # Cập nhật hoặc gửi báo cáo mới
        try:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=processing_msg.message_id,
                text=formatted_report,
                parse_mode='HTML'
            )
        except Exception:
            # Nếu không thể chỉnh sửa tin nhắn (có thể quá dài), gửi tin nhắn mới
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

async def data_stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh để admin xem thống kê dữ liệu hệ thống."""
    user_id = update.message.from_user.id
    if str(user_id) != ADMIN_ID:
        await update.message.reply_text("❌ Chỉ admin dùng được lệnh này!")
        return
        
    await update.message.reply_text("⏳ Đang tổng hợp thống kê dữ liệu...")
    
    try:
        # Khởi tạo các lớp cần thiết
        data_loader = DataLoader()
        data_quality = DataQualityControl()
        data_processor = AdvancedDataProcessor()
        
        data_manager = DataAutomationManager(data_loader, data_quality, data_processor)
        stats = await data_manager.get_data_statistics()
        
        # Tạo báo cáo
        report = "📊 <b>THỐNG KÊ DỮ LIỆU HỆ THỐNG</b>\n\n"
        report += f"🔢 Tổng số mã: {stats['total_symbols']}\n"
        report += f"📈 Tổng số điểm dữ liệu: {stats['total_datapoints']:,}\n"
        report += f"🗄️ Tổng số khóa cache: {stats['total_cache_keys']}\n\n"
        
        if stats['problem_symbols']:
            report += "⚠️ <b>MÃ CÓ VẤN ĐỀ CHẤT LƯỢNG:</b>\n"
            for symbol_info in stats['problem_symbols'][:10]:  # Chỉ hiển thị 10 mã đầu tiên
                report += f"- {symbol_info['symbol']}: {symbol_info['score']:.2f}/1.0\n"
                
            if len(stats['problem_symbols']) > 10:
                report += f"... và {len(stats['problem_symbols']) - 10} mã khác\n"
        else:
            report += "✅ Tất cả mã đều có chất lượng dữ liệu tốt\n"
            
        # Thông tin bộ nhớ
        if stats['memory_usage']:
            report += "\n💾 <b>SỬ DỤNG BỘ NHỚ REDIS:</b>\n"
            report += f"- Đã dùng: {stats['memory_usage'].get('used_memory', 'N/A')}\n"
            report += f"- Đỉnh: {stats['memory_usage'].get('used_memory_peak', 'N/A')}\n"
            report += f"- Tổng bộ nhớ hệ thống: {stats['memory_usage'].get('total_system_memory', 'N/A')}\n"
            
        await update.message.reply_text(report, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Lỗi lấy thống kê dữ liệu: {str(e)}")
        await update.message.reply_text(f"❌ Lỗi khi lấy thống kê dữ liệu: {str(e)}")

async def refresh_data_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh để làm mới dữ liệu cho một mã cụ thể."""
    user_id = update.message.from_user.id
    if not await is_user_approved(user_id):
        await notify_admin_new_user(update, context)
        return
        
    args = context.args
    if not args:
        await update.message.reply_text("❌ Vui lòng nhập mã chứng khoán cần làm mới dữ liệu.")
        return
        
    symbol = args[0].upper()
    await update.message.reply_text(f"⏳ Đang làm mới dữ liệu cho {symbol}...")
    
    try:
        # Khởi tạo các lớp cần thiết
        data_loader = DataLoader()
        timeframes = ['1D', '1W', '1M']
        
        # Xóa cache hiện tại
        for tf in timeframes:
            cache_key = f"data_vnstock_{symbol}_{tf}_{DEFAULT_CANDLES}"
            await redis_manager.redis_client.delete(cache_key)
            cache_key = f"data_yahoo_{symbol}_{tf}_{DEFAULT_CANDLES}"
            await redis_manager.redis_client.delete(cache_key)
            
        # Tải dữ liệu mới
        results = []
        for tf in timeframes:
            try:
                df, report = await data_loader.load_data(symbol, tf, DEFAULT_CANDLES)
                results.append(f"✅ {tf}: {len(df)} nến")
            except Exception as e:
                results.append(f"❌ {tf}: {str(e)}")
                
        # Báo cáo kết quả
        report = f"🔄 <b>LÀM MỚI DỮ LIỆU: {symbol}</b>\n\n" + "\n".join(results)
        await update.message.reply_text(report, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Lỗi làm mới dữ liệu cho {symbol}: {str(e)}")
        await update.message.reply_text(f"❌ Lỗi làm mới dữ liệu: {str(e)}")

async def check_timestamp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh để kiểm tra và sửa timestamp cho một mã cụ thể."""
    user_id = update.message.from_user.id
    if not await is_user_approved(user_id):
        await notify_admin_new_user(update, context)
        return
        
    args = context.args
    if not args or len(args) < 1:
        await update.message.reply_text("❌ Vui lòng nhập: /checkts [Mã] [Khung thời gian: 1D, 1W, 1M (mặc định 1D)]")
        return
        
    symbol = args[0].upper()
    timeframe = args[1].upper() if len(args) > 1 else '1D'
    
    if timeframe not in ['1D', '1W', '1M']:
        await update.message.reply_text("❌ Khung thời gian không hợp lệ. Sử dụng: 1D, 1W, hoặc 1M")
        return
        
    await update.message.reply_text(f"⏳ Đang kiểm tra timestamp cho {symbol} ({timeframe})...")
    
    try:
        # Khởi tạo loader và timestamp aligner
        data_loader = DataLoader(primary_source='vnstock', backup_sources=['yahoo'])
        
        # Tải dữ liệu
        regular_df, _ = await data_loader.load_data(symbol, timeframe, 30)
        precise_df = await data_loader.get_precise_timestamp_data(symbol, timeframe, 30)
        
        # Tạo báo cáo
        report = f"🕒 <b>KIỂM TRA TIMESTAMP CHO {symbol} ({timeframe})</b>\n\n"
        
        # So sánh số lượng nến
        regular_count = len(regular_df) if regular_df is not None else 0
        precise_count = len(precise_df) if precise_df is not None else 0
        
        report += f"📊 <b>SỐ LƯỢNG NẾN:</b>\n"
        report += f"- Dữ liệu thông thường: {regular_count} nến\n"
        report += f"- Dữ liệu đã căn chỉnh: {precise_count} nến\n\n"
        
        # Thông tin timestamp
        if precise_df is not None and not precise_df.empty:
            first_date = precise_df.index[0].strftime('%Y-%m-%d %H:%M')
            last_date = precise_df.index[-1].strftime('%Y-%m-%d %H:%M')
            
            report += f"🗓️ <b>PHẠM VI THỜI GIAN:</b>\n"
            report += f"- Từ: {first_date}\n"
            report += f"- Đến: {last_date}\n\n"
            
            # Kiểm tra timezone
            timezone = str(precise_df.index[0].tz)
            report += f"🌐 <b>TIMEZONE:</b> {timezone}\n\n"
            
            # Kiểm tra thời gian trong ngày
            hours = [idx.hour for idx in precise_df.index]
            minutes = [idx.minute for idx in precise_df.index]
            
            if len(set(hours)) == 1 and len(set(minutes)) == 1:
                report += f"✅ <b>CHUẨN HÓA THỜI GIAN:</b> Tất cả timestamp đều vào {hours[0]}:{minutes[0]}\n\n"
            else:
                report += f"⚠️ <b>CHUẨN HÓA THỜI GIAN:</b> Timestamp không đồng nhất!\n"
                report += f"- Giờ khác nhau: {set(hours)}\n"
                report += f"- Phút khác nhau: {set(minutes)}\n\n"
                
            # Kiểm tra ngày giao dịch
            weekdays = [idx.weekday() for idx in precise_df.index]
            weekday_counts = {
                0: "Thứ 2", 1: "Thứ 3", 2: "Thứ 4", 
                3: "Thứ 5", 4: "Thứ 6", 5: "Thứ 7", 6: "Chủ nhật"
            }
            
            if any(wd >= 5 for wd in weekdays):
                report += "⚠️ <b>NGÀY GIAO DỊCH:</b> Phát hiện ngày cuối tuần trong dữ liệu!\n"
                for wd, count in sorted({wd: weekdays.count(wd) for wd in set(weekdays)}.items()):
                    report += f"- {weekday_counts[wd]}: {count} nến\n"
            else:
                report += "✅ <b>NGÀY GIAO DỊCH:</b> Tất cả đều là ngày trong tuần (Thứ 2-6)\n"
                for wd, count in sorted({wd: weekdays.count(wd) for wd in set(weekdays)}.items()):
                    report += f"- {weekday_counts[wd]}: {count} nến\n"
        else:
            report += "❌ Không có dữ liệu sau khi căn chỉnh timestamp"
            
        await update.message.reply_text(report, parse_mode='HTML')
        
    except Exception as e:
        logger.error(f"Lỗi kiểm tra timestamp cho {symbol}: {str(e)}")
        await update.message.reply_text(f"❌ Lỗi: {str(e)}")

# Thêm lệnh mới vào main
def add_timestamp_commands(app):
    app.add_handler(CommandHandler("checkts", check_timestamp_command))
    logger.info("Đã đăng ký lệnh kiểm tra timestamp")

# ---------- MAIN & DEPLOY ----------
async def main():
    await init_db()

    # Khởi tạo các lớp xử lý dữ liệu nâng cao
    data_loader = DataLoader(primary_source='vnstock', backup_sources=['yahoo'])
    data_quality_control = DataQualityControl()
    data_processor = AdvancedDataProcessor()
    
    # Thiết lập scheduler
    scheduler = AsyncIOScheduler()
    
    # Tự động hóa quản lý dữ liệu
    data_automation = DataAutomationManager(
        data_loader=data_loader,
        quality_control=data_quality_control,
        data_processor=data_processor,
        scheduler=scheduler
    )
    
    # Thêm tác vụ auto training
    scheduler.add_job(auto_train_models, 'cron', hour=2, minute=0, id='auto_train_models', replace_existing=True)
    
    # Thiết lập tự động hóa dữ liệu
    data_automation.setup_data_automation()
    
    # Nếu đã có dữ liệu lịch sử, thiết lập các mã ưu tiên
    training_symbols = await get_training_symbols()
    if training_symbols:
        data_automation.set_priority_symbols(training_symbols)
        logger.info(f"Đã thiết lập {len(training_symbols)} mã ưu tiên từ lịch sử")
    
    # Khởi động scheduler
    scheduler.start()
    logger.info("Các tác vụ tự động đã được khởi động")

    # Khởi tạo ứng dụng Telegram
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("getid", get_id))
    app.add_handler(CommandHandler("approve", approve_user))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, notify_admin_new_user))
    logger.info("🤖 Bot khởi động! Phiên bản V18.8.1T (Nâng cấp tải dữ liệu)")

    BASE_URL = os.getenv("RENDER_EXTERNAL_URL", f"https://{os.getenv('RENDER_SERVICE_NAME')}.onrender.com")
    WEBHOOK_URL = f"{BASE_URL}/{TELEGRAM_TOKEN}"
    await app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        webhook_url=WEBHOOK_URL,
        url_path=TELEGRAM_TOKEN
    )

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