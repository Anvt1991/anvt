#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot Chứng Khoán Toàn Diện Phiên Bản V18.9 (Nâng cấp):
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
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, ForeignKey, select, LargeBinary

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

# ---------- CHUẨN HÓA DỮ LIỆU ----------
class DataNormalizer:
    """
    Lớp chuẩn hóa dữ liệu cho chứng khoán:
    - Chuẩn hóa tên cột
    - Xử lý giá trị ngoại lai
    - Điền giá trị thiếu
    - Xác thực dữ liệu
    """
    
    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa DataFrame để đảm bảo định dạng nhất quán"""
        if df is None or df.empty:
            raise ValueError("DataFrame rỗng, không thể chuẩn hóa")
        
        # Chuẩn hóa tên cột
        column_mapping = {
            'time': 'date', 'Time': 'date', 'Date': 'date',
            'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        df = df.rename(columns={col: column_mapping.get(col, col) for col in df.columns})
        
        # Đảm bảo có đủ các cột cần thiết
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Thiếu các cột: {missing_columns}")
        
        # Chuyển đổi định dạng index thành datetime nếu chưa phải
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df.index):
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        
        return df
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> (bool, str):
        """Xác thực tính hợp lệ của dữ liệu chứng khoán"""
        if df is None or df.empty:
            return False, "DataFrame rỗng"
        
        errors = []
        
        # Kiểm tra dữ liệu cần thiết
        if 'close' not in df.columns:
            errors.append("Thiếu cột 'close'")
        
        # Kiểm tra high >= low
        if 'high' in df.columns and 'low' in df.columns:
            invalid_hl = (~(df['high'] >= df['low'])).sum()
            if invalid_hl > 0:
                errors.append(f"Có {invalid_hl} hàng với giá high < low")
        
        # Kiểm tra low <= close <= high
        if all(col in df.columns for col in ['low', 'close', 'high']):
            invalid_range = (~((df['close'] >= df['low']) & (df['close'] <= df['high']))).sum()
            if invalid_range > 0:
                errors.append(f"Có {invalid_range} hàng với giá close nằm ngoài khoảng [low, high]")
        
        # Kiểm tra volume âm
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                errors.append(f"Có {negative_volume} hàng với volume âm")
        
        return len(errors) == 0, "\n".join(errors)
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns=['open', 'high', 'low', 'close'], 
                         method='zscore', threshold=3) -> (pd.DataFrame, str):
        """Phát hiện giá trị ngoại lai trong dữ liệu"""
        if df is None or df.empty:
            return df, "Không có dữ liệu để phát hiện outlier"
        
        report_lines = []
        df = df.copy()
        df['is_outlier'] = False
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > threshold
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            else:
                raise ValueError(f"Phương pháp phát hiện outlier '{method}' không được hỗ trợ")
            
            # Ghi nhận outlier
            df.loc[outliers, 'is_outlier'] = True
            outlier_rows = df[outliers]
            
            if not outlier_rows.empty:
                report_lines.append(f"Phát hiện {len(outlier_rows)} giá trị bất thường trong cột {col}:")
                for idx, row in outlier_rows.iterrows():
                    report_lines.append(f"- {idx.strftime('%Y-%m-%d')}: {row[col]:.2f}")
        
        return df, "\n".join(report_lines) if report_lines else "Không có giá trị bất thường"
    
    @staticmethod
    def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Điền các giá trị bị thiếu trong dữ liệu"""
        if df is None or df.empty:
            return df
        
        # Kiểm tra giá trị NaN
        if not df.isna().any().any():
            return df
        
        df_filled = df.copy()
        
        # Điền cột close
        if 'close' in df.columns and df['close'].isna().any():
            df_filled['close'] = df_filled['close'].fillna(method='ffill')
        
        # Điền các cột giá còn lại
        for col in ['open', 'high', 'low']:
            if col in df.columns and df[col].isna().any():
                # Sử dụng giá close nếu có
                if 'close' in df.columns:
                    df_filled[col] = df_filled[col].fillna(df_filled['close'])
                else:
                    df_filled[col] = df_filled[col].fillna(method='ffill')
        
        # Điền volume
        if 'volume' in df.columns and df['volume'].isna().any():
            df_filled['volume'] = df_filled['volume'].fillna(0)
        
        return df_filled
    
    @staticmethod
    def standardize_for_db(data: dict) -> dict:
        """Chuẩn hóa dữ liệu cho lưu trữ database"""
        standardized_data = {}
        for key, value in data.items():
            if isinstance(value, np.float64):
                standardized_data[key] = float(value)
            elif isinstance(value, np.int64):
                standardized_data[key] = int(value)
            elif isinstance(value, pd.Timestamp):
                standardized_data[key] = value.to_pydatetime()
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                standardized_data[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                standardized_data[key] = value.tolist()
            else:
                standardized_data[key] = value
        return standardized_data

# ---------- CLASS VALIDATOR DỮ LIỆU ----------
class DataValidator:
    """
    Lớp validator nâng cao cho dữ liệu chứng khoán:
    - Chuẩn hóa cột dữ liệu
    - Xử lý giá trị thiếu
    - Phát hiện và xử lý outlier theo IQR/Z-Score
    - Validate ticker, timeframe, số lượng nến
    - Căn chỉnh timestamp theo khung thời gian
    """
    
    VALID_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1D', '1W', '1M']
    MIN_CANDLES = 20
    VALID_TICKERS_REGEX = r'^[A-Z0-9]{3,10}$'
    
    @staticmethod
    def validate_ticker(ticker: str) -> (bool, str):
        """Kiểm tra tính hợp lệ của mã chứng khoán"""
        if not ticker:
            return False, "Mã chứng khoán không được để trống"
        
        ticker = ticker.upper()
        if not re.match(DataValidator.VALID_TICKERS_REGEX, ticker):
            return False, "Mã chứng khoán không hợp lệ"
        
        # Kiểm tra mã VN 
        if len(ticker) == 3 and not is_index(ticker):
            return True, f"Mã hợp lệ: {ticker}"
            
        if is_index(ticker):
            return True, f"Chỉ số hợp lệ: {ticker}"
            
        return True, f"Mã hợp lệ: {ticker}"
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> (bool, str):
        """Kiểm tra tính hợp lệ của khung thời gian"""
        if not timeframe:
            return False, "Khung thời gian không được để trống"
        
        timeframe = timeframe.upper()
        if timeframe not in DataValidator.VALID_TIMEFRAMES:
            return False, f"Khung thời gian không hợp lệ. Hỗ trợ: {', '.join(DataValidator.VALID_TIMEFRAMES)}"
        
        return True, f"Khung thời gian hợp lệ: {timeframe}"
    
    @staticmethod
    def validate_candles_count(count: int) -> (bool, str):
        """Kiểm tra tính hợp lệ của số lượng nến"""
        if count < DataValidator.MIN_CANDLES:
            return False, f"Số lượng nến phải ít nhất {DataValidator.MIN_CANDLES}"
        
        if count > 5000:
            return False, "Số lượng nến không nên vượt quá 5000 để tránh quá tải"
        
        return True, f"Số lượng nến hợp lệ: {count}"
    
    @staticmethod
    def detect_and_handle_outliers(df: pd.DataFrame, method='iqr', replace_method='mean') -> (pd.DataFrame, dict):
        """
        Phát hiện và xử lý outlier
        
        Parameters:
        - df: DataFrame cần xử lý
        - method: 'iqr' hoặc 'zscore'
        - replace_method: 'mean', 'median', 'ffill', hoặc 'none' (chỉ báo cáo, không thay thế)
        
        Returns:
        - DataFrame đã xử lý
        - Dict báo cáo outlier
        """
        if df is None or df.empty:
            return df, {"error": "DataFrame rỗng"}
        
        columns = ['open', 'high', 'low', 'close']
        columns = [col for col in columns if col in df.columns]
        
        df_processed = df.copy()
        outlier_report = {"total": 0, "columns": {}}
        
        for column in columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound))
            elif method == 'zscore':
                z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
                outliers = z_scores > 3
            else:
                continue
                
            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_dates = df.index[outliers].tolist()
                outlier_values = df.loc[outliers, column].tolist()
                
                outlier_report["total"] += outlier_count
                outlier_report["columns"][column] = {
                    "count": int(outlier_count),
                    "samples": [{"date": d.strftime('%Y-%m-%d %H:%M'), "value": float(v)} 
                               for d, v in zip(outlier_dates[:5], outlier_values[:5])]
                }
                
                # Xử lý outlier nếu cần
                if replace_method != 'none':
                    if replace_method == 'mean':
                        replacement = df[column].mean()
                    elif replace_method == 'median':
                        replacement = df[column].median()
                    elif replace_method == 'ffill':
                        df_processed[column] = df_processed[column].mask(outliers).ffill().bfill()
                        continue
                    
                    df_processed.loc[outliers, column] = replacement
        
        return df_processed, outlier_report
    
    @staticmethod
    def align_timestamps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Căn chỉnh timestamps chuẩn theo khung thời gian"""
        if df is None or df.empty or not pd.api.types.is_datetime64_any_dtype(df.index):
            return df
        
        df_aligned = df.copy()
        
        # Chuyển tzinfo nếu cần
        if df_aligned.index.tz is None:
            df_aligned.index = df_aligned.index.tz_localize('Asia/Bangkok')
        
        # Căn chỉnh timestamp theo timeframe
        if timeframe == '5m':
            df_aligned.index = df_aligned.index.floor('5min')
        elif timeframe == '15m':
            df_aligned.index = df_aligned.index.floor('15min')
        elif timeframe == '30m':
            df_aligned.index = df_aligned.index.floor('30min')
        elif timeframe == '1h':
            df_aligned.index = df_aligned.index.floor('H')
        elif timeframe == '4h':
            df_aligned.index = df_aligned.index.floor('4H')
        elif timeframe == '1D':
            df_aligned.index = df_aligned.index.floor('D')
        elif timeframe == '1W':
            df_aligned.index = df_aligned.index.floor('W')
        elif timeframe == '1M':
            df_aligned.index = df_aligned.index.floor('MS')
            
        # Xử lý dữ liệu bị trùng sau khi căn chỉnh
        if df_aligned.index.duplicated().any():
            # Gộp các dòng lại với nhau
            df_aligned = df_aligned.groupby(level=0).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
        
        return df_aligned
    
    @staticmethod
    def process_dataframe(df: pd.DataFrame, timeframe: str, handle_outliers: bool = True) -> (pd.DataFrame, dict):
        """
        Xử lý toàn diện DataFrame:
        1. Chuẩn hóa dữ liệu
        2. Điền giá trị thiếu
        3. Phát hiện và xử lý outlier
        4. Căn chỉnh timestamp
        
        Returns:
        - DataFrame đã xử lý
        - Dict báo cáo quá trình
        """
        if df is None or df.empty:
            return df, {"error": "DataFrame rỗng"}
        
        report = {"steps": []}
        
        # 1. Chuẩn hóa dữ liệu
        df = DataNormalizer.normalize_dataframe(df)
        report["steps"].append("Chuẩn hóa dữ liệu thành công")
        
        # 2. Xác thực dữ liệu
        is_valid, error_msg = DataNormalizer.validate_data(df)
        report["validation"] = {"valid": is_valid, "message": error_msg if not is_valid else "Dữ liệu hợp lệ"}
        
        # 3. Điền giá trị thiếu
        na_count_before = df.isna().sum().sum()
        if na_count_before > 0:
            df = DataNormalizer.fill_missing_values(df)
            na_count_after = df.isna().sum().sum()
            report["steps"].append(f"Điền {na_count_before - na_count_after} giá trị thiếu")
        
        # 4. Phát hiện và xử lý outlier
        if handle_outliers:
            df, outlier_report = DataValidator.detect_and_handle_outliers(df)
            report["outliers"] = outlier_report
            if outlier_report["total"] > 0:
                report["steps"].append(f"Xử lý {outlier_report['total']} giá trị outlier")
        
        # 5. Căn chỉnh timestamp
        df = DataValidator.align_timestamps(df, timeframe)
        report["steps"].append(f"Căn chỉnh timestamp theo khung thời gian {timeframe}")
        
        return df, report

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

class SchemaVersion(Base):
    __tablename__ = 'schema_version'
    id = Column(Integer, primary_key=True)
    version = Column(String, nullable=False)
    applied_at = Column(DateTime, default=datetime.now)
    description = Column(String, nullable=True)
    
    @classmethod
    async def get_current_version(cls, session):
        result = await session.execute(select(cls).order_by(cls.id.desc()).limit(1))
        version = result.scalars().first()
        return version.version if version else "0.0.0"
        
    @classmethod
    async def update_version(cls, session, version, description=None):
        new_version = cls(version=version, description=description)
        session.add(new_version)
        await session.commit()
        return new_version

class TrainedModel(Base):
    __tablename__ = 'trained_models'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    performance = Column(Float, nullable=True)
    # Thêm các trường mới
    version = Column(String, nullable=True)
    params = Column(Text, nullable=True)  # JSON serialized parameters
    timeframe = Column(String, nullable=True)  # e.g. 1D, 1W, etc.

class ModelPerformanceHistory(Base):
    __tablename__ = 'model_performance_history'
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('trained_models.id'), nullable=False)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    version = Column(String, nullable=True)
    timeframe = Column(String, nullable=True)
    performance = Column(Float, nullable=False)
    training_date = Column(DateTime, default=datetime.now)
    test_period_start = Column(DateTime, nullable=True)
    test_period_end = Column(DateTime, nullable=True)
    metrics = Column(Text, nullable=True)  # JSON serialized metrics
    
    # Relationship to trained model
    model = relationship("TrainedModel", backref="performance_history")

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ---------- DATABASE MIGRATION ----------
async def migrate_database():
    """
    Kiểm tra và cập nhật schema database nếu cần
    """
    CURRENT_VERSION = "1.1.0"  # Tăng version khi có thay đổi schema
    
    async with SessionLocal() as session:
        try:
            # Kiểm tra version hiện tại
            current_version = await SchemaVersion.get_current_version(session)
            logger.info(f"Phiên bản database hiện tại: {current_version}")
            
            if current_version == CURRENT_VERSION:
                logger.info("Database đã ở phiên bản mới nhất.")
                return
            
            # Migration từ version cũ (0.0.0) lên 1.0.0
            if current_version == "0.0.0":
                logger.info("Đang nâng cấp database lên phiên bản 1.0.0...")
                
                # Tạo bản ghi đầu tiên trong SchemaVersion
                await SchemaVersion.update_version(session, "1.0.0", "Ban đầu: thêm bảng SchemaVersion, ModelPerformanceHistory")
                current_version = "1.0.0"
                
                # Kiểm tra các model hiện có và cập nhật trường mới
                query = await session.execute(select(TrainedModel))
                existing_models = query.scalars().all()
                
                for model in existing_models:
                    # Cập nhật các trường mới nếu là null
                    if model.version is None:
                        model.version = "1.0.0"
                    if model.timeframe is None:
                        model.timeframe = "1D"
                    if model.params is None:
                        model.params = json.dumps({})
                
                await session.commit()
                logger.info(f"Đã cập nhật {len(existing_models)} model cũ")
            
            # Migration từ 1.0.0 lên 1.1.0
            if current_version == "1.0.0":
                logger.info("Đang nâng cấp database lên phiên bản 1.1.0...")
                
                # Thêm logic migration từ 1.0.0 lên 1.1.0 ở đây
                await SchemaVersion.update_version(session, "1.1.0", "Thêm hỗ trợ intraday timeframes và optimizations")
                current_version = "1.1.0"
            
            logger.info(f"Nâng cấp database lên phiên bản {CURRENT_VERSION} thành công")
        
        except Exception as e:
            logger.error(f"Lỗi trong quá trình migration database: {str(e)}")
            raise

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
    async def store_trained_model(self, symbol: str, model_type: str, model, performance: float = None, 
                                 version: str = "1.0.0", params: dict = None, timeframe: str = "1D"):
        try:
            model_blob = pickle.dumps(model)
            params_json = json.dumps(params) if params else None
            
            async with self.Session() as session:
                result = await session.execute(select(TrainedModel).filter_by(symbol=symbol, model_type=model_type))
                existing = result.scalar_one_or_none()
                
                if existing:
                    # Lưu hiệu suất cũ vào lịch sử trước khi cập nhật
                    if existing.performance is not None:
                        await self.track_model_performance(existing.id, symbol, model_type, existing.performance, 
                                                          existing.version, existing.timeframe, 
                                                          json.loads(existing.params) if existing.params else None)
                    
                    # Cập nhật mô hình hiện có
                    existing.model_blob = model_blob
                    existing.created_at = datetime.now()
                    existing.performance = performance
                    existing.version = version
                    existing.params = params_json
                    existing.timeframe = timeframe
                    model_id = existing.id
                else:
                    # Tạo mô hình mới
                    new_model = TrainedModel(
                        symbol=symbol, 
                        model_type=model_type, 
                        model_blob=model_blob, 
                        performance=performance,
                        version=version,
                        params=params_json,
                        timeframe=timeframe
                    )
                    session.add(new_model)
                    await session.flush()  # Để lấy ID của model mới
                    model_id = new_model.id
                
                await session.commit()
                
                # Thêm vào lịch sử hiệu suất
                if performance is not None:
                    await self.track_model_performance(model_id, symbol, model_type, performance, 
                                                     version, timeframe, params)
                
            logger.info(f"Lưu mô hình {model_type} cho {symbol} (timeframe: {timeframe}) thành công với hiệu suất: {performance}")
        except Exception as e:
            logger.error(f"Lỗi lưu mô hình {model_type} cho {symbol}: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_trained_model(self, symbol: str, model_type: str, timeframe: str = "1D"):
        try:
            async with self.Session() as session:
                result = await session.execute(
                    select(TrainedModel)
                    .filter_by(symbol=symbol, model_type=model_type)
                    .filter((TrainedModel.timeframe == timeframe) | (TrainedModel.timeframe == None))
                )
                model_record = result.scalar_one_or_none()
                if model_record:
                    model = pickle.loads(model_record.model_blob)
                    
                    model_info = {
                        'performance': model_record.performance,
                        'version': model_record.version,
                        'created_at': model_record.created_at,
                        'timeframe': model_record.timeframe or 'unknown',
                        'params': json.loads(model_record.params) if model_record.params else None
                    }
                    
                    return model, model_info
            
            # Nếu không tìm thấy model với timeframe chính xác, thử tải model bất kỳ
            if timeframe != "1D":
                logger.info(f"Không tìm thấy model {model_type} cho {symbol} với timeframe {timeframe}, thử tải model mặc định")
                return await self.load_trained_model(symbol, model_type)
                
            return None, None
        except Exception as e:
            logger.error(f"Lỗi tải mô hình {model_type} cho {symbol}: {str(e)}")
            return None, None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def track_model_performance(self, model_id: int, symbol: str, model_type: str, 
                                     performance: float, version: str = None, 
                                     timeframe: str = None, params: dict = None,
                                     test_period_start: datetime = None, 
                                     test_period_end: datetime = None,
                                     metrics: dict = None):
        """
        Theo dõi lịch sử hiệu suất của mô hình qua các lần huấn luyện
        """
        try:
            metrics_json = json.dumps(metrics) if metrics else None
            
            async with self.Session() as session:
                history_entry = ModelPerformanceHistory(
                    model_id=model_id,
                    symbol=symbol,
                    model_type=model_type,
                    version=version,
                    timeframe=timeframe,
                    performance=performance,
                    test_period_start=test_period_start,
                    test_period_end=test_period_end,
                    metrics=metrics_json
                )
                session.add(history_entry)
                await session.commit()
                
            logger.info(f"Đã lưu lịch sử hiệu suất mô hình {model_type} cho {symbol} với hiệu suất: {performance}")
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu lịch sử hiệu suất mô hình: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get_model_performance_history(self, symbol: str = None, model_type: str = None, 
                                           timeframe: str = None, limit: int = 10):
        """
        Lấy lịch sử hiệu suất của mô hình
        """
        try:
            async with self.Session() as session:
                query = select(ModelPerformanceHistory).order_by(ModelPerformanceHistory.training_date.desc())
                
                if symbol:
                    query = query.filter(ModelPerformanceHistory.symbol == symbol)
                if model_type:
                    query = query.filter(ModelPerformanceHistory.model_type == model_type)
                if timeframe:
                    query = query.filter(ModelPerformanceHistory.timeframe == timeframe)
                    
                query = query.limit(limit)
                result = await session.execute(query)
                history = result.scalars().all()
                
                return [
                    {
                        'id': entry.id,
                        'model_id': entry.model_id,
                        'symbol': entry.symbol,
                        'model_type': entry.model_type,
                        'version': entry.version,
                        'timeframe': entry.timeframe,
                        'performance': entry.performance,
                        'training_date': entry.training_date.isoformat(),
                        'test_period': {
                            'start': entry.test_period_start.isoformat() if entry.test_period_start else None,
                            'end': entry.test_period_end.isoformat() if entry.test_period_end else None
                        },
                        'metrics': json.loads(entry.metrics) if entry.metrics else None
                    }
                    for entry in history
                ]
        except Exception as e:
            logger.error(f"Lỗi lấy lịch sử hiệu suất mô hình: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get_latest_model_version(self, symbol: str = None, model_type: str = None):
        """
        Lấy phiên bản mới nhất của mô hình
        """
        try:
            async with self.Session() as session:
                query = select(TrainedModel).order_by(TrainedModel.created_at.desc())
                
                if symbol:
                    query = query.filter(TrainedModel.symbol == symbol)
                if model_type:
                    query = query.filter(TrainedModel.model_type == model_type)
                    
                query = query.limit(1)
                result = await session.execute(query)
                model = result.scalar_one_or_none()
                
                if not model:
                    return "0.0.0"
                    
                return model.version or "1.0.0"
        except Exception as e:
            logger.error(f"Lỗi lấy phiên bản mô hình: {str(e)}")
            return "0.0.0"

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
        # Chuẩn hóa timeframe
        timeframe_map = {
            '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '4h',
            '1d': '1D', 'd': '1D', 'day': '1D',
            '1w': '1W', 'w': '1W', 'week': '1W',
            '1mo': '1M', 'mo': '1M', 'month': '1M'
        }
        timeframe = timeframe_map.get(timeframe.lower(), timeframe).upper()
        
        # Validate các tham số đầu vào
        valid_ticker, ticker_msg = DataValidator.validate_ticker(symbol)
        if not valid_ticker:
            raise ValueError(ticker_msg)
            
        valid_timeframe, timeframe_msg = DataValidator.validate_timeframe(timeframe)
        if not valid_timeframe:
            raise ValueError(timeframe_msg)
            
        valid_candles, candles_msg = DataValidator.validate_candles_count(num_candles)
        if not valid_candles:
            raise ValueError(candles_msg)
        
        # Thiết lập thời gian cache dựa trên timeframe
        if timeframe in ['5M', '15M', '30M']:
            expire = CACHE_EXPIRE_SHORT  # Dữ liệu intraday hết hạn nhanh hơn
        elif timeframe in ['1H', '4H', '1D']:
            expire = CACHE_EXPIRE_MEDIUM
        else:
            expire = CACHE_EXPIRE_LONG
        
        cache_key = f"data_{self.source}_{symbol}_{timeframe}_{num_candles}"
        cached_data = await redis_manager.get(cache_key)
        if cached_data is not None:
            return cached_data, "Dữ liệu từ cache, không kiểm tra outlier"

        try:
            if self.source == 'vnstock':
                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
                def fetch_vnstock():
                    stock = Vnstock().stock(symbol=symbol, source='TCBS')
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    
                    # Điều chỉnh số ngày lấy dữ liệu dựa trên timeframe
                    if timeframe in ['5M', '15M', '30M', '1H']:
                        # Dữ liệu intraday chỉ có trong 7-30 ngày gần nhất
                        days_back = 30
                    elif timeframe == '4H':
                        days_back = 60
                    elif timeframe == '1D':
                        days_back = (num_candles + 1) * 3  # 3x cho ngày nghỉ
                    elif timeframe == '1W':
                        days_back = (num_candles + 1) * 7 * 2  # 2x cho tuần nghỉ
                    else:  # 1M
                        days_back = (num_candles + 1) * 30 * 1.5  # 1.5x cho tháng
                    
                    start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
                    
                    try:
                        # Lấy dữ liệu theo timeframe
                        if timeframe in ['5M', '15M', '30M', '1H', '4H']:
                            df = stock.quote.intraday(timeframe.lower(), from_date=start_date, to_date=end_date)
                        else:
                            df = stock.quote.history(start=start_date, end=end_date, interval=timeframe)
                    except Exception as e:
                        logger.error(f"Lỗi tải dữ liệu {timeframe} cho {symbol}: {str(e)}")
                        # Nếu không lấy được dữ liệu intraday, thử dùng dữ liệu daily và resampling
                        if timeframe in ['5M', '15M', '30M', '1H', '4H']:
                            logger.warning(f"Thử lấy dữ liệu daily cho {symbol} và resampling về {timeframe}")
                            # Lấy dữ liệu daily và resample sau
                            df = stock.quote.history(start=start_date, end=end_date, interval='1D')
                            raise ValueError(f"Dữ liệu intraday cho {symbol} không khả dụng")
                    
                    if df is None or df.empty or len(df) < DataValidator.MIN_CANDLES:
                        raise ValueError(f"Không đủ dữ liệu cho {'chỉ số' if is_index(symbol) else 'mã'} {symbol}")
                    
                    # Xử lý và chuẩn hóa dữ liệu
                    df, report = DataValidator.process_dataframe(df, timeframe)
                    
                    if len(df) < num_candles:
                        logger.warning(f"Chỉ có {len(df)} nến cho {symbol}, yêu cầu {num_candles}")
                    
                    return df.tail(min(num_candles + 1, len(df)))
                
                df = await run_in_thread(fetch_vnstock)
            
            elif self.source == 'yahoo':
                # Mapping giữa timeframe của chúng ta với Yahoo
                yahoo_period_map = {
                    '1D': '1d', '5M': '5m', '15M': '15m', '30M': '30m', 
                    '1H': '1h', '4H': '4h', '1W': '1wk', '1M': '1mo'
                }
                
                if timeframe not in yahoo_period_map:
                    raise ValueError(f"Yahoo không hỗ trợ timeframe {timeframe}")
                
                # Điều chỉnh interval và period theo timeframe
                interval = yahoo_period_map.get(timeframe)
                
                # Tính toán period dựa trên timeframe
                if timeframe in ['5M', '15M', '30M']:
                    period = '7d'  # Yahoo có dữ liệu intraday trong khoảng 7 ngày
                elif timeframe in ['1H', '4H']:
                    period = '60d'  # Có thể lấy được dữ liệu 60 ngày với 1h và 4h
                elif timeframe == '1D':
                    period = f"{num_candles * 3}d"  # 3x cho ngày nghỉ
                elif timeframe == '1W':
                    period = f"{int(num_candles * 1.5)}mo"  # Weekly data
                else:  # 1M
                    period = f"{num_candles * 2}y"  # Monthly data
                
                df = await self._download_yahoo_data(symbol, num_candles + 1, interval, period)
                if df is None or df.empty or len(df) < DataValidator.MIN_CANDLES:
                    raise ValueError(f"Không đủ dữ liệu cho {symbol} từ Yahoo Finance")
                
                # Xử lý và chuẩn hóa dữ liệu
                df, report = DataValidator.process_dataframe(df, timeframe)
                
                if len(df) < num_candles:
                    logger.warning(f"Chỉ có {len(df)} nến cho {symbol}, yêu cầu {num_candles}")
            else:
                raise ValueError("Nguồn dữ liệu không hợp lệ")

            # Lọc ngày giao dịch (chỉ áp dụng cho 1D trở lên)
            if timeframe in ['1D', '1W', '1M']:
                trading_df = filter_trading_days(df)
            else:
                trading_df = df
                
            # Phát hiện outlier cuối cùng
            trading_df, outlier_report = DataNormalizer.detect_outliers(trading_df)
            await redis_manager.set(cache_key, trading_df, expire=expire)
            return trading_df, outlier_report
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu cho {symbol}: {str(e)}")
            raise ValueError(f"Không thể tải dữ liệu: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
    async def _download_yahoo_data(self, symbol: str, num_candles: int, interval: str, period: str) -> pd.DataFrame:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                # Xác định khoảng thời gian dựa trên interval
                if interval in ['5m', '15m', '30m', '1h', '4h']:
                    # Với dữ liệu intraday, Yahoo giới hạn khoảng thời gian ngắn hơn
                    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                           f"?interval={interval}&range={period}")
                else:
                    # Với dữ liệu daily trở lên, sử dụng cách tải trực tiếp
                    start_ts = int((datetime.now() - timedelta(days=num_candles * 3)).timestamp())
                    end_ts = int(datetime.now().timestamp())
                    url = (f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
                           f"?period1={start_ts}&period2={end_ts}&interval={interval}&events=history")
                
                async with asyncio.wait_for(session.get(url), timeout=15) as response:
                    if response.status != 200:
                        raise ValueError(f"Không thể tải dữ liệu từ Yahoo, HTTP {response.status}")
                    
                    if interval in ['5m', '15m', '30m', '1h', '4h']:
                        # Parse JSON cho dữ liệu intraday
                        data = await response.json()
                        
                        if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                            raise ValueError("Cấu trúc dữ liệu Yahoo không hợp lệ")
                        
                        result = data['chart']['result'][0]
                        timestamps = result['timestamp']
                        quote = result['indicators']['quote'][0]
                        
                        df = pd.DataFrame({
                            'date': pd.to_datetime(timestamps, unit='s'),
                            'open': quote.get('open', []),
                            'high': quote.get('high', []),
                            'low': quote.get('low', []),
                            'close': quote.get('close', []),
                            'volume': quote.get('volume', [])
                        })
                        
                        df = df.set_index('date')
                    else:
                        # Parse CSV cho dữ liệu daily
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
        Chuẩn bị dữ liệu toàn diện cho một mã chứng khoán, bao gồm:
        - Dữ liệu kỹ thuật đa khung thời gian
        - Dữ liệu cơ bản
        - Dự báo và các tín hiệu khác
        
        Args:
            symbol: Mã chứng khoán
            timeframes: Danh sách khung thời gian cần lấy, mặc định ['1D', '1W']
            num_candles: Số lượng nến cần lấy, mặc định 200
            
        Returns:
            Dictionary chứa tất cả dữ liệu đã xử lý
        """
        if timeframes is None:
            timeframes = ['1D', '1W']
        
        # Validate timeframes
        valid_timeframes = []
        for tf in timeframes:
            is_valid, _ = DataValidator.validate_timeframe(tf)
            if is_valid:
                valid_timeframes.append(tf)
            else:
                logger.warning(f"Bỏ qua timeframe không hợp lệ: {tf}")
                
        if not valid_timeframes:
            valid_timeframes = ['1D']  # Mặc định nếu không có timeframe nào hợp lệ
        
        # Validate symbol
        is_valid_symbol, msg = DataValidator.validate_ticker(symbol)
        if not is_valid_symbol:
            raise ValueError(msg)
        
        # Validate số lượng nến
        is_valid_candles, msg = DataValidator.validate_candles_count(num_candles)
        if not is_valid_candles:
            num_candles = DataValidator.MIN_CANDLES
            logger.warning(f"{msg}. Sử dụng số lượng nến tối thiểu: {num_candles}")
        
        # Dữ liệu thị trường
        loader = self.data_loader
        analyzer = self.tech_analyzer
        
        # Lấy dữ liệu đa khung thời gian
        data_dict = {}
        outlier_reports = {}
        tech_data = {}
        tasks = []
        
        for tf in valid_timeframes:
            tasks.append(loader.load_data(symbol, tf, num_candles))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, tf in enumerate(valid_timeframes):
            if isinstance(results[i], Exception):
                logger.error(f"Lỗi tải dữ liệu cho {symbol} timeframe {tf}: {str(results[i])}")
                continue
                
            df, outlier_report = results[i]
            if df is not None and not df.empty:
                data_dict[tf] = df
                outlier_reports[tf] = outlier_report
                tech_data[tf] = analyzer.calculate_indicators(df)
        
        # Nếu không có dữ liệu nào, báo lỗi
        if not data_dict:
            raise ValueError(f"Không thể tải dữ liệu cho {symbol} với bất kỳ timeframe nào")
        
        # Dữ liệu cơ bản
        try:
            fundamental_data = await loader.get_fundamental_data(symbol)
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu cơ bản cho {symbol}: {str(e)}")
            fundamental_data = {"error": str(e)}
        
        # Dự báo và tín hiệu
        forecast_data = {}
        signal_data = {}
        
        # Ưu tiên sử dụng dữ liệu daily cho dự báo
        if '1D' in tech_data:
            # Dự báo Prophet
            try:
                prophet_model, _ = await model_db_manager.load_trained_model(symbol, 'prophet', '1D')
                if prophet_model:
                    df_prophet = prepare_data_for_prophet(data_dict['1D'])
                    forecast, _ = forecast_with_prophet(df_prophet, model=prophet_model)
                    forecast_data['prophet'] = forecast.to_dict(orient='records')
            except Exception as e:
                logger.error(f"Lỗi dự báo Prophet cho {symbol}: {str(e)}")
            
            # Tín hiệu XGBoost
            try:
                xgb_model, _ = await model_db_manager.load_trained_model(symbol, 'xgboost', '1D')
                if xgb_model:
                    signal, prob = predict_xgboost_signal(tech_data['1D'], features=[
                        'sma20', 'sma50', 'rsi', 'macd', 'signal', 'bb_high', 'bb_low', 'volume'
                    ])
                    signal_data['xgboost'] = {'signal': int(signal), 'probability': float(prob)}
            except Exception as e:
                logger.error(f"Lỗi dự báo XGBoost cho {symbol}: {str(e)}")
        
        # Mẫu nến và patterns
        pattern_data = self.extract_patterns(tech_data)
        
        # Tổng hợp dữ liệu
        return {
            'symbol': symbol,
            'timeframes': list(data_dict.keys()),
            'data': data_dict,
            'technical': tech_data,
            'fundamental': fundamental_data,
            'outliers': outlier_reports,
            'forecast': forecast_data,
            'signals': signal_data,
            'patterns': pattern_data,
            'last_update': datetime.now().isoformat()
        }
    
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

async def train_models_for_symbol(symbol: str, timeframes: list = None):
    """
    Huấn luyện các mô hình cho một mã chứng khoán
    """
    if timeframes is None:
        timeframes = ['1D']  # Mặc định, chỉ train cho khung thời gian ngày
    
    loader = DataLoader()
    
    for timeframe in timeframes:
        try:
            # Lấy thông tin phiên bản mới nhất cho model
            current_version = await model_db_manager.get_latest_model_version(symbol=symbol)
            version_parts = current_version.split('.')
            new_version = f"{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}"
            
            # Tải dữ liệu theo timeframe
            df, _ = await loader.load_data(symbol, timeframe, 365 if timeframe == '1D' else 500)
            if df is None or len(df) < 100:
                logger.warning(f"Không đủ dữ liệu cho {symbol} trong timeframe {timeframe} để huấn luyện")
                continue
            
            # Huấn luyện mô hình Prophet
            if timeframe in ['1D', '1W']:  # Prophet tốt cho dữ liệu ngày, tuần
                df_prophet = prepare_data_for_prophet(df)
                try:
                    prophet_model, prophet_perf = train_prophet_model(df_prophet)
                    
                    # Lưu mô hình với các thông tin mới
                    prophet_params = {
                        "seasonality_mode": "multiplicative",
                        "yearly_seasonality": True,
                        "weekly_seasonality": True,
                        "daily_seasonality": False if timeframe != '1D' else True,
                        "datapoints": len(df)
                    }
                    
                    await model_db_manager.store_trained_model(
                        symbol=symbol, 
                        model_type='prophet', 
                        model=prophet_model, 
                        performance=prophet_perf,
                        version=new_version,
                        params=prophet_params,
                        timeframe=timeframe
                    )
                    
                    logger.info(f"Huấn luyện mô hình Prophet cho {symbol} ({timeframe}) thành công, MAPE: {prophet_perf:.4f}")
                except Exception as e:
                    logger.error(f"Lỗi huấn luyện Prophet cho {symbol} ({timeframe}): {str(e)}")
            
            # Huấn luyện mô hình XGBoost
            features = ['sma20', 'sma50', 'rsi', 'macd', 'signal', 'bb_high', 'bb_low', 'volume']
            df_xgb = TechnicalAnalyzer()._calculate_common_indicators(df)
            df_xgb = df_xgb.dropna()
            
            if len(df_xgb) < 100:
                logger.warning(f"Không đủ dữ liệu cho {symbol} ({timeframe}) để huấn luyện XGBoost sau khi tính toán chỉ báo")
                continue
                
            try:
                xgb_model, xgb_perf = train_xgboost_model(df_xgb, features)
                
                # Lưu mô hình với các thông tin mới
                xgb_params = {
                    "features": features,
                    "target": "signal",
                    "datapoints": len(df_xgb),
                    "model_params": {
                        "max_depth": 6,
                        "learning_rate": 0.1,
                        "n_estimators": 100
                    }
                }
                
                await model_db_manager.store_trained_model(
                    symbol=symbol, 
                    model_type='xgboost', 
                    model=xgb_model, 
                    performance=xgb_perf,
                    version=new_version,
                    params=xgb_params,
                    timeframe=timeframe
                )
                
                logger.info(f"Huấn luyện mô hình XGBoost cho {symbol} ({timeframe}) thành công, Accuracy: {xgb_perf:.4f}")
            except Exception as e:
                logger.error(f"Lỗi huấn luyện XGBoost cho {symbol} ({timeframe}): {str(e)}")
        
        except Exception as e:
            logger.error(f"Lỗi tổng thể khi huấn luyện cho {symbol} ({timeframe}): {str(e)}")
    
    return True

async def auto_train_models():
    """Tự động huấn luyện lại các mô hình định kỳ"""
    try:
        logger.info("Bắt đầu quá trình auto training cho tất cả mã...")
        symbols = await get_training_symbols()
        if not symbols:
            logger.warning("Không có mã nào để huấn luyện")
            return
        
        # Danh sách timeframes để train - chỉ train các khung thời gian phổ biến
        timeframes = ['1D', '1W', '1H', '4H']
        
        tasks = []
        for symbol in symbols:
            logger.info(f"Lên lịch auto training cho mã: {symbol}")
            tasks.append(train_models_for_symbol(symbol, timeframes))
        
        if tasks:
            # Chạy tối đa 5 mã cùng lúc để tránh quá tải
            for batch in [tasks[i:i+5] for i in range(0, len(tasks), 5)]:
                await asyncio.gather(*batch)
                # Nghỉ một chút giữa các batch
                await asyncio.sleep(10)
            
            logger.info(f"Đã hoàn thành auto training cho {len(symbols)} mã")
        else:
            logger.warning("Không có task training nào được lên lịch")
    
    except Exception as e:
        logger.error(f"Lỗi trong auto_train_models: {str(e)}")

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
        
    def calculate_support_resistance_levels(self, df: pd.DataFrame, num_levels: int = 3) -> dict:
        """
        Tự tính toán các mức hỗ trợ và kháng cự dựa trên phương pháp:
        1. Xác định đỉnh và đáy trong dữ liệu lịch sử
        2. Phân cụm các đỉnh và đáy gần nhau
        3. Sắp xếp theo tần suất xuất hiện
        """
        if len(df) < 20:
            return {"support_levels": [], "resistance_levels": []}
            
        # Tìm đỉnh và đáy cục bộ
        n = 5  # Window size cho việc xác định đỉnh/đáy
        df_ext = df.copy()
        
        # 1. Xác định đỉnh
        df_ext['is_peak'] = False
        for i in range(n, len(df_ext) - n):
            if all(df_ext['high'].iloc[i] > df_ext['high'].iloc[i-j] for j in range(1, n+1)) and \
               all(df_ext['high'].iloc[i] > df_ext['high'].iloc[i+j] for j in range(1, n+1)):
                df_ext.loc[df_ext.index[i], 'is_peak'] = True
        
        # 2. Xác định đáy
        df_ext['is_valley'] = False
        for i in range(n, len(df_ext) - n):
            if all(df_ext['low'].iloc[i] < df_ext['low'].iloc[i-j] for j in range(1, n+1)) and \
               all(df_ext['low'].iloc[i] < df_ext['low'].iloc[i+j] for j in range(1, n+1)):
                df_ext.loc[df_ext.index[i], 'is_valley'] = True
        
        # 3. Lấy giá trị đỉnh và đáy
        peaks = df_ext[df_ext['is_peak']]['high'].tolist()
        valleys = df_ext[df_ext['is_valley']]['low'].tolist()
        
        # 4. Phân cụm giá trị gần nhau (đơn giản hóa)
        def cluster_prices(prices, threshold_pct=0.02):
            if not prices:
                return []
            # Sắp xếp giá
            sorted_prices = sorted(prices)
            clusters = []
            current_cluster = [sorted_prices[0]]
            
            for price in sorted_prices[1:]:
                # Nếu giá hiện tại gần với trung bình cụm hiện tại
                if price <= current_cluster[0] * (1 + threshold_pct):
                    current_cluster.append(price)
                else:
                    # Tính trung bình cụm và thêm vào danh sách
                    clusters.append(sum(current_cluster) / len(current_cluster))
                    current_cluster = [price]
            
            # Thêm cụm cuối cùng
            if current_cluster:
                clusters.append(sum(current_cluster) / len(current_cluster))
            
            return clusters
        
        resistance_clusters = cluster_prices(peaks)
        support_clusters = cluster_prices(valleys)
        
        # 5. Lấy N mức có tần suất xuất hiện cao nhất
        current_price = df['close'].iloc[-1]
        
        # Lọc kháng cự phía trên giá hiện tại
        resistance_levels = sorted([r for r in resistance_clusters if r > current_price])[:num_levels]
        
        # Lọc hỗ trợ phía dưới giá hiện tại
        support_levels = sorted([s for s in support_clusters if s < current_price], reverse=True)[:num_levels]
        
        # Format kết quả
        support_levels = [round(float(price), 2) for price in support_levels]
        resistance_levels = [round(float(price), 2) for price in resistance_levels]
        
        return {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels
        }

    async def analyze_with_openrouter(self, technical_data):
        if not OPENROUTER_API_KEY:
            raise Exception("Chưa có OPENROUTER_API_KEY")

        # Tính toán mức hỗ trợ/kháng cự từ dữ liệu candlestick
        df = pd.DataFrame(technical_data["candlestick_data"])
        calculated_levels = self.calculate_support_resistance_levels(df)
        
        prompt = (
            "Bạn là chuyên gia phân tích kỹ thuật chứng khoán với 20 năm kinh nghiệm."
            " Dựa trên dữ liệu dưới đây, hãy nhận diện các mẫu hình nến phổ biến"
            " sóng Elliott, mô hình Wyckoff, và các vùng hỗ trợ/kháng cự với phương pháp sau:"
            "\n\n1. Mức hỗ trợ: Xác định các mức giá thấp lặp lại nhiều lần và luôn bật lên, các đáy rõ ràng"
            "\n2. Mức kháng cự: Xác định các mức giá cao lặp lại nhiều lần và thường bị bán mạnh, các đỉnh rõ ràng"
            "\n3. Chọn các mức có tần suất kiểm tra cao (giá chạm nhiều lần)"
            "\n4. Mức hỗ trợ phải thấp hơn giá hiện tại, mức kháng cự phải cao hơn giá hiện tại"
            "\n5. Giá trị mức hỗ trợ và kháng cự phải tương đồng về độ lớn với giá hiện tại"
            f"\n\nGiá hiện tại: {df['close'].iloc[-1]:.2f}"
            "\n\nChỉ trả về kết quả ở dạng JSON như sau, không thêm giải thích nào khác và không bọc trong cặp dấu ```:\n"
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
        "🚀 **V18.9 - THUA GIA CÁT LƯỢNG MỖI CÁI QUẠT!**\n"
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
        if num_candles < 20:
            raise ValueError("Số nến phải lớn hơn hoặc bằng 20 để tính toán chỉ báo!")
        if num_candles > 500:
            raise ValueError("Tối đa 500 nến!")
        
        # Sử dụng pipeline chuẩn hóa
        data_pipeline = DataPipeline()
        ai_analyzer = AIAnalyzer()
        
        # Chuẩn bị dữ liệu với pipeline
        await update.message.reply_text(f"⏳ Đang chuẩn bị dữ liệu cho {symbol}...")
        pipeline_result = await data_pipeline.prepare_symbol_data(symbol, timeframes=['1D', '1W', '1M'], num_candles=num_candles)
        
        if pipeline_result['errors']:
            error_message = f"⚠️ Một số lỗi xảy ra trong quá trình chuẩn bị dữ liệu:\n"
            error_message += "\n".join(pipeline_result['errors'])
            await update.message.reply_text(error_message)
        
        if not pipeline_result['dataframes']:
            raise ValueError(f"Không thể tải dữ liệu cho {symbol}")
        
        # Tạo báo cáo với AI
        await update.message.reply_text(f"⏳ Đang phân tích {symbol} với AI...")
        report = await ai_analyzer.generate_report(
            pipeline_result['dataframes'], 
            symbol, 
            pipeline_result['fundamental_data'], 
            pipeline_result['outlier_reports']
        )
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
    logger.info("Khởi động bot...")
    
    # Khởi tạo database
    await init_db()
    
    # Thực hiện migrate database nếu cần
    await migrate_database()
    
    # Khởi động scheduler cho auto training
    scheduler = AsyncIOScheduler()
    scheduler.add_job(auto_train_models, 'cron', hour=2, minute=0)
    scheduler.start()
    logger.info("Auto training scheduler đã khởi động.")

    # Thiết lập Telegram bot với webhook tối ưu cho Render Cloud
    BASE_URL = os.getenv("RENDER_EXTERNAL_URL", f"https://{os.getenv('RENDER_SERVICE_NAME')}.onrender.com")
    WEBHOOK_URL = f"{BASE_URL}/{TELEGRAM_TOKEN}"
    WEBHOOK_PATH = f"/{TELEGRAM_TOKEN}"

    # Khởi tạo bot application
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("getid", get_id))
    application.add_handler(CommandHandler("approve", approve_user))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, notify_admin_new_user))
    
    # Tạo ứng dụng web aiohttp
    async def setup_webhook():
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                webhook_info = await application.bot.get_webhook_info()
                if webhook_info.url != WEBHOOK_URL:
                    await application.bot.set_webhook(url=WEBHOOK_URL)
                    logger.info(f"Webhook đã được thiết lập thành công: {WEBHOOK_URL}")
                else:
                    logger.info(f"Webhook đã được thiết lập trước đó: {WEBHOOK_URL}")
                return
            except Exception as e:
                retry_count += 1
                logger.error(f"Lỗi thiết lập webhook (thử lần {retry_count}): {str(e)}")
                await asyncio.sleep(5)
    
    # Tạo web server với aiohttp
    from aiohttp import web
    
    async def webhook_handler(request):
        # Đọc và xử lý update từ Telegram
        update_data = await request.json()
        await application.update_queue.put(update_data)
        return web.Response(status=200)
    
    async def health_check(request):
        # Route để kiểm tra trạng thái hoạt động của server
        return web.Response(text='OK', status=200)
    
    async def start_webhook():
        # Thiết lập webhook
        await setup_webhook()
        
        # Khởi tạo ứng dụng web
        web_app = web.Application()
        web_app.router.add_post(WEBHOOK_PATH, webhook_handler)
        web_app.router.add_get('/health', health_check)
        
        # Khởi động web server
        runner = web.AppRunner(web_app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', PORT)
        
        logger.info(f"🤖 Bot khởi động với webhook tại: {WEBHOOK_URL}")
        logger.info(f"Server lắng nghe tại: 0.0.0.0:{PORT}")
        
        # Khởi động webhook processing
        await application.start()
        await site.start()
        
        # Giữ cho ứng dụng chạy liên tục
        while True:
            await asyncio.sleep(3600)  # Kiểm tra mỗi giờ
            
            # Kiểm tra và thiết lập lại webhook nếu cần
            try:
                webhook_info = await application.bot.get_webhook_info()
                if not webhook_info.url or webhook_info.url != WEBHOOK_URL:
                    logger.warning(f"Webhook không hoạt động hoặc không đúng. Thiết lập lại.")
                    await application.bot.set_webhook(url=WEBHOOK_URL)
            except Exception as e:
                logger.error(f"Lỗi kiểm tra webhook: {str(e)}")
    
    # Khởi động webhook server
    await start_webhook()

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