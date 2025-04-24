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
from datetime import datetime, timedelta, time
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
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, select, LargeBinary, ForeignKey

import xgboost as xgb
from sklearn.metrics import accuracy_score
from prophet import Prophet

import matplotlib.pyplot as plt
import holidays
import html

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tenacity import retry, stop_after_attempt, wait_exponential

import aiohttp
from aiohttp import web  # Thêm import này
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

class DataValidator:
    """
    Lớp kiểm tra và xác thực dữ liệu nâng cao:
    - Xác thực ticker hợp lệ
    - Xác thực timeframe hợp lệ
    - Căn chỉnh timestamp theo khung thời gian
    - Xử lý nâng cao hơn cho outlier và missing values
    """
    
    # Các timeframe được hỗ trợ
    SUPPORTED_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1D', '1W', '1M']
    
    # Ánh xạ timeframe sang số phút
    TIMEFRAME_MINUTES = {
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1D': 1440,
        '1W': 10080,
        '1M': 43200
    }
    
    # Thời gian giao dịch (Vietnam)
    MARKET_HOURS = {
        'open': time(9, 0),
        'close': time(15, 0)
    }
    
    # Ngày giao dịch (Vietnam)
    TRADING_DAYS = [0, 1, 2, 3, 4]  # 0: Monday, 4: Friday
    
    @staticmethod
    def validate_ticker(ticker: str) -> (bool, str):
        """Xác thực mã chứng khoán hợp lệ"""
        if not ticker:
            return False, "Mã chứng khoán không được để trống"
        
        # Kiểm tra định dạng mã VN
        if re.match(r'^[A-Z0-9]{3,4}$', ticker):
            return True, f"{ticker} là mã hợp lệ"
        
        # Nếu là mã chỉ số
        indices = ['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX', 'UPCOM']
        if ticker.upper() in indices:
            return True, f"{ticker} là chỉ số hợp lệ"
            
        return False, f"{ticker} không phải là mã chứng khoán hợp lệ"
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> (bool, str):
        """Xác thực khung thời gian hợp lệ"""
        # Chuẩn hóa timeframe
        if timeframe.lower() in ['1d', 'daily', 'd']:
            timeframe = '1D'
        elif timeframe.lower() in ['1w', 'weekly', 'w']:
            timeframe = '1W'
        elif timeframe.lower() in ['1m', 'monthly', 'm']:
            timeframe = '1M'
        elif timeframe.lower() in ['1h', 'hourly', 'h']:
            timeframe = '1h'
        elif timeframe.lower() in ['4h', '4 hours']:
            timeframe = '4h'
        
        if timeframe not in DataValidator.SUPPORTED_TIMEFRAMES:
            return False, f"Khung thời gian {timeframe} không được hỗ trợ. Hỗ trợ: {', '.join(DataValidator.SUPPORTED_TIMEFRAMES)}"
        
        return True, f"Khung thời gian {timeframe} hợp lệ"
    
    @staticmethod
    def validate_candle_count(count: int, min_count: int = 10, max_count: int = 5000) -> (bool, str):
        """Xác thực số lượng nến hợp lệ"""
        if not isinstance(count, int) or count < min_count:
            return False, f"Số lượng nến phải là số nguyên và ít nhất {min_count}"
        
        if count > max_count:
            return False, f"Số lượng nến tối đa là {max_count}"
            
        return True, f"Số lượng nến {count} hợp lệ"
    
    @staticmethod
    def align_timestamp(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Căn chỉnh timestamp theo đúng khung thời gian"""
        if df is None or df.empty:
            return df
            
        # Chỉ xử lý cho các timeframe intraday
        if timeframe not in ['5m', '15m', '30m', '1h', '4h']:
            return df
            
        # Chuyển đổi index thành datetime nếu chưa phải
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        
        # Lấy các phút căn chỉnh dựa trên timeframe
        minutes_map = {
            '5m': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
            '15m': [0, 15, 30, 45],
            '30m': [0, 30],
            '1h': [0],
            '4h': [0]
        }
        aligned_minutes = minutes_map[timeframe]
        
        # Hàm chuyển đổi timestamp thành timestamp căn chỉnh dựa trên timeframe
        def align_to_timeframe(timestamp):
            # Với 4h, căn chỉnh vào 1h, 5h, 9h, 13h, 17h, 21h
            if timeframe == '4h':
                hour = timestamp.hour
                aligned_hour = (hour // 4) * 4
                return timestamp.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
            
            # Với các TF khác, tìm phút gần nhất trong danh sách aligned_minutes
            minute = timestamp.minute
            closest_minute = min(aligned_minutes, key=lambda x: abs(x - minute))
            
            # Nếu đang làm tròn lên và cần tăng giờ
            if closest_minute < minute and closest_minute == 0:
                return (timestamp + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            
            return timestamp.replace(minute=closest_minute, second=0, microsecond=0)
        
        # Căn chỉnh index
        df_aligned = df.copy()
        df_aligned.index = df_aligned.index.map(align_to_timeframe)
        
        # Loại bỏ các timestamp trùng lặp (giữ bản ghi đầu tiên)
        df_aligned = df_aligned[~df_aligned.index.duplicated(keep='first')]
        
        return df_aligned
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5, 
                        columns: list = ['open', 'high', 'low', 'close'], action: str = 'mark') -> (pd.DataFrame, str):
        """
        Xử lý nâng cao cho outliers
        
        Parameters:
        - method: 'iqr' hoặc 'zscore'
        - threshold: ngưỡng cho phương pháp (1.5 cho IQR, 3.0 cho zscore)
        - columns: các cột cần kiểm tra
        - action: 'mark' (đánh dấu) hoặc 'remove' (loại bỏ) hoặc 'winsorize' (thay thế)
        """
        if df is None or df.empty:
            return df, "Không có dữ liệu để xử lý outlier"
        
        report_lines = []
        df_result = df.copy()
        outlier_mask = pd.Series(False, index=df.index)
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = z_scores > threshold
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            else:
                raise ValueError(f"Phương pháp phát hiện outlier '{method}' không được hỗ trợ")
            
            # Cập nhật mặt nạ outlier tổng hợp
            outlier_mask = outlier_mask | col_outliers
            
            # Xử lý từng cột theo action
            if action == 'winsorize':
                if method == 'zscore':
                    # Đối với zscore, thay thế bằng giá trị mean +/- threshold*std
                    mean, std = df[col].mean(), df[col].std()
                    df_result.loc[df[col] > mean + threshold * std, col] = mean + threshold * std
                    df_result.loc[df[col] < mean - threshold * std, col] = mean - threshold * std
                elif method == 'iqr':
                    # Đối với IQR, thay thế bằng upper/lower bound
                    df_result.loc[df[col] > upper_bound, col] = upper_bound
                    df_result.loc[df[col] < lower_bound, col] = lower_bound
                
                count = col_outliers.sum()
                if count > 0:
                    report_lines.append(f"Đã winsorize {count} giá trị ngoại lai trong cột {col}")
            elif action == 'mark':
                # Chỉ đánh dấu outlier vào báo cáo
                outlier_rows = df[col_outliers]
                if not outlier_rows.empty:
                    report_lines.append(f"Phát hiện {len(outlier_rows)} giá trị ngoại lai trong cột {col}:")
                    for idx, row in outlier_rows.iterrows():
                        report_lines.append(f"- {idx.strftime('%Y-%m-%d %H:%M')}: {row[col]:.2f}")
        
        # Đánh dấu is_outlier
        df_result['is_outlier'] = outlier_mask
        
        # Nếu action là remove, loại bỏ tất cả các hàng có outlier
        if action == 'remove':
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                df_result = df_result[~outlier_mask]
                report_lines.append(f"Đã loại bỏ {outlier_count} hàng có giá trị ngoại lai")
        
        return df_result, "\n".join(report_lines) if report_lines else "Không có giá trị ngoại lai"
    
    @staticmethod
    def fill_missing_by_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Điền giá trị thiếu thông minh dựa theo khung thời gian
        - Với timeframe intraday: Chỉ điền trong giờ giao dịch
        - Với D/W/M: Điền theo phương pháp phù hợp
        """
        if df is None or df.empty or not df.isna().any().any():
            return df
        
        df_filled = df.copy()
        
        # Kiểm tra có ít nhất một hàng không NA
        if df.dropna().empty:
            logger.warning("DataFrame chỉ chứa giá trị NA, không thể điền")
            return df
        
        # Xử lý điền giá trị khác nhau cho khung thời gian khác nhau
        if timeframe in ['5m', '15m', '30m', '1h', '4h']:
            # Với intraday, chi điền trong giờ giao dịch (9h-15h)
            trading_hours = df_filled.index.indexer_between_time('09:00', '15:00')
            trading_df = df_filled.iloc[trading_hours]
            
            # Điền giá trị cho dữ liệu trong giờ giao dịch
            if 'close' in df_filled.columns:
                trading_df['close'] = trading_df['close'].fillna(method='ffill')
                
            for col in ['open', 'high', 'low']:
                if col in trading_df.columns:
                    # Nếu open bị thiếu, dùng close của nến trước
                    if col == 'open':
                        trading_df['open'] = trading_df['open'].fillna(method='ffill')
                    else:
                        # high, low lấy từ close nếu bị thiếu
                        trading_df[col] = trading_df[col].fillna(trading_df['close'])
                    
            if 'volume' in trading_df.columns:
                trading_df['volume'] = trading_df['volume'].fillna(0)
                
            # Áp dụng lại vào df_filled
            df_filled.iloc[trading_hours] = trading_df
        else:
            # Với D/W/M, dùng phương pháp của DataNormalizer
            if 'close' in df_filled.columns:
                df_filled['close'] = df_filled['close'].fillna(method='ffill')
            
            for col in ['open', 'high', 'low']:
                if col in df_filled.columns:
                    if col == 'open' and df_filled['open'].isna().any():
                        # Nếu open bị thiếu, dùng close của ngày trước
                        df_filled['open'] = df_filled['open'].fillna(method='ffill')
                    elif 'close' in df_filled.columns:
                        df_filled[col] = df_filled[col].fillna(df_filled['close'])
            
            if 'volume' in df_filled.columns:
                df_filled['volume'] = df_filled['volume'].fillna(0)
        
        return df_filled

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
    version = Column(String, nullable=True)
    params = Column(Text, nullable=True)  # JSON string of model parameters
    timeframe = Column(String, nullable=True)  # '1D', '1W', '1M', '1h', etc.

class SchemaVersion(Base):
    __tablename__ = 'schema_version'
    id = Column(Integer, primary_key=True)
    version = Column(String, nullable=False)
    applied_at = Column(DateTime, default=datetime.now)
    description = Column(String, nullable=True)

class ModelPerformanceHistory(Base):
    __tablename__ = 'model_performance_history'
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('trained_models.id'), nullable=False)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    performance = Column(Float, nullable=False)
    prediction_date = Column(DateTime, nullable=False)
    actual_result = Column(Float, nullable=False)  # Actual price or value
    predicted_result = Column(Float, nullable=False)  # Predicted price or value
    error = Column(Float, nullable=False)  # Prediction error
    version = Column(String, nullable=True)
    timeframe = Column(String, nullable=True)
    recorded_at = Column(DateTime, default=datetime.now)

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
    async def store_trained_model(self, symbol: str, model_type: str, model, performance: float = None,
                                 version: str = "1.0", params: dict = None, timeframe: str = '1D'):
        try:
            model_blob = pickle.dumps(model)
            params_json = json.dumps(params) if params else None
            
            async with self.Session() as session:
                result = await session.execute(select(TrainedModel).filter_by(symbol=symbol, model_type=model_type, timeframe=timeframe))
                existing = result.scalar_one_or_none()
                if existing:
                    existing.model_blob = model_blob
                    existing.created_at = datetime.now()
                    existing.performance = performance
                    existing.version = version
                    existing.params = params_json
                    existing.timeframe = timeframe
                    model_id = existing.id
                else:
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
                    await session.flush()
                    model_id = new_model.id
                
                await session.commit()
            logger.info(f"Lưu mô hình {model_type} cho {symbol} ({timeframe}) thành công với hiệu suất: {performance}, phiên bản: {version}")
            return model_id
        except Exception as e:
            logger.error(f"Lỗi lưu mô hình {model_type} cho {symbol}: {str(e)}")
            return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_trained_model(self, symbol: str, model_type: str, timeframe: str = '1D'):
        try:
            async with self.Session() as session:
                result = await session.execute(
                    select(TrainedModel)
                    .filter_by(symbol=symbol, model_type=model_type, timeframe=timeframe)
                    .order_by(TrainedModel.created_at.desc())
                )
                model_record = result.scalar_one_or_none()
                if model_record:
                    model = pickle.loads(model_record.model_blob)
                    params = json.loads(model_record.params) if model_record.params else None
                    return model, model_record.performance, model_record.version, params, model_record.id
            return None, None, None, None, None
        except Exception as e:
            logger.error(f"Lỗi tải mô hình {model_type} cho {symbol}: {str(e)}")
            return None, None, None, None, None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def record_model_performance(self, model_id: int, symbol: str, model_type: str, 
                                     performance: float, prediction_date: datetime,
                                     actual_result: float, predicted_result: float,
                                     version: str = None, timeframe: str = '1D'):
        try:
            error = abs(actual_result - predicted_result)
            async with self.Session() as session:
                new_history = ModelPerformanceHistory(
                    model_id=model_id,
                    symbol=symbol,
                    model_type=model_type,
                    performance=performance,
                    prediction_date=prediction_date,
                    actual_result=actual_result,
                    predicted_result=predicted_result,
                    error=error,
                    version=version,
                    timeframe=timeframe
                )
                session.add(new_history)
                await session.commit()
            logger.info(f"Ghi nhận hiệu suất mô hình {model_type} cho {symbol}: {performance}")
            return True
        except Exception as e:
            logger.error(f"Lỗi ghi nhận hiệu suất mô hình {model_type} cho {symbol}: {str(e)}")
            return False
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get_model_performance_history(self, symbol: str, model_type: str, timeframe: str = '1D', limit: int = 10):
        try:
            async with self.Session() as session:
                result = await session.execute(
                    select(ModelPerformanceHistory)
                    .filter_by(symbol=symbol, model_type=model_type, timeframe=timeframe)
                    .order_by(ModelPerformanceHistory.prediction_date.desc())
                    .limit(limit)
                )
                history = result.scalars().all()
                return [
                    {
                        "id": h.id,
                        "model_id": h.model_id,
                        "symbol": h.symbol,
                        "model_type": h.model_type,
                        "performance": h.performance,
                        "prediction_date": h.prediction_date.isoformat(),
                        "actual_result": h.actual_result,
                        "predicted_result": h.predicted_result,
                        "error": h.error,
                        "version": h.version,
                        "timeframe": h.timeframe
                    }
                    for h in history
                ]
        except Exception as e:
            logger.error(f"Lỗi lấy lịch sử hiệu suất mô hình cho {symbol}: {str(e)}")
            return []
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def save_schema_version(self, version: str, description: str = None):
        try:
            async with self.Session() as session:
                new_version = SchemaVersion(
                    version=version, 
                    description=description,
                    applied_at=datetime.now()
                )
                session.add(new_version)
                await session.commit()
            logger.info(f"Cập nhật phiên bản schema: {version}")
            return True
        except Exception as e:
            logger.error(f"Lỗi cập nhật phiên bản schema: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get_current_schema_version(self):
        try:
            async with self.Session() as session:
                result = await session.execute(
                    select(SchemaVersion).order_by(SchemaVersion.applied_at.desc()).limit(1)
                )
                version = result.scalar_one_or_none()
                return version.version if version else None
        except Exception as e:
            logger.error(f"Lỗi lấy phiên bản schema hiện tại: {str(e)}")
            return None

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
    """
    Lọc lại ngày giao dịch cho data khung ngày (1D)
    Chỉ loại bỏ cuối tuần và ngày lễ cho khung ngày
    """
    if df.empty:
        return df
    
    # Chỉ lọc ngày giao dịch, giữ nguyên data trên khung tuần và tháng
    df = df[df.index.weekday < 5]  # Loại bỏ T7, CN
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
        # Verify and standardize timeframe
        is_valid, message = DataValidator.validate_timeframe(timeframe)
        if not is_valid:
            raise ValueError(message)
            
        # Mapping timeframes to standard format
        timeframe_map = {
            '1d': '1D', 'd': '1D', 'daily': '1D',
            '1w': '1W', 'w': '1W', 'weekly': '1W', 
            '1mo': '1M', 'mo': '1M', 'monthly': '1M',
            '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', 'h': '1h', 'hourly': '1h',
            '4h': '4h'
        }
        timeframe = timeframe_map.get(timeframe.lower(), timeframe)
        
        # Set cache expiration times based on timeframe
        if timeframe in ['5m', '15m', '30m', '1h', '4h']:
            expire = CACHE_EXPIRE_SHORT // 2  # Shorter cache for intraday
        elif timeframe == '1D':
            expire = CACHE_EXPIRE_SHORT
        elif timeframe == '1W':
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
                    
                    # Tính toán start_date phù hợp dựa vào timeframe
                    if timeframe in ['5m', '15m', '30m', '1h']:
                        # For intraday, we need to adjust the period to get enough data
                        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    elif timeframe == '4h':
                        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                    elif timeframe == '1W':
                        # Cho weekly data, lấy dữ liệu gấp 3 lần số nến cần thiết để tránh thiếu dữ liệu
                        start_date = (datetime.now() - timedelta(weeks=num_candles * 3)).strftime('%Y-%m-%d')
                    elif timeframe == '1M':
                        # Cho monthly data, lấy dữ liệu gấp 3 lần số nến cần thiết để tránh thiếu dữ liệu
                        start_date = (datetime.now() - timedelta(days=num_candles * 90)).strftime('%Y-%m-%d')
                    else:
                        # Daily timeframe
                        start_date = (datetime.now() - timedelta(days=num_candles * 3)).strftime('%Y-%m-%d')
                    
                    # Fetch data with appropriate interval - đảm bảo interval truyền vào chính xác
                    # TCBS API hỗ trợ các interval: 1D, 1W, 1M
                    tcbs_interval = timeframe
                    if timeframe in ['5m', '15m', '30m', '1h', '4h']:
                        # Intraday API có thể khác, kiểm tra tài liệu API của TCBS
                        tcbs_interval = timeframe
                    
                    df = stock.quote.history(start=start_date, end=end_date, interval=tcbs_interval)
                    if df is None or df.empty or len(df) < 20:
                        raise ValueError(f"Không đủ dữ liệu cho {'chỉ số' if is_index(symbol) else 'mã'} {symbol} (timeframe: {timeframe})")
                    
                    # Chuẩn hóa dữ liệu
                    df = df.rename(columns={'time': 'date', 'open': 'open', 'high': 'high',
                                            'low': 'low', 'close': 'close', 'volume': 'volume'})
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    df = DataNormalizer.normalize_dataframe(df)
                    
                    # Align timestamps for intraday data
                    if timeframe in ['5m', '15m', '30m', '1h', '4h']:
                        df = DataValidator.align_timestamp(df, timeframe)
                    
                    # Đảm bảo múi giờ chính xác
                    df.index = df.index.tz_localize('Asia/Bangkok')
                    
                    # Xác thực dữ liệu
                    is_valid, error_msg = DataNormalizer.validate_data(df)
                    if not is_valid:
                        logger.warning(f"Dữ liệu không hợp lệ cho {symbol}: {error_msg}")
                    
                    # Điền giá trị thiếu theo timeframe
                    df = DataValidator.fill_missing_by_timeframe(df, timeframe)
                    
                    if len(df) < 200 and timeframe not in ['5m', '15m', '30m', '1h', '4h']:
                        logger.warning(f"Dữ liệu cho {symbol} dưới 200 nến, SMA200 có thể không chính xác")
                    
                    return df.tail(num_candles + 1)
                    
                df = await run_in_thread(fetch_vnstock)
            elif self.source == 'yahoo':
                # Map timeframes to Yahoo Finance intervals
                period_map = {
                    '1D': 'd', '1W': 'wk', '1M': 'mo',
                    '5m': '5m', '15m': '15m', '30m': '30m', 
                    '1h': '60m', '4h': '4h'
                }
                yahoo_interval = period_map.get(timeframe, 'd')
                
                df = await self._download_yahoo_data(symbol, num_candles + 1, yahoo_interval, timeframe)
                if df is None or df.empty or len(df) < 20:
                    raise ValueError(f"Không đủ dữ liệu cho {symbol} từ Yahoo Finance (timeframe: {timeframe})")
                
                # Chuẩn hóa dữ liệu
                df = DataNormalizer.normalize_dataframe(df)
                
                # Align timestamps for intraday data
                if timeframe in ['5m', '15m', '30m', '1h', '4h']:
                    df = DataValidator.align_timestamp(df, timeframe)
                
                # Fill missing values based on timeframe
                df = DataValidator.fill_missing_by_timeframe(df, timeframe)
                df.index = df.index.tz_localize('Asia/Bangkok')
                
                # Xác thực dữ liệu
                is_valid, error_msg = DataNormalizer.validate_data(df)
                if not is_valid:
                    logger.warning(f"Dữ liệu không hợp lệ cho {symbol}: {error_msg}")
                
                if len(df) < 200 and timeframe not in ['5m', '15m', '30m', '1h', '4h']:
                    logger.warning(f"Dữ liệu cho {symbol} dưới 200 nến, SMA200 có thể không chính xác")
            else:
                raise ValueError("Nguồn dữ liệu không hợp lệ")

            # Filter trading days for daily timeframe only
            if timeframe == '1D':
                df = filter_trading_days(df)
                
            # Detect and handle outliers
            df, outlier_report = DataValidator.handle_outliers(
                df, method='iqr', threshold=1.5, 
                columns=['open', 'high', 'low', 'close'], 
                action='mark'
            )
            
            await redis_manager.set(cache_key, df, expire=expire)
            return df, outlier_report
        except Exception as e:
            logger.error(f"Lỗi tải dữ liệu cho {symbol} (timeframe: {timeframe}): {str(e)}")
            raise ValueError(f"Không thể tải dữ liệu: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
    async def _download_yahoo_data(self, symbol: str, num_candles: int, period: str, timeframe: str) -> pd.DataFrame:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                start_ts = int((datetime.now() - timedelta(days=num_candles * 3)).timestamp())
                end_ts = int(datetime.now().timestamp())
                url = (f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}"
                       f"?period1={start_ts}&period2={end_ts}&interval={period}&events=history")
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
        Chuẩn bị dữ liệu đầy đủ cho một mã chứng khoán bao gồm:
        - Dữ liệu đa khung thời gian 
        - Chỉ báo kỹ thuật
        - Phát hiện outlier
        - Dữ liệu cơ bản (nếu có)
        """
        if timeframes is None:
            timeframes = ['1D', '1W', '1M']
        
        # Validate timeframes
        valid_timeframes = []
        for tf in timeframes:
            is_valid, _ = DataValidator.validate_timeframe(tf)
            if is_valid:
                valid_timeframes.append(tf)
            else:
                logger.warning(f"Khung thời gian không hợp lệ: {tf}")
        
        if not valid_timeframes:
            raise ValueError(f"Không có khung thời gian hợp lệ trong: {timeframes}")
        
        # Validate symbol
        is_valid, message = DataValidator.validate_ticker(symbol)
        if not is_valid:
            raise ValueError(message)
        
        result = {
            'symbol': symbol,
            'dataframes': {},
            'indicators': {},
            'outlier_reports': {},
            'fundamental_data': {},
            'errors': []
        }
        
        # Tải dữ liệu đa khung thời gian
        for tf in valid_timeframes:
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

async def train_models_for_symbol(symbol: str, timeframes = None):
    try:
        if timeframes is None:
            timeframes = ['1D']  # Default to daily data
            
        logger.info(f"Bắt đầu auto training cho mã: {symbol} với timeframes: {timeframes}")
        loader = DataLoader()
        tech_analyzer = TechnicalAnalyzer()
        
        for timeframe in timeframes:
            logger.info(f"Training mô hình cho {symbol} với timeframe {timeframe}")
            # Validate timeframe
            is_valid, msg = DataValidator.validate_timeframe(timeframe)
            if not is_valid:
                logger.warning(f"Bỏ qua timeframe không hợp lệ {timeframe}: {msg}")
                continue
                
            # Load data for timeframe
            df, _ = await loader.load_data(symbol, timeframe, 500)
            df = tech_analyzer.calculate_indicators(df)
            
            # Define features and model version
            features = ['sma20', 'sma50', 'sma200', 'rsi', 'macd', 'signal',
                        'bb_high', 'bb_low', 'ichimoku_a', 'ichimoku_b', 'vwap', 'mfi']
            version = "2.0"  # Can be dynamic based on model configuration
            
            # Define model parameters
            prophet_params = {
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10,
                "seasonality_mode": "multiplicative",
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": True,
                "timeframe": timeframe
            }
            
            xgb_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 100,
                "objective": "binary:logistic",
                "booster": "gbtree",
                "features": features,
                "timeframe": timeframe
            }
            
            # Train models asynchronously
            task_prophet = asyncio.to_thread(train_prophet_model, df)
            task_xgb = asyncio.to_thread(train_xgboost_model, df, features)
            
            (prophet_model, prophet_perf), (xgb_model, xgb_perf) = await asyncio.gather(task_prophet, task_xgb)
            
            # Store models with parameters
            prophet_id = await model_db_manager.store_trained_model(
                symbol, 'prophet', prophet_model, prophet_perf, 
                version=version, params=prophet_params, timeframe=timeframe
            )
            
            xgb_id = await model_db_manager.store_trained_model(
                symbol, 'xgboost', xgb_model, xgb_perf,
                version=version, params=xgb_params, timeframe=timeframe
            )
            
            logger.info(f"Training hoàn tất cho {symbol} với timeframe {timeframe}. Prophet: {prophet_perf:.4f}, XGBoost: {xgb_perf:.4f}")
            
            # Record initial performance
            now = datetime.now()
            if prophet_id:
                await model_db_manager.record_model_performance(
                    model_id=prophet_id,
                    symbol=symbol,
                    model_type='prophet',
                    performance=prophet_perf,
                    prediction_date=now,
                    actual_result=df['close'].iloc[-1],
                    predicted_result=df['close'].iloc[-1],  # Placeholder
                    version=version,
                    timeframe=timeframe
                )
                
            if xgb_id:
                await model_db_manager.record_model_performance(
                    model_id=xgb_id,
                    symbol=symbol,
                    model_type='xgboost',
                    performance=xgb_perf,
                    prediction_date=now,
                    actual_result=df['close'].iloc[-1],
                    predicted_result=df['close'].iloc[-1],  # Placeholder
                    version=version,
                    timeframe=timeframe
                )
            
        logger.info(f"Auto training cho {symbol} hoàn tất cho tất cả timeframes")
    except Exception as e:
        logger.error(f"Lỗi auto training cho {symbol}: {str(e)}")
        
async def auto_train_models():
    try:
        symbols = await get_training_symbols()
        if not symbols:
            logger.info("Không có mã nào trong ReportHistory, bỏ qua auto training.")
            return
            
        timeframes = ['1D', '1W']  # Default timeframes to train - can be expanded to include intraday
        tasks = [train_models_for_symbol(symbol, timeframes) for symbol in symbols]
        await asyncio.gather(*tasks)
        
        # Record schema version after successful training
        await model_db_manager.save_schema_version(
            "2.0", 
            description="Enhanced model training with multiple timeframes and parameters"
        )
        
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

    async def generate_report(self, dfs: dict, symbol: str, fundamental_data: dict, outlier_reports: dict, primary_timeframe: str = '1D') -> str:
        try:
            tech_analyzer = TechnicalAnalyzer()
            indicators = tech_analyzer.calculate_multi_timeframe_indicators(dfs)
            news = await get_news(symbol=symbol)
            news_text = "\n".join([f"📰 **{n['title']}**\n🔗 {n['link']}\n📝 {n['summary']}" for n in news])
            
            # Sử dụng timeframe chính cho phân tích
            df_primary = dfs.get(primary_timeframe)
            if df_primary is None:
                if '1D' in dfs:
                    df_primary = dfs.get('1D')
                    logger.warning(f"Không tìm thấy khung {primary_timeframe} cho {symbol}, sử dụng 1D thay thế")
                else:
                    # Lấy timeframe đầu tiên có sẵn
                    primary_timeframe = list(dfs.keys())[0]
                    df_primary = dfs.get(primary_timeframe)
                    logger.warning(f"Không tìm thấy khung 1D cho {symbol}, sử dụng {primary_timeframe} thay thế")
            
            close_today = df_primary['close'].iloc[-1]
            close_yesterday = df_primary['close'].iloc[-2]
            price_action = self.analyze_price_action(df_primary)
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
                "candlestick_data": df_primary.tail(50).to_dict(orient="records"),
                "technical_indicators": indicators.get(primary_timeframe, indicators.get('1D', {}))
            }
            openrouter_result = await self.analyze_with_openrouter(technical_data)
            support_levels = openrouter_result.get('support_levels', [])
            resistance_levels = openrouter_result.get('resistance_levels', [])
            patterns = openrouter_result.get('patterns', [])

            forecast, prophet_model = forecast_with_prophet(df_primary, periods=7)
            prophet_perf = evaluate_prophet_performance(df_primary, forecast)
            future_forecast = forecast[forecast['ds'] > df_primary.index[-1].tz_localize(None)]
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
            xgb_signal, xgb_perf = predict_xgboost_signal(df_primary.copy(), features)
            if isinstance(xgb_signal, int):
                xgb_text = "Tăng" if xgb_signal == 1 else "Giảm"
            else:
                xgb_text = xgb_signal
            
            if is_index(symbol):
                xgb_summary = f"**XGBoost dự đoán xu hướng tiếp theo** (Hiệu suất: {xgb_perf:.2f}): {xgb_text}\n"
            else:
                xgb_summary = f"**XGBoost dự đoán tín hiệu giao dịch** (Hiệu suất: {xgb_perf:.2f}): {xgb_text}\n"

            outlier_text = "\n".join([f"**{tf}**: {report}" for tf, report in outlier_reports.items() if tf in dfs])

            # Tự tính toán thêm mức hỗ trợ/kháng cự để đối chiếu
            calculated_levels = self.calculate_support_resistance_levels(df_primary)
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
- Khung thời gian: {primary_timeframe}
- Giá hôm qua: {close_yesterday:.2f}
- Giá hôm nay: {close_today:.2f} ({((close_today-close_yesterday)/close_yesterday*100):.2f}%)

**Diễn biến chỉ số:**
{price_action}

**Lịch sử dự đoán:**
{past_report}

**Chất lượng dữ liệu:**
{outlier_text}
```
            else:
                # Phân tích cho cổ phiếu
                fundamental_report = f"📊 **Thông tin cơ bản:**\n"
                for key, value in fundamental_data.items():
                    if isinstance(value, (int, float)):
                        fundamental_report += f"- {key}: {value:,.2f}\n"
                    else:
                        fundamental_report += f"- {key}: {value}\n"
                
                prompt = f"""
Bạn là chuyên gia phân tích kỹ thuật, phân tích thị trường chứng khoán Việt Nam với 30 năm kinh nghiệm. 
Hãy viết báo cáo chi tiết cho CỔ PHIẾU {symbol}:

**Thông tin cơ bản:**
- Ngày: {datetime.now().strftime('%d/%m/%Y')}
- Khung thời gian: {primary_timeframe}
- Giá hôm qua: {close_yesterday:.2f}
- Giá hôm nay: {close_today:.2f} ({((close_today-close_yesterday)/close_yesterday*100):.2f}%)

**Diễn biến giá:**
{price_action}

**Lịch sử dự đoán:**
{past_report}

**Chất lượng dữ liệu:**
{outlier_text}
```

            prompt += f"""
**Dự báo giá:**
{forecast_summary}

**Dự đoán XGBoost:**
{xgb_summary}

**Mức hỗ trợ/kháng cự (OpenRouter):**
- Hỗ trợ: {", ".join([f"{level:.2f}" for level in support_levels])}
- Kháng cự: {", ".join([f"{level:.2f}" for level in resistance_levels])}

**Mức hỗ trợ/kháng cự (Tự tính):**
- Hỗ trợ: {calc_support_str}
- Kháng cự: {calc_resistance_str}

**Mẫu hình nến:**
{", ".join([f"{p['name']} ({p['description']})" for p in patterns])}

**Tin tức:**
{news_text}

**Thông tin cơ bản:**
{fundamental_report}

**Chỉ báo kỹ thuật:**
{json.dumps(indicators.get(primary_timeframe, indicators.get('1D', {})), ensure_ascii=False, indent=2)}

**Kết luận:**
"""

            response = await self.generate_content(prompt)
            report = response.text
            await self.save_report_history(symbol, report, close_today, close_yesterday)
            return report
        except Exception as e:
            logger.error(f"Lỗi trong generate_report: {str(e)}")
            return f"❌ Lỗi: {str(e)}"

# ---------- MAIN & DEPLOY ----------
async def main():
    # Khởi tạo bot application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("chart", chart_command))
    app.add_handler(CommandHandler("getid", get_id))
    app.add_handler(CommandHandler("approve", approve_user))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, notify_admin_new_user))
    logger.info("🤖 Bot khởi động!")

    # Khởi động bot
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await app.updater.idle()

if __name__ == "__main__":
    asyncio.run(main())