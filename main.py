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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps
import time as time_module  # Đổi tên để tránh xung đột

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
import signal  # Thêm import signal

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

# Tối ưu thread pool và process pool cho xử lý bất đồng bộ
thread_executor = ThreadPoolExecutor(max_workers=10)  # Cho I/O bound tasks
process_executor = ProcessPoolExecutor(max_workers=4)  # Cho CPU bound tasks

# Decorator để đo thời gian thực thi của hàm
def measure_execution_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time_module.time()
        result = await func(*args, **kwargs)
        end_time = time_module.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

async def run_in_thread(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(thread_executor, lambda: func(*args, **kwargs))

# Hàm thực thi trong process pool cho tác vụ nặng về CPU
async def run_in_process(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(process_executor, lambda: func(*args, **kwargs))

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
            self.redis_client = redis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=False,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30
            )
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
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def delete(self, key):
        """Xóa một key khỏi cache"""
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis delete: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def exists(self, key):
        """Kiểm tra một key có tồn tại trong cache không"""
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Lỗi Redis exists: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def flush_all(self):
        """Xóa toàn bộ cache"""
        try:
            await self.redis_client.flushall()
            logger.info("Đã xóa toàn bộ cache Redis")
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis flush_all: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def pattern_delete(self, pattern):
        """Xóa tất cả các key theo pattern"""
        try:
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
                logger.info(f"Đã xóa {len(keys)} key theo pattern: {pattern}")
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis pattern_delete: {str(e)}")
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
                    
                    # Calculate appropriate start date based on timeframe
                    if timeframe in ['5m', '15m', '30m', '1h']:
                        # For intraday, we need to adjust the period to get enough data
                        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    elif timeframe == '4h':
                        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                    elif timeframe == '1D':
                        # Daily timeframe
                        start_date = (datetime.now() - timedelta(days=(num_candles + 1) * 3)).strftime('%Y-%m-%d')
                        df = stock.quote.history(start=start_date, end=end_date, interval=timeframe)
                    elif timeframe == '1W':
                        # Weekly timeframe - fetch more data and resample properly
                        start_date = (datetime.now() - timedelta(days=(num_candles + 1) * 7 * 3)).strftime('%Y-%m-%d')
                        # Fetch daily data
                        df_daily = stock.quote.history(start=start_date, end=end_date, interval='1D')
                        if df_daily is None or df_daily.empty:
                            raise ValueError(f"Không đủ dữ liệu hàng ngày cho {'chỉ số' if is_index(symbol) else 'mã'} {symbol}")
                        
                        # Chuẩn hóa dữ liệu hàng ngày
                        df_daily = df_daily.rename(columns={'time': 'date', 'open': 'open', 'high': 'high',
                                                'low': 'low', 'close': 'close', 'volume': 'volume'})
                        df_daily['date'] = pd.to_datetime(df_daily['date'])
                        df_daily = df_daily.set_index('date')
                        
                        # Resample to weekly data (Monday to Sunday)
                        df = df_daily.resample('W-MON').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        })
                        # Remove any future weeks
                        df = df[df.index <= datetime.now()]
                    elif timeframe == '1M':
                        # Monthly timeframe - fetch more data and resample properly
                        start_date = (datetime.now() - timedelta(days=(num_candles + 1) * 31 * 3)).strftime('%Y-%m-%d')
                        # Fetch daily data
                        df_daily = stock.quote.history(start=start_date, end=end_date, interval='1D')
                        if df_daily is None or df_daily.empty:
                            raise ValueError(f"Không đủ dữ liệu hàng ngày cho {'chỉ số' if is_index(symbol) else 'mã'} {symbol}")
                        
                        # Chuẩn hóa dữ liệu hàng ngày
                        df_daily = df_daily.rename(columns={'time': 'date', 'open': 'open', 'high': 'high',
                                                'low': 'low', 'close': 'close', 'volume': 'volume'})
                        df_daily['date'] = pd.to_datetime(df_daily['date'])
                        df_daily = df_daily.set_index('date')
                        
                        # Resample to monthly data (calendar month)
                        df = df_daily.resample('MS').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        })
                        # Remove any future months
                        df = df[df.index <= datetime.now()]
                    else:
                        # Get enough data based on the number of candles needed
                        start_date = (datetime.now() - timedelta(days=(num_candles + 1) * 3)).strftime('%Y-%m-%d')
                        df = stock.quote.history(start=start_date, end=end_date, interval=timeframe)
                    
                    if df is None or df.empty or len(df) < 20:
                        raise ValueError(f"Không đủ dữ liệu cho {'chỉ số' if is_index(symbol) else 'mã'} {symbol} (timeframe: {timeframe})")
                    
                    # Chuẩn hóa dữ liệu
                    if 'time' in df.columns:
                        df = df.rename(columns={'time': 'date'})
                    if 'date' not in df.index.names:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                    
                    df = DataNormalizer.normalize_dataframe(df)
                    
                    # Align timestamps for intraday data
                    if timeframe in ['5m', '15m', '30m', '1h', '4h']:
                        df = DataValidator.align_timestamp(df, timeframe)
                    
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
                
                if timeframe in ['1W', '1M']:
                    # For weekly and monthly, we'll use the correct Yahoo Finance interval
                    # but we need to adjust the period to get enough candles
                    if timeframe == '1W':
                        # For weekly timeframe, get enough days for the requested candles
                        period_days = num_candles * 7 * 3  # 3x safety factor
                    else:  # 1M
                        # For monthly timeframe, get enough days
                        period_days = num_candles * 31 * 3  # 3x safety factor
                    
                    df = await self._download_yahoo_data(symbol, period_days, yahoo_interval, timeframe)
                else:
                    # For other timeframes, proceed normally
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

            # Filter trading days for daily and higher timeframes
            if timeframe not in ['5m', '15m', '30m', '1h', '4h']:
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
                    
                    # Đảm bảo đúng số nến cho mỗi timeframe
                    if timeframe == '1W':
                        # Chỉ lấy số nến tuần theo yêu cầu
                        return df.tail(num_candles // 7 + 5)  # +5 để đảm bảo đủ dữ liệu
                    elif timeframe == '1M':
                        # Chỉ lấy số nến tháng theo yêu cầu
                        return df.tail(num_candles // 31 + 5)  # +5 để đảm bảo đủ dữ liệu
                    else:
                        # Các timeframe khác
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
                # Điều chỉnh số nến cần lấy theo timeframe
                adjusted_candles = num_candles
                if tf == '1W':
                    # Để có đủ dữ liệu cho phân tích tuần, cần nhiều nến hơn
                    adjusted_candles = min(num_candles * 2, 500)  # Tối đa 500 nến
                elif tf == '1M':
                    # Để có đủ dữ liệu cho phân tích tháng, cần nhiều nến hơn
                    adjusted_candles = min(num_candles * 3, 500)  # Tối đa 500 nến

                df, outlier_report = await self.data_loader.load_data(symbol, tf, adjusted_candles)
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

                    # Thêm thông tin xu hướng cho từng khung thời gian
                    if tf == '1D':
                        prompt += f"- Xu hướng Ngắn hạn: {DataPipeline.extract_patterns({'1D': dfs.get('1D', None)}).get('1D_trend', 'N/A')}\n"
                    elif tf == '1W':
                        prompt += f"- Xu hướng Trung hạn: {DataPipeline.extract_patterns({'1W': dfs.get('1W', None)}).get('1W_trend', 'N/A')}\n"
                    elif tf == '1M':
                        prompt += f"- Xu hướng Dài hạn: {DataPipeline.extract_patterns({'1M': dfs.get('1M', None)}).get('1M_trend', 'N/A')}\n"
                    
                    # Thêm thông tin nến cuối với nến trước đó
                    if dfs.get(tf) is not None and not dfs[tf].empty and len(dfs[tf]) >= 2:
                        last_candle = dfs[tf].iloc[-1]
                        prev_candle = dfs[tf].iloc[-2]
                        change = last_candle['close'] - prev_candle['close']
                        change_pct = (change / prev_candle['close']) * 100
                        
                        if tf == '1D':
                            prompt += f"- Thay đổi Ngày: {change:.2f} ({change_pct:.2f}%)\n"
                        elif tf == '1W':
                            prompt += f"- Thay đổi Tuần: {change:.2f} ({change_pct:.2f}%)\n"
                        elif tf == '1M':
                            prompt += f"- Thay đổi Tháng: {change:.2f} ({change_pct:.2f}%)\n"

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
2. Phân tích đa khung thời gian, phải tách biệt rõ ràng:
   - Xu hướng ngắn hạn (1D): phân tích dựa trên dữ liệu ngày.
   - Xu hướng trung hạn (1W): phân tích dựa trên dữ liệu tuần.
   - Xu hướng dài hạn (1M): phân tích dựa trên dữ liệu tháng.
   * QUAN TRỌNG: Xu hướng mỗi khung thời gian phải được phân tích riêng biệt, không trộn lẫn.
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
2. Phân tích đa khung thời gian, phải tách biệt rõ ràng:
   - Xu hướng ngắn hạn (1D): phân tích dựa trên dữ liệu ngày.
   - Xu hướng trung hạn (1W): phân tích dựa trên dữ liệu tuần.
   - Xu hướng dài hạn (1M): phân tích dựa trên dữ liệu tháng.
   * QUAN TRỌNG: Xu hướng mỗi khung thời gian phải được phân tích riêng biệt, không trộn lẫn.
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
    
    # Gửi tin nhắn loading ngay khi bắt đầu
    waiting_msg = await update.message.reply_text("⏳ Đang xử lý yêu cầu...")
    
    try:
        args = context.args
        if not args:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=waiting_msg.message_id,
                text="❌ Vui lòng nhập mã chứng khoán (e.g., VNINDEX, SSI)."
            )
            return
            
        symbol = args[0].upper()
        
        # Kiểm tra số lượng nến
        try:
            num_candles = int(args[1]) if len(args) > 1 else DEFAULT_CANDLES
            if num_candles < 20:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=waiting_msg.message_id,
                    text="❌ Số nến phải lớn hơn hoặc bằng 20 để tính toán chỉ báo!"
                )
                return
            if num_candles > 500:
                await context.bot.edit_message_text(
                    chat_id=update.effective_chat.id,
                    message_id=waiting_msg.message_id,
                    text="❌ Tối đa 500 nến!"
                )
                return
        except ValueError:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=waiting_msg.message_id,
                text="❌ Số lượng nến không hợp lệ. Vui lòng nhập số nguyên."
            )
            return
        
        # Sử dụng pipeline chuẩn hóa
        data_pipeline = DataPipeline()
        ai_analyzer = AIAnalyzer()
        
        # Cập nhật tin nhắn chờ
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=waiting_msg.message_id,
            text=f"⏳ Đang chuẩn bị dữ liệu cho {symbol}..."
        )
        
        # Chuẩn bị dữ liệu với pipeline
        start_time = time_module.time()
        pipeline_result = await data_pipeline.prepare_symbol_data(symbol, timeframes=['1D', '1W', '1M'], num_candles=num_candles)
        data_time = time_module.time() - start_time
        
        if pipeline_result['errors']:
            error_message = f"⚠️ Một số lỗi xảy ra trong quá trình chuẩn bị dữ liệu:\n"
            error_message += "\n".join(pipeline_result['errors'])
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=waiting_msg.message_id,
                text=error_message
            )
        
        if not pipeline_result['dataframes']:
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=waiting_msg.message_id,
                text=f"❌ Không thể tải dữ liệu cho {symbol}. Vui lòng kiểm tra mã chứng khoán."
            )
            return
        
        # Cập nhật tin nhắn chờ
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=waiting_msg.message_id,
            text=f"⏳ Đang phân tích {symbol} với AI..."
        )
        
        # Tạo báo cáo với AI
        start_time = time_module.time()
        report = await ai_analyzer.generate_report(
            pipeline_result['dataframes'], 
            symbol, 
            pipeline_result['fundamental_data'], 
            pipeline_result['outlier_reports']
        )
        ai_time = time_module.time() - start_time
        
        # Lưu báo cáo vào cache
        await redis_manager.set(f"report_{symbol}_{num_candles}", report, expire=CACHE_EXPIRE_SHORT)

        formatted_report = f"<b>📈 Báo cáo phân tích cho {symbol}</b>\n\n"
        formatted_report += f"<pre>{html.escape(report)}</pre>"
        
        # Thông tin hiệu suất (chỉ hiển thị trong môi trường debug)
        performance_info = f"\n<i>Thời gian tải dữ liệu: {data_time:.2f}s | Thời gian phân tích: {ai_time:.2f}s</i>"
        
        # Gửi báo cáo cuối cùng
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=waiting_msg.message_id,
            text=formatted_report + performance_info,
            parse_mode='HTML'
        )
        
    except ValueError as e:
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=waiting_msg.message_id,
            text=f"❌ Lỗi: {str(e)}"
        )
    except Exception as e:
        # Lưu lại thông tin chi tiết về lỗi để debug
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Lỗi trong analyze_command: {str(e)}\n{error_traceback}")
        
        # Phân loại và trả về thông báo lỗi thân thiện với người dùng
        error_message = f"❌ Lỗi không xác định"
        
        if "No module named" in str(e):
            error_message = "❌ Thiếu module cần thiết. Vui lòng liên hệ admin."
        elif "HTTP" in str(e) and "429" in str(e):
            error_message = "❌ Đã vượt quá giới hạn API. Vui lòng thử lại sau ít phút."
        elif "timeout" in str(e).lower():
            error_message = "❌ Yêu cầu bị timeout. Vui lòng thử lại sau."
        elif "connection" in str(e).lower():
            error_message = "❌ Lỗi kết nối. Hãy kiểm tra kết nối mạng và thử lại."
        else:
            error_message = f"❌ Lỗi không xác định: {str(e)}"
        
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=waiting_msg.message_id,
            text=error_message
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
    """Hàm chính của bot"""
    
    logger.info("Khởi động bot...")
    
    # Khởi tạo database
    await init_db()
    
    # Kiểm tra và khởi tạo phiên bản schema
    current_version = await model_db_manager.get_current_schema_version()
    if not current_version:
        logger.info("Khởi tạo phiên bản schema mới")
        await model_db_manager.save_schema_version(
            "2.0", 
            description="Schema mới với hỗ trợ đa timeframe và lưu tham số mô hình"
        )
    else:
        logger.info(f"Phiên bản schema hiện tại: {current_version}")

    # Bật auto training mô hình
    auto_train_task = None

    # Khởi tạo Redis Manager
    global redis_manager
    redis_manager = RedisManager()

    # Khởi tạo DB Manager
    global db
    db = DBManager()

    # Khởi tạo scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(auto_train_models, 'cron', hour=2, minute=0)
    scheduler.start()
    logger.info("Auto training scheduler đã khởi động.")

    # Khởi tạo bot application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", analyze_command))
    app.add_handler(CommandHandler("getid", get_id))
    app.add_handler(CommandHandler("approve", approve_user))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, notify_admin_new_user))
    logger.info("🤖 Bot khởi động!")

    # Thiết lập webhook cho môi trường Render
    BASE_URL = os.getenv("RENDER_EXTERNAL_URL", f"https://{os.getenv('RENDER_SERVICE_NAME')}.onrender.com")
    WEBHOOK_URL = f"{BASE_URL}/{TELEGRAM_TOKEN}"
    
    # Khởi động bot với webhook trong try-except để xử lý lỗi
    try:
        # Thiết lập webhook
        await app.bot.set_webhook(url=WEBHOOK_URL)
        
        # Khởi động ứng dụng
        await app.initialize()
        await app.start()
        
        # Thiết lập web server để xử lý webhook
        webapp = web.Application()
        
        # Webhook handler
        async def webhook_handler(request):
            # Loại bỏ kiểm tra request.match_info.get('token') không chính xác
            try:
                request_body_bytes = await request.read()
                await app.update_queue.put(
                    Update.de_json(json.loads(request_body_bytes), app.bot)
                )
                return web.Response()
            except json.JSONDecodeError:
                logger.error("Lỗi decode JSON từ request webhook")
                return web.Response(status=400) # Bad Request
            except Exception as e:
                logger.error(f"Lỗi xử lý webhook: {str(e)}")
                return web.Response(status=500) # Internal Server Error
        
        # Đăng ký route
        webapp.router.add_post(f'/{TELEGRAM_TOKEN}', webhook_handler)
        
        # Khởi động web server
        runner = web.AppRunner(webapp)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', PORT)
        
        # Khởi động site
        await site.start()
        
        logger.info(f"Webhook đã thiết lập tại {WEBHOOK_URL}")
        logger.info(f"Bot đang lắng nghe trên 0.0.0.0:{PORT}")
        
        # Giữ ứng dụng chạy
        shutdown_event = asyncio.Event()
        
        # Thiết lập signal handlers cho graceful shutdown
        async def shutdown(signal_):
            """Xử lý graceful shutdown khi nhận signal"""
            logger.info(f"Nhận signal {signal_.name}, đang tắt bot...")
            
            # Đóng các kết nối và tài nguyên
            try:
                # Đóng các thread pool và process pool
                thread_executor.shutdown(wait=False)
                process_executor.shutdown(wait=False)
                
                # Đóng Redis connection
                await redis_manager.redis_client.close()
                logger.info("Đã đóng kết nối Redis")
                
                # Dừng webhook và web server
                await site.stop()
                await runner.cleanup()
                logger.info("Đã dừng web server")
                
                # Dừng bot application
                await app.stop()
                await app.shutdown()
                logger.info("Đã dừng bot application")
                
                # Dừng scheduler
                scheduler.shutdown()
                logger.info("Đã dừng scheduler")
            except Exception as e:
                logger.error(f"Lỗi khi shutdown: {str(e)}")
            finally:
                # Đánh thức shutdown event để thoát main loop
                shutdown_event.set()
        
        # Đăng ký các signal handlers
        loop = asyncio.get_running_loop()
        for s in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(s, lambda s=s: asyncio.create_task(shutdown(s)))
        
        logger.info("Đã cài đặt signal handlers cho graceful shutdown")
        
        # Chờ signal shutdown
        await shutdown_event.wait()
        
        logger.info("Bot đã tắt một cách an toàn")
        
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập webhook: {str(e)}")
        # Quay lại chế độ polling nếu webhook thất bại
        logger.info("Chuyển sang chế độ polling...")
        await app.run_polling()

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

        class TestDataValidator(unittest.TestCase):
            def setUp(self):
                # Tạo dữ liệu test
                self.df = pd.DataFrame({
                    'open': [100, 105, 110, None, 107],
                    'high': [110, 115, 120, 118, 112],
                    'low': [98, 103, 108, 105, 102],
                    'close': [105, 112, 116, 107, 105],
                    'volume': [10000, 12000, 15000, 9000, 11000]
                }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
                
            def test_validate_ticker(self):
                # Test mã hợp lệ
                is_valid, _ = DataValidator.validate_ticker('FPT')
                self.assertTrue(is_valid)
                
                # Test mã index hợp lệ
                is_valid, _ = DataValidator.validate_ticker('VNINDEX')
                self.assertTrue(is_valid)
                
                # Test mã không hợp lệ
                is_valid, _ = DataValidator.validate_ticker('ABCDE')
                self.assertFalse(is_valid)
                
                # Test mã rỗng
                is_valid, _ = DataValidator.validate_ticker('')
                self.assertFalse(is_valid)
                
            def test_validate_timeframe(self):
                # Test các timeframe hợp lệ
                for tf in ['1D', '1W', '1M', '5m', '15m', '30m', '1h', '4h']:
                    is_valid, _ = DataValidator.validate_timeframe(tf)
                    self.assertTrue(is_valid)
                    
                # Test các alias timeframe
                for tf in ['d', 'daily', 'w', 'weekly', 'h', 'hourly']:
                    is_valid, _ = DataValidator.validate_timeframe(tf)
                    self.assertTrue(is_valid)
                
                # Test timeframe không hợp lệ
                is_valid, _ = DataValidator.validate_timeframe('10m')
                self.assertFalse(is_valid)
                
            def test_align_timestamp(self):
                # Tạo dữ liệu intraday không căn chỉnh
                intraday_df = pd.DataFrame({
                    'open': [100, 105, 110, 107, 108],
                    'high': [110, 115, 120, 112, 115],
                    'low': [98, 103, 108, 102, 105],
                    'close': [105, 112, 116, 105, 110],
                    'volume': [10000, 12000, 15000, 11000, 13000]
                }, index=pd.to_datetime([
                    '2023-01-01 09:02:30',
                    '2023-01-01 09:17:45',
                    '2023-01-01 09:32:15',
                    '2023-01-01 09:45:00',
                    '2023-01-01 10:01:20'
                ]))
                
                # Test căn chỉnh 15m
                aligned_df = DataValidator.align_timestamp(intraday_df, '15m')
                # Kiểm tra timestamps đã căn chỉnh đúng
                for idx in aligned_df.index:
                    self.assertEqual(idx.minute % 15, 0)
                    
                # Test căn chỉnh 1h
                aligned_df = DataValidator.align_timestamp(intraday_df, '1h')
                # Kiểm tra timestamps đã căn chỉnh đúng
                for idx in aligned_df.index:
                    self.assertEqual(idx.minute, 0)
                
            def test_handle_outliers(self):
                # Tạo dữ liệu có outlier
                df_outliers = pd.DataFrame({
                    'open': [100, 105, 110, 200, 107],  # 200 là outlier
                    'high': [110, 115, 120, 300, 112],  # 300 là outlier
                    'low': [98, 103, 108, 105, 102],
                    'close': [105, 112, 116, 250, 105]  # 250 là outlier
                }, index=pd.date_range('2023-01-01', periods=5, freq='D'))
                
                # Test đánh dấu outlier
                df_marked, report = DataValidator.handle_outliers(df_outliers, method='iqr', action='mark')
                self.assertTrue('is_outlier' in df_marked.columns)
                self.assertTrue(df_marked.iloc[3]['is_outlier'])  # 4th row has outliers
                
                # Test winsorize outlier
                df_fixed, _ = DataValidator.handle_outliers(df_outliers, method='iqr', action='winsorize')
                # Kiểm tra giá trị đã được thay thế
                self.assertLess(df_fixed.iloc[3]['open'], 200)  # Should be capped
                
            def test_fill_missing_by_timeframe(self):
                # Tạo dữ liệu intraday có missing values
                intraday_df = pd.DataFrame({
                    'open': [100, None, 110, None, 108],
                    'high': [110, 115, None, 112, 115],
                    'low': [98, 103, 108, None, 105],
                    'close': [105, 112, 116, 105, None],
                    'volume': [10000, None, 15000, 11000, 13000]
                }, index=pd.to_datetime([
                    '2023-01-01 09:00:00',
                    '2023-01-01 09:15:00',
                    '2023-01-01 09:30:00',
                    '2023-01-01 09:45:00',
                    '2023-01-01 10:00:00'
                ]))
                
                # Test fill missing theo 15m
                filled_df = DataValidator.fill_missing_by_timeframe(intraday_df, '15m')
                self.assertFalse(filled_df.isna().any().any())  # No NaN values
        
        unittest.main(argv=[sys.argv[0]])
    else:
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.run(main())