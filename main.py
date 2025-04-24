#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot Chứng Khoán Toàn Diện Phiên Bản V20.0 (Nâng cấp):
- Tích hợp AI OpenRouter cho phân tích mẫu hình, sóng, và nến nhật.
- Sử dụng mô hình deepseek/deepseek-chat-v3-0324:free.
- Chuẩn hóa dữ liệu và pipeline xử lý.
- Tích hợp fallback sang Groq AI khi Gemini gặp vấn đề.
- Hệ thống cache thông minh với cachetools.
- Phát hiện và xử lý outlier nâng cao.
- Phân tích kỹ thuật đa khung thời gian cải tiến.
- Hệ thống dự báo hybrid cải tiến.
- Thu thập tin tức đa nguồn cải tiến.
- Đảm bảo các chức năng và công nghệ hiện có không bị ảnh hưởng.
"""

import os
import sys
import io
import logging
import pickle
from datetime import datetime, timedelta, time as time_type
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import traceback

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
import pytz

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tenacity import retry, stop_after_attempt, wait_exponential

import aiohttp
from aiohttp import web  # Thêm import này
import json
import re
import cachetools

# ---------- CẤU HÌNH & LOGGING ----------
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "deepseek-r1-distill-llama-70b"  # Model ổn định cho fallback
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
TZ = pytz.timezone('Asia/Bangkok')

# ---------- CACHE THÔNG MINH VỚI CACHETOOLS ----------
cache = cachetools.TTLCache(maxsize=100, ttl=CACHE_EXPIRE_SHORT)

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
    Lớp xác thực và chuẩn hóa dữ liệu toàn diện.
    Cung cấp các phương thức để xác thực, chuẩn hóa và căn chỉnh dữ liệu chứng khoán.
    """
    # Danh sách khung thời gian hợp lệ
    SUPPORTED_TIMEFRAMES = [
        '5m', '15m', '30m', '1h', '4h',  # Khung thời gian mới
        '1D', '1W', '1M'                 # Khung thời gian legacy
    ]
    
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
        'open': time_type(9, 0),
        'close': time_type(15, 0)
    }
    
    # Ngày giao dịch (Vietnam)
    TRADING_DAYS = [0, 1, 2, 3, 4]  # 0: Monday, 4: Friday
    
    # Danh sách các mã chỉ số
    INDICES = ['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX', 'UPCOM']
    
    # Định dạng mã hợp lệ
    TICKER_PATTERN = r'^[A-Z0-9]{3,6}$'
    
    @staticmethod
    def validate_ticker(ticker: str) -> (bool, str):
        """Xác thực mã chứng khoán và trả về kết quả kèm thông báo"""
        import re
        ticker = ticker.upper().strip()
        
        # Kiểm tra xem ticker có phải là chỉ số không
        if ticker in DataValidator.INDICES:
            return True, f"{ticker} là chỉ số hợp lệ"
            
        # Nếu không phải chỉ số, kiểm tra định dạng
        if re.match(DataValidator.TICKER_PATTERN, ticker):
            return True, f"{ticker} là mã cổ phiếu hợp lệ"
            
        return False, f"Mã chứng khoán không hợp lệ: {ticker}"
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> (bool, str):
        """Xác thực khung thời gian và trả về kết quả kèm thông báo"""
        timeframe = timeframe.upper() if timeframe.lower() not in ['5m', '15m', '30m', '1h', '4h'] else timeframe
        
        # Bảng ánh xạ chuẩn hóa
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
        
        if normalized in DataValidator.SUPPORTED_TIMEFRAMES:
            return True, normalized
            
        return False, f"Khung thời gian không hợp lệ: {timeframe}. Khung thời gian hợp lệ: {', '.join(DataValidator.SUPPORTED_TIMEFRAMES)}"
    
    @staticmethod
    def validate_candle_count(count: int, min_count: int = 10, max_count: int = 5000) -> (bool, str):
        """Xác thực số lượng nến và trả về kết quả kèm thông báo"""
        try:
            count = int(count)
            if min_count <= count <= max_count:
                return True, f"Số lượng nến hợp lệ: {count}"
            else:
                return False, f"Số lượng nến phải trong khoảng {min_count} - {max_count}. Nhận được: {count}"
        except ValueError:
            return False, f"Giá trị không phải là số nguyên: {count}"
    
    @staticmethod
    def align_timestamp(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
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
        
        def align_to_timeframe(timestamp):
            # Với 4h, căn chỉnh vào 1h, 5h, 9h, 13h, 17h, 21h
            if timeframe == '4h':
                hour = timestamp.hour
                aligned_hour = (hour // 4) * 4 + 1
                return timestamp.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
            
            # Với 1h, căn chỉnh vào đầu giờ
            elif timeframe == '1h':
                return timestamp.replace(minute=0, second=0, microsecond=0)
                
            # Với 30m, căn chỉnh vào 0 và 30 phút
            elif timeframe == '30m':
                minute = 0 if timestamp.minute < 30 else 30
                return timestamp.replace(minute=minute, second=0, microsecond=0)
                
            # Với 15m, căn chỉnh vào 0, 15, 30, 45 phút
            elif timeframe == '15m':
                minute = (timestamp.minute // 15) * 15
                return timestamp.replace(minute=minute, second=0, microsecond=0)
                
            # Với 5m, căn chỉnh vào mỗi 5 phút
            elif timeframe == '5m':
                minute = (timestamp.minute // 5) * 5
                return timestamp.replace(minute=minute, second=0, microsecond=0)
                
            # Với 1D, căn chỉnh vào 15:00 (đóng cửa thị trường VN)
            elif timeframe == '1D':
                return timestamp.replace(hour=15, minute=0, second=0, microsecond=0)
                
            # Với 1W, căn chỉnh vào 15:00 thứ 6
            elif timeframe == '1W':
                days_to_friday = (4 - timestamp.weekday()) % 7
                friday = timestamp + timedelta(days=days_to_friday)
                return friday.replace(hour=15, minute=0, second=0, microsecond=0)
                
            # Với 1M, căn chỉnh vào ngày cuối cùng của tháng
            elif timeframe == '1M':
                next_month = timestamp.replace(day=28) + timedelta(days=4)
                last_day = next_month - timedelta(days=next_month.day)
                return last_day.replace(hour=15, minute=0, second=0, microsecond=0)
                
            return timestamp
        
        df.index = pd.DatetimeIndex([align_to_timeframe(ts) for ts in df.index])
        return df.sort_index()
    
    @staticmethod
    def handle_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5, 
                        columns: list = ['open', 'high', 'low', 'close'], action: str = 'mark') -> (pd.DataFrame, str):
        """
        Phát hiện và xử lý giá trị ngoại lai trong dữ liệu.
        
        Các phương pháp phát hiện:
        - 'iqr': Sử dụng phạm vi tứ phân vị - Interquartile Range (IQR)
        - 'zscore': Sử dụng Z-score
        - 'modified_zscore': Z-score biến thể với median thay vì mean
        
        Các hành động xử lý:
        - 'mark': Chỉ đánh dấu các giá trị ngoại lai
        - 'clip': Cắt giá trị về giới hạn gần nhất
        - 'fill': Điền các giá trị ngoại lai bằng giá trị gần nhất không phải ngoại lai
        - 'remove': Xóa các hàng chứa giá trị ngoại lai
        """
        if df.empty:
            return df, "DataFrame rỗng, không thể xử lý outlier"
        
        df_out = df.copy()
        outlier_info = []
        detected_outliers = {}
        
        for col in columns:
            if col not in df.columns:
                continue
                
            col_data = df[col]
            outliers_mask = pd.Series(False, index=df.index)
            
            # Phát hiện outlier theo phương pháp đã chọn
            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
                bounds_info = f"Phạm vi IQR cho {col}: [{lower_bound:.2f}, {upper_bound:.2f}]"
                
            elif method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                z_scores = np.abs((col_data - mean) / std)
                outliers_mask = z_scores > threshold
                bounds_info = f"Ngưỡng Z-score cho {col}: {threshold} (mean={mean:.2f}, std={std:.2f})"
                
            elif method == 'modified_zscore':
                median = col_data.median()
                mad = np.median(np.abs(col_data - median))
                modified_z_scores = 0.6745 * np.abs(col_data - median) / mad if mad > 0 else np.zeros_like(col_data)
                outliers_mask = modified_z_scores > threshold
                bounds_info = f"Ngưỡng Modified Z-score cho {col}: {threshold} (median={median:.2f}, MAD={mad:.2f})"
            
            else:
                return df, f"Phương pháp xử lý outlier không hợp lệ: {method}"
            
            outlier_indices = df.index[outliers_mask]
            detected_outliers[col] = outlier_indices
            
            outlier_count = len(outlier_indices)
            if outlier_count > 0:
                outlier_info.append(f"{outlier_count} outlier trong cột {col} ({bounds_info})")
                
                # Xử lý outlier theo hành động đã chọn
                if action == 'mark':
                    df_out.loc[outliers_mask, f'{col}_is_outlier'] = True
                    
                elif action == 'clip':
                    if method == 'iqr':
                        df_out.loc[col_data < lower_bound, col] = lower_bound
                        df_out.loc[col_data > upper_bound, col] = upper_bound
                    elif method in ['zscore', 'modified_zscore']:
                        std_or_mad = std if method == 'zscore' else mad
                        center = mean if method == 'zscore' else median
                        df_out.loc[outliers_mask, col] = np.clip(
                            df_out.loc[outliers_mask, col],
                            center - threshold * std_or_mad,
                            center + threshold * std_or_mad
                        )
                    outlier_info.append(f"  - Các giá trị outlier trong {col} đã được clip")
                    
                elif action == 'fill':
                    # Điền các giá trị bằng phương pháp ffill rồi bfill
                    df_out.loc[outliers_mask, col] = np.nan
                    df_out[col] = df_out[col].fillna(method='ffill').fillna(method='bfill')
                    outlier_info.append(f"  - Các giá trị outlier trong {col} đã được thay thế")
                    
                elif action == 'remove':
                    # Đánh dấu các hàng sẽ bị xóa
                    rows_to_remove = outliers_mask
                    outlier_info.append(f"  - {rows_to_remove.sum()} hàng có outlier trong {col} sẽ bị xóa")
                    
                else:
                    return df, f"Hành động xử lý outlier không hợp lệ: {action}"
        
        # Thực hiện xóa hàng nếu action là 'remove' và có ít nhất một cột có outlier
        if action == 'remove' and any(detected_outliers.values()):
            all_outlier_rows = pd.Series(False, index=df.index)
            for col, indices in detected_outliers.items():
                all_outlier_rows = all_outlier_rows | df.index.isin(indices)
            
            df_out = df_out.loc[~all_outlier_rows]
            outlier_info.append(f"Tổng cộng đã xóa {all_outlier_rows.sum()} hàng có outlier")
        
        outlier_summary = "\n".join(outlier_info) if outlier_info else "Không phát hiện outlier"
        
        return df_out, outlier_summary
    
    @staticmethod
    def fill_missing_by_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Điền các giá trị bị thiếu trong dữ liệu với xử lý khác nhau tùy theo khung thời gian.
        Các khung thời gian intraday (5m, 15m, ...) được xử lý chặt chẽ hơn.
        """
        if df.empty:
            return df
        
        df_filled = df.copy()
        
        # Kiểm tra nếu không có giá trị NaN
        if not df_filled.isna().any().any():
            return df_filled
        
        # Xử lý điền giá trị thiếu cho cột giá
        price_columns = ['open', 'high', 'low', 'close', 'adj_close'] if 'adj_close' in df.columns else ['open', 'high', 'low', 'close']
        
        # Xác định phương pháp điền cho từng timeframe
        if timeframe in ['5m', '15m', '30m', '1h', '4h']:  # Khung thời gian intraday
            # Đối với dữ liệu intraday, chỉ điền giá trị trong cùng ngày giao dịch
            # Chia thành các nhóm theo ngày
            df_filled.index = pd.to_datetime(df_filled.index)
            dates = pd.DatetimeIndex([d.date() for d in df_filled.index])
            
            for date in dates.unique():
                # Lấy các hàng trong ngày hiện tại
                mask = pd.Series([d.date() == date for d in df_filled.index], index=df_filled.index)
                day_slice = df_filled.loc[mask].copy()
                
                # Điền dữ liệu cho nhóm ngày
                if day_slice[price_columns].isna().any().any():
                    # Đối với giá, sử dụng phương pháp linear interpolation trong ngày
                    day_slice[price_columns] = day_slice[price_columns].interpolate(method='linear', limit_direction='both')
                    
                    # Nếu còn giá trị thiếu ở đầu/cuối ngày, sử dụng ffill/bfill
                    day_slice[price_columns] = day_slice[price_columns].fillna(method='ffill').fillna(method='bfill')
                    
                    # Cập nhật lại vào DataFrame chính
                    df_filled.loc[mask, price_columns] = day_slice[price_columns]
                
                # Đối với volume, điền 0 cho giá trị thiếu
                if 'volume' in df.columns and day_slice['volume'].isna().any():
                    df_filled.loc[mask, 'volume'] = day_slice['volume'].fillna(0)
            
        else:  # Khung thời gian ngày, tuần, tháng
            # Điền các giá trị thiếu theo phương pháp ffill trước, rồi bfill sau
            df_filled[price_columns] = df_filled[price_columns].fillna(method='ffill')
            df_filled[price_columns] = df_filled[price_columns].fillna(method='bfill')
            
            # Đối với volume, điền 0 cho giá trị thiếu
            if 'volume' in df.columns:
                df_filled['volume'] = df_filled['volume'].fillna(0)
        
        return df_filled
    
    @staticmethod
    def validate_fundamental_data(data: dict) -> (dict, list):
        """
        Xác thực dữ liệu cơ bản và trả về những lưu ý quan trọng.
        """
        if not data:
            return {}, ["Không có dữ liệu cơ bản"]
        
        validated_data = {}
        observations = []
        
        # Chuẩn hóa dữ liệu và chuyển đổi kiểu dữ liệu
        for key, value in data.items():
            # Bỏ qua các giá trị không hợp lệ
            if value is None or (isinstance(value, (float, int)) and (np.isnan(value) or np.isinf(value))):
                continue
                
            # Chuẩn hóa kiểu dữ liệu số
            if isinstance(value, (np.float32, np.float64)):
                validated_data[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                validated_data[key] = int(value)
            else:
                validated_data[key] = value
        
        # Kiểm tra các thông số cơ bản quan trọng
        if 'marketCap' in validated_data:
            market_cap = validated_data['marketCap']
            if market_cap < 1e9:  # Nhỏ hơn 1 tỷ
                observations.append(f"Vốn hóa thị trường thấp: {market_cap:,.0f} (< 1 tỷ)")
        
        # Kiểm tra P/E
        if 'trailingPE' in validated_data:
            pe = validated_data['trailingPE']
            if pe < 0:
                observations.append(f"P/E âm: {pe:.2f} (Doanh nghiệp có thể đang lỗ)")
            elif pe > 50:
                observations.append(f"P/E rất cao: {pe:.2f} (> 50)")
        
        # Kiểm tra P/B
        if 'priceToBook' in validated_data:
            pb = validated_data['priceToBook']
            if pb < 1:
                observations.append(f"P/B thấp: {pb:.2f} (< 1)")
            elif pb > 10:
                observations.append(f"P/B rất cao: {pb:.2f} (> 10)")
        
        # Kiểm tra ROE
        if 'returnOnEquity' in validated_data:
            roe = validated_data['returnOnEquity'] * 100 if validated_data['returnOnEquity'] < 1 else validated_data['returnOnEquity']
            if roe < 0:
                observations.append(f"ROE âm: {roe:.2f}% (Doanh nghiệp có thể đang không hiệu quả)")
            elif roe > 30:
                observations.append(f"ROE rất cao: {roe:.2f}% (> 30%)")
        
        # Kiểm tra biến động giá
        if 'beta' in validated_data:
            beta = validated_data['beta']
            if beta > 2:
                observations.append(f"Beta cao: {beta:.2f} (Cổ phiếu biến động mạnh)")
            elif beta < 0:
                observations.append(f"Beta âm: {beta:.2f} (Cổ phiếu biến động ngược thị trường)")
        
        # Kiểm tra thanh khoản
        if 'averageVolume' in validated_data and 'volume' in validated_data:
            avg_vol = validated_data['averageVolume']
            cur_vol = validated_data['volume']
            if cur_vol > avg_vol * 3:
                observations.append(f"Khối lượng giao dịch cao bất thường: {cur_vol:,.0f} (gấp {cur_vol/avg_vol:.1f} lần trung bình)")
            elif cur_vol < avg_vol * 0.3:
                observations.append(f"Khối lượng giao dịch thấp: {cur_vol:,.0f} (chỉ bằng {cur_vol/avg_vol:.1f} lần trung bình)")
                
        return validated_data, observations

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
                    else:
                        # Daily or longer timeframes
                        start_date = (datetime.now() - timedelta(days=(num_candles + 1) * 3)).strftime('%Y-%m-%d')
                    
                    # Fetch data with appropriate interval
                    df = stock.quote.history(start=start_date, end=end_date, interval=timeframe)
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
    """Tải nội dung từ feed RSS."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.error(f"Lỗi khi tải RSS từ {url}: HTTP {response.status}")
                    return ""
    except Exception as e:
        logger.error(f"Lỗi khi tải RSS từ {url}: {str(e)}")
        return ""

async def fetch_rss(url: str, keywords: list = None, is_symbol_search: bool = False) -> list:
    """
    Tải và lọc tin tức từ nguồn RSS.
    
    Args:
        url: URL của feed RSS
        keywords: Danh sách từ khóa cần lọc
        is_symbol_search: True nếu đang tìm kiếm mã chứng khoán cụ thể
        
    Returns:
        Danh sách bài viết phù hợp
    """
    try:
        content = await fetch_rss_feed(url)
        if not content:
            return []
            
        feed = feedparser.parse(content)
        
        if not feed.entries:
            return []
            
        result = []
        for entry in feed.entries[:20]:  # Chỉ xử lý 20 bài mới nhất
            title = entry.get('title', '')
            link = entry.get('link', '')
            published = entry.get('published', entry.get('pubDate', ''))
            description = html.unescape(re.sub(r'<[^>]+>', '', entry.get('description', '')))
            
            # Chuyển đổi thời gian nếu có
            try:
                date = pd.to_datetime(published).strftime('%Y-%m-%d %H:%M')
            except:
                date = published
                
            # Nếu không có từ khóa, lấy tất cả tin
            if not keywords:
                result.append({
                    'title': title,
                    'link': link,
                    'date': date,
                    'description': description[:200] + '...' if len(description) > 200 else description
                })
                continue
                
            # Lọc theo từ khóa
            title_lower = title.lower()
            desc_lower = description.lower()
            
            if is_symbol_search:
                # Tìm kiếm chính xác hơn cho mã chứng khoán
                # Cần tìm mã chính xác (ví dụ "VIC" không phải "VICA" hay "VICTEX")
                for keyword in keywords:
                    ticker_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    if (re.search(ticker_pattern, title_lower) or 
                        re.search(ticker_pattern, desc_lower)):
                        result.append({
                            'title': title,
                            'link': link,
                            'date': date,
                            'description': description[:200] + '...' if len(description) > 200 else description
                        })
                        break
            else:
                # Tìm kiếm thông thường
                if any(keyword.lower() in title_lower or keyword.lower() in desc_lower 
                      for keyword in keywords):
                    result.append({
                        'title': title,
                        'link': link,
                        'date': date,
                        'description': description[:200] + '...' if len(description) > 200 else description
                    })
                    
        return result
        
    except Exception as e:
        logger.error(f"Lỗi khi xử lý RSS {url}: {str(e)}")
        return []

async def get_news(symbol: str = None, limit: int = 3) -> list:
    """
    Lấy tin tức chứng khoán từ nhiều nguồn.
    
    Args:
        symbol: Mã chứng khoán cần tìm tin
        limit: Số lượng tin tối đa
        
    Returns:
        Danh sách các tin tức
    """
    # Danh sách nguồn RSS
    rss_sources = [
        "https://vneconomy.vn/rss/chung-khoan-24.rss",
        "https://cafef.vn/feed/thi-truong-chung-khoan.rss",
        "https://vietstock.vn/rss/event-news-chung-khoan.rss",
        "https://tinnhanhchungkhoan.vn/rss/chung-khoan.rss",
        "https://ndh.vn/feed/chung-khoan",
        "https://cafef.vn/feed/timeline-chung-khoan.rss",
        "https://vn.investing.com/rss/news_25.rss",  # Tin tức chứng khoán từ Investing
        "https://ttvn.toquoc.vn/rss/chung-khoan.rss"
    ]
    
    redis_client = None
    
    try:
        # Check cache first
        cache_key = f"news:{symbol or 'market'}"
        redis_client = await redis.from_url(REDIS_URL)
        cached_news = await redis_client.get(cache_key)
        
        if cached_news:
            return json.loads(cached_news)
            
        # Không có cache, tải tin mới
        keywords = None
        is_symbol_search = False
        
        if symbol:
            if is_index(symbol):
                if symbol == "VNINDEX":
                    keywords = ["VN-Index", "VNIndex", "chỉ số VN", "thị trường chứng khoán"]
                elif symbol == "VN30":
                    keywords = ["VN30", "nhóm VN30", "rổ VN30", "30 cổ phiếu lớn"]
                elif symbol == "HNX30":
                    keywords = ["HNX30", "nhóm HNX30", "rổ HNX30", "sàn HNX"]
                elif symbol == "HNXINDEX":
                    keywords = ["HNX-Index", "HNXIndex", "chỉ số HNX", "sàn Hà Nội"]
                elif symbol == "UPCOM":
                    keywords = ["UPCOM", "UPCoM", "UpCom", "sàn Upcom"]
            else:
                keywords = [symbol]
                is_symbol_search = True
                
                # Tìm kiếm thêm về công ty
                try:
                    company_info = await run_in_thread(lambda: Vnstock().company_profile(symbol))
                    if company_info and 'companyName' in company_info:
                        company_name = company_info['companyName'].split('(')[0].strip()
                        keywords.append(company_name)
                except:
                    pass
        
        # Fetch từ nhiều nguồn song song
        tasks = [fetch_rss(url, keywords, is_symbol_search) for url in rss_sources]
        results = await asyncio.gather(*tasks)
        
        # Gộp và sắp xếp kết quả
        all_news = []
        for result in results:
            all_news.extend(result)
        
        # Loại bỏ tin trùng lặp dựa trên tiêu đề
        unique_news = []
        seen_titles = set()
        
        for news in all_news:
            title = news['title'].lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(news)
        
        # Sắp xếp theo ngày giảm dần (mới nhất trước) nếu có thể parse được ngày
        try:
            unique_news.sort(key=lambda x: pd.to_datetime(x['date'], errors='coerce'), reverse=True)
        except:
            pass
        
        # Lấy số lượng tin theo yêu cầu
        result = unique_news[:limit] if limit > 0 else unique_news
        
        # Lưu vào cache
        if redis_client and result:
            await redis_client.set(cache_key, json.dumps(result), ex=NEWS_CACHE_EXPIRE)
        
        return result
        
    except Exception as e:
        logger.error(f"Lỗi khi lấy tin tức: {str(e)}")
        return []
    finally:
        if redis_client:
            await redis_client.close()

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
    # Đánh giá dự báo Prophet bằng tính MAPE trên tập kiểm tra
    y_true = df['close'].iloc[-7:].values
    y_pred = forecast['yhat'][:7].values
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return 100 - mape  # Đổi MAPE thành độ chính xác phần trăm

def predict_xgboost_signal(df: pd.DataFrame, features: list) -> (int, float):
    # Predict buy/sell signal using XGBoost
    try:
        # Đảm bảo đủ dữ liệu để train
        if len(df) < 60:
            return 0, 0  # Không đủ dữ liệu
            
        # Tạo target: 1 nếu giá đóng cửa ngày mai > giá đóng cửa hôm nay, ngược lại 0
        df = df.copy()
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Xử lý giá trị NaN
        df = df.dropna()
        
        # Chia tập train và test
        train_size = int(len(df) * 0.8)
        X_train = df[features][:train_size]
        y_train = df['target'][:train_size]
        X_test = df[features][train_size:]
        y_test = df['target'][train_size:]
        
        # Huấn luyện mô hình
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
        model.fit(X_train, y_train)
        
        # Đánh giá mô hình
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Dự đoán tín hiệu cho ngày mai
        last_features = df[features].iloc[-1].values.reshape(1, -1)
        prediction = model.predict(last_features)[0]
        probability = model.predict_proba(last_features)[0][prediction]
        
        return prediction, probability * accuracy
    except Exception as e:
        logger.error(f"Error in XGBoost prediction: {str(e)}")
        return 0, 0

class EnhancedPredictor:
    """
    Lớp thực hiện dự báo nâng cao kết hợp các phương pháp khác nhau.
    - Kết hợp Prophet với XGBoost
    - Xử lý đa khung thời gian
    - Cân nhắc yếu tố mùa vụ và độ mạnh thị trường
    - Kiểm tra tính hợp lý của dự báo
    """
    
    def __init__(self):
        """Khởi tạo bộ dự báo nâng cao"""
        self.vietnam_holidays = get_vietnam_holidays([datetime.now().year - 1, datetime.now().year, datetime.now().year + 1])
    
    def hybrid_predict(self, df: pd.DataFrame, days: int = 5):
        """
        Dự báo sử dụng mô hình kết hợp (hybrid) giữa Prophet, XGBoost và phân tích kỹ thuật.
        
        Args:
            df: DataFrame dữ liệu giá theo ngày
            days: Số ngày dự báo trong tương lai
            
        Returns:
            dict: Kết quả dự báo với nhiều chỉ số
        """
        try:
            if len(df) < 60:
                return {"error": "Không đủ dữ liệu (cần ít nhất 60 điểm dữ liệu)"}
            
            result = {}
            
            # 1. Dự báo bằng Prophet
            prophet_data = prepare_data_for_prophet(df)
            
            model = Prophet(
                changepoint_prior_scale=0.05,
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=self.vietnam_holidays
            )
            
            # Thêm ngày trong tuần của Việt Nam (thứ 2 đến thứ 6)
            model.add_seasonality(
                name='weekday',
                period=5,
                fourier_order=3,
                condition_name='is_weekday'
            )
            
            # Thêm điều kiện ngày giao dịch
            prophet_data['is_weekday'] = prophet_data['ds'].dt.dayofweek.isin([0, 1, 2, 3, 4])
            
            model.fit(prophet_data)
            
            # Tạo dữ liệu dự báo
            future = model.make_future_dataframe(periods=days)
            future['is_weekday'] = future['ds'].dt.dayofweek.isin([0, 1, 2, 3, 4])
            forecast = model.predict(future)
            
            # Lấy giá dự báo cho các ngày tiếp theo
            future_predictions = forecast.tail(days)
            
            # Tính độ tin cậy của dự báo Prophet
            prophet_performance = self.evaluate_prophet_performance(df, forecast)
            
            # 2. Dự báo tín hiệu mua/bán bằng XGBoost
            features = ['sma20', 'sma50', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bollinger_hband', 'bollinger_lband']
            features = [f for f in features if f in df.columns]
            
            signal, signal_confidence = self.predict_xgboost_signal(df, features)
            
            # 3. Phân tích xu hướng từ chỉ số kỹ thuật
            trend_analysis = self._analyze_technical_trend(df)
            
            # 4. Kết hợp các dự báo
            last_price = df['close'].iloc[-1]
            
            # Các ngày dự báo trong tương lai (chỉ lấy ngày giao dịch)
            trading_days = []
            predictions = []
            confidences = []
            trends = []
            
            # Chỉ lấy những ngày giao dịch trong dự báo
            for i, row in future_predictions.iterrows():
                day_of_week = row['ds'].dayofweek
                if day_of_week in [0, 1, 2, 3, 4]:  # Thứ 2 đến thứ 6
                    trading_days.append(row['ds'])
                    predictions.append(row['yhat'])
                    
                    # Tính độ tin cậy cho từng ngày
                    days_ahead = (row['ds'] - prophet_data['ds'].iloc[-1]).days
                    confidence = max(0, prophet_performance - days_ahead * 3)
                    confidences.append(confidence / 100)
                    
                    # Xu hướng theo ngày
                    if i > 0:
                        prev_pred = future_predictions.iloc[i-1]['yhat']
                        trends.append(1 if row['yhat'] > prev_pred else (-1 if row['yhat'] < prev_pred else 0))
                    else:
                        trends.append(1 if row['yhat'] > last_price else (-1 if row['yhat'] < last_price else 0))
            
            # Kiểm tra tính hợp lý của dự báo
            predictions = self._apply_sanity_checks(predictions, last_price)
            
            # 5. Tính toán các chỉ số khác
            price_change = [(price - last_price) / last_price * 100 for price in predictions]
            
            # Tính xác suất tăng/giảm
            if signal == 1:
                prob_up = signal_confidence
                prob_down = 1 - signal_confidence
            else:
                prob_up = 1 - signal_confidence
                prob_down = signal_confidence
            
            # Điều chỉnh xác suất dựa trên độ mạnh của xu hướng hiện tại
            if trend_analysis['strength'] > 0.7:
                if trend_analysis['direction'] == 'up':
                    prob_up = min(0.95, prob_up * 1.2)
                    prob_down = 1 - prob_up
                else:
                    prob_down = min(0.95, prob_down * 1.2)
                    prob_up = 1 - prob_down
            
            # 6. Tạo kết quả trả về
            result = {
                'last_price': last_price,
                'forecast_dates': trading_days,
                'forecast_prices': predictions,
                'price_change_percent': price_change,
                'confidence': confidences,
                'trend_direction': trends,
                'signal': 'Mua' if signal == 1 else 'Bán',
                'signal_confidence': signal_confidence * 100,
                'probability_up': prob_up * 100,
                'probability_down': prob_down * 100,
                'overall_trend': trend_analysis['direction'],
                'trend_strength': trend_analysis['strength'] * 100,
                'prophet_performance': prophet_performance
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Lỗi trong dự báo hybrid: {str(e)}\n{traceback.format_exc()}")
            return {"error": f"Lỗi dự báo: {str(e)}"}
    
    def _analyze_technical_trend(self, df: pd.DataFrame) -> dict:
        """Phân tích xu hướng dựa trên các chỉ số kỹ thuật"""
        # Lấy dữ liệu gần đây nhất
        recent_data = df.tail(20).copy()
        
        # Xác định các tín hiệu
        signals = {
            'price_above_sma20': recent_data['close'].iloc[-1] > recent_data['sma20'].iloc[-1] if 'sma20' in recent_data.columns else None,
            'price_above_sma50': recent_data['close'].iloc[-1] > recent_data['sma50'].iloc[-1] if 'sma50' in recent_data.columns else None,
            'sma20_above_sma50': recent_data['sma20'].iloc[-1] > recent_data['sma50'].iloc[-1] if 'sma20' in recent_data.columns and 'sma50' in recent_data.columns else None,
            'rsi_above_50': recent_data['rsi'].iloc[-1] > 50 if 'rsi' in recent_data.columns else None,
            'macd_above_signal': recent_data['macd'].iloc[-1] > recent_data['macd_signal'].iloc[-1] if 'macd' in recent_data.columns and 'macd_signal' in recent_data.columns else None,
            'price_trend_up': recent_data['close'].iloc[-1] > recent_data['close'].iloc[-5] if len(recent_data) >= 5 else None
        }
        
        # Đếm số tín hiệu tăng/giảm
        up_signals = sum(1 for signal in signals.values() if signal is True)
        down_signals = sum(1 for signal in signals.values() if signal is False)
        valid_signals = sum(1 for signal in signals.values() if signal is not None)
        
        # Tính cường độ xu hướng
        if valid_signals > 0:
            if up_signals > down_signals:
                direction = 'up'
                strength = up_signals / valid_signals
            else:
                direction = 'down'
                strength = down_signals / valid_signals
        else:
            direction = 'neutral'
            strength = 0
        
        return {'direction': direction, 'strength': strength, 'signals': signals}
    
    def _apply_sanity_checks(self, predictions, last_value):
        """Kiểm tra và điều chỉnh các dự báo không hợp lý"""
        max_daily_change = 0.07  # Giới hạn thay đổi tối đa 7% mỗi ngày
        
        adjusted_predictions = [predictions[0]]  # Giữ nguyên giá trị đầu tiên
        
        # Kiểm tra từ ngày thứ 2 trở đi
        for i in range(1, len(predictions)):
            prev_prediction = adjusted_predictions[i-1]
            curr_prediction = predictions[i]
            
            # Tính % thay đổi
            pct_change = abs(curr_prediction - prev_prediction) / prev_prediction
            
            if pct_change > max_daily_change:
                # Giới hạn thay đổi trong phạm vi cho phép
                if curr_prediction > prev_prediction:
                    adjusted_predictions.append(prev_prediction * (1 + max_daily_change))
                else:
                    adjusted_predictions.append(prev_prediction * (1 - max_daily_change))
            else:
                adjusted_predictions.append(curr_prediction)
        
        return adjusted_predictions
    
    def evaluate_prophet_performance(self, df: pd.DataFrame, forecast: pd.DataFrame) -> float:
        """Đánh giá hiệu suất dự báo của Prophet"""
        # Lấy dữ liệu cho đánh giá
        evaluation_days = min(7, len(df) // 10)  # Lấy tối đa 7 ngày gần nhất hoặc 10% dữ liệu
        y_true = df['close'].tail(evaluation_days).values
        
        # Lấy giá trị dự báo tương ứng
        forecast_dates = pd.to_datetime(df.index[-evaluation_days:])
        
        # Tìm các dự báo trùng với ngày thực tế
        y_pred = []
        for date in forecast_dates:
            match = forecast[forecast['ds'] == date]
            if not match.empty:
                y_pred.append(match['yhat'].iloc[0])
            else:
                # Tìm ngày gần nhất
                closest_date = forecast['ds'].iloc[(forecast['ds'] - date).abs().argsort()[0]]
                match = forecast[forecast['ds'] == closest_date]
                y_pred.append(match['yhat'].iloc[0])
        
        # Tính MAPE
        mape = np.mean(np.abs((y_true - np.array(y_pred)) / y_true)) * 100
        accuracy = max(0, 100 - mape)  # Convert to accuracy percentage
        
        return min(accuracy, 95)  # Cap at 95% to avoid overconfidence
    
    def predict_xgboost_signal(self, df: pd.DataFrame, features: list) -> (int, float):
        """Dự đoán tín hiệu giao dịch sử dụng XGBoost"""
        try:
            # Đảm bảo đủ dữ liệu để train
            if len(df) < 60 or not all(feature in df.columns for feature in features):
                return 0, 0.5  # Không đủ dữ liệu hoặc thiếu features
                
            # Tạo target: 1 nếu giá đóng cửa ngày mai > giá đóng cửa hôm nay, ngược lại 0
            df = df.copy()
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
            
            # Xử lý giá trị NaN
            df = df.dropna()
            
            # Chia tập train và test
            train_size = int(len(df) * 0.8)
            X_train = df[features][:train_size]
            y_train = df['target'][:train_size]
            X_test = df[features][train_size:]
            y_test = df['target'][train_size:]
            
            # Huấn luyện mô hình
            model = xgb.XGBClassifier(
                n_estimators=100, 
                max_depth=3, 
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Đánh giá mô hình
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Dự đoán tín hiệu cho ngày mai
            last_features = df[features].iloc[-1].values.reshape(1, -1)
            prediction = model.predict(last_features)[0]
            probability = model.predict_proba(last_features)[0][prediction]
            
            # Điều chỉnh độ tin cậy dựa trên hiệu suất đã kiểm chứng
            confidence = probability * accuracy
            
            # Giới hạn độ tin cậy
            confidence = min(max(confidence, 0.5), 0.9)
            
            return prediction, confidence
        except Exception as e:
            logger.error(f"Lỗi trong dự đoán XGBoost: {str(e)}")
            return 0, 0.5

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

async def auto_train_models():
    """
    Tự động huấn luyện các mô hình dự báo cho các mã được theo dõi.
    Chạy theo lịch (mỗi ngày lúc 2 giờ sáng).
    """
    logger.info("Bắt đầu huấn luyện tự động các mô hình...")
    try:
        # Lấy danh sách các mã cần huấn luyện
        training_symbols = await get_training_symbols()
        
        if not training_symbols:
            logger.info("Không có mã nào cần huấn luyện")
            return
        
        logger.info(f"Huấn luyện mô hình cho {len(training_symbols)} mã: {', '.join(training_symbols)}")
        
        # Khởi tạo ModelDBManager và DataLoader
        model_db = ModelDBManager()
        data_loader = DataLoader()
        
        # Timeframes cần huấn luyện
        timeframes = ['1D', '1W']  # Daily và Weekly
        
        # Huấn luyện mô hình cho từng mã
        for symbol in training_symbols:
            try:
                logger.info(f"Đang huấn luyện mô hình cho {symbol}...")
                
                # Huấn luyện cho từng timeframe
                for timeframe in timeframes:
                    # Tải dữ liệu
                    data, error_msg = await data_loader.load_data(symbol, timeframe, 500)
                    
                    if data is None or data.empty or len(data) < 100:
                        logger.warning(f"Không đủ dữ liệu để huấn luyện cho {symbol} với timeframe {timeframe}: {error_msg}")
                        continue
                    
                    # Chuẩn hóa dữ liệu
                    data = DataNormalizer.normalize_dataframe(data)
                    
                    # Điền giá trị thiếu
                    data = DataValidator.fill_missing_by_timeframe(data, timeframe)
                    
                    # Kiểm tra và xử lý outlier
                    data, outlier_info = DataValidator.handle_outliers(
                        data, method='iqr', threshold=2.0, action='clip'
                    )
                    
                    if "Không phát hiện outlier" not in outlier_info:
                        logger.info(f"Xử lý outlier cho {symbol} ({timeframe}): {outlier_info}")
                    
                    # Tính toán các chỉ báo kỹ thuật
                    analyzer = TechnicalAnalyzer()
                    data = analyzer.calculate_indicators(data)
                    
                    # Huấn luyện mô hình Prophet
                    try:
                        # Dữ liệu cho prophet
                        prophet_data = prepare_data_for_prophet(data)
                        
                        # Tạo mô hình với cấu hình nâng cao
                        holidays_df = get_vietnam_holidays([datetime.now().year - 1, datetime.now().year, datetime.now().year + 1])
                        
                        model = Prophet(
                            changepoint_prior_scale=0.05,
                            yearly_seasonality=True,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            holidays=holidays_df
                        )
                        
                        # Thêm các yếu tố mùa vụ đặc biệt
                        model.add_seasonality(
                            name='weekday',
                            period=5,
                            fourier_order=3,
                            condition_name='is_weekday'
                        )
                        
                        # Chuẩn bị điều kiện ngày giao dịch
                        prophet_data['is_weekday'] = prophet_data['ds'].dt.dayofweek.isin([0, 1, 2, 3, 4])
                        
                        # Huấn luyện
                        model.fit(prophet_data)
                        
                        # Đánh giá
                        future = model.make_future_dataframe(periods=7)
                        future['is_weekday'] = future['ds'].dt.dayofweek.isin([0, 1, 2, 3, 4])
                        forecast = model.predict(future)
                        
                        # Tính độ chính xác
                        predictor = EnhancedPredictor()
                        performance = predictor.evaluate_prophet_performance(data, forecast)
                        
                        # Lưu mô hình
                        await model_db.store_trained_model(
                            symbol=symbol,
                            model_type="prophet",
                            model=model,
                            performance=performance,
                            version="2.0",
                            params={"changepoint_prior_scale": 0.05},
                            timeframe=timeframe
                        )
                        
                        logger.info(f"Đã huấn luyện và lưu mô hình Prophet cho {symbol} ({timeframe}) với độ chính xác {performance:.2f}%")
                    
                    except Exception as e:
                        logger.error(f"Lỗi khi huấn luyện mô hình Prophet cho {symbol} ({timeframe}): {str(e)}")
                    
                    # Huấn luyện mô hình XGBoost
                    try:
                        # Lấy các tính năng cho XGBoost
                        features = ['sma20', 'sma50', 'rsi', 'macd', 'macd_signal', 'macd_hist', 
                                  'bollinger_hband', 'bollinger_lband', 'volume']
                        available_features = [f for f in features if f in data.columns]
                        
                        if len(available_features) < 5:
                            logger.warning(f"Không đủ tính năng cho XGBoost {symbol} ({timeframe}): chỉ có {len(available_features)}")
                            continue
                        
                        # Tạo target: 1 nếu giá đóng cửa ngày mai > giá đóng cửa hôm nay, ngược lại 0
                        model_data = data.copy()
                        model_data['target'] = (model_data['close'].shift(-1) > model_data['close']).astype(int)
                        
                        # Xử lý giá trị NaN
                        model_data = model_data.dropna()
                        
                        if len(model_data) < 60:
                            logger.warning(f"Không đủ dữ liệu cho XGBoost {symbol} ({timeframe}): chỉ có {len(model_data)} hàng")
                            continue
                        
                        # Chia tập train và test
                        train_size = int(len(model_data) * 0.8)
                        X_train = model_data[available_features][:train_size]
                        y_train = model_data['target'][:train_size]
                        X_test = model_data[available_features][train_size:]
                        y_test = model_data['target'][train_size:]
                        
                        # Xây dựng mô hình
                        model = xgb.XGBClassifier(
                            n_estimators=100, 
                            max_depth=3, 
                            learning_rate=0.1,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42
                        )
                        
                        # Huấn luyện
                        model.fit(X_train, y_train)
                        
                        # Đánh giá
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred) * 100
                        
                        # Lưu mô hình
                        await model_db.store_trained_model(
                            symbol=symbol,
                            model_type="xgboost",
                            model=model,
                            performance=accuracy,
                            version="2.0",
                            params={"n_estimators": 100, "max_depth": 3, "features": available_features},
                            timeframe=timeframe
                        )
                        
                        logger.info(f"Đã huấn luyện và lưu mô hình XGBoost cho {symbol} ({timeframe}) với độ chính xác {accuracy:.2f}%")
                    
                    except Exception as e:
                        logger.error(f"Lỗi khi huấn luyện mô hình XGBoost cho {symbol} ({timeframe}): {str(e)}")
                
                logger.info(f"Hoàn tất huấn luyện mô hình cho {symbol}")
            
            except Exception as e:
                logger.error(f"Lỗi khi huấn luyện cho {symbol}: {str(e)}")
        
        logger.info("Hoàn tất huấn luyện tự động các mô hình")
    
    except Exception as e:
        logger.error(f"Lỗi trong quá trình auto_train_models: {str(e)}")

# ---------- AI VÀ BÁO CÁO ----------
class AIAnalyzer:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-pro')
        self.db_manager = DBManager()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_content(self, prompt):
        try:
            response = await run_in_thread(lambda: self.model.generate_content(prompt))
            return response.text
        except Exception as e:
            logger.error(f"Lỗi Gemini API: {str(e)}")
            # Thử sử dụng Groq khi Gemini thất bại
            try:
                return await self.generate_with_groq(prompt)
            except Exception as groq_error:
                logger.error(f"Fallback sang Groq cũng thất bại: {str(groq_error)}")
                return f"Không thể tạo phân tích do lỗi API. Chi tiết: {str(e)}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_with_groq(self, prompt: str) -> str:
        """
        Tạo nội dung sử dụng Groq API khi Gemini gặp vấn đề.
        """
        if not GROQ_API_KEY:
            raise ValueError("Không có GROQ_API_KEY trong cấu hình")
            
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": GROQ_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "Bạn là trợ lý phân tích chứng khoán chuyên nghiệp. Hãy cung cấp phân tích chi tiết, khách quan với ngôn ngữ rõ ràng, chuyên nghiệp. Phân tích của bạn cần dựa trên dữ liệu được cung cấp."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 2048
            }
            
            try:
                async with session.post(GROQ_API_URL, json=payload, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Groq API trả về lỗi {response.status}: {error_text}")
                        
                    data = await response.json()
                    
                    if not data.get("choices") or len(data["choices"]) == 0:
                        raise ValueError("Groq API trả về kết quả không hợp lệ")
                        
                    return data["choices"][0]["message"]["content"]
            except aiohttp.ClientError as e:
                raise ValueError(f"Lỗi kết nối đến Groq API: {str(e)}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_with_groq(self, technical_data):
        """
        Phân tích dữ liệu kỹ thuật bằng Groq API.
        
        Args:
            technical_data: Dữ liệu kỹ thuật cần phân tích
            
        Returns:
            Dict chứa kết quả phân tích
        """
        # Chuyển dữ liệu thành JSON an toàn (xử lý các kiểu dữ liệu NumPy, pandas...)
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (pd.Timestamp, datetime)):
                    return obj.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, pd.Series):
                    return obj.to_dict()
                elif isinstance(obj, pd.DataFrame):
                    return obj.to_dict(orient='records')
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(CustomJSONEncoder, self).default(obj)
                
        # Xây dựng prompt
        try:
            # Chuẩn bị dữ liệu để phân tích
            symbol = technical_data.get('symbol', 'Không rõ')
            timeframe = technical_data.get('timeframe', '1D')
            
            # Lấy thông tin giá hiện tại
            current_data = technical_data.get('current_data', {})
            close = current_data.get('close', 'N/A')
            
            # Phân tích kỹ thuật
            indicators = technical_data.get('indicators', {})
            
            # Dự báo
            forecast = technical_data.get('forecast', {})
            
            # Mẫu hình & sóng giá
            price_patterns = technical_data.get('price_patterns', [])
            waves = technical_data.get('waves', {})
            
            # Hỗ trợ kháng cự
            support_resistance = technical_data.get('support_resistance', {})
            
            # Cấu trúc prompt
            prompt = f"""
            # YÊU CẦU PHÂN TÍCH CHỨNG KHOÁN TOÀN DIỆN
            
            Hãy phân tích mã chứng khoán {symbol} với khung thời gian {timeframe} dựa trên dữ liệu sau:
            
            ## DỮ LIỆU HIỆN TẠI
            - Giá hiện tại: {close}
            
            ## CHỈ BÁO KỸ THUẬT
            {json.dumps(indicators, indent=2, cls=CustomJSONEncoder)}
            
            ## MẪU HÌNH & SÓNG GIÁ
            {json.dumps(price_patterns, indent=2, cls=CustomJSONEncoder)}
            {json.dumps(waves, indent=2, cls=CustomJSONEncoder)}
            
            ## HỖ TRỢ & KHÁNG CỰ
            {json.dumps(support_resistance, indent=2, cls=CustomJSONEncoder)}
            
            ## DỰ BÁO
            {json.dumps(forecast, indent=2, cls=CustomJSONEncoder)}
            
            # YÊU CẦU PHÂN TÍCH
            
            Hãy phân tích toàn diện dựa trên dữ liệu trên và trả về phân tích theo cấu trúc JSON với các trường như sau:
            - summary: Tóm tắt ngắn gọn tình hình hiện tại
            - trend_analysis: Phân tích xu hướng
            - indicator_signals: Tín hiệu từ các chỉ báo
            - support_resistance_analysis: Phân tích vùng giá
            - pattern_analysis: Phân tích mẫu hình kỹ thuật
            - forecast_analysis: Nhận định dự báo
            - recommendation: Khuyến nghị (Mua mạnh/Mua/Nắm giữ/Bán/Bán mạnh)
            - risk_level: Mức độ rủi ro (1-5)
            
            Yêu cầu: Cung cấp phân tích chi tiết nhưng ngắn gọn, dựa hoàn toàn vào dữ liệu đã cho.
            """
            
            # Gọi API Groq
            response_text = await self.generate_with_groq(prompt)
            
            # Tìm và trích xuất phần JSON trong phản hồi
            try:
                # Tìm các dấu ngoặc nhọn đầu tiên và cuối cùng
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_text = response_text[start_idx:end_idx+1]
                    result = json.loads(json_text)
                    return result
                else:
                    # Nếu không tìm thấy JSON, xử lý văn bản
                    return {
                        "summary": "Không thể phân tích dữ liệu dưới dạng JSON",
                        "analysis": response_text,
                        "recommendation": "Không có khuyến nghị"
                    }
            except json.JSONDecodeError:
                # Nếu không phân tích được JSON, trả về toàn bộ văn bản
                return {
                    "summary": "Không thể phân tích dữ liệu dưới dạng JSON",
                    "analysis": response_text,
                    "recommendation": "Không có khuyến nghị"
                }
                
        except Exception as e:
            logger.error(f"Lỗi khi phân tích với Groq: {str(e)}\n{traceback.format_exc()}")
            return {
                "summary": "Lỗi khi phân tích dữ liệu",
                "analysis": f"Đã xảy ra lỗi: {str(e)}",
                "recommendation": "Không có khuyến nghị do lỗi phân tích"
            }
    
    async def load_report_history(self, symbol: str) -> list:
        return await self.db_manager.load_report_history(symbol)
    
    async def save_report_history(self, symbol: str, report: str, close_today: float, close_yesterday: float) -> None:
        await self.db_manager.save_report_history(symbol, report, close_today, close_yesterday)
    
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
            f"\n\nGiá hiện tại: {float(df['close'].iloc[-1]):.2f}"
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
        # Chuẩn bị dữ liệu cho phân tích
        df = dfs.get(DEFAULT_TIMEFRAME, None)
        if df is None or df.empty:
            return f"Không có dữ liệu cho khung thời gian {DEFAULT_TIMEFRAME} của mã {symbol}"
        
        # Lấy dữ liệu giá hiện tại
        current_price = df['close'].iloc[-1]
        previous_price = df['close'].iloc[-2] if len(df) > 1 else None
        price_change = ((current_price - previous_price) / previous_price * 100) if previous_price else 0
        
        # Phân tích kỹ thuật
        ta_analyzer = TechnicalAnalyzer()
        indicators = {}
        
        for tf, data in dfs.items():
            if not data.empty:
                indicators[tf] = ta_analyzer.calculate_indicators(data)
        
        # Phân tích price action
        price_action_analysis = self.analyze_price_action(df)
        
        # Tính toán mức hỗ trợ/kháng cự
        support_resistance = self.calculate_support_resistance_levels(df)
        
        # Dự báo giá
        try:
            predictor = EnhancedPredictor()
            forecast_data = predictor.hybrid_predict(df)
        except Exception as e:
            logger.error(f"Lỗi dự báo giá: {str(e)}")
            forecast_data = {"error": f"Không thể dự báo giá: {str(e)}"}
        
        # Lấy tin tức
        news = await get_news(symbol, 5)
        
        # Lấy lịch sử báo cáo
        report_history = await self.load_report_history(symbol)
        
        # Phân tích OpenRouter
        is_index_flag = is_index(symbol)
        
        # Dữ liệu kỹ thuật để truyền vào AI
        technical_data = {
            "symbol": symbol,
            "timeframe": DEFAULT_TIMEFRAME,
            "is_index": is_index_flag,
            "current_data": {
                "date": df.index[-1].strftime('%Y-%m-%d') if not df.empty else "N/A",
                "open": float(df['open'].iloc[-1]) if not df.empty else None,
                "high": float(df['high'].iloc[-1]) if not df.empty else None,
                "low": float(df['low'].iloc[-1]) if not df.empty else None,
                "close": float(df['close'].iloc[-1]) if not df.empty else None,
                "volume": int(df['volume'].iloc[-1]) if not df.empty and 'volume' in df else None,
                "previous_close": float(df['close'].iloc[-2]) if len(df) > 1 else None,
                "price_change": price_change
            },
            "indicators": indicators.get(DEFAULT_TIMEFRAME, {}),
            "multi_timeframe_indicators": indicators,
            "price_action": price_action_analysis,
            "support_resistance": support_resistance,
            "forecast": forecast_data,
            "outlier_reports": outlier_reports
        }
        
        # Tìm mẫu hình kỹ thuật và phân tích sóng
        try:
            ai_analysis = await self.analyze_with_openrouter(technical_data)
            technical_data.update(ai_analysis)
        except Exception as e:
            logger.error(f"Lỗi khi phân tích với OpenRouter: {str(e)}")
            # Fallback sang Groq khi OpenRouter thất bại
            try:
                ai_analysis = await self.analyze_with_groq(technical_data)
                technical_data.update(ai_analysis)
            except Exception as groq_error:
                logger.error(f"Fallback sang Groq cũng thất bại: {str(groq_error)}")
                technical_data["error"] = f"Không thể phân tích mẫu hình: {str(e)}"
        
        # Tạo prompt cho Gemini
        if is_index_flag:
            prompt_template = """
            # PHÂN TÍCH CHỈ SỐ CHỨNG KHOÁN: {symbol}

            ## TỔNG QUAN
            - Chỉ số hiện tại: {current_price} ({price_change:+.2f}%)
            - Khung thời gian: {timeframe}
            
            ## PHÂN TÍCH KỸ THUẬT
            {technical_indicators}
            
            ## PHÂN TÍCH ĐA KHUNG THỜI GIAN
            {multi_timeframe_analysis}
            
            ## MẪU HÌNH & SÓNG GIÁ
            {patterns_waves}
            
            ## HỖ TRỢ & KHÁNG CỰ
            {support_resistance}
            
            ## DỰ BÁO
            {forecast}
            
            ## TIN TỨC LIÊN QUAN
            {news}

            ## NHẬN XÉT KỸ THUẬT
            {openrouter_analysis}
            
            ## CẢNH BÁO DỮ LIỆU
            {data_warnings}

            ## YÊU CẦU
            Dựa vào dữ liệu trên, hãy viết một báo cáo phân tích thị trường dành cho nhà đầu tư. Phân tích này nên bao gồm:
            1. Nhận định tổng quan về xu hướng hiện tại của chỉ số
            2. Phân tích chi tiết các chỉ báo kỹ thuật, nhận định về động lượng, xu hướng
            3. So sánh phân tích ở các khung thời gian khác nhau để có cái nhìn toàn diện
            4. Đánh giá về các mẫu hình kỹ thuật (nếu có) và phân tích sóng
            5. Dự báo trong thời gian tới, kèm các mức hỗ trợ và kháng cự quan trọng
            6. Đưa ra nhận định tổng thể về thị trường và đề xuất chiến lược phù hợp
            
            Viết ngắn gọn, súc tích, trình bày theo cấu trúc rõ ràng, dễ đọc nhưng vẫn đảm bảo đầy đủ thông tin.
            """
        else:
            prompt_template = """
            # PHÂN TÍCH CỔ PHIẾU: {symbol}

            ## TỔNG QUAN
            - Giá hiện tại: {current_price} ({price_change:+.2f}%)
            - Khung thời gian: {timeframe}
            
            ## THÔNG TIN CÔNG TY
            {fundamental_info}
            
            ## PHÂN TÍCH KỸ THUẬT
            {technical_indicators}
            
            ## PHÂN TÍCH ĐA KHUNG THỜI GIAN
            {multi_timeframe_analysis}
            
            ## MẪU HÌNH & SÓNG GIÁ
            {patterns_waves}
            
            ## HỖ TRỢ & KHÁNG CỰ
            {support_resistance}
            
            ## DỰ BÁO
            {forecast}
            
            ## TIN TỨC LIÊN QUAN
            {news}
            
            ## BÁO CÁO TRƯỚC ĐÓ
            {previous_report}

            ## NHẬN XÉT KỸ THUẬT
            {openrouter_analysis}

            ## CẢNH BÁO DỮ LIỆU
            {data_warnings}

            ## YÊU CẦU
            Dựa vào dữ liệu trên, hãy viết một báo cáo phân tích cổ phiếu dành cho nhà đầu tư. Phân tích này nên bao gồm:
            1. Nhận định tổng quan về tình hình và xu hướng hiện tại
            2. Phân tích cơ bản ngắn gọn về công ty (nếu có dữ liệu)
            3. Phân tích chi tiết các chỉ báo kỹ thuật ở khung thời gian chính
            4. So sánh phân tích ở các khung thời gian khác nhau để có cái nhìn toàn diện
            5. Đánh giá về các mẫu hình kỹ thuật (nếu có) và phân tích sóng
            6. Dự báo trong thời gian tới, kèm các mức hỗ trợ và kháng cự quan trọng
            7. Đưa ra khuyến nghị cụ thể (Mua/Bán/Nắm giữ) kèm lý do
            
            Viết ngắn gọn, súc tích, trình bày theo cấu trúc rõ ràng, dễ đọc nhưng vẫn đảm bảo đầy đủ thông tin.
            """
        
        # Chuẩn bị các thành phần cho prompt
        technical_indicators = "N/A"
        main_indicators = indicators.get(DEFAULT_TIMEFRAME, {})
        if main_indicators:
            technical_indicators = f"""
            - RSI: {main_indicators.get('rsi', 'N/A'):.2f}
            - MACD: {main_indicators.get('macd', 'N/A'):.2f} (Signal: {main_indicators.get('macd_signal', 'N/A'):.2f})
            - SMA20: {main_indicators.get('sma20', 'N/A'):.2f}
            - SMA50: {main_indicators.get('sma50', 'N/A'):.2f}
            - SMA200: {main_indicators.get('sma200', 'N/A'):.2f}
            - Bollinger Bands: Upper={main_indicators.get('bollinger_hband', 'N/A'):.2f}, Lower={main_indicators.get('bollinger_lband', 'N/A'):.2f}
            """
        
        # Phân tích đa khung thời gian
        multi_timeframe_analysis = "Không có dữ liệu đa khung thời gian"
        if len(indicators) > 1:
            multi_timeframe_analysis = ""
            for tf, tf_indicators in indicators.items():
                if tf == DEFAULT_TIMEFRAME or not tf_indicators:
                    continue
                
                multi_timeframe_analysis += f"### Khung thời gian {tf}:\n"
                multi_timeframe_analysis += f"- RSI: {tf_indicators.get('rsi', 'N/A'):.2f}\n"
                multi_timeframe_analysis += f"- MACD: {tf_indicators.get('macd', 'N/A'):.2f} (Signal: {tf_indicators.get('macd_signal', 'N/A'):.2f})\n"
                multi_timeframe_analysis += f"- SMA20: {tf_indicators.get('sma20', 'N/A'):.2f}\n"
                
                # Thêm thông tin hỗ trợ/kháng cự
                if dfs.get(tf) is not None and not dfs[tf].empty:
                    try:
                        tf_support_resistance = self.calculate_support_resistance_levels(dfs[tf])
                        if tf_support_resistance and "support_levels" in tf_support_resistance:
                            support_levels = tf_support_resistance["support_levels"]
                            resistance_levels = tf_support_resistance["resistance_levels"]
                            
                            if support_levels:
                                multi_timeframe_analysis += f"- Hỗ trợ: {', '.join([f'{level:.2f}' for level in support_levels[:2]])}\n"
                            if resistance_levels:
                                multi_timeframe_analysis += f"- Kháng cự: {', '.join([f'{level:.2f}' for level in resistance_levels[:2]])}\n"
                    except Exception as e:
                        logger.error(f"Lỗi khi phân tích hỗ trợ/kháng cự cho {tf}: {str(e)}")
        
        fundamental_info = "Không có dữ liệu cơ bản"
        if fundamental_data and not is_index_flag:
            try:
                validated_data, observations = DataValidator.validate_fundamental_data(fundamental_data)
                pe = validated_data.get('trailingPE', 'N/A')
                pb = validated_data.get('priceToBook', 'N/A')
                market_cap = validated_data.get('marketCap', 'N/A')
                eps = validated_data.get('trailingEps', 'N/A')
                dividend_yield = validated_data.get('dividendYield', 'N/A')
                beta = validated_data.get('beta', 'N/A')
                
                # Định dạng các giá trị
                if market_cap != 'N/A':
                    market_cap = f"{market_cap:,.0f}"
                if dividend_yield != 'N/A' and dividend_yield is not None:
                    if isinstance(dividend_yield, (float, int)) and dividend_yield < 1:  # Nếu là tỷ lệ phần trăm đang ở dạng thập phân
                        dividend_yield = f"{dividend_yield * 100:.2f}%"
                    else:
                        dividend_yield = f"{dividend_yield:.2f}%"
                
                fundamental_info = f"""
                - P/E: {pe}
                - P/B: {pb}
                - EPS: {eps}
                - Vốn hóa: {market_cap}
                - Tỷ suất cổ tức: {dividend_yield}
                - Beta: {beta}
                """
                
                # Thêm các quan sát đáng chú ý
                if observations:
                    fundamental_info += "\n### CHÚ Ý:\n"
                    fundamental_info += "\n".join(f"- {obs}" for obs in observations)
            except Exception as e:
                logger.error(f"Lỗi xử lý dữ liệu cơ bản: {str(e)}")
                fundamental_info = f"Lỗi khi xử lý dữ liệu cơ bản: {str(e)}"
        
        patterns_waves = "Không phát hiện mẫu hình đặc biệt"
        if "price_patterns" in technical_data and technical_data["price_patterns"]:
            patterns = technical_data["price_patterns"]
            if isinstance(patterns, list) and patterns:
                patterns_waves = "\n".join([f"- {pattern}" for pattern in patterns])
            elif isinstance(patterns, str):
                patterns_waves = patterns
        
        support_resistance_text = "Không có dữ liệu"
        if support_resistance and "support_levels" in support_resistance and "resistance_levels" in support_resistance:
            support_levels = support_resistance["support_levels"]
            resistance_levels = support_resistance["resistance_levels"]
            
            support_text = "Không có" if not support_levels else ", ".join([f"{level:.2f}" for level in support_levels])
            resistance_text = "Không có" if not resistance_levels else ", ".join([f"{level:.2f}" for level in resistance_levels])
            
            support_resistance_text = f"- Kháng cự: {resistance_text}\n- Hỗ trợ: {support_text}"
        
        forecast_text = "Không có dữ liệu dự báo"
        if "error" not in forecast_data:
            try:
                last_price = forecast_data.get('last_price', current_price)
                signal = forecast_data.get('signal', 'Không xác định')
                signal_confidence = forecast_data.get('signal_confidence', 0)
                probability_up = forecast_data.get('probability_up', 0)
                probability_down = forecast_data.get('probability_down', 0)
                overall_trend = forecast_data.get('overall_trend', 'Không xác định')
                trend_strength = forecast_data.get('trend_strength', 0)
                
                forecast_text = f"""
                - Tín hiệu: {signal} (độ tin cậy: {signal_confidence:.1f}%)
                - Xu hướng tổng thể: {overall_trend.capitalize()} (độ mạnh: {trend_strength:.1f}%)
                - Xác suất tăng: {probability_up:.1f}%
                - Xác suất giảm: {probability_down:.1f}%
                """
                
                # Thêm giá dự báo nếu có
                if 'forecast_prices' in forecast_data and 'forecast_dates' in forecast_data:
                    prices = forecast_data['forecast_prices']
                    dates = forecast_data['forecast_dates']
                    
                    if prices and dates and len(prices) == len(dates):
                        forecast_text += "\nGiá dự báo:\n"
                        for i in range(min(5, len(prices))):
                            date_str = dates[i].strftime('%Y-%m-%d') if isinstance(dates[i], (datetime, pd.Timestamp)) else dates[i]
                            price = prices[i]
                            change = (price - last_price) / last_price * 100
                            forecast_text += f"- {date_str}: {price:.2f} ({change:+.2f}%)\n"
            except Exception as e:
                logger.error(f"Lỗi định dạng dữ liệu dự báo: {str(e)}")
                forecast_text = f"Lỗi định dạng dữ liệu dự báo: {str(e)}"
        
        news_text = "Không có tin tức liên quan"
        if news:
            news_text = ""
            for item in news:
                date = item.get('date', '')
                title = item.get('title', '')
                link = item.get('link', '')
                desc = item.get('description', '')
                
                news_text += f"- {date}: {title}\n"
                if desc:
                    news_text += f"  {desc}\n"
                news_text += f"  Link: {link}\n\n"
        
        previous_report = "Chưa có báo cáo trước đó"
        if report_history and len(report_history) > 0:
            last_report = report_history[0]
            report_date = last_report.get('date', 'N/A')
            report_text = last_report.get('report', 'N/A')
            
            if report_text:
                # Lấy 300 ký tự đầu tiên nhưng không cắt giữa từ
                if len(report_text) > 300:
                    short_text = report_text[:300].rsplit(' ', 1)[0] + '...'
                else:
                    short_text = report_text
                
                previous_report = f"""Báo cáo gần nhất ({report_date}):\n{short_text}"""
        
        openrouter_analysis = "Không có phân tích mẫu hình"
        if "summary" in technical_data:
            openrouter_analysis = technical_data["summary"]
            if "trend_analysis" in technical_data:
                openrouter_analysis += f"\n\nXu hướng: {technical_data['trend_analysis']}"
            if "pattern_analysis" in technical_data:
                openrouter_analysis += f"\n\nMẫu hình: {technical_data['pattern_analysis']}"
            if "recommendation" in technical_data:
                openrouter_analysis += f"\n\nKhuyến nghị: {technical_data['recommendation']}"
        
        data_warnings = "Không có cảnh báo dữ liệu"
        if outlier_reports:
            warnings = []
            for tf, report in outlier_reports.items():
                if report and "Không phát hiện outlier" not in report:
                    warnings.append(f"Khung thời gian {tf}: {report}")
            
            if warnings:
                data_warnings = "\n".join(warnings)
        
        # Điền thông tin vào template
        prompt = prompt_template.format(
            symbol=symbol,
            current_price=current_price,
            price_change=price_change,
            timeframe=DEFAULT_TIMEFRAME,
            technical_indicators=technical_indicators,
            multi_timeframe_analysis=multi_timeframe_analysis,
            fundamental_info=fundamental_info,
            patterns_waves=patterns_waves,
            support_resistance=support_resistance_text,
            forecast=forecast_text,
            news=news_text,
            previous_report=previous_report,
            openrouter_analysis=openrouter_analysis,
            data_warnings=data_warnings
        )
        
        try:
            # Gọi model để tạo báo cáo
            report = await self.generate_content(prompt)
            
            # Lưu báo cáo vào lịch sử
            await self.save_report_history(
                symbol=symbol,
                report=report,
                close_today=current_price,
                close_yesterday=previous_price if previous_price else current_price
            )
            
            # Lưu report vào cache
            await redis_manager.set(f"report_{symbol}_{DEFAULT_TIMEFRAME}_{DEFAULT_CANDLES}", 
                                   report, expire=CACHE_EXPIRE_SHORT)
            
            return report
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo: {str(e)}")
            return f"Lỗi khi tạo báo cáo: {str(e)}"

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
    # Kiểm tra xem người dùng có được phép sử dụng bot hay không
    user_id = update.message.from_user.id
    if not await is_user_approved(str(user_id)):
        await update.message.reply_text("⛔ Bạn chưa được cấp quyền sử dụng bot. Vui lòng liên hệ admin.")
        await notify_admin_new_user(update, context)
        return

    start_time = time_module.time()
    
    try:
        args = context.args
        if not args:
            await update.message.reply_text("⚠️ Sử dụng: /analyze <mã chứng khoán> [khung thời gian] [số nến]")
            return
    
        # Mã chứng khoán và các tham số
        symbol = args[0].upper()
        
        # Xác thực mã chứng khoán
        is_valid, message = DataValidator.validate_ticker(symbol)
        if not is_valid:
            await update.message.reply_text(f"⚠️ {message}")
            return
        
        # Xử lý các tham số tùy chọn
        timeframe = DEFAULT_TIMEFRAME
        num_candles = DEFAULT_CANDLES
        
        if len(args) >= 2:
            tf_valid, tf_info = DataValidator.validate_timeframe(args[1])
            if tf_valid:
                timeframe = tf_info
            else:
                try:
                    num_candles = int(args[1])
                    candles_valid, candles_info = DataValidator.validate_candle_count(num_candles)
                    if not candles_valid:
                        await update.message.reply_text(f"⚠️ {candles_info}")
                        return
                except ValueError:
                    await update.message.reply_text(f"⚠️ Khung thời gian không hợp lệ: {args[1]}")
                    return
        
        if len(args) >= 3:
            try:
                num_candles = int(args[2])
                candles_valid, candles_info = DataValidator.validate_candle_count(num_candles)
                if not candles_valid:
                    await update.message.reply_text(f"⚠️ {candles_info}")
                    return
            except ValueError:
                await update.message.reply_text(f"⚠️ Số nến không hợp lệ: {args[2]}")
                return
        
        # Thông báo cho người dùng rằng bot đang xử lý
        processing_message = await update.message.reply_text(
            f"⏳ Đang phân tích {symbol} với khung thời gian {timeframe}... Vui lòng đợi trong giây lát."
        )

        # Sử dụng pipeline chuẩn hóa
        pipeline = DataPipeline()
        
        # Xác định các khung thời gian cần phân tích
        timeframes_to_analyze = [timeframe]
        
        # Thêm các khung thời gian đa chiều
        if timeframe == '1D':
            timeframes_to_analyze.extend(['1W', '1M'])
        elif timeframe == '1h':
            timeframes_to_analyze.extend(['4h', '1D'])
        elif timeframe == '15m':
            timeframes_to_analyze.extend(['1h', '4h'])
        
        # Chuẩn bị dữ liệu
        data = await pipeline.prepare_symbol_data(symbol, timeframes=timeframes_to_analyze, num_candles=num_candles)
        
        if 'error' in data:
            await processing_message.edit_text(f"❌ Lỗi: {data['error']}")
            return
        
        # Kiểm tra dữ liệu
        if not data['dfs'] or timeframe not in data['dfs'] or data['dfs'][timeframe].empty:
            await processing_message.edit_text(f"❌ Không có dữ liệu cho {symbol} với khung thời gian {timeframe}")
            return
        
        # Lấy báo cáo outlier
        outlier_reports = {}
        for tf, df in data['dfs'].items():
            if not df.empty:
                _, report = DataValidator.handle_outliers(df, method='iqr', threshold=2.0, action='mark')
                outlier_reports[tf] = report
        
        # Tạo phân tích
        ai_analyzer = AIAnalyzer()
        
        # Đo thời gian tạo báo cáo
        report_start_time = time_module.time()
        report = await ai_analyzer.generate_report(data['dfs'], symbol, data['fundamental_data'], outlier_reports)
        report_duration = time_module.time() - report_start_time
        
        # Gửi báo cáo
        total_duration = time_module.time() - start_time
        
        # Thêm footer với thông tin thời gian xử lý
        footer = f"\n\n⏱️ Thời gian xử lý: {total_duration:.2f}s (phân tích AI: {report_duration:.2f}s)"
        
        # Thêm thông tin outlier nếu phát hiện
        outlier_warnings = []
        for tf, report_text in outlier_reports.items():
            if report_text and "Không phát hiện outlier" not in report_text:
                outlier_warnings.append(f"⚠️ {tf}: Phát hiện dữ liệu bất thường")
        
        if outlier_warnings:
            footer += "\n" + "\n".join(outlier_warnings)
            
        # Thêm thông tin phiên bản bot
        footer += "\n\nBot Chứng Khoán V20.0"
        
        # Gửi báo cáo hoàn chỉnh
        await processing_message.edit_text(report + footer)
        
        # Lưu vào cache
        try:
            await redis_manager.set(f"report_{symbol}_{timeframe}_{num_candles}", 
                                   report, expire=CACHE_EXPIRE_SHORT)
        except Exception as e:
            logger.error(f"Lỗi khi lưu report vào cache: {str(e)}")
        
    except Exception as e:
        logger.error(f"Lỗi khi phân tích {symbol}: {str(e)}\n{traceback.format_exc()}")
        await update.message.reply_text(f"❌ Lỗi khi phân tích: {str(e)}")

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

    # Khởi tạo Redis và Database
    global redis_manager, db_manager, model_db_manager
    redis_manager = RedisManager()
    db_manager = DBManager()
    model_db_manager = ModelDBManager()

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
            # Hiện tại thiếu try/except và đóng kết nối
            try:
                # Xử lý webhook
                ...
            except Exception as e:
                logger.error(f"Lỗi webhook: {str(e)}")
                return web.Response(status=500)
        
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
        await shutdown_event.wait()
        
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập webhook: {str(e)}")
        # Quay lại chế độ polling nếu webhook thất bại
        logger.info("Chuyển sang chế độ polling...")
        await app.run_polling()
    finally:
        # Đóng kết nối
        try:
            await redis_manager.close()
        except:
            pass
        try:
            await db_manager.close()
        except:
            pass

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