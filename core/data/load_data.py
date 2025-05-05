#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module for loading stock data with improved async handling
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
import time
import io
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import pytz
import requests
from dotenv import load_dotenv
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
import random

# Import DataValidator từ module data_validator thay vì định nghĩa lại
from core.data.data_validator import DataValidator

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set timezone for Vietnam
TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# Check if vnstock is available
try:
    import vnstock
    from vnstock import Vnstock
    VNSTOCK_AVAILABLE = True
except ImportError:
    VNSTOCK_AVAILABLE = False
    logger.warning("vnstock package not available. VN stock data will be limited.")

# Cache để lưu trữ dữ liệu tạm thời
cache = {}

class CacheManager:
    """
    Manages a cache that expires after a certain time
    """
    
    def __init__(self):
        """
        Initialize the cache with a TTL (time to live) of 12 hours
        """
        self.cache = TTLCache(maxsize=1000, ttl=12*3600)  # 12 hours TTL
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        try:
            if key in self.cache:
                logger.debug(f"Cache hit for {key}")
                return self.cache[key]
            else:
                logger.debug(f"Cache miss for {key}")
                return default
        except Exception as e:
            logger.warning(f"Error accessing cache for key {key}: {str(e)}")
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        try:
            self.cache[key] = value
            logger.debug(f"Cached value for {key}")
        except Exception as e:
            logger.warning(f"Error setting cache for key {key}: {str(e)}")
    
    def clear(self) -> None:
        """
        Clear the cache
        """
        try:
            self.cache.clear()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing cache: {str(e)}")

# Sử dụng DataValidator từ module data_validator

class DataLoader:
    """
    Lớp tải và xử lý dữ liệu chứng khoán
    """
    
    def __init__(self, source: str = 'vnstock'):
        """Khởi tạo với nguồn dữ liệu"""
        self.source = source
        self.validator = DataValidator()
        self.cache = CacheManager()
        logger.info(f"DataLoader được khởi tạo với nguồn: {source}")
        
    def detect_outliers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """Phát hiện giá trị ngoại lệ trong dữ liệu"""
        return self.validator.detect_and_handle_outliers(df)
        
    def clean_data(self, df: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, str]:
        """
        Làm sạch dữ liệu chứng khoán (sync version)
        Args:
            df: DataFrame cần làm sạch
            symbol: Mã chứng khoán
        Returns:
            DataFrame đã làm sạch và message
        """
        if df.empty:
            return df, "DataFrame rỗng"
        # Chuẩn hóa dataframe
        df = self.validator.normalize_dataframe(df)
        # Phát hiện và xử lý outliers
        df_clean, message = self.validator.detect_and_handle_outliers(df)
        logger.info(f"Dữ liệu {symbol} sau khi làm sạch: {len(df_clean)} dòng. {message}")
        return df_clean, message

    def load_raw_data(self, symbol: str, timeframe: str, num_candles: int) -> pd.DataFrame:
        """
        Tải dữ liệu chứng khoán raw từ nguồn được chọn (sync version)
        """
        symbol = self.validator.validate_ticker(symbol)
        timeframe = self.validator.normalize_timeframe(timeframe)
        num_candles = self.validator.validate_candles(num_candles)
        end_date = datetime.now()
        tf_map = {
            '5m': pd.DateOffset(minutes=5),
            '15m': pd.DateOffset(minutes=15),
            '30m': pd.DateOffset(minutes=30),
            '1h': pd.DateOffset(hours=1),
            '4h': pd.DateOffset(hours=4),
            '1D': pd.DateOffset(days=1),
            '1W': pd.DateOffset(weeks=1),
            '1M': pd.DateOffset(months=1)
        }
        offset = tf_map.get(timeframe, pd.DateOffset(days=1))
        start_date = end_date - (offset * num_candles)
        try:
            logger.info(f"Tải dữ liệu {symbol} từ {start_date} đến {end_date} với khung thời gian {timeframe}")
            if self.source.lower() == 'vnstock' and VNSTOCK_AVAILABLE:
                # Sử dụng API sync nếu có, nếu không fallback yfinance
                try:
                    start_str = start_date.strftime('%Y-%m-%d')
                    end_str = end_date.strftime('%Y-%m-%d')
                    stock = Vnstock().stock(symbol=symbol, source='VCI')
                    df = stock.quote.history(start=start_str, end=end_str, interval=timeframe)
                    if df is None or df.empty:
                        logger.warning(f"Không có dữ liệu trả về từ vnstock cho {symbol}")
                        return pd.DataFrame()
                    df = DataValidator.normalize_dataframe(df)
                    df = DataValidator.validate_schema(df)
                    logger.info(f"[DEBUG] System now: {datetime.now()} | Last data index: {df.index[-1] if not df.empty else 'N/A'}")
                    logger.info(f"Đã tải {len(df)} dòng dữ liệu từ vnstock cho {symbol}")
                    return df
                except Exception as e:
                    logger.error(f"Lỗi khi tải dữ liệu từ vnstock cho {symbol}: {str(e)}")
                    # Fallback yfinance
            return self._load_from_yfinance(symbol, timeframe, num_candles)
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu {symbol}: {str(e)}")
            return pd.DataFrame()

    def load_data(self, symbol: str, timeframe: str, num_candles: int) -> tuple:
        """
        Tải và xử lý dữ liệu toàn diện (sync version)
        """
        raw_df = self.load_raw_data(symbol, timeframe, num_candles)
        cleaned_df, outlier_report = self.clean_data(raw_df, symbol)
        return raw_df, cleaned_df, outlier_report

    def load_and_clean_data(self, symbol: str, timeframe: str, num_candles: int) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
        """
        Wrapper sync hoàn toàn cho pipeline
        """
        try:
            raw_df, cleaned_df, outlier_report = self.load_data(symbol, timeframe, num_candles)
            if not cleaned_df.empty:
                try:
                    start_date = cleaned_df.index[0].date()
                    end_date = cleaned_df.index[-1].date()
                    trading_days = DataValidator.calculate_trading_days(start_date, end_date)
                    actual_days = len(cleaned_df)
                    expected_days = len(trading_days)
                    if expected_days > 0 and actual_days < 0.9 * expected_days:
                        logger.warning(f"Dữ liệu {symbol} chỉ có {actual_days}/{expected_days} phiên giao dịch thực tế (<90%). Có thể thiếu phiên hoặc dữ liệu không liên tục.")
                except Exception as date_ex:
                    logger.warning(f"Lỗi khi kiểm tra số phiên giao dịch cho {symbol}: {date_ex}")
            return raw_df, cleaned_df, outlier_report
        except Exception as e:
            logger.error(f"Lỗi trong quá trình load_and_clean_data: {e}")
            return pd.DataFrame(), pd.DataFrame(), f"Lỗi tải dữ liệu: {str(e)}"

    def get_price_data(self, symbol: str, period: str = "6mo", interval: str = "1d", use_cache: bool = True) -> Dict[str, Any]:
        """
        Lấy dữ liệu giá chứng khoán
        
        Args:
            symbol: Mã chứng khoán
            period: Khoảng thời gian (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, ytd, max)
            interval: Tần suất dữ liệu (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Sử dụng dữ liệu cache nếu có
            
        Returns:
            Dict chứa dữ liệu giá
        """
        # Xử lý symbol
        if not symbol:
            logger.error("Mã chứng khoán không hợp lệ")
            return {"error": "Mã chứng khoán không hợp lệ"}
        
        # Chuẩn hóa symbol
        try:
            symbol = self.validator.validate_ticker(symbol)
        except Exception as e:
            logger.warning(f"Lỗi khi chuẩn hóa mã {symbol}: {e}")
            # Tiếp tục sử dụng symbol gốc

        # Tạo khóa cache
        cache_key = f"price_data_{symbol}_{period}_{interval}"
        
        # Kiểm tra cache
        if use_cache:
            try:
                if cache_key in cache:
                    cached_data = cache[cache_key]
                    # Kiểm tra thời gian cache để quyết định có sử dụng hay không
                    if "last_updated" in cached_data:
                        last_updated = datetime.fromisoformat(cached_data["last_updated"].replace('Z', '+00:00'))
                        cache_age = (datetime.now(TZ) - last_updated).total_seconds()
                        
                        # Sử dụng cache nếu chưa quá cũ (dưới 30 phút)
                        if cache_age < 1800:  # 30 phút
                            logger.info(f"Sử dụng dữ liệu cache cho {symbol} (cache {cache_age:.0f}s cũ)")
                            return cached_data
                        else:
                            logger.info(f"Cache quá cũ ({cache_age:.0f}s) cho {symbol}, tải lại dữ liệu mới")
                    else:
                        logger.info(f"Sử dụng dữ liệu cache cho {symbol}")
                        return cached_data
            except Exception as cache_error:
                logger.warning(f"Lỗi khi kiểm tra cache: {cache_error}")
                # Tiếp tục tải dữ liệu mới
        
        try:
            # Chuẩn hóa tham số
            timeframe = self.validator.normalize_timeframe(interval)
            
            # Tính số lượng nến dựa trên period
            period_to_candles = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90, 
                "6mo": 180, "1y": 365, "2y": 730, "5y": 1825,
                "ytd": (datetime.now() - datetime(datetime.now().year, 1, 1)).days,
                "max": 3650  # 10 năm
            }
            num_candles = period_to_candles.get(period, 180)
            
            # Tải và làm sạch dữ liệu
            raw_df, cleaned_df, outlier_report = self.load_and_clean_data(symbol, timeframe, num_candles)
            
            # CHUẨN HÓA DỮ LIỆU ĐẦU RA
            cleaned_df = self.validator.normalize_dataframe(cleaned_df)
            cleaned_df = self.validator.validate_schema(cleaned_df)
            
            # Tạo kết quả trả về
            price_data = {
                "symbol": symbol,
                "source": self.source,
                "period": period,
                "interval": interval,
                "last_updated": datetime.now(TZ).isoformat(),
                "currency": "VND" if symbol.upper() in self.validator.INDICES or not symbol.endswith(".VN") else "USD",
                "price_history": [],
                "outlier_report": outlier_report
            }
            
            if not cleaned_df.empty:
                # Chuyển DataFrame thành list
                try:
                    for date, row in cleaned_df.iterrows():
                        price_data["price_history"].append({
                            "date": date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(date, pd.Timestamp) else str(date),
                            "open": float(row["open"]),
                            "high": float(row["high"]),
                            "low": float(row["low"]),
                            "close": float(row["close"]),
                            "volume": int(row["volume"]),
                            "adj_close": float(row.get("adj_close", row["close"]))
                        })
                except Exception as convert_error:
                    logger.error(f"Lỗi khi chuyển đổi dữ liệu: {convert_error}")
                    # Trong trường hợp lỗi, thử phương pháp khác
                    try:
                        for i in range(len(cleaned_df)):
                            row = cleaned_df.iloc[i]
                            date = cleaned_df.index[i]
                            price_data["price_history"].append({
                                "date": date.strftime('%Y-%m-%d %H:%M:%S') if isinstance(date, pd.Timestamp) else str(date),
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": int(row["volume"]),
                                "adj_close": float(row.get("adj_close", row["close"]))
                            })
                    except Exception as fallback_error:
                        logger.error(f"Lỗi khi chuyển đổi dữ liệu (phương pháp thay thế): {fallback_error}")
                
                # Tính toán thông tin hiện tại
                if price_data["price_history"]:
                    current = price_data["price_history"][-1]
                    previous = price_data["price_history"][-2] if len(price_data["price_history"]) > 1 else current
                    
                    price_data["current"] = {
                        "price": current["close"],
                        "change": current["close"] - previous["close"],
                        "change_percent": ((current["close"] - previous["close"]) / previous["close"]) * 100 if previous["close"] > 0 else 0,
                        "volume": current["volume"],
                        "date": current["date"]
                    }
            
            # Lưu vào cache
            try:
                cache[cache_key] = price_data
            except Exception as cache_error:
                logger.warning(f"Lỗi khi lưu cache: {cache_error}")
            
            return price_data
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu giá cho {symbol}: {e}")
            return {"error": str(e), "symbol": symbol}
    
    def load_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Tải dữ liệu chứng khoán và chuyển đổi thành DataFrame
        
        Args:
            symbol: Mã chứng khoán
            period: Khoảng thời gian (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Chu kỳ (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame chứa dữ liệu chứng khoán
        """
        try:
            # Gọi get_price_data
            price_data = self.get_price_data(symbol, period, interval)
            
            if "error" in price_data:
                logger.error(f"Lỗi khi tải dữ liệu cho {symbol}: {price_data['error']}")
                return pd.DataFrame()
            
            # Chuyển đổi dữ liệu sang DataFrame
            price_history = price_data.get("price_history", [])
            if not price_history:
                logger.warning(f"Không có dữ liệu lịch sử giá cho {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(price_history)
            
            # Chuyển đổi cột ngày thành index
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            # CHUẨN HÓA TIMEZONE: luôn ép về Asia/Ho_Chi_Minh (tz-aware)
            tz = pytz.timezone('Asia/Ho_Chi_Minh')
            if isinstance(df.index, pd.DatetimeIndex):
                if df.index.tz is None:
                    df.index = df.index.tz_localize(tz)
                else:
                    df.index = df.index.tz_convert(tz)
            # Đảm bảo có đủ các cột cần thiết
            required_columns = ["open", "high", "low", "close", "volume"]
            for col in required_columns:
                if col not in df.columns:
                    logger.warning(f"Cột {col} không có trong dữ liệu {symbol}")
                    df[col] = 0
            return df
            
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu chứng khoán cho {symbol}: {e}")
            return pd.DataFrame()
    
    def clear_cache(self):
        """Xóa tất cả cache"""
        global cache
        cache = {}
        logger.info("Đã xóa toàn bộ cache")
        
    def get_popular_stocks(self) -> List[str]:
        """Trả về danh sách các mã chứng khoán phổ biến"""
        return [
            'VNM', 'VHM', 'VIC', 'VCB', 'FPT', 'HPG', 
            'MSN', 'MWG', 'VRE', 'BID', 'CTG', 'GAS'
        ]
        
    def get_vn30_stocks(self) -> List[str]:
        """Trả về danh sách các mã trong chỉ số VN30"""
        vn30_stocks = [
            'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 
            'GVR', 'HDB', 'HPG', 'KDH', 'MBB', 'MSN', 'MWG', 
            'NVL', 'PDR', 'PLX', 'POW', 'SAB', 'SSI', 'STB', 
            'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 
            'VNM', 'VPB'
        ]
        return vn30_stocks
    
    def get_all_stocks(self) -> List[str]:
        """Trả về danh sách tất cả các mã chứng khoán"""
        # Nếu có vnstock, sử dụng để lấy danh sách đầy đủ
        if VNSTOCK_AVAILABLE:
            try:
                # Lấy danh sách từ vnstock
                listing_companies = vnstock.listing_companies()
                if isinstance(listing_companies, pd.DataFrame) and not listing_companies.empty:
                    return listing_companies['ticker'].tolist()
            except Exception as e:
                logger.warning(f"Không thể lấy danh sách mã từ vnstock: {str(e)}")
        
        # Fallback: trả về danh sách mã phổ biến + VN30
        return list(set(self.get_popular_stocks() + self.get_vn30_stocks() + [
            # Thêm một số mã khác vào đây
            'VRE', 'SSB', 'VND', 'VIX', 'PNJ', 'DGC', 'HVN', 'HAG', 
            'VTP', 'HSG', 'DCM', 'SHB', 'GMD', 'PC1', 'NKG', 'HCM', 
            'DIG', 'DPM', 'VOS', 'DBC', 'PVD', 'VCI', 'SHS', 'EVF'
        ]))
    
    def get_stock_industry(self, symbol: str) -> str:
        """
        Lấy thông tin ngành của mã chứng khoán
        
        Args:
            symbol: Mã chứng khoán
            
        Returns:
            Tên ngành của mã
        """
        # Thông tin ngành mặc định cho một số mã phổ biến
        industry_map = {
            'VNM': 'Thực phẩm & Đồ uống',
            'VHM': 'Bất động sản',
            'VIC': 'Đa ngành',
            'VCB': 'Ngân hàng',
            'FPT': 'Công nghệ thông tin',
            'HPG': 'Thép',
            'MSN': 'Đa ngành',
            'MWG': 'Bán lẻ',
            'VRE': 'Bất động sản',
            'BID': 'Ngân hàng',
            'CTG': 'Ngân hàng',
            'GAS': 'Dầu khí',
            'TCB': 'Ngân hàng',
            'VPB': 'Ngân hàng',
            'TPB': 'Ngân hàng',
            'MBB': 'Ngân hàng',
            'HDB': 'Ngân hàng',
            'SSI': 'Chứng khoán',
            'VND': 'Chứng khoán',
            'HCM': 'Chứng khoán',
            'PNJ': 'Trang sức',
            'DBC': 'Thực phẩm & Đồ uống',
            'DGC': 'Hóa chất',
            'DCM': 'Hóa chất',
            'HSG': 'Thép',
            'NKG': 'Thép',
            'VTP': 'Vận tải & Logistics',
            'GMD': 'Cảng biển'
        }
        
        # Nếu có trong mapping, trả về ngành
        if symbol in industry_map:
            return industry_map[symbol]
        
        # Nếu có vnstock, thử lấy thông tin từ vnstock
        if VNSTOCK_AVAILABLE:
            try:
                listing_companies = vnstock.listing_companies()
                if isinstance(listing_companies, pd.DataFrame) and not listing_companies.empty:
                    company = listing_companies[listing_companies['ticker'] == symbol]
                    if not company.empty and 'industry' in company.columns:
                        return company['industry'].iloc[0]
            except Exception as e:
                logger.warning(f"Không thể lấy thông tin ngành từ vnstock: {str(e)}")
        
        # Nếu không có thông tin, trả về 'Khác'
        return 'Khác'