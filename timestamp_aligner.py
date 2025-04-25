#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module căn chỉnh timestamp cho dữ liệu chứng khoán Việt Nam.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import holidays

logger = logging.getLogger(__name__)

class TimestampAligner:
    def __init__(self, exchange_timezone='Asia/Bangkok'):
        """
        Khởi tạo bộ căn chỉnh timestamp
        
        Args:
            exchange_timezone: Múi giờ của sàn giao dịch
        """
        self.exchange_timezone = exchange_timezone
        self.trading_hours = {
            'morning_open': 9,   # 9:00 AM
            'morning_close': 11, # 11:30 AM
            'afternoon_open': 13, # 1:00 PM
            'afternoon_close': 15, # 3:00 PM (15:00)
        }
        
    def align_timestamps(self, df):
        """
        Căn chỉnh tất cả timestamp trong DataFrame
        
        Args:
            df: DataFrame chứa dữ liệu với index là timestamp
            
        Returns:
            DataFrame với timestamp đã được căn chỉnh
        """
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
            
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        aligned_df = df.copy()
        
        # Đảm bảo index là datetime có timezone
        if aligned_df.index.tz is None:
            aligned_df.index = aligned_df.index.tz_localize(self.exchange_timezone)
        elif aligned_df.index.tz.zone != self.exchange_timezone:
            aligned_df.index = aligned_df.index.tz_convert(self.exchange_timezone)
            
        # Chuẩn hóa timestamp về giờ đóng cửa
        close_hour = self.trading_hours['afternoon_close']
        new_index = pd.DatetimeIndex([
            idx.replace(hour=close_hour, minute=0, second=0, microsecond=0)
            for idx in aligned_df.index
        ])
        
        aligned_df.index = new_index
        
        # Loại bỏ trùng lặp
        aligned_df = aligned_df[~aligned_df.index.duplicated(keep='last')]
        
        return aligned_df.sort_index()
        
    def detect_frequency(self, df):
        """
        Phát hiện tần suất dữ liệu (ngày, tuần, tháng)
        
        Args:
            df: DataFrame để phát hiện tần suất
            
        Returns:
            Chuỗi '1D', '1W', hoặc '1M' chỉ định tần suất dữ liệu
        """
        if len(df) < 2:
            return '1D'
            
        # Tính khoảng cách giữa các timestamp
        timedeltas = []
        sorted_df = df.sort_index()
        for i in range(1, min(len(sorted_df), 5)):
            delta = (sorted_df.index[i] - sorted_df.index[i-1]).total_seconds() / (60 * 60 * 24)
            timedeltas.append(delta)
            
        avg_delta = np.median(timedeltas)
        
        if avg_delta < 2:
            return '1D'  # Dữ liệu ngày
        elif avg_delta < 10:
            return '1W'  # Dữ liệu tuần
        else:
            return '1M'  # Dữ liệu tháng
    
    def standardize_timeframe(self, df, freq='1D'):
        """
        Chuẩn hóa DataFrame theo khung thời gian cụ thể
        
        Args:
            df: DataFrame cần chuẩn hóa
            freq: Tần suất mục tiêu ('1D', '1W', '1M')
            
        Returns:
            DataFrame đã chuẩn hóa
        """
        if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
            return df
            
        # Sắp xếp dữ liệu theo thời gian
        df = df.sort_index()
        
        if freq == '1D':
            # Đặt tất cả timestamp về 15:00
            return self.align_timestamps(df)
        elif freq == '1W':
            # Gom nhóm theo tuần, lấy ngày cuối tuần
            # Đầu tiên căn chỉnh timezone
            aligned_df = df.copy()
            if aligned_df.index.tz is None:
                aligned_df.index = aligned_df.index.tz_localize(self.exchange_timezone)
                
            # Gom nhóm theo tuần và lấy dòng cuối cùng
            weekly_df = aligned_df.resample('W-FRI').last().dropna()
            
            # Căn chỉnh timestamp cho weekly_df
            return self.align_timestamps(weekly_df)
        elif freq == '1M':
            # Gom nhóm theo tháng, lấy ngày cuối tháng
            aligned_df = df.copy()
            if aligned_df.index.tz is None:
                aligned_df.index = aligned_df.index.tz_localize(self.exchange_timezone)
                
            # Gom nhóm theo tháng và lấy dòng cuối cùng
            monthly_df = aligned_df.resample('M').last().dropna()
            
            # Căn chỉnh timestamp cho monthly_df
            return self.align_timestamps(monthly_df)
        else:
            return self.align_timestamps(df)
    
    def filter_trading_days(self, df):
        """
        Lọc ra các ngày giao dịch hợp lệ
        
        Args:
            df: DataFrame cần lọc
            
        Returns:
            DataFrame chỉ chứa các ngày giao dịch hợp lệ
        """
        if df is None or df.empty:
            return df
            
        # Lọc các ngày cuối tuần (thứ 7, chủ nhật)
        trading_days_mask = (df.index.weekday < 5)  # 0-4 = Thứ 2 đến Thứ 6
        
        # Lọc các ngày nghỉ lễ Việt Nam
        years = df.index.year.unique()
        vn_holidays = holidays.Vietnam(years=list(years))
        holiday_dates = set(vn_holidays.keys())
        
        # Tạo mảng boolean để lọc ngày nghỉ lễ
        not_holiday_mask = pd.Series(
            ~np.array([idx.date() in holiday_dates for idx in df.index]),
            index=df.index
        )
        
        # Áp dụng cả hai bộ lọc
        valid_days_mask = trading_days_mask & not_holiday_mask
        filtered_df = df[valid_days_mask]
        
        return filtered_df
    
    def fix_timestamp_issues(self, df):
        """
        Sửa các vấn đề thường gặp với timestamp
        
        Args:
            df: DataFrame cần sửa
            
        Returns:
            DataFrame đã được sửa các vấn đề timestamp
        """
        if df is None or df.empty:
            return df
            
        fixed_df = df.copy()
        
        # 1. Xử lý timezone
        if not isinstance(fixed_df.index, pd.DatetimeIndex):
            # Nếu index không phải là DatetimeIndex, thử chuyển đổi
            try:
                fixed_df.index = pd.to_datetime(fixed_df.index)
            except Exception as e:
                logger.error(f"Không thể chuyển đổi index sang DatetimeIndex: {str(e)}")
                return df
        
        # 2. Sửa lỗi timezone
        if fixed_df.index.tz is None:
            fixed_df.index = fixed_df.index.tz_localize(self.exchange_timezone)
        elif fixed_df.index.tz.zone != self.exchange_timezone:
            fixed_df.index = fixed_df.index.tz_convert(self.exchange_timezone)
        
        # 3. Xử lý trùng lặp timestamp
        if fixed_df.index.duplicated().any():
            fixed_df = fixed_df[~fixed_df.index.duplicated(keep='last')]
            logger.info(f"Đã loại bỏ {sum(fixed_df.index.duplicated())} timestamp trùng lặp")
        
        # 4. Đảm bảo các timestamp được sắp xếp
        if not fixed_df.index.is_monotonic_increasing:
            fixed_df = fixed_df.sort_index()
            logger.info("Sắp xếp lại timestamp theo thứ tự tăng dần")
        
        return fixed_df
    
    def merge_dataframes_with_alignment(self, dfs, freq=None):
        """
        Hợp nhất nhiều DataFrame với căn chỉnh timestamp
        
        Args:
            dfs: Danh sách DataFrame cần hợp nhất
            freq: Tần suất để chuẩn hóa (nếu không cung cấp, tự động phát hiện)
            
        Returns:
            DataFrame hợp nhất với timestamp đã được căn chỉnh
        """
        if not dfs:
            return pd.DataFrame()
            
        # Loại bỏ các DataFrame rỗng
        valid_dfs = [df for df in dfs if df is not None and not df.empty]
        
        if not valid_dfs:
            return pd.DataFrame()
            
        # Nếu không cung cấp freq, phát hiện từ DataFrame đầu tiên
        if freq is None:
            freq = self.detect_frequency(valid_dfs[0])
            
        # Chuẩn hóa từng DataFrame
        aligned_dfs = [self.standardize_timeframe(df, freq=freq) for df in valid_dfs]
        
        # Hợp nhất các DataFrame
        merged_df = pd.concat(aligned_dfs)
        
        # Xử lý trùng lặp
        merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
        
        # Sắp xếp theo thời gian
        merged_df = merged_df.sort_index()
        
        return merged_df 