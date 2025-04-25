#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Timestamp Utilities Module for Stock Market Bot:
- Provides timestamp alignment utilities for different timeframes
- Contains helper functions for time series manipulations
- Handles timezone conversions for financial data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
from typing import Dict, List, Tuple, Optional, Union

# TZ-aware objects for common timezones
UTC = pytz.UTC
VIETNAM_TZ = pytz.timezone('Asia/Ho_Chi_Minh')

def align_timestamps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Align timestamps according to the specified timeframe
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with DatetimeIndex
    timeframe : str
        Timeframe to align to ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo')
        
    Returns:
    --------
    DataFrame with aligned timestamps
    """
    if df is None or df.empty:
        return df
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Normalize timeframe string
    timeframe = normalize_timeframe(timeframe)
    
    # For intraday data
    if timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']:
        # Map to pandas frequency strings
        freq_map = {
            '1m': 'T',      # minute
            '5m': '5T',     # 5 minutes
            '15m': '15T',   # 15 minutes
            '30m': '30T',   # 30 minutes
            '1h': 'H',      # hour
            '4h': '4H'      # 4 hours
        }
        freq = freq_map[timeframe]
        
        # Floor timestamps to exact intervals
        # e.g., 09:03:45 -> 09:00:00 for 1h timeframe
        df.index = df.index.floor(freq)
        
    # For daily data
    elif timeframe == '1d':
        # Normalize to midnight
        df.index = df.index.normalize()
        
    # For weekly data
    elif timeframe == '1w':
        # Set to Monday of the week
        df.index = df.index - pd.to_timedelta(df.index.dayofweek, unit='D')
        df.index = df.index.normalize()
        
    # For monthly data
    elif timeframe == '1mo':
        # Set to first day of the month
        df.index = df.index.to_period('M').to_timestamp()
        
    # Remove duplicates that might occur after rounding
    df = df[~df.index.duplicated(keep='last')]
    
    # Sort by index
    df = df.sort_index()
    
    return df

def normalize_timeframe(timeframe: str) -> str:
    """
    Normalize timeframe string to standard format
    
    Parameters:
    -----------
    timeframe : str
        Timeframe string in various formats
        
    Returns:
    --------
    str
        Normalized timeframe string
    """
    # Convert to lowercase and remove spaces
    timeframe = timeframe.lower().replace(' ', '')
    
    # Handle special cases and aliases
    aliases = {
        '1minute': '1m', '1min': '1m', 'm': '1m', 'minute': '1m',
        '5minute': '5m', '5min': '5m',
        '15minute': '15m', '15min': '15m',
        '30minute': '30m', '30min': '30m',
        '1hour': '1h', '1hr': '1h', '60m': '1h', '60min': '1h', 'h': '1h', 'hour': '1h',
        '4hour': '4h', '4hr': '4h', '240m': '4h', '240min': '4h',
        '1day': '1d', 'daily': '1d', 'day': '1d', 'd': '1d',
        '1week': '1w', 'weekly': '1w', 'week': '1w', 'w': '1w',
        '1month': '1mo', 'monthly': '1mo', 'month': '1mo', 'mo': '1mo',
        # Upper case aliases
        '1D': '1d', '1W': '1w', '1M': '1mo', '1H': '1h'
    }
    
    return aliases.get(timeframe, timeframe)

def resample_data(df: pd.DataFrame, to_timeframe: str, agg_methods: Optional[Dict] = None) -> pd.DataFrame:
    """
    Resample data to a different timeframe
    
    Parameters:
    -----------
    df : pandas DataFrame
        OHLCV data with DatetimeIndex
    to_timeframe : str
        Target timeframe to resample to
    agg_methods : dict, optional
        Custom aggregation methods for each column. If None, uses default OHLCV methods
        
    Returns:
    --------
    DataFrame with resampled data
    """
    if df is None or df.empty:
        return df
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Make sure required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Default aggregation methods for OHLCV data
    default_methods = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Use custom methods if provided, otherwise use defaults
    methods = agg_methods if agg_methods is not None else default_methods
    
    # Normalize the target timeframe
    to_timeframe = normalize_timeframe(to_timeframe)
    
    # Map to pandas frequency strings
    freq_map = {
        '1m': 'T',      # minute
        '5m': '5T',     # 5 minutes
        '15m': '15T',   # 15 minutes
        '30m': '30T',   # 30 minutes
        '1h': 'H',      # hour
        '4h': '4H',     # 4 hours
        '1d': 'D',      # day
        '1w': 'W-MON',  # week (Monday start)
        '1mo': 'MS'     # month start
    }
    
    if to_timeframe not in freq_map:
        raise ValueError(f"Unsupported target timeframe: {to_timeframe}")
    
    # Resample the data
    resampled = df.resample(freq_map[to_timeframe]).agg(methods)
    
    # For OHLCV data, ensure we have volume as 0 for periods with no trades
    if 'volume' in resampled.columns:
        resampled['volume'] = resampled['volume'].fillna(0)
    
    # Handle periods with no data
    if resampled.isna().any().any():
        # For price data, forward fill is often most appropriate
        for col in ['open', 'high', 'low', 'close']:
            if col in resampled.columns:
                resampled[col] = resampled[col].fillna(method='ffill')
    
    return resampled

def ensure_timezone(df: pd.DataFrame, target_tz: str = 'Asia/Ho_Chi_Minh') -> pd.DataFrame:
    """
    Ensure DataFrame index has the correct timezone
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with DatetimeIndex
    target_tz : str
        Target timezone to convert to
        
    Returns:
    --------
    DataFrame with timezone-aware index
    """
    if df is None or df.empty:
        return df
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Convert timezone
    if df.index.tz is None:
        # Localize naive datetimes
        df.index = df.index.tz_localize(target_tz)
    elif str(df.index.tz) != target_tz:
        # Convert from one timezone to another
        df.index = df.index.tz_convert(target_tz)
    
    return df

def get_trading_periods(timeframe: str, date: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    Get start and end of trading period for a specific date
    
    Parameters:
    -----------
    timeframe : str
        Timeframe ('1d', '1w', etc.)
    date : datetime, optional
        Reference date (defaults to today)
        
    Returns:
    --------
    Tuple of (start_time, end_time) for the trading period
    """
    if date is None:
        date = datetime.now(VIETNAM_TZ)
    elif date.tzinfo is None:
        date = date.replace(tzinfo=VIETNAM_TZ)
    
    timeframe = normalize_timeframe(timeframe)
    
    # Vietnam stock market trading hours: 9:00-11:30, 13:00-15:00
    if timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']:
        # For intraday, use exact trading hours
        day_start = datetime.combine(date.date(), time(9, 0), tzinfo=VIETNAM_TZ)
        day_end = datetime.combine(date.date(), time(15, 0), tzinfo=VIETNAM_TZ)
        
        # Check if weekend
        if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            # Find next Monday
            days_ahead = 7 - date.weekday()
            next_monday = date + timedelta(days=days_ahead)
            day_start = datetime.combine(next_monday.date(), time(9, 0), tzinfo=VIETNAM_TZ)
            day_end = datetime.combine(next_monday.date(), time(15, 0), tzinfo=VIETNAM_TZ)
    
    elif timeframe == '1d':
        # For daily, use calendar day
        day_start = datetime.combine(date.date(), time(0, 0), tzinfo=VIETNAM_TZ)
        day_end = datetime.combine(date.date(), time(23, 59, 59), tzinfo=VIETNAM_TZ)
    
    elif timeframe == '1w':
        # For weekly, use Monday-Friday
        monday = date - timedelta(days=date.weekday())
        friday = monday + timedelta(days=4)
        day_start = datetime.combine(monday.date(), time(0, 0), tzinfo=VIETNAM_TZ)
        day_end = datetime.combine(friday.date(), time(23, 59, 59), tzinfo=VIETNAM_TZ)
    
    elif timeframe == '1mo':
        # For monthly, use calendar month
        month_start = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        # Get last day of month
        if month_start.month == 12:
            next_month = month_start.replace(year=month_start.year+1, month=1)
        else:
            next_month = month_start.replace(month=month_start.month+1)
        last_day = next_month - timedelta(days=1)
        day_end = datetime.combine(last_day.date(), time(23, 59, 59), tzinfo=VIETNAM_TZ)
        day_start = month_start
    
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    return day_start, day_end

def fill_missing_periods(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Fill missing periods in time series data
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with DatetimeIndex
    timeframe : str
        Timeframe of the data
        
    Returns:
    --------
    DataFrame with missing periods filled
    """
    if df is None or df.empty:
        return df
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Normalize timeframe
    timeframe = normalize_timeframe(timeframe)
    
    # Map to pandas frequency strings
    freq_map = {
        '1m': 'T',      # minute
        '5m': '5T',     # 5 minutes
        '15m': '15T',   # 15 minutes
        '30m': '30T',   # 30 minutes
        '1h': 'H',      # hour
        '4h': '4H',     # 4 hours
        '1d': 'B',      # business day
        '1w': 'W-MON',  # week (Monday start)
        '1mo': 'MS'     # month start
    }
    
    if timeframe not in freq_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    # Create a complete index
    if timeframe == '1d':
        # For daily data, use business days
        full_idx = pd.bdate_range(start=df.index.min(), end=df.index.max(), freq=freq_map[timeframe])
    else:
        # For other timeframes, use regular frequency
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq_map[timeframe])
    
    # Reindex the DataFrame
    df_filled = df.reindex(full_idx)
    
    # Fill missing values
    # For price, forward fill is usually appropriate
    price_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'close']]
    df_filled[price_cols] = df_filled[price_cols].fillna(method='ffill')
    
    # For volume, fill with 0
    if 'volume' in df.columns:
        df_filled['volume'] = df_filled['volume'].fillna(0)
    
    return df_filled

def is_trading_time(timestamp: datetime = None) -> bool:
    """
    Check if a given timestamp is during Vietnam trading hours
    
    Parameters:
    -----------
    timestamp : datetime, optional
        Timestamp to check (defaults to now)
        
    Returns:
    --------
    bool
        True if the timestamp is during trading hours
    """
    if timestamp is None:
        timestamp = datetime.now(VIETNAM_TZ)
    elif timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=VIETNAM_TZ)
    
    # Convert to Vietnam time if in different timezone
    if timestamp.tzinfo != VIETNAM_TZ:
        timestamp = timestamp.astimezone(VIETNAM_TZ)
    
    # Check if weekend
    if timestamp.weekday() >= 5:  # 5=Saturday, 6=Sunday
        return False
    
    # Morning session: 9:00-11:30
    morning_start = time(9, 0)
    morning_end = time(11, 30)
    
    # Afternoon session: 13:00-15:00
    afternoon_start = time(13, 0)
    afternoon_end = time(15, 0)
    
    current_time = timestamp.time()
    
    # Check if current time is in either session
    is_morning_session = morning_start <= current_time <= morning_end
    is_afternoon_session = afternoon_start <= current_time <= afternoon_end
    
    return is_morning_session or is_afternoon_session

def merge_timeframes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge data from multiple timeframes into a single DataFrame
    
    Parameters:
    -----------
    dfs : dict
        Dictionary of {timeframe: dataframe} pairs
        
    Returns:
    --------
    DataFrame with merged data
    """
    if not dfs:
        return pd.DataFrame()
    
    # Sort timeframes from smallest to largest
    timeframe_order = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1mo']
    
    # Normalized timeframes
    normalized_dfs = {}
    for tf, df in dfs.items():
        normalized_tf = normalize_timeframe(tf)
        normalized_dfs[normalized_tf] = df
    
    # Find the smallest timeframe as the base
    available_tfs = [tf for tf in timeframe_order if tf in normalized_dfs]
    if not available_tfs:
        return pd.DataFrame()
    
    base_tf = available_tfs[0]
    base_df = normalized_dfs[base_tf].copy()
    
    # Add suffix to base columns
    base_df = base_df.add_suffix(f'_{base_tf}')
    
    # Merge with larger timeframes
    for tf in available_tfs[1:]:
        if tf in normalized_dfs:
            # Add suffix to current timeframe columns
            current_df = normalized_dfs[tf].copy()
            current_df = current_df.add_suffix(f'_{tf}')
            
            # Resample the current DF to match the base DF frequency
            current_df = current_df.asfreq(base_df.index.freq, method='pad')
            
            # Merge on index
            base_df = pd.merge(
                base_df, 
                current_df,
                left_index=True,
                right_index=True,
                how='left'
            )
    
    return base_df 