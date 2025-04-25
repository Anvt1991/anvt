#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Data Loader Module for Stock Market Bot:
- Supports 1h, 4h timeframes from Yahoo Finance
- Provides automatic fallback between data sources
- Implements intelligent error handling and retry logic
- Tracks and reports data source reliability
"""

import os
import io
import logging
import pickle
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional, Union

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Create module-specific logger
logger = logging.getLogger(__name__)

class DataSourceTracker:
    """Tracks reliability metrics for data sources"""
    def __init__(self, redis_manager=None):
        self.reliability_scores = {
            'vnstock': 1.0,
            'yahoo': 0.95,
            'binance': 0.90,
            'alphavantage': 0.85
        }
        self.failure_penalty = 0.05
        self.success_reward = 0.01
        self.min_score = 0.3
        self.max_score = 1.0
        self.redis_manager = redis_manager
        
    async def load_scores(self):
        """Load reliability scores from Redis if available"""
        if self.redis_manager:
            scores = await self.redis_manager.get("data_source_reliability")
            if scores:
                self.reliability_scores = scores
                
    async def save_scores(self):
        """Save reliability scores to Redis if available"""
        if self.redis_manager:
            await self.redis_manager.set("data_source_reliability", 
                                        self.reliability_scores, 
                                        expire=604800)  # 7 days
                
    def get_sources_by_reliability(self) -> List[str]:
        """Return data sources ordered by reliability"""
        return sorted(self.reliability_scores.keys(), 
                      key=lambda x: self.reliability_scores[x], 
                      reverse=True)
    
    def record_success(self, source: str):
        """Record successful data retrieval for a source"""
        if source in self.reliability_scores:
            self.reliability_scores[source] = min(
                self.max_score,
                self.reliability_scores[source] + self.success_reward
            )
            asyncio.create_task(self.save_scores())
    
    def record_failure(self, source: str):
        """Record failed data retrieval for a source"""
        if source in self.reliability_scores:
            self.reliability_scores[source] = max(
                self.min_score,
                self.reliability_scores[source] - self.failure_penalty
            )
            asyncio.create_task(self.save_scores())
    
    def get_reliability(self, source: str) -> float:
        """Get reliability score for a source"""
        return self.reliability_scores.get(source, 0.5)

class EnhancedDataLoader:
    def __init__(self, redis_manager=None, vnstock_client=None, run_in_thread=None):
        self.redis_manager = redis_manager
        self.vnstock_client = vnstock_client
        self.run_in_thread = run_in_thread
        self.source_tracker = DataSourceTracker(redis_manager)
        self.session = None
        self.all_timeframes = {
            'yahoo': ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'],
            'vnstock': ['1D', '1W', '1M']
        }
        self.timeframe_mapping = {
            # Standard mapping
            '1m': {'yahoo': '1m', 'vnstock': None},
            '5m': {'yahoo': '5m', 'vnstock': None},
            '15m': {'yahoo': '15m', 'vnstock': None},
            '30m': {'yahoo': '30m', 'vnstock': None},
            '1h': {'yahoo': '60m', 'vnstock': None},
            '4h': {'yahoo': '90m', 'vnstock': None},  # Not exact but closest available
            '1d': {'yahoo': '1d', 'vnstock': '1D'},
            '1w': {'yahoo': '1wk', 'vnstock': '1W'},
            '1mo': {'yahoo': '1mo', 'vnstock': '1M'},
            
            # Aliases
            '1D': {'yahoo': '1d', 'vnstock': '1D'},
            '1W': {'yahoo': '1wk', 'vnstock': '1W'},
            '1M': {'yahoo': '1mo', 'vnstock': '1M'},
            '1H': {'yahoo': '60m', 'vnstock': None},
            '4H': {'yahoo': '90m', 'vnstock': None},
        }
        
        # Cache expiration settings based on timeframe
        self.cache_expire_settings = {
            '1m': 300,       # 5 minutes
            '5m': 600,       # 10 minutes
            '15m': 900,      # 15 minutes
            '30m': 1800,     # 30 minutes
            '1h': 3600,      # 1 hour
            '4h': 7200,      # 2 hours
            '1d': 21600,     # 6 hours
            '1w': 86400,     # 24 hours
            '1mo': 259200    # 3 days
        }
    
    async def initialize(self):
        """Initialize the data loader including reliability scores"""
        await self.source_tracker.load_scores()
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close resources"""
        if self.session:
            await self.session.close()
    
    def _normalize_timeframe(self, timeframe: str) -> str:
        """Normalize timeframe string to standard format"""
        timeframe = timeframe.lower().replace(' ', '')
        
        # Handle special cases and aliases
        aliases = {
            '1hour': '1h', '1hr': '1h', '60m': '1h', '60min': '1h',
            '4hour': '4h', '4hr': '4h', '240m': '4h', '240min': '4h',
            '1day': '1d', 'daily': '1d', 'day': '1d',
            '1week': '1w', 'weekly': '1w', 'week': '1w',
            '1month': '1mo', 'monthly': '1mo', 'month': '1mo'
        }
        
        return aliases.get(timeframe, timeframe)
    
    def _get_cache_expiry(self, timeframe: str) -> int:
        """Get appropriate cache expiration time based on timeframe"""
        normalized_tf = self._normalize_timeframe(timeframe)
        return self.cache_expire_settings.get(normalized_tf, 3600)  # Default 1 hour
    
    async def _get_supported_sources(self, timeframe: str) -> List[str]:
        """Get list of sources that support the given timeframe, ordered by reliability"""
        normalized_tf = self._normalize_timeframe(timeframe)
        mapping = self.timeframe_mapping.get(normalized_tf, {})
        
        # Get sources that support this timeframe
        supported_sources = [
            source for source, mapped_tf in mapping.items() 
            if mapped_tf is not None
        ]
        
        # Sort by reliability
        all_sources_by_reliability = self.source_tracker.get_sources_by_reliability()
        return [s for s in all_sources_by_reliability if s in supported_sources]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _fetch_yahoo_data(self, symbol: str, timeframe: str, num_candles: int) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            normalized_tf = self._normalize_timeframe(timeframe)
            mapped_tf = self.timeframe_mapping.get(normalized_tf, {}).get('yahoo')
            
            if not mapped_tf:
                raise ValueError(f"Timeframe {timeframe} not supported by Yahoo Finance")
            
            # Determine interval and period values
            interval_mapping = {
                '1m': '1m', '2m': '2m', '5m': '5m', '15m': '15m', '30m': '30m', 
                '60m': '60m', '90m': '90m', '1h': '60m', '1d': '1d', 
                '5d': '5d', '1wk': '1wk', '1mo': '1mo', '3mo': '3mo'
            }
            
            interval = interval_mapping.get(mapped_tf, mapped_tf)
            
            # Calculate start and end timestamps
            # For intraday data, Yahoo limits history, adjust accordingly
            if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
                # Intraday data has limitations - Yahoo only provides last 7 days for 1m
                # and last 60 days for other intraday intervals
                days_back = 7 if interval == '1m' else 60
                start_ts = int((datetime.now() - timedelta(days=days_back)).timestamp())
            else:
                # For daily and above, we can go back much further
                days_multiplier = {
                    '1d': 1.5, '5d': 5, '1wk': 7, '1mo': 30, '3mo': 90
                }.get(interval, 1.5)
                
                start_ts = int((datetime.now() - timedelta(days=num_candles * days_multiplier)).timestamp())
            
            end_ts = int(datetime.now().timestamp())
            
            # Build URL with correct ticker symbol format
            ticker = symbol if '.' in symbol else f"{symbol}.VN"
            url = (f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
                  f"?period1={start_ts}&period2={end_ts}&interval={interval}&events=history")
            
            async with self.session.get(url, timeout=15) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP error {response.status} from Yahoo Finance")
                
                text = await response.text()
                if not text or text.startswith("<!DOCTYPE html>"):
                    raise ValueError("Invalid response from Yahoo Finance")
                
                df = pd.read_csv(io.StringIO(text))
                if df.empty:
                    raise ValueError("Empty data returned from Yahoo Finance")
                
                # Process DataFrame
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                df = df.rename(columns={
                    'Open': 'open', 'High': 'high', 'Low': 'low',
                    'Close': 'close', 'Volume': 'volume'
                })
                
                # Ensure all required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        raise ValueError(f"Missing required column {col} in Yahoo data")
                
                # Align timestamp with timeframe
                df = self._align_timestamps(df, normalized_tf)
                
                # For intraday data, adjust timezone info
                if interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m']:
                    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Bangkok')
                else:
                    df.index = df.index.tz_localize('Asia/Bangkok')
                
                # Basic validation
                if not (df['high'] >= df['low']).all():
                    logger.warning(f"Some high values are less than low values in Yahoo data for {symbol}")
                
                result = df.tail(num_candles)
                self.source_tracker.record_success('yahoo')
                return result
            
        except Exception as e:
            self.source_tracker.record_failure('yahoo')
            logger.error(f"Error fetching Yahoo data for {symbol}, {timeframe}: {str(e)}")
            raise
    
    async def _fetch_vnstock_data(self, symbol: str, timeframe: str, num_candles: int) -> pd.DataFrame:
        """Fetch data from VNStock"""
        try:
            if not self.vnstock_client or not self.run_in_thread:
                raise ValueError("VNStock client not initialized")
            
            normalized_tf = self._normalize_timeframe(timeframe)
            mapped_tf = self.timeframe_mapping.get(normalized_tf, {}).get('vnstock')
            
            if not mapped_tf:
                raise ValueError(f"Timeframe {timeframe} not supported by VNStock")
            
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
            def fetch_data():
                stock = self.vnstock_client.stock(symbol=symbol, source='TCBS')
                end_date = datetime.now().strftime('%Y-%m-%d')
                # Request more days than needed to account for weekends/holidays
                start_date = (datetime.now() - timedelta(days=(num_candles + 1) * 3)).strftime('%Y-%m-%d')
                df = stock.quote.history(start=start_date, end=end_date, interval=mapped_tf)
                
                if df is None or df.empty:
                    raise ValueError(f"Empty data returned from VNStock for {symbol}")
                
                # Standardize column names
                df = df.rename(columns={
                    'time': 'date', 'open': 'open', 'high': 'high',
                    'low': 'low', 'close': 'close', 'volume': 'volume'
                })
                
                # Ensure date is properly formatted
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df.index = df.index.tz_localize('Asia/Bangkok')
                
                # Select only necessary columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in VNStock data: {missing_cols}")
                
                df = df[required_cols].dropna()
                
                # Basic validation
                if not (df['high'] >= df['low']).all():
                    logger.warning(f"Some high values are less than low values in VNStock data for {symbol}")
                
                return df.tail(num_candles + 1)
            
            # Run VNStock functions in a thread pool to avoid blocking
            df = await self.run_in_thread(fetch_data)
            
            # Align timestamps with timeframe
            df = self._align_timestamps(df, normalized_tf)
            
            self.source_tracker.record_success('vnstock')
            return df
            
        except Exception as e:
            self.source_tracker.record_failure('vnstock')
            logger.error(f"Error fetching VNStock data for {symbol}, {timeframe}: {str(e)}")
            raise
    
    def _align_timestamps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Align timestamps according to the specified timeframe"""
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Normalize the timestamp based on timeframe
        if timeframe in ['1m', '5m', '15m', '30m', '1h', '4h']:
            # For intraday data
            freq_map = {
                '1m': 'T', '5m': '5T', '15m': '15T', 
                '30m': '30T', '1h': 'H', '4h': '4H'
            }
            freq = freq_map.get(timeframe, 'D')
            
            # Round timestamps to exact intervals
            # e.g., 09:03:45 -> 09:00:00 for 1h timeframe
            df.index = df.index.floor(freq)
            
            # Remove duplicates that might occur after rounding
            df = df[~df.index.duplicated(keep='last')]
            
        elif timeframe in ['1d', '1w', '1mo']:
            # For daily/weekly/monthly data - normalize to midnight
            df.index = df.index.normalize()
            df = df[~df.index.duplicated(keep='last')]
            
        return df
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'zscore', threshold: float = 3.0) -> Tuple[pd.DataFrame, str]:
        """
        Detect outliers in the data using various methods
        
        Parameters:
        -----------
        df : pandas DataFrame
            The data to check for outliers
        method : str
            Method to use for outlier detection ('zscore', 'iqr', 'dbscan')
        threshold : float
            Threshold for outlier detection
            
        Returns:
        --------
        df : pandas DataFrame
            DataFrame with 'is_outlier' column added
        report : str
            Text report describing outlier detection results
        """
        if df is None or df.empty or 'close' not in df.columns:
            return df, "No data available for outlier detection"
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        df['is_outlier'] = False
        
        try:
            if method == 'zscore':
                # Z-score method (default)
                z_scores = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
                df['is_outlier'] = z_scores > threshold
                
            elif method == 'iqr':
                # IQR method
                Q1 = df['close'].quantile(0.25)
                Q3 = df['close'].quantile(0.75)
                IQR = Q3 - Q1
                df['is_outlier'] = ((df['close'] < (Q1 - 1.5 * IQR)) | 
                                    (df['close'] > (Q3 + 1.5 * IQR)))
                
            elif method == 'rolling':
                # Rolling window method - more suitable for time series
                window = min(20, len(df) // 4)  # Adjust window size based on data length
                rolling_mean = df['close'].rolling(window=window, center=True).mean()
                rolling_std = df['close'].rolling(window=window, center=True).std()
                df['is_outlier'] = np.abs(df['close'] - rolling_mean) > (threshold * rolling_std)
                
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
            
            # Generate report
            outliers = df[df['is_outlier']]
            if outliers.empty:
                report = "Không có giá trị bất thường trong dữ liệu"
            else:
                report = f"Phát hiện {len(outliers)} giá trị bất thường bằng phương pháp {method}:\n"
                for idx, row in outliers.iterrows():
                    date_str = idx.strftime('%Y-%m-%d %H:%M:%S') if idx.hour > 0 else idx.strftime('%Y-%m-%d')
                    report += f"- {date_str}: {row['close']:.2f}\n"
            
            return df, report
            
        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            df['is_outlier'] = False
            return df, f"Lỗi phát hiện outliers: {str(e)}"
    
    async def load_data(self, symbol: str, timeframe: str, num_candles: int, 
                        detect_outliers: bool = True, outlier_method: str = 'zscore',
                        use_cache: bool = True) -> Tuple[pd.DataFrame, str]:
        """
        Load financial data with automatic source fallback
        
        Parameters:
        -----------
        symbol : str
            The ticker symbol to load data for
        timeframe : str
            Timeframe for the data (e.g., '1m', '1h', '1d', '1w')
        num_candles : int
            Number of candles to retrieve
        detect_outliers : bool
            Whether to detect outliers in the data
        outlier_method : str
            Method to use for outlier detection
        use_cache : bool
            Whether to use cached data if available
            
        Returns:
        --------
        df : pandas DataFrame
            DataFrame with OHLCV data
        report : str
            Report on data quality and outliers
        """
        normalized_tf = self._normalize_timeframe(timeframe)
        cache_key = f"enhanced_data_{symbol}_{normalized_tf}_{num_candles}"
        cache_expire = self._get_cache_expiry(normalized_tf)
        
        # Try to get from cache if enabled
        if use_cache and self.redis_manager:
            cached_data = await self.redis_manager.get(cache_key)
            if cached_data is not None:
                return cached_data, "Data from cache"
        
        # Get list of available sources for this timeframe
        sources = await self._get_supported_sources(normalized_tf)
        if not sources:
            raise ValueError(f"No data sources available for timeframe {timeframe}")
        
        # Try each source in order of reliability
        errors = []
        for source in sources:
            try:
                if source == 'yahoo':
                    df = await self._fetch_yahoo_data(symbol, normalized_tf, num_candles)
                elif source == 'vnstock':
                    df = await self._fetch_vnstock_data(symbol, normalized_tf, num_candles)
                else:
                    continue
                
                # If we got here, we have data
                if df is not None and not df.empty:
                    # Filter for trading days if needed
                    if normalized_tf in ['1d', '1w', '1mo', '1D', '1W', '1M']:
                        from .utils import filter_trading_days
                        df = filter_trading_days(df)
                    
                    # Detect outliers if requested
                    outlier_report = ""
                    if detect_outliers:
                        df, outlier_report = self.detect_outliers(df, method=outlier_method)
                    
                    # Store in cache
                    if self.redis_manager:
                        await self.redis_manager.set(cache_key, df, expire=cache_expire)
                    
                    return df, f"Data from {source}. {outlier_report}"
            
            except Exception as e:
                error_msg = f"Source {source} failed: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)
                continue
        
        # If we got here, all sources failed
        error_summary = "\n".join(errors)
        raise ValueError(f"All data sources failed for {symbol}/{timeframe}:\n{error_summary}")
    
    async def load_multiple_timeframes(self, symbol: str, timeframes: List[str], 
                                      num_candles: int) -> Dict[str, Tuple[pd.DataFrame, str]]:
        """Load data for multiple timeframes"""
        results = {}
        tasks = []
        
        for tf in timeframes:
            task = asyncio.create_task(
                self.load_data(symbol, tf, num_candles)
            )
            tasks.append((tf, task))
            
        for tf, task in tasks:
            try:
                df, report = await task
                results[tf] = (df, report)
            except Exception as e:
                logger.error(f"Failed to load {tf} data for {symbol}: {str(e)}")
                results[tf] = (None, f"Error: {str(e)}")
                
        return results
    
    async def get_fundamental_data(self, symbol: str) -> dict:
        """Get fundamental data with automatic source fallback"""
        if '.' in symbol or symbol.upper() in ['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX', 'UPCOM']:
            return {"error": f"{symbol} is an index or non-local symbol, no fundamental data available"}
        
        cache_key = f"fundamental_enhanced_{symbol}"
        cached_data = await self.redis_manager.get(cache_key) if self.redis_manager else None
        
        if cached_data is not None:
            return cached_data
        
        # Try sources in reliability order
        sources = self.source_tracker.get_sources_by_reliability()
        for source in sources:
            try:
                if source == 'vnstock':
                    fundamental_data = await self._get_vnstock_fundamental(symbol)
                elif source == 'yahoo':
                    fundamental_data = await self._get_yahoo_fundamental(symbol)
                else:
                    continue
                    
                if fundamental_data and len(fundamental_data) > 2:  # Must have at least 2 metrics
                    # Cache the result
                    if self.redis_manager:
                        await self.redis_manager.set(cache_key, fundamental_data, expire=86400)  # 1 day
                    
                    self.source_tracker.record_success(source)
                    return fundamental_data
                    
            except Exception as e:
                logger.warning(f"Failed to get fundamental data from {source}: {str(e)}")
                self.source_tracker.record_failure(source)
                
        return {"error": f"Could not retrieve fundamental data for {symbol} from any source"}
    
    async def _get_vnstock_fundamental(self, symbol: str) -> dict:
        """Get fundamental data from VNStock"""
        if not self.vnstock_client or not self.run_in_thread:
            raise ValueError("VNStock client not initialized")
            
        def fetch():
            stock = self.vnstock_client.stock(symbol=symbol, source='TCBS')
            fundamental_data = {}
            
            # Get various fundamental metrics
            try:
                ratios = stock.finance.ratio()
                if ratios is not None and not ratios.empty:
                    fundamental_data.update(ratios.iloc[-1].to_dict())
            except Exception as e:
                logger.warning(f"Failed to get ratios from VNStock: {str(e)}")
                
            try:
                if hasattr(stock.finance, 'valuation'):
                    valuation = stock.finance.valuation()
                    if valuation is not None and not valuation.empty:
                        fundamental_data.update(valuation.iloc[-1].to_dict())
            except Exception as e:
                logger.warning(f"Failed to get valuation from VNStock: {str(e)}")
                
            # Add more data sources as needed
            
            if not fundamental_data:
                raise ValueError("No fundamental data from VNStock")
                
            return fundamental_data
            
        try:
            return await self.run_in_thread(fetch)
        except Exception as e:
            logger.error(f"Error getting VNStock fundamental data: {str(e)}")
            raise
    
    async def _get_yahoo_fundamental(self, symbol: str) -> dict:
        """Get fundamental data from Yahoo Finance"""
        ticker = symbol if '.' in symbol else f"{symbol}.VN"
        
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{ticker}?modules=summaryProfile,summaryDetail,defaultKeyStatistics,financialData"
            
            async with self.session.get(url, timeout=15) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP error {response.status} from Yahoo Finance")
                    
                data = await response.json()
                if not data or 'quoteSummary' not in data or 'result' not in data['quoteSummary'] or not data['quoteSummary']['result']:
                    raise ValueError("Invalid or empty response from Yahoo Finance")
                    
                result = data['quoteSummary']['result'][0]
                
                # Extract and combine metrics from different modules
                fundamental_data = {}
                
                if 'summaryDetail' in result:
                    sd = result['summaryDetail']
                    mappings = {
                        'trailingPE': 'P/E',
                        'priceToBook': 'P/B',
                        'dividendYield': 'Dividend Yield',
                        'marketCap': 'Market Cap',
                        'fiftyTwoWeekHigh': '52 Week High',
                        'fiftyTwoWeekLow': '52 Week Low'
                    }
                    for yahoo_key, display_key in mappings.items():
                        if yahoo_key in sd and sd[yahoo_key] is not None and 'raw' in sd[yahoo_key]:
                            fundamental_data[display_key] = sd[yahoo_key]['raw']
                
                if 'defaultKeyStatistics' in result:
                    ks = result['defaultKeyStatistics']
                    mappings = {
                        'trailingEps': 'EPS',
                        'bookValue': 'Book Value Per Share',
                        'returnOnEquity': 'ROE',
                        'returnOnAssets': 'ROA',
                        'netIncomeToCommon': 'Net Income'
                    }
                    for yahoo_key, display_key in mappings.items():
                        if yahoo_key in ks and ks[yahoo_key] is not None and 'raw' in ks[yahoo_key]:
                            fundamental_data[display_key] = ks[yahoo_key]['raw']
                
                if 'financialData' in result:
                    fd = result['financialData']
                    mappings = {
                        'totalRevenue': 'Revenue',
                        'grossMargins': 'Gross Margin',
                        'operatingMargins': 'Operating Margin',
                        'profitMargins': 'Profit Margin',
                        'debtToEquity': 'Debt to Equity'
                    }
                    for yahoo_key, display_key in mappings.items():
                        if yahoo_key in fd and fd[yahoo_key] is not None and 'raw' in fd[yahoo_key]:
                            fundamental_data[display_key] = fd[yahoo_key]['raw']
                
                # Calculate additional metrics if possible
                if 'EPS' in fundamental_data and 'Market Cap' in fundamental_data:
                    try:
                        shares = fundamental_data['Market Cap'] / fundamental_data['EPS']
                        fundamental_data['Shares Outstanding'] = shares
                    except:
                        pass
                
                if 'Book Value Per Share' in fundamental_data and 'Shares Outstanding' in fundamental_data:
                    try:
                        fundamental_data['Book Value'] = fundamental_data['Book Value Per Share'] * fundamental_data['Shares Outstanding']
                    except:
                        pass
                
                return fundamental_data
                
        except Exception as e:
            logger.error(f"Error getting Yahoo fundamental data: {str(e)}")
            raise 