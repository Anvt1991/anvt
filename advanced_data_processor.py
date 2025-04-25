#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced Data Processor Module for Stock Market Bot:
- Provides sophisticated methods for handling financial data
- Multiple strategies for outlier detection and handling
- Fill missing data with various imputation techniques
- Create rich derived features for technical analysis
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy import stats
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Create module-specific logger
logger = logging.getLogger(__name__)

class AdvancedDataProcessor:
    """
    Advanced data processing for financial time series
    
    Provides methods for:
    - Outlier detection and treatment
    - Missing data imputation
    - Feature engineering
    - Data transformation
    - Data normalization
    """
    
    def __init__(self, redis_manager=None):
        """Initialize the advanced data processor"""
        self.redis_manager = redis_manager
    
    # ----- OUTLIER DETECTION AND HANDLING -----
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'zscore', 
                        column: str = 'close', threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in financial data using various methods
        
        Parameters:
        -----------
        df : pandas DataFrame
            The data to process
        method : str
            Method to use ('zscore', 'iqr', 'isolation_forest', 'dbscan', 'rolling')
        column : str
            Column to check for outliers
        threshold : float
            Threshold for outlier detection (meaning depends on method)
            
        Returns:
        --------
        DataFrame with 'is_outlier' column added
        """
        if df is None or df.empty or column not in df.columns:
            return df
        
        # Create a copy to avoid modifying the original
        df_result = df.copy()
        
        try:
            if method == 'zscore':
                # Z-score method (standard deviations from mean)
                z_scores = np.abs((df_result[column] - df_result[column].mean()) / df_result[column].std())
                df_result['is_outlier'] = z_scores > threshold
                
            elif method == 'iqr':
                # IQR method (Inter-quartile range)
                Q1 = df_result[column].quantile(0.25)
                Q3 = df_result[column].quantile(0.75)
                IQR = Q3 - Q1
                df_result['is_outlier'] = ((df_result[column] < (Q1 - threshold * IQR)) | 
                                          (df_result[column] > (Q3 + threshold * IQR)))
                
            elif method == 'isolation_forest':
                # Isolation Forest (more sophisticated ML-based method)
                clf = IsolationForest(contamination=threshold/10, random_state=42)
                df_result['is_outlier'] = clf.fit_predict(df_result[[column]]) == -1
                
            elif method == 'dbscan':
                # DBSCAN clustering-based method
                X = df_result[[column]].values
                db = DBSCAN(eps=threshold, min_samples=5).fit(X)
                df_result['is_outlier'] = db.labels_ == -1
                
            elif method == 'rolling':
                # Rolling window method (good for time series)
                window = min(20, len(df) // 4)  # Adaptive window size
                rolling_mean = df_result[column].rolling(window=window, center=True).mean()
                rolling_std = df_result[column].rolling(window=window, center=True).std()
                df_result['is_outlier'] = np.abs(df_result[column] - rolling_mean) > (threshold * rolling_std)
                
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
        
        except Exception as e:
            logger.error(f"Error in outlier detection ({method}): {str(e)}")
            df_result['is_outlier'] = False
        
        return df_result
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'winsorize', 
                        column: str = 'close', outlier_column: str = 'is_outlier') -> pd.DataFrame:
        """
        Handle detected outliers with various methods
        
        Parameters:
        -----------
        df : pandas DataFrame
            Data with outliers detected (has outlier_column)
        method : str
            Method to handle outliers: 'winsorize', 'replace_mean', 'replace_median', 'replace_nearest', 'remove'
        column : str
            Column to process for outliers
        outlier_column : str
            Column indicating outliers (boolean)
            
        Returns:
        --------
        DataFrame with outliers handled
        """
        if df is None or df.empty or column not in df.columns or outlier_column not in df.columns:
            return df
        
        # Create a copy to avoid modifying the original
        df_result = df.copy()
        
        # Skip if no outliers
        if not df_result[outlier_column].any():
            return df_result
        
        try:
            if method == 'winsorize':
                # Winsorize - replace with values at percentile limits
                # Good for financial data as it preserves trends
                limits = (0.01, 0.01)  # 1% on each end
                winsorized = stats.mstats.winsorize(df_result[column].values, limits=limits)
                df_result.loc[df_result[outlier_column], column] = winsorized[df_result[outlier_column]]
                
            elif method == 'replace_mean':
                # Replace with mean
                mean_value = df_result[~df_result[outlier_column]][column].mean()
                df_result.loc[df_result[outlier_column], column] = mean_value
                
            elif method == 'replace_median':
                # Replace with median (more robust than mean)
                median_value = df_result[~df_result[outlier_column]][column].median()
                df_result.loc[df_result[outlier_column], column] = median_value
                
            elif method == 'replace_nearest':
                # Replace with nearest non-outlier value
                for idx in df_result[df_result[outlier_column]].index:
                    # Find closest non-outlier index
                    non_outlier_idx = df_result[~df_result[outlier_column]].index
                    if len(non_outlier_idx) > 0:
                        # Find closest datetime index
                        time_diffs = abs(non_outlier_idx - idx)
                        closest_idx = non_outlier_idx[np.argmin(time_diffs)]
                        df_result.loc[idx, column] = df_result.loc[closest_idx, column]
                
            elif method == 'remove':
                # Simply remove outliers (not recommended for time series)
                df_result = df_result[~df_result[outlier_column]]
                
            else:
                raise ValueError(f"Unsupported outlier handling method: {method}")
                
            # Ensure other columns are consistent with the handled outliers
            if method != 'remove' and column == 'close':
                # Update high/low if necessary
                for idx in df_result[df_result[outlier_column]].index:
                    # Ensure high >= close and low <= close
                    if 'high' in df_result.columns and df_result.loc[idx, 'high'] < df_result.loc[idx, column]:
                        df_result.loc[idx, 'high'] = df_result.loc[idx, column]
                    if 'low' in df_result.columns and df_result.loc[idx, 'low'] > df_result.loc[idx, column]:
                        df_result.loc[idx, 'low'] = df_result.loc[idx, column]
            
        except Exception as e:
            logger.error(f"Error in outlier handling ({method}): {str(e)}")
        
        return df_result
    
    # ----- MISSING DATA HANDLING -----
    
    def detect_missing_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect and analyze missing data in the DataFrame
        
        Parameters:
        -----------
        df : pandas DataFrame
            The data to check for missing values
            
        Returns:
        --------
        Tuple with:
        - Original DataFrame
        - Dictionary with missing data statistics
        """
        if df is None or df.empty:
            return df, {"error": "Empty DataFrame"}
        
        # Calculate missing data statistics
        missing_count = df.isna().sum()
        missing_percent = (missing_count / len(df)) * 100
        
        # Get columns with missing data
        cols_with_missing = missing_percent[missing_percent > 0].index.tolist()
        
        # Check for missing dates in time series
        if isinstance(df.index, pd.DatetimeIndex):
            # Check for gaps in dates
            date_gaps = []
            if len(df) > 1:
                sorted_df = df.sort_index()
                date_diffs = sorted_df.index[1:] - sorted_df.index[:-1]
                median_diff = pd.Timedelta(seconds=np.median([d.total_seconds() for d in date_diffs]))
                
                # Find gaps larger than 2x median
                large_gaps = date_diffs > (2 * median_diff)
                if large_gaps.any():
                    gap_indices = np.where(large_gaps)[0]
                    for i in gap_indices:
                        gap_start = sorted_df.index[i]
                        gap_end = sorted_df.index[i+1]
                        date_gaps.append((gap_start, gap_end, gap_end - gap_start))
        else:
            date_gaps = "Not a time series with DatetimeIndex"
        
        # Prepare missing data report
        missing_data_stats = {
            "missing_counts": missing_count.to_dict(),
            "missing_percent": missing_percent.to_dict(),
            "columns_with_missing": cols_with_missing,
            "date_gaps": date_gaps if isinstance(date_gaps, list) else None,
            "total_missing_cells": missing_count.sum(),
            "total_cells": df.size,
            "overall_missing_percent": (missing_count.sum() / df.size) * 100
        }
        
        return df, missing_data_stats
    
    def fill_missing_data(self, df: pd.DataFrame, method: str = 'ffill', 
                          columns: List[str] = None) -> pd.DataFrame:
        """
        Fill missing data using various imputation methods
        
        Parameters:
        -----------
        df : pandas DataFrame
            Data with missing values
        method : str
            Method to fill missing values:
            - 'ffill': Forward fill (good for time series)
            - 'bfill': Backward fill 
            - 'linear': Linear interpolation
            - 'cubic': Cubic spline interpolation
            - 'mean': Fill with column mean
            - 'median': Fill with column median
            - 'knn': K-Nearest Neighbors imputation
            - 'time_weighted': Time-weighted average (for time series)
        columns : list of str
            Columns to fill (None = all columns)
            
        Returns:
        --------
        DataFrame with missing values filled
        """
        if df is None or df.empty:
            return df
        
        # Create a copy to avoid modifying the original
        df_result = df.copy()
        
        # Select columns to process
        cols_to_fill = columns if columns is not None else df_result.columns
        cols_to_fill = [c for c in cols_to_fill if c in df_result.columns]
        
        try:
            if method == 'ffill':
                # Forward fill (use previous valid value)
                df_result[cols_to_fill] = df_result[cols_to_fill].fillna(method='ffill')
                
            elif method == 'bfill':
                # Backward fill (use next valid value)
                df_result[cols_to_fill] = df_result[cols_to_fill].fillna(method='bfill')
                
            elif method == 'linear':
                # Linear interpolation
                df_result[cols_to_fill] = df_result[cols_to_fill].interpolate(method='linear')
                
            elif method == 'cubic':
                # Cubic spline interpolation (smoother than linear)
                df_result[cols_to_fill] = df_result[cols_to_fill].interpolate(method='cubic')
                
            elif method == 'mean':
                # Fill with column mean
                for col in cols_to_fill:
                    mean_val = df_result[col].mean()
                    df_result[col] = df_result[col].fillna(mean_val)
                
            elif method == 'median':
                # Fill with column median (more robust than mean)
                for col in cols_to_fill:
                    median_val = df_result[col].median()
                    df_result[col] = df_result[col].fillna(median_val)
                
            elif method == 'knn':
                # K-Nearest Neighbors imputation
                imputer = KNNImputer(n_neighbors=5)
                df_imputed = pd.DataFrame(
                    imputer.fit_transform(df_result[cols_to_fill]),
                    columns=cols_to_fill,
                    index=df_result.index
                )
                df_result[cols_to_fill] = df_imputed
                
            elif method == 'time_weighted':
                # Time-weighted average - for time series data
                if not isinstance(df_result.index, pd.DatetimeIndex):
                    raise ValueError("Time-weighted imputation requires DatetimeIndex")
                
                # For each column with missing values
                for col in cols_to_fill:
                    # Get indices with missing values
                    missing_idx = df_result[df_result[col].isna()].index
                    
                    for idx in missing_idx:
                        # Find nearest valid values before and after
                        valid_before = df_result.loc[:idx, col].dropna()
                        valid_after = df_result.loc[idx:, col].dropna()
                        
                        if not valid_before.empty and not valid_after.empty:
                            before_idx = valid_before.index[-1]
                            after_idx = valid_after.index[0]
                            
                            if before_idx != after_idx:  # Ensure we have different points
                                before_val = df_result.loc[before_idx, col]
                                after_val = df_result.loc[after_idx, col]
                                
                                # Calculate time differences
                                t_diff_before = (idx - before_idx).total_seconds()
                                t_diff_after = (after_idx - idx).total_seconds()
                                t_diff_total = t_diff_before + t_diff_after
                                
                                # Time-weighted average
                                weight_after = t_diff_before / t_diff_total
                                weight_before = t_diff_after / t_diff_total
                                weighted_val = (before_val * weight_before) + (after_val * weight_after)
                                
                                df_result.loc[idx, col] = weighted_val
                
            else:
                raise ValueError(f"Unsupported missing data imputation method: {method}")
                
            # Handle any remaining missing values with forward fill
            df_result[cols_to_fill] = df_result[cols_to_fill].fillna(method='ffill')
            df_result[cols_to_fill] = df_result[cols_to_fill].fillna(method='bfill')
            
        except Exception as e:
            logger.error(f"Error in missing data imputation ({method}): {str(e)}")
        
        return df_result
    
    # ----- DERIVED FEATURES -----
    
    def create_derived_features(self, df: pd.DataFrame, feature_set: str = 'basic') -> pd.DataFrame:
        """
        Create derived features for technical analysis
        
        Parameters:
        -----------
        df : pandas DataFrame
            OHLCV data
        feature_set : str
            Set of features to create:
            - 'basic': Basic price and volume features
            - 'momentum': Momentum indicators
            - 'trend': Trend indicators
            - 'volatility': Volatility indicators
            - 'candle': Candlestick patterns
            - 'volume': Volume indicators
            - 'all': All available features
            
        Returns:
        --------
        DataFrame with additional feature columns
        """
        if df is None or df.empty:
            return df
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for derived features: {missing_cols}")
        
        # Create a copy to avoid modifying the original
        df_result = df.copy()
        
        try:
            # Basic Price and Return features
            if feature_set in ['basic', 'all']:
                # Price changes and returns
                df_result['return_1d'] = df_result['close'].pct_change(1)
                df_result['return_5d'] = df_result['close'].pct_change(5)
                df_result['log_return'] = np.log(df_result['close'] / df_result['close'].shift(1))
                
                # Price ranges
                df_result['range'] = df_result['high'] - df_result['low']
                df_result['range_pct'] = df_result['range'] / df_result['close']
                
                # Body size
                df_result['body_size'] = np.abs(df_result['close'] - df_result['open'])
                df_result['body_pct'] = df_result['body_size'] / df_result['close']
                
                # Upper and lower shadows
                df_result['upper_shadow'] = df_result.apply(
                    lambda x: x['high'] - max(x['open'], x['close']), axis=1)
                df_result['lower_shadow'] = df_result.apply(
                    lambda x: min(x['open'], x['close']) - x['low'], axis=1)
                
                # Volume features
                df_result['volume_change'] = df_result['volume'].pct_change(1)
                df_result['volume_ma5'] = df_result['volume'].rolling(window=5).mean()
                df_result['volume_ma20'] = df_result['volume'].rolling(window=20).mean()
                df_result['volume_ratio'] = df_result['volume'] / df_result['volume_ma20']
            
            # Momentum indicators
            if feature_set in ['momentum', 'all']:
                # RSI
                delta = df_result['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                df_result['rsi_14'] = 100 - (100 / (1 + rs))
                
                # Stochastic Oscillator
                low_14 = df_result['low'].rolling(window=14).min()
                high_14 = df_result['high'].rolling(window=14).max()
                df_result['stoch_k'] = 100 * ((df_result['close'] - low_14) / (high_14 - low_14))
                df_result['stoch_d'] = df_result['stoch_k'].rolling(window=3).mean()
                
                # ROC (Rate of Change)
                df_result['roc_12'] = ((df_result['close'] / df_result['close'].shift(12)) - 1) * 100
                
                # Williams %R
                df_result['williams_r'] = -100 * ((high_14 - df_result['close']) / (high_14 - low_14))
            
            # Trend indicators
            if feature_set in ['trend', 'all']:
                # Moving Averages
                df_result['sma_5'] = df_result['close'].rolling(window=5).mean()
                df_result['sma_10'] = df_result['close'].rolling(window=10).mean()
                df_result['sma_20'] = df_result['close'].rolling(window=20).mean()
                df_result['sma_50'] = df_result['close'].rolling(window=50).mean()
                df_result['sma_200'] = df_result['close'].rolling(window=200).mean()
                
                # Exponential Moving Averages
                df_result['ema_12'] = df_result['close'].ewm(span=12, adjust=False).mean()
                df_result['ema_26'] = df_result['close'].ewm(span=26, adjust=False).mean()
                
                # MACD
                df_result['macd'] = df_result['ema_12'] - df_result['ema_26']
                df_result['macd_signal'] = df_result['macd'].ewm(span=9, adjust=False).mean()
                df_result['macd_hist'] = df_result['macd'] - df_result['macd_signal']
                
                # Moving Average Crossovers
                df_result['cross_5_20'] = (df_result['sma_5'] > df_result['sma_20']).astype(int)
                df_result['cross_50_200'] = (df_result['sma_50'] > df_result['sma_200']).astype(int)
                
                # ADX (Average Directional Index)
                # Simplified calculation
                df_result['tr'] = np.maximum(
                    df_result['high'] - df_result['low'],
                    np.maximum(
                        np.abs(df_result['high'] - df_result['close'].shift(1)),
                        np.abs(df_result['low'] - df_result['close'].shift(1))
                    )
                )
                df_result['atr_14'] = df_result['tr'].rolling(window=14).mean()
            
            # Volatility indicators
            if feature_set in ['volatility', 'all']:
                # ATR (Average True Range)
                df_result['atr_14'] = df_result['tr'].rolling(window=14).mean() if 'tr' in df_result.columns else np.nan
                
                # Bollinger Bands
                df_result['bb_middle'] = df_result['close'].rolling(window=20).mean()
                df_result['bb_std'] = df_result['close'].rolling(window=20).std()
                df_result['bb_upper'] = df_result['bb_middle'] + (2 * df_result['bb_std'])
                df_result['bb_lower'] = df_result['bb_middle'] - (2 * df_result['bb_std'])
                df_result['bb_width'] = (df_result['bb_upper'] - df_result['bb_lower']) / df_result['bb_middle']
                
                # Historical Volatility
                df_result['hist_vol_20'] = df_result['log_return'].rolling(window=20).std() * np.sqrt(252)
                
                # Chaikin Volatility
                df_result['chaikin_vol'] = (df_result['high'] - df_result['low']).rolling(window=10).std()
            
            # Candlestick patterns
            if feature_set in ['candle', 'all']:
                # Doji (open and close are almost equal)
                doji_threshold = 0.001  # 0.1% of price
                df_result['doji'] = (np.abs(df_result['close'] - df_result['open']) <= 
                                   doji_threshold * df_result['close']).astype(int)
                
                # Hammer (lower shadow is at least twice the body size)
                df_result['hammer'] = ((df_result['lower_shadow'] >= 2 * df_result['body_size']) & 
                                     (df_result['upper_shadow'] <= 0.5 * df_result['body_size'])).astype(int)
                
                # Shooting Star (upper shadow is at least twice the body size)
                df_result['shooting_star'] = ((df_result['upper_shadow'] >= 2 * df_result['body_size']) & 
                                           (df_result['lower_shadow'] <= 0.5 * df_result['body_size'])).astype(int)
                
                # Bullish/Bearish Engulfing
                df_result['bullish_engulfing'] = ((df_result['open'] < df_result['close']) & 
                                               (df_result['open'] <= df_result['close'].shift(1)) & 
                                               (df_result['close'] >= df_result['open'].shift(1)) & 
                                               (df_result['open'].shift(1) > df_result['close'].shift(1))).astype(int)
                
                df_result['bearish_engulfing'] = ((df_result['open'] > df_result['close']) & 
                                               (df_result['open'] >= df_result['close'].shift(1)) & 
                                               (df_result['close'] <= df_result['open'].shift(1)) & 
                                               (df_result['open'].shift(1) < df_result['close'].shift(1))).astype(int)
            
            # Volume indicators
            if feature_set in ['volume', 'all']:
                # On-Balance Volume (OBV)
                obv = [0]
                for i in range(1, len(df_result)):
                    if df_result['close'].iloc[i] > df_result['close'].iloc[i-1]:
                        obv.append(obv[-1] + df_result['volume'].iloc[i])
                    elif df_result['close'].iloc[i] < df_result['close'].iloc[i-1]:
                        obv.append(obv[-1] - df_result['volume'].iloc[i])
                    else:
                        obv.append(obv[-1])
                df_result['obv'] = obv
                
                # Volume Weighted Average Price (VWAP)
                df_result['vwap'] = (df_result['volume'] * (df_result['high'] + df_result['low'] + df_result['close']) / 3).cumsum() / df_result['volume'].cumsum()
                
                # Chaikin A/D Line
                mfm = ((df_result['close'] - df_result['low']) - (df_result['high'] - df_result['close'])) / (df_result['high'] - df_result['low'])
                mfm = mfm.replace([np.inf, -np.inf], 0)
                mfv = mfm * df_result['volume']
                df_result['chaikin_ad'] = mfv.cumsum()
                
                # Chaikin Money Flow
                df_result['cmf'] = mfv.rolling(window=20).sum() / df_result['volume'].rolling(window=20).sum()
                
                # Money Flow Index (MFI)
                typical_price = (df_result['high'] + df_result['low'] + df_result['close']) / 3
                raw_money_flow = typical_price * df_result['volume']
                
                positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
                negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
                
                positive_mf = positive_flow.rolling(window=14).sum()
                negative_mf = negative_flow.rolling(window=14).sum()
                
                mf_ratio = positive_mf / negative_mf
                df_result['mfi'] = 100 - (100 / (1 + mf_ratio))
            
        except Exception as e:
            logger.error(f"Error creating derived features ({feature_set}): {str(e)}")
            
        return df_result
    
    # ----- DATA TRANSFORMATION -----
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax',
                      columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize data using various methods
        
        Parameters:
        -----------
        df : pandas DataFrame
            Data to normalize
        method : str
            Normalization method:
            - 'minmax': Scale to [0, 1] range
            - 'zscore': Standardize to zero mean and unit variance
            - 'robust': Robust scaling using IQR
            - 'log': Natural logarithm transformation
            - 'pct_change': Percentage change from previous
        columns : list of str
            Columns to normalize (None = all numeric columns)
            
        Returns:
        --------
        DataFrame with normalized values
        """
        if df is None or df.empty:
            return df
        
        # Create a copy to avoid modifying the original
        df_result = df.copy()
        
        # Select numeric columns if none specified
        if columns is None:
            columns = df_result.select_dtypes(include=['number']).columns.tolist()
        else:
            # Ensure all specified columns exist
            columns = [c for c in columns if c in df_result.columns]
        
        try:
            if method == 'minmax':
                # Min-max scaling to [0, 1]
                for col in columns:
                    min_val = df_result[col].min()
                    max_val = df_result[col].max()
                    if max_val > min_val:
                        df_result[f"{col}_norm"] = (df_result[col] - min_val) / (max_val - min_val)
                    else:
                        df_result[f"{col}_norm"] = 0.5  # Handle constant values
                
            elif method == 'zscore':
                # Z-score standardization
                for col in columns:
                    mean_val = df_result[col].mean()
                    std_val = df_result[col].std()
                    if std_val > 0:
                        df_result[f"{col}_norm"] = (df_result[col] - mean_val) / std_val
                    else:
                        df_result[f"{col}_norm"] = 0  # Handle constant values
                
            elif method == 'robust':
                # Robust scaling using IQR
                for col in columns:
                    q1 = df_result[col].quantile(0.25)
                    q3 = df_result[col].quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        df_result[f"{col}_norm"] = (df_result[col] - q1) / iqr
                    else:
                        df_result[f"{col}_norm"] = 0  # Handle constant values
                
            elif method == 'log':
                # Log transformation (handles zeros by adding 1)
                for col in columns:
                    if (df_result[col] <= 0).any():
                        min_val = abs(df_result[col].min()) + 1 if df_result[col].min() < 0 else 1
                        df_result[f"{col}_norm"] = np.log(df_result[col] + min_val)
                    else:
                        df_result[f"{col}_norm"] = np.log(df_result[col])
                
            elif method == 'pct_change':
                # Percentage change
                for col in columns:
                    df_result[f"{col}_norm"] = df_result[col].pct_change()
                
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
                
        except Exception as e:
            logger.error(f"Error in data normalization ({method}): {str(e)}")
            
        return df_result
    
    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage by converting data types
        
        Parameters:
        -----------
        df : pandas DataFrame
            Data to optimize
            
        Returns:
        --------
        Memory-optimized DataFrame
        """
        if df is None or df.empty:
            return df
        
        df_result = df.copy()
        
        # Record original memory usage
        original_mem = df_result.memory_usage(deep=True).sum()
        
        # Optimize integers
        int_cols = df_result.select_dtypes(include=['int']).columns
        for col in int_cols:
            df_result[col] = pd.to_numeric(df_result[col], downcast='integer')
        
        # Optimize floats
        float_cols = df_result.select_dtypes(include=['float']).columns
        for col in float_cols:
            df_result[col] = pd.to_numeric(df_result[col], downcast='float')
        
        # Optimize categoricals (for columns with few unique values)
        obj_cols = df_result.select_dtypes(include=['object']).columns
        for col in obj_cols:
            num_unique = df_result[col].nunique()
            if num_unique < len(df_result) * 0.5:  # If less than 50% unique values
                df_result[col] = df_result[col].astype('category')
        
        # Report memory savings
        new_mem = df_result.memory_usage(deep=True).sum()
        savings = (1 - new_mem/original_mem) * 100
        logger.info(f"Memory optimization: {original_mem/1024**2:.2f} MB → {new_mem/1024**2:.2f} MB ({savings:.2f}% saved)")
        
        return df_result 