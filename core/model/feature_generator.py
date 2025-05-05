#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature Engineering for stock market prediction
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
try:
    import talib
except ImportError:
    talib = None
from sklearn.preprocessing import StandardScaler, RobustScaler
from core.technical import TechnicalAnalyzer
from core.data.data_validator import DataValidator

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    Feature Engineer for stock market prediction models
    Prepares technical indicators and other features for ML models
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the FeatureEngineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scaler = None
        self._feature_names = []
        
        # Default configuration
        self.default_config = {
            "use_technical_indicators": True,
            "use_trend_features": True,
            "use_volatility_features": True,
            "use_lagged_features": True,
            "lag_periods": [1, 3, 5, 10, 20],
            "ma_periods": [5, 10, 20, 50, 100],
            "scaling": "standard",  # "standard", "robust" or None
            "fill_method": "ffill"  # Forward fill missing values
        }
        
        # Update config with default values for missing keys
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        logger.info("FeatureEngineer initialized")
        
    def prepare_features(self, data: pd.DataFrame, 
                          tech_analysis: Optional[Dict[str, Any]] = None, 
                          news_sentiment: Optional[Dict[str, Any]] = None,
                          market_condition: Optional[Dict[str, Any]] = None,
                          feature_set: str = "full",
                          for_inference: bool = False) -> pd.DataFrame:
        """
        Prepare features for ML models
        
        Args:
            data: DataFrame with OHLCV data
            tech_analysis: Technical analysis results
            news_sentiment: News sentiment analysis results
            market_condition: Market condition analysis
            feature_set: Type of feature set to create ("full", "minimal", "basic")
            for_inference: If True, removes target columns and ensures feature consistency
            
        Returns:
            DataFrame with features
        """
        if data is None or data.empty:
            logger.error("No data provided for feature engineering")
            return pd.DataFrame()
        
        try:
            # CHUẨN HÓA DỮ LIỆU ĐẦU VÀO
            data = DataValidator.normalize_dataframe(data)
            data = DataValidator.validate_schema(data)
            # Make a copy to avoid modifying the original
            df = data.copy()
            
            # Kiểm tra cột bắt buộc
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Thiếu cột {col} trong dữ liệu đầu vào. Thêm cột này với giá trị 0.")
                    df[col] = 0.0
            
            # Chuyển đổi đảm bảo kiểu dữ liệu số
            for col in df.columns:
                if df[col].dtype != 'object':  # Bỏ qua các cột không phải số
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        logger.warning(f"Không thể chuyển đổi cột {col} sang số: {e}")
            
            # Loại bỏ duplicate columns
            if len(df.columns) != len(set(df.columns)):
                duplicates = df.columns[df.columns.duplicated()].tolist()
                logger.warning(f"Duplicate columns detected: {duplicates}. Keeping first occurrence.")
                df = df.loc[:, ~df.columns.duplicated()]
            
            # Kiểm tra và xử lý dữ liệu NaN trước khi tiếp tục
            nan_columns = df.columns[df.isna().any()].tolist()
            if nan_columns:
                logger.warning(f"Phát hiện dữ liệu NaN trong các cột: {nan_columns}. Áp dụng ffill.")
                df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Sinh feature
            if feature_set == "minimal":
                df = self._generate_minimal_features(df)
            elif feature_set == "basic":
                df = self._generate_basic_features(df)
            else:
                try:
                    df = self._generate_technical_features(df)
                except Exception as e:
                    logger.error(f"Lỗi khi tạo technical features: {e}")
                    
                try:
                    df = self._generate_lagged_features(df)
                except Exception as e:
                    logger.error(f"Lỗi khi tạo lagged features: {e}")
                    
                # Add sentiment and market features if available
                if news_sentiment:
                    try:
                        df = self._add_sentiment_features(df, news_sentiment)
                    except Exception as e:
                        logger.error(f"Lỗi khi thêm sentiment features: {e}")
                
                if market_condition:
                    try:
                        df = self._add_market_features(df, market_condition)
                    except Exception as e:
                        logger.error(f"Lỗi khi thêm market features: {e}")
            
            # Xử lý missing value
            df = self._handle_missing_values(df)
            
            # Scale features if needed
            if self.config["scaling"]:
                df = self._scale_features(df)
            
            # Nếu for_inference, loại bỏ cột target
            if for_inference:
                target_cols = ['target', 'future_close', 'price_change', 'price_pct_change', 'y']
                cols_to_drop = [col for col in target_cols if col in df.columns]
                if cols_to_drop:
                    logger.info(f"Removing target columns for inference: {cols_to_drop}")
                    df = df.drop(columns=cols_to_drop, errors='ignore')
            
            # Validate output
            if df.isnull().any().any():
                logger.warning("Output features có giá trị NaN. Sẽ fill bằng 0.")
                df = df.fillna(0)
            
            # Kiểm tra và loại bỏ các cột có giá trị inf
            inf_columns = np.any(np.isinf(df.select_dtypes(include=[np.number])), axis=0)
            if inf_columns.any():
                inf_col_names = df.select_dtypes(include=[np.number]).columns[inf_columns].tolist()
                logger.warning(f"Phát hiện giá trị inf trong các cột: {inf_col_names}. Thay thế bằng 0.")
                df = df.replace([np.inf, -np.inf], 0)
            
            # Store feature names for later use
            self._feature_names = list(df.columns)
            
            logger.info(f"Prepared {df.shape[1]} features with {df.shape[0]} rows (feature_set: {feature_set})")
            return df
            
        except Exception as e:
            logger.error(f"Error in feature preparation: {str(e)}")
            # Trả về DataFrame cơ bản nếu có lỗi
            if data is not None and not data.empty:
                try:
                    # Cố gắng trả về dữ liệu tối thiểu
                    basic_df = data.copy()
                    if 'close' in basic_df.columns:
                        basic_df['return_1d'] = basic_df['close'].pct_change(1).fillna(0)
                    return basic_df
                except:
                    pass
            return pd.DataFrame()
    
    def _generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate technical indicators as features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical features added
        """
        if not self.config["use_technical_indicators"]:
            return df
        
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning("Missing required columns for technical indicators")
                return df
            
            # Ép kiểu float64 cho các cột OHLCV để tránh lỗi khi dùng TA-Lib
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
            # Copy to avoid warnings about modifying original
            result = df.copy()
            
            # Calculate common indicators using talib
            # Moving Averages
            for period in self.config["ma_periods"]:
                result[f'sma_{period}'] = talib.SMA(result['close'].values, timeperiod=period)
                result[f'ema_{period}'] = talib.EMA(result['close'].values, timeperiod=period)
            
            # Price relative to moving averages
            for period in self.config["ma_periods"]:
                result[f'close_over_sma_{period}'] = result['close'] / result[f'sma_{period}'] - 1
                result[f'close_over_ema_{period}'] = result['close'] / result[f'ema_{period}'] - 1
            
            # RSI
            result['rsi_14'] = talib.RSI(result['close'].values, timeperiod=14)
            result['rsi_7'] = talib.RSI(result['close'].values, timeperiod=7)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(
                result['close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            result['macd'] = macd
            result['macd_signal'] = macd_signal
            result['macd_hist'] = macd_hist
            result['macd_crossover'] = np.where(result['macd'] > result['macd_signal'], 1, -1)
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                result['close'].values, 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2, 
                matype=0
            )
            result['bb_upper'] = upper
            result['bb_middle'] = middle
            result['bb_lower'] = lower
            result['bb_width'] = (upper - lower) / middle
            result['bb_position'] = (result['close'] - lower) / (upper - lower)
            
            # Momentum indicators
            result['mom_10'] = talib.MOM(result['close'].values, timeperiod=10)
            result['roc_10'] = talib.ROC(result['close'].values, timeperiod=10)
            
            # Volume indicators
            result['obv'] = talib.OBV(result['close'].values, result['volume'].values)
            result['obv_ma'] = talib.SMA(result['obv'].values, timeperiod=20)
            result['obv_ratio'] = result['obv'] / result['obv_ma']
            
            # ADX - Trend strength
            result['adx'] = talib.ADX(result['high'].values, result['low'].values, result['close'].values, timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                result['high'].values, 
                result['low'].values, 
                result['close'].values, 
                fastk_period=14, 
                slowk_period=3, 
                slowk_matype=0, 
                slowd_period=3, 
                slowd_matype=0
            )
            result['stoch_k'] = slowk
            result['stoch_d'] = slowd
            
            # Volatility
            if self.config["use_volatility_features"]:
                # ATR - Average True Range
                result['atr'] = talib.ATR(result['high'].values, result['low'].values, result['close'].values, timeperiod=14)
                result['atr_percent'] = result['atr'] / result['close'] * 100
                
                # Historical volatility
                result['returns'] = result['close'].pct_change()
                result['volatility_10'] = result['returns'].rolling(window=10).std() * np.sqrt(252)
                result['volatility_20'] = result['returns'].rolling(window=20).std() * np.sqrt(252)
                
                # High-Low Range
                result['high_low_range'] = (result['high'] - result['low']) / result['close'] * 100
                result['high_low_range_ma10'] = result['high_low_range'].rolling(window=10).mean()
            
            logger.info(f"Generated {result.shape[1] - df.shape[1]} technical features")
            return result
            
        except Exception as e:
            logger.error(f"Error generating technical features: {str(e)}")
            return df
    
    def _generate_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate lagged and rolling features
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with lagged features added
        """
        if not self.config["use_lagged_features"]:
            return df
        
        try:
            result = df.copy()
            
            # Calculate returns
            result['returns'] = result['close'].pct_change()
            
            # Create lagged returns
            for lag in self.config["lag_periods"]:
                result[f'returns_lag_{lag}'] = result['returns'].shift(lag)
            
            # Create rolling return features
            result['returns_mean_5'] = result['returns'].rolling(window=5).mean()
            result['returns_mean_10'] = result['returns'].rolling(window=10).mean()
            result['returns_mean_20'] = result['returns'].rolling(window=20).mean()
            
            # Create rolling std features
            result['returns_std_5'] = result['returns'].rolling(window=5).std()
            result['returns_std_10'] = result['returns'].rolling(window=10).std()
            result['returns_std_20'] = result['returns'].rolling(window=20).std()
            
            # Create rolling min/max features
            result['close_min_5'] = result['close'].rolling(window=5).min() / result['close'] - 1
            result['close_max_5'] = result['close'].rolling(window=5).max() / result['close'] - 1
            result['close_min_10'] = result['close'].rolling(window=10).min() / result['close'] - 1
            result['close_max_10'] = result['close'].rolling(window=10).max() / result['close'] - 1
            
            logger.info(f"Generated {result.shape[1] - df.shape[1]} lagged and rolling features")
            return result
            
        except Exception as e:
            logger.error(f"Error generating lagged features: {str(e)}")
            return df
    
    def _add_sentiment_features(self, df: pd.DataFrame, sentiment: Dict[str, Any]) -> pd.DataFrame:
        """
        Add sentiment analysis features
        
        Args:
            df: DataFrame with price data
            sentiment: Sentiment analysis results
            
        Returns:
            DataFrame with sentiment features added
        """
        try:
            result = df.copy()
            
            # Process today's sentiment
            if 'positive' in sentiment and 'negative' in sentiment:
                result['sentiment_positive'] = sentiment['positive'] / 100.0
                result['sentiment_negative'] = sentiment['negative'] / 100.0
                result['sentiment_ratio'] = sentiment['positive'] / (sentiment['negative'] + 1e-5)  # Avoid division by zero
                result['sentiment_score'] = (sentiment['positive'] - sentiment['negative']) / 100.0
            
            # If we have sentiment history, add it
            if 'history' in sentiment and isinstance(sentiment['history'], list):
                # Convert list to a dataframe with dates
                senti_df = pd.DataFrame(sentiment['history'])
                if 'date' in senti_df.columns and 'score' in senti_df.columns:
                    senti_df['date'] = pd.to_datetime(senti_df['date'])
                    senti_df.set_index('date', inplace=True)
                    
                    # Join with main dataframe on date
                    if isinstance(result.index, pd.DatetimeIndex):
                        result = result.join(senti_df, how='left')
                    else:
                        # If index is not datetime, use nearest date matching
                        logger.warning("DataFrame index is not DatetimeIndex, sentiment history may not align correctly")
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding sentiment features: {str(e)}")
            return df
    
    def _add_market_features(self, df: pd.DataFrame, market: Dict[str, Any]) -> pd.DataFrame:
        """
        Add market condition features
        
        Args:
            df: DataFrame with price data
            market: Market condition analysis
            
        Returns:
            DataFrame with market features added
        """
        try:
            result = df.copy()
            
            # Process market data
            if 'vnindex_change_pct' in market:
                result['market_change'] = market['vnindex_change_pct'] / 100.0
            
            # Market trend as numeric
            if 'market_trend' in market:
                trend = market['market_trend']
                trend_score = 0.0
                
                if "Tăng mạnh" in trend:
                    trend_score = 1.0
                elif "Tăng" in trend:
                    trend_score = 0.5
                elif "Phục hồi" in trend:
                    trend_score = 0.3
                elif "Đi ngang" in trend:
                    trend_score = 0.0
                elif "Điều chỉnh" in trend:
                    trend_score = -0.3
                elif "Giảm" in trend:
                    trend_score = -0.5
                elif "Giảm mạnh" in trend:
                    trend_score = -1.0
                
                result['market_trend_score'] = trend_score
            
            # Market breadth - advancers vs decliners
            if 'advancers' in market and 'decliners' in market:
                total = market['advancers'] + market['decliners'] + market.get('unchanged', 0)
                result['market_breadth'] = (market['advancers'] - market['decliners']) / total if total > 0 else 0
            
            # Market liquidity
            if 'volume' in market and 'avg_volume' in market:
                result['market_volume_ratio'] = market['volume'] / market['avg_volume']
            
            # Market sentiment
            if 'market_sentiment' in market and 'overall' in market['market_sentiment']:
                sentiment = market['market_sentiment']['overall']
                senti_score = 0.0
                
                if sentiment == 'Tích cực':
                    senti_score = 0.7
                elif sentiment == 'Tiêu cực':
                    senti_score = -0.7
                
                result['market_sentiment_score'] = senti_score
            
            return result
            
        except Exception as e:
            logger.error(f"Error adding market features: {str(e)}")
            return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the features
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with missing values handled
        """
        if df.empty:
            return df
        
        try:
            # Get numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Fill numeric missing values
            if self.config["fill_method"] == "ffill":
                df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
                # For any remaining NAs at the beginning, use backfill
                df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
            elif self.config["fill_method"] == "zero":
                df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # For remaining NA values, use column mean
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            # Check if we still have missing values
            if df[numeric_cols].isna().any().any():
                logger.warning("Still have missing values after handling")
                # Fill any remaining with zeros
                df[numeric_cols] = df[numeric_cols].fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale features to standardize them
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with scaled features
        """
        if df.empty:
            return df
        
        try:
            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Skip date/index, target variables and categorical columns
            exclude_cols = ['date', 'close', 'open', 'high', 'low', 'volume', 'returns'] 
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if not feature_cols:
                return df
            
            # Create and fit scaler
            if self.config["scaling"] == "standard":
                self.scaler = StandardScaler()
            elif self.config["scaling"] == "robust":
                self.scaler = RobustScaler()
            else:
                logger.warning(f"Unknown scaling method: {self.config['scaling']}")
                return df
            
            # Fit and transform
            scaled_features = self.scaler.fit_transform(df[feature_cols])
            
            # Create a new dataframe with scaled features
            scaled_df = pd.DataFrame(scaled_features, index=df.index, columns=feature_cols)
            
            # Drop original feature columns and add scaled ones
            result = df.drop(columns=feature_cols)
            for col in feature_cols:
                result[col] = scaled_df[col]
            
            return result
            
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            return df
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features generated
        
        Returns:
            List of feature names
        """
        if not self._feature_names:
            logger.warning("Feature names not available. Run prepare_features first.")
            return []
        return self._feature_names.copy()
    
    def save_scaler(self, filepath: str) -> bool:
        """
        Save the scaler to a file
        
        Args:
            filepath: Path to save the scaler
            
        Returns:
            True if successful, False otherwise
        """
        import joblib
        try:
            if self.scaler is not None:
                joblib.dump(self.scaler, filepath)
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving scaler: {str(e)}")
            return False
    
    def load_scaler(self, filepath: str) -> bool:
        """
        Load the scaler from a file
        
        Args:
            filepath: Path to load the scaler from
            
        Returns:
            True if successful, False otherwise
        """
        import joblib
        try:
            self.scaler = joblib.load(filepath)
            return True
        except Exception as e:
            logger.error(f"Error loading scaler: {str(e)}")
            return False

    def _generate_minimal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate minimal essential features for basic prediction
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with minimal features added
        """
        try:
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.warning("Missing required columns for minimal features")
                return df
            
            # Convert columns to float64
            for col in required_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
            
            # Create a new DataFrame to store results
            result = df.copy()
            
            # Simple price features (returns)
            result['return_1d'] = result['close'].pct_change(1)
            result['return_5d'] = result['close'].pct_change(5)
            
            # Simple moving averages
            result['sma_5'] = talib.SMA(result['close'].values, timeperiod=5)
            result['sma_20'] = talib.SMA(result['close'].values, timeperiod=20)
            
            # Price relative to moving averages
            result['close_over_sma5'] = result['close'] / result['sma_5'] - 1
            result['close_over_sma20'] = result['close'] / result['sma_20'] - 1
            
            # Basic RSI
            result['rsi_14'] = talib.RSI(result['close'].values, timeperiod=14)
            
            # Simple volume features
            result['volume_change'] = result['volume'].pct_change(1)
            result['volume_ma5'] = talib.SMA(result['volume'].values, timeperiod=5)
            result['volume_ratio'] = result['volume'] / result['volume_ma5']
            
            # Basic volatility
            result['high_low_ratio'] = result['high'] / result['low']
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating minimal features: {str(e)}")
            return df
    
    def _generate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate basic features without advanced indicators
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with basic features added
        """
        try:
            # Start with minimal features
            result = self._generate_minimal_features(df)
            
            # Add some intermediate level indicators
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                result['close'].values, 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2, 
                matype=0
            )
            result['bb_upper'] = upper
            result['bb_lower'] = lower
            result['bb_position'] = (result['close'] - lower) / (upper - lower)
            
            # MACD
            macd, macd_signal, _ = talib.MACD(
                result['close'].values, 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            result['macd'] = macd
            result['macd_signal'] = macd_signal
            result['macd_crossover'] = np.where(result['macd'] > result['macd_signal'], 1, -1)
            
            # Momentum
            result['mom_10'] = talib.MOM(result['close'].values, timeperiod=10)
            
            # Simple trend feature
            result['is_uptrend'] = np.where(result['sma_5'] > result['sma_20'], 1, 0)
            
            # Add a few lagged features
            result['close_lag1'] = result['close'].shift(1)
            result['close_lag5'] = result['close'].shift(5)
            result['return_lag1'] = result['return_1d'].shift(1)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating basic features: {str(e)}")
            return df 

    def get_report_indicators(self, df: pd.DataFrame) -> dict:
        """
        Lấy chỉ số kỹ thuật chuẩn hóa cho báo cáo, không tự tính lại MA, RSI, MACD.
        """
        return TechnicalAnalyzer.get_technical_indicators(df) 

    def generate_prophet_features(self, df: pd.DataFrame, date_col: str = 'date', value_col: str = 'close') -> pd.DataFrame:
        """
        Chuẩn hóa dữ liệu cho Prophet: cột 'ds' (datetime), 'y' (giá trị)
        """
        prophet_df = df.copy()
        if date_col not in prophet_df.columns:
            prophet_df['ds'] = pd.to_datetime(prophet_df.index)
        else:
            prophet_df['ds'] = pd.to_datetime(prophet_df[date_col])
        prophet_df['y'] = prophet_df[value_col] if value_col in prophet_df.columns else prophet_df.iloc[:, 0]
        prophet_df = prophet_df[['ds', 'y']].dropna().sort_values('ds')
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize('Asia/Ho_Chi_Minh', ambiguous='NaT', nonexistent='shift_forward') if prophet_df['ds'].dt.tz is None else prophet_df['ds']
        return prophet_df 

    def get_features_for_inference(self, data: pd.DataFrame,
                                   tech_analysis: Optional[Dict[str, Any]] = None,
                                   news_sentiment: Optional[Dict[str, Any]] = None,
                                   market_condition: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Prepare features specifically for model inference
        
        Args:
            data: DataFrame with OHLCV data
            tech_analysis: Technical analysis results
            news_sentiment: News sentiment analysis results
            market_condition: Market condition analysis
            
        Returns:
            DataFrame with features ready for model inference
        """
        return self.prepare_features(data, tech_analysis, news_sentiment, market_condition, 
                                   feature_set="full", for_inference=True) 