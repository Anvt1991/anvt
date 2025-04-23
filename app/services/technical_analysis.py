import logging
import pandas as pd
import numpy as np

# Thử import ta-lib, nếu không có thì sử dụng các hàm thay thế
try:
    import talib as ta
    TALIB_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("TA-Lib được tìm thấy và sẽ được sử dụng cho phân tích kỹ thuật.")
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib không khả dụng. Sẽ sử dụng các hàm thay thế với pandas.")

from app.utils.config import TECHNICAL_INDICATOR_PERIODS

class TechnicalAnalyzer:
    """
    Performs technical analysis calculations on price data
    """
    
    @staticmethod
    def _calculate_common_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate common technical indicators."""
        if df is None or df.empty:
            return df
            
        result_df = df.copy()
        
        # Make sure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in result_df.columns:
                logger.warning(f"Thiếu cột {col} trong dữ liệu, một số chỉ báo sẽ không được tính")
        
        if TALIB_AVAILABLE:
            # Sử dụng TA-Lib nếu có
            # Moving Averages for multiple periods
            for period in TECHNICAL_INDICATOR_PERIODS:
                if 'close' in result_df.columns:
                    result_df[f'sma_{period}'] = ta.SMA(result_df['close'], timeperiod=period)
                    result_df[f'ema_{period}'] = ta.EMA(result_df['close'], timeperiod=period)
                    
            # Bollinger Bands (20, 2)
            if 'close' in result_df.columns:
                upper, middle, lower = ta.BBANDS(
                    result_df['close'], 
                    timeperiod=20, 
                    nbdevup=2, 
                    nbdevdn=2, 
                    matype=0
                )
                result_df['bb_upper'] = upper
                result_df['bb_middle'] = middle
                result_df['bb_lower'] = lower
                
            # RSI (14)
            if 'close' in result_df.columns:
                result_df['rsi_14'] = ta.RSI(result_df['close'], timeperiod=14)
                
            # MACD (12, 26, 9)
            if 'close' in result_df.columns:
                macd, macd_signal, macd_hist = ta.MACD(
                    result_df['close'], 
                    fastperiod=12, 
                    slowperiod=26, 
                    signalperiod=9
                )
                result_df['macd'] = macd
                result_df['macd_signal'] = macd_signal
                result_df['macd_hist'] = macd_hist
                
            # Stochastic (14, 3, 3)
            if all(col in result_df.columns for col in ['high', 'low', 'close']):
                slowk, slowd = ta.STOCH(
                    result_df['high'], 
                    result_df['low'], 
                    result_df['close'], 
                    fastk_period=14, 
                    slowk_period=3, 
                    slowk_matype=0, 
                    slowd_period=3, 
                    slowd_matype=0
                )
                result_df['stoch_k'] = slowk
                result_df['stoch_d'] = slowd
                
            # ADX (14)
            if all(col in result_df.columns for col in ['high', 'low', 'close']):
                result_df['adx'] = ta.ADX(
                    result_df['high'], 
                    result_df['low'], 
                    result_df['close'], 
                    timeperiod=14
                )
                
            # OBV - On Balance Volume
            if all(col in result_df.columns for col in ['close', 'volume']):
                result_df['obv'] = ta.OBV(result_df['close'], result_df['volume'])
                
            # ATR - Average True Range (14)
            if all(col in result_df.columns for col in ['high', 'low', 'close']):
                result_df['atr'] = ta.ATR(
                    result_df['high'], 
                    result_df['low'], 
                    result_df['close'], 
                    timeperiod=14
                )
        else:
            # Sử dụng pandas và numpy khi không có TA-Lib
            logger.info("Sử dụng pandas để tính toán các chỉ báo kỹ thuật")
            
            # Moving Averages for multiple periods
            for period in TECHNICAL_INDICATOR_PERIODS:
                if 'close' in result_df.columns:
                    # SMA - Simple Moving Average
                    result_df[f'sma_{period}'] = result_df['close'].rolling(window=period).mean()
                    # EMA - Exponential Moving Average
                    result_df[f'ema_{period}'] = result_df['close'].ewm(span=period, adjust=False).mean()
            
            # Bollinger Bands (20, 2)
            if 'close' in result_df.columns:
                result_df['bb_middle'] = result_df['close'].rolling(window=20).mean()
                result_df['bb_std'] = result_df['close'].rolling(window=20).std()
                result_df['bb_upper'] = result_df['bb_middle'] + 2 * result_df['bb_std']
                result_df['bb_lower'] = result_df['bb_middle'] - 2 * result_df['bb_std']
                result_df.drop('bb_std', axis=1, inplace=True, errors='ignore')
            
            # RSI (14)
            if 'close' in result_df.columns:
                delta = result_df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                # Phương pháp SMA
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                
                # Tính RSI
                rs = avg_gain / avg_loss
                result_df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD (12, 26, 9)
            if 'close' in result_df.columns:
                ema12 = result_df['close'].ewm(span=12, adjust=False).mean()
                ema26 = result_df['close'].ewm(span=26, adjust=False).mean()
                result_df['macd'] = ema12 - ema26
                result_df['macd_signal'] = result_df['macd'].ewm(span=9, adjust=False).mean()
                result_df['macd_hist'] = result_df['macd'] - result_df['macd_signal']
                
            # Stochastic Oscillator (14, 3, 3)
            if all(col in result_df.columns for col in ['high', 'low', 'close']):
                # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
                low_14 = result_df['low'].rolling(window=14).min()
                high_14 = result_df['high'].rolling(window=14).max()
                result_df['stoch_k'] = 100 * ((result_df['close'] - low_14) / (high_14 - low_14))
                result_df['stoch_d'] = result_df['stoch_k'].rolling(window=3).mean()
                
            # OBV - On Balance Volume - Đơn giản hóa
            if all(col in result_df.columns for col in ['close', 'volume']):
                # Khởi tạo cột OBV với giá trị đầu tiên là khối lượng
                obv = pd.Series(index=result_df.index, dtype='float64')
                obv.iloc[0] = 0
                
                # Tính OBV
                for i in range(1, len(result_df)):
                    if result_df['close'].iloc[i] > result_df['close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] + result_df['volume'].iloc[i]
                    elif result_df['close'].iloc[i] < result_df['close'].iloc[i-1]:
                        obv.iloc[i] = obv.iloc[i-1] - result_df['volume'].iloc[i]
                    else:
                        obv.iloc[i] = obv.iloc[i-1]
                        
                result_df['obv'] = obv
                
            # ATR - Average True Range (14) - Đơn giản hóa
            if all(col in result_df.columns for col in ['high', 'low', 'close']):
                # True Range
                tr1 = result_df['high'] - result_df['low']
                tr2 = abs(result_df['high'] - result_df['close'].shift())
                tr3 = abs(result_df['low'] - result_df['close'].shift())
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                # ATR
                result_df['atr'] = tr.rolling(window=14).mean()
                
            # ADX không được triển khai vì quá phức tạp nếu không có ta-lib
            result_df['adx'] = np.nan
            
        return result_df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for a DataFrame."""
        # First calculate common indicators
        df_with_indicators = self._calculate_common_indicators(df)
        
        # Add more complex or custom indicators here
        
        return df_with_indicators
    
    def calculate_multi_timeframe_indicators(self, dfs: dict) -> dict:
        """Calculate indicators for multiple timeframes."""
        result = {}
        
        for timeframe, df in dfs.items():
            # Skip if the dataframe is empty
            if df is None or df.empty:
                logger.warning(f"Dataframe cho timeframe {timeframe} rỗng, bỏ qua")
                continue
                
            # Calculate indicators for this timeframe
            result[timeframe] = self.calculate_indicators(df)
            
        return result

    def detect_patterns(self, dfs: dict) -> dict:
        """Detect candlestick patterns in the data."""
        patterns = {}
        
        for timeframe, df in dfs.items():
            if df is None or df.empty:
                continue
                
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Dictionary to store pattern detection results
                timeframe_patterns = {}
                
                if TALIB_AVAILABLE:
                    # Detect với TA-Lib nếu có
                    # Detect bullish patterns
                    timeframe_patterns['cdl_hammer'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
                    timeframe_patterns['cdl_morning_star'] = ta.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
                    timeframe_patterns['cdl_engulfing_bullish'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
                    timeframe_patterns['cdl_piercing'] = ta.CDLPIERCING(df['open'], df['high'], df['low'], df['close'])
                    
                    # Detect bearish patterns
                    timeframe_patterns['cdl_shooting_star'] = ta.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
                    timeframe_patterns['cdl_evening_star'] = ta.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
                    timeframe_patterns['cdl_hanging_man'] = ta.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
                    timeframe_patterns['cdl_dark_cloud_cover'] = ta.CDLDARKCLOUDCOVER(df['open'], df['high'], df['low'], df['close'])
                    
                    # Add more patterns as needed
                    timeframe_patterns['cdl_doji'] = ta.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
                    timeframe_patterns['cdl_three_inside'] = ta.CDL3INSIDE(df['open'], df['high'], df['low'], df['close'])
                else:
                    # Detect patterns với logic đơn giản nếu không có TA-Lib
                    logger.info("Sử dụng logic đơn giản để phát hiện mẫu hình nến")
                    
                    # Tạo Series với giá trị 0 
                    zeros = pd.Series(0, index=df.index)
                    
                    # Detect Doji (Simplified)
                    body_size = abs(df['close'] - df['open'])
                    avg_body_size = body_size.rolling(window=14).mean()
                    is_doji = body_size < (0.1 * avg_body_size)
                    timeframe_patterns['cdl_doji'] = zeros.copy()
                    timeframe_patterns['cdl_doji'][is_doji] = 100
                    
                    # Detect Bullish Engulfing (Simplified)
                    prev_bearish = df['open'].shift() > df['close'].shift()
                    curr_bullish = df['close'] > df['open']
                    curr_open_below_prev_close = df['open'] < df['close'].shift()
                    curr_close_above_prev_open = df['close'] > df['open'].shift()
                    
                    bullish_engulfing = prev_bearish & curr_bullish & curr_open_below_prev_close & curr_close_above_prev_open
                    timeframe_patterns['cdl_engulfing_bullish'] = zeros.copy()
                    timeframe_patterns['cdl_engulfing_bullish'][bullish_engulfing] = 100
                    
                    # Các mẫu hình khác sẽ không được triển khai nếu không có TA-Lib
                    # vì độ phức tạp cao, chỉ trả về series với giá trị 0
                    for pattern in ['cdl_hammer', 'cdl_morning_star', 'cdl_piercing', 
                                   'cdl_shooting_star', 'cdl_evening_star', 'cdl_hanging_man',
                                   'cdl_dark_cloud_cover', 'cdl_three_inside']:
                        timeframe_patterns[pattern] = zeros.copy()
                
                # Format pattern information for most recent candle
                latest_patterns = {}
                for pattern_name, pattern_values in timeframe_patterns.items():
                    # Get the most recent non-zero pattern value
                    recent_values = pattern_values[-5:]  # Last 5 values
                    recent_nonzero = [v for v in recent_values if v != 0]
                    
                    if recent_nonzero:
                        strength = recent_nonzero[-1]  # Most recent non-zero value
                        bullish = strength > 0
                        latest_patterns[pattern_name] = {
                            'detected': True,
                            'bullish': bullish,
                            'strength': abs(strength),
                            'period_detected': len(pattern_values) - list(pattern_values)[::-1].index(strength) - 1
                        }
                
                patterns[timeframe] = latest_patterns
                
        return patterns

    def extract_last_candle_info(self, df: pd.DataFrame) -> dict:
        """Extract information about the last candle."""
        if df is None or df.empty:
            return {}
            
        try:
            last_row = df.iloc[-1]
            previous_row = df.iloc[-2] if len(df) > 1 else None
            
            info = {
                'date': last_row.name.strftime('%Y-%m-%d') if hasattr(last_row.name, 'strftime') else str(last_row.name),
                'open': last_row.get('open'),
                'high': last_row.get('high'),
                'low': last_row.get('low'),
                'close': last_row.get('close'),
                'volume': last_row.get('volume')
            }
            
            # Calculate percentage change from previous close
            if previous_row is not None and 'close' in previous_row and previous_row['close'] > 0:
                info['change_percent'] = ((last_row['close'] - previous_row['close']) / previous_row['close']) * 100
            
            # Add technical indicators if available
            indicator_keys = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
            for key in indicator_keys:
                info[key] = last_row.get(key)
                
            return info
            
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất thông tin nến cuối cùng: {str(e)}")
            return {} 