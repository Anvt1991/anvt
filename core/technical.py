#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Technical analyzer for stock data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timezone
from core.data.data_validator import DataValidator

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """
    Technical analyzer for stock data
    """
    
    def __init__(self):
        """Initialize the TechnicalAnalyzer"""
        logger.info("TechnicalAnalyzer initialized")
    
    @staticmethod
    def _validate_input(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Validate and standardize input DataFrame: ensure OHLCV columns, DatetimeIndex, timezone, remove duplicate columns.
        """
        if df is None or df.empty:
            logger.warning(f"[TechnicalAnalyzer] Dữ liệu đầu vào rỗng{f' cho {symbol}' if symbol else ''}.")
            # Trả về DataFrame rỗng với cấu trúc chuẩn
            empty_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
            empty_df.index.name = 'date'
            return empty_df
            
        # Tạo bản sao để tránh thay đổi dữ liệu gốc
        df = df.copy()
        
        # Chuẩn hóa tên cột về chữ thường
        df.columns = [str(col).lower() for col in df.columns]
        
        # Kiểm tra và thêm các cột thiếu
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"[TechnicalAnalyzer] Thiếu cột {col} trong dữ liệu đầu vào{f' cho {symbol}' if symbol else ''}. Thêm cột này với giá trị 0.")
                df[col] = 0.0
        
        # Chuyển đổi cột sang kiểu số
        for col in df.columns:
            if col in required_cols:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"[TechnicalAnalyzer] Không thể chuyển cột {col} thành số: {e}")
                    df[col] = 0.0
        
        # Xử lý giá trị NaN
        if df[required_cols].isna().any().any():
            nan_cols = df[required_cols].columns[df[required_cols].isna().any()].tolist()
            logger.warning(f"[TechnicalAnalyzer] Dữ liệu có giá trị NaN trong cột: {nan_cols}. Áp dụng ffill và bfill.")
            df[required_cols] = df[required_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove duplicate columns
        if len(df.columns) != len(set(df.columns)):
            duplicates = df.columns[df.columns.duplicated()].tolist()
            logger.warning(f"[TechnicalAnalyzer] Duplicate columns detected: {duplicates}. Keeping first occurrence.")
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Ensure DatetimeIndex and timezone
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.set_index('date')
                except Exception as e:
                    logger.warning(f"[TechnicalAnalyzer] Lỗi khi chuyển cột 'date' thành index: {e}")
            else:
                logger.warning(f"[TechnicalAnalyzer] Không có index kiểu DatetimeIndex và không có cột 'date'.")
                # Tạo index ngày mới
                df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
                
        # Đảm bảo index không có giá trị NaT
        if isinstance(df.index, pd.DatetimeIndex) and df.index.isna().any():
            logger.warning(f"[TechnicalAnalyzer] Index có giá trị NaT. Tạo lại index.")
            valid_indices = ~df.index.isna()
            if valid_indices.any():
                df = df.loc[valid_indices]
            else:
                df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='D')
                
        # Set timezone
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            try:
                df.index = df.index.tz_localize('Asia/Ho_Chi_Minh', ambiguous='NaT', nonexistent='shift_forward')
            except Exception as e:
                logger.warning(f"[TechnicalAnalyzer] Lỗi khi set timezone: {e}")
                try:
                    # Fallback: try a different approach
                    df.index = pd.DatetimeIndex([idx.replace(tzinfo=timezone('Asia/Ho_Chi_Minh')) 
                                              if not idx.tzinfo else idx 
                                              for idx in df.index])
                except Exception as e2:
                    logger.warning(f"[TechnicalAnalyzer] Fallback timezone cũng lỗi: {e2}")
        
        return df

    def analyze(self, df: pd.DataFrame, period: str = 'short', symbol: str = None) -> Dict[str, Any]:
        """
        Analyze stock data using technical indicators
        Returns: Dictionary of technical analysis results (chuẩn hóa)
        """
        # CHUẨN HÓA DỮ LIỆU ĐẦU VÀO
        df = DataValidator.normalize_dataframe(df)
        df = DataValidator.validate_schema(df)
        # Validate input
        df = self._validate_input(df, symbol)
        if df is None or df.empty:
            fallback_support = [{"level": None, "basis": "Fallback: thiếu dữ liệu"}]
            fallback_resistance = [{"level": None, "basis": "Fallback: thiếu dữ liệu"}]
            return {
                "error": "No data available for analysis",
                "support_resistance": {"support": fallback_support, "resistance": fallback_resistance},
                "indicators": {},
                "signals": {},
                "trend_analysis": {},
                "bollinger_bands": {},
            }
        try:
            # Set period parameters
            if period == 'short':
                rsi_period = 14
                macd_fast = 12
                macd_slow = 26
                macd_signal = 9
                bb_period = 20
            elif period == 'medium':
                rsi_period = 21
                macd_fast = 19
                macd_slow = 39
                macd_signal = 9
                bb_period = 30
            else:  # long
                rsi_period = 30
                macd_fast = 24
                macd_slow = 52
                macd_signal = 18
                bb_period = 50
                
            # Kết quả mặc định - tránh lỗi nếu tính toán fails
            result = {
                "indicators": {
                    "rsi": None,
                    "ma_20": None,
                    "ma_50": None, 
                    "ma_200": None,
                    "macd": {
                        "macd_value": None,
                        "signal_line": None,
                        "histogram": None,
                        "interpretation": "Không đủ dữ liệu"
                    },
                },
                "signals": {
                    "rsi_signal": "Neutral",
                    "macd_signal": "Neutral",
                    "bb_signal": "Neutral",
                },
                "support_resistance": {
                    "support": [{"level": None, "basis": "Không đủ dữ liệu"}],
                    "resistance": [{"level": None, "basis": "Không đủ dữ liệu"}],
                },
                "trend_analysis": {
                    "trend": "Không xác định",
                    "ma_interpretation": "Không đủ dữ liệu",
                    "rsi_interpretation": "Không đủ dữ liệu",
                    "macd_interpretation": "Không đủ dữ liệu",
                },
                "bollinger_bands": {
                    "upper": None,
                    "middle": None,
                    "lower": None,
                },
            }
            
            # RSI
            current_rsi = None
            try:
                df['rsi'] = self._calculate_rsi(df, period=rsi_period)
                current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns and len(df) > 0 else None
            except Exception as e:
                logger.warning(f"[TechnicalAnalyzer] Không thể tính RSI: {e}{f' cho {symbol}' if symbol else ''}. Sử dụng giá trị None.")
                
            # MACD
            macd_val = signal_val = hist_val = None
            try:
                macd_result = self._calculate_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
                if macd_result is not None:
                    df['macd'] = macd_result[0]
                    df['macd_signal_line'] = macd_result[1]
                    df['macd_histogram'] = macd_result[2]
                    if len(df) > 0:
                        macd_val = float(df['macd'].iloc[-1]) if 'macd' in df.columns else None
                        signal_val = float(df['macd_signal_line'].iloc[-1]) if 'macd_signal_line' in df.columns else None
                        hist_val = float(df['macd_histogram'].iloc[-1]) if 'macd_histogram' in df.columns else None
            except Exception as e:
                logger.warning(f"[TechnicalAnalyzer] Không thể tính MACD: {e}{f' cho {symbol}' if symbol else ''}. Sử dụng giá trị None.")
                
            # Bollinger Bands
            upper = middle = lower = None
            try:
                upper, middle, lower = self._calculate_bollinger_bands(df, window=bb_period)
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
            except Exception as e:
                logger.warning(f"[TechnicalAnalyzer] Không thể tính Bollinger Bands: {e}{f' cho {symbol}' if symbol else ''}. Sử dụng giá trị None.")
                
            # MA values
            ma20 = ma50 = ma200 = None
            try:
                if len(df) >= 20:
                    ma20 = df['close'].rolling(window=20).mean().iloc[-1]
                if len(df) >= 50:
                    ma50 = df['close'].rolling(window=50).mean().iloc[-1]
                if len(df) >= 200:
                    ma200 = df['close'].rolling(window=200).mean().iloc[-1]
            except Exception as e:
                logger.warning(f"[TechnicalAnalyzer] Không thể tính MA: {e}{f' cho {symbol}' if symbol else ''}. Sử dụng giá trị None.")
                
            # Trend analysis
            try:
                trend = self.analyze_trend(df)
            except Exception as e:
                logger.warning(f"[TechnicalAnalyzer] Không thể phân tích trend: {e}{f' cho {symbol}' if symbol else ''}. Sử dụng giá trị mặc định.")
                trend = "Không xác định"
                
            # Support and Resistance
            try:
                support, resistance, support_basis, resistance_basis = self._find_support_resistance(df, return_basis=True)
                support_levels = [{"level": float(l) if l is not None else None, "basis": b} for l, b in zip(support, support_basis)]
                resistance_levels = [{"level": float(l) if l is not None else None, "basis": b} for l, b in zip(resistance, resistance_basis)]
            except Exception as e:
                logger.warning(f"[TechnicalAnalyzer] Không thể tìm hỗ trợ/kháng cự: {e}{f' cho {symbol}' if symbol else ''}. Sử dụng giá trị mặc định.")
                support_levels = [{"level": None, "basis": "Lỗi khi tính toán"}]
                resistance_levels = [{"level": None, "basis": "Lỗi khi tính toán"}]
                
            # RSI interpretation
            if current_rsi is not None:
                if current_rsi > 70:
                    rsi_interpretation = "Quá mua (Overbought)"
                elif current_rsi < 30:
                    rsi_interpretation = "Quá bán (Oversold)"
                else:
                    rsi_interpretation = "Trung tính (Neutral)"
            else:
                rsi_interpretation = "Không đủ dữ liệu"
                
            # MACD interpretation
            if macd_val is not None and signal_val is not None and hist_val is not None:
                if macd_val > signal_val and hist_val > 0:
                    macd_interpretation = "MACD cắt lên signal line, histogram dương → Tín hiệu mua mạnh"
                elif macd_val < signal_val and hist_val < 0:
                    macd_interpretation = "MACD cắt xuống signal line, histogram âm → Tín hiệu bán mạnh"
                elif macd_val > signal_val:
                    macd_interpretation = "MACD trên signal line → Xu hướng tăng"
                elif macd_val < signal_val:
                    macd_interpretation = "MACD dưới signal line → Xu hướng giảm"
                else:
                    macd_interpretation = "MACD trung tính"
            else:
                macd_interpretation = "MACD không đầy đủ dữ liệu"
                
            # MA interpretation
            if ma20 and ma50 and ma200:
                if ma20 > ma50 > ma200:
                    ma_interpretation = "MA20 > MA50 > MA200: Xu hướng tăng mạnh"
                elif ma20 < ma50 < ma200:
                    ma_interpretation = "MA20 < MA50 < MA200: Xu hướng giảm mạnh"
                elif ma20 > ma50 and ma50 < ma200:
                    ma_interpretation = "MA20 > MA50 < MA200: Có thể phục hồi"
                elif ma20 < ma50 and ma50 > ma200:
                    ma_interpretation = "MA20 < MA50 > MA200: Đang điều chỉnh"
                else:
                    ma_interpretation = "Các đường MA đang hội tụ hoặc đi ngang"
            else:
                ma_interpretation = "Không đủ dữ liệu MA"
                
            # BB Signal
            last_close = df['close'].iloc[-1] if len(df) > 0 else None
            if last_close is not None and upper is not None and lower is not None:
                if last_close > upper.iloc[-1]:
                    bb_signal = 'Overbought'
                elif last_close < lower.iloc[-1]:
                    bb_signal = 'Oversold'
                else:
                    bb_signal = 'Within Bands'
            else:
                bb_signal = 'Neutral'
                
            # MACD Signal
            macd_signal = 'Neutral'
            if df['macd'].iloc[-1] > df['macd_signal_line'].iloc[-1] and df['macd'].iloc[-2] <= df['macd_signal_line'].iloc[-2] if 'macd' in df.columns and 'macd_signal_line' in df.columns and len(df) > 1 else False:
                macd_signal = 'Buy (Bullish Crossover)'
            elif df['macd'].iloc[-1] < df['macd_signal_line'].iloc[-1] and df['macd'].iloc[-2] >= df['macd_signal_line'].iloc[-2] if 'macd' in df.columns and 'macd_signal_line' in df.columns and len(df) > 1 else False:
                macd_signal = 'Sell (Bearish Crossover)'
            elif df['macd'].iloc[-1] > df['macd_signal_line'].iloc[-1] if 'macd' in df.columns and 'macd_signal_line' in df.columns and len(df) > 0 else False:
                macd_signal = 'Bullish'
            elif df['macd'].iloc[-1] < df['macd_signal_line'].iloc[-1] if 'macd' in df.columns and 'macd_signal_line' in df.columns and len(df) > 0 else False:
                macd_signal = 'Bearish'
                
            # Chuẩn hóa trả về
            def safe_float(val):
                try:
                    if pd.isna(val) or val is None:
                        return None
                    return float(val)
                except Exception:
                    return None
                    
            # Cập nhật kết quả với các giá trị đã tính
            result["indicators"]["rsi"] = safe_float(current_rsi)
            result["indicators"]["ma_20"] = safe_float(ma20)
            result["indicators"]["ma_50"] = safe_float(ma50)
            result["indicators"]["ma_200"] = safe_float(ma200)
            result["indicators"]["macd"]["macd_value"] = safe_float(macd_val)
            result["indicators"]["macd"]["signal_line"] = safe_float(signal_val)
            result["indicators"]["macd"]["histogram"] = safe_float(hist_val)
            result["indicators"]["macd"]["interpretation"] = macd_interpretation
            
            result["signals"]["rsi_signal"] = 'Oversold' if current_rsi is not None and current_rsi < 30 else ('Overbought' if current_rsi is not None and current_rsi > 70 else 'Neutral')
            result["signals"]["macd_signal"] = macd_signal
            result["signals"]["bb_signal"] = bb_signal
            
            result["support_resistance"]["support"] = support_levels
            result["support_resistance"]["resistance"] = resistance_levels
            
            result["trend_analysis"]["trend"] = trend
            result["trend_analysis"]["ma_interpretation"] = ma_interpretation
            result["trend_analysis"]["rsi_interpretation"] = rsi_interpretation
            result["trend_analysis"]["macd_interpretation"] = macd_interpretation
            
            result["bollinger_bands"]["upper"] = safe_float(upper.iloc[-1] if upper is not None and len(upper) > 0 else None)
            result["bollinger_bands"]["middle"] = safe_float(middle.iloc[-1] if middle is not None and len(middle) > 0 else None)
            result["bollinger_bands"]["lower"] = safe_float(lower.iloc[-1] if lower is not None and len(lower) > 0 else None)
            
            return result
            
        except Exception as e:
            logger.error(f"[TechnicalAnalyzer] Error in technical analysis: {str(e)}{f' cho {symbol}' if symbol else ''}")
            fallback_support = [{"level": None, "basis": "Fallback: lỗi phân tích"}]
            fallback_resistance = [{"level": None, "basis": "Fallback: lỗi phân tích"}]
            return {
                "error": f"Technical analysis failed: {str(e)}",
                "support_resistance": {"support": fallback_support, "resistance": fallback_resistance},
                "indicators": {},
                "signals": {},
                "trend_analysis": {},
                "bollinger_bands": {},
            }
    
    @staticmethod
    def _calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        
        Args:
            df: DataFrame with price data
            period: RSI period
            
        Returns:
            Series with RSI values
        """
        # Thử sử dụng ta-lib nếu có thể
        try:
            import talib
            return talib.RSI(df['close'].values, timeperiod=period)
        except ImportError:
            # Nếu không có ta-lib, tính toán thủ công với cải tiến
            delta = df['close'].diff()
            
            # Tách gain và loss
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # Tính toán avg_gain và avg_loss sử dụng EMA thay vì rolling mean
            # Tính giá trị SMA ban đầu
            avg_gain = gain.rolling(window=period).mean().iloc[period]
            avg_loss = loss.rolling(window=period).mean().iloc[period]
            
            # Tính RSI sử dụng phương pháp Wilder
            avg_gains = np.zeros_like(delta)
            avg_losses = np.zeros_like(delta)
            
            for i in range(period, len(df)):
                if i == period:
                    avg_gains[i] = avg_gain
                    avg_losses[i] = avg_loss
                else:
                    avg_gains[i] = (avg_gains[i-1] * (period - 1) + gain.iloc[i]) / period
                    avg_losses[i] = (avg_losses[i-1] * (period - 1) + loss.iloc[i]) / period
            
            avg_gains = pd.Series(avg_gains, index=delta.index)
            avg_losses = pd.Series(avg_losses, index=delta.index)
            
            # Tránh chia cho 0
            avg_losses = avg_losses.replace(0, np.finfo(float).eps)
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    @staticmethod
    def _calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (MACD Line, Signal Line, Histogram)
        """
        # Thử sử dụng ta-lib nếu có thể
        try:
            import talib
            macd, signal_line, histogram = talib.MACD(
                df['close'].values,
                fastperiod=fast,
                slowperiod=slow,
                signalperiod=signal
            )
            return pd.Series(macd, index=df.index), pd.Series(signal_line, index=df.index), pd.Series(histogram, index=df.index)
        except ImportError:
            # Nếu không có ta-lib, tính toán thủ công với cải tiến
            # Tính EMA đúng chuẩn
            def calculate_ema(prices, period):
                k = 2 / (period + 1)
                ema = prices.copy()
                # Dùng SMA cho giá trị đầu tiên
                ema[:period] = prices[:period].mean()
                # Tính EMA theo công thức
                for i in range(period, len(prices)):
                    ema[i] = prices[i] * k + ema[i-1] * (1-k)
                return ema
            
            close_prices = df['close'].values
            
            # Tính các EMA
            ema_fast = pd.Series(calculate_ema(close_prices, fast), index=df.index)
            ema_slow = pd.Series(calculate_ema(close_prices, slow), index=df.index)
            
            # Tính MACD
            macd_line = ema_fast - ema_slow
            
            # Tính Signal line (EMA của MACD)
            signal_line = pd.Series(
                calculate_ema(macd_line.values, signal),
                index=df.index
            )
            
            # Tính Histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
    
    @staticmethod
    def _calculate_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with price data
            window: Window for moving average
            num_std: Number of standard deviations
            
        Returns:
            Tuple of (Upper Band, Middle Band, Lower Band)
        """
        # Manual calculation
        middle = df['close'].rolling(window=window).mean()
        std_dev = df['close'].rolling(window=window).std()
        
        upper = middle + (std_dev * num_std)
        lower = middle - (std_dev * num_std)
        
        return upper, middle, lower
    
    def _find_support_resistance(self, df: pd.DataFrame, return_basis: bool = False) -> Tuple[List[float], List[float], List[str], List[str]]:
        """
        Find support and resistance levels using improved price action analysis
        
        Args:
            df: DataFrame with price data
            return_basis: Whether to return basis for each level
            
        Returns:
            Tuple of (Support Levels, Resistance Levels, Support Basis, Resistance Basis)
        """
        try:
            # Đảm bảo có đủ dữ liệu
            if len(df) < 30:
                logger.warning(f"Không đủ dữ liệu để xác định hỗ trợ/kháng cự chính xác. Có {len(df)} dòng.")
                fallback_support = [df['close'].iloc[-1] * 0.95]
                fallback_resistance = [df['close'].iloc[-1] * 1.05]
                if return_basis:
                    return fallback_support, fallback_resistance, ["Fallback: thiếu dữ liệu"], ["Fallback: thiếu dữ liệu"]
                return fallback_support, fallback_resistance
                
            # Sử dụng nhiều phiên hơn cho kết quả tin cậy
            recent_df = df.iloc[-120:] if len(df) >= 120 else df
            
            # Tìm các đỉnh và đáy tiềm năng
            window = min(10, len(recent_df) // 10)  # Điều chỉnh window size theo kích thước dữ liệu
            
            # Phương pháp nâng cao: Tính toán các mức hỗ trợ/kháng cự bằng cả hai phương pháp
            # 1. Fractal method
            highs = recent_df['high'].values
            lows = recent_df['low'].values
            
            # Fractals - xác định các điểm swing high/low
            swing_highs_idx = []
            swing_lows_idx = []
            
            # Tìm swing highs (điểm có high cao hơn w điểm xung quanh)
            for i in range(window, len(recent_df) - window):
                if all(highs[i] > highs[i-j] for j in range(1, window+1)) and \
                   all(highs[i] > highs[i+j] for j in range(1, window+1)):
                    swing_highs_idx.append(i)
                    
            # Tìm swing lows (điểm có low thấp hơn w điểm xung quanh)
            for i in range(window, len(recent_df) - window):
                if all(lows[i] < lows[i-j] for j in range(1, window+1)) and \
                   all(lows[i] < lows[i+j] for j in range(1, window+1)):
                    swing_lows_idx.append(i)
                    
            # 2. Phương pháp 2: Volume Profile - tìm các vùng giao dịch tập trung
            # Chia giá thành nhiều vùng
            price_min = recent_df['low'].min()
            price_max = recent_df['high'].max()
            price_range = price_max - price_min
            num_zones = 100
            zone_height = price_range / num_zones
            
            # Tính tổng khối lượng cho mỗi vùng giá
            volume_profile = np.zeros(num_zones)
            for i in range(len(recent_df)):
                # Xác định vùng giá của nến
                candle_min_zone = max(0, min(num_zones-1, int((recent_df['low'].iloc[i] - price_min) / zone_height)))
                candle_max_zone = max(0, min(num_zones-1, int((recent_df['high'].iloc[i] - price_min) / zone_height)))
                
                # Phân bổ khối lượng vào các vùng giá
                for zone in range(candle_min_zone, candle_max_zone + 1):
                    volume_profile[zone] += recent_df['volume'].iloc[i] / (candle_max_zone - candle_min_zone + 1)
            
            # Tìm các đỉnh cục bộ trong volume profile
            volume_peaks = []
            for i in range(5, num_zones - 5):
                if all(volume_profile[i] >= volume_profile[i-j] for j in range(1, 5)) and \
                   all(volume_profile[i] >= volume_profile[i+j] for j in range(1, 5)):
                    volume_peaks.append(i)
            
            # Chuyển đổi chỉ số vùng thành giá
            volume_peak_prices = [price_min + (i + 0.5) * zone_height for i in volume_peaks]
            
            # 3. Kết hợp kết quả từ hai phương pháp
            # Lấy giá từ các swing points
            swing_high_prices = [highs[i] for i in swing_highs_idx]
            swing_low_prices = [lows[i] for i in swing_lows_idx]
            
            # Kết hợp với volume peak prices và phân loại thành hỗ trợ/kháng cự
            current_price = df['close'].iloc[-1]
            
            resistance_raw = []
            resistance_basis = []
            for p in swing_high_prices:
                if p > current_price:
                    resistance_raw.append(p)
                    resistance_basis.append("Swing high")
            for p in volume_peak_prices:
                if p > current_price:
                    resistance_raw.append(p)
                    resistance_basis.append("Volume profile peak")
            support_raw = []
            support_basis = []
            for p in swing_low_prices:
                if p < current_price:
                    support_raw.append(p)
                    support_basis.append("Swing low")
            for p in volume_peak_prices:
                if p < current_price:
                    support_raw.append(p)
                    support_basis.append("Volume profile peak")
            
            # Loại bỏ các mức quá gần nhau (clustering)
            def cluster_levels(levels, basis, threshold_pct=0.01, reverse=False):
                if not levels:
                    return [], []
                if reverse:
                    levels = list(reversed(levels))
                    basis = list(reversed(basis))
                threshold = current_price * threshold_pct
                clustered = [levels[0]]
                clustered_basis = [basis[0]]
                for idx, level in enumerate(levels[1:], 1):
                    if abs(level - clustered[-1]) > threshold:
                        clustered.append(level)
                        clustered_basis.append(basis[idx])
                if reverse:
                    clustered = list(reversed(clustered))
                    clustered_basis = list(reversed(clustered_basis))
                return clustered, clustered_basis
                
            resistance_levels, resistance_basis = cluster_levels(resistance_raw, resistance_basis)
            support_levels, support_basis = cluster_levels(support_raw, support_basis, reverse=True)
            
            # Giới hạn số lượng mức
            resistance_levels = resistance_levels[:5]  # Lấy tối đa 5 mức kháng cự
            support_levels = support_levels[:5]  # Lấy tối đa 5 mức hỗ trợ
            
            if return_basis:
                return support_levels, resistance_levels, support_basis, resistance_basis
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Lỗi khi tìm mức hỗ trợ/kháng cự: {str(e)}")
            # Fallback nếu có lỗi
            fallback_support = [df['close'].iloc[-1] * 0.95]
            fallback_resistance = [df['close'].iloc[-1] * 1.05]
            return fallback_support, fallback_resistance, ["Fallback: lỗi phân tích"], ["Fallback: lỗi phân tích"]
    
    def analyze_trend(self, df: pd.DataFrame) -> str:
        """
        Analyze the trend of a stock or index
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Trend description (string)
        """
        try:
            # Check if we have enough data
            if len(df) < 50:
                return "Không đủ dữ liệu"
                
            # Calculate moving averages if not already present
            if 'ma50' not in df.columns:
                df['ma50'] = df['close'].rolling(window=50).mean()
            if 'ma200' not in df.columns:
                df['ma200'] = df['close'].rolling(window=200).mean()
            
            # Get last values
            last_close = df['close'].iloc[-1]
            last_ma50 = df['ma50'].iloc[-1]
            last_ma200 = df['ma200'].iloc[-1]
            
            # Check price movement over different periods
            change_1w = (last_close / df['close'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
            change_1m = (last_close / df['close'].iloc[-22] - 1) * 100 if len(df) >= 22 else 0
            change_3m = (last_close / df['close'].iloc[-66] - 1) * 100 if len(df) >= 66 else 0
            
            # Determine trend
            if last_close > last_ma50 and last_ma50 > last_ma200:
                if change_1w > 3 or change_1m > 10:
                    return "Tăng mạnh (Uptrend)"
                return "Tăng (Uptrend)"
            elif last_close < last_ma50 and last_ma50 < last_ma200:
                if change_1w < -3 or change_1m < -10:
                    return "Giảm mạnh (Downtrend)"
                return "Giảm (Downtrend)"
            elif last_close > last_ma50 and last_ma50 < last_ma200:
                return "Phục hồi (Recovery)"
            elif last_close < last_ma50 and last_ma50 > last_ma200:
                return "Điều chỉnh (Correction)"
            else:
                return "Đi ngang (Sideways)"
                
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return "Không xác định"
    
    def generate_chart(self, df: pd.DataFrame, symbol: str) -> Figure:
        """
        Generate a technical analysis chart
        
        Args:
            df: DataFrame with price data
            symbol: Stock symbol
            
        Returns:
            Matplotlib Figure object
        """
        try:
            # Create figure
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Calculate indicators if not already in the dataframe
            if 'ma50' not in df.columns:
                df['ma50'] = df['close'].rolling(window=50).mean()
            if 'ma200' not in df.columns:
                df['ma200'] = df['close'].rolling(window=200).mean()
            if 'rsi' not in df.columns:
                df['rsi'] = self._calculate_rsi(df)
            if 'macd' not in df.columns:
                macd_result = self._calculate_macd(df)
                df['macd'] = macd_result[0]
                df['macd_signal_line'] = macd_result[1]
                df['macd_histogram'] = macd_result[2]
            if 'bb_upper' not in df.columns:
                upper, middle, lower = self._calculate_bollinger_bands(df)
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
            
            # Plot price and moving averages
            ax1.plot(df.index, df['close'], label='Price', color='black')
            ax1.plot(df.index, df['ma50'], label='MA50', color='blue', alpha=0.7)
            ax1.plot(df.index, df['ma200'], label='MA200', color='red', alpha=0.7)
            ax1.plot(df.index, df['bb_upper'], label='BB Upper', color='grey', linestyle='--', alpha=0.5)
            ax1.plot(df.index, df['bb_middle'], label='BB Middle', color='grey', linestyle='--', alpha=0.5)
            ax1.plot(df.index, df['bb_lower'], label='BB Lower', color='grey', linestyle='--', alpha=0.5)
            
            # Format first subplot
            ax1.set_title(f'{symbol} Technical Analysis')
            ax1.set_ylabel('Price')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            
            # Plot RSI
            ax2.plot(df.index, df['rsi'], color='purple')
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.set_ylabel('RSI')
            ax2.grid(True, alpha=0.3)
            
            # Plot MACD
            ax3.plot(df.index, df['macd'], label='MACD', color='blue')
            ax3.plot(df.index, df['macd_signal_line'], label='Signal', color='red')
            ax3.bar(df.index, df['macd_histogram'], label='Histogram', color='grey', alpha=0.5)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax3.set_ylabel('MACD')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left')
            
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            # Return a simple error figure
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Error generating chart: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center')
            ax.set_axis_off()
            return fig 

    @staticmethod
    def get_technical_indicators(df: pd.DataFrame) -> dict:
        """
        Chuẩn hóa tính toán chỉ số kỹ thuật, trả về dict scalar, đồng nhất key, kiểm tra NaN.
        """
        result = {}
        # MA
        for period in [20, 50, 200]:
            ma_col = f"ma_{period}"
            if len(df) >= period:
                ma_val = df['close'].rolling(window=period).mean().iloc[-1]
                result[ma_col] = float(ma_val) if not pd.isna(ma_val) and ma_val is not None else None
            else:
                result[ma_col] = None
        # RSI
        if len(df) >= 14:
            try:
                import talib
                rsi = talib.RSI(df['close'].values, timeperiod=14)
                rsi_val = rsi[-1]
            except ImportError:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = rsi.iloc[-1]
            result['rsi_14'] = float(rsi_val) if not pd.isna(rsi_val) else None
        else:
            result['rsi_14'] = None
        # MACD
        if len(df) >= 26:
            try:
                import talib
                macd, macd_signal, macd_hist = talib.MACD(df['close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
                macd_val = macd[-1]
                macd_signal_val = macd_signal[-1]
                macd_hist_val = macd_hist[-1]
            except ImportError:
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                macd_val = ema12.iloc[-1] - ema26.iloc[-1]
                macd_signal_val = df['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                macd_hist_val = macd_val - macd_signal_val
            result['macd'] = float(macd_val) if not pd.isna(macd_val) else None
            result['macd_signal'] = float(macd_signal_val) if not pd.isna(macd_signal_val) else None
            result['macd_hist'] = float(macd_hist_val) if not pd.isna(macd_hist_val) else None
        else:
            result['macd'] = result['macd_signal'] = result['macd_hist'] = None
        return result 