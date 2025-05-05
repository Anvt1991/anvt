#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module kiểm tra và xác thực dữ liệu chứng khoán
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional
from datetime import datetime, timedelta
import pytz
import re
import logging
import holidays
# Thêm thư viện cho outlier detection và schema validation
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from pipeline.utils import is_trading_day

# Thiết lập logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Timezone cho Việt Nam
TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# Định nghĩa schema cho dữ liệu chứng khoán
stock_schema = DataFrameSchema({
    "open": Column(float, Check.gt(0)),
    "high": Column(float, Check.gt(0)),
    "low": Column(float, Check.gt(0)),
    "close": Column(float, Check.gt(0)),
    "volume": Column(int, Check.ge(0)),
}, strict=False)  # strict=False để tránh lỗi với cột không xác định

class DataValidator:
    """
    Lớp xác thực và chuẩn hóa dữ liệu toàn diện.
    Cung cấp các phương thức để xác thực, chuẩn hóa và căn chỉnh dữ liệu chứng khoán.
    """
    # Danh sách khung thời gian hợp lệ
    VALID_TIMEFRAMES = [
        '5m', '15m', '30m', '1h', '4h',  # Khung thời gian mới
        '1D', '1W', '1M'                 # Khung thời gian legacy
    ]
    
    # Danh sách các mã chỉ số
    INDICES = ['VNINDEX', 'VN30', 'HNX30', 'HNXINDEX', 'UPCOM', 'HNX', 'VN100']
    
    # Định dạng mã hợp lệ
    TICKER_PATTERN = r'^[A-Z0-9]{3,6}$'
    
    @staticmethod
    def is_valid_timeframe(timeframe: str) -> bool:
        """Kiểm tra tính hợp lệ của khung thời gian"""
        return timeframe.upper() in [tf.upper() for tf in DataValidator.VALID_TIMEFRAMES]
    
    @staticmethod
    def normalize_timeframe(timeframe: str) -> str:
        """
        Chuẩn hóa khung thời gian
        
        Args:
            timeframe: Khung thời gian cần chuẩn hóa
            
        Returns:
            Khung thời gian đã được chuẩn hóa
        """
        # Chuyển sang chữ thường và loại bỏ khoảng trắng
        tf = timeframe.strip().lower()
        
        # Ánh xạ các biến thể phổ biến
        tf_map = {
            # Khung thời gian ngày
            '1d': '1D', 'd': '1D', 'day': '1D', 'daily': '1D', '1day': '1D',
            # Khung thời gian tuần
            '1w': '1W', 'w': '1W', 'week': '1W', 'weekly': '1W', '1week': '1W',
            # Khung thời gian tháng
            '1m': '1M', 'm': '1M', 'month': '1M', 'monthly': '1M', '1month': '1M',
            # Khung thời gian phút
            '5m': '5m', '5min': '5m', '5': '5m',
            '15m': '15m', '15min': '15m', '15': '15m',
            '30m': '30m', '30min': '30m', '30': '30m',
            # Khung thời gian giờ
            '60m': '1h', '1h': '1h', '1hour': '1h', 'h': '1h', 'hour': '1h',
            '4h': '4h', '4hour': '4h', '240m': '4h'
        }
        
        # Trả về khung thời gian đã chuẩn hóa
        normalized = tf_map.get(tf, timeframe)
        if DataValidator.is_valid_timeframe(normalized):
            return normalized
        
        # Nếu không tìm thấy, trả về mặc định
        logger.warning(f"Khung thời gian không hợp lệ: {timeframe}, sử dụng 1D")
        return '1D'
    
    @staticmethod
    def validate_ticker(symbol: str) -> str:
        """
        Kiểm tra tính hợp lệ của mã chứng khoán
        
        Args:
            symbol: Mã chứng khoán cần kiểm tra
            
        Returns:
            Mã chứng khoán đã được chuẩn hóa
        """
        # Loại bỏ khoảng trắng và chuyển sang chữ hoa
        symbol = symbol.strip().upper()
        
        # Kiểm tra nếu mã là chỉ số
        if DataValidator.is_index(symbol):
            return symbol
        
        # Kiểm tra định dạng mã
        if not re.match(DataValidator.TICKER_PATTERN, symbol):
            logger.warning(f"Mã chứng khoán không hợp lệ: {symbol}")
            raise ValueError(f"Mã chứng khoán không hợp lệ: {symbol}")
        
        return symbol
    
    @staticmethod
    def is_index(symbol: str) -> bool:
        """
        Kiểm tra xem mã có phải là chỉ số không
        
        Args:
            symbol: Mã cần kiểm tra
            
        Returns:
            True nếu là chỉ số, False nếu không phải
        """
        return symbol.upper() in DataValidator.INDICES
    
    @staticmethod
    def validate_candles(num_candles: int) -> int:
        """
        Xác thực số lượng nến
        
        Args:
            num_candles: Số lượng nến cần kiểm tra
            
        Returns:
            Số lượng nến hợp lệ
        """
        try:
            num = int(num_candles)
            if num <= 0:
                logger.warning(f"Số lượng nến không hợp lệ: {num}, sử dụng giá trị mặc định 100")
                return 100
            elif num > 10000:
                logger.warning(f"Số lượng nến quá lớn: {num}, giới hạn ở 10000")
                return 10000
            return num
        except (ValueError, TypeError):
            logger.warning(f"Số lượng nến không hợp lệ: {num_candles}, sử dụng giá trị mặc định 100")
            return 100
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Tuple[datetime, datetime]:
        """
        Xác thực khoảng thời gian
        
        Args:
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            
        Returns:
            Tuple (ngày bắt đầu, ngày kết thúc) đã được chuẩn hóa
        """
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            if start >= end:
                logger.warning("Ngày bắt đầu phải trước ngày kết thúc")
                end = start + timedelta(days=1)
            if end > datetime.now(TZ):
                end = datetime.now(TZ)
            return start, end
        except Exception as e:
            logger.error(f"Khoảng thời gian không hợp lệ: {str(e)}")
            # Trả về mặc định 30 ngày
            end = datetime.now(TZ)
            start = end - timedelta(days=30)
            return start, end
    
    @staticmethod
    def align_timestamps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Căn chỉnh dấu thời gian theo khung thời gian chính xác
        
        Args:
            df: DataFrame cần căn chỉnh
            timeframe: Khung thời gian
            
        Returns:
            DataFrame đã được căn chỉnh
        """
        if df.empty:
            return df
        
        df = df.copy()
        # Đảm bảo index là datetime và có múi giờ
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Nếu không có múi giờ, giả định là múi giờ Việt Nam
        if df.index.tz is None:
            df.index = df.index.tz_localize(TZ)
        
        # Căn chỉnh timestamp dựa trên khung thời gian
        if timeframe == '1D':
            # Đối với dữ liệu ngày, lấy giá mỗi ngày lúc EOD (15:00)
            df.index = df.index.normalize() + pd.Timedelta(hours=15)
        elif timeframe == '1W':
            # Đối với dữ liệu tuần, lấy giá vào cuối tuần (Thứ 6, 15:00)
            df.index = df.index.to_period('W').to_timestamp() + pd.Timedelta(days=4, hours=15)
        elif timeframe == '1M':
            # Đối với dữ liệu tháng, lấy giá vào cuối tháng
            df.index = df.index.to_period('M').to_timestamp() + pd.Timedelta(hours=15)
        elif timeframe == '5m':
            # Căn chỉnh theo mỗi 5 phút
            df.index = df.index.floor('5min')
        elif timeframe == '15m':
            # Căn chỉnh theo mỗi 15 phút
            df.index = df.index.floor('15min')
        elif timeframe == '30m':
            # Căn chỉnh theo mỗi 30 phút
            df.index = df.index.floor('30min')
        elif timeframe == '1h':
            # Căn chỉnh theo mỗi giờ
            df.index = df.index.floor('H')
        elif timeframe == '4h':
            # Căn chỉnh theo mỗi 4 giờ
            df.index = df.index.floor('4H')
        
        # Sắp xếp theo thời gian tăng dần
        df = df.sort_index()
        
        return df
    
    @staticmethod
    def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
        """
        Loại bỏ các dòng có volume = 0 hoặc giá = 0 ở đầu/đuôi chuỗi
        """
        if df.empty:
            return df
        # Loại bỏ các dòng volume = 0 hoặc giá = 0 ở đầu chuỗi
        while not df.empty and ((df.iloc[0][['open','high','low','close']].min() <= 0) or (df.iloc[0]['volume'] == 0)):
            df = df.iloc[1:]
        # Loại bỏ các dòng volume = 0 hoặc giá = 0 ở cuối chuỗi
        while not df.empty and ((df.iloc[-1][['open','high','low','close']].min() <= 0) or (df.iloc[-1]['volume'] == 0)):
            df = df.iloc[:-1]
        return df

    @staticmethod
    def remove_non_trading_days(df: pd.DataFrame) -> pd.DataFrame:
        """
        Loại bỏ các dòng có ngày là ngày nghỉ lễ hoặc cuối tuần
        """
        if df.empty:
            return df
        from pipeline.utils import is_trading_day
        # Đảm bảo index là DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
        # Lọc các ngày giao dịch
        mask = df.index.map(lambda d: is_trading_day(d.date()))
        return df[mask]

    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa DataFrame chứa dữ liệu chứng khoán
        
        Args:
            df: DataFrame cần chuẩn hóa
            
        Returns:
            DataFrame đã được chuẩn hóa
        """
        if df.empty:
            return df
        
        # Tạo bản sao để tránh thay đổi df gốc
        df_norm = df.copy()
        
        # Chuẩn hóa tên cột
        col_map = {
            'date': 'date', 'time': 'date', 'Date': 'date', 'Time': 'date', 'tradingdate': 'date', 'TradingDate': 'date',
            'open': 'open', 'Open': 'open', 'OPEN': 'open',
            'high': 'high', 'High': 'high', 'HIGH': 'high',
            'low': 'low', 'Low': 'low', 'LOW': 'low',
            'close': 'close', 'Close': 'close', 'CLOSE': 'close', 'pricebasic': 'close',
            'volume': 'volume', 'Volume': 'volume', 'VOLUME': 'volume',
            'adj close': 'adj_close', 'Adj Close': 'adj_close', 'ADJCLOSE': 'adj_close'
        }
        
        # Đổi tên các cột
        renamed_cols = {}
        for old_col in df_norm.columns:
            if old_col in col_map:
                renamed_cols[old_col] = col_map[old_col]
        
        if renamed_cols:
            df_norm = df_norm.rename(columns=renamed_cols)
        
        # Đảm bảo các cột OHLCV tồn tại
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df_norm.columns:
                if col == 'volume':
                    df_norm[col] = 0  # Thêm cột volume với giá trị 0
                elif col == 'close' and 'adj_close' in df_norm.columns:
                    df_norm[col] = df_norm['adj_close']  # Sử dụng adj_close nếu không có close
                elif col in ['open', 'high', 'low'] and 'close' in df_norm.columns:
                    df_norm[col] = df_norm['close']  # Sử dụng giá đóng cửa nếu không có
                else:
                    logger.warning(f"Không thể thêm cột {col} thiếu")
        
        # Đảm bảo chỉ số ngày có định dạng datetime
        if not isinstance(df_norm.index, pd.DatetimeIndex):
            # Nếu có cột date, sử dụng làm chỉ số
            if 'date' in df_norm.columns:
                df_norm['date'] = pd.to_datetime(df_norm['date'])
                df_norm = df_norm.set_index('date')
                
                # Kiểm tra nếu chỉ số đã có timezone
                if df_norm.index.tzinfo is None:
                    df_norm.index = df_norm.index.tz_localize(TZ)
        
        # Xử lý giá trị null trong chuỗi thời gian
        if any(df_norm.isnull().sum()):
            # Phương pháp điền: forward fill cho giá mở/đóng/cao/thấp
            df_norm[['open', 'high', 'low', 'close']] = df_norm[['open', 'high', 'low', 'close']].fillna(method='ffill')
            
            # Các giá trị vẫn còn thiếu (ở đầu chuỗi) sẽ được điền bằng backward fill
            df_norm[['open', 'high', 'low', 'close']] = df_norm[['open', 'high', 'low', 'close']].fillna(method='bfill')
            
            # Khối lượng thiếu được điền 0
            df_norm['volume'] = df_norm['volume'].fillna(0)
        
        # Kiểm tra tính nhất quán của dữ liệu
        # - Giá cao >= giá đóng, mở, thấp
        # - Giá thấp <= giá đóng, mở, cao
        df_norm['high'] = df_norm[['high', 'open', 'close']].max(axis=1)
        df_norm['low'] = df_norm[['low', 'open', 'close']].min(axis=1)
        
        # Chuyển đổi kiểu dữ liệu
        for col in required_cols:
            df_norm[col] = pd.to_numeric(df_norm[col], errors='coerce')
        
        # Loại bỏ các hàng có giá trị âm trong giá
        df_norm = df_norm[(df_norm['open'] > 0) & (df_norm['high'] > 0) & (df_norm['low'] > 0) & (df_norm['close'] > 0)]
        # Loại bỏ các dòng volume = 0 hoặc giá = 0 ở đầu/đuôi chuỗi
        df_norm = DataValidator.remove_invalid_rows(df_norm)
        # Loại bỏ các dòng có ngày là ngày nghỉ lễ/cuối tuần
        df_norm = DataValidator.remove_non_trading_days(df_norm)
        
        return df_norm
    
    @staticmethod
    def detect_and_handle_outliers(df: pd.DataFrame, method: str = 'ensemble', threshold: float = 3.0, ml_params: Dict[str, Any] = None) -> Tuple[pd.DataFrame, str]:
        """
        Phát hiện và cảnh báo các giá trị ngoại lệ (outliers) trong dữ liệu giá, nhưng không loại bỏ tự động.
        Chỉ loại bỏ các điểm dữ liệu chắc chắn là lỗi (giá/volume âm, giá = 0, volume = 0, ngày nghỉ) ở các bước normalize khác.
        """
        if df.empty:
            return df, "DataFrame rỗng"
        # Đảm bảo các cột cần thiết
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            message = f"Thiếu cột dữ liệu: {', '.join(missing_columns)}"
            logger.warning(message)
            return df, message
        # Thống kê trước khi xử lý
        total_rows = len(df)
        # Phát hiện outlier nhưng không loại bỏ
        outlier_report = "Không loại bỏ outlier tự động. Chỉ cảnh báo các điểm bất thường."
        # Có thể log các điểm nghi ngờ là outlier nếu muốn
        # (Có thể bổ sung log chi tiết ở đây nếu cần)
        return df, outlier_report
    
    @staticmethod
    def validate_fundamental_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xác thực dữ liệu cơ bản của cổ phiếu
        
        Args:
            data: Dictionary chứa dữ liệu cơ bản
            
        Returns:
            Dictionary đã được xác thực
        """
        if not data:
            return {"error": "Không có dữ liệu cơ bản"}
        
        valid_data = {}
        # Danh sách các chỉ số cơ bản và phạm vi hợp lệ
        key_ranges = {
            'EPS': (-1000, 100000),
            'P/E': (0, 1000),
            'P/B': (0, 100),
            'ROE': (-100, 100),
            'ROA': (-100, 100),
            'Dividend Yield': (0, 100),
            'Market Cap': (0, 1e12),
            'Revenue': (0, 1e12),
            'Profit': (-1e12, 1e12)
        }
        
        # Chuẩn hóa tên key
        key_mapping = {
            'earningpershare': 'EPS',
            'eps': 'EPS',
            'pricetoearning': 'P/E',
            'pe': 'P/E',
            'pricetobook': 'P/B',
            'pb': 'P/B',
            'returnonequity': 'ROE',
            'roe': 'ROE',
            'returnonasset': 'ROA',
            'roa': 'ROA',
            'dividendyield': 'Dividend Yield',
            'dividend': 'Dividend Yield',
            'marketcap': 'Market Cap',
            'market_cap': 'Market Cap',
            'revenue': 'Revenue',
            'profit': 'Profit'
        }
        
        # Chuẩn hóa và xác thực dữ liệu
        for key, value in data.items():
            normalized_key = key_mapping.get(key.lower(), key)
            
            # Nếu là key quan trọng, kiểm tra phạm vi
            if normalized_key in key_ranges:
                min_val, max_val = key_ranges[normalized_key]
                try:
                    # Chuyển đổi sang float
                    numeric_value = float(value)
                    if min_val <= numeric_value <= max_val:
                        valid_data[normalized_key] = numeric_value
                    else:
                        logger.warning(f"Dữ liệu '{normalized_key}' = {numeric_value} ngoài phạm vi hợp lệ ({min_val}, {max_val})")
                except (ValueError, TypeError):
                    logger.warning(f"Không thể chuyển đổi '{normalized_key}' = {value} sang số")
            else:
                # Các key khác giữ nguyên
                valid_data[normalized_key] = value
        
        return valid_data
    
    @staticmethod
    def calculate_trading_days(start_date: datetime, end_date: datetime, country: str = 'VN') -> pd.DatetimeIndex:
        """
        Tính toán các ngày giao dịch trong khoảng thời gian, loại bỏ ngày nghỉ và cuối tuần
        
        Args:
            start_date: Ngày bắt đầu
            end_date: Ngày kết thúc
            country: Mã quốc gia để xác định ngày nghỉ
            
        Returns:
            DatetimeIndex chứa các ngày giao dịch
        """
        if start_date >= end_date:
            raise ValueError("Ngày bắt đầu phải trước ngày kết thúc")
        
        try:
            # Lấy danh sách ngày nghỉ lễ
            country_holidays = holidays.country_holidays(country, years=range(start_date.year, end_date.year + 1))
            
            # Tạo DatetimeIndex với tần suất ngày làm việc (loại bỏ cuối tuần)
            business_days = pd.date_range(start=start_date, end=end_date, freq='B')
            
            # Loại bỏ ngày nghỉ lễ
            trading_days = [day for day in business_days if day not in country_holidays]
            
            return pd.DatetimeIndex(trading_days)
        except Exception as e:
            logger.warning(f"Lỗi khi tính toán ngày giao dịch: {str(e)}, trả về tất cả ngày làm việc")
            return pd.date_range(start=start_date, end=end_date, freq='B')
    
    @staticmethod
    def validate_api_response(response: Dict[str, Any], expected_keys: List[str]) -> bool:
        """
        Xác thực phản hồi API có đầy đủ các trường dữ liệu cần thiết hay không
        
        Args:
            response: Phản hồi API
            expected_keys: Danh sách các key cần có
            
        Returns:
            True nếu hợp lệ, False nếu không
        """
        if not response:
            return False
        return all(key in response for key in expected_keys)
    
    @staticmethod
    def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
        """
        Kiểm tra cấu trúc và kiểu dữ liệu theo Pandera schema.
        
        Args:
            df: DataFrame cần kiểm tra
            
        Returns:
            DataFrame đã được validate
            
        Raises:
            pa.errors.SchemaError: Nếu dữ liệu không đáp ứng schema
        """
        try:
            # Kiểm tra nếu df không phải DataFrame rỗng
            if df is None or df.empty:
                logger.warning("DataFrame rỗng, bỏ qua validate schema")
                return df
                
            # Đảm bảo các cột cần thiết tồn tại
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.warning(f"Cột {col} không tồn tại, bỏ qua validate schema")
                    return df
            
            # Chỉ validate các cột chính, bỏ qua các cột khác
            df_subset = df[required_cols].copy()
            
            # Chuyển đổi volume sang kiểu int nếu cần
            if df_subset['volume'].dtype != 'int64':
                df_subset['volume'] = df_subset['volume'].astype(int)
            
            # Validate schema
            validated_df = stock_schema.validate(df_subset)
            
            # Đặt lại tất cả các cột của df gốc
            for col in required_cols:
                df[col] = validated_df[col]
                
            logger.info("Schema validation thành công")
            return df
        except pa.errors.SchemaError as e:
            logger.error(f"Schema validation thất bại: {e}")
            # Không crash chương trình, trả về df gốc và log warning
            logger.warning("Tiếp tục với DataFrame chưa được validate")
            return df 
    
    @staticmethod
    def detect_outliers_isolation_forest(df: pd.DataFrame, 
                                        columns: List[str] = None,
                                        contamination: float = 0.01, 
                                        random_state: int = 42) -> pd.Index:
        """
        Phát hiện outlier bằng Isolation Forest
        
        Args:
            df: DataFrame cần phát hiện outlier
            columns: Danh sách cột sử dụng để phát hiện (mặc định là OHLC)
            contamination: Tỷ lệ outlier dự kiến (0.01 = 1%)
            random_state: Random seed để kết quả nhất quán
            
        Returns:
            Index của các outlier
        """
        if df.empty:
            return pd.Index([])
            
        # Sử dụng các cột giá mặc định nếu không chỉ định
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
            # Kiểm tra xem các cột cần thiết có tồn tại không
            columns = [col for col in columns if col in df.columns]
            if not columns:
                logger.warning("Không đủ cột để phát hiện outlier")
                return pd.Index([])
        
        # Tạo bản sao của dữ liệu và chuẩn hóa
        X = df[columns].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Huấn luyện mô hình
        model = IsolationForest(contamination=contamination, random_state=random_state)
        y_pred = model.fit_predict(X_scaled)
        
        # y_pred = -1 là outlier, 1 là inlier
        outlier_indices = df.index[y_pred == -1]
        logger.info(f"Isolation Forest đã phát hiện {len(outlier_indices)} outlier")
        
        return outlier_indices
        
    @staticmethod
    def detect_outliers_lof(df: pd.DataFrame, 
                           columns: List[str] = None,
                           n_neighbors: int = 20, 
                           contamination: float = 0.01) -> pd.Index:
        """
        Phát hiện outlier bằng Local Outlier Factor
        
        Args:
            df: DataFrame cần phát hiện outlier
            columns: Danh sách cột sử dụng để phát hiện (mặc định là OHLC)
            n_neighbors: Số lượng neighbors xem xét
            contamination: Tỷ lệ outlier dự kiến (0.01 = 1%)
            
        Returns:
            Index của các outlier
        """
        if df.empty:
            return pd.Index([])
            
        # Sử dụng các cột giá mặc định nếu không chỉ định
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
            # Kiểm tra xem các cột cần thiết có tồn tại không
            columns = [col for col in columns if col in df.columns]
            if not columns:
                logger.warning("Không đủ cột để phát hiện outlier")
                return pd.Index([])
        
        # Tạo bản sao của dữ liệu và chuẩn hóa
        X = df[columns].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Huấn luyện mô hình
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        y_pred = lof.fit_predict(X_scaled)
        
        # y_pred = -1 là outlier, 1 là inlier
        outlier_indices = df.index[y_pred == -1]
        logger.info(f"Local Outlier Factor đã phát hiện {len(outlier_indices)} outlier")
        
        return outlier_indices
        
    @staticmethod
    def detect_outliers_dbscan(df: pd.DataFrame, 
                              columns: List[str] = None,
                              eps: float = 0.5, 
                              min_samples: int = 5) -> pd.Index:
        """
        Phát hiện outlier bằng DBSCAN
        
        Args:
            df: DataFrame cần phát hiện outlier
            columns: Danh sách cột sử dụng để phát hiện (mặc định là OHLC)
            eps: Khoảng cách tối đa giữa hai mẫu để được coi là cùng vùng
            min_samples: Số mẫu tối thiểu trong một vùng
            
        Returns:
            Index của các outlier
        """
        if df.empty:
            return pd.Index([])
            
        # Sử dụng các cột giá mặc định nếu không chỉ định
        if columns is None:
            columns = ['open', 'high', 'low', 'close']
            # Kiểm tra xem các cột cần thiết có tồn tại không
            columns = [col for col in columns if col in df.columns]
            if not columns:
                logger.warning("Không đủ cột để phát hiện outlier")
                return pd.Index([])
                
        # Tạo bản sao của dữ liệu và chuẩn hóa
        X = df[columns].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Huấn luyện mô hình
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_scaled)
        
        # label = -1 là outlier
        outlier_indices = df.index[labels == -1]
        logger.info(f"DBSCAN đã phát hiện {len(outlier_indices)} outlier")
        
        return outlier_indices 

    @staticmethod
    def normalize_for_prophet(df: pd.DataFrame, date_col: str = 'date', value_col: str = 'close') -> pd.DataFrame:
        """
        Chuẩn hóa DataFrame cho Prophet: cột 'ds' (datetime), 'y' (giá trị dự báo)
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