import numpy as np
import pandas as pd
import logging
from datetime import datetime, date

logger = logging.getLogger(__name__)

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
        if data is None:
            return {}
            
        standardized_data = {}
        for key, value in data.items():
            if value is None:
                standardized_data[key] = None
            elif isinstance(value, (int, float, str, bool)):
                # Các kiểu dữ liệu cơ bản không cần chuyển đổi
                standardized_data[key] = value
            elif isinstance(value, (np.int8, np.int16, np.int32, np.int64)):
                standardized_data[key] = int(value)
            elif isinstance(value, (np.float16, np.float32, np.float64, np.float128)):
                standardized_data[key] = float(value)
            elif isinstance(value, (datetime, date)):
                standardized_data[key] = value.isoformat()
            elif isinstance(value, np.bool_):
                standardized_data[key] = bool(value)
            elif isinstance(value, pd.Timestamp):
                standardized_data[key] = value.to_pydatetime().isoformat()
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                # Chuyển đổi Series hoặc DataFrame thành dict
                try:
                    standardized_data[key] = value.to_dict()
                except:
                    # Fallback nếu to_dict() không hoạt động
                    standardized_data[key] = str(value)
            elif isinstance(value, np.ndarray):
                # Xử lý các trường hợp kiểu dữ liệu đặc biệt trong ndarray
                if np.issubdtype(value.dtype, np.integer):
                    standardized_data[key] = value.astype(int).tolist()
                elif np.issubdtype(value.dtype, np.floating):
                    standardized_data[key] = value.astype(float).tolist()
                elif np.issubdtype(value.dtype, np.bool_):
                    standardized_data[key] = value.astype(bool).tolist()
                else:
                    # Chuyển đổi sang list với phương thức an toàn
                    try:
                        standardized_data[key] = value.tolist()
                    except:
                        standardized_data[key] = list(map(str, value))
            elif isinstance(value, dict):
                # Đệ quy xử lý từng phần tử trong dict
                standardized_data[key] = DataNormalizer.standardize_for_db(value)
            elif isinstance(value, (list, tuple)):
                # Chuyển đổi từng phần tử trong list
                try:
                    standardized_data[key] = [
                        DataNormalizer.standardize_for_db(item) if isinstance(item, dict) 
                        else item if isinstance(item, (int, float, str, bool))
                        else float(item) if isinstance(item, (np.float16, np.float32, np.float64))
                        else int(item) if isinstance(item, (np.int8, np.int16, np.int32, np.int64))
                        else item.isoformat() if isinstance(item, (datetime, date, pd.Timestamp))
                        else str(item)
                        for item in value
                    ]
                except:
                    # Fallback an toàn
                    standardized_data[key] = str(value)
            else:
                # Cho các kiểu dữ liệu không xác định, chuyển sang string
                try:
                    standardized_data[key] = str(value)
                except:
                    standardized_data[key] = "Error: Không thể chuyển đổi kiểu dữ liệu"
                    logger.warning(f"Không thể chuyển đổi kiểu dữ liệu cho key {key}: {type(value)}")
        
        return standardized_data 