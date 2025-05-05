#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script kiểm tra tích hợp để đảm bảo DataValidator mới tương thích với các module khác
"""

import pandas as pd
import numpy as np
import logging
import sys
from datetime import datetime, timedelta
import pytz

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_integration')

# Import DataValidator từ tệp đã được cập nhật
from core.data.data_validator import DataValidator

# Hàm giả lập của module DataLoader
def mock_load_stock_data(symbol):
    """Tạo dữ liệu chứng khoán giả lập"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
    
    # Dữ liệu cơ bản
    df = pd.DataFrame({
        'open': np.random.normal(100, 5, n),
        'high': np.random.normal(105, 5, n),
        'low': np.random.normal(95, 5, n),
        'close': np.random.normal(102, 5, n),
        'volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    
    # Đảm bảo high > low và giá > 0
    for i in range(n):
        df.iloc[i, 1] = max(df.iloc[i, [0, 1, 2, 3]])  # high = max(open, high, low, close)
        df.iloc[i, 2] = min(df.iloc[i, [0, 1, 2, 3]])  # low = min(open, high, low, close)
    
    # Thêm một số outlier
    if symbol == 'VNM':
        df.iloc[10, 0] = 200  # open outlier
        df.iloc[60, 4] = 100000  # volume outlier
    
    return df

def test_pipeline():
    """
    Test pipeline xử lý dữ liệu để đảm bảo DataValidator mới
    tương thích với quy trình hiện tại
    """
    logger.info("===== TEST PIPELINE =====")
    symbols = ['VNM', 'FPT', 'VIC']
    
    for symbol in symbols:
        logger.info(f"\nXử lý dữ liệu cho {symbol}")
        
        # Bước 1: Load dữ liệu
        df = mock_load_stock_data(symbol)
        logger.info(f"Dữ liệu gốc shape: {df.shape}")
        
        # Bước 2: Validate schema (tính năng mới)
        df = DataValidator.validate_schema(df)
        logger.info(f"Sau validate schema shape: {df.shape}")
        
        # Bước 3: Chuẩn hóa DataFrame (tính năng cũ)
        df = DataValidator.normalize_dataframe(df)
        logger.info(f"Sau normalize shape: {df.shape}")
        
        # Bước 4: Phát hiện và xử lý outlier
        # Sử dụng cả phương pháp cũ và mới để so sánh
        methods = ['iqr', 'isolation_forest']
        
        for method in methods:
            logger.info(f"\nPhương pháp phát hiện outlier: {method}")
            df_copy = df.copy()
            
            ml_params = {}
            if method == 'isolation_forest':
                ml_params = {'contamination': 0.05, 'random_state': 42}
                
            df_clean, report = DataValidator.detect_and_handle_outliers(
                df_copy, method=method, ml_params=ml_params
            )
            
            logger.info(f"Sau xử lý outlier shape: {df_clean.shape}")
            logger.info(f"Report summary: {report.split(':', 1)[0]}")
            
            # Bước 5: Căn chỉnh timestamps (tính năng cũ)
            df_aligned = DataValidator.align_timestamps(df_clean, '1D')
            logger.info(f"Sau align timestamps shape: {df_aligned.shape}")
    
    logger.info("\n===== TÍCH HỢP THÀNH CÔNG =====")
    return True

def test_normalize_validation():
    """
    Test quy trình chuẩn hóa và validate với dữ liệu không hợp lệ
    để đảm bảo vẫn hoạt động với phần cũ và mới
    """
    logger.info("\n===== TEST NORMALIZE & VALIDATION =====")
    
    # Tạo DataFrame có vấn đề cần chuẩn hóa
    df = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=10),
        'Open': np.random.normal(100, 5, 10),
        'High': np.random.normal(105, 5, 10),
        'Low': np.random.normal(95, 5, 10),
        'Close': np.random.normal(102, 5, 10),
        'Volume': np.random.randint(1000, 10000, 10),
        'extra_col': np.random.randn(10)
    })
    
    # Chèn một vài giá trị null
    df.loc[2, 'Open'] = None
    df.loc[5, 'Close'] = None
    
    logger.info(f"DataFrame gốc columns: {df.columns.tolist()}")
    
    # Đầu tiên chuẩn hóa
    df_norm = DataValidator.normalize_dataframe(df)
    logger.info(f"Sau normalize columns: {df_norm.columns.tolist()}")
    logger.info(f"Null values: {df_norm.isnull().sum().sum()}")
    
    # Sau đó validate schema
    df_valid = DataValidator.validate_schema(df_norm)
    
    # Test quy trình ngược lại: validate trước, normalize sau
    df2 = df.copy()
    # Đổi tên cột để đảm bảo schema pass
    df2.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'extra_col']
    
    # Validate schema
    df2_valid = DataValidator.validate_schema(df2)
    
    # Sau đó normalize
    df2_norm = DataValidator.normalize_dataframe(df2_valid)
    
    logger.info(f"Quy trình 1 (norm -> valid) shape: {df_valid.shape}")
    logger.info(f"Quy trình 2 (valid -> norm) shape: {df2_norm.shape}")
    
    # So sánh kết quả
    if df_valid.equals(df2_norm):
        logger.info("Hai quy trình cho kết quả giống nhau")
    else:
        logger.info("Hai quy trình cho kết quả khác nhau")
    
    return df_valid, df2_norm

if __name__ == "__main__":
    logger.info("===== BẮT ĐẦU KIỂM TRA TÍCH HỢP =====")
    
    # Test quy trình chuẩn hóa và validate
    df_valid, df2_norm = test_normalize_validation()
    
    # Test pipeline
    test_pipeline()
    
    logger.info("===== HOÀN TẤT KIỂM TRA TÍCH HỢP =====") 