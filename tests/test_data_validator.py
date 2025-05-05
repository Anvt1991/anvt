#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script test cho DataValidator, đặc biệt là các tính năng mới
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
import sys

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_validator')

# Import DataValidator
from core.data.data_validator import DataValidator
from core.technical import TechnicalAnalyzer

def create_test_dataframe(with_outliers=True):
    """Tạo DataFrame test với các outlier"""
    # Tạo dữ liệu giả lập
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
    
    # Thêm outlier nếu cần
    if with_outliers:
        # Thêm một số outlier
        outlier_indices = [10, 25, 50, 75]
        
        for idx in outlier_indices:
            # Outlier giá
            if idx == 10:
                df.iloc[idx, 0] = 200  # open
            elif idx == 25:
                df.iloc[idx, 3] = 50   # close
            elif idx == 50:
                df.iloc[idx, 1] = 300  # high
            elif idx == 75:
                df.iloc[idx, 2] = 30   # low
                
        # Outlier khối lượng
        df.iloc[60, 4] = 100000  # volume
    
    return df

def test_schema_validation():
    """Test tính năng validate schema"""
    logger.info("\n== TEST SCHEMA VALIDATION ==")
    
    # Tạo DataFrame hợp lệ
    df = create_test_dataframe(with_outliers=False)
    
    # Chuyển volume sang float để test kiểu dữ liệu
    df['volume'] = df['volume'].astype(float)
    
    logger.info(f"DataFrame gốc:\n{df.head()}")
    
    # Validate schema
    try:
        validated_df = DataValidator.validate_schema(df)
        logger.info("Schema validation thành công")
        logger.info(f"Kiểu dữ liệu sau validation:\n{validated_df.dtypes}")
    except Exception as e:
        logger.error(f"Schema validation thất bại: {e}")
    
    # Test với dữ liệu không hợp lệ
    df_invalid = df.copy()
    df_invalid.iloc[0, 0] = -10  # open < 0
    
    try:
        validated_df = DataValidator.validate_schema(df_invalid)
        logger.info("Đã xử lý dữ liệu không hợp lệ")
    except Exception as e:
        logger.error(f"Lỗi với dữ liệu không hợp lệ: {e}")
    
    return validated_df

def test_outlier_detection(method='isolation_forest'):
    """Test các phương pháp phát hiện outlier"""
    logger.info(f"\n== TEST OUTLIER DETECTION: {method} ==")
    
    # Tạo DataFrame có outlier
    df = create_test_dataframe(with_outliers=True)
    
    logger.info(f"DataFrame gốc shape: {df.shape}")
    
    # Cấu hình ml_params
    ml_params = {}
    if method == 'isolation_forest':
        ml_params = {'contamination': 0.05, 'random_state': 42}
    elif method == 'lof':
        ml_params = {'n_neighbors': 10, 'contamination': 0.05}
    elif method == 'dbscan':
        ml_params = {'eps': 10, 'min_samples': 3}
    
    # Phát hiện và xử lý outlier
    df_clean, report = DataValidator.detect_and_handle_outliers(df, method=method, ml_params=ml_params)
    
    logger.info(f"DataFrame sau xử lý outlier shape: {df_clean.shape}")
    logger.info(f"Báo cáo outlier:\n{report}")
    
    # Trực quan hóa kết quả nếu có thay đổi
    if df.shape != df_clean.shape:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot giá đóng cửa
        axes[0, 0].plot(df.index, df['close'], 'b-', label='Gốc')
        axes[0, 0].plot(df_clean.index, df_clean['close'], 'r--', label='Đã xử lý')
        axes[0, 0].set_title('Giá đóng cửa')
        axes[0, 0].legend()
        
        # Plot khối lượng
        axes[0, 1].plot(df.index, df['volume'], 'b-', label='Gốc')
        axes[0, 1].plot(df_clean.index, df_clean['volume'], 'r--', label='Đã xử lý')
        axes[0, 1].set_title('Khối lượng')
        axes[0, 1].legend()
        
        # Plot high-low range
        axes[1, 0].fill_between(df.index, df['high'], df['low'], alpha=0.3, label='Gốc')
        axes[1, 0].fill_between(df_clean.index, df_clean['high'], df_clean['low'], alpha=0.3, color='r', label='Đã xử lý')
        axes[1, 0].set_title('Biên độ High-Low')
        axes[1, 0].legend()
        
        # Box plot các giá trị close
        axes[1, 1].boxplot([df['close'], df_clean['close']], labels=['Gốc', 'Đã xử lý'])
        axes[1, 1].set_title('Box plot giá đóng cửa')
        
        plt.tight_layout()
        plt.savefig(f'outlier_detection_{method}.png')
        logger.info(f"Đã lưu biểu đồ vào outlier_detection_{method}.png")
    
    return df_clean

def test_compare_methods():
    """So sánh các phương pháp phát hiện outlier"""
    logger.info("\n== COMPARE OUTLIER DETECTION METHODS ==")
    
    # Tạo DataFrame có outlier
    df = create_test_dataframe(with_outliers=True)
    
    # Các phương pháp cần test
    methods = ['iqr', 'zscore', 'isolation_forest', 'lof', 'dbscan']
    results = {}
    
    for method in methods:
        logger.info(f"Kiểm tra phương pháp: {method}")
        df_clean, report = DataValidator.detect_and_handle_outliers(
            df, method=method, 
            ml_params={'contamination': 0.05} if method in ['isolation_forest', 'lof'] else {}
        )
        
        # Lưu kết quả
        results[method] = {
            'shape': df_clean.shape,
            'report': report
        }
        
        logger.info(f"Phương pháp {method}: từ {df.shape} -> {df_clean.shape}")
    
    # So sánh kết quả
    logger.info("\n=== KẾT QUẢ SO SÁNH ===")
    for method, result in results.items():
        logger.info(f"{method}: {df.shape} -> {result['shape']}")
    
    return results

def test_find_support_resistance():
    """Kiểm tra phương thức tìm mức hỗ trợ và kháng cự"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from technical_analyzer import TechnicalAnalyzer
    
    # Tạo dữ liệu mẫu
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    
    # Tạo một mẫu dữ liệu với xu hướng tăng, có các điểm swing points rõ ràng
    # Giá dao động từ 10 đến 20, với các điểm hỗ trợ ở 12 và 15, kháng cự ở 17 và 19
    close_prices = []
    for i in range(100):
        if i < 20:
            # Tạo một xu hướng ban đầu đi ngang quanh 12
            close_prices.append(12 + np.random.normal(0, 0.2))
        elif i < 30:
            # Tăng lên mức 15
            close_prices.append(12 + (i-20)*0.3 + np.random.normal(0, 0.2))
        elif i < 50:
            # Đi ngang quanh 15
            close_prices.append(15 + np.random.normal(0, 0.3))
        elif i < 60:
            # Tăng lên mức 17
            close_prices.append(15 + (i-50)*0.2 + np.random.normal(0, 0.2))
        elif i < 70:
            # Đi ngang tại 17, thử phá vỡ 19 nhưng thất bại
            if i == 65:
                close_prices.append(19)  # Thử chạm mức 19
            else:
                close_prices.append(17 + np.random.normal(0, 0.3))
        elif i < 80:
            # Điều chỉnh xuống về 15
            close_prices.append(17 - (i-70)*0.2 + np.random.normal(0, 0.2))
        else:
            # Tăng lại lên 17
            close_prices.append(15 + (i-80)*0.2 + np.random.normal(0, 0.2))
    
    # Tạo dữ liệu OHLC
    high_prices = [p + np.random.uniform(0.1, 0.5) for p in close_prices]
    low_prices = [p - np.random.uniform(0.1, 0.5) for p in close_prices]
    open_prices = [low_prices[i] + np.random.uniform(0, high_prices[i] - low_prices[i]) for i in range(100)]
    volume = [int(1000000 * np.random.uniform(0.5, 1.5)) for _ in range(100)]
    
    # Tạo DataFrame
    data = {
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }
    df = pd.DataFrame(data, index=dates)
    
    # Tiến hành test
    analyzer = TechnicalAnalyzer()
    support, resistance = analyzer._find_support_resistance(df)
    
    # In kết quả để kiểm tra
    print(f"Dữ liệu mẫu - Giá đóng cửa hiện tại: {close_prices[-1]}")
    print(f"Mức hỗ trợ phát hiện được: {support}")
    print(f"Mức kháng cự phát hiện được: {resistance}")
    
    # Kiểm tra xem hỗ trợ có dưới giá hiện tại không
    assert support < close_prices[-1], f"Mức hỗ trợ ({support}) phải thấp hơn giá hiện tại ({close_prices[-1]})"
    
    # Kiểm tra xem kháng cự có trên giá hiện tại không
    assert resistance > close_prices[-1], f"Mức kháng cự ({resistance}) phải cao hơn giá hiện tại ({close_prices[-1]})"
    
    # Kiểm tra các mức đã biết trước
    # Do dữ liệu là ngẫu nhiên, chúng ta kiểm tra khoảng giá trị
    assert 11.5 <= support <= 12.5 or 14.5 <= support <= 15.5, f"Mức hỗ trợ phải gần với 12 hoặc 15, nhận được {support}"
    assert 16.5 <= resistance <= 17.5 or 18.5 <= resistance <= 19.5, f"Mức kháng cự phải gần với 17 hoặc 19, nhận được {resistance}"
    
    # Kiểm tra với dữ liệu trống và nhỏ
    empty_df = pd.DataFrame()
    small_df = df.iloc[-5:]
    
    # Kiểm tra xem có xử lý tốt dữ liệu trống không
    fallback_support, fallback_resistance = analyzer._find_support_resistance(empty_df)
    assert fallback_support < fallback_resistance, "Ngay cả với dữ liệu trống, mức hỗ trợ phải thấp hơn mức kháng cự"
    
    # Kiểm tra dữ liệu quá ít
    small_support, small_resistance = analyzer._find_support_resistance(small_df)
    assert small_support < small_resistance, "Với dữ liệu nhỏ, mức hỗ trợ vẫn phải thấp hơn mức kháng cự"
    
    print("Test tìm mức hỗ trợ và kháng cự thành công!")

if __name__ == "__main__":
    logger.info("=== BẮT ĐẦU KIỂM TRA DATA VALIDATOR ===")
    
    # Test schema validation
    validated_df = test_schema_validation()
    
    # Test từng phương pháp phát hiện outlier
    test_outlier_detection(method='iqr')  # Phương pháp cũ
    test_outlier_detection(method='isolation_forest')  # Phương pháp mới
    test_outlier_detection(method='lof')  # Phương pháp mới
    test_outlier_detection(method='dbscan')  # Phương pháp mới
    
    # So sánh các phương pháp
    results = test_compare_methods()
    
    # Test tìm mức hỗ trợ và kháng cự
    test_find_support_resistance()
    
    logger.info("=== HOÀN TẤT KIỂM TRA ===") 