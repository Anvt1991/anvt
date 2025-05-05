#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kiểm tra nhanh các module đã sửa đổi
"""

import pandas as pd
import numpy as np
import os
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Thêm thư mục hiện tại vào PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Import các module cần thiết
from core.model.feature_generator import FeatureGenerator
from core.model.ml_model import MLPredictor
from core.model.model_trainer import ModelTrainer
from core.model.enhanced_predictor import EnhancedPredictor
from core.model.recommendation import RecommendationEngine
from core.model.signal_aggregator import SignalAggregator

def create_test_data(n=100, symbol="VNM"):
    """
    Tạo dữ liệu test
    """
    np.random.seed(42)
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
        
    # Thêm cột target để train
    df['target'] = np.where(df['close'].shift(-5) > df['close'], 1, 0)
    
    return df

def test_feature_generator():
    """
    Kiểm tra FeatureGenerator
    """
    print("\n=== Kiểm tra FeatureGenerator ===")
    data = create_test_data()
    
    # Khởi tạo FeatureGenerator
    feature_generator = FeatureGenerator()
    
    # Tạo features cho train
    features = feature_generator.prepare_features(data, feature_set="minimal")
    print(f"Số lượng features: {features.shape[1]}")
    print(f"Số lượng dòng: {features.shape[0]}")
    
    # Tạo features cho inference 
    inference_features = feature_generator.get_features_for_inference(data)
    print(f"Số lượng features cho inference: {inference_features.shape[1]}")
    
    # Kiểm tra feature_names
    feature_names = feature_generator.get_feature_names()
    print(f"Số lượng feature_names: {len(feature_names)}")
    
    return features

def test_ml_predictor(features):
    """
    Kiểm tra MLPredictor
    """
    print("\n=== Kiểm tra MLPredictor ===")
    data = create_test_data()
    
    # Đảm bảo index phù hợp
    if isinstance(features.index, pd.DatetimeIndex):
        # Lấy các ngày có sẵn trong cả hai DataFrame
        common_dates = data.index.intersection(features.index)
        features = features.loc[common_dates]
        y = data.loc[common_dates, 'target']
    else:
        # Tạo target mới nếu index không phải datetime
        y = pd.Series(np.random.randint(0, 2, len(features)), index=features.index)
    
    # Khởi tạo MLPredictor
    config = {
        "model_type": "classifier",
        "model_name": "random_forest",
        "random_state": 42,
        "train_test_split": 0.8,
        "hyperparameters": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10
            }
        }
    }
    
    predictor = MLPredictor(config=config)
    
    # Train model
    result = predictor.train(features, y)
    print(f"Kết quả train: {result}")
    
    # Lưu model tạm thời
    os.makedirs("models", exist_ok=True)
    predictor.save_model("models/test_model.joblib")
    print("Đã lưu model tạm thời")
    
    # Load model
    predictor2 = MLPredictor(config=config)
    predictor2.load_model("models/test_model.joblib")
    print(f"Đã load model với {len(predictor2.feature_names)} features")
    
    # Predict
    sample = features.iloc[-5:].copy()
    predictions = predictor2.batch_predict(sample)
    print(f"Predictions: {predictions[:2]}")
    
    return predictor

def test_model_trainer():
    """
    Kiểm tra ModelTrainer
    """
    print("\n=== Kiểm tra ModelTrainer ===")
    data = create_test_data()
    
    # Khởi tạo ModelTrainer
    config = {
        "model_type": "classifier",
        "model_name": "random_forest",
        "random_state": 42,
        "feature_sets": ["minimal", "full"],
        "train_test_split": 0.8,
        "feature_selection": {
            "correlation_threshold": 0.9,
            "importance_threshold": 0.01,
            "mutual_info_threshold": 0.01
        },
        "hyperparameters": {
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10
            }
        }
    }
    
    feature_generator = FeatureGenerator(config)
    trainer = ModelTrainer(config, feature_generator)
    
    # Train model
    result = trainer.train_model("TEST", data)
    print(f"Kết quả train: {result}")
    
    # Check sync
    synced = trainer.ensure_model_synced("TEST", data)
    print(f"Model synced: {synced}")
    
    return trainer

def test_enhanced_predictor():
    """
    Kiểm tra EnhancedPredictor
    """
    print("\n=== Kiểm tra EnhancedPredictor ===")
    data = create_test_data()
    
    # Khởi tạo EnhancedPredictor
    config = {
        "model_type": "classifier",
        "model_name": "random_forest",
        "random_state": 42,
        "feature_sets": ["minimal", "full"],
        "train_test_split": 0.8,
        "weights": {
            "ml_short": 0.5,
            "ml_medium": 0.3,
            "technical": 0.1,
            "sentiment": 0.1
        }
    }
    
    predictor = EnhancedPredictor(config)
    
    # Create some tech signals
    tech_analysis = {
        "signals": {
            "macd": "BUY",
            "rsi": "BUY",
            "stoch": "SELL"
        }
    }
    
    # Add some sentiment data
    news_sentiment = {
        "overall_score": 0.3,
        "news_count": 5
    }
    
    # Predict
    result = predictor.predict("TEST", data, tech_analysis, news_sentiment)
    print(f"Kết quả dự đoán: {result}")
    
    return predictor

def run_all_tests():
    """
    Chạy tất cả các tests
    """
    print("=== BẮT ĐẦU KIỂM TRA NHANH ===")
    try:
        features = test_feature_generator()
        try:
            test_ml_predictor(features)
        except Exception as e:
            print(f"Lỗi trong MLPredictor: {str(e)}")
        try:
            test_model_trainer()
        except Exception as e:
            print(f"Lỗi trong ModelTrainer: {str(e)}")
        try:
            test_enhanced_predictor()
        except Exception as e:
            print(f"Lỗi trong EnhancedPredictor: {str(e)}")
    except Exception as e:
        print(f"Lỗi chung: {str(e)}")
    print("\n=== ĐÃ HOÀN THÀNH KIỂM TRA ===")

if __name__ == "__main__":
    run_all_tests() 