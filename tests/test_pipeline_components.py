#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests cho các thành phần riêng lẻ của Pipeline
Đảm bảo mỗi module được triển khai đầy đủ và hoạt động đúng
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from typing import Optional, Dict, Any, List

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_pipeline_components')

# Import các module cần thiết
from pipeline.interfaces import (
    StockData, ValidationResult, FeatureEngineeringResult, 
    TechnicalAnalysisResult, PredictionResult, NewsResult,
    AIAnalysisResult, EnhancedPredictionResult, PortfolioResult,
    BacktestResult, ReportResult, PipelineData
)
from core.data.data_validator import DataValidator
from core.model.feature_generator import FeatureGenerator
from core.technical import TechnicalAnalyzer
from core.model.ml_model import MLPredictor
from core.news.news import NewsLoader
from core.model.enhanced_predictor import EnhancedPredictor
from core.strategy.backtester import Backtester

class TestPipelineComponents(unittest.TestCase):
    """Test cases cho từng thành phần của pipeline"""
    
    def setUp(self):
        """Khởi tạo môi trường test"""
        self.symbol = "VNM"
        self.stock_data = self._create_mock_stock_data()
        
    def _create_mock_stock_data(self) -> StockData:
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
        
        return StockData(
            df=df,
            symbol=self.symbol,
            start_date=dates[0],
            end_date=dates[-1],
            timeframe="daily"
        )
    
    def test_data_validator(self):
        """Kiểm tra module DataValidator"""
        # Kiểm tra các phương thức tĩnh cần thiết
        self.assertTrue(hasattr(DataValidator, 'validate_schema'))
        self.assertTrue(hasattr(DataValidator, 'normalize_dataframe'))
        self.assertTrue(hasattr(DataValidator, 'detect_and_handle_outliers'))
        
        # Tạo instance
        validator = DataValidator()
        
        # Kiểm tra phương thức validate
        result = validator.validate(self.stock_data)
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid)
    
    def test_feature_engineer(self):
        """Kiểm tra module FeatureEngineer"""
        # Tạo instance
        engineer = FeatureGenerator()
        
        # Kiểm tra các phương thức chính
        self.assertTrue(hasattr(engineer, 'engineer_features'))
        
        # Kiểm tra kết quả
        result = engineer.engineer_features(self.stock_data)
        self.assertIsInstance(result, FeatureEngineeringResult)
        self.assertTrue(len(result.features) > 0)
    
    def test_technical_analyzer(self):
        """Kiểm tra module TechnicalAnalyzer"""
        # Tạo instance
        analyzer = TechnicalAnalyzer()
        
        # Kiểm tra các phương thức chính
        self.assertTrue(hasattr(analyzer, 'analyze'))
        
        # Tạo feature engineering result
        feature_result = FeatureEngineeringResult(
            features={'rsi': pd.Series(np.random.rand(len(self.stock_data.df)))},
            feature_importance={'rsi': 0.8}
        )
        
        # Kiểm tra kết quả
        result = analyzer.analyze(self.stock_data, feature_result)
        self.assertIsInstance(result, TechnicalAnalysisResult)
        self.assertTrue(len(result.indicators) > 0)
    
    def test_ml_predictor(self):
        """Kiểm tra module MLPredictor"""
        # Tạo instance
        predictor = MLPredictor()
        
        # Kiểm tra các phương thức chính
        self.assertTrue(hasattr(predictor, 'predict'))
        
        # Tạo technical analysis result
        tech_result = TechnicalAnalysisResult(
            indicators={'macd': pd.Series(np.random.rand(len(self.stock_data.df)))},
            signals={'trend': 'up'}
        )
        
        # Kiểm tra kết quả
        result = predictor.predict(self.stock_data, tech_result)
        self.assertIsInstance(result, PredictionResult)
        self.assertIn(result.prediction, ['bullish', 'bearish', 'neutral'])
    
    def test_news_loader(self):
        """Kiểm tra module NewsLoader"""
        # Tạo instance
        news_loader = NewsLoader()
        
        # Kiểm tra các phương thức chính
        self.assertTrue(hasattr(news_loader, 'get_news'))
        self.assertTrue(hasattr(news_loader, 'analyze_news'))
        
        # Kiểm tra kết quả
        result = news_loader.analyze_news(self.symbol)
        self.assertIsInstance(result, NewsResult)
    
    def test_portfolio_optimizer(self):
        """Kiểm tra module PortfolioOptimizer"""
        # Tạo instance
        optimizer = PortfolioOptimizer()
        
        # Kiểm tra các phương thức chính
        self.assertTrue(hasattr(optimizer, 'optimize'))
        
        # Tạo enhanced prediction result
        pred_result = EnhancedPredictionResult(
            prediction='bullish',
            confidence=0.9,
            prediction_horizon='short-term'
        )
        
        # Tạo technical analysis result
        tech_result = TechnicalAnalysisResult(
            indicators={},
            signals={'trend': 'up'}
        )
        
        # Kiểm tra kết quả
        result = optimizer.optimize(pred_result, tech_result)
        self.assertIsInstance(result, PortfolioResult)
        self.assertIn(result.position_type, ['buy', 'sell', 'hold'])
    
    def test_backtester(self):
        """Kiểm tra module Backtester"""
        # Tạo instance
        backtester = Backtester()
        
        # Kiểm tra các phương thức chính
        self.assertTrue(hasattr(backtester, 'backtest'))
        
        # Tạo portfolio result
        portfolio_result = PortfolioResult(
            position_type='buy',
            position_size=0.1,
            stop_loss=90.0,
            take_profit=120.0
        )
        
        # Kiểm tra kết quả
        result = backtester.backtest(self.stock_data, portfolio_result)
        self.assertIsInstance(result, BacktestResult)
        self.assertIsNotNone(result.total_return)
    
    def test_component_interfaces(self):
        """Kiểm tra từng thành phần tuân thủ giao diện"""
        # Danh sách các module cần kiểm tra
        components = [
            (DataValidator, ValidationResult),
            (FeatureGenerator, FeatureEngineeringResult),
            (TechnicalAnalyzer, TechnicalAnalysisResult),
            (MLPredictor, PredictionResult),
            (NewsLoader, NewsResult),
            (EnhancedPredictor, EnhancedPredictionResult),
            (Backtester, BacktestResult)
        ]
        
        for component_class, result_class in components:
            # Kiểm tra khởi tạo component
            component = component_class()
            self.assertIsNotNone(component)
            
            # Kiểm tra các phương thức phổ biến
            # Lưu ý: Đây chỉ là kiểm tra cấu trúc, không phải logic
            if hasattr(component, 'validate'):
                self.assertTrue(callable(getattr(component, 'validate')))
            
            if hasattr(component, 'analyze'):
                self.assertTrue(callable(getattr(component, 'analyze')))
            
            if hasattr(component, 'predict'):
                self.assertTrue(callable(getattr(component, 'predict')))
            
            if hasattr(component, 'generate'):
                self.assertTrue(callable(getattr(component, 'generate')))

if __name__ == "__main__":
    unittest.main() 