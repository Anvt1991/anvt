#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests cho PipelineProcessor và các thành phần của nó
Đảm bảo các thành phần được đăng ký và hoạt động đúng
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
from typing import Optional, Dict, Any

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('test_pipeline')

# Import các module cần thiết
from pipeline.processor import PipelineProcessor
from pipeline.interfaces import (
    StockData, ValidationResult, FeatureEngineeringResult, 
    TechnicalAnalysisResult, PredictionResult, NewsResult,
    AIAnalysisResult, EnhancedPredictionResult, PortfolioResult,
    BacktestResult, ReportResult, PipelineResult
)
from core.data.data_validator import DataValidator
from core.model.feature_generator import FeatureGenerator
from core.technical import TechnicalAnalyzer
from core.model.ml_model import MLPredictor
from core.news.news import NewsLoader
from core.model.enhanced_predictor import EnhancedPredictor
from core.strategy.backtester import Backtester

class TestPipelineProcessor(unittest.TestCase):
    """Test cases cho PipelineProcessor"""
    
    def setUp(self):
        """Khởi tạo môi trường test"""
        self.pipeline = PipelineProcessor()
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
    
    def test_register_components(self):
        """Kiểm tra đăng ký các thành phần vào pipeline"""
        
        # Đăng ký các thành phần
        self.pipeline.register_data_validator(self._mock_data_validator)
        self.pipeline.register_feature_engineer(self._mock_feature_engineer)
        self.pipeline.register_technical_analyzer(self._mock_technical_analyzer)
        self.pipeline.register_prediction_generator(self._mock_prediction_generator)
        self.pipeline.register_news_analyzer(self._mock_news_analyzer)
        self.pipeline.register_ai_analyzer(self._mock_ai_analyzer)
        self.pipeline.register_prediction_enhancer(self._mock_prediction_enhancer)
        self.pipeline.register_portfolio_optimizer(self._mock_portfolio_optimizer)
        self.pipeline.register_backtester(self._mock_backtester)
        self.pipeline.register_report_generator(self._mock_report_generator)
        self.pipeline.register_database_saver(self._mock_database_saver)
        
        # Kiểm tra số lượng thành phần đã đăng ký
        self.assertEqual(len(self.pipeline.data_validators), 1)
        self.assertEqual(len(self.pipeline.feature_engineers), 1)
        self.assertEqual(len(self.pipeline.technical_analyzers), 1)
        self.assertEqual(len(self.pipeline.prediction_generators), 1)
        self.assertEqual(len(self.pipeline.news_analyzers), 1)
        self.assertEqual(len(self.pipeline.ai_analyzers), 1)
        self.assertEqual(len(self.pipeline.prediction_enhancers), 1)
        self.assertEqual(len(self.pipeline.portfolio_optimizers), 1)
        self.assertEqual(len(self.pipeline.backtesters), 1)
        self.assertEqual(len(self.pipeline.report_generators), 1)
        self.assertEqual(len(self.pipeline.database_savers), 1)
        
    def test_pipeline_execution(self):
        """Kiểm tra quá trình thực thi pipeline"""
        
        # Đăng ký các thành phần mock
        self.pipeline.register_data_validator(self._mock_data_validator)
        self.pipeline.register_feature_engineer(self._mock_feature_engineer)
        self.pipeline.register_technical_analyzer(self._mock_technical_analyzer)
        self.pipeline.register_prediction_generator(self._mock_prediction_generator)
        self.pipeline.register_news_analyzer(self._mock_news_analyzer)
        self.pipeline.register_ai_analyzer(self._mock_ai_analyzer)
        self.pipeline.register_prediction_enhancer(self._mock_prediction_enhancer)
        self.pipeline.register_portfolio_optimizer(self._mock_portfolio_optimizer)
        self.pipeline.register_backtester(self._mock_backtester)
        self.pipeline.register_report_generator(self._mock_report_generator)
        self.pipeline.register_database_saver(self._mock_database_saver)
        
        # Thực thi pipeline
        result = self.pipeline.process(self.symbol, self.stock_data)
        
        # Kiểm tra kết quả
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.symbol, self.symbol)
        self.assertIsNotNone(result.validation_result)
        self.assertIsNotNone(result.feature_engineering_result)
        self.assertIsNotNone(result.technical_analysis_result)
        self.assertIsNotNone(result.prediction_result)
        self.assertIsNotNone(result.news_result)
        self.assertIsNotNone(result.ai_analysis_result)
        self.assertIsNotNone(result.enhanced_prediction_result)
        self.assertIsNotNone(result.portfolio_result)
        self.assertIsNotNone(result.backtest_result)
        self.assertIsNotNone(result.report_result)
        
    def test_error_handling(self):
        """Kiểm tra xử lý lỗi trong pipeline"""
        
        # Đăng ký thành phần gây lỗi
        self.pipeline.register_data_validator(self._error_data_validator)
        
        # Cấu hình pipeline để dừng khi có lỗi
        self.pipeline.update_config({'error_tolerance': 'strict'})
        
        # Thực thi pipeline và kiểm tra lỗi
        result = self.pipeline.process(self.symbol, self.stock_data)
        
        # Kiểm tra kết quả
        self.assertIsInstance(result, PipelineResult)
        self.assertTrue(result.error is not None)
        self.assertIn("Simulated error in data validator", result.error)
    
    def test_parallel_execution(self):
        """Kiểm tra thực thi song song"""
        
        # Đăng ký các thành phần mock
        self.pipeline.register_data_validator(self._mock_data_validator)
        self.pipeline.register_feature_engineer(self._mock_feature_engineer)
        
        # Cấu hình pipeline cho thực thi song song
        self.pipeline.update_config({
            'parallel_execution': True,
            'max_workers': 2
        })
        
        # Thực thi song song với nhiều symbols
        symbols = ['VNM', 'FPT', 'VIC']
        results = self.pipeline.parallel_process(symbols)
        
        # Kiểm tra kết quả
        self.assertEqual(len(results), 3)
        for symbol in symbols:
            self.assertIn(symbol, results)
            self.assertIsInstance(results[symbol], PipelineResult)
    
    # Mock functions cho các thành phần
    def _mock_data_validator(self, stock_data: StockData) -> ValidationResult:
        return ValidationResult(is_valid=True)
    
    def _mock_feature_engineer(self, stock_data: StockData) -> FeatureEngineeringResult:
        return FeatureEngineeringResult(
            features={'rsi': pd.Series(np.random.rand(len(stock_data.df)))},
            feature_importance={'rsi': 0.8}
        )
    
    def _mock_technical_analyzer(self, stock_data: StockData, feature_result: Optional[FeatureEngineeringResult] = None) -> TechnicalAnalysisResult:
        return TechnicalAnalysisResult(
            indicators={'macd': pd.Series(np.random.rand(len(stock_data.df)))},
            signals={'trend': 'up'},
            patterns={'pattern1': 'bullish'}
        )
    
    def _mock_prediction_generator(self, stock_data: StockData, tech_result: TechnicalAnalysisResult) -> PredictionResult:
        return PredictionResult(
            prediction='bullish',
            confidence=0.85,
            prediction_horizon='short-term',
            target_price=110.5
        )
    
    def _mock_news_analyzer(self, symbol: str, stock_df: Optional[pd.DataFrame] = None) -> NewsResult:
        return NewsResult(
            news_items=[],
            sentiment_score=0.7
        )
    
    def _mock_ai_analyzer(self, stock_data: StockData, tech_result: TechnicalAnalysisResult, news_result: Optional[NewsResult] = None) -> AIAnalysisResult:
        return AIAnalysisResult(
            sentiment='bullish',
            confidence=0.8,
            insights=['Strong buying pressure']
        )
    
    def _mock_prediction_enhancer(self, pred_result: PredictionResult, ai_result: AIAnalysisResult, news_result: NewsResult) -> EnhancedPredictionResult:
        return EnhancedPredictionResult(
            prediction='bullish',
            confidence=0.9,
            prediction_horizon='short-term',
            target_price=115.0
        )
    
    def _mock_portfolio_optimizer(self, pred_result: EnhancedPredictionResult, tech_result: TechnicalAnalysisResult) -> PortfolioResult:
        return PortfolioResult(
            position_type='buy',
            position_size=0.2,
            stop_loss=95.0,
            take_profit=120.0
        )
    
    def _mock_backtester(self, stock_data: StockData, portfolio_result: PortfolioResult) -> BacktestResult:
        return BacktestResult(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=0.05
        )
    
    def _mock_report_generator(self, pipeline_data) -> ReportResult:
        return ReportResult(
            summary="This is a test report",
            sections={
                'technical': 'Technical analysis looks positive',
                'sentiment': 'Market sentiment is bullish'
            }
        )
    
    def _mock_database_saver(self, pipeline_result: PipelineResult) -> bool:
        return True
    
    # Mock function gây lỗi
    def _error_data_validator(self, stock_data: StockData) -> ValidationResult:
        raise ValueError("Simulated error in data validator")

if __name__ == "__main__":
    unittest.main() 