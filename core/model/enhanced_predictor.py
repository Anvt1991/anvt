#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Predictor for stock market analysis
Combines technical, ML, fundamental, and sentiment analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import os
import json
import matplotlib.pyplot as plt
from core.model.feature_generator import FeatureGenerator
from core.model.ml_model import MLPredictor
from core.strategy.backtester import Backtester
from core.data.data_validator import DataValidator
import joblib
from core.model.model_trainer import ModelTrainer
from core.model.signal_aggregator import SignalAggregator
from core.model.recommendation import RecommendationEngine
from core.model.config import ConfigManager

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPredictor:
    """
    Orchestrator: Kết hợp các thành phần ModelTrainer, SignalAggregator, RecommendationEngine, ConfigManager
    """
    def __init__(self, config_path=None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.feature_generator = FeatureGenerator(config=self.config.get("feature_engineering", {}))
        self.model_trainer = ModelTrainer(self.config, self.feature_generator)
        self.signal_aggregator = SignalAggregator(self.config)
        self.recommendation_engine = RecommendationEngine(self.config)

    def predict(self, symbol: str, data: pd.DataFrame,
                tech_analysis: Optional[Dict[str, Any]] = None,
                news_sentiment: Optional[Dict[str, Any]] = None, 
                market_condition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Thực hiện dự đoán nâng cao cho một cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu
            data: DataFrame với dữ liệu giá
            tech_analysis: Kết quả phân tích kỹ thuật (optional)
            news_sentiment: Sentiment từ tin tức (optional)
            market_condition: Điều kiện thị trường (optional)
            
        Returns:
            Dict với kết quả dự đoán
        """
        try:
            logger.info(f"Starting enhanced prediction for {symbol}")
            # Đảm bảo model đã được đồng bộ trước khi dự đoán
            model_synced = self.model_trainer.ensure_model_synced(symbol, data, tech_analysis, news_sentiment, market_condition)
            if not model_synced:
                logger.error(f"Failed to sync model for {symbol}")
                return {"error": "Model sync failed"}
                
            # Generate features for prediction
            features = self.feature_generator.get_features_for_inference(data, tech_analysis, news_sentiment, market_condition)
            
            if features is None or features.empty:
                logger.error(f"Feature generator returned empty DataFrame for {symbol}")
                return {"error": "No features generated for prediction"}
                
            # Predict ML score
            ml_scores = self.model_trainer.ml_predictor.predict_ml_score(symbol, features)
            
            # Check if ml_scores contains an error
            if "error" in ml_scores:
                logger.error(f"ML prediction error for {symbol}: {ml_scores['error']}")
                return {"error": ml_scores["error"]}
                
            # Get technical signals
            tech_signals = {}
            if tech_analysis:
                tech_signals = tech_analysis.get('signals', {})
                
            # Combine recommendations
            recommendation = self.recommendation_engine.generate(
                ml_scores, tech_signals, news_sentiment
            )
            
            # Combine all results
            result = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "ml_scores": ml_scores,
                "technical_signals": tech_signals,
                "recommendation": recommendation
            }
            
            # Ensure proper format for downstream consumers
            if "short_term" not in result["ml_scores"]:
                result["ml_scores"]["short_term"] = 0
            if "medium_term" not in result["ml_scores"]:
                result["ml_scores"]["medium_term"] = 0
            if "long_term" not in result["ml_scores"]:
                result["ml_scores"]["long_term"] = 0
            
            # Log success
            logger.info(f"Enhanced prediction completed for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction for {symbol}: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}

    def train_model(self, symbol: str, data: pd.DataFrame, tech_analysis=None, news_sentiment=None, market_condition=None):
        return self.model_trainer.train_model(symbol, data, tech_analysis, news_sentiment, market_condition)

    def _calculate_technical_score(self, tech_analysis: Optional[Dict[str, Any]]) -> float:
        if not tech_analysis:
            return 0.0
        try:
            signals = []
            weights = []
            if 'sma' in tech_analysis:
                sma_data = tech_analysis['sma']
                for period, value in sma_data.items():
                    if isinstance(value, dict) and 'signal' in value:
                        signals.append(1 if value['signal'] == 'buy' else -1 if value['signal'] == 'sell' else 0)
                        weights.append(1)
            if 'macd' in tech_analysis and isinstance(tech_analysis['macd'], dict):
                macd_data = tech_analysis['macd']
                if 'signal' in macd_data:
                    signals.append(1 if macd_data['signal'] == 'buy' else -1 if macd_data['signal'] == 'sell' else 0)
                    weights.append(1.5)
            if 'rsi' in tech_analysis and isinstance(tech_analysis['rsi'], dict):
                rsi_data = tech_analysis['rsi']
                if 'value' in rsi_data:
                    rsi_value = rsi_data['value']
                    if rsi_value < 30:
                        signals.append(1)
                    elif rsi_value > 70:
                        signals.append(-1)
                    else:
                        signals.append(0)
                    weights.append(1.2)
            if 'bollinger' in tech_analysis and isinstance(tech_analysis['bollinger'], dict):
                bb_data = tech_analysis['bollinger']
                if 'signal' in bb_data:
                    signals.append(1 if bb_data['signal'] == 'buy' else -1 if bb_data['signal'] == 'sell' else 0)
                    weights.append(1.2)
            for indicator, data in tech_analysis.items():
                if indicator not in ['sma', 'macd', 'rsi', 'bollinger'] and isinstance(data, dict) and 'signal' in data:
                    signals.append(1 if data['signal'] == 'buy' else -1 if data['signal'] == 'sell' else 0)
                    weights.append(1)
            if signals and weights:
                weighted_sum = sum(s * w for s, w in zip(signals, weights))
                total_weight = sum(weights)
                score = weighted_sum / total_weight
                return max(min(score, 1), -1)
            return 0.0
        except Exception:
            return 0.0

    def _calculate_sentiment_score(self, news_sentiment: Optional[Dict[str, Any]]) -> float:
        if not news_sentiment:
            return 0.0
        try:
            sentiment_values = []
            weights = []
            if 'overall_sentiment' in news_sentiment:
                overall = news_sentiment['overall_sentiment']
                if isinstance(overall, (int, float)):
                    sentiment_values.append(overall * 2 - 1)
                    weights.append(2)
            if 'news_items' in news_sentiment and isinstance(news_sentiment['news_items'], list):
                for item in news_sentiment['news_items']:
                    if isinstance(item, dict) and 'sentiment' in item:
                        sentiment = item['sentiment']
                        recency = item.get('recency', 1.0)
                        norm_sentiment = sentiment * 2 - 1
                        sentiment_values.append(norm_sentiment)
                        weights.append(recency)
            if 'social_sentiment' in news_sentiment and isinstance(news_sentiment['social_sentiment'], (int, float)):
                social = news_sentiment['social_sentiment']
                sentiment_values.append(social * 2 - 1)
                weights.append(1.5)
            if sentiment_values and weights:
                weighted_sum = sum(s * w for s, w in zip(sentiment_values, weights))
                total_weight = sum(weights)
                score = weighted_sum / total_weight
                return max(min(score, 1), -1)
            return 0.0
        except Exception:
            return 0.0

    def _calculate_market_score(self, market_condition: Optional[Dict[str, Any]]) -> float:
        if not market_condition:
            return 0.0
        try:
            signals = []
            weights = []
            if 'trend' in market_condition:
                trend = market_condition['trend']
                if isinstance(trend, str):
                    signals.append(1 if trend.lower() == 'bullish' else -1 if trend.lower() == 'bearish' else 0)
                    weights.append(2)
                elif isinstance(trend, (int, float)):
                    signals.append(trend)
                    weights.append(2)
            if 'sector_performance' in market_condition:
                sector = market_condition['sector_performance']
                if isinstance(sector, (int, float)):
                    sector_score = min(max(sector, -1), 1)
                    signals.append(sector_score)
                    weights.append(1.5)
            if 'vix' in market_condition:
                vix = market_condition['vix']
                if isinstance(vix, (int, float)):
                    if vix > 30:
                        vix_score = -0.8
                    elif vix > 20:
                        vix_score = -0.3
                    elif vix < 15:
                        vix_score = 0.5
                    else:
                        vix_score = 0
                    signals.append(vix_score)
                    weights.append(1)
            for indicator, value in market_condition.items():
                if indicator not in ['trend', 'sector_performance', 'vix']:
                    if isinstance(value, dict) and 'signal' in value:
                        signals.append(1 if value['signal'] == 'buy' else -1 if value['signal'] == 'sell' else 0)
                        weights.append(1)
                    elif isinstance(value, (int, float)) and -1 <= value <= 1:
                        signals.append(value)
                        weights.append(1)
            if signals and weights:
                weighted_sum = sum(s * w for s, w in zip(signals, weights))
                total_weight = sum(weights)
                score = weighted_sum / total_weight
                return max(min(score, 1), -1)
            return 0.0
        except Exception:
            return 0.0

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        try:
            if 'close' not in data.columns:
                return 0.05
            daily_returns = data['close'].pct_change().dropna()
            if len(daily_returns) < 5:
                return 0.05
            volatility = daily_returns.std() * np.sqrt(252)
            volatility = max(min(volatility, 0.5), 0.01)
            return volatility
        except Exception:
            return 0.05

    def batch_predict(self, symbols: List[str], 
                     data_dict: Dict[str, pd.DataFrame],
                     tech_analysis_dict: Optional[Dict[str, Dict[str, Any]]] = None,
                     news_sentiment_dict: Optional[Dict[str, Dict[str, Any]]] = None,
                     market_condition: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Thực hiện dự đoán nâng cao cho nhiều cổ phiếu
        
        Args:
            symbols: Danh sách mã cổ phiếu
            data_dict: Dictionary của DataFrames với dữ liệu giá
            tech_analysis_dict: Dictionary kết quả phân tích kỹ thuật (optional)
            news_sentiment_dict: Dictionary sentiment từ tin tức (optional)
            market_condition: Điều kiện thị trường (optional)
            
        Returns:
            Dict với kết quả dự đoán cho mỗi symbol
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = data_dict.get(symbol)
                tech_analysis = tech_analysis_dict.get(symbol) if tech_analysis_dict else None
                news_sentiment = news_sentiment_dict.get(symbol) if news_sentiment_dict else None
                
                if data is not None:
                    result = self.predict(symbol, data, tech_analysis, news_sentiment, market_condition)
                    results[symbol] = result
                else:
                    results[symbol] = {"error": f"No data available for {symbol}"}
            except Exception as e:
                logger.error(f"Error in batch prediction for {symbol}: {str(e)}")
                results[symbol] = {"error": f"Prediction failed: {str(e)}"}
                
        # Ensure all results have the same structure
        for symbol, result in results.items():
            if "error" not in result:
                if "ml_scores" not in result:
                    result["ml_scores"] = {"short_term": 0, "medium_term": 0, "long_term": 0}
                if "technical_signals" not in result:
                    result["technical_signals"] = {}
                if "recommendation" not in result:
                    result["recommendation"] = {"action": "HOLD", "confidence": 0, "reasoning": "Default recommendation due to missing data"}
                
        return results 