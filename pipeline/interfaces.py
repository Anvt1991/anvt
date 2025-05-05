#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataclass chuẩn hóa cho Pipeline phân tích chứng khoán
Module này định nghĩa các dataclass để truyền dữ liệu giữa các bước trong pipeline.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np

# Timezone constant
TZ = datetime.now().astimezone().tzinfo

@dataclass
class StockData:
    """Stock price and volume data"""
    df: pd.DataFrame
    symbol: str
    start_date: datetime
    end_date: datetime
    timeframe: str = "daily"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    validation_details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering"""
    features: Dict[str, pd.Series] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_correlation: Optional[pd.DataFrame] = None
    rolling_features: Dict[str, Any] = field(default_factory=dict)
    volatility_features: Dict[str, Any] = field(default_factory=dict)
    momentum_features: Dict[str, Any] = field(default_factory=dict)
    extra_features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TechnicalAnalysisResult:
    """Result of technical analysis"""
    indicators: Dict[str, pd.Series] = field(default_factory=dict)
    signals: Dict[str, str] = field(default_factory=dict)
    patterns: Dict[str, str] = field(default_factory=dict)
    support_resistance: Dict[str, List[float]] = field(default_factory=dict)
    trend_analysis: Dict[str, str] = field(default_factory=dict)
    bollinger_bands: Dict[str, Any] = field(default_factory=dict)
    atr: Optional[float] = None
    volatility: Optional[float] = None
    obv: Optional[float] = None
    momentum: Optional[float] = None
    rolling: Dict[str, Any] = field(default_factory=dict)
    market: Dict[str, Any] = field(default_factory=dict)
    liquidity: Dict[str, Any] = field(default_factory=dict)
    abnormal_events: List[Any] = field(default_factory=list)
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class NewsItem:
    """Single news item"""
    title: str
    date: datetime
    source: str
    url: Optional[str] = None
    content: Optional[str] = None
    sentiment: Optional[float] = None
    relevance: Optional[float] = None

@dataclass
class NewsResult:
    """Result of news analysis"""
    news_items: List[NewsItem] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    sentiment_detail: Dict[str, Any] = field(default_factory=dict)
    abnormal_events: List[Any] = field(default_factory=list)
    liquidity: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AIAnalysisResult:
    """Result of AI analysis"""
    sentiment: str  # bullish, bearish, neutral
    confidence: float
    insights: List[str] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[str] = None
    timestamp: Optional[datetime] = None
    prediction_detail: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PredictionResult:
    """Result of price prediction"""
    prediction: str  # bullish, bearish, neutral
    confidence: float
    prediction_horizon: str  # short-term, medium-term, long-term
    target_price: Optional[float] = None
    direction: str = "neutral"  # up, down, neutral
    details: Dict[str, Any] = field(default_factory=dict)
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class EnhancedPredictionResult:
    """Result of enhanced prediction with multiple inputs"""
    prediction: str  # bullish, bearish, neutral
    confidence: float
    prediction_horizon: str  # short-term, medium-term, long-term
    contributing_factors: Dict[str, Any] = field(default_factory=dict)
    reasoning: Optional[str] = None
    target_price: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    version: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class PortfolioResult:
    """Result of portfolio optimization"""
    position_type: str  # buy, sell, hold
    position_size: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    expected_return: Optional[float] = None
    max_drawdown: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class BacktestResult:
    """Result of strategy backtesting"""
    total_return: float = 0.0
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    profit_factor: Optional[float] = None
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: Optional[str] = None
    timestamp: Optional[datetime] = None

@dataclass
class ReportResult:
    """Result of report generation"""
    summary: str
    date_generated: datetime = field(default_factory=datetime.now)
    sections: Dict[str, str] = field(default_factory=dict)
    charts: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class PipelineResult:
    """Combined result of all pipeline stages"""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    validation_result: Optional[ValidationResult] = None
    feature_engineering_result: Optional[FeatureEngineeringResult] = None
    technical_analysis_result: Optional[TechnicalAnalysisResult] = None
    news_result: Optional[NewsResult] = None
    ai_analysis_result: Optional[AIAnalysisResult] = None
    prediction_result: Optional[PredictionResult] = None
    enhanced_prediction_result: Optional[EnhancedPredictionResult] = None
    portfolio_result: Optional[PortfolioResult] = None
    backtest_result: Optional[BacktestResult] = None
    report_result: Optional[ReportResult] = None
    execution_time: float = 0.0
    error: Optional[str] = None
    warning_messages: List[str] = field(default_factory=list)
    
    @classmethod
    def from_pipeline_data(cls, pipeline_data):
        """Create a PipelineResult from PipelineData"""
        return cls(
            symbol=pipeline_data.symbol,
            validation_result=pipeline_data.validation_result,
            feature_engineering_result=pipeline_data.feature_engineering_result,
            technical_analysis_result=pipeline_data.technical_analysis_result,
            news_result=pipeline_data.news_result,
            ai_analysis_result=pipeline_data.ai_analysis_result,
            prediction_result=pipeline_data.prediction_result,
            enhanced_prediction_result=pipeline_data.enhanced_prediction_result,
            portfolio_result=pipeline_data.portfolio_result,
            backtest_result=pipeline_data.backtest_result,
            report_result=pipeline_data.report_result,
            execution_time=pipeline_data.execution_times.get('total', 0.0),
            error=pipeline_data.error,
            warning_messages=pipeline_data.warnings
        )

@dataclass
class PipelineData:
    """Data container for pipeline processing"""
    symbol: str
    stock_data: Optional[StockData] = None
    validation_result: Optional[ValidationResult] = None
    feature_engineering_result: Optional[FeatureEngineeringResult] = None
    technical_analysis_result: Optional[TechnicalAnalysisResult] = None
    news_result: Optional[NewsResult] = None
    ai_analysis_result: Optional[AIAnalysisResult] = None
    prediction_result: Optional[PredictionResult] = None
    enhanced_prediction_result: Optional[EnhancedPredictionResult] = None
    portfolio_result: Optional[PortfolioResult] = None
    backtest_result: Optional[BacktestResult] = None
    report_result: Optional[ReportResult] = None
    pipeline_result: Optional[PipelineResult] = None
    
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    execution_times: Dict[str, float] = field(default_factory=dict)
    
    error: Optional[str] = None
    has_error: bool = False
    warnings: List[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProphetForecastResult:
    """Kết quả dự báo Prophet chuẩn hóa cho pipeline"""
    forecast_df: pd.DataFrame
    model_type: str = "prophet"
    version: str = "1.0"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

# Converters để chuyển đổi giữa dict và dataclass
def dict_to_dataclass(data_dict: Dict[str, Any], dataclass_type):
    """Chuyển đổi từ dict sang dataclass"""
    if dataclass_type is StockData and 'data' in data_dict and isinstance(data_dict['data'], pd.DataFrame):
        return StockData(**data_dict)
    
    # Lọc các key không có trong dataclass
    valid_fields = {f.name for f in dataclass_type.__dataclass_fields__}
    filtered_dict = {k: v for k, v in data_dict.items() if k in valid_fields}
    
    return dataclass_type(**filtered_dict)

def dataclass_to_dict(dataclass_obj) -> Dict[str, Any]:
    """Chuyển đổi từ dataclass sang dict"""
    result = {}
    for field in dataclass_obj.__dataclass_fields__:
        value = getattr(dataclass_obj, field)
        if isinstance(value, pd.DataFrame):
            # Bỏ qua DataFrame, chỉ lưu metadata
            continue
        elif hasattr(value, '__dataclass_fields__'):
            # Chuyển đổi đệ quy các dataclass lồng nhau
            result[field] = dataclass_to_dict(value)
        elif isinstance(value, (datetime, np.ndarray)):
            # Xử lý các kiểu dữ liệu đặc biệt
            continue
        else:
            result[field] = value
    return result

# Các interface mà các module phải tuân theo
class DataLoaderInterface:
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Tải dữ liệu chứng khoán"""
        raise NotImplementedError

class DataValidatorInterface:
    def validate_and_clean_dataframe(self, df: pd.DataFrame) -> tuple:
        """Xác thực và làm sạch dữ liệu"""
        raise NotImplementedError
    
    def detect_and_handle_outliers(self, df: pd.DataFrame) -> tuple:
        """Phát hiện và xử lý outliers"""
        raise NotImplementedError
    
    def normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Chuẩn hóa DataFrame"""
        raise NotImplementedError

class FeatureEngineeringInterface:
    def prepare_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Tạo đặc trưng cho mô hình"""
        raise NotImplementedError
    
    def get_feature_names(self) -> List[str]:
        """Lấy danh sách tên đặc trưng"""
        raise NotImplementedError

class TechnicalAnalyzerInterface:
    def analyze(self, df: pd.DataFrame, period: str = 'short') -> Dict[str, Any]:
        """Phân tích kỹ thuật"""
        raise NotImplementedError
    
    def calculate_adx(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tính chỉ số ADX"""
        raise NotImplementedError
    
    def calculate_bollinger_bands(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tính Bollinger Bands"""
        raise NotImplementedError
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Tính các mức Fibonacci"""
        raise NotImplementedError

class NewsLoaderInterface:
    def get_news(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """Lấy tin tức liên quan đến mã chứng khoán"""
        raise NotImplementedError
    
    def analyze_sentiment(self, news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phân tích sentiment của tin tức"""
        raise NotImplementedError

class AIAnalyzerInterface:
    def predict(self, symbol: str, data: pd.DataFrame, 
               technical_indicators: Dict[str, Any] = None, 
               news_sentiment: Dict[str, Any] = None) -> Dict[str, Any]:
        """Dự đoán xu hướng giá"""
        raise NotImplementedError

class EnhancedPredictorInterface:
    def predict(self, symbol: str, data: pd.DataFrame, 
               tech_analysis: Dict[str, Any] = None,
               news_sentiment: Dict[str, Any] = None,
               market_condition: Dict[str, Any] = None) -> Dict[str, Any]:
        """Dự đoán nâng cao"""
        raise NotImplementedError

class BacktesterInterface:
    def run(self, data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Chạy backtest"""
        raise NotImplementedError

class ReportGeneratorInterface:
    def generate_all_reports(self, analysis_data: Dict[str, Any], report_text: str) -> Dict[str, str]:
        """Tạo tất cả các báo cáo"""
        raise NotImplementedError 