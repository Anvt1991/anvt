import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps
from typing import List, Dict, Any, Callable, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from pipeline.interfaces import (
    StockData,
    ValidationResult,
    FeatureEngineeringResult,
    TechnicalAnalysisResult,
    NewsResult,
    AIAnalysisResult,
    PredictionResult,
    EnhancedPredictionResult,
    PortfolioResult,
    BacktestResult,
    ReportResult,
    PipelineData,
    PipelineResult
)
from core.data.data import DataLoader
from core.data.data_validator import DataValidator
from core.model.feature_generator import FeatureGenerator
from core.technical import TechnicalAnalyzer
from core.news.news import NewsLoader
from core.model.enhanced_predictor import EnhancedPredictor
from core.strategy.backtester import Backtester
from core.data.db import DBManager
from core.ai.groq import GroqHandler
from core.ai.gemini import GeminiHandler
from core.model.prophet_model import ProphetModel
from pipeline.utils import ensure_valid_data
from core.report.manager import ReportManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PipelineProcessor')

def handle_errors(func):
    """Decorator to handle errors and set error flag in pipeline data"""
    @wraps(func)
    def wrapper(self, pipeline_data: PipelineData, *args, **kwargs):
        if getattr(pipeline_data, 'has_error', False):
            logger.warning(f"Skipping {func.__name__} due to previous errors")
            return pipeline_data
        
        try:
            return func(self, pipeline_data, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            pipeline_data.has_error = True
            pipeline_data.error = f"Error in {func.__name__}: {str(e)}"
            return pipeline_data
    
    return wrapper

def time_execution(func):
    """Decorator to measure execution time"""
    @wraps(func)
    def wrapper(self, pipeline_data: PipelineData, *args, **kwargs):
        start_time = time.time()
        result = func(self, pipeline_data, *args, **kwargs)
        exec_time = time.time() - start_time
        
        # Store execution time in pipeline data
        if hasattr(result, 'execution_times'):
            result.execution_times[func.__name__] = exec_time
            
        logger.info(f"{func.__name__} executed in {exec_time:.2f} seconds")
        return result
    
    return wrapper

def to_primitive(obj):
    """Recursively convert pd.Series, pd.DataFrame, np types, Timestamp, etc. to primitive types for JSON serialization."""
    if isinstance(obj, pd.DataFrame):
        # Convert index to string to avoid data loss with datetime/tz
        try:
            if not obj.empty:
                logger.debug(f"[to_primitive] DataFrame index min: {obj.index.min()}, max: {obj.index.max()}, len: {len(obj)}")
            obj = obj.copy()
            obj.index = obj.index.astype(str)
        except Exception as e:
            # fallback: just convert
            pass
        return obj.rename_axis(None).to_dict(orient="index")
    elif isinstance(obj, pd.Series):
        return {str(k): to_primitive(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.ndarray, list, tuple)):
        return [to_primitive(x) for x in obj]
    elif isinstance(obj, dict):
        return {str(k): to_primitive(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif hasattr(obj, "__dict__"):
        return to_primitive(vars(obj))
    else:
        return obj

class PipelineProcessor:
    """
    Main processor class for the stock analysis and prediction pipeline.
    Orchestrates the execution of various components in the pipeline.
    """
    
    def __init__(self):
        """Initialize the pipeline processor with empty component lists"""
        self.data_validators = []
        self.feature_engineers = []
        self.technical_analyzers = []
        self.prediction_generators = []
        self.news_analyzers = []
        self.prediction_enhancers = []
        self.portfolio_optimizers = []
        self.backtesters = []
        self.report_generators = []
        self.database_savers = []
        
        # Configuration
        self.config = {
            'parallel_execution': False,
            'max_workers': 4,
            'error_tolerance': 'strict',  # 'strict' or 'lenient'
            'log_level': 'INFO'
        }
        
        # Set logger level based on config
        logger.setLevel(getattr(logging, self.config['log_level']))
        
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.feature_engineer = FeatureGenerator()
        self.technical_analyzer = TechnicalAnalyzer()
        self.news_loader = NewsLoader()
        self.enhanced_predictor = EnhancedPredictor()
        self.backtester = Backtester()
        self.db = DBManager()
        self.groq_handler = GroqHandler()
        self.gemini_handler = GeminiHandler()
        self.prophet_model = ProphetModel()
    
    def register_data_validator(self, validator_func: Callable[[StockData], ValidationResult]):
        """Register a data validation function"""
        self.data_validators.append(validator_func)
        logger.debug(f"Registered data validator: {validator_func.__name__}")
        return self
    
    def register_feature_engineer(self, engineer_func: Callable[[StockData], FeatureEngineeringResult]):
        """Register a feature engineering function"""
        self.feature_engineers.append(engineer_func)
        logger.debug(f"Registered feature engineer: {engineer_func.__name__}")
        return self
    
    def register_technical_analyzer(self, analyzer_func: Callable[[StockData, Optional[FeatureEngineeringResult]], TechnicalAnalysisResult]):
        """Register a technical analysis function"""
        self.technical_analyzers.append(analyzer_func)
        logger.debug(f"Registered technical analyzer: {analyzer_func.__name__}")
        return self
    
    def register_prediction_generator(self, predictor_func: Callable[[StockData, TechnicalAnalysisResult], PredictionResult]):
        """Register a prediction generator function"""
        self.prediction_generators.append(predictor_func)
        logger.debug(f"Registered prediction generator: {predictor_func.__name__}")
        return self

    def register_news_analyzer(self, analyzer_func: Callable[[str, Optional[pd.DataFrame]], NewsResult]):
        """Register a news analysis function"""
        self.news_analyzers.append(analyzer_func)
        logger.debug(f"Registered news analyzer: {analyzer_func.__name__}")
        return self
    
    def register_prediction_enhancer(self, enhancer_func: Callable[[PredictionResult, AIAnalysisResult, NewsResult], EnhancedPredictionResult]):
        """Register a prediction enhancement function"""
        self.prediction_enhancers.append(enhancer_func)
        logger.debug(f"Registered prediction enhancer: {enhancer_func.__name__}")
        return self
    
    def register_portfolio_optimizer(self, optimizer_func: Callable[[EnhancedPredictionResult, TechnicalAnalysisResult], PortfolioResult]):
        """Register a portfolio optimization function"""
        self.portfolio_optimizers.append(optimizer_func)
        logger.debug(f"Registered portfolio optimizer: {optimizer_func.__name__}")
        return self
    
    def register_backtester(self, backtester_func: Callable[[StockData, PortfolioResult], BacktestResult]):
        """Register a backtesting function"""
        self.backtesters.append(backtester_func)
        logger.debug(f"Registered backtester: {backtester_func.__name__}")
        return self
    
    def register_report_generator(self, report_func: Callable[[PipelineData], ReportResult]):
        """Register a report generation function"""
        self.report_generators.append(report_func)
        logger.debug(f"Registered report generator: {report_func.__name__}")
        return self
    
    def register_database_saver(self, saver_func: Callable[[PipelineResult], bool]):
        """Register a database saving function"""
        self.database_savers.append(saver_func)
        logger.debug(f"Registered database saver: {saver_func.__name__}")
        return self
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update pipeline configuration"""
        self.config.update(config_updates)
        
        # Update logger level if log_level is changed
        if 'log_level' in config_updates:
            logger.setLevel(getattr(logging, self.config['log_level']))
            
        logger.info(f"Updated pipeline configuration: {config_updates}")
        return self
    
    @time_execution
    @handle_errors
    def _pipeline_data_loader(self, pipeline_data: PipelineData) -> PipelineData:
        """Load stock data for the given symbol"""
        symbol = pipeline_data.symbol
        try:
            df = self.data_loader.load_stock_data(symbol)
            if df is None or df.empty:
                pipeline_data.has_error = True
                pipeline_data.error = f"Không thể tải dữ liệu cho {symbol}"
                return pipeline_data
            # Tạo StockData (giả sử các trường phù hợp với interfaces.py)
            stock_data = StockData(
                symbol=symbol,
                df=df,
                start_date=df.index[0] if not df.empty else None,
                end_date=df.index[-1] if not df.empty else None,
                timeframe="daily",
                metadata={}
            )
            pipeline_data.stock_data = stock_data
            # Đảm bảo result có symbol
            if not hasattr(pipeline_data, 'result'):
                pipeline_data.result = {}
            pipeline_data.result["symbol"] = symbol
        except Exception as e:
            pipeline_data.has_error = True
            pipeline_data.error = f"Lỗi khi tải dữ liệu: {str(e)}"
            if not hasattr(pipeline_data, 'result'):
                pipeline_data.result = {}
            pipeline_data.result["symbol"] = symbol
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_data_validation(self, pipeline_data: PipelineData) -> PipelineData:
        """Validate and clean stock data (gộp logic từ analyze.py)"""
        logger.info(f"Starting data validation for {pipeline_data.symbol}")
        symbol = pipeline_data.symbol
        if not pipeline_data.stock_data or not hasattr(pipeline_data.stock_data, 'df') or pipeline_data.stock_data.df is None:
            pipeline_data.has_error = True
            pipeline_data.error = "No stock data provided"
            if not hasattr(pipeline_data, 'result'):
                pipeline_data.result = {}
            pipeline_data.result["symbol"] = symbol
            return pipeline_data
        df = pipeline_data.stock_data.df
        # Bước 1: Chuẩn hóa dữ liệu
        clean_df = self.data_validator.normalize_dataframe(df)
        # Bước 2: Validate schema
        clean_df = self.data_validator.validate_schema(clean_df)
        validation_results = []  # Có thể bổ sung cảnh báo nếu validate_schema trả về warning
        # Bước 3: Xử lý outliers (chỉ cảnh báo, không loại bỏ)
        clean_df, outlier_report = self.data_validator.detect_and_handle_outliers(clean_df)
        outliers = []  # Không loại bỏ, chỉ cảnh báo
        # Tạo ValidationResult
        validation_result = ValidationResult(
            is_valid=True,
            error_message=None,
            warnings=validation_results if isinstance(validation_results, list) else [],
            validation_details={},
        )
        # Cập nhật PipelineData
        pipeline_data.validation_result = validation_result
        # Cập nhật StockData với dữ liệu đã làm sạch
        pipeline_data.stock_data.df = clean_df
        # Cập nhật result (cho tương thích ngược)
        if not hasattr(pipeline_data, 'result'):
            pipeline_data.result = {}
        pipeline_data.result["data_validation"] = {
            "validation_results": validation_results,
            "outliers_detected": 0,  # Không loại bỏ outlier
            "outlier_report": outlier_report,
            "missing_values": df.isna().sum().sum(),
            "rows_before_cleaning": len(df),
            "rows_after_cleaning": len(clean_df)
        }
        pipeline_data.result["symbol"] = symbol
        logger.info(f"✓ Dữ liệu {symbol} đã được xác thực và làm sạch")
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_feature_engineering(self, pipeline_data: PipelineData) -> PipelineData:
        """Tạo đặc trưng (features) cho dữ liệu chứng khoán (gộp logic từ analyze.py)"""
        logger.info(f"Starting feature engineering for {pipeline_data.symbol}")
        symbol = pipeline_data.symbol
        df = pipeline_data.stock_data.df if pipeline_data.stock_data else None
        if df is None or df.empty:
            pipeline_data.has_error = True
            pipeline_data.error = "No data for feature engineering"
            return pipeline_data
        # CHUẨN HÓA DỮ LIỆU ĐẦU VÀO
        df = DataValidator.normalize_dataframe(df)
        df = DataValidator.validate_schema(df)
        # Tạo đặc trưng
        features_df = self.feature_engineer.prepare_features(df)
        # Lấy danh sách tên đặc trưng trực tiếp từ DataFrame
        feature_names = list(features_df.columns) if features_df is not None else []
        # Tạo FeatureEngineeringResult
        feature_engineering_result = FeatureEngineeringResult(
            features=features_df.to_dict(orient='series') if hasattr(features_df, 'to_dict') else {},
            feature_importance={},
            feature_correlation=None,
            rolling_features={},
            volatility_features={},
            momentum_features={},
            extra_features={}
        )
        pipeline_data.feature_engineering_result = feature_engineering_result
        # Cập nhật result (cho tương thích ngược)
        if not hasattr(pipeline_data, 'result'):
            pipeline_data.result = {}
        pipeline_data.result["feature_engineering"] = {
            "features_count": features_df.shape[1] if features_df is not None else 0,
            "feature_names": feature_names
        }
        logger.info(f"✓ Đã tạo {features_df.shape[1] if features_df is not None else 0} đặc trưng cho {symbol}")
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_technical_analysis(self, pipeline_data: PipelineData) -> PipelineData:
        """Phân tích kỹ thuật (gộp logic từ analyze.py)"""
        logger.info(f"Starting technical analysis for {pipeline_data.symbol}")
        symbol = pipeline_data.symbol
        df = pipeline_data.stock_data.df if pipeline_data.stock_data else None
        if df is None or df.empty:
            pipeline_data.has_error = True
            pipeline_data.error = "No data for technical analysis"
            if not hasattr(pipeline_data, 'result'):
                pipeline_data.result = {}
            pipeline_data.result["symbol"] = symbol
            return pipeline_data
        period = getattr(pipeline_data, 'period', 'short')
        # CHUẨN HÓA DỮ LIỆU ĐẦU VÀO
        df = DataValidator.normalize_dataframe(df)
        df = DataValidator.validate_schema(df)
        try:
            # Phân tích kỹ thuật trả về dict chuẩn hóa
            ta_dict = self.technical_analyzer.analyze(df, period)
            pipeline_data.result = pipeline_data.result or {}
            pipeline_data.result["technical_analysis"] = ta_dict
            pipeline_data.result["symbol"] = symbol
            logger.info(f"✓ Phân tích kỹ thuật hoàn tất cho {symbol}")
        except Exception as e:
            pipeline_data.has_error = True
            pipeline_data.error = f"Lỗi khi thực hiện phân tích kỹ thuật: {str(e)}"
            pipeline_data.result = pipeline_data.result or {}
            pipeline_data.result["symbol"] = symbol
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_groq_analysis(self, pipeline_data: PipelineData) -> PipelineData:
        """Phân tích mẫu hình, nến, sóng, wyckoff bằng Groq AI"""
        logger.info(f"Starting Groq pattern analysis for {pipeline_data.symbol}")
        symbol = pipeline_data.symbol
        df = pipeline_data.stock_data.df if pipeline_data.stock_data else None
        if df is None or df.empty:
            pipeline_data.has_error = True
            pipeline_data.error = "No data for Groq pattern analysis"
            return pipeline_data
        groq_result = None
        try:
            # Chuẩn bị prompt cho phân tích mẫu hình
            system_prompt = "Bạn là chuyên gia phân tích kỹ thuật. Hãy phân tích mẫu hình, nến, sóng, Wyckoff cho mã chứng khoán được cung cấp."
            prompt_text = f"Mã: {symbol}\nDữ liệu:\n{df.tail(100).to_string()}"
            temperature = 0.3
            # Gọi GroqHandler để phân tích mẫu hình, nến, sóng, wyckoff
            groq_result = self.groq_handler.generate_content_sync(
                prompt=prompt_text,
                system_prompt=system_prompt,
                temperature=temperature
            )
        except Exception as e:
            logger.warning(f"Groq pattern analysis exception: {str(e)}")
            groq_result = None
        # Kiểm tra kết quả lỗi hoặc quota
        fallback_needed = False
        if groq_result is None:
            fallback_needed = True
        elif isinstance(groq_result, str):
            error_str = groq_result.lower()
            if any(x in error_str for x in ["quota", "429", "error", "limit", "exceeded", "unavailable"]):
                fallback_needed = True
        if fallback_needed:
            logger.warning("Groq trả về lỗi quota hoặc lỗi khác, fallback sang OpenRouter.")
            try:
                from core.ai.openrouter import OpenRouterHandler
                openrouter = OpenRouterHandler()
                messages = [
                    {"role": "system", "content": "Bạn là chuyên gia phân tích kỹ thuật. Hãy phân tích mẫu hình, nến, sóng, Wyckoff cho mã chứng khoán sau."},
                    {"role": "user", "content": f"Mã: {symbol}\nDữ liệu: {df.tail(100).to_string()}"}
                ]
                groq_result = openrouter.generate_response(messages)
                pipeline_data.groq_analysis_result = {"openrouter_pattern_report": groq_result}
                if not hasattr(pipeline_data, 'result'):
                    pipeline_data.result = {}
                pipeline_data.result["groq_analysis"] = {"openrouter_pattern_report": groq_result}
                logger.info(f"✓ OpenRouter pattern analysis completed for {symbol}")
            except Exception as oe:
                pipeline_data.has_error = True
                pipeline_data.error = f"Lỗi khi phân tích mẫu hình Groq và OpenRouter: {str(oe)}"
        else:
            pipeline_data.groq_analysis_result = groq_result
            if not hasattr(pipeline_data, 'result'):
                pipeline_data.result = {}
            pipeline_data.result["groq_analysis"] = groq_result
            logger.info(f"✓ Groq pattern analysis completed for {symbol}")
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_news_analysis(self, pipeline_data: PipelineData) -> PipelineData:
        """Phân tích tin tức (gộp logic từ analyze.py)"""
        logger.info(f"Starting news analysis for {pipeline_data.symbol}")
        symbol = pipeline_data.symbol
        # Tải và phân tích tin tức
        news = self.news_loader.get_news(symbol, days=7)
        news_sentiment = self.news_loader.analyze_sentiment(news)
        # Tính toán số lượng tin tức theo sentiment
        positive_count = sum(1 for n in news if n.get('sentiment', '') == 'positive')
        negative_count = sum(1 for n in news if n.get('sentiment', '') == 'negative')
        neutral_count = sum(1 for n in news if n.get('sentiment', '') == 'neutral')
        # Tạo NewsResult
        news_result = NewsResult(
            news_items=news,
            sentiment_score=news_sentiment.get('score') if isinstance(news_sentiment, dict) else None,
            impact_analysis=news_sentiment if isinstance(news_sentiment, dict) else {},
            sentiment_detail=news_sentiment if isinstance(news_sentiment, dict) else {},
            abnormal_events=[],
            liquidity={}
        )
        pipeline_data.news_result = news_result
        # Cập nhật result (cho tương thích ngược)
        if not hasattr(pipeline_data, 'result'):
            pipeline_data.result = {}
        pipeline_data.result["news"] = news
        pipeline_data.result["news_sentiment"] = news_sentiment
        pipeline_data.result["news_count"] = len(news) if news else 0
        logger.info(f"✓ Phân tích tin tức hoàn tất: {len(news) if news else 0} tin")
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_enhanced_prediction(self, pipeline_data: PipelineData) -> PipelineData:
        """Dự báo và tăng cường dự báo bằng EnhancedPredictor"""
        logger.info(f"Starting enhanced prediction for {pipeline_data.symbol}")
        symbol = pipeline_data.symbol
        # Bỏ qua bước ML cho VNINDEX và VN30
        if symbol.upper() in ["VNINDEX", "VN30"]:
            logger.info(f"Bỏ qua bước enhanced prediction (ML) cho {symbol}")
            return pipeline_data
        df = pipeline_data.stock_data.df if pipeline_data.stock_data else None
        tech = pipeline_data.result.get("technical_analysis", {}) if pipeline_data.result else None
        news = pipeline_data.result.get("news_sentiment") if pipeline_data.result else None
        market = None  # Có thể bổ sung nếu pipeline_data có trường này
        try:
            prediction_result = self.enhanced_predictor.predict(symbol, df, tech, news, market)
            pipeline_data.result = pipeline_data.result or {}
            pipeline_data.result["enhanced_prediction"] = prediction_result
            logger.info(f"✓ Enhanced prediction completed for {symbol}")
        except Exception as e:
            pipeline_data.has_error = True
            pipeline_data.error = f"Lỗi khi enhanced prediction: {str(e)}"
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_backtest(self, pipeline_data: PipelineData) -> PipelineData:
        """Backtest chiến lược đầu tư (gộp logic từ analyze.py)"""
        logger.info(f"Starting backtesting for {pipeline_data.symbol}")
        symbol = pipeline_data.symbol
        df = pipeline_data.stock_data.df if pipeline_data.stock_data else None
        if df is None or len(df) < 100:
            logger.info(f"Bỏ qua backtest do không đủ dữ liệu cho {symbol}")
            return pipeline_data
        try:
            # Tạo tín hiệu mô phỏng dựa trên phân tích kỹ thuật
            signals_df = pd.DataFrame(index=df.index)
            signals_df['signal'] = 0
            # Tạo tín hiệu mua/bán đơn giản dựa trên Golden Cross/Death Cross
            ta = pipeline_data.result.get("technical_analysis", {}) if pipeline_data.result else {}
            if 'signals' in ta and 'ma_cross' in ta['signals']:
                ma_cross = ta['signals']['ma_cross']
                if ma_cross == 'Golden Cross':
                    signals_df.iloc[-1, signals_df.columns.get_loc('signal')] = 1
                elif ma_cross == 'Death Cross':
                    signals_df.iloc[-1, signals_df.columns.get_loc('signal')] = -1
            # Tạo dữ liệu cho backtest
            stock_data = {symbol: df}
            signals = {symbol: signals_df}
            # Thực hiện backtest
            backtest_dict = self.backtester.run(stock_data, signals)
            # Chuyển đổi kết quả sang BacktestResult
            backtest_result = BacktestResult(
                total_return=backtest_dict.get("total_return", 0.0),
                sharpe_ratio=backtest_dict.get("performance_metrics", {}).get("sharpe_ratio"),
                max_drawdown=backtest_dict.get("performance_metrics", {}).get("max_drawdown"),
                win_rate=backtest_dict.get("performance_metrics", {}).get("win_rate"),
                profit_factor=backtest_dict.get("performance_metrics", {}).get("profit_factor"),
                trades=backtest_dict.get("trades", []),
                equity_curve=None,
                details=backtest_dict.get("performance_metrics", {}),
                metadata={},
                version=None,
                timestamp=None
            )
            pipeline_data.backtest_result = backtest_result
            # Cập nhật result (cho tương thích ngược)
            if not hasattr(pipeline_data, 'result'):
                pipeline_data.result = {}
            pipeline_data.result["backtest"] = {
                "total_return": backtest_result.total_return,
                "final_equity": backtest_dict.get("final_equity", 0.0),
                "performance_metrics": backtest_dict.get("performance_metrics", {}),
                "win_rate": backtest_result.win_rate,
                "profit_factor": backtest_result.profit_factor,
                "max_drawdown": backtest_result.max_drawdown
            }
            logger.info(f"✓ Backtest hoàn tất, tổng lợi nhuận: {backtest_result.total_return:.2f}%")
        except Exception as e:
            pipeline_data.has_error = True
            pipeline_data.error = f"Lỗi khi thực hiện backtest: {str(e)}"
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_prophet_forecast(self, pipeline_data: PipelineData, periods: int = 14) -> PipelineData:
        """Dự báo Prophet cho chỉ số (VNINDEX, VN30, v.v.)"""
        symbol = pipeline_data.symbol
        df = pipeline_data.stock_data.df if pipeline_data.stock_data else None
        if df is None or df.empty:
            pipeline_data.has_error = True
            pipeline_data.error = "No data for Prophet forecast"
            return pipeline_data
        try:
            forecast_result = self.prophet_model.forecast_pipeline_result(symbol, df, periods)
            pipeline_data.prophet_forecast = forecast_result
            if not hasattr(pipeline_data, 'result'):
                pipeline_data.result = {}
            pipeline_data.result["prophet_forecast"] = forecast_result
            logger.info(f"✓ Prophet forecast completed for {symbol}")
        except Exception as e:
            pipeline_data.has_error = True
            pipeline_data.error = f"Lỗi khi dự báo Prophet: {str(e)}"
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_report_generation(self, pipeline_data: PipelineData, aggregated_data: dict = None) -> PipelineData:
        logger.info(f"Starting report generation for {pipeline_data.symbol}")
        symbol = pipeline_data.symbol
        # Kiểm tra hợp lệ pipeline_data trước khi sinh báo cáo
        if not ensure_valid_data(pipeline_data):
            logger.error(f"Không thể tạo báo cáo cho {symbol}: Dữ liệu pipeline không hợp lệ hoặc có lỗi trước đó: {getattr(pipeline_data, 'error', '')}")
            pipeline_data.has_error = True
            pipeline_data.error = f"Không thể tạo báo cáo cho {symbol}: Dữ liệu pipeline không hợp lệ hoặc có lỗi trước đó."
            if not hasattr(pipeline_data, 'result'):
                pipeline_data.result = {}
            pipeline_data.result["report"] = None
            return pipeline_data
        try:
            merged_data = aggregated_data if aggregated_data is not None else self.aggregate_pipeline_result(pipeline_data)
            # Rút gọn log merged_data: chỉ log các trường chính
            raw_data = merged_data.get('raw_data', {})
            price_info = merged_data.get('price_info', {})
            ta = merged_data.get('technical_analysis', {})
            indicators = ta.get('indicators', {})
            last_date = None
            if isinstance(raw_data, dict) and raw_data:
                try:
                    last_date = max(raw_data.keys())
                except Exception:
                    last_date = None
            logger.info(f"[DEBUG] merged_data: keys={list(merged_data.keys())}, raw_data_rows={len(raw_data)}, last_date={last_date}, close={price_info.get('close')}, prev_close={price_info.get('previous_close')}, RSI={indicators.get('rsi') or indicators.get('rsi_14')}, MA20={indicators.get('ma_20')}, MA50={indicators.get('ma_50')}, MA200={indicators.get('ma_200')}, MACD={indicators.get('macd') or indicators.get('macd_value')}")
            # Lấy thông tin giá để truyền vào meta
            meta = {
                "close_today": price_info.get("close"),
                "close_yesterday": price_info.get("previous_close"),
                "timestamp": merged_data.get("timestamp"),
                "data": merged_data
            }
            report_manager = ReportManager()
            report_result = report_manager.create_and_send_report(merged_data, symbol, meta=meta)
            if not hasattr(pipeline_data, 'result'):
                pipeline_data.result = {}
            if report_result.get("success"):
                pipeline_data.result["report"] = report_result["report"]
                logger.info(f"✓ Tạo báo cáo hoàn tất (ReportManager)")
            else:
                pipeline_data.has_error = True
                pipeline_data.error = report_result.get("error", "Không thể tạo báo cáo")
                pipeline_data.result["report"] = None
        except Exception as e:
            pipeline_data.has_error = True
            pipeline_data.error = f"Lỗi khi tạo báo cáo: {str(e)}"
        return pipeline_data
    
    @time_execution
    @handle_errors
    def _pipeline_save_to_database(self, pipeline_data: PipelineData, aggregated_data: dict = None) -> PipelineData:
        # Đã tích hợp lưu/gửi báo cáo vào ReportManager, không cần xử lý ở đây nữa
        logger.info(f"Bỏ qua lưu/gửi báo cáo trong _pipeline_save_to_database (đã tích hợp ReportManager)")
        return pipeline_data
    
    @time_execution
    def process(self, symbol: str, stock_data: Optional[StockData] = None, mode: str = 'predict') -> PipelineResult:
        logger.info(f"Starting pipeline processing for {symbol} (mode={mode})")
        pipeline_data = PipelineData(symbol=symbol, stock_data=stock_data)
        pipeline_data.execution_start = datetime.now()
        try:
            # 1. DataLoader
            pipeline_data = self._pipeline_data_loader(pipeline_data)
            if pipeline_data.has_error and self.config['error_tolerance'] == 'strict':
                logger.error(f"Pipeline processing stopped due to data loading error: {pipeline_data.error}")
                return PipelineResult.from_pipeline_data(pipeline_data)
            # 2. DataValidator
            pipeline_data = self._pipeline_data_validation(pipeline_data)
            if pipeline_data.has_error and self.config['error_tolerance'] == 'strict':
                logger.error(f"Pipeline processing stopped due to data validation error: {pipeline_data.error}")
                return PipelineResult.from_pipeline_data(pipeline_data)

            # Prophet forecast cho chỉ số
            if symbol.upper() in ["VNINDEX", "VN30"]:
                pipeline_data = self._pipeline_prophet_forecast(pipeline_data, periods=14)
                if pipeline_data.has_error and self.config['error_tolerance'] == 'strict':
                    logger.error(f"Pipeline processing stopped due to Prophet forecast error: {pipeline_data.error}")
                    return PipelineResult.from_pipeline_data(pipeline_data)

            # 3. Chạy song song các bước độc lập: TechnicalAnalyzer, NewsLoader, Groq, FeatureEngineer
            from concurrent.futures import ThreadPoolExecutor, as_completed
            parallel_steps = [
                (self._pipeline_technical_analysis, "technical_analysis_result"),
                (self._pipeline_news_analysis, "news_result"),
                (self._pipeline_groq_analysis, "groq_analysis_result"),
                (self._pipeline_feature_engineering, "feature_engineering_result"),
            ]
            results_map = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_step = {executor.submit(func, pipeline_data): (func, attr) for func, attr in parallel_steps}
                for future in as_completed(future_to_step):
                    func, attr = future_to_step[future]
                    try:
                        result = future.result()
                        # Cập nhật pipeline_data với kết quả từng bước
                        setattr(pipeline_data, attr, getattr(result, attr, None))
                        # Nếu có .result dict thì merge vào pipeline_data.result
                        if hasattr(result, 'result') and isinstance(result.result, dict):
                            if not hasattr(pipeline_data, 'result'):
                                pipeline_data.result = {}
                            pipeline_data.result.update(result.result)
                        # Nếu có lỗi, dừng pipeline nếu strict
                        if getattr(result, 'has_error', False) and self.config['error_tolerance'] == 'strict':
                            logger.error(f"Pipeline processing stopped due to error in {func.__name__}: {getattr(result, 'error', '')}")
                            return PipelineResult.from_pipeline_data(result)
                    except Exception as e:
                        logger.error(f"Error in parallel step {func.__name__}: {str(e)}", exc_info=True)
                        pipeline_data.has_error = True
                        pipeline_data.error = f"Error in {func.__name__}: {str(e)}"
                        if self.config['error_tolerance'] == 'strict':
                            return PipelineResult.from_pipeline_data(pipeline_data)

            # 4. EnhancedPredictor
            pipeline_data = self._pipeline_enhanced_prediction(pipeline_data)
            if pipeline_data.has_error and self.config['error_tolerance'] == 'strict':
                logger.error(f"Pipeline processing stopped due to enhanced prediction error: {pipeline_data.error}")
                return PipelineResult.from_pipeline_data(pipeline_data)
            # 5. Backtester
            pipeline_data = self._pipeline_backtest(pipeline_data)
            if pipeline_data.has_error and self.config['error_tolerance'] == 'strict':
                logger.error(f"Pipeline processing stopped due to backtesting error: {pipeline_data.error}")
                return PipelineResult.from_pipeline_data(pipeline_data)
            # 7. Tổng hợp các trường dữ liệu
            aggregated_data = self.aggregate_pipeline_result(pipeline_data)
            # 8. Gemini (sinh báo cáo)
            pipeline_data = self._pipeline_report_generation(pipeline_data, aggregated_data=aggregated_data)
            if pipeline_data.has_error and self.config['error_tolerance'] == 'strict':
                logger.error(f"Pipeline processing stopped due to report generation error: {pipeline_data.error}")
                return PipelineResult.from_pipeline_data(pipeline_data)
            # 9. DBManager (lưu + gửi telegram)
            pipeline_data = self._pipeline_save_to_database(pipeline_data, aggregated_data=aggregated_data)
        except Exception as e:
            pipeline_data.has_error = True
            pipeline_data.error = f"Unhandled error in pipeline processing: {str(e)}"
            logger.error(pipeline_data.error, exc_info=True)
        pipeline_data.execution_end = datetime.now()
        total_time = (pipeline_data.execution_end - pipeline_data.execution_start).total_seconds()
        pipeline_data.execution_times['total'] = total_time
        logger.info(f"Pipeline processing completed for {symbol} in {total_time:.2f} seconds")
        return PipelineResult.from_pipeline_data(pipeline_data)

    def parallel_process(self, symbols: List[str]) -> Dict[str, PipelineResult]:
        """
        Process multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols to process
            
        Returns:
            Dictionary mapping symbols to their pipeline results
        """
        if not self.config['parallel_execution']:
            logger.info("Parallel execution is disabled. Processing sequentially.")
            results = {}
            for symbol in symbols:
                results[symbol] = self.process(symbol)
            return results
        
        logger.info(f"Processing {len(symbols)} symbols in parallel with {self.config['max_workers']} workers")
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.config['max_workers']) as executor:
            # Submit all processing tasks
            future_to_symbol = {executor.submit(self.process, symbol): symbol for symbol in symbols}
            
            # Collect results as they complete
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
                    # Create an error result
                    results[symbol] = PipelineResult(
                        symbol=symbol,
                        error=str(e)
                    )
        
        logger.info(f"Completed parallel processing of {len(symbols)} symbols")
        return results 

    def aggregate_pipeline_result(self, pipeline_data: PipelineData) -> dict:
        """
        Tổng hợp tất cả kết quả từ pipeline_data thành một dict duy nhất.
        Ưu tiên lấy từ các thuộc tính chuẩn hóa, fallback sang pipeline_data.result nếu cần.
        ĐẢM BẢO CHUẨN HÓA: Chỉ truyền indicators duy nhất ở cấp 1, luôn là giá trị mới nhất.
        """
        merged_data = {}
        # 1. Thông tin cơ bản
        merged_data["symbol"] = pipeline_data.symbol
        # Đảm bảo timestamp luôn hợp lệ, tránh lỗi NoneType
        exec_end = getattr(pipeline_data, "execution_end", None)
        if exec_end is None:
            exec_end = datetime.now()
        merged_data["timestamp"] = exec_end.isoformat()
        # 2. Dữ liệu gốc
        if hasattr(pipeline_data, "stock_data") and pipeline_data.stock_data is not None:
            merged_data["raw_data"] = to_primitive(pipeline_data.stock_data.df) if hasattr(pipeline_data.stock_data, "df") else {}
        # 3. Kết quả từng bước
        for field in [
            "validation_result", "feature_engineering_result", "technical_analysis_result",
            "news_result", "ai_analysis_result", "prediction_result", "enhanced_prediction_result",
            "portfolio_result", "backtest_result", "report_result"
        ]:
            value = getattr(pipeline_data, field, None)
            if value is not None:
                merged_data[field] = to_primitive(value)
        # 4. Các trường tổng hợp từ pipeline_data.result (nếu có)
        if hasattr(pipeline_data, "result") and isinstance(pipeline_data.result, dict):
            merged_data.update(to_primitive(pipeline_data.result))
        # 5. Đảm bảo các trường tối thiểu luôn có (dù là dict rỗng)
        for key in ["technical_analysis", "news", "ml_scores", "portfolio", "backtest"]:
            if key not in merged_data:
                merged_data[key] = {}
        # 6. Thông tin giá & các trường quan trọng từ stock_data.df
        current_price = price_change = price_change_percent = None
        close_today = close_yesterday = None
        if hasattr(pipeline_data, "stock_data") and pipeline_data.stock_data and hasattr(pipeline_data.stock_data, "df"):
            df = pipeline_data.stock_data.df
            if not df.empty and "close" in df.columns:
                close_today = float(df["close"].iloc[-1])
                current_price = close_today
                if len(df) > 1:
                    close_yesterday = float(df["close"].iloc[-2])
                    price_change = close_today - close_yesterday
                    price_change_percent = (price_change / close_yesterday) * 100 if close_yesterday != 0 else 0
        merged_data["price_info"] = {
            "close": close_today,
            "previous_close": close_yesterday
        }
        merged_data["current_price"] = current_price
        merged_data["price_change"] = price_change
        merged_data["price_change_percent"] = price_change_percent
        # Thêm kết quả Prophet forecast nếu có
        if hasattr(pipeline_data, "prophet_forecast") and pipeline_data.prophet_forecast is not None:
            merged_data["prophet_forecast"] = to_primitive(pipeline_data.prophet_forecast)
        # === CHUẨN HÓA INDICATORS DUY NHẤT ===
        # Lấy indicators mới nhất từ technical_analysis hoặc tự tính lại nếu thiếu
        indicators = None
        # Ưu tiên lấy từ technical_analysis
        if "technical_analysis" in merged_data and isinstance(merged_data["technical_analysis"], dict):
            indicators = merged_data["technical_analysis"].get("indicators")
        # Nếu chưa có, tự động tính lại từ stock_data.df
        if not indicators and hasattr(pipeline_data, "stock_data") and pipeline_data.stock_data and hasattr(pipeline_data.stock_data, "df"):
            df = pipeline_data.stock_data.df
            if df is not None and not df.empty:
                try:
                    from core.technical import TechnicalAnalyzer
                    indicators = TechnicalAnalyzer.get_technical_indicators(df)
                except Exception as e:
                    indicators = {k: None for k in ["rsi_14", "ma_20", "ma_50", "ma_200", "macd", "macd_signal", "macd_hist"]}
            else:
                indicators = {k: None for k in ["rsi_14", "ma_20", "ma_50", "ma_200", "macd", "macd_signal", "macd_hist"]}
        merged_data["indicators"] = indicators
        # Đồng bộ lại các nhánh phụ: xóa indicators ở technical_analysis, technical_analysis_result nếu có
        if "technical_analysis" in merged_data and isinstance(merged_data["technical_analysis"], dict):
            merged_data["technical_analysis"]["indicators"] = None
        if "technical_analysis_result" in merged_data and isinstance(merged_data["technical_analysis_result"], dict):
            merged_data["technical_analysis_result"]["indicators"] = None
        # Bổ sung trường 'vnindex_last' và các trường tổng hợp nếu là VNINDEX
        if pipeline_data.symbol.upper() == "VNINDEX":
            close_today = close_yesterday = None
            volume = volume_avg_10 = volume_avg_20 = volume_change_pct = None
            df = None
            if hasattr(pipeline_data, "stock_data") and pipeline_data.stock_data and hasattr(pipeline_data.stock_data, "df"):
                df = pipeline_data.stock_data.df
                if not df.empty and "close" in df.columns:
                    close_today = float(df["close"].iloc[-1])
                    if len(df) > 1:
                        close_yesterday = float(df["close"].iloc[-2])
                if not df.empty and "volume" in df.columns:
                    volume = float(df["volume"].iloc[-1])
                    if len(df) >= 10:
                        volume_avg_10 = float(df["volume"].iloc[-10:].mean())
                    if len(df) >= 20:
                        volume_avg_20 = float(df["volume"].iloc[-20:].mean())
                    if volume_avg_10 and volume_avg_10 > 0:
                        volume_change_pct = (volume - volume_avg_10) / volume_avg_10 * 100
            # Tính thay đổi điểm số và %
            vnindex_change = close_today - close_yesterday if close_today is not None and close_yesterday is not None else 0
            vnindex_change_pct = ((close_today / close_yesterday - 1) * 100) if close_today and close_yesterday else 0
            # Lấy sentiment/trend nếu có
            market_sentiment = merged_data.get("news_sentiment", {}).get("overall", "N/A")
            market_trend = merged_data.get("technical_analysis", {}).get("trend", "N/A")
            # Các trường khác (nếu có)
            support = merged_data.get("technical_analysis", {}).get("support", [])
            resistance = merged_data.get("technical_analysis", {}).get("resistance", [])
            top_influencers = merged_data.get("top_influencers", [])
            group_leaders = merged_data.get("group_leaders", {})
            market_breadth = merged_data.get("market_breadth", {})
            # Gán vào merged_data
            merged_data.update({
                "vnindex_last": close_today,
                "vnindex_change": vnindex_change,
                "vnindex_change_pct": vnindex_change_pct,
                "volume": volume if volume is not None else "N/A",
                "volume_avg_10": volume_avg_10 if volume_avg_10 is not None else "N/A",
                "volume_avg_20": volume_avg_20 if volume_avg_20 is not None else "N/A",
                "volume_change_pct": volume_change_pct if volume_change_pct is not None else "N/A",
                "market_sentiment": market_sentiment,
                "market_trend": market_trend,
                "support": support,
                "resistance": resistance,
                "top_influencers": top_influencers,
                "group_leaders": group_leaders,
                "market_breadth": market_breadth
            })
        # Cuối cùng: chuẩn hóa toàn bộ merged_data về primitive
        merged_data = to_primitive(merged_data)
        # Bổ sung log chi tiết kiểm tra dữ liệu
        logger.info(f"[MERGE] keys: {list(merged_data.keys())}")
        logger.info(f"[MERGE] raw_data last date: {max(merged_data['raw_data'].keys()) if merged_data.get('raw_data') else 'N/A'}")
        logger.info(f"[MERGE] current_price: {merged_data.get('current_price')}, price_change: {merged_data.get('price_change')}, price_change_percent: {merged_data.get('price_change_percent')}, indicators: {merged_data.get('indicators')}")
        return merged_data 