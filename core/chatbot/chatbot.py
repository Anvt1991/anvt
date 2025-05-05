#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Chatbot tư vấn chứng khoán sử dụng AI
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import pytz
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import re
import warnings
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib as mpl
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import joblib
from dotenv import load_dotenv
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import traceback
import asyncio
import sqlite3

# Local imports
from core.technical import TechnicalAnalyzer
from core.news.news import NewsLoader
from core.data.data import DataLoader
from core.strategy.strategy import StrategyOptimizer
from core.model.enhanced_predictor import EnhancedPredictor
from core.data.data_validator import DataValidator
from core.ai.groq import GroqHandler
from core.ai.gemini import GeminiHandler
from core.data.db import DBManager
from pipeline.processor import PipelineProcessor

# Thiết lập logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Thiết lập warnings
warnings.filterwarnings('ignore')

# Thiết lập múi giờ
TZ = pytz.timezone('Asia/Ho_Chi_Minh')

# Load environment variables
load_dotenv()

# Thiết lập plot
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 12
sns.set_style('whitegrid')

class ChatbotAI:
    """
    Chatbot tư vấn chứng khoán sử dụng AI
    """
    
    def __init__(self):
        logger.info("Khởi tạo ChatbotAI...")
        # Chỉ khởi tạo pipeline, các thành phần khác lấy từ pipeline
        self.pipeline = PipelineProcessor()
        self.data_loader = self.pipeline.data_loader
        self.technical_analyzer = self.pipeline.technical_analyzer
        self.news_loader = self.pipeline.news_loader
        self.strategy_optimizer = getattr(self.pipeline, 'strategy_optimizer', None)
        self.enhanced_predictor = self.pipeline.enhanced_predictor
        self.groq_handler = self.pipeline.groq_handler
        self.gemini_handler = self.pipeline.gemini_handler
        self.db = self.pipeline.db
        # Cache, kết quả, figure cache giữ nguyên
        self.data_cache = {}
        self.last_update = {}
        self.models = {}
        self.figure_cache = {}
        self.current_results = {}
        logger.info("✓ ChatbotAI đã được khởi tạo thành công")
        
    def get_db_connection(self):
        """
        Tạo kết nối SQLite mới cho thread hiện tại
        
        Returns:
            Kết nối SQLite mới
        """
        conn = sqlite3.connect(self.db.db_file)
        conn.row_factory = sqlite3.Row
        return conn
    
    def load_symbol_data(self, symbol: str, reload: bool = False) -> pd.DataFrame:
        """
        Tải dữ liệu cho mã chứng khoán
        
        Args:
            symbol: Mã chứng khoán
            reload: Buộc tải lại dữ liệu
            
        Returns:
            DataFrame chứa dữ liệu lịch sử
        """
        current_time = datetime.now(TZ)
        
        # Kiểm tra cache
        if not reload and symbol in self.data_cache:
            last_update = self.last_update.get(symbol, datetime.min.replace(tzinfo=TZ))
            # Nếu dữ liệu đã tải trong vòng 4 giờ, sử dụng cache
            if (current_time - last_update).total_seconds() < 14400:  # 4 giờ = 14400 giây
                logger.info(f"Sử dụng dữ liệu cache cho {symbol}")
                return self.data_cache[symbol]
        
        # Tải dữ liệu mới
        logger.info(f"Tải dữ liệu mới cho {symbol}")
        try:
            # Sử dụng load_stock_data cho tất cả các loại dữ liệu kể cả chỉ số
            df = self.data_loader.load_stock_data(symbol)
            
            # Kiểm tra và xử lý dữ liệu
            if df is not None and not df.empty:
                df = DataValidator.detect_and_handle_outliers(df)[0]
                self.data_cache[symbol] = df
                self.last_update[symbol] = current_time
                return df
            else:
                logger.error(f"Không thể tải dữ liệu cho {symbol}")
                return None
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu cho {symbol}: {str(e)}")
            return None
    
    def analyze_symbol(self, symbol: str, period: str = 'short', reload: bool = False) -> Dict[str, Any]:
        """
        Phân tích mã chứng khoán sử dụng pipeline chuẩn hóa (PipelineProcessor)
        """
        logger.info(f"[ChatbotAI] Gọi pipeline chuẩn để phân tích {symbol} ({period})")
        result_obj = self.pipeline.process(symbol)
        # Ưu tiên lấy .result nếu có, nếu không thì lấy các trường public
        if hasattr(result_obj, 'to_dict') and callable(getattr(result_obj, 'to_dict')):
            result = result_obj.to_dict()
        elif hasattr(result_obj, 'result'):
            result = result_obj.result
        else:
            # fallback: lấy tất cả các trường public không bắt đầu bằng _ và không phải callable
            result = {k: getattr(result_obj, k) for k in dir(result_obj) if not k.startswith('_') and not callable(getattr(result_obj, k))}
        return result
    
    def suggest_portfolio(self, risk_level: str = 'moderate', 
                        stock_count: int = 5, 
                        industry_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Đề xuất danh mục đầu tư
        
        Args:
            risk_level: Mức độ rủi ro ('low', 'moderate', 'high')
            stock_count: Số lượng mã cổ phiếu
            industry_filter: Danh sách ngành nghề lọc
            
        Returns:
            Đề xuất danh mục
        """
        # Tải danh sách cổ phiếu
        stock_list = self.data_loader.get_vn30_stocks() if risk_level != 'high' else self.data_loader.get_all_stocks()
        
        # Lọc ngành nếu có yêu cầu
        if industry_filter:
            stock_list = [s for s in stock_list if self.data_loader.get_stock_industry(s) in industry_filter]
        
        # Tải dữ liệu cho từng mã
        stock_data = {}
        for symbol in stock_list[:min(30, len(stock_list))]:  # Giới hạn số lượng để tránh quá tải
            df = self.load_symbol_data(symbol)
            if df is not None and not df.empty:
                stock_data[symbol] = df
        
        # Tính toán danh mục tối ưu
        portfolio = self.portfolio_optimizer.optimize(
            stock_data=stock_data,
            risk_level=risk_level,
            stock_count=stock_count
        )
        
        return portfolio
    
    def generate_technical_chart(self, symbol: str, period: str = '6m') -> Optional[Figure]:
        """
        Tạo biểu đồ phân tích kỹ thuật
        
        Args:
            symbol: Mã chứng khoán
            period: Khoảng thời gian ('1m', '3m', '6m', '1y', '3y')
            
        Returns:
            Matplotlib Figure object
        """
        # Tải dữ liệu
        df = self.load_symbol_data(symbol)
        if df is None or df.empty:
            return None
        
        # Map period to number of days
        period_days = {
            '1m': 30,
            '3m': 90,
            '6m': 180,
            '1y': 365,
            '3y': 365*3
        }
        days = period_days.get(period, 180)
        
        # Lọc dữ liệu theo khoảng thời gian
        start_date = datetime.now(TZ) - timedelta(days=days)
        df = df[df.index >= start_date]
        
        # Tạo biểu đồ
        return self.technical_analyzer.generate_chart(df, symbol)
    
    def process_query(self, query: str) -> str:
        """
        Xử lý truy vấn từ người dùng
        
        Args:
            query: Truy vấn từ người dùng
            
        Returns:
            Phản hồi từ chatbot
        """
        query = query.strip().upper()
        
        # Kiểm tra nếu query là VNINDEX (đặc biệt xử lý)
        if query == "VNINDEX":
            logger.info("Xử lý truy vấn cho chỉ số VNINDEX")
            result = self.analyze_symbol("VNINDEX")
            
            if result is not None and "error" in result:
                if result.get("error") and "Không thể tải dữ liệu" in result["error"]:
                    return f"Không thể phân tích chỉ số VNINDEX: Không tìm thấy dữ liệu giá. Vui lòng thử lại sau."
                else:
                    return f"Không thể phân tích chỉ số VNINDEX: {result['error']}"
            
            # Xử lý tương tự như phân tích mã chứng khoán
            last_price = result.get("last_price")
            tech_analysis = result.get("technical_analysis", {})
            sentiment = result.get("news_sentiment", {})
            prediction = result.get("enhanced_prediction", {})
            pattern = result.get("pattern_analysis", {})
            
            response = f"Phân tích chỉ số VNINDEX:\n"
            if last_price is not None:
                response += f"Giá hiện tại: {last_price:,.2f} điểm\n\n"
            else:
                response += f"Giá hiện tại: Không có dữ liệu\n\n"
            
            # Phần kỹ thuật cơ bản
            response += "Phân tích kỹ thuật:\n"
            if "trend" in tech_analysis:
                response += f"- Xu hướng: {tech_analysis['trend']}\n"
            if "rsi" in tech_analysis and "rsi_signal" in tech_analysis:
                response += f"- RSI: {tech_analysis['rsi']:.2f} ({tech_analysis['rsi_signal']})\n"
            if "macd_signal" in tech_analysis:
                response += f"- MACD: {tech_analysis['macd_signal']}\n"
            if "bb_signal" in tech_analysis:
                response += f"- Bollinger Bands: {tech_analysis['bb_signal']}\n"
            if "support" in tech_analysis and "resistance" in tech_analysis:
                support_values = tech_analysis['support']
                resistance_values = tech_analysis['resistance']
                
                # Format support and resistance lists
                if isinstance(support_values, list) and isinstance(resistance_values, list):
                    def get_level(x):
                        if isinstance(x, dict):
                            return x.get('level', None)
                        return x
                    support_str = '/'.join([f"{get_level(s):.2f}" for s in support_values if get_level(s) is not None])
                    resistance_str = '/'.join([f"{get_level(r):.2f}" for r in resistance_values if get_level(r) is not None])
                    response += f"- Hỗ trợ/Kháng cự: {support_str}/{resistance_str}\n"
                else:
                    # Fallback for backwards compatibility if they're not lists
                    response += f"- Hỗ trợ/Kháng cự: {tech_analysis['support']:.2f}/{tech_analysis['resistance']:.2f}\n"
            response += "\n"
            
            # Thêm nhắc nhở về báo cáo chi tiết
            response += "Nhập 'THỊ TRƯỜNG CHI TIẾT' để xem báo cáo đầy đủ từ AI."
            
            return response
        
        # Mẫu regex cho mã chứng khoán 3 ký tự
        stock_pattern = r'\b[A-Z]{3}\b'
        
        # Kiểm tra xem có chứa mã chứng khoán không
        stock_matches = re.findall(stock_pattern, query)
        
        # Nếu tìm thấy mã, phân tích mã đó
        if stock_matches:
            symbol = stock_matches[0]
            logger.info(f"Xử lý truy vấn cho mã: {symbol}")
            
            # Kiểm tra lịch sử phân tích trong DB
            has_history = False
            history_text = ""
            try:
                # Sử dụng API của DBManager để truy vấn lịch sử
                history_reports = self.db.load_history_report(symbol, 1)
                
                if history_reports and len(history_reports) > 0:
                    history = history_reports[0]
                    creation_time = datetime.fromisoformat(history['created_at']) if 'created_at' in history else None
                    if creation_time and (datetime.now(TZ) - creation_time).total_seconds() < 86400:  # 24 giờ
                        has_history = True
                        last_price = history.get('close_price', 0)
                        prev_price = history.get('previous_close', 0)
                        change = last_price - prev_price
                        change_pct = (change / prev_price * 100) if prev_price > 0 else 0
                        
                        history_text = f"Đã tìm thấy phân tích gần đây ({creation_time.strftime('%H:%M %d/%m/%Y')}).\n"
                        history_text += f"Giá lúc đó: {last_price:,.2f} ({'+' if change >= 0 else ''}{change:,.2f} / {change_pct:.2f}%)\n\n"
            except Exception as e:
                logger.error(f"Lỗi khi truy vấn lịch sử DB: {e}")
            
            # Kiểm tra trong bộ nhớ cache trước
            if symbol in self.current_results:
                result = self.current_results[symbol]
                cache_time = result.get("timestamp")
                
                # Nếu cache còn mới (dưới 30 phút), sử dụng lại
                if cache_time and (datetime.now(TZ) - cache_time).total_seconds() < 1800:
                    logger.info(f"Sử dụng kết quả cache cho {symbol}")
                else:
                    # Nếu cache cũ, phân tích lại
                    result = self.analyze_symbol(symbol)
            else:
                # Nếu không có trong cache, phân tích mới
                result = self.analyze_symbol(symbol)
            
            if result is not None and "error" in result:
                if result.get("error") and "Không thể tải dữ liệu" in result["error"]:
                    return f"Không thể phân tích mã {symbol}: Không tìm thấy dữ liệu giá cho mã này. Vui lòng kiểm tra lại mã chứng khoán hoặc thử một mã khác."
                else:
                    return f"Không thể phân tích mã {symbol}: {result['error']}"
            
            # Tạo phản hồi
            last_price = result.get("last_price")
            tech_analysis = result.get("technical_analysis", {})
            sentiment = result.get("news_sentiment", {})
            prediction = result.get("enhanced_prediction", {})
            pattern = result.get("pattern_analysis", {})
            
            # Phần thông tin chung
            response = ""
            if has_history:
                response += history_text
            
            response += f"Phân tích mã {symbol}:\n"
            if last_price is not None:
                response += f"Giá hiện tại: {last_price:,.2f} VND\n\n"
            else:
                response += f"Giá hiện tại: Không có dữ liệu\n\n"
            
            # Phần kỹ thuật cơ bản
            response += "Phân tích kỹ thuật:\n"
            if "trend" in tech_analysis:
                response += f"- Xu hướng: {tech_analysis['trend']}\n"
            if "rsi" in tech_analysis and "rsi_signal" in tech_analysis:
                response += f"- RSI: {tech_analysis['rsi']:.2f} ({tech_analysis['rsi_signal']})\n"
            if "macd_signal" in tech_analysis:
                response += f"- MACD: {tech_analysis['macd_signal']}\n"
            if "bb_signal" in tech_analysis:
                response += f"- Bollinger Bands: {tech_analysis['bb_signal']}\n"
            if "support" in tech_analysis and "resistance" in tech_analysis:
                support_values = tech_analysis['support']
                resistance_values = tech_analysis['resistance']
                
                # Format support and resistance lists
                if isinstance(support_values, list) and isinstance(resistance_values, list):
                    def get_level(x):
                        if isinstance(x, dict):
                            return x.get('level', None)
                        return x
                    support_str = '/'.join([f"{get_level(s):.2f}" for s in support_values if get_level(s) is not None])
                    resistance_str = '/'.join([f"{get_level(r):.2f}" for r in resistance_values if get_level(r) is not None])
                    response += f"- Hỗ trợ/Kháng cự: {support_str}/{resistance_str}\n"
                else:
                    # Fallback for backwards compatibility if they're not lists
                    response += f"- Hỗ trợ/Kháng cự: {tech_analysis['support']:.2f}/{tech_analysis['resistance']:.2f}\n"
            response += "\n"
            
            # Phần phân tích pattern từ Groq
            if pattern and "technical_patterns" in pattern and len(pattern["technical_patterns"]) > 0:
                response += "Mẫu hình kỹ thuật (Groq):\n"
                for p in pattern["technical_patterns"][:2]:  # Giới hạn 2 mẫu hình quan trọng nhất
                    confidence = p.get("confidence", "")
                    implications = p.get("implications", "")
                    response += f"- {p.get('pattern_name', 'N/A')} ({confidence}): {implications}\n"
                response += "\n"
            
            # Phần tin tức
            if sentiment:
                response += f"Phân tích tin tức: {sentiment.get('overall', 'N/A')}\n"
                response += f"- Tích cực: {sentiment.get('positive', 0):.1f}%, Tiêu cực: {sentiment.get('negative', 0):.1f}%\n\n"
            
            # Phần dự đoán
            if prediction:
                short_term = prediction.get('short_term', {})
                medium_term = prediction.get('medium_term', {})
                
                response += "Dự đoán:\n"
                if short_term:
                    response += f"- Ngắn hạn (1-7 ngày): {short_term.get('direction', '')} {abs(short_term.get('change_pct', 0)):.2f}%\n"
                if medium_term:
                    response += f"- Trung hạn (1-3 tháng): {medium_term.get('direction', '')} {abs(medium_term.get('change_pct', 0)):.2f}%\n"
                response += f"- Xác suất tăng: {prediction.get('probability_up', 0):.1f}%\n\n"
                
                # Khuyến nghị
                response += f"Khuyến nghị: {prediction.get('recommendation', 'Không có')}"
            
            # Thêm báo cáo AI (Gemini)
            if "report" in result:
                response += "\n\nBáo cáo phân tích chi tiết:\n" + result["report"]
            
            return response
        
        # Trợ giúp và lịch sử gần đây
        if "TRỢ GIÚP" in query or "HELP" in query:
            help_text = """Chatbot tư vấn chứng khoán: 
- Nhập mã chứng khoán (VD: FPT) để phân tích chi tiết
- Nhập "THỊ TRƯỜNG" để xem tình hình thị trường
- Nhập "DANH MỤC" để đề xuất danh mục đầu tư
- Nhập "LỊCH SỬ" để xem các phân tích gần đây"""
            return help_text
            
        # Kiểm tra lịch sử báo cáo
        if "LỊCH SỬ" in query or "HISTORY" in query:
            try:
                # Lấy các báo cáo gần đây từ DB
                histories = []
                
                # Lấy các symbol gần đây từ DB (lấy từ table report_history)
                symbols = []
                try:
                    conn = sqlite3.connect(self.db.db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT symbol FROM report_history ORDER BY id DESC LIMIT 10")
                    symbols = [row[0] for row in cursor.fetchall()]
                    conn.close()
                except Exception as e:
                    logger.error(f"Lỗi khi truy vấn DB: {e}")
                
                if symbols:
                    response = "Các mã đã phân tích gần đây:\n\n"
                    for idx, symbol in enumerate(symbols, 1):
                        response += f"{idx}. {symbol}\n"
                    response += "\nNhập mã để xem phân tích chi tiết."
                    return response
                else:
                    return "Chưa có phân tích nào được lưu trong lịch sử."
            except Exception as e:
                logger.error(f"Lỗi khi lấy lịch sử: {e}")
                return "Không thể lấy lịch sử phân tích do lỗi hệ thống."
        
        # Đề xuất danh mục
        if "DANH MỤC" in query or "ĐẦU TƯ" in query:
            risk_level = "moderate"
            if "RỦI RO THẤP" in query or "AN TOÀN" in query:
                risk_level = "low"
            elif "RỦI RO CAO" in query or "TĂNG TRƯỞNG" in query:
                risk_level = "high"
                
            portfolio = self.suggest_portfolio(risk_level=risk_level)
            
            response = f"Đề xuất danh mục đầu tư (mức độ rủi ro: {risk_level}):\n\n"
            for i, (symbol, weight) in enumerate(portfolio["weights"].items()):
                response += f"{i+1}. {symbol}: {weight*100:.1f}%\n"
            
            response += f"\nLợi nhuận kỳ vọng: {portfolio['expected_return']*100:.2f}%\n"
            response += f"Rủi ro (độ lệch chuẩn): {portfolio['risk']*100:.2f}%\n"
            response += f"Tỷ lệ Sharpe: {portfolio['sharpe_ratio']:.2f}"
            
            return response
        
        # Phân tích thị trường
        if "THỊ TRƯỜNG" in query:
            logger.info("Xử lý truy vấn phân tích thị trường")
            
            # Kiểm tra xem có muốn báo cáo chi tiết không
            detailed_report = "CHI TIẾT" in query or "ĐẦY ĐỦ" in query
            
            if detailed_report:
                # Tạo báo cáo chi tiết từ pipeline chuẩn (lấy từ result['report'])
                logger.info("Tạo báo cáo chi tiết VNINDEX với pipeline chuẩn")
                result = self.analyze_symbol("VNINDEX")
                if "report" in result:
                    return result["report"]
                else:
                    return "Không có báo cáo chi tiết từ AI cho VNINDEX."
            else:
                # Lấy thông tin từ analyze_market_condition
                market = self.analyze_market_condition()
                
                if "error" in market:
                    return f"Không thể phân tích thị trường: {market['error']}"
                
                response = "Phân tích thị trường:\n"
                response += f"VN-Index: {market['vnindex_last']:,.2f} "
                
                change = market['vnindex_change']
                change_pct = market['vnindex_change_pct']
                if change >= 0:
                    response += f"↑ +{change:,.2f} (+{change_pct:.2f}%)\n\n"
                else:
                    response += f"↓ {change:,.2f} ({change_pct:.2f}%)\n\n"
                
                # Thêm thông tin về khối lượng nếu có
                if market.get('volume') and market.get('volume_avg_10'):
                    vol_change = market.get('volume_change_pct')
                    vol_text = f"Khối lượng: {market['volume']:,.0f} "
                    if vol_change:
                        vol_text += f"({'+'if vol_change >= 0 else ''}{vol_change:.1f}% so với TB 10 phiên)\n"
                    response += vol_text
                
                # Thêm RSI và MACD nếu có
                if market.get('rsi'):
                    rsi_value = market['rsi']
                    rsi_signal = "Quá bán" if rsi_value < 30 else "Quá mua" if rsi_value > 70 else "Trung tính"
                    response += f"RSI: {rsi_value:.1f} ({rsi_signal})\n"
                
                response += f"Xu hướng: {market['trend']}\n"
                response += f"Tâm lý thị trường: {market['market_sentiment']['overall']}\n\n"
                
                # Thêm nhắc nhở về báo cáo chi tiết
                response += "Nhập 'THỊ TRƯỜNG CHI TIẾT' để xem báo cáo đầy đủ từ AI."
                
                return response
        
        # Mặc định nếu không hiểu truy vấn
        return """Tôi không hiểu yêu cầu của bạn. Vui lòng:
- Nhập mã chứng khoán (ví dụ: FPT) để phân tích chi tiết
- Nhập "THỊ TRƯỜNG" để xem tình hình thị trường chung
- Nhập "DANH MỤC" để nhận đề xuất danh mục đầu tư"""

    def validate_input(self, text: str) -> Tuple[bool, str]:
        """
        Kiểm tra tính hợp lệ của đầu vào
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Tuple (is_valid, message)
        """
        if not text.strip():
            return False, "Vui lòng nhập nội dung truy vấn"
        
        # Kiểm tra độ dài
        if len(text) > 500:
            return False, "Truy vấn quá dài, vui lòng giới hạn trong 500 ký tự"
        
        return True, ""

    # XÓA HÀM BÁO CÁO AI CŨ (Gemini/Groq/OpenRouter)
    # def generate_vnindex_report(self): ...

class ChatbotGUI:
    """
    GUI class for the chatbot
    """
    def __init__(self, root, config=None):
        """Initialize the GUI"""
        self.root = root
        self.root.title("Quân Sư Quạt Mo - Trợ lý đầu tư chứng khoán")
        self.root.geometry("800x600")
        self.root.minsize(width=700, height=500)
        
        # Set up icon if available
        try:
            self.root.iconbitmap("assets/icon.ico")
        except:
            pass
        
        # Initialize tkinter StringVar for status
        self.status_var = tk.StringVar()
        self.status_var.set("Sẵn sàng phục vụ")
        
        # Configure ttk styles
        self.configure_styles()
        
        # Initialize chatbot
        try:
            self.chatbot = ChatbotAI()
            logger.info("✓ ChatbotAI backend đã khởi tạo")
        except Exception as e:
            logger.error(f"Error initializing ChatbotAI backend: {str(e)}")
            self.chatbot = None
        
        # Apply configuration
        if config:
            self._apply_config_to_chatbot(config)
        
        # Flag to prevent multiple requests
        self.processing = False
        
        # Initialize thread and queue for handling requests
        self.queue = queue.Queue()
        
        # Create GUI elements
        self.create_widgets()
        
        # Check database connection
        self.check_database_connection()
        
        # Start checking queue
        self.check_queue()
        
        # Center window
        self.center_window()
        
        logger.info("✓ ChatbotGUI initialization complete")
        
    def _apply_config_to_chatbot(self, config):
        """Apply configuration to chatbot components"""
        if not self.chatbot or not config:
            return
            
        try:
            # Configure data loader
            data_config = config.get('data', {})
            if hasattr(self.chatbot, 'data_loader') and data_config:
                self.chatbot.data_loader.default_source = data_config.get('default_source', 'vnstock')
                self.chatbot.data_loader.cache_enabled = data_config.get('cache_enabled', True)
                self.chatbot.data_loader.default_period = data_config.get('default_period', '1y')
            
            # Configure technical analyzer
            ta_config = config.get('technical_analysis', {})
            if hasattr(self.chatbot, 'technical_analyzer') and ta_config:
                self.chatbot.technical_analyzer.indicators = ta_config.get('indicators', [])
            
            # Configure AI providers
            ai_config = config.get('ai', {}).get('providers', {})
            
            # Configure Groq if enabled
            groq_config = ai_config.get('groq', {})
            if hasattr(self.chatbot, 'groq_handler') and groq_config:
                self.chatbot.groq_handler.enabled = groq_config.get('enabled', False)
                self.chatbot.groq_handler.api_key = groq_config.get('api_key', os.getenv('GROQ_API_KEY', ''))
                self.chatbot.groq_handler.model = groq_config.get('model', 'llama3-8b-8192')
            
            # Configure Gemini if enabled
            gemini_config = ai_config.get('gemini', {})
            if hasattr(self.chatbot, 'gemini_handler') and gemini_config:
                self.chatbot.gemini_handler.enabled = gemini_config.get('enabled', False)
                self.chatbot.gemini_handler.api_key = gemini_config.get('api_key', os.getenv('GEMINI_API_KEY', ''))
                self.chatbot.gemini_handler.model = gemini_config.get('model', 'gemini-pro')
            
            # Configure enhanced predictor
            predictor_config = config.get('predictor', {})
            if hasattr(self.chatbot, 'enhanced_predictor') and predictor_config:
                self.chatbot.enhanced_predictor.confidence_threshold = predictor_config.get('confidence_threshold', 0.7)
                self.chatbot.enhanced_predictor.max_forecast_days = predictor_config.get('max_forecast_days', 30)
                
            # Configure database connection
            db_config = config.get('database', {})
            if hasattr(self.chatbot, 'db') and db_config:
                # Apply database settings that don't require reinitialization
                pass
                
            logger.info("✓ Configuration applied to ChatbotAI components")
        except Exception as e:
            logger.error(f"Error applying configuration: {str(e)}")
    
    def check_database_connection(self):
        """Check database connection"""
        try:
            if not self.chatbot:
                logger.error("Lỗi kết nối database: chatbot instance is None")
                messagebox.showwarning("Cảnh báo", "Không thể kết nối đến database. Một số chức năng có thể không hoạt động.")
                return False
                
            conn = self.chatbot.get_db_connection()
            if conn:
                logger.info("Kết nối database thành công")
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Lỗi kết nối database: {str(e)}")
            messagebox.showwarning("Cảnh báo", "Không thể kết nối đến database. Một số chức năng có thể không hoạt động.")
            return False
    
    def create_widgets(self):
        """Tạo các widget cho giao diện"""
        # Tạo main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Tạo header frame
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title và logo
        title_label = ttk.Label(header_frame, text="QUÂN SƯ QUẠT MO - Tư vấn chứng khoán AI", style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Khu vực hiển thị tin nhắn
        chat_frame = ttk.LabelFrame(main_frame, text="Cuộc trò chuyện", padding="5")
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Text widget với scrollbar
        chat_display_frame = ttk.Frame(chat_frame)
        chat_display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.chat_display = scrolledtext.ScrolledText(
            chat_display_frame, 
            wrap=tk.WORD, 
            width=70, 
            height=20,
            font=('Helvetica', 10),
            background='white'
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Cấu hình tags cho định dạng văn bản
        self.chat_display.tag_configure('user', foreground='#000080', font=('Helvetica', 10, 'bold'))
        self.chat_display.tag_configure('user_message', foreground='black', font=('Helvetica', 10))
        self.chat_display.tag_configure('bot', foreground='#800000', font=('Helvetica', 10, 'bold'))
        self.chat_display.tag_configure('bot_message', foreground='black', font=('Helvetica', 10))
        self.chat_display.tag_configure('error', foreground='red', font=('Helvetica', 10, 'bold'))
        self.chat_display.tag_configure('timestamp', foreground='gray', font=('Helvetica', 8))
        self.chat_display.tag_configure('suggestion', foreground='blue', font=('Helvetica', 10, 'italic'))
        
        # Khu vực nhập tin nhắn
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        # Label hướng dẫn
        prompt_label = ttk.Label(input_frame, text="Nhập mã chứng khoán hoặc câu hỏi:", anchor=tk.W)
        prompt_label.pack(fill=tk.X, pady=(0, 5))
        
        # Entry và button
        entry_button_frame = ttk.Frame(input_frame)
        entry_button_frame.pack(fill=tk.X)
        
        self.message_entry = ttk.Entry(entry_button_frame, font=('Helvetica', 10))
        self.message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.message_entry.bind("<Return>", self.send_message)
        self.message_entry.focus_set()
        
        self.send_button = ttk.Button(entry_button_frame, text="Gửi", command=self.send_message, width=10)
        self.send_button.pack(side=tk.RIGHT)
        
        # Thanh trạng thái
        status_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, padding=(5, 2))
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        status_indicator = ttk.Label(status_frame, text="●", foreground="green", font=('Helvetica', 10))
        status_indicator.pack(side=tk.LEFT, padx=(0, 5))
        
        status_label = ttk.Label(status_frame, textvariable=self.status_var, style='Status.TLabel', anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Quick action buttons
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=5)
        
        # Market button
        market_button = ttk.Button(
            action_frame, 
            text="Thị trường", 
            command=lambda: self.process_quick_action("THỊ TRƯỜNG"),
            width=15
        )
        market_button.pack(side=tk.LEFT, padx=5)
        
        # Portfolio button
        portfolio_button = ttk.Button(
            action_frame, 
            text="Danh mục đầu tư", 
            command=lambda: self.process_quick_action("DANH MỤC"),
            width=15
        )
        portfolio_button.pack(side=tk.LEFT, padx=5)
        
        # Top stocks button
        stocks_button = ttk.Button(
            action_frame, 
            text="Cổ phiếu VN30", 
            command=lambda: self.process_quick_action("VN30"),
            width=15
        )
        stocks_button.pack(side=tk.LEFT, padx=5)
        
        # Help button
        help_button = ttk.Button(
            action_frame, 
            text="Trợ giúp", 
            command=lambda: self.process_quick_action("TRỢ GIÚP"),
            width=15
        )
        help_button.pack(side=tk.LEFT, padx=5)
        
        # Hiển thị tin nhắn chào mừng
        self.display_welcome_message()
    
    def display_welcome_message(self):
        """Hiển thị tin nhắn chào mừng"""
        welcome_message = """Xin chào! Em là Gia Cát Quạt Mo, trợ lý AI về chứng khoán.

Em có thể:
1. Phân tích cổ phiếu (VD: FPT, VNM, ACB,...)
2. Xem tình hình thị trường (Gõ: THỊ TRƯỜNG)
3. Đề xuất danh mục đầu tư (Gõ: DANH MỤC)
4. Trả lời câu hỏi về chứng khoán

Đại ca cần gì ạ?"""
        
        self.display_message("ĐỆ", welcome_message)
        
        # Hiển thị gợi ý
        self.display_suggestion("Gợi ý: Nhập mã cổ phiếu (VD: FPT) để phân tích chi tiết")
    
    def display_message(self, sender, message):
        """
        Hiển thị tin nhắn lên giao diện
        
        Args:
            sender: Người gửi tin nhắn
            message: Nội dung tin nhắn
        """
        self.chat_display.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Định dạng khác nhau cho người dùng và chatbot
        if sender == "Bạn":
            self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.chat_display.insert(tk.END, f"{sender}: ", "user")
            self.chat_display.insert(tk.END, f"{message}\n\n", "user_message")
        elif sender == "Lỗi":
            self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.chat_display.insert(tk.END, f"{sender}: ", "error")
            self.chat_display.insert(tk.END, f"{message}\n\n", "error")
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] ", "timestamp")
            self.chat_display.insert(tk.END, f"{sender}: ", "bot")
            
            # Xử lý định dạng đặc biệt trong tin nhắn 
            # (sử dụng ** để in đậm, __ để in nghiêng)
            processed_message = self.format_message(message)
            self.chat_display.insert(tk.END, f"{processed_message}\n\n", "bot_message")
        
        # Cuộn xuống để hiển thị tin nhắn mới nhất
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def format_message(self, message):
        """
        Xử lý định dạng đặc biệt trong tin nhắn
        
        Args:
            message: Tin nhắn cần xử lý
            
        Returns:
            Tin nhắn đã xử lý
        """
        # TODO: Implement message formatting with markdown-like syntax
        # This is just a placeholder for future enhancement
        return message
    
    def display_suggestion(self, suggestion):
        """
        Hiển thị gợi ý
        
        Args:
            suggestion: Nội dung gợi ý
        """
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, f"{suggestion}\n\n", "suggestion")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.see(tk.END)
    
    def send_message(self, event=None):
        """
        Send a message from the user to the chatbot
        
        Args:
            event: Event object (used for Enter key binding)
        """
        # Get message from entry widget
        message = self.message_entry.get().strip()
        
        # Clear entry widget
        self.message_entry.delete(0, tk.END)
        
        # If message is empty, do nothing
        if not message:
            return
        
        # Display user message in chat display
        self.display_message("Bạn", message)
        
        # Process message
        if self.processing:
            # If already processing a message, display error
            self.display_message("Lỗi", "Vui lòng đợi xử lý xong tin nhắn trước đó")
            return
        
        # Disable send button while processing
        self.send_button.config(state=tk.DISABLED)
        self.status_var.set("Đang xử lý...")
        
        # Check if input is valid
        if self.chatbot:
            is_valid, error_msg = self.chatbot.validate_input(message)
        else:
            is_valid = True  # Allow messages even if chatbot is None
            error_msg = None
            
        if not is_valid:
            self.display_message("Lỗi", error_msg)
            self.send_button.config(state=tk.NORMAL)
            self.status_var.set("Sẵn sàng phục vụ")
            return
        
        # Start processing thread
        self.processing = True
        processing_thread = threading.Thread(target=self.process_message_thread, args=(message,))
        processing_thread.daemon = True
        processing_thread.start()
    
    def process_quick_action(self, action):
        """
        Xử lý hành động nhanh từ nút bấm
        
        Args:
            action: Hành động cần xử lý
        """
        if self.processing:
            return
            
        self.display_message("Bạn", action)
        
        # Cập nhật trạng thái
        self.processing = True
        self.status_var.set("Đang xử lý...")
        self.send_button.config(state=tk.DISABLED)
        
        # Tạo luồng mới để xử lý tin nhắn
        thread = threading.Thread(target=self.process_message_thread, args=(action,))
        thread.daemon = True
        thread.start()
        
        # Kiểm tra hàng đợi định kỳ
        self.root.after(100, self.check_queue)
    
    def process_message_thread(self, message):
        """
        Process a message in a separate thread
        
        Args:
            message: Message to process
        """
        try:
            if not self.chatbot:
                # If chatbot is not initialized, display error message
                # Avoid any string formatting issues
                result = "Hệ thống chưa sẵn sàng. Lỗi kết nối cơ sở dữ liệu, vui lòng khởi động lại ứng dụng."
            else:
                try:
                    # Process message with ChatbotAI
                    result = self.chatbot.process_query(message)
                except Exception as internal_e:
                    # Handle any errors from the chatbot's process_query method
                    logger.error(f"Error in chatbot.process_query: {str(internal_e)}", exc_info=True)
                    result = f"Lỗi xử lý: {str(internal_e)}"
            
            # Put result in queue for main thread to display
            self.queue.put(("ĐỆ", result))
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", exc_info=True)
            error_message = "Rất tiếc, đã xảy ra lỗi khi xử lý tin nhắn của bạn."
            self.queue.put(("Lỗi", error_message))
        finally:
            # Reset processing flag
            self.processing = False
            # Update status
            self.status_var.set("Sẵn sàng phục vụ")
            # Enable send button
            if hasattr(self, "send_button"):
                self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))
    
    def check_queue(self):
        """
        Check the queue for messages to display
        """
        try:
            # Check for pending messages
            while not self.queue.empty():
                # Get message from queue
                sender, message = self.queue.get_nowait()
                
                # Display message based on type
                if sender == "Lỗi":
                    self.display_message("Lỗi", message)
                elif sender == "ĐỆ":
                    self.display_message("ĐỆ", message)
                elif sender == "Gợi ý":
                    self.display_suggestion(message)
                
                # Mark as done
                self.queue.task_done()
        except Exception as e:
            logger.error(f"Error checking queue: {str(e)}")
        
        # Schedule next check
        self.root.after(100, self.check_queue)

    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() - width) // 2
        y = (self.root.winfo_screenheight() - height) // 2
        self.root.geometry(f"+{x}+{y}")

    def configure_styles(self):
        """Configure ttk styles for the GUI"""
        style = ttk.Style()
        
        # Try to use a modern theme if available
        try:
            style.theme_use('clam')  # 'clam' is generally available on most platforms
        except:
            pass  # Use default theme if 'clam' is not available
        
        # Configure styles for various widgets
        style.configure('TFrame', background='#f5f5f5')
        style.configure('TButton', font=('Helvetica', 10))
        style.configure('TLabel', font=('Helvetica', 10))
        style.configure('Title.TLabel', font=('Helvetica', 14, 'bold'))
        style.configure('Status.TLabel', font=('Helvetica', 9))
