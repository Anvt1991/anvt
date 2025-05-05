import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
import httpx
from loguru import logger
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Tải biến môi trường từ file .env
load_dotenv()

class GeminiHandler:
    """
    Lớp quản lý tương tác với Google Gemini API để tổng hợp và phân tích báo cáo
    """
    
    def __init__(self, api_key: str = None):
        """
        Khởi tạo GeminiHandler
        
        Args:
            api_key: API key của Google Gemini (mặc định lấy từ biến môi trường GEMINI_API_KEY)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY không được cung cấp và không tìm thấy trong biến môi trường")
        
        # Cấu hình Google Generative AI
        genai.configure(api_key=self.api_key)
        
        # Thông số model
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        self.model = genai.GenerativeModel(self.model_name)
        self.generation_config = {
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
        
        self.retry_config = {
            "max_retries": int(os.getenv("GEMINI_MAX_RETRIES", 3)),
            "retry_delay": float(os.getenv("GEMINI_RETRY_DELAY", 1.0))
        }
        
        logger.info(f"Khởi tạo GeminiHandler với model: {self.model_name}")
    
    def build_technical_prompt(self, pattern_data: Dict[str, Any], output_format: str = "markdown"):
        """
        Tạo prompt cho báo cáo phân tích kỹ thuật
        
        Args:
            pattern_data: Dữ liệu phân tích mẫu hình đã được tối ưu
            output_format: Định dạng đầu ra
            
        Returns:
            Tuple (system_prompt, user_prompt)
        """
        symbol = pattern_data.get("symbol", "N/A")
        
        # Kiểm tra và ghi log cảnh báo nếu thiếu thông tin quan trọng
        if not symbol or symbol == "N/A":
            logger.warning("Dữ liệu không có mã cổ phiếu (symbol)")
        if not pattern_data.get("indicators"):
            logger.warning(f"Dữ liệu thiếu thông tin chỉ báo kỹ thuật cho mã {symbol}")
        if not pattern_data.get("support_resistance"):
            logger.warning(f"Dữ liệu thiếu thông tin hỗ trợ/kháng cự cho mã {symbol}")
        if not pattern_data.get("overall_rating"):
            logger.warning(f"Dữ liệu thiếu đánh giá tổng thể cho mã {symbol}")
            
        system_prompt = f"""Bạn là chuyên gia phân tích kỹ thuật chứng khoán Việt Nam với nhiều năm kinh nghiệm.
Hãy viết báo cáo phân tích kỹ thuật cho mã {symbol} , nội dung dựa trên dữ liệu được cung cấp.
Báo cáo cần ngắn gọn, súc tích, dễ hiểu, dùng ngôn ngữ chuyên nghiệp dành cho nhà đầu tư.
Tập trung vào các xu hướng chính, các mô hình giá, các chỉ báo kỹ thuật quan trọng, và đưa ra nhận định.
"""
        
        user_prompt = f"Dữ liệu phân tích cho mã {symbol}:\n{pattern_data}\n\nHãy tạo báo cáo phân tích kỹ thuật bằng định dạng {output_format}."
        
        return system_prompt, user_prompt

    def build_market_prompt(self, pattern_data: Dict[str, Any], output_format: str = "markdown"):
        """
        Tạo prompt cho báo cáo phân tích thị trường
        
        Args:
            pattern_data: Dữ liệu phân tích thị trường đã được tối ưu
            output_format: Định dạng đầu ra
            
        Returns:
            Tuple (system_prompt, user_prompt)
        """
        # Kiểm tra và ghi log cảnh báo nếu thiếu thông tin quan trọng
        if not pattern_data.get("vnindex_last"):
            logger.warning("Dữ liệu thiếu thông tin chỉ số VNINDEX")
            
        system_prompt = """Bạn là chuyên gia phân tích thị trường chứng khoán Việt Nam với hơn 15 năm kinh nghiệm.
Hãy viết báo cáo phân tích thị trường VNINDEX, nội dung dựa trên dữ liệu được cung cấp.
Báo cáo cần ngắn gọn, súc tích, nhưng đầy đủ thông tin quan trọng, dùng ngôn ngữ chuyên nghiệp dành cho nhà đầu tư.

Báo cáo cần có các phần sau:
1. Tóm tắt diễn biến thị trường
2. Phân tích kỹ thuật VNINDEX
3. Phân tích dòng tiền và thanh khoản
4. Phân tích nhóm ngành dẫn dắt và cổ phiếu ảnh hưởng
5. Phân tích hoạt động khối ngoại (nếu có dữ liệu)
6. Tâm lý thị trường và tin tức vĩ mô
7. Nhận định chung và triển vọng ngắn hạn
"""
        
        user_prompt = f"Dữ liệu phân tích thị trường:\n{pattern_data}\n\nHãy tạo báo cáo phân tích thị trường bằng định dạng {output_format}."
        
        return system_prompt, user_prompt

    def generate_report(self, 
                       pattern_data: Dict[str, Any], 
                       market_data: Dict[str, Any] = None,
                       report_type: str = "technical",
                       output_format: str = "markdown") -> str:
        """
        Tạo báo cáo từ dữ liệu phân tích bằng Gemini API (sync)
        """
        try:
            if not pattern_data:
                raise ValueError("Dữ liệu phân tích không được cung cấp")
            symbol = pattern_data.get("symbol", "N/A")
            if report_type.lower() == "market":
                system_prompt, user_prompt = self.build_market_prompt(pattern_data, output_format)
            else:
                system_prompt, user_prompt = self.build_technical_prompt(pattern_data, output_format)
            response = self._call_gemini_api_sync(system_prompt, user_prompt)
            if not response:
                logger.error(f"Không nhận được phản hồi từ Gemini API khi tạo báo cáo cho {symbol}")
                return f"Không thể tạo báo cáo cho {symbol} do lỗi API."
            logger.info(f"Đã tạo báo cáo cho mã {symbol}")
            return response
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo: {str(e)}")
            return f"Lỗi khi tạo báo cáo: {str(e)}"

    def _call_gemini_api_sync(self, system_prompt: str, user_prompt: str) -> str:
        """
        Gọi Gemini API phiên bản đồng bộ (sync) với retry logic
        
        Args:
            system_prompt: Nội dung system prompt
            user_prompt: Nội dung user prompt
            
        Returns:
            Nội dung phản hồi hoặc None nếu lỗi
        """
        retry_count = 0
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        while retry_count < self.retry_config["max_retries"]:
            try:
                # Cấu hình model
                model = genai.GenerativeModel(self.model_name)
                
                # Tạo chat session
                chat = model.start_chat(history=[])
                
                # Gửi tin nhắn
                response = chat.send_message(
                    combined_prompt,
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings
                )
                
                return response.text
                
            except Exception as e:
                retry_count += 1
                wait_time = self.retry_config["retry_delay"] * (2 ** retry_count)
                
                logger.warning(f"Lỗi khi gọi Gemini API: {str(e)}. Thử lại lần {retry_count} sau {wait_time} giây...")
                time.sleep(wait_time)
        
        logger.error(f"Đã thử {self.retry_config['max_retries']} lần gọi Gemini API nhưng không thành công")
        return None

    def set_model(self, model_name: str) -> None:
        """
        Thay đổi model của Gemini
        
        Args:
            model_name: Tên model mới
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"Đã chuyển sang model: {model_name}")
    
    def test_connection(self) -> bool:
        """
        Kiểm tra kết nối với Gemini API
        
        Returns:
            True nếu kết nối thành công, False nếu không
        """
        try:
            # Cấu hình đơn giản để kiểm tra kết nối
            generation_config = {
                "temperature": 0.1,
                "max_output_tokens": 32,
            }
            
            # Thực hiện kiểm tra với timeout 15 giây
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    lambda: genai.GenerativeModel(self.model_name).generate_content("Hello, world!").text is not None
                )
                try:
                    is_connected = future.result(timeout=15)
                    if is_connected:
                        logger.info("Kết nối Gemini API thành công")
                    else:
                        logger.error("Kết nối Gemini API thất bại")
                    return is_connected
                except TimeoutError:
                    logger.error("Kết nối Gemini API bị timeout")
                    return False
            
        except Exception as e:
            logger.error(f"Lỗi không xác định khi kết nối Gemini API: {str(e)}")
            return False

    def generate_report_sync(self, pattern_data: Dict[str, Any] = None, market_data: Dict[str, Any] = None,
                            report_type: str = "technical", output_format: str = "markdown") -> str:
        """
        Phiên bản đồng bộ của hàm generate_report
        
        Args:
            pattern_data: Dữ liệu phân tích đã được tối ưu
            market_data: Dữ liệu thị trường bổ sung (không sử dụng khi đã tối ưu)
            report_type: Loại báo cáo - "technical" hoặc "market"
            output_format: Định dạng đầu ra
            
        Returns:
            Nội dung báo cáo hoặc thông báo lỗi
        """
        try:
            # Kiểm tra tham số đầu vào
            if not pattern_data:
                logger.error("Tham số pattern_data là bắt buộc")
                return "ERROR: Thiếu dữ liệu phân tích"
            
            symbol = pattern_data.get("symbol", "Unknown")
            logger.info(f"Đang tạo báo cáo {report_type} cho mã {symbol}")
            
            # Xác định loại prompt và tạo prompt tương ứng
            if report_type.lower() == "market":
                system_prompt, user_prompt = self.build_market_prompt(pattern_data, output_format)
            else:
                system_prompt, user_prompt = self.build_technical_prompt(pattern_data, output_format)
            
            # Gọi API
            response = self._call_gemini_api_sync(system_prompt, user_prompt)
            
            # Kiểm tra kết quả
            if not response:
                logger.error(f"Không nhận được phản hồi từ Gemini API khi tạo báo cáo cho {symbol}")
                return None
            
            logger.info(f"Đã tạo báo cáo cho mã {symbol}")
            return response
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo báo cáo đồng bộ: {str(e)}")
            return None

    # Xóa hoặc chuyển _call_gemini_api thành private sync nếu không còn dùng async. 