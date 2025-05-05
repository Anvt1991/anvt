#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script để tạo báo cáo tự động theo lịch trình
Có thể chạy bằng crontab trên hệ thống Linux/Mac hoặc Task Scheduler trên Windows
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
import asyncio
from typing import List, Dict, Any

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/scheduled_reports.log")
    ]
)
logger = logging.getLogger("scheduled_reports")

# Import các module cần thiết
from core.data.db import DBManager
from core.data.load_data import DataLoader
from pipeline.processor import PipelineProcessor
from core.report.manager import ReportManager
from core.telegram.notify import send_report_to_telegram
from core.ai.groq import GroqHandler
from core.ai.gemini import GeminiHandler

class ScheduledReportGenerator:
    """
    Lớp quản lý tạo báo cáo tự động theo lịch trình
    """
    
    def __init__(self, config_path: str = None):
        """
        Khởi tạo Generator
        
        Args:
            config_path: Đường dẫn đến file cấu hình
        """
        # Tải cấu hình
        self.config = self._load_config(config_path)
        
        # Khởi tạo các thành phần core
        self.db = DBManager()
        self.data_loader = DataLoader()
        self.pipeline = PipelineProcessor()
        self.groq_handler = GroqHandler()
        self.gemini_handler = GeminiHandler()
        self.report_manager = ReportManager(self.db, self.gemini_handler)
        
        logger.info("ScheduledReportGenerator đã khởi tạo thành công")
    
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """
        Tải cấu hình từ file
        
        Args:
            config_path: Đường dẫn đến file cấu hình
            
        Returns:
            Dict chứa thông tin cấu hình
        """
        # Đường dẫn mặc định
        if not config_path:
            config_path = os.path.join("config", "telegram_config.json")
        
        try:
            # Kiểm tra file tồn tại
            if not os.path.exists(config_path):
                logger.warning(f"Không tìm thấy file cấu hình {config_path}, sử dụng cấu hình mặc định")
                return {
                    "telegram": {
                        "daily_report": {
                            "enabled": True,
                            "symbols": ["VNINDEX", "VN30"]
                        }
                    }
                }
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Đã tải cấu hình từ {config_path}")
            return config
        except Exception as e:
            logger.error(f"Lỗi khi tải cấu hình: {str(e)}")
            return {
                "telegram": {
                    "daily_report": {
                        "enabled": True,
                        "symbols": ["VNINDEX", "VN30"]
                    }
                }
            }
    
    def generate_report_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Tạo báo cáo cho một mã cổ phiếu (sync)
        """
        try:
            logger.info(f"Bắt đầu tạo báo cáo cho {symbol}")
            
            # Sử dụng pipeline processor để phân tích
            pipeline_data = self.pipeline.process(symbol)
            
            if not pipeline_data or (hasattr(pipeline_data, 'has_error') and pipeline_data.has_error):
                error_msg = getattr(pipeline_data, 'error', 'Không rõ lỗi') if pipeline_data else 'Không nhận được dữ liệu'
                logger.error(f"Lỗi khi phân tích {symbol}: {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Tổng hợp dữ liệu
            aggregated_data = self.pipeline.aggregate_pipeline_result(pipeline_data)
            
            # Sinh báo cáo
            report_result = self.report_manager.generate_report(
                aggregated_data, 
                symbol,
                meta={
                    "close_today": aggregated_data.get("current_price", 0),
                    "close_yesterday": aggregated_data.get("previous_price", 0)
                }
            )
            
            if not report_result.get("success") or not report_result.get("report"):
                error_msg = report_result.get("error", "Không thể sinh báo cáo")
                logger.error(f"Lỗi khi sinh báo cáo cho {symbol}: {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Lưu báo cáo vào DB
            saved = self.report_manager.save_report(
                symbol,
                report_result["report"],
                aggregated_data.get("current_price", 0),
                aggregated_data.get("previous_price", 0)
            )
            
            logger.info(f"Đã tạo báo cáo cho {symbol} thành công")
            return {
                "success": True, 
                "report": report_result["report"],
                "saved": saved
            }
        except Exception as e:
            logger.error(f"Lỗi khi xử lý {symbol}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def send_report(self, report: str, symbol: str) -> bool:
        """
        Gửi báo cáo qua Telegram (sync)
        """
        try:
            # Lấy thông tin bot_token và chat_id từ cấu hình
            bot_token = self.config.get('telegram', {}).get('bot_token') or os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = self.config.get('telegram', {}).get('chat_id') or os.getenv("TELEGRAM_CHAT_ID")
            
            # Tạo message header
            current_time = datetime.now().strftime("%H:%M:%S %d/%m/%Y")
            header = f"📊 *BÁO CÁO TỰ ĐỘNG: {symbol}*\n⏱️ *Thời gian:* {current_time}\n\n"
            
            # Gửi báo cáo với header
            success = send_report_to_telegram(
                header + report,
                chat_id=chat_id,
                bot_token=bot_token,
                parse_mode="MARKDOWN"
            )
            
            if success:
                logger.info(f"Đã gửi báo cáo cho {symbol} thành công")
            else:
                logger.error(f"Không thể gửi báo cáo cho {symbol}")
            
            return success
        except Exception as e:
            logger.error(f"Lỗi khi gửi báo cáo: {str(e)}")
            return False
    
    def generate_daily_reports(self):
        """
        Tạo các báo cáo hàng ngày theo cấu hình (sync)
        """
        # Kiểm tra cấu hình báo cáo hàng ngày
        daily_report_config = self.config.get('telegram', {}).get('daily_report', {})
        if not daily_report_config.get('enabled', False):
            logger.info("Báo cáo hàng ngày không được bật trong cấu hình")
            return
        
        # Lấy danh sách mã cần báo cáo
        symbols = daily_report_config.get('symbols', [])
        if not symbols:
            logger.warning("Không có mã nào được cấu hình cho báo cáo hàng ngày")
            return
        
        logger.info(f"Bắt đầu tạo báo cáo hàng ngày cho {len(symbols)} mã: {', '.join(symbols)}")
        
        # Tạo báo cáo cho từng mã
        for symbol in symbols:
            try:
                # Tạo báo cáo
                report_result = self.generate_report_for_symbol(symbol)
                
                if report_result.get("success") and report_result.get("report"):
                    # Gửi báo cáo
                    self.send_report(report_result["report"], symbol)
                else:
                    error_msg = report_result.get("error", "Không rõ lỗi")
                    logger.error(f"Không thể tạo báo cáo cho {symbol}: {error_msg}")
            except Exception as e:
                logger.error(f"Lỗi khi xử lý báo cáo cho {symbol}: {str(e)}", exc_info=True)
        
        logger.info("Hoàn thành tạo báo cáo hàng ngày")

def main():
    """
    Hàm chính để tạo báo cáo theo lịch trình (sync)
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Tạo báo cáo tự động theo lịch trình")
    parser.add_argument("--config", type=str, help="Đường dẫn đến file cấu hình")
    parser.add_argument("--symbol", type=str, help="Mã cổ phiếu cụ thể (nếu không cung cấp, sẽ dùng danh sách trong cấu hình)")
    parser.add_argument("--debug", action="store_true", help="Bật chế độ debug")
    args = parser.parse_args()
    
    # Cấu hình logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Khởi tạo generator
        generator = ScheduledReportGenerator(args.config)
        
        # Nếu có symbol cụ thể, chỉ tạo báo cáo cho mã đó
        if args.symbol:
            symbol = args.symbol.upper()
            logger.info(f"Tạo báo cáo cho mã cụ thể: {symbol}")
            
            report_result = generator.generate_report_for_symbol(symbol)
            
            if report_result.get("success") and report_result.get("report"):
                generator.send_report(report_result["report"], symbol)
                logger.info(f"Đã tạo và gửi báo cáo cho {symbol} thành công")
            else:
                error_msg = report_result.get("error", "Không rõ lỗi")
                logger.error(f"Không thể tạo báo cáo cho {symbol}: {error_msg}")
        else:
            # Tạo báo cáo hàng ngày theo cấu hình
            generator.generate_daily_reports()
        
    except Exception as e:
        logger.error(f"Lỗi không xử lý được: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 