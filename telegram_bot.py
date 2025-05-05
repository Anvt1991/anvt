#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Phiên bản BotChatAI chỉ sử dụng Telegram - không có GUI
"""

import os
import sys
import logging
import json
import time
import asyncio
import traceback
from datetime import datetime
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, Response, HTTPException

# Cấu hình logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

file_handler = logging.FileHandler("logs/telegram_bot.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
logger = logging.getLogger("telegram_bot_main")

# Import các module cần thiết
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackContext
from telegram.constants import ParseMode
from core.data.db import DBManager
from core.data.load_data import DataLoader
from pipeline.processor import PipelineProcessor
from core.report.manager import ReportManager
from core.telegram.notify import send_report_to_telegram

class TelegramOnlyBot:
    """
    Lớp quản lý bot Telegram cho BotChatAI (phiên bản không có GUI)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Khởi tạo bot và các thành phần cần thiết
        
        Args:
            config_path: Đường dẫn đến file cấu hình (nếu có)
        """
        # Tải cấu hình
        self.config = self._load_config(config_path)
        
        # Thiết lập môi trường
        self._setup_environment()
        
        # Lấy token và chat_id từ cấu hình hoặc biến môi trường
        self.bot_token = self.config.get('telegram', {}).get('bot_token') or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = self.config.get('telegram', {}).get('chat_id') or os.getenv("TELEGRAM_CHAT_ID")
        
        # Tự động duyệt admin
        self.ADMIN_ID = os.getenv("ADMIN_ID") or self.config.get('telegram', {}).get('admin_id')
        
        if not self.bot_token:
            logger.error("Không tìm thấy TELEGRAM_BOT_TOKEN! Hãy cung cấp thông qua biến môi trường hoặc file config.")
            sys.exit(1)
        
        # Khởi tạo các thành phần core
        self.db = DBManager()
        self.data_loader = DataLoader()
        self.pipeline = PipelineProcessor()
        self.report_manager = ReportManager(self.db, self.pipeline.gemini_handler)
        
        # Khởi tạo Telegram Bot
        self.bot = Bot(token=self.bot_token)
        self.application = Application.builder().token(self.bot_token).build()
        
        # Đăng ký các lệnh
        self.whitelist_path = os.path.join("cache", "approved_users.json")
        self.approved_users = self._load_whitelist()
        # Tự động duyệt admin nếu chưa có
        if self.ADMIN_ID and self.ADMIN_ID not in self.approved_users:
            self.approved_users.add(self.ADMIN_ID)
            self._save_whitelist()
            logger.info(f"Tự động duyệt admin: {self.ADMIN_ID}")
        self._register_handlers()
        
        logger.info("BotChatAI (phiên bản Telegram only) đã khởi tạo thành công")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Tải cấu hình từ file
        
        Args:
            config_path: Đường dẫn tới file cấu hình
            
        Returns:
            Dict chứa thông tin cấu hình
        """
        # Đường dẫn mặc định
        if not config_path:
            config_path = os.path.join("config", "telegram_config.json")
        
        # Kiểm tra file tồn tại
        if not os.path.exists(config_path):
            logger.warning(f"Không tìm thấy file cấu hình {config_path}, sử dụng cấu hình mặc định")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Đã tải cấu hình từ {config_path}")
            return config
        except Exception as e:
            logger.error(f"Lỗi khi tải cấu hình: {str(e)}")
            return {}
    
    def _setup_environment(self):
        """
        Thiết lập môi trường làm việc
        """
        # Tạo các thư mục cần thiết
        dirs_to_create = [
            "logs",
            "cache",
            "reports",
            "models"
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
        
        # Thiết lập múi giờ
        os.environ['TZ'] = 'Asia/Ho_Chi_Minh'
    
    def _load_whitelist(self):
        if os.path.exists(self.whitelist_path):
            try:
                with open(self.whitelist_path, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def _save_whitelist(self):
        with open(self.whitelist_path, "w", encoding="utf-8") as f:
            json.dump(list(self.approved_users), f)

    def _is_admin(self, update):
        return str(update.effective_chat.id) == str(self.chat_id)

    def _is_approved(self, update):
        return self._is_admin(update) or str(update.effective_user.id) in self.approved_users

    def _register_handlers(self):
        # Thêm lại lệnh start, giữ các lệnh approve, remove, update, analyze, status
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("approve", self.approve_command))
        self.application.add_handler(CommandHandler("remove", self.remove_command))
        self.application.add_handler(CommandHandler("update", self.update_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_approved(update):
            await update.message.reply_text("⛔ Bạn chưa được duyệt sử dụng bot. Vui lòng liên hệ admin.")
            return
        welcome_text = (
            "🤖 *Chào mừng bạn đến với BotChatAI!*\n\n"
            "Bạn đã được duyệt sử dụng bot.\n"
            "\n*Lệnh chính:*\n"
            "/analyze <mã> - Phân tích mã chứng khoán (ví dụ: /analyze FPT)\n"
            "/status - Kiểm tra trạng thái hệ thống\n"
            "/update - Cập nhật dữ liệu (chỉ admin)\n"
            "/approve <user_id> - Duyệt user (chỉ admin)\n"
            "/remove <user_id> - Xóa user khỏi whitelist (chỉ admin)\n"
            "\n_Bot đang trong giai đoạn phát triển, vui lòng báo lỗi nếu phát hiện._"
        )
        await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)

    async def approve_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update):
            await update.message.reply_text("⛔ Bạn không có quyền sử dụng lệnh này.")
            return
        if not context.args:
            await update.message.reply_text("⚠️ Vui lòng nhập user_id cần duyệt. Ví dụ: /approve 123456789")
            return
        user_id = context.args[0]
        self.approved_users.add(user_id)
        self._save_whitelist()
        await update.message.reply_text(f"✅ Đã duyệt user_id: {user_id}")

    async def remove_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update):
            await update.message.reply_text("⛔ Bạn không có quyền sử dụng lệnh này.")
            return
        if not context.args:
            await update.message.reply_text("⚠️ Vui lòng nhập user_id cần xóa. Ví dụ: /remove 123456789")
            return
        user_id = context.args[0]
        if user_id in self.approved_users:
            self.approved_users.remove(user_id)
            self._save_whitelist()
            await update.message.reply_text(f"✅ Đã xóa user_id: {user_id} khỏi whitelist")
        else:
            await update.message.reply_text(f"❌ user_id: {user_id} không có trong whitelist")

    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_approved(update):
            await update.message.reply_text("⛔ Bạn chưa được duyệt sử dụng bot. Vui lòng liên hệ admin.")
            return
        if not context.args:
            await update.message.reply_text("⚠️ Vui lòng nhập mã chứng khoán. Ví dụ: /analyze FPT")
            return
        
        symbol = context.args[0].upper()
        loading_message = await update.message.reply_text(f"⏳ Đang phân tích {symbol}...")
        
        try:
            pipeline_result = self.process_symbol(symbol)
            if pipeline_result and pipeline_result.get("report"):
                await loading_message.edit_text(pipeline_result["report"], parse_mode=ParseMode.MARKDOWN)
            else:
                error_msg = pipeline_result.get("error", "Không rõ lỗi")
                await loading_message.edit_text(f"❌ Không thể phân tích {symbol}: {error_msg}")
        except Exception as e:
            logger.error(f"Lỗi khi phân tích {symbol}: {str(e)}", exc_info=True)
            await loading_message.edit_text(f"❌ Lỗi khi phân tích {symbol}: {str(e)}")
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_approved(update):
            await update.message.reply_text("⛔ Bạn chưa được duyệt sử dụng bot. Vui lòng liên hệ admin.")
            return
        loading_message = await update.message.reply_text("⏳ Đang kiểm tra trạng thái hệ thống...")
        
        try:
            # Kiểm tra kết nối đến các API
            groq_ok = self.pipeline.groq_handler.test_connection()
            gemini_ok = self.pipeline.gemini_handler.test_connection()
            db_ok = self.db.test_connection()
            
            # Lấy thông tin hệ thống
            status_text = f"""
🔍 *TRẠNG THÁI HỆ THỐNG*

📡 *Kết nối API:*
- Groq API: {"✅ OK" if groq_ok else "❌ Lỗi"}
- Gemini API: {"✅ OK" if gemini_ok else "❌ Lỗi"}
- Database: {"✅ OK" if db_ok else "❌ Lỗi"}

🤖 *Thông tin phiên bản:*
- Bot version: {self.config.get('version', '2.0')}
- Groq model: {self.pipeline.groq_handler.model_name}
- Gemini model: {self.pipeline.gemini_handler.model_name}

⏱️ *Cập nhật:* {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
            """
            
            await loading_message.edit_text(status_text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra trạng thái: {str(e)}")
            await loading_message.edit_text(f"❌ Lỗi khi kiểm tra trạng thái: {str(e)}")
    
    async def update_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_admin(update):
            await update.message.reply_text("⛔ Bạn không có quyền sử dụng lệnh này.")
            return
        loading_message = await update.message.reply_text("⏳ Đang cập nhật dữ liệu thị trường...")
        
        try:
            # Cập nhật dữ liệu cho các mã VN30
            symbols = self.data_loader.get_vn30_stocks()
            
            if not symbols:
                await loading_message.edit_text("❌ Không thể lấy danh sách mã VN30.")
                return
            
            # Thêm VNINDEX
            if "VNINDEX" not in symbols:
                symbols.append("VNINDEX")
            
            # Cập nhật từng mã
            success_count = 0
            error_symbols = []
            
            for symbol in symbols:
                try:
                    # Tải lại dữ liệu
                    self.data_loader.load_stock_data(symbol, force_reload=True)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Lỗi khi cập nhật {symbol}: {str(e)}")
                    error_symbols.append(symbol)
            
            # Thông báo kết quả
            status_text = f"""
🔄 *CẬP NHẬT DỮ LIỆU THÀNH CÔNG*

✅ Đã cập nhật: {success_count}/{len(symbols)} mã
{f"❌ Lỗi: {', '.join(error_symbols)}" if error_symbols else ""}

⏱️ *Cập nhật lúc:* {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
            """
            
            await loading_message.edit_text(status_text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật dữ liệu: {str(e)}")
            await loading_message.edit_text(f"❌ Lỗi khi cập nhật dữ liệu: {str(e)}")
    
    def process_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Xử lý phân tích mã cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu cần phân tích
            
        Returns:
            Dict với kết quả phân tích
        """
        try:
            # Sử dụng pipeline processor để phân tích (sync)
            pipeline_data = self.pipeline.process(symbol)
            # Lấy kết quả chuẩn hóa
            if hasattr(pipeline_data, 'to_dict') and callable(getattr(pipeline_data, 'to_dict')):
                result = pipeline_data.to_dict()
            elif hasattr(pipeline_data, 'result'):
                result = pipeline_data.result
            else:
                result = pipeline_data
            # Kiểm tra lỗi
            if not result or (isinstance(result, dict) and result.get('has_error')):
                error_msg = result.get('error', 'Không rõ lỗi') if isinstance(result, dict) else 'Không nhận được dữ liệu'
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
            self.report_manager.save_report(
                symbol,
                report_result["report"],
                aggregated_data.get("current_price", 0),
                aggregated_data.get("previous_price", 0)
            )
            return {"success": True, "report": report_result["report"]}
        except Exception as e:
            logger.error(f"Lỗi khi xử lý {symbol}: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def run_polling(self):
        """
        Chạy bot ở chế độ polling (gọi liên tục đến Telegram API)
        """
        logger.info("Khởi động bot ở chế độ polling...")
        self.application.run_polling()

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="BotChatAI - Telegram only version")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def create_app():
    bot = TelegramOnlyBot()
    app = FastAPI()

    @app.post("/webhook/{bot_token}")
    async def telegram_webhook(request: Request, bot_token: str):
        # Bảo mật: chỉ xử lý nếu token đúng
        if bot_token != bot.bot_token:
            return Response(content="Forbidden", status_code=403)
        # Đảm bảo Application đã được initialize
        if not bot.application._initialized:
            await bot.application.initialize()
        # Đảm bảo Bot đã được initialize
        if not bot.bot._initialized:
            await bot.bot.initialize()
        data = await request.json()
        update = Update.de_json(data, bot.bot)
        await bot.application.process_update(update)
        return Response(content="OK", status_code=200)

    @app.get("/setup")
    async def setup_webhook(secret: str, url: str):
        setup_secret = os.getenv("SETUP_SECRET", "botchatai_secret")
        if secret != setup_secret:
            raise HTTPException(status_code=403, detail="Secret không hợp lệ")
        webhook_url = f"{url}/webhook/{bot.bot_token}"
        result = await bot.bot.set_webhook(webhook_url)
        if result:
            return {"success": True, "message": f"Webhook đã được thiết lập tại {webhook_url}"}
        else:
            raise HTTPException(status_code=500, detail="Không thể thiết lập webhook")

    @app.get("/")
    async def root():
        return {"status": "online", "bot": "BotChatAI Telegram Bot"}

    return app

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    # Set debug mode if specified
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        # Initialize bot (for local dev/testing)
        bot = TelegramOnlyBot(args.config)
        # Thông báo không hỗ trợ polling nữa
        logger.info("Chế độ polling đã bị loại bỏ. Hãy chạy bằng Uvicorn/FastAPI để sử dụng webhook.")
    except Exception as e:
        logger.critical(f"Lỗi nghiêm trọng: {str(e)}", exc_info=True)
        sys.exit(1)

# FastAPI app cho Render/production
app = create_app() 