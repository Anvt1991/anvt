#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bot Telegram cho BotChatAI sử dụng webhook.
"""

import os
import logging
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime

from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackContext
from telegram.constants import ParseMode
from fastapi import FastAPI, Request, Response, HTTPException, Depends

# Import module BotChatAI
from pipeline.processor import PipelineProcessor
from core.data.load_data import DataLoader
from core.data.db import DBManager

# Cấu hình logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class BotChatAITelegramBot:
    """
    Bot Telegram cho BotChatAI với các lệnh phân tích chứng khoán.
    Hỗ trợ các lệnh: /analyze, /market, /symbols, /history, /help
    """

    def __init__(self, bot_token: str = None, setup_secret: str = None):
        """
        Khởi tạo Bot Telegram BotChatAI

        Args:
            bot_token: Telegram Bot Token, nếu None sẽ lấy từ biến môi trường TELEGRAM_BOT_TOKEN
            setup_secret: Secret để thiết lập webhook, nếu None sẽ lấy từ biến môi trường SETUP_SECRET
        """
        self.token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        if not self.token:
            logger.error("Không tìm thấy TELEGRAM_BOT_TOKEN trong biến môi trường!")
            raise ValueError("TELEGRAM_BOT_TOKEN không được cung cấp")

        self.setup_secret = setup_secret or os.environ.get("SETUP_SECRET", "botchatai_secret")
        
        # Khởi tạo các thành phần cần thiết
        self.pipeline = PipelineProcessor()
        self.data_loader = DataLoader()
        self.db = DBManager()
        
        # Khởi tạo bot và application
        self.bot = Bot(token=self.token)
        self.application = Application.builder().token(self.token).build()
        
        # Đăng ký các command handler
        self._register_handlers()
        
        # Tạo FastAPI app
        self.app = FastAPI(title="BotChatAI Telegram Bot")
        self._setup_routes()
    
    def _register_handlers(self):
        """Đăng ký các command handler cho bot"""
        self.application.add_handler(CommandHandler("start", self.help_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("market", self.market_command))
        self.application.add_handler(CommandHandler("symbols", self.symbols_command))
        self.application.add_handler(CommandHandler("history", self.history_command))
    
    def _setup_routes(self):
        """Thiết lập các routes cho FastAPI app"""
        
        @self.app.post(f"/webhook/{self.token}")
        async def telegram_webhook(request: Request):
            """Xử lý webhook từ Telegram"""
            try:
                data = await request.json()
                # Chuyển JSON thành update object
                update = Update.de_json(data, self.bot)
                # Xử lý update
                await self.application.process_update(update)
                return Response(content="OK", status_code=200)
            except Exception as e:
                logger.error(f"Lỗi khi xử lý webhook: {str(e)}")
                return Response(content=f"Error: {str(e)}", status_code=500)
        
        @self.app.get("/setup")
        async def setup_webhook_route(secret: str, url: str):
            """Thiết lập webhook URL cho bot"""
            if secret != self.setup_secret:
                raise HTTPException(status_code=403, detail="Secret không hợp lệ")
            
            webhook_url = f"{url}/webhook/{self.token}"
            result = await self.bot.set_webhook(webhook_url)
            
            if result:
                return {
                    "success": True, 
                    "message": f"Webhook đã được thiết lập tại {webhook_url}"
                }
            else:
                raise HTTPException(status_code=500, detail="Không thể thiết lập webhook")
        
        @self.app.get("/")
        async def root():
            """Endpoint mặc định cho health check"""
            return {"status": "online", "bot": "BotChatAI Telegram Bot"}
    
    def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Xử lý lệnh /help hoặc /start (sync)
        """
        help_text = """
🤖 *BotChatAI - Trợ lý phân tích chứng khoán*

*Các lệnh có sẵn:*
/analyze <mã> - Phân tích mã cụ thể (vd: /analyze FPT)
/market - Thông tin thị trường hiện tại
/symbols - Danh sách mã chứng khoán
/history <mã> - Lịch sử phân tích mã
/status - Kiểm tra trạng thái hệ thống
/help - Hiển thị trợ giúp này

_Bot đang trong giai đoạn phát triển, vui lòng báo lỗi nếu phát hiện._
"""
        update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)

    def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Xử lý lệnh /analyze <mã> (sync)
        """
        if not context.args:
            update.message.reply_text("⚠️ Vui lòng nhập mã chứng khoán. Ví dụ: /analyze FPT")
            return
        symbol = context.args[0].upper()
        loading_message = update.message.reply_text(f"⏳ Đang phân tích {symbol}...")
        try:
            pipeline_result = self.process_symbol(symbol)
            if pipeline_result and pipeline_result.get("report"):
                loading_message.edit_text(pipeline_result["report"], parse_mode=ParseMode.MARKDOWN)
            else:
                error_msg = pipeline_result.get("error", "Không rõ lỗi")
                loading_message.edit_text(f"❌ Không thể phân tích {symbol}: {error_msg}")
        except Exception as e:
            logger.error(f"Lỗi khi phân tích {symbol}: {str(e)}", exc_info=True)
            loading_message.edit_text(f"❌ Lỗi khi phân tích {symbol}: {str(e)}")

    def market_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Hiển thị thông tin thị trường khi người dùng nhắn /market (sync)
        """
        loading_message = update.message.reply_text("⏳ Đang lấy thông tin thị trường...")
        try:
            market_info = self.pipeline.get_market_condition()
            text = "📊 *THÔNG TIN THỊ TRƯỜNG*\n\n"
            if market_info:
                vnindex = market_info.get("vnindex_last", 0)
                change = market_info.get("vnindex_change", 0)
                change_pct = market_info.get("vnindex_change_pct", 0)
                sentiment = market_info.get("market_sentiment", {})
                trend = market_info.get("market_trend", "N/A")
                text += f"📈 *VN-Index:* {vnindex:,.2f} ({'+' if change >= 0 else ''}{change:,.2f} | {change_pct:.2f}%)\n"
                text += f"📉 *Xu hướng:* {trend}\n"
                if sentiment:
                    score = sentiment.get("score", 0)
                    label = sentiment.get("label", "trung tính")
                    text += f"🤔 *Tâm lý thị trường:* {label} (score: {score:.2f})\n\n"
                text += f"⏱️ *Cập nhật:* {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
            else:
                text += "❌ Không thể lấy thông tin thị trường."
            loading_message.edit_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Lỗi khi lấy thông tin thị trường: {str(e)}")
            loading_message.edit_text(f"❌ Lỗi khi lấy thông tin thị trường: {str(e)}")

    def symbols_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Hiển thị danh sách mã chứng khoán khi người dùng nhắn /symbols (sync)
        """
        try:
            vn30_symbols = self.data_loader.get_vn30_stocks()
            if vn30_symbols:
                text = "📋 *DANH SÁCH MÃ VN30*\n\n"
                chunks = [vn30_symbols[i:i+5] for i in range(0, len(vn30_symbols), 5)]
                for chunk in chunks:
                    text += " - ".join(chunk) + "\n"
                text += "\n💡 *Cách dùng:* /analyze <mã> để phân tích mã cụ thể"
            else:
                text = "❌ Không thể lấy danh sách mã."
            update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Lỗi khi lấy danh sách mã: {str(e)}")
            update.message.reply_text(f"❌ Lỗi khi lấy danh sách mã: {str(e)}")

    def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Hiển thị lịch sử phân tích mã khi người dùng nhắn /history <mã> (sync)
        """
        if not context.args:
            update.message.reply_text("⚠️ Vui lòng nhập mã chứng khoán. Ví dụ: /history FPT")
            return
        symbol = context.args[0].upper()
        loading_message = update.message.reply_text(f"⏳ Đang lấy lịch sử phân tích {symbol}...")
        try:
            history = self.db.load_report_history(symbol)
            if history and len(history) > 0:
                text = f"📜 *LỊCH SỬ PHÂN TÍCH {symbol}*\n\n"
                for i, report in enumerate(history[:5]):
                    date = report.get("date", "N/A")
                    close = report.get("close_today", 0)
                    prev_close = report.get("close_yesterday", 0)
                    change = close - prev_close
                    change_pct = (change / prev_close * 100) if prev_close else 0
                    text += f"📊 *Báo cáo {i+1}:* {date}\n"
                    text += f"💰 Giá: {close:,.2f} ({'+' if change >= 0 else ''}{change:,.2f} | {change_pct:.2f}%)\n"
                    text += "---------------------\n"
                text += f"\n💡 Sử dụng /analyze {symbol} để phân tích mới nhất"
            else:
                text = f"❌ Không có lịch sử phân tích cho {symbol}."
            loading_message.edit_text(text, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            logger.error(f"Lỗi khi lấy lịch sử phân tích {symbol}: {str(e)}")
            loading_message.edit_text(f"❌ Lỗi khi lấy lịch sử phân tích {symbol}: {str(e)}")
    
    def format_analysis_result(self, result: Dict[str, Any]) -> str:
        """Format kết quả phân tích thành văn bản Telegram"""
        if not result or "error" in result:
            return f"❌ Lỗi khi phân tích: {result.get('error', 'Không rõ')}"
        
        symbol = result.get("symbol", "???")
        last_price = result.get("last_price", 0)
        prev_price = result.get("previous_price", 0)
        
        change = last_price - prev_price
        change_pct = (change / prev_price * 100) if prev_price else 0
        
        # Lấy các phân tích
        tech_analysis = result.get("technical_analysis", {})
        ai_analysis = result.get("ai_analysis", {})
        prediction = result.get("prediction", {})
        
        # Tổng hợp tín hiệu kỹ thuật
        signals = tech_analysis.get("signals", {})
        bullish = sum(1 for s in signals.values() if s == "bullish")
        bearish = sum(1 for s in signals.values() if s == "bearish")
        neutral = sum(1 for s in signals.values() if s == "neutral")
        
        # Tạo văn bản phân tích
        text = f"📊 *Phân tích {symbol}*\n\n"
        text += f"💰 *Giá hiện tại:* {last_price:,.2f} ({'+' if change >= 0 else ''}{change:,.2f} | {change_pct:.2f}%)\n\n"
        
        text += f"📈 *Tín hiệu kỹ thuật:* {bullish} tăng | {bearish} giảm | {neutral} trung tính\n"
        
        # Xu hướng
        trend = tech_analysis.get("trend_analysis", {}).get("trend", "N/A")
        text += f"📉 *Xu hướng:* {trend}\n\n"
        
        # Phân tích AI
        if ai_analysis:
            ai_summary = ai_analysis.get("summary", "Không có thông tin")
            text += f"🤖 *Phân tích AI:*\n{ai_summary}\n\n"
        
        # Dự đoán
        if prediction:
            pred_direction = prediction.get("direction", "?")
            pred_confidence = prediction.get("confidence", 0) * 100
            text += f"🔮 *Dự đoán:* {pred_direction} (độ tin cậy: {pred_confidence:.1f}%)\n\n"
        
        # Hỗ trợ và kháng cự
        sr = tech_analysis.get("support_resistance", {})
        supports = sr.get("support_levels", [])
        resistances = sr.get("resistance_levels", [])
        
        if supports:
            text += f"⬇️ *Hỗ trợ:* {', '.join([f'{s:,.2f}' for s in supports[:3]])}\n"
        if resistances:
            text += f"⬆️ *Kháng cự:* {', '.join([f'{r:,.2f}' for r in resistances[:3]])}\n\n"
        
        # Thêm thời gian phân tích
        text += f"⏱️ *Cập nhật:* {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
        
        return text

async def setup_webhook(bot: Bot, webhook_url: str) -> bool:
    """
    Thiết lập webhook URL cho bot
    
    Args:
        bot: Bot Telegram 
        webhook_url: URL webhook (không bao gồm /webhook/{token})
        
    Returns:
        True nếu thành công, False nếu thất bại
    """
    token = bot.token
    full_url = f"{webhook_url}/webhook/{token}"
    try:
        result = await bot.set_webhook(full_url)
        if result:
            logger.info(f"Webhook đã được thiết lập tại {full_url}")
            return True
        logger.error(f"Không thể thiết lập webhook tại {full_url}")
        return False
    except Exception as e:
        logger.error(f"Lỗi khi thiết lập webhook: {str(e)}")
        return False

def start_bot(host: str = "0.0.0.0", port: int = 10000):
    """
    Khởi động bot Telegram với FastAPI server
    
    Args:
        host: Host để binding server
        port: Port để binding server
    """
    import uvicorn
    
    try:
        # Khởi tạo bot
        bot = BotChatAITelegramBot()
        
        # Chạy FastAPI
        logger.info(f"Khởi động BotChatAI Telegram Bot trên {host}:{port}")
        uvicorn.run(bot.app, host=host, port=port)
    except Exception as e:
        logger.error(f"Lỗi khi khởi động bot: {str(e)}")
        raise 