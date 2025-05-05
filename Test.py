from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

async def get_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat
    print(f"Chat Type: {chat.type}")  # Kiểm tra loại chat
    print(f"Chat Title: {chat.title}")  # Kiểm tra tên kênh
    print(f"Chat ID: {chat.id}")  # Lấy ID của kênh

async def main():
    # Thay "YOUR_BOT_TOKEN" bằng token thực tế của bot của bạn
    application = Application.builder().token("7780930655:AAEPJ77fbGwtDeCj-jCzVUuA-rZgGxPsuMM").build()

    # Đăng ký handler để bắt tin nhắn
    application.add_handler(MessageHandler(filters.TEXT & filters.ChatType.CHANNEL, get_chat_id))

    # Bắt đầu bot
    await application.run_polling()

# Khởi chạy bot
import asyncio
asyncio.run(main())
