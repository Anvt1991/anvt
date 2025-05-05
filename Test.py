import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CallbackContext

# Thiết lập logging để theo dõi lỗi
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Hàm xử lý tin nhắn từ kênh
async def get_chat_id(update: Update, context: CallbackContext):
    chat = update.effective_chat
    print(f"Chat Type: {chat.type}")  # Kiểm tra loại chat
    print(f"Chat Title: {chat.title}")  # Kiểm tra tên kênh
    print(f"Chat ID: {chat.id}")  # Lấy ID của kênh

# Hàm chính để khởi động bot

def main():
    # Tạo bot application với token của bạn
    application = Application.builder().token("7780930655:AAEPJ77fbGwtDeCj-jCzVUuA-rZgGxPsuMM").build()

    # Thêm handler để xử lý mọi loại tin nhắn
    application.add_handler(MessageHandler(filters.ALL, get_chat_id))
    application.run_polling()

if __name__ == "__main__":
    main()
