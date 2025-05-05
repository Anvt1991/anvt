# BotChatAI - Phiên bản Telegram Only

BotChatAI là trợ lý phân tích chứng khoán AI tích hợp với Telegram, cho phép bạn dễ dàng nhận được phân tích kỹ thuật, phân tích thị trường và các báo cáo thông qua ứng dụng Telegram.

## Tính năng chính

- **Phân tích kỹ thuật**: Phân tích mã cổ phiếu với chỉ báo kỹ thuật, nhận diện mẫu hình, và đưa ra khuyến nghị.
- **Phân tích thị trường**: Thông tin tổng quan về thị trường VNINDEX.
- **Báo cáo tự động**: Nhận báo cáo tự động cho các mã chứng khoán quan tâm.
- **Truy vấn dễ dàng**: Sử dụng lệnh đơn giản qua Telegram để truy vấn thông tin.
- **Tích hợp AI**: Sử dụng Groq và Gemini để tạo báo cáo chi tiết, dễ hiểu.

## Yêu cầu

- Python 3.8+
- Các thư viện được liệt kê trong `requirements.txt`
- Tài khoản Groq API (hoặc Gemini API)
- Bot Telegram đã tạo qua BotFather

## Cài đặt và thiết lập

1. **Clone repository**:
   ```
   git clone https://github.com/yourusername/botchatai.git
   cd botchatai
   ```

2. **Cài đặt các gói phụ thuộc**:
   ```
   pip install -r requirements.txt
   ```

3. **Thiết lập biến môi trường**:
   Tạo file `.env` với nội dung sau:
   ```
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   ```

4. **Cấu hình Telegram Bot**:
   - Chỉnh sửa file `config/telegram_config.json` để thiết lập bot token, chat ID và người dùng admin.
   - Bạn có thể lấy bot token bằng cách tạo bot mới qua [@BotFather](https://t.me/BotFather).
   - Để lấy chat ID, có thể sử dụng [@userinfobot](https://t.me/userinfobot).

5. **Kiểm tra cài đặt**:
   ```
   python telegram_bot.py --debug
   ```

## Sử dụng

### Các lệnh cơ bản

- `/start` hoặc `/help` - Hiển thị hướng dẫn sử dụng
- `/analyze <mã>` - Phân tích mã cổ phiếu (ví dụ: `/analyze FPT`)
- `/market` - Xem thông tin thị trường hiện tại
- `/symbols` - Hiển thị danh sách mã chứng khoán VN30
- `/history <mã>` - Xem lịch sử phân tích của mã cổ phiếu
- `/status` - Kiểm tra trạng thái hệ thống
- `/update` - Cập nhật dữ liệu thị trường (chỉ admin)

### Cấu hình báo cáo tự động

Để nhận báo cáo tự động hàng ngày, cấu hình trong file `config/telegram_config.json`:

```json
"daily_report": {
    "enabled": true,
    "time": "17:30",
    "symbols": ["VNINDEX", "VN30", "VCB", "VNM", "VHM", "FPT", "HPG"]
}
```

## Chạy ứng dụng

### Chạy trực tiếp

```
python telegram_bot.py
```

### Chạy bằng Docker

1. Xây dựng Docker image:
   ```
   docker build -t botchatai-telegram .
   ```

2. Chạy container:
   ```
   docker run -d --name botchatai -v $(pwd)/config:/app/config -v $(pwd)/data:/app/data botchatai-telegram
   ```

## Giải quyết vấn đề

### Các vấn đề thường gặp

1. **Bot không phản hồi**:
   - Kiểm tra kết nối internet
   - Xác nhận Telegram Bot Token hợp lệ
   - Đảm bảo bot đang chạy với lệnh `ps aux | grep telegram_bot.py`

2. **Lỗi khi phân tích**:
   - Kiểm tra API keys (Groq/Gemini)
   - Đảm bảo đã tải đủ dữ liệu với lệnh `/update`
   - Xem logs trong thư mục `logs/`

3. **Báo cáo tự động không hoạt động**:
   - Kiểm tra cài đặt thời gian trong `telegram_config.json`
   - Đảm bảo bot đang chạy liên tục

## Tài liệu và liên kết hữu ích

- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Groq API Documentation](https://console.groq.com/docs)
- [Gemini API Documentation](https://ai.google.dev/docs)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 