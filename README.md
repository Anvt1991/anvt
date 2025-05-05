# Bot News Gemini

Bot Telegram tự động lấy tin tức tài chính, kinh tế, vĩ mô, chứng khoán (tiếng Việt & tiếng Anh) từ Google News RSS, phân tích bằng AI Gemini, dịch tự động sang tiếng Việt (nếu cần), lưu vào PostgreSQL và gửi cho user theo từ khóa quan tâm.

## Tính năng nổi bật
- Lấy tin từ nhiều nguồn Google News RSS (chủ đề: kinh tế, chứng khoán, vĩ mô, chiến tranh, lãi suất, Fed, ...)
- Phân tích, tóm tắt, nhận định, cảm xúc bằng AI Gemini (Google)
- Dịch tự động nhận định AI sang tiếng Việt nếu tin gốc là tiếng Anh (LibreTranslate)
- Lưu trữ tin và phân tích vào PostgreSQL
- Chỉ gửi tin cho user đã định sẵn và theo từ khóa quan tâm
- Chống gửi trùng tin nhờ Redis
- Chạy định kỳ (14 phút/lần)

## Yêu cầu hệ thống
- Python 3.9+
- PostgreSQL
- Redis

## Cài đặt
```bash
# Clone code
 git clone <repo-url>
 cd <thư mục bot>

# Cài đặt thư viện
pip install -r requirements.txt
```

## Cấu hình biến môi trường (.env)
Tạo file `.env` với nội dung mẫu:
```env
BOT_TOKEN=token_telegram_bot
WEBHOOK_URL=https://your-domain.com/webhook
REDIS_URL=redis://localhost
DATABASE_URL=postgresql://user:password@host:port/dbname
OPENROUTER_API_KEY=your_openrouter_api_key
CHANNEL_ID=-1001234567890  # Thay bằng chat_id của channel Telegram
```

## Cấu hình user và từ khóa nhận tin
Mở file `Bot_News.py`, sửa biến sau:
```python
user_keywords = {
    123456789: ["bitcoin", "vn-index"],  # Thay 123456789 bằng user_id thực tế
    # Thêm user_id và từ khóa khác nếu cần
}
```

## Khởi tạo database
Tạo bảng PostgreSQL:
```sql
CREATE TABLE news_insights (
    id SERIAL PRIMARY KEY,
    title TEXT,
    link TEXT UNIQUE,
    summary TEXT,
    sentiment TEXT,
    ai_opinion TEXT
);
```

## Quy trình gửi tin
- Bot sẽ tự động gửi tất cả tin tức phù hợp vào channel Telegram đã cấu hình.
- Không cần đăng ký user, không cần duyệt admin.

## Gửi tin vào channel Telegram
- Tạo channel mới trên Telegram (public hoặc private).
- Thêm bot vào channel với quyền admin (ít nhất là gửi tin nhắn).
- Lấy chat_id của channel:
    - Channel public: dùng @username (ví dụ: @tenkenh)
    - Channel private: forward 1 tin nhắn từ channel sang @userinfobot để lấy chat_id (dạng -100xxxxxxxxxx)
- Sửa biến `CHANNEL_ID` trong file `.env`.
- Mọi tin tức sẽ được gửi vào channel này, không cần duyệt user.

## Chạy bot
```bash
python Bot_News.py
```

## Ghi chú
- Để lấy user_id Telegram: nhắn tin cho @userinfobot trên Telegram.
- Có thể chỉnh sửa danh sách nguồn tin trong biến `FEED_URLS` ở đầu file `Bot_News.py`.
- Nếu muốn đổi chu kỳ quét tin, sửa số phút trong dòng `await asyncio.sleep(14 * 60)`.
- Bot không nhận lệnh từ user, chỉ gửi tin tự động.

## Phụ thuộc
- aiogram==3.4
- aiohttp
- feedparser
- asyncpg
- aioredis
- httpx
- python-dotenv

## Bản quyền
- Chỉ sử dụng cho mục đích cá nhân/nghiên cứu. Không thương mại hóa khi chưa xin phép các nguồn tin và API liên quan. 