# Bot News Chứng Khoán - Telegram

## Mục đích
Bot này tự động thu thập, lọc và gửi các tin tức tài chính, kinh tế, chứng khoán, chính trị quan trọng cho người dùng Telegram, ưu tiên tin nóng, loại trừ spam/PR, đảm bảo không bỏ lỡ tin khi bot bị sleep (Render).

## Tính năng nổi bật
- **Lọc tin thông minh:**
  - Chỉ lấy tin trong 2 ngày gần nhất.
  - Lọc theo từ khóa tài chính/chính trị, loại trừ spam/PR.
  - Ưu tiên gửi tin nóng (breaking news).
- **Hàng đợi tin tức:**
  - Tin được quét và lưu vào queue (Redis), gửi dần cho user.
  - Không bỏ lỡ tin khi bot bị cold start.
- **Quản lý từ khóa động:**
  - Người dùng có thể thêm/xem/xóa từ khóa lọc tin.
- **Hệ thống duyệt user:**
  - Chỉ user được admin duyệt mới nhận tin.
- **Tương thích Render, Heroku, VPS...**

## Pipeline tổng quát
1. **Job 1 (mỗi 1h):** Quét toàn bộ RSS, lọc tin thông minh, đẩy vào queue (ưu tiên tin nóng).
2. **Job 2 (mỗi 800s):** Lấy 1 tin từ queue (ưu tiên hot_news_queue), gửi cho user đã duyệt.
3. **Tin đã gửi được đánh dấu, không gửi lại.**

## Hướng dẫn cài đặt
### 1. Yêu cầu
- Python 3.9+
- Redis server
- PostgreSQL
- Tài khoản Telegram Bot (lấy token từ @BotFather)

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường
Tạo file `.env` hoặc cấu hình trực tiếp trên Render:
```
BOT_TOKEN=...                # Token Telegram Bot
WEBHOOK_URL=...              # (nếu dùng webhook)
REDIS_URL=redis://...        # URL Redis
DATABASE_URL=postgresql://...# URL PostgreSQL
ADMIN_ID=...                 # Telegram user id của admin
REDIS_TTL=21600              # (tuỳ chọn) TTL cho cache (giây)
NEWS_JOB_INTERVAL=800        # (tuỳ chọn) Chu kỳ gửi tin (giây)
HOURLY_JOB_INTERVAL=3600     # (tuỳ chọn) Chu kỳ quét RSS (giây)
FETCH_LIMIT_DAYS=2           # (tuỳ chọn) Chỉ lấy tin trong N ngày gần nhất
```

### 4. Khởi động bot
```bash
python Bot_News.py
```

## Các lệnh chính trên Telegram
- `/start` - Khởi động bot, nhận hướng dẫn
- `/help` - Xem hướng dẫn sử dụng
- `/register` - Đăng ký sử dụng bot (chờ admin duyệt)
- `/keywords` - Xem danh sách từ khóa lọc tin
- `/set_keywords <từ khóa>` - Thêm từ khóa bổ sung (cách nhau bởi dấu phẩy)
- `/clear_keywords` - Xóa toàn bộ từ khóa bổ sung

## Quản lý user
- User mới phải đăng ký và được admin duyệt (qua nút bấm trên Telegram).
- Admin có thể duyệt/từ chối user trực tiếp trên Telegram.

## Lưu ý vận hành trên Render/Heroku
- **Bot sẽ tự động khởi động lại và không bỏ lỡ tin khi bị sleep.**
- Nên dùng Redis và PostgreSQL bản cloud hoặc add-on.
- Nếu dùng webhook, cần cấu hình đúng `WEBHOOK_URL` và mở port phù hợp.

## Tuỳ chỉnh nâng cao
- Sửa các biến trong class `Config` để thay đổi chu kỳ, từ khóa mặc định, v.v.
- Có thể mở rộng thêm các RSS feed trong `FEED_URLS`.

## Đóng góp & liên hệ
- Mọi ý kiến đóng góp, báo lỗi xin gửi về [admin Telegram](https://t.me/your_admin_username)

---
**Copyright © 2024** 