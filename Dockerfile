FROM python:3.9-slim

WORKDIR /app

# Đặt biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Ho_Chi_Minh

# Cài đặt các gói hỗ trợ
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Sao chép requirements trước để tận dụng caching của Docker
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Sao chép mã nguồn
COPY . .

# Tạo thư mục cần thiết
RUN mkdir -p logs cache models reports data

# Port mặc định (nếu cần để healthcheck hoặc API future)
EXPOSE 8080

# Lệnh khởi động
CMD ["python", "telegram_bot.py"] 