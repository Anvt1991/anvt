# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements nếu có
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn
COPY . .

# Cổng mặc định
EXPOSE 8000

# Lệnh chạy bot
CMD ["python", "Bot_News.py"] 