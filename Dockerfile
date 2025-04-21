# Dựa trên Python image
FROM python:3.10-slim

# Cài đặt thư mục làm việc
WORKDIR /app

# Sao chép các file
COPY . /app

# Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# Cổng mà bot sẽ chạy
ENV PORT=10000

# Chạy ứng dụng
CMD ["python", "main.py"]
