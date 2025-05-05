# BotChatAI - Tư vấn Chứng khoán

BotChatAI là một ứng dụng chatbot phân tích và tư vấn chứng khoán sử dụng các công nghệ AI tiên tiến từ Groq và Gemini. Ứng dụng cung cấp khả năng phân tích kỹ thuật, phân tích mẫu hình giá, phân tích tin tức và dự đoán giá chứng khoán.

## Tính năng

- Phân tích kỹ thuật chứng khoán
- Phân tích mẫu hình giá sử dụng AI (Groq)
- Tạo báo cáo phân tích chi tiết (Gemini)
- Phân tích tin tức và đánh giá tâm lý thị trường
- Dự đoán giá và tạo khuyến nghị đầu tư
- Đề xuất danh mục đầu tư tối ưu
- Lưu trữ và theo dõi lịch sử phân tích
- API để tích hợp với các hệ thống khác

## Kiến trúc hệ thống

### Sơ đồ pipeline dạng text dễ hiểu

Bắt đầu từ API hoặc giao diện người dùng, dữ liệu sẽ đi qua các bước sau:

```
BẮT ĐẦU (API/GUI gọi)
   |
   v
1. Tải dữ liệu
   (DataLoader - core/data/load_data.py)
   → Lấy dữ liệu giá chứng khoán từ nhiều nguồn (vnstock, yfinance).
   |
   v
2. Làm sạch dữ liệu
   (DataValidator - core/data/data_validator.py)
   → Kiểm tra, chuẩn hóa, loại bỏ lỗi và giá trị ngoại lệ trong dữ liệu.
   |
   v
3. Tạo đặc trưng
   (FeatureEngineer - core/feature_engineering.py)
   → Sinh các chỉ số, đặc trưng phục vụ cho phân tích kỹ thuật và AI.
   |
   v
4. Phân tích kỹ thuật
   (TechnicalAnalyzer - core/technical.py)
   → Tính toán các chỉ báo kỹ thuật như RSI, MACD, Bollinger Bands, v.v.
   |
   v
5. Phân tích tin tức
   (NewsLoader - core/news/news.py)
   → Lấy tin tức liên quan, phân tích sentiment và tác động đến cổ phiếu.
   |
   v
6. Phân tích AI
   (AIAnalyzer - core/ai/ai_analyzer.py, ai_interface.py, groq.py, gemini.py)
   → Sử dụng AI (Groq, Gemini) để phân tích nâng cao, tổng hợp tín hiệu.
   |
   v
7. Dự đoán & Khuyến nghị
   (EnhancedPredictor - core/ai/enhanced_predictor.py, ml_model.py)
   → Kết hợp các kết quả phân tích để dự đoán xu hướng, đưa ra khuyến nghị đầu tư.
   |
   v
8. Backtest chiến lược
   (Backtester - core/backtester.py)
   → Kiểm thử hiệu quả các chiến lược đầu tư trên dữ liệu lịch sử.
   |
   v
10. Sinh báo cáo
    (ReportGenerator - pipeline/report_generator.py)
    → Xuất báo cáo tổng hợp kết quả phân tích.
   |
   v
11. Lưu kết quả
    (DBManager - core/db.py)
    → Lưu trữ kết quả, lịch sử, báo cáo vào cơ sở dữ liệu.
   |
   v
KẾT THÚC
```

### Sơ đồ pipeline chi tiết, dễ hiểu

```mermaid
flowchart TD
    Start([Bắt đầu<br/>(API/GUI gọi)])
    step1([1. Tải dữ liệu<br/>DataLoader<br/>(core/data/load_data.py)])
    step2([2. Làm sạch dữ liệu<br/>DataValidator<br/>(core/data/data_validator.py)])
    step3([3. Tạo đặc trưng<br/>FeatureEngineer<br/>(core/feature_engineering.py)])
    step4([4. Phân tích kỹ thuật<br/>TechnicalAnalyzer<br/>(core/technical.py)])
    step5([5. Phân tích tin tức<br/>NewsLoader<br/>(core/news/news.py)])
    step6([6. Phân tích AI<br/>AIAnalyzer<br/>(core/ai/ai_analyzer.py, ai_interface.py, groq.py, gemini.py)])
    step7([7. Dự đoán & Khuyến nghị<br/>EnhancedPredictor<br/>(core/ai/enhanced_predictor.py, ml_model.py)])
    step9([9. Backtest chiến lược<br/>Backtester<br/>(core/backtester.py)])
    step10([10. Sinh báo cáo<br/>ReportGenerator<br/>(pipeline/report_generator.py)])
    step11([11. Lưu kết quả<br/>DBManager<br/>(core/db.py)])
    End([Kết thúc])

    Start --> step1
    step1 --> step2
    step2 --> step3
    step3 --> step4
    step4 --> step5
    step5 --> step6
    step6 --> step7
    step7 --> step9
    step9 --> step10
    step10 --> step11
    step11 --> End
```

**Chú thích:**
- Màu xanh: các module pipeline (điều phối, chuẩn hóa, tiện ích, báo cáo).
- Màu vàng: các module core (xử lý nghiệp vụ từng bước).
- Mũi tên thể hiện luồng dữ liệu và điều phối giữa các module.

Các mô tả chi tiết về module:

- **app/**: Các entrypoint chạy ứng dụng (GUI, API, legacy)
  - `main.py`: Entrypoint giao diện người dùng (Tkinter GUI)
  - `run_api.py`: Entrypoint chạy API server (uvicorn)
  - `bot_api.py`: Định nghĩa FastAPI app và các endpoint
  - `app_legacy.py`, `main_legacy.py`: Giao diện và entrypoint legacy (Streamlit)
- **core/**: Các module lõi xử lý nghiệp vụ
  - `ai_analyzer.py`: Phân tích AI truyền thống cho cổ phiếu
  - `ai_interface.py`: Giao tiếp với các dịch vụ AI (Groq, Gemini)
  - `backtester.py`: Kiểm thử hiệu suất chiến lược đầu tư
  - `chatbot.py`: Logic chính của chatbot AI
  - `chatbot_legacy.py`: Chatbot cho giao diện legacy
  - `data.py`: Module chuyển tiếp, tương thích ngược cho DataLoader
  - `data_validator.py`: Kiểm tra, làm sạch, chuẩn hóa dữ liệu
  - `db.py`: Quản lý cơ sở dữ liệu
  - `enhanced_predictor.py`: Dự đoán nâng cao kết hợp ML, kỹ thuật, sentiment
  - `feature_engineering.py`: Xử lý, tạo đặc trưng dữ liệu
  - `gemini.py`, `groq.py`: Tích hợp AI Gemini và Groq
  - `load_data.py`: Module chính tải dữ liệu chứng khoán từ nhiều nguồn
  - `logging_config.py`: Cấu hình logging
  - `ml_model.py`: Xây dựng, huấn luyện, dự đoán mô hình ML
  - `news.py`: Tải và phân tích tin tức
  - `portfolio.py`: Tối ưu hóa danh mục đầu tư
  - `strategy.py`: Tối ưu hóa chiến lược đầu tư
  - `technical.py`: Phân tích kỹ thuật (RSI, MACD, Bollinger Bands...)
- **pipeline/**: Xây dựng pipeline phân tích chuẩn hóa
  - `analyze.py`: Pipeline chuẩn hóa cho phân tích cổ phiếu
  - `interfaces.py`: Định nghĩa dataclass, interface cho pipeline
  - `processor.py`: Xử lý pipeline, tích hợp các bước
  - `report_generator.py`: Sinh báo cáo phân tích, xuất file
  - `utils.py`: Tiện ích hỗ trợ pipeline
- **tests/**: Thư mục kiểm thử các module
  - `test_data_loader.py`, `test_data_validator.py`, `test_pipeline.py`, ...
  - `example.py`: Ví dụ sử dụng DataLoader
- **config/**: Thư mục cấu hình
  - `default_config.json`: Cấu hình mặc định cho hệ thống
- **models/**: Thư mục chứa mô hình đã train (ML, scaler...)
- **data/**: Thư mục chứa dữ liệu mẫu
- **cache/**: Cache dữ liệu tải về
- **requirements.txt**: Thư viện cần thiết
- **README.md**: Tài liệu tổng hợp

## Cài đặt

### Yêu cầu

- Python 3.8+
- Các thư viện Python (xem `requirements.txt`)
- API keys cho Groq và Gemini (tạo file `.env`)

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Cấu hình

Tạo file `.env` với nội dung sau:

```
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Sử dụng

### Chạy ứng dụng chatbot (GUI)

```bash
python -m app.main
```

### Chạy API server

```bash
python -m app.run_api
```

hoặc với các tham số:

```bash
python -m app.run_api --host 127.0.0.1 --port 8888 --reload
```

## API Endpoints

API của BotChatAI cung cấp các endpoints sau:

- `GET /`: Thông tin API
- `POST /api/analyze`: Phân tích mã chứng khoán
- `GET /api/symbols`: Danh sách mã chứng khoán
- `GET /api/market`: Tình hình thị trường
- `GET /api/history/{symbol}`: Lịch sử phân tích cho mã chứng khoán
- `POST /api/train_model`: Huấn luyện lại mô hình ML cho một mã cổ phiếu
- `POST /api/optimize_model`: Tối ưu hyperparameter cho mô hình ML
- `POST /api/backtest`: Backtest chiến lược đầu tư cho một mã cổ phiếu
- `POST /api/data_validation`: Kiểm thử chất lượng dữ liệu cho một mã cổ phiếu

### Ví dụ gọi API

Phân tích mã chứng khoán:

```bash
curl -X POST "http://localhost:8000/api/analyze" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "FPT", "period": "short", "reload": false}'
```

Huấn luyện lại mô hình ML:

```bash
curl -X POST "http://localhost:8000/api/train_model" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "FPT", "timeframe": "1D", "num_candles": 200}'
```

Tối ưu hyperparameter cho mô hình ML:

```bash
curl -X POST "http://localhost:8000/api/optimize_model" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "FPT", "timeframe": "1D", "num_candles": 200}'
```

Backtest chiến lược đầu tư:

```bash
curl -X POST "http://localhost:8000/api/backtest" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "FPT", "timeframe": "1D", "num_candles": 200}'
```

Kiểm thử chất lượng dữ liệu:

```bash
curl -X POST "http://localhost:8000/api/data_validation" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "FPT", "timeframe": "1D", "num_candles": 200}'
```

## Pipeline phân tích

Pipeline phân tích của BotChatAI bao gồm các bước:

1. **Load data**: Tải dữ liệu giá cho mã chứng khoán
2. **Data validation**: Kiểm tra, làm sạch dữ liệu
3. **Feature engineering**: Tạo đặc trưng dữ liệu
4. **Technical analysis**: Phân tích kỹ thuật (RSI, MACD, Bollinger Bands...)
5. **News analysis**: Phân tích tin tức và tâm lý thị trường
6. **AI analysis**: Phân tích sử dụng AI (Groq & Gemini)
7. **Backtest**: Kiểm thử hiệu suất chiến lược
9. **Report generation**: Sinh báo cáo phân tích
10. **Save to database**: Lưu kết quả phân tích vào database

## Đóng góp

Đóng góp cho dự án rất được hoan nghênh. Vui lòng tạo Pull Request hoặc mở Issue để đóng góp.

## Giấy phép

Dự án được cấp phép theo [MIT License](LICENSE).