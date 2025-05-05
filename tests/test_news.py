import pytest
from core.news.news import NewsLoader

def test_analyze_sentiment_basic():
    loader = NewsLoader()
    news_list = [
        {"title": "VN-Index tăng mạnh", "summary": "Thị trường chứng khoán khởi sắc", "source": "cafef.vn"},
        {"title": "Cổ phiếu giảm sâu", "summary": "Nhà đầu tư lo ngại rủi ro", "source": "vnexpress.net"},
        {"title": "Thị trường ổn định", "summary": "Không có biến động lớn", "source": "cafef.vn"},
        {"title": "Cổ phiếu A vừa tăng vừa giảm", "summary": "Tích cực nhưng cũng có rủi ro", "source": "cafef.vn"},
    ]
    result = loader.analyze_sentiment(news_list)
    assert "overall" in result
    assert "positive" in result
    assert "negative" in result
    assert "mixed" in result
    assert "neutral" in result
    assert len(result["details"]) == 4
    # Kiểm tra nhãn mixed xuất hiện
    assert any(d["sentiment"] == "Mixed" for d in result["details"])

# Có thể bổ sung test cho fetch_rss bằng cách mock aiohttp nếu cần 