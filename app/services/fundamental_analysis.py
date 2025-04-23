import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def deep_fundamental_analysis(fundamental_data: dict) -> str:
    """
    Analyze fundamental data and provide insights
    """
    if not fundamental_data or "error" in fundamental_data:
        return "Không có đủ dữ liệu cơ bản để phân tích."
    
    analysis = []
    
    # Extract key metrics
    symbol = fundamental_data.get("symbol", "")
    sector = fundamental_data.get("sector", "Unknown")
    industry = fundamental_data.get("industry", "Unknown")
    market_cap = fundamental_data.get("market_cap")
    pe_ratio = fundamental_data.get("pe_ratio")
    pb_ratio = fundamental_data.get("pb_ratio")
    dividend_yield = fundamental_data.get("dividend_yield")
    
    # General company information
    analysis.append(f"**Phân tích cơ bản {symbol}**")
    if sector and sector != "Unknown":
        analysis.append(f"- Ngành: {sector}")
    if industry and industry != "Unknown":
        analysis.append(f"- Lĩnh vực: {industry}")
    
    # Market cap analysis
    if market_cap:
        market_cap_formatted = format_market_cap(market_cap)
        analysis.append(f"- Vốn hóa thị trường: {market_cap_formatted}")
        
        # Market cap size classification
        if market_cap >= 10_000_000_000:  # $10B+
            analysis.append("- Phân loại: Cổ phiếu vốn hóa lớn (Large Cap)")
        elif market_cap >= 2_000_000_000:  # $2B-$10B
            analysis.append("- Phân loại: Cổ phiếu vốn hóa trung bình (Mid Cap)")
        else:
            analysis.append("- Phân loại: Cổ phiếu vốn hóa nhỏ (Small Cap)")
    
    # P/E analysis
    if pe_ratio:
        analysis.append(f"- Chỉ số P/E: {pe_ratio:.2f}")
        
        if pe_ratio < 0:
            analysis.append("  → Công ty đang lỗ (P/E âm)")
        elif pe_ratio < 10:
            analysis.append("  → P/E thấp, có thể là cổ phiếu giá trị hoặc có vấn đề về triển vọng tăng trưởng")
        elif pe_ratio < 20:
            analysis.append("  → P/E ở mức hợp lý")
        elif pe_ratio < 50:
            analysis.append("  → P/E cao, cổ phiếu có thể được định giá dựa trên kỳ vọng tăng trưởng")
        else:
            analysis.append("  → P/E rất cao, tiềm ẩn rủi ro định giá quá cao")
    
    # P/B analysis
    if pb_ratio:
        analysis.append(f"- Chỉ số P/B: {pb_ratio:.2f}")
        
        if pb_ratio < 1:
            analysis.append("  → P/B dưới 1, cổ phiếu có thể đang được giao dịch dưới giá trị sổ sách")
        elif pb_ratio < 3:
            analysis.append("  → P/B ở mức hợp lý")
        else:
            analysis.append("  → P/B cao, công ty có thể có ROE cao hoặc được định giá cao")
    
    # Dividend analysis
    if dividend_yield:
        dividend_yield_percent = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
        analysis.append(f"- Tỷ suất cổ tức: {dividend_yield_percent:.2f}%")
        
        if dividend_yield_percent > 5:
            analysis.append("  → Tỷ suất cổ tức cao, có thể phù hợp cho đầu tư thu nhập")
        elif dividend_yield_percent > 2:
            analysis.append("  → Tỷ suất cổ tức ở mức hợp lý")
        elif dividend_yield_percent > 0:
            analysis.append("  → Tỷ suất cổ tức thấp")
        else:
            analysis.append("  → Không trả cổ tức")
    
    # Business summary if available
    summary = fundamental_data.get("summary", "")
    if summary and len(summary) > 20:
        # Truncate summary if too long
        max_length = 300
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        analysis.append(f"\n**Tóm tắt công ty:**\n{summary}")
    
    # Conclusion
    analysis.append("\n**Nhận định:**")
    
    # Generate some basic conclusions based on the metrics
    conclusions = generate_conclusions(fundamental_data)
    analysis.extend(conclusions)
    
    return "\n".join(analysis)

def format_market_cap(market_cap: float) -> str:
    """Format market cap in a readable format."""
    if market_cap is None:
        return "Không có dữ liệu"
        
    if market_cap >= 1_000_000_000_000:  # Trillion
        return f"{market_cap / 1_000_000_000_000:.2f} Nghìn tỷ"
    elif market_cap >= 1_000_000_000:  # Billion
        return f"{market_cap / 1_000_000_000:.2f} Tỷ"
    elif market_cap >= 1_000_000:  # Million
        return f"{market_cap / 1_000_000:.2f} Triệu"
    else:
        return f"{market_cap:,.0f}"

def generate_conclusions(data: dict) -> list:
    """Generate conclusion statements based on fundamental data."""
    conclusions = []
    
    pe_ratio = data.get("pe_ratio")
    pb_ratio = data.get("pb_ratio")
    dividend_yield = data.get("dividend_yield")
    
    # Valuation conclusion
    if pe_ratio and pb_ratio:
        if pe_ratio < 15 and pb_ratio < 1.5:
            conclusions.append("- Định giá ở mức thấp, có thể xem xét nếu công ty có nền tảng cơ bản tốt")
        elif pe_ratio > 30 or pb_ratio > 3:
            conclusions.append("- Định giá đang ở mức cao, cần thận trọng và nghiên cứu thêm về triển vọng tăng trưởng")
        else:
            conclusions.append("- Định giá ở mức hợp lý so với thị trường")
    
    # Dividend conclusion
    if dividend_yield:
        dividend_yield_percent = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
        if dividend_yield_percent > 4:
            conclusions.append("- Tỷ suất cổ tức hấp dẫn, phù hợp cho chiến lược đầu tư thu nhập")
    
    # Add general recommendation if we have enough data
    if len(conclusions) > 0:
        if pe_ratio and pb_ratio and dividend_yield:
            # This is a very simplistic investment approach for demonstration
            # Real investment decisions require much more thorough analysis
            score = 0
            
            # Low P/E is good
            if pe_ratio < 10:
                score += 2
            elif pe_ratio < 20:
                score += 1
            
            # Low P/B is good
            if pb_ratio < 1:
                score += 2
            elif pb_ratio < 2:
                score += 1
            
            # High dividend is good
            dividend_yield_percent = dividend_yield * 100 if dividend_yield < 1 else dividend_yield
            if dividend_yield_percent > 5:
                score += 2
            elif dividend_yield_percent > 2:
                score += 1
            
            # Generate recommendation based on simple score
            if score >= 4:
                conclusions.append("- Dựa trên các chỉ số cơ bản, cổ phiếu có vẻ hấp dẫn cho đầu tư giá trị")
            elif score >= 2:
                conclusions.append("- Các chỉ số cơ bản ở mức trung bình, cần phân tích sâu hơn về triển vọng tăng trưởng và ngành")
            else:
                conclusions.append("- Định giá hiện tại có vẻ cao, nhà đầu tư cần thận trọng và tìm hiểu thêm")
    
    if not conclusions:
        conclusions.append("- Cần thêm dữ liệu để đưa ra nhận định chi tiết hơn")
    
    return conclusions 