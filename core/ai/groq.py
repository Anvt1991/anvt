import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
import httpx
from loguru import logger
import pandas as pd
from dotenv import load_dotenv
import re
from core.technical import TechnicalAnalyzer
from datetime import datetime
from core.data.data_validator import DataValidator

# Tải biến môi trường từ file .env
load_dotenv()

class GroqHandler:
    """
    Lớp quản lý tương tác với Groq API để phân tích mẫu hình kỹ thuật
    """
    
    def __init__(self, api_key: str = None):
        """
        Khởi tạo GroqHandler
        
        Args:
            api_key: API key của Groq (mặc định lấy từ biến môi trường GROQ_API_KEY)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY không được cung cấp và không tìm thấy trong biến môi trường")
        
        # Endpoint và header
        self.api_endpoint = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Thông số model
        self.model_name = os.getenv("GROQ_MODEL", "deepseek-r1-distill-llama-70b")
        self.generation_config = {
            "temperature": 0.3,
            "max_tokens": 4096,
            "top_p": 0.9,
        }
        
        self.retry_config = {
            "max_retries": int(os.getenv("GROQ_MAX_RETRIES", 3)),
            "retry_delay": float(os.getenv("GROQ_RETRY_DELAY", 1.0))
        }
        
        logger.info(f"Khởi tạo GroqHandler với model: {self.model_name}")
    
    def analyze_pattern(self, 
                      stock_data: pd.DataFrame, 
                      symbol: str, 
                      additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Phân tích mẫu hình kỹ thuật từ dữ liệu giá cổ phiếu (sync)
        """
        try:
            if stock_data.empty:
                raise ValueError("Dữ liệu cổ phiếu không được cung cấp hoặc rỗng")
            
            logger.info(f"Bắt đầu phân tích mẫu hình cho {symbol}")
            logger.debug(f"Các cột đầu vào: {stock_data.columns.tolist()}")
            logger.debug(f"Dữ liệu mẫu đầu vào:\n{stock_data.head(3)}")
            
            stock_data_clean = DataValidator.normalize_dataframe(stock_data.copy())
            logger.info(f"Các cột sau khi chuẩn hóa: {stock_data_clean.columns.tolist()}")
            try:
                stock_data_clean = DataValidator.validate_schema(stock_data_clean)
            except Exception as schema_error:
                logger.warning(f"Cảnh báo khi validate schema: {str(schema_error)}")
            if 'close' not in stock_data_clean.columns:
                raise ValueError(f"Dữ liệu đầu vào cho {symbol} vẫn thiếu cột 'close' sau khi chuẩn hóa. Các cột hiện tại: {list(stock_data_clean.columns)}")

            # 1. Lấy giá trị cần thiết (dùng tên cột thường)
            current_price = float(stock_data_clean["close"].iloc[-1])
            previous_price = float(stock_data_clean["close"].iloc[-2])
            price_change = round(current_price - previous_price, 2)
            price_change_percent = round((price_change / previous_price) * 100, 2)
            price_info = {
                "current_price": current_price,
                "previous_price": previous_price,
                "price_change": price_change,
                "price_change_percent": price_change_percent
            }

            # 2. Khi tạo prompt cho Groq, mới đổi tên cột
            prompt_df = stock_data_clean.copy()
            prompt_df = prompt_df.reset_index()
            if 'date' in prompt_df.columns:
                prompt_df = prompt_df.rename(columns={'date': 'Date'})
            elif prompt_df.index.name and prompt_df.index.name in prompt_df.columns:
                prompt_df = prompt_df.rename(columns={prompt_df.index.name: 'Date'})
            prompt_df = prompt_df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            })

            logger.info(f"Các cột sau khi chuẩn hóa toàn diện (cho prompt): {prompt_df.columns.tolist()}")

            # Tự tính toán chỉ báo kỹ thuật
            calculated_indicators = TechnicalAnalyzer.get_technical_indicators(stock_data_clean)
            technical_analyzer = TechnicalAnalyzer()
            support_levels, resistance_levels = technical_analyzer._find_support_resistance(stock_data_clean)
            trend = technical_analyzer.analyze_trend(stock_data_clean)

            # Chuẩn bị dữ liệu giá để đưa vào prompt (chỉ một số nến gần đây)
            num_bars = min(50, len(prompt_df))
            recent_data = prompt_df.tail(num_bars).reset_index(drop=True)

            # Đảm bảo cột Date tồn tại và đúng định dạng
            if 'Date' not in recent_data.columns:
                date_col = recent_data.columns[0]
                recent_data.rename(columns={date_col: 'Date'}, inplace=True)
            if pd.api.types.is_datetime64_any_dtype(recent_data['Date']):
                recent_data["Date"] = recent_data["Date"].dt.strftime("%Y-%m-%d")
            else:
                try:
                    recent_data["Date"] = pd.to_datetime(recent_data["Date"]).dt.strftime("%Y-%m-%d")
                except:
                    recent_data["Date"] = recent_data["Date"].astype(str)

            # Format dữ liệu thành text
            data_text = "Ngày,Mở cửa,Cao nhất,Thấp nhất,Đóng cửa,Khối lượng\n"
            for _, row in recent_data.iterrows():
                try:
                    data_text += f"{row['Date']},{row['Open']:.2f},{row['High']:.2f},{row['Low']:.2f},{row['Close']:.2f},{int(row['Volume'])}\n"
                except:
                    date_val = row.get('Date', 'N/A')
                    open_val = f"{row.get('Open', 0):.2f}" if row.get('Open') is not None else "N/A"
                    high_val = f"{row.get('High', 0):.2f}" if row.get('High') is not None else "N/A"
                    low_val = f"{row.get('Low', 0):.2f}" if row.get('Low') is not None else "N/A"
                    close_val = f"{row.get('Close', 0):.2f}" if row.get('Close') is not None else "N/A"
                    volume_val = int(row.get('Volume', 0)) if row.get('Volume') is not None else "N/A"
                    data_text += f"{date_val},{open_val},{high_val},{low_val},{close_val},{volume_val}\n"

            # Sử dụng hàm mới để tạo prompt nâng cao
            system_prompt, user_prompt = self._create_enhanced_prompt(
                symbol=symbol,
                data_text=data_text,
                price_info=price_info,
                technical_indicators=calculated_indicators,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend=trend,
                additional_context=additional_context
            )

            # Gọi Groq API với prompt nâng cao
            json_response = self._call_groq_api_sync(system_prompt, user_prompt)
            
            if not json_response:
                logger.error(f"Không nhận được phản hồi từ Groq API khi phân tích {symbol}")
                return {
                    "symbol": symbol,
                    "error": "Không thể phân tích do lỗi API"
                }
            
            # Xử lý và kiểm tra kết quả
            try:
                # Xử lý trường hợp JSON được bọc trong các dấu ```
                json_response = self._extract_json_content(json_response)
                
                # Parse JSON
                result = json.loads(json_response)
                
                # Đảm bảo dữ liệu đủ các trường cần thiết
                if "symbol" not in result:
                    result["symbol"] = symbol
                if "current_price" not in result:
                    result["current_price"] = price_info["current_price"]
                if "price_change" not in result:
                    result["price_change"] = price_info["price_change"]
                if "price_change_percent" not in result:
                    result["price_change_percent"] = price_info["price_change_percent"]
                
                # Luôn sử dụng chỉ báo kỹ thuật đã tính
                if "indicators" not in result:
                    result["indicators"] = {}
                
                # Thêm chỉ báo RSI từ dữ liệu đã tính
                result["indicators"]["rsi"] = {
                    "value": calculated_indicators.get("rsi_14"),
                    "interpretation": "Quá mua" if calculated_indicators.get("rsi_14", 0) > 70 else 
                                      "Quá bán" if calculated_indicators.get("rsi_14", 0) < 30 else "Trung tính"
                }
                
                # Thêm chỉ báo MACD từ dữ liệu đã tính
                result["indicators"]["macd"] = {
                    "macd_value": calculated_indicators.get("macd"),
                    "signal_line": calculated_indicators.get("macd_signal"),
                    "histogram": calculated_indicators.get("macd_hist"),
                    "interpretation": "Tích cực" if calculated_indicators.get("macd_hist", 0) > 0 else "Tiêu cực"
                }
                
                # Thêm Moving Averages từ dữ liệu đã tính
                result["indicators"]["moving_averages"] = {
                    "ma_20": calculated_indicators.get("ma_20"),
                    "ma_50": calculated_indicators.get("ma_50"),
                    "ma_200": calculated_indicators.get("ma_200"),
                    "interpretation": "Đang lên" if (calculated_indicators.get("ma_20", 0) or 0) > (calculated_indicators.get("ma_50", 0) or 0) else "Đang xuống"
                }
                
                # Thêm support/resistance levels đã tính
                result["support_resistance"] = {
                    "support_levels": support_levels,
                    "resistance_levels": resistance_levels
                }
                
                # Kiểm tra tính logic của khuyến nghị với chỉ báo và xu hướng
                if "overall_rating" in result and "recommendation" in result["overall_rating"]:
                    recommendation = result["overall_rating"]["recommendation"]
                    # Kiểm tra mâu thuẫn nghiêm trọng giữa khuyến nghị và xu hướng/chỉ báo
                    if recommendation == "Mua" and trend.startswith("Giảm") and calculated_indicators.get("rsi_14", 50) > 70:
                        result["overall_rating"]["recommendation"] = "Nắm giữ"
                        if "rationale" in result["overall_rating"]:
                            result["overall_rating"]["rationale"] += " (Điều chỉnh từ Mua sang Nắm giữ do mâu thuẫn với xu hướng giảm và RSI quá mua)"
                        # Thêm cảnh báo vào special_notes nếu trường này tồn tại
                        if "special_notes" in result["overall_rating"]:
                            result["overall_rating"]["special_notes"] += " CHÚ Ý: Khuyến nghị đã được điều chỉnh do mâu thuẫn với xu hướng và chỉ báo RSI."
                        else:
                            result["overall_rating"]["special_notes"] = "CHÚ Ý: Khuyến nghị đã được điều chỉnh do mâu thuẫn với xu hướng và chỉ báo RSI."
                    elif recommendation == "Bán" and trend.startswith("Tăng") and calculated_indicators.get("rsi_14", 50) < 30:
                        result["overall_rating"]["recommendation"] = "Nắm giữ"
                        if "rationale" in result["overall_rating"]:
                            result["overall_rating"]["rationale"] += " (Điều chỉnh từ Bán sang Nắm giữ do mâu thuẫn với xu hướng tăng và RSI quá bán)"
                        # Thêm cảnh báo vào special_notes nếu trường này tồn tại
                        if "special_notes" in result["overall_rating"]:
                            result["overall_rating"]["special_notes"] += " CHÚ Ý: Khuyến nghị đã được điều chỉnh do mâu thuẫn với xu hướng và chỉ báo RSI."
                        else:
                            result["overall_rating"]["special_notes"] = "CHÚ Ý: Khuyến nghị đã được điều chỉnh do mâu thuẫn với xu hướng và chỉ báo RSI."
                
                # Thêm metadata về phân tích
                result["meta"] = {
                    "analyzed_at": datetime.now().isoformat(),
                    "trend": trend,
                    "data_points": len(stock_data_clean),
                    "version": "2.0"  # Đánh dấu phiên bản nâng cao của prompt
                }
                
                # Trả về kết quả đã xử lý
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Lỗi JSON từ Groq ({symbol}): {str(e)} - Phản hồi: {json_response[:200]}...")
                
                # Cố gắng sửa chữa JSON
                try:
                    fixed_json = self._fix_json_format(json_response)
                    result = json.loads(fixed_json)
                    
                    # Đảm bảo dữ liệu đủ các trường cần thiết
                    if "symbol" not in result:
                        result["symbol"] = symbol
                    if "current_price" not in result:
                        result["current_price"] = price_info["current_price"]
                    if "price_change" not in result:
                        result["price_change"] = price_info["price_change"]
                    if "price_change_percent" not in result:
                        result["price_change_percent"] = price_info["price_change_percent"]
                    
                    # Thêm các chỉ báo kỹ thuật đã tính vào kết quả
                    # Tương tự như phần trên
                    if "indicators" not in result:
                        result["indicators"] = {}
                    
                    result["indicators"]["rsi"] = {
                        "value": calculated_indicators.get("rsi_14"),
                        "interpretation": "Quá mua" if calculated_indicators.get("rsi_14", 0) > 70 else 
                                          "Quá bán" if calculated_indicators.get("rsi_14", 0) < 30 else "Trung tính"
                    }
                    
                    result["indicators"]["macd"] = {
                        "macd_value": calculated_indicators.get("macd"),
                        "signal_line": calculated_indicators.get("macd_signal"),
                        "histogram": calculated_indicators.get("macd_hist"),
                        "interpretation": "Tích cực" if calculated_indicators.get("macd_hist", 0) > 0 else "Tiêu cực"
                    }
                    
                    result["indicators"]["moving_averages"] = {
                        "ma_20": calculated_indicators.get("ma_20"),
                        "ma_50": calculated_indicators.get("ma_50"),
                        "ma_200": calculated_indicators.get("ma_200"),
                        "interpretation": "Đang lên" if (calculated_indicators.get("ma_20", 0) or 0) > (calculated_indicators.get("ma_50", 0) or 0) else "Đang xuống"
                    }
                    
                    result["support_resistance"] = {
                        "support_levels": support_levels,
                        "resistance_levels": resistance_levels
                    }
                    
                    result["meta"] = {
                        "analyzed_at": datetime.now().isoformat(),
                        "trend": trend,
                        "data_points": len(stock_data_clean),
                        "json_fixed": True,
                        "version": "2.0"
                    }
                    
                    logger.info(f"Đã sửa chữa được JSON cho {symbol}")
                    return result
                    
                except Exception as fix_error:
                    logger.error(f"Không thể sửa chữa JSON: {str(fix_error)}")
                
                # Trả về dữ liệu tối thiểu nếu xảy ra lỗi
                return {
                    "symbol": symbol,
                    "current_price": price_info["current_price"],
                    "price_change": price_info["price_change"],
                    "price_change_percent": price_info["price_change_percent"],
                    "indicators": {
                        "rsi": {"value": calculated_indicators.get("rsi_14"), 
                                "interpretation": "Quá mua" if calculated_indicators.get("rsi_14", 0) > 70 else 
                                                 "Quá bán" if calculated_indicators.get("rsi_14", 0) < 30 else "Trung tính"},
                        "macd": {"macd_value": calculated_indicators.get("macd"), 
                                "signal_line": calculated_indicators.get("macd_signal"),
                                "histogram": calculated_indicators.get("macd_hist"),
                                "interpretation": "Tích cực" if calculated_indicators.get("macd_hist", 0) > 0 else "Tiêu cực"},
                        "moving_averages": {
                            "ma_20": calculated_indicators.get("ma_20"),
                            "ma_50": calculated_indicators.get("ma_50"),
                            "ma_200": calculated_indicators.get("ma_200"),
                            "interpretation": "Đang lên" if (calculated_indicators.get("ma_20", 0) or 0) > (calculated_indicators.get("ma_50", 0) or 0) else "Đang xuống"
                        }
                    },
                    "support_resistance": {
                        "support_levels": support_levels,
                        "resistance_levels": resistance_levels
                    },
                    "overall_rating": {
                        "recommendation": "Nắm giữ",  # Khuyến nghị an toàn khi có lỗi
                        "risk_level": "Trung bình",
                        "summary": "Không thể phân tích đầy đủ do lỗi từ API",
                        "rationale": "Khuyến nghị thận trọng do hệ thống không thể phân tích hoàn chỉnh mẫu hình kỹ thuật",
                        "time_frame": "Ngắn hạn",
                        "special_notes": "Dữ liệu mẫu hình không thể phân tích do lỗi API. Chỉ sử dụng chỉ báo kỹ thuật để đưa ra đánh giá sơ bộ."
                    },
                    "technical_patterns": [],
                    "candlestick_patterns": [],
                    "error": "Không thể phân tích phản hồi JSON",
                    "meta": {
                        "analyzed_at": datetime.now().isoformat(),
                        "trend": trend,
                        "data_points": len(stock_data_clean),
                        "json_error": True,
                        "version": "2.0"
                    }
                }
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích mẫu hình cho {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "error": str(e)
            }
    
    def _extract_json_content(self, text: str) -> str:
        """
        Trích xuất nội dung JSON từ văn bản, loại bỏ các markdown code blocks nếu có.
        
        Args:
            text: Chuỗi văn bản chứa JSON
            
        Returns:
            Chuỗi JSON đã được trích xuất
        """
        # Pattern để trích xuất JSON từ markdown code block
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        matches = re.findall(pattern, text)
        
        if matches:
            # Lấy khối JSON lớn nhất
            return max(matches, key=len).strip()
        else:
            # Nếu không có code blocks, trả về văn bản gốc
            return text.strip()
    
    def _validate_and_fix_json_structure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xác thực và sửa cấu trúc JSON nếu cần, đảm bảo tính chính xác của các giá trị số
        
        Args:
            data: Dictionary chứa dữ liệu JSON
            
        Returns:
            Dictionary đã được sửa cấu trúc và xác thực
        """
        # Lấy giá hiện tại để kiểm tra tính hợp lý của các mức hỗ trợ/kháng cự
        current_price = data.get("current_price", 0)
        if not isinstance(current_price, (int, float)) or current_price <= 0:
            logger.warning(f"Giá hiện tại không hợp lệ: {current_price}, sử dụng giá mặc định")
            # Nếu không có giá hợp lệ, không thể xác thực các mức giá khác
            return data
            
        # 1. Xác thực và chuẩn hóa technical_patterns
        if "technical_patterns" not in data:
            data["technical_patterns"] = []
        elif not isinstance(data["technical_patterns"], list):
            data["technical_patterns"] = []
        else:
            validated_patterns = []
            for pattern in data["technical_patterns"]:
                if not isinstance(pattern, dict):
                    continue
                # Đảm bảo có pattern_name
                if "pattern_name" not in pattern or not pattern["pattern_name"]:
                    continue
                # Chuẩn hóa confidence (chuyển từ string "High/Medium/Low" thành số)
                confidence = pattern.get("confidence", "Medium")
                if isinstance(confidence, str):
                    if confidence.lower() == "high":
                        confidence = 0.9
                    elif confidence.lower() == "medium":
                        confidence = 0.6
                    elif confidence.lower() == "low":
                        confidence = 0.3
                    else:
                        try:
                            confidence = confidence.replace("%", "").strip()
                            confidence = float(confidence)
                            if confidence > 1 and confidence <= 100:
                                confidence = confidence / 100
                        except:
                            confidence = 0.5  # Giá trị mặc định
                elif isinstance(confidence, (int, float)):
                    if confidence > 1:
                        confidence = min(confidence, 100) / 100
                else:
                    confidence = 0.5
                pattern["confidence"] = round(confidence, 2)
                # Bổ sung implications nếu thiếu
                if not pattern.get("implications"):
                    name = pattern.get("pattern_name", "").lower()
                    if any(x in name for x in ["double bottom", "bullish", "breakout", "inverse head", "cup and handle"]):
                        pattern["implications"] = "Tín hiệu đảo chiều tăng/mua"
                    elif any(x in name for x in ["double top", "bearish", "breakdown", "head and shoulders"]):
                        pattern["implications"] = "Tín hiệu đảo chiều giảm/bán"
                    else:
                        pattern["implications"] = "Cần quan sát xác nhận thêm"
                validated_patterns.append(pattern)
            data["technical_patterns"] = validated_patterns
        
        # 2. Xác thực và chuẩn hóa support_resistance
        if "support_resistance" not in data:
            data["support_resistance"] = {
                "support_levels": [],
                "resistance_levels": []
            }
        elif not isinstance(data["support_resistance"], dict):
            data["support_resistance"] = {
                "support_levels": [],
                "resistance_levels": []
            }
        else:
            # Đảm bảo các trường con tồn tại
            if "support_levels" not in data["support_resistance"]:
                data["support_resistance"]["support_levels"] = []
            if "resistance_levels" not in data["support_resistance"]:
                data["support_resistance"]["resistance_levels"] = []
                
            # Chuẩn hóa support levels
            support_levels = []
            raw_support = data["support_resistance"]["support_levels"]
            if isinstance(raw_support, list):
                for level in raw_support:
                    if isinstance(level, (int, float)):
                        # Kiểm tra xem mức hỗ trợ có thấp hơn giá hiện tại không
                        if 0 < level < current_price * 0.99:  # Thấp hơn ít nhất 1%
                            support_levels.append(round(level, 2))
                    elif isinstance(level, dict) and "value" in level:
                        value = level["value"]
                        if isinstance(value, (int, float)) and 0 < value < current_price * 0.99:
                            support_levels.append(round(value, 2))
                    elif isinstance(level, str):
                        try:
                            value = float(level.replace(',', '').strip())
                            if 0 < value < current_price * 0.99:
                                support_levels.append(round(value, 2))
                        except:
                            continue
                            
            # Chuẩn hóa resistance levels
            resistance_levels = []
            raw_resistance = data["support_resistance"]["resistance_levels"]
            if isinstance(raw_resistance, list):
                for level in raw_resistance:
                    if isinstance(level, (int, float)):
                        # Kiểm tra xem mức kháng cự có cao hơn giá hiện tại không
                        if level > current_price * 1.01:  # Cao hơn ít nhất 1%
                            resistance_levels.append(round(level, 2))
                    elif isinstance(level, dict) and "value" in level:
                        value = level["value"]
                        if isinstance(value, (int, float)) and value > current_price * 1.01:
                            resistance_levels.append(round(value, 2))
                    elif isinstance(level, str):
                        try:
                            value = float(level.replace(',', '').strip())
                            if value > current_price * 1.01:
                                resistance_levels.append(round(value, 2))
                        except:
                            continue
            
            # Nếu không có đủ mức hỗ trợ/kháng cự sau khi lọc
            if len(support_levels) < 2:
                # Thêm mức hỗ trợ theo tỷ lệ phần trăm giá hiện tại
                support_levels = [
                    round(current_price * 0.95, 2),  # -5%
                    round(current_price * 0.97, 2)   # -3%
                ]
                
            if len(resistance_levels) < 2:
                # Thêm mức kháng cự theo tỷ lệ phần trăm giá hiện tại
                resistance_levels = [
                    round(current_price * 1.05, 2),  # +5%
                    round(current_price * 1.03, 2)   # +3%
                ]
                
            # Sắp xếp các mức
            support_levels = sorted(set(support_levels))  # Loại bỏ trùng lặp
            resistance_levels = sorted(set(resistance_levels), reverse=True)  # Loại bỏ trùng lặp, sắp xếp giảm dần
            
            data["support_resistance"]["support_levels"] = support_levels
            data["support_resistance"]["resistance_levels"] = resistance_levels
        
        # 3. Xác thực và chuẩn hóa indicators
        if "indicators" not in data:
            data["indicators"] = {
                "rsi": {"value": None, "interpretation": ""},
                "macd": {"value": None, "interpretation": ""},
                "moving_averages": {"ma_20": None, "ma_50": None, "ma_200": None, "interpretation": ""}
            }
        elif not isinstance(data["indicators"], dict):
            data["indicators"] = {
                "rsi": {"value": None, "interpretation": ""},
                "macd": {"value": None, "interpretation": ""},
                "moving_averages": {"ma_20": None, "ma_50": None, "ma_200": None, "interpretation": ""}
            }
        else:
            # Xác thực RSI
            if "rsi" not in data["indicators"] or not isinstance(data["indicators"]["rsi"], dict):
                data["indicators"]["rsi"] = {"value": None, "interpretation": ""}
            else:
                rsi = data["indicators"]["rsi"].get("value")
                if rsi is not None:
                    if isinstance(rsi, (int, float)):
                        # RSI nằm trong khoảng 0-100
                        if rsi < 0 or rsi > 100:
                            rsi = max(0, min(100, rsi))
                        data["indicators"]["rsi"]["value"] = round(rsi, 2)
                    else:
                        try:
                            rsi = float(str(rsi).replace('%', ''))
                            if rsi < 0 or rsi > 100:
                                rsi = max(0, min(100, rsi))
                            data["indicators"]["rsi"]["value"] = round(rsi, 2)
                        except:
                            data["indicators"]["rsi"]["value"] = None
            
            # Xác thực MACD
            if "macd" not in data["indicators"] or not isinstance(data["indicators"]["macd"], dict):
                data["indicators"]["macd"] = {"value": None, "interpretation": ""}
            else:
                macd = data["indicators"]["macd"].get("value")
                if macd is not None and not isinstance(macd, (int, float)):
                    try:
                        macd = float(str(macd))
                        data["indicators"]["macd"]["value"] = round(macd, 4)
                    except:
                        data["indicators"]["macd"]["value"] = None
                elif macd is not None:
                    data["indicators"]["macd"]["value"] = round(macd, 4)
            
            # Xác thực Moving Averages
            if "moving_averages" not in data["indicators"] or not isinstance(data["indicators"]["moving_averages"], dict):
                data["indicators"]["moving_averages"] = {"ma_20": None, "ma_50": None, "ma_200": None, "interpretation": ""}
            else:
                for ma_key in ["ma_20", "ma_50", "ma_200"]:
                    ma_value = data["indicators"]["moving_averages"].get(ma_key)
                    if ma_value is not None:
                        if isinstance(ma_value, (int, float)):
                            # MA nên dương và trong khoảng hợp lý so với giá hiện tại
                            if ma_value <= 0 or ma_value > current_price * 2:
                                data["indicators"]["moving_averages"][ma_key] = None
                            else:
                                data["indicators"]["moving_averages"][ma_key] = round(ma_value, 2)
                        else:
                            try:
                                ma_value = float(str(ma_value).replace(',', ''))
                                if ma_value <= 0 or ma_value > current_price * 2:
                                    data["indicators"]["moving_averages"][ma_key] = None
                                else:
                                    data["indicators"]["moving_averages"][ma_key] = round(ma_value, 2)
                            except:
                                data["indicators"]["moving_averages"][ma_key] = None
        
        # 4. Xác thực wave_analysis
        if "wave_analysis" not in data:
            data["wave_analysis"] = {
                "direction": "",
                "main_wave": "",
                "subwaves": [],
                "interpretation": "",
                "confidence": ""
            }
        elif not isinstance(data["wave_analysis"], dict):
            data["wave_analysis"] = {
                "direction": "",
                "main_wave": "",
                "subwaves": [],
                "interpretation": "",
                "confidence": ""
            }
        else:
            # Đảm bảo các trường con tồn tại
            if "direction" not in data["wave_analysis"]:
                data["wave_analysis"]["direction"] = ""
            if "main_wave" not in data["wave_analysis"]:
                data["wave_analysis"]["main_wave"] = ""
            if "subwaves" not in data["wave_analysis"]:
                data["wave_analysis"]["subwaves"] = []
            if "interpretation" not in data["wave_analysis"]:
                data["wave_analysis"]["interpretation"] = ""
            if "confidence" not in data["wave_analysis"]:
                data["wave_analysis"]["confidence"] = ""
        
        # 5. Xác thực overall_rating
        if "overall_rating" not in data:
            data["overall_rating"] = {
                "recommendation": "",
                "risk_level": "",
                "summary": "",
                "rationale": ""
            }
        elif not isinstance(data["overall_rating"], dict):
            data["overall_rating"] = {
                "recommendation": "",
                "risk_level": "",
                "summary": "",
                "rationale": ""
            }
        else:
            # Chuẩn hóa recommendation
            recommendation = data["overall_rating"].get("recommendation", "")
            if isinstance(recommendation, str):
                recommendation = recommendation.strip().lower()
                if recommendation in ["mua", "buy", "strong buy", "mua mạnh"]:
                    data["overall_rating"]["recommendation"] = "Mua"
                elif recommendation in ["bán", "sell", "strong sell", "bán mạnh"]:
                    data["overall_rating"]["recommendation"] = "Bán"
                elif recommendation in ["nắm giữ", "hold", "neutral", "trung lập"]:
                    data["overall_rating"]["recommendation"] = "Nắm giữ"
                else:
                    data["overall_rating"]["recommendation"] = "N/A"
            else:
                data["overall_rating"]["recommendation"] = "N/A"
                
            # Chuẩn hóa risk_level
            risk_level = data["overall_rating"].get("risk_level", "")
            if isinstance(risk_level, str):
                risk_level = risk_level.strip().lower()
                if risk_level in ["cao", "high"]:
                    data["overall_rating"]["risk_level"] = "Cao"
                elif risk_level in ["trung bình", "medium"]:
                    data["overall_rating"]["risk_level"] = "Trung bình"
                elif risk_level in ["thấp", "low"]:
                    data["overall_rating"]["risk_level"] = "Thấp"
                else:
                    data["overall_rating"]["risk_level"] = "N/A"
            else:
                data["overall_rating"]["risk_level"] = "N/A"
        
        return data
    
    def _fix_json_format(self, text: str) -> str:
        """
        Sửa chữa các lỗi phổ biến trong định dạng JSON từ LLM
        
        Args:
            text: Chuỗi JSON cần sửa chữa
            
        Returns:
            Chuỗi JSON đã được sửa chữa
        """
        # Loại bỏ các dấu ``` nếu có
        cleaned_text = re.sub(r'```json\s*|\s*```', '', text)
        
        # Sửa các lỗi dấu nháy không đúng
        cleaned_text = re.sub(r'(?<!")"(?!")', '"', cleaned_text)
        
        # Sửa lỗi dấu phẩy cuối cùng trong mảng và đối tượng
        cleaned_text = re.sub(r',\s*}', '}', cleaned_text)
        cleaned_text = re.sub(r',\s*]', ']', cleaned_text)
        
        # Sửa lỗi cặp key-value không hợp lệ (thiếu giá trị)
        cleaned_text = re.sub(r'"([^"]+)":\s*,', r'"\1": null,', cleaned_text)
        
        # Sửa trường hợp thiếu dấu ngoặc đóng
        open_braces = cleaned_text.count('{')
        close_braces = cleaned_text.count('}')
        if open_braces > close_braces:
            cleaned_text += '}' * (open_braces - close_braces)
        
        open_brackets = cleaned_text.count('[')
        close_brackets = cleaned_text.count(']')
        if open_brackets > close_brackets:
            cleaned_text += ']' * (open_brackets - close_brackets)
        
        return cleaned_text
    
    def _call_groq_api_sync(self, system_prompt: str, user_prompt: str) -> str:
        """
        Gọi Groq API phiên bản đồng bộ (sync)
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or "Bạn là chuyên gia phân tích kỹ thuật."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.generation_config["temperature"],
            "max_tokens": self.generation_config["max_tokens"],
            "top_p": self.generation_config["top_p"],
            "response_format": {"type": "json_object"}
        }
        retries = 0
        while retries < self.retry_config["max_retries"]:
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(
                        self.api_endpoint,
                        headers=self.headers,
                        json=payload
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Groq API trả về lỗi: {response.status_code} - {response.text}")
                        if 500 <= response.status_code < 600:
                            retries += 1
                            time.sleep(self.retry_config["retry_delay"] * (2 ** retries))
                            continue
                        return None
            except Exception as e:
                logger.error(f"Lỗi khi gọi Groq API (sync): {str(e)}")
                retries += 1
                if retries < self.retry_config["max_retries"]:
                    time.sleep(self.retry_config["retry_delay"] * (2 ** retries))
                else:
                    return None
        return None
    
    def set_model(self, model_name: str) -> None:
        """
        Thiết lập model cho Groq API
        
        Args:
            model_name: Tên model
        """
        self.model_name = model_name
        logger.info(f"Đã thiết lập model Groq thành: {model_name}")
    
    def test_connection(self) -> bool:
        """
        Kiểm tra kết nối với Groq API
        
        Returns:
            True nếu kết nối thành công, False nếu không
        """
        try:
            # Tạo payload cho lời gọi API đơn giản
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "Kiểm tra kết nối."},
                    {"role": "user", "content": "Hello, world!"}
                ],
                "max_tokens": 10,
                "response_format": {"type": "text"}  # Sử dụng định dạng text đơn giản cho kiểm tra kết nối
            }
            
            # Sử dụng httpx vì đã được sử dụng trong các phương thức khác
            with httpx.Client(timeout=10.0) as client:  # Thêm timeout 10 giây
                response = client.post(
                    self.api_endpoint,
                    headers=self.headers,
                    json=payload
                )
            
            # Kiểm tra phản hồi HTTP
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    logger.info("Kết nối Groq API thành công")
                    return True
                else:
                    logger.error(f"Phản hồi Groq API không đúng định dạng: {response.text}")
                    return False
            else:
                logger.error(f"Lỗi kết nối Groq API: {response.status_code} - {response.text}")
                return False
                
        except httpx.TimeoutException:
            logger.error("Kết nối Groq API bị timeout")
            return False
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error khi kết nối Groq API: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Lỗi không xác định khi kết nối Groq API: {str(e)}")
            return False
    
    def _validate_numerical_values(self, data: Dict[str, Any], stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Xác thực tính hợp lý của các giá trị số trong kết quả phân tích và
        sử dụng chỉ số kỹ thuật đã tính sẵn thay vì tính lại.
        """
        try:
            warnings = []
            # Đảm bảo có dữ liệu giá hiện tại
            if not isinstance(data.get("current_price"), (int, float)) or data["current_price"] <= 0:
                if not stock_data.empty and "Close" in stock_data.columns:
                    data["current_price"] = float(stock_data["Close"].iloc[-1])
                else:
                    data["current_price"] = 1.0  # Giá trị mặc định nếu không có dữ liệu
            current_price = data["current_price"]
            
            # --- NÂNG CẤP: Sử dụng chỉ báo kỹ thuật đã tính sẵn ---
            calculated_indicators = TechnicalAnalyzer.get_technical_indicators(stock_data)
            
            # Kiểm tra và điều chỉnh RSI
            if "indicators" in data and "rsi" in data["indicators"]:
                rsi_value = data["indicators"]["rsi"].get("value")
                calc_rsi = calculated_indicators.get("rsi_14")
                if rsi_value is not None and calc_rsi is not None:
                    if abs(rsi_value - calc_rsi) > 10:  # Giảm ngưỡng sai lệch từ 20 xuống 10
                        warnings.append(f"RSI từ LLM ({rsi_value}) lệch lớn so với tính toán ({round(calc_rsi,2)}), đã thay thế.")
                        data["indicators"]["rsi"]["value"] = round(calc_rsi, 2)
                # Đảm bảo RSI trong khoảng hợp lệ
                if rsi_value is not None and not (0 <= rsi_value <= 100):
                    warnings.append(f"RSI ngoài khoảng hợp lệ: {rsi_value}")
                    data["indicators"]["rsi"]["value"] = max(0, min(100, rsi_value))
                # Nếu không có giá trị RSI, sử dụng giá trị đã tính
                if rsi_value is None and calc_rsi is not None:
                    data["indicators"]["rsi"]["value"] = round(calc_rsi, 2)
                    data["indicators"]["rsi"]["interpretation"] = "Quá mua" if calc_rsi > 70 else "Quá bán" if calc_rsi < 30 else "Trung tính"
            
            # Kiểm tra MA
            if "indicators" in data and "moving_averages" in data["indicators"]:
                for ma_key, calc_key in [("ma_20", "ma_20"), ("ma_50", "ma_50"), ("ma_200", "ma_200")]:
                    ma_value = data["indicators"]["moving_averages"].get(ma_key)
                    calc_ma = calculated_indicators.get(calc_key)
                    if ma_value is not None and calc_ma is not None:
                        if abs(ma_value - calc_ma) / (calc_ma if calc_ma != 0 else 1) > 0.1:  # Giảm ngưỡng sai lệch từ 0.2 xuống 0.1
                            warnings.append(f"{ma_key.upper()} từ LLM ({ma_value}) lệch lớn so với tính toán ({round(calc_ma,2)}), đã thay thế.")
                            data["indicators"]["moving_averages"][ma_key] = round(calc_ma, 2)
                    # Nếu không có giá trị MA, sử dụng giá trị đã tính
                    if ma_value is None and calc_ma is not None:
                        data["indicators"]["moving_averages"][ma_key] = round(calc_ma, 2)
            
            # Kiểm tra MACD
            if "indicators" in data and "macd" in data["indicators"]:
                macd_value = data["indicators"]["macd"].get("value")
                calc_macd = calculated_indicators.get("macd")
                if macd_value is not None and calc_macd is not None:
                    if abs(macd_value - calc_macd) > max(0.2, abs(calc_macd) * 0.3):  # Giảm ngưỡng sai lệch
                        warnings.append(f"MACD từ LLM ({macd_value}) lệch lớn so với tính toán ({round(calc_macd,4)}), đã thay thế.")
                        data["indicators"]["macd"]["value"] = round(calc_macd, 4)
                # Nếu không có giá trị MACD, sử dụng giá trị đã tính
                if macd_value is None and calc_macd is not None:
                    data["indicators"]["macd"]["value"] = round(calc_macd, 4)
                    data["indicators"]["macd"]["interpretation"] = "Tích cực" if calculated_indicators.get("macd_hist", 0) > 0 else "Tiêu cực"
            
            # Kiểm tra support levels - sử dụng TechnicalAnalyzer thay vì tự xây dựng
            if "support_resistance" in data and "support_levels" in data["support_resistance"]:
                support_from_llm = data["support_resistance"]["support_levels"]
                
                # Tính toán mức hỗ trợ từ dữ liệu thực
                technical_analyzer = TechnicalAnalyzer()
                support_levels, _ = technical_analyzer._find_support_resistance(stock_data)
                
                # Chỉ thông báo nếu có sự khác biệt lớn
                if support_from_llm and support_levels:
                    # Đếm số lượng mức hỗ trợ từ LLM và từ tính toán thực tế khác nhau quá 5%
                    diff_count = 0
                    for s_llm in support_from_llm:
                        if all(abs(s_llm - s_calc) / current_price > 0.05 for s_calc in support_levels):
                            diff_count += 1
                    
                    if diff_count > len(support_from_llm) // 2:
                        warnings.append(f"Mức hỗ trợ từ LLM khác biệt lớn so với tính toán, đã thay thế.")
                        data["support_resistance"]["support_levels"] = support_levels
                
                # Nếu không có mức hỗ trợ hợp lệ nào, sử dụng mức đã tính
                if not support_from_llm:
                    data["support_resistance"]["support_levels"] = support_levels
            
            # Kiểm tra resistance levels - tương tự như support
            if "support_resistance" in data and "resistance_levels" in data["support_resistance"]:
                resistance_from_llm = data["support_resistance"]["resistance_levels"]
                
                # Tính toán mức kháng cự từ dữ liệu thực
                technical_analyzer = TechnicalAnalyzer()
                _, resistance_levels = technical_analyzer._find_support_resistance(stock_data)
                
                # Chỉ thông báo nếu có sự khác biệt lớn
                if resistance_from_llm and resistance_levels:
                    # Đếm số lượng mức kháng cự từ LLM và từ tính toán thực tế khác nhau quá 5%
                    diff_count = 0
                    for r_llm in resistance_from_llm:
                        if all(abs(r_llm - r_calc) / current_price > 0.05 for r_calc in resistance_levels):
                            diff_count += 1
                    
                    if diff_count > len(resistance_from_llm) // 2:
                        warnings.append(f"Mức kháng cự từ LLM khác biệt lớn so với tính toán, đã thay thế.")
                        data["support_resistance"]["resistance_levels"] = resistance_levels
                
                # Nếu không có mức kháng cự hợp lệ nào, sử dụng mức đã tính
                if not resistance_from_llm:
                    data["support_resistance"]["resistance_levels"] = resistance_levels
            
            # Kiểm tra sự phù hợp của khuyến nghị với các chỉ báo kỹ thuật
            if "overall_rating" in data and "recommendation" in data["overall_rating"]:
                recommendation = data["overall_rating"]["recommendation"]
                
                # Xác định xu hướng từ giá trị MA
                ma20 = calculated_indicators.get("ma_20")
                ma50 = calculated_indicators.get("ma_50")
                ma_trend = None
                if ma20 is not None and ma50 is not None:
                    ma_trend = "Tăng" if ma20 > ma50 else "Giảm"
                
                # Kiểm tra mâu thuẫn giữa khuyến nghị và xu hướng MA/RSI
                if recommendation == "Mua" and ma_trend == "Giảm" and calculated_indicators.get("rsi_14", 50) > 70:
                    warnings.append(f"Khuyến nghị {recommendation} mâu thuẫn với xu hướng MA {ma_trend} và RSI quá mua")
                    if "rationale" in data["overall_rating"]:
                        data["overall_rating"]["rationale"] += " (Lưu ý: Khuyến nghị này đi ngược xu hướng MA và RSI)"
                elif recommendation == "Bán" and ma_trend == "Tăng" and calculated_indicators.get("rsi_14", 50) < 30:
                    warnings.append(f"Khuyến nghị {recommendation} mâu thuẫn với xu hướng MA {ma_trend} và RSI quá bán")
                    if "rationale" in data["overall_rating"]:
                        data["overall_rating"]["rationale"] += " (Lưu ý: Khuyến nghị này đi ngược xu hướng MA và RSI)"
            
            # Thêm cảnh báo nếu có
            if warnings:
                data["validation_warnings"] = warnings
            return data
        except Exception as e:
            logger.error(f"Lỗi khi xác thực giá trị số: {str(e)}")
            return data
    
    def get_report_indicators(self, stock_data):
        """
        Lấy chỉ số kỹ thuật chuẩn hóa cho báo cáo, không tự tính lại MA, RSI, MACD.
        """
        return TechnicalAnalyzer.get_technical_indicators(stock_data)

    def _create_enhanced_prompt(self, 
                               symbol: str,
                               data_text: str, 
                               price_info: Dict[str, float], 
                               technical_indicators: Dict[str, Any],
                               support_levels: List[float],
                               resistance_levels: List[float],
                               trend: str,
                               additional_context: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Tạo prompt nâng cao cho phân tích mẫu hình, tách biệt vai trò giữa tính toán kỹ thuật và nhận diện mẫu hình
        
        Args:
            symbol: Mã cổ phiếu
            data_text: Dữ liệu giá định dạng text
            price_info: Thông tin giá hiện tại, thay đổi, phần trăm thay đổi
            technical_indicators: Chỉ báo kỹ thuật đã tính toán
            support_levels: Mức hỗ trợ đã tính
            resistance_levels: Mức kháng cự đã tính
            trend: Xu hướng hiện tại
            additional_context: Thông tin bổ sung về thị trường
            
        Returns:
            Tuple (system_prompt, user_prompt)
        """
        # Chuẩn bị system prompt tập trung vào việc nhận diện mẫu hình
        system_prompt = """Bạn là chuyên gia phân tích kỹ thuật chứng khoán với 20+ năm kinh nghiệm. Nhiệm vụ của bạn bao gồm:

1. NHẬN DIỆN MẪU HÌNH KỸ THUẬT:
   - Mẫu hình đảo chiều: Head and Shoulders, Double/Triple Top/Bottom, Rounding Bottom/Top, Island Reversal
   - Mẫu hình tiếp diễn: Flags, Pennants, Triangles (Ascending/Descending/Symmetrical), Wedges, Rectangles, Cup and Handle
   - Mẫu nến: Doji, Hammer, Shooting Star, Engulfing, Morning/Evening Star, Harami, Three White Soldiers, Three Black Crows

2. PHÂN TÍCH SÓNG ELLIOTT:
   - Xác định vị trí hiện tại trong chu kỳ sóng 5-3
   - Dự đoán điểm đảo chiều tiềm năng dựa trên nguyên lý sóng

3. ĐƯA RA KHUYẾN NGHỊ ĐẦU TƯ:
   - Dựa trên mẫu hình và tín hiệu kỹ thuật đã nhận diện
   - Xác định mục tiêu giá và mức độ rủi ro
   - Phù hợp với chỉ báo kỹ thuật đã cung cấp

LƯU Ý QUAN TRỌNG:
- KHÔNG TỰ TÍNH các chỉ báo kỹ thuật (RSI, MACD, MA)
- KHÔNG TỰ TÍNH các mức hỗ trợ/kháng cự
- KHÔNG TỰ XÁC ĐỊNH xu hướng
- Chỉ sử dụng các giá trị đã cung cấp và tập trung vào việc nhận diện mẫu hình

Hãy đảm bảo kết quả phân tích:
- Có mức độ tin cậy rõ ràng (High/Medium/Low) cho mỗi mẫu hình
- Có ý nghĩa cụ thể của mẫu hình đối với triển vọng giá
- Có khuyến nghị phù hợp với xu hướng và chỉ báo được cung cấp

Chỉ trả về JSON, không thêm văn bản giới thiệu/kết luận."""

        # Chuẩn bị thông tin kỹ thuật đã tính
        indicators_text = f"""
CHỈ BÁO KỸ THUẬT (ĐÃ TÍNH SẴN - KHÔNG CẦN TÍNH LẠI):
- RSI(14): {technical_indicators.get('rsi_14', 'N/A')}
- MACD: {technical_indicators.get('macd', 'N/A')}
- MACD Signal: {technical_indicators.get('macd_signal', 'N/A')}
- MACD Histogram: {technical_indicators.get('macd_hist', 'N/A')}
- MA20: {technical_indicators.get('ma_20', 'N/A')}
- MA50: {technical_indicators.get('ma_50', 'N/A')}
- MA200: {technical_indicators.get('ma_200', 'N/A')}

MỨC GIÁ QUAN TRỌNG (ĐÃ TÍNH SẴN - KHÔNG CẦN TÍNH LẠI):
- Hỗ trợ: {', '.join(map(str, support_levels))}
- Kháng cự: {', '.join(map(str, resistance_levels))}

XU HƯỚNG HIỆN TẠI: {trend}
"""

        # Chuẩn bị user prompt
        user_prompt = f"""Phân tích kỹ thuật cho cổ phiếu {symbol}:

DỮ LIỆU GIÁ GẦN ĐÂY:
{data_text}

THÔNG TIN GIÁ HIỆN TẠI:
- Giá hiện tại: {price_info.get('current_price', 'N/A')}
- Thay đổi: {price_info.get('price_change', 'N/A')} ({price_info.get('price_change_percent', 'N/A')}%)

{indicators_text}
"""

        # Thêm thông tin thị trường nếu có
        if additional_context:
            user_prompt += "THÔNG TIN THỊ TRƯỜNG BỔ SUNG:\n"
            for key, value in additional_context.items():
                user_prompt += f"- {key}: {value}\n"
        
        # Cấu trúc JSON đầu ra, đơn giản hóa và tập trung vào mẫu hình
        json_structure = """
YÊU CẦU: Trả về kết quả phân tích dưới dạng JSON với cấu trúc sau:
{
  "symbol": "Mã cổ phiếu",
  "current_price": giá hiện tại (số),
  "price_change": thay đổi giá (số),
  "price_change_percent": phần trăm thay đổi (số),
  
  "technical_patterns": [
    {
      "pattern_name": "Tên mẫu hình kỹ thuật",
      "confidence": "Độ tin cậy (chỉ dùng: High, Medium, Low)",
      "description": "Mô tả ngắn gọn về mẫu hình đã phát hiện",
      "implications": "Ý nghĩa đối với xu hướng giá trong tương lai",
      "price_target": "Mục tiêu giá dựa trên mẫu hình (nếu xác định được)"
    }
  ],
  
  "candlestick_patterns": [
    {
      "pattern_name": "Tên mẫu hình nến",
      "confidence": "Độ tin cậy (High/Medium/Low)",
      "implications": "Ý nghĩa của mẫu hình nến này"
    }
  ],
  
  "wave_analysis": {
    "direction": "Hướng sóng chính (Up/Down/Sideways)",
    "main_wave": "Tên/số sóng chính",
    "subwaves": ["Tên/số sóng con 1", "Tên/số sóng con 2", ...],
    "interpretation": "Diễn giải ý nghĩa sóng",
    "confidence": "Độ tin cậy (High/Medium/Low)",
    "next_move": "Dự đoán bước tiếp theo của chu kỳ sóng (nếu có thể)"
  },
  
  "overall_rating": {
    "recommendation": "Khuyến nghị (chỉ dùng: Mua/Bán/Nắm giữ)",
    "time_frame": "Khung thời gian (Ngắn/Trung/Dài hạn)",
    "risk_level": "Mức độ rủi ro (Thấp/Trung bình/Cao)",
    "summary": "Tóm tắt đánh giá tổng thể",
    "rationale": "Lý do chi tiết cho khuyến nghị dựa trên mẫu hình kỹ thuật",
    "special_notes": "Lưu ý đặc biệt về điểm cần theo dõi (nếu có)"
  }
}

LƯU Ý QUAN TRỌNG: 
1. JSON phải hợp lệ 100%, không có lỗi cú pháp
2. Phân tích phải phù hợp với chỉ báo kỹ thuật đã cung cấp
3. Không tự tính lại các chỉ báo, chỉ tập trung vào nhận diện mẫu hình
4. Chỉ trả về JSON, không thêm văn bản giới thiệu hoặc kết luận
5. Đảm bảo khuyến nghị (Mua/Bán/Nắm giữ) phù hợp với mẫu hình đã nhận diện"""

        return system_prompt, user_prompt

    def generate_content_sync(self, prompt: str, system_prompt: str = None, temperature: float = None, max_tokens: int = None, top_p: float = None) -> str:
        """
        Gọi Groq API đồng bộ để sinh nội dung (dạng chat completion, trả về string JSON hoặc text)
        Args:
            prompt: Nội dung user prompt
            system_prompt: Nội dung system prompt (nếu có)
            temperature: Nhiệt độ sampling (nếu có)
            max_tokens: Số token tối đa (nếu có)
            top_p: top_p sampling (nếu có)
        Returns:
            Nội dung phản hồi (string) hoặc None nếu lỗi
        """
        # Đảm bảo prompt hoặc system_prompt có từ 'json' (bắt buộc với response_format json_object)
        combined_prompt = (prompt or "") + " " + (system_prompt or "")
        if "json" not in combined_prompt.lower():
            prompt += "\nHãy trả về kết quả dưới dạng JSON hợp lệ."
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or "Bạn là chuyên gia phân tích kỹ thuật."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature if temperature is not None else self.generation_config["temperature"],
            "max_tokens": max_tokens if max_tokens is not None else self.generation_config["max_tokens"],
            "top_p": top_p if top_p is not None else self.generation_config["top_p"],
            "response_format": {"type": "json_object"}
        }
        retries = 0
        while retries < self.retry_config["max_retries"]:
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.post(
                        self.api_endpoint,
                        headers=self.headers,
                        json=payload
                    )
                    if response.status_code == 200:
                        result = response.json()
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Groq API trả về lỗi: {response.status_code} - {response.text}")
                        if 500 <= response.status_code < 600:
                            retries += 1
                            time.sleep(self.retry_config["retry_delay"] * (2 ** retries))
                            continue
                        return None
            except Exception as e:
                logger.error(f"Lỗi khi gọi Groq API (sync): {str(e)}")
                retries += 1
                if retries < self.retry_config["max_retries"]:
                    time.sleep(self.retry_config["retry_delay"] * (2 ** retries))
                else:
                    return None
        return None 