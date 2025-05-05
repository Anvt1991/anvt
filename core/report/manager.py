import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio

logger = logging.getLogger("report_manager")

def optimize_merged_data_for_ai(merged_data, max_news=5, max_raw=30, max_prophet=5):
    # Chỉ giữ các trường chính
    keys_to_keep = [
        "symbol", "current_price", "price_change", "price_change_percent",
        "indicators", "technical_patterns", "candlestick_patterns",
        "support_resistance", "overall_rating", "trend",
        "news_sentiment", "meta",
        # Bổ sung các trường do Groq/ML sinh ra
        "wave_analysis", "ml_scores", "abnormal_events", "group_leaders",
        "market_breadth", "top_influencers", "liquidity_note",
        "wyckoff_analysis"  # Bổ sung trường Wyckoff
    ]
    # Thêm các trường đặc biệt nếu có
    if "prophet_forecast" in merged_data:
        keys_to_keep.append("prophet_forecast")
    if "news" in merged_data:
        keys_to_keep.append("news")
    if "vnindex_last" in merged_data:
        keys_to_keep.append("vnindex_last")
    if "vnindex_change" in merged_data:
        keys_to_keep.append("vnindex_change")
    if "vnindex_change_pct" in merged_data:
        keys_to_keep.append("vnindex_change_pct")
    if "volume" in merged_data:
        keys_to_keep.append("volume")
    # Tạo dict rút gọn
    optimized = {k: merged_data[k] for k in keys_to_keep if k in merged_data}
    # Rút gọn news
    if "news" in optimized and isinstance(optimized["news"], list):
        optimized["news"] = optimized["news"][:max_news]
    # Rút gọn raw_data nếu cần
    if "raw_data" in merged_data and isinstance(merged_data["raw_data"], dict):
        last_keys = sorted(merged_data["raw_data"].keys())[-max_raw:]
        optimized["raw_data"] = {k: merged_data["raw_data"][k] for k in last_keys}
    # Rút gọn prophet_forecast nếu có
    if "prophet_forecast" in optimized and isinstance(optimized["prophet_forecast"], dict):
        pf = optimized["prophet_forecast"]
        if "forecast_df" in pf and hasattr(pf["forecast_df"], "head"):
            optimized["prophet_forecast"]["forecast_df"] = pf["forecast_df"].head(max_prophet)
    return optimized

class ReportManager:
    def __init__(self, db_manager: Optional[object] = None, gemini_handler: Optional[object] = None):
        from core.data.db import DBManager
        if db_manager is not None and not isinstance(db_manager, DBManager):
            logger.warning(f"[ReportManager] db_manager truyền vào không phải DBManager mà là {type(db_manager)}. Sẽ tự động khởi tạo DBManager mới.")
        # Nếu db_manager không phải là DBManager, khởi tạo DBManager
        self.db = db_manager if isinstance(db_manager, DBManager) else DBManager()
        self.gemini = gemini_handler
        self.openrouter = None

    async def generate_report(self, merged_data: Dict[str, Any], symbol: str, meta: dict = None) -> Dict[str, Any]:
        # Rút gọn merged_data trước khi truyền vào AI
        merged_data = optimize_merged_data_for_ai(merged_data)
        # Import GeminiHandler và OpenRouterHandler động
        try:
            from core.ai.gemini import GeminiHandler
            from core.ai.openrouter import OpenRouterHandler
        except ImportError as e:
            logger.error(f"Không thể import AI handler: {e}")
            return {"success": False, "error": str(e), "report": None}
        gemini = self.gemini or GeminiHandler()
        openrouter = self.openrouter or OpenRouterHandler()
        # Ưu tiên Gemini, fallback OpenRouter nếu lỗi hoặc không có báo cáo
        try:
            report = None
            gemini_error = None
            try:
                report = gemini.generate_report_sync(
                    pattern_data=merged_data,
                    market_data=merged_data,
                    report_type="market" if symbol.upper() == "VNINDEX" else "technical",
                    output_format="markdown"
                )
                # Nếu Gemini trả về None hoặc báo cáo lỗi, fallback
                if not report or "Không thể tạo báo cáo" in str(report) or "ERROR" in str(report):
                    gemini_error = f"Gemini trả về lỗi hoặc None: {report}"
                    raise Exception(gemini_error)
            except Exception as e:
                logger.warning(f"Gemini lỗi: {e}, fallback OpenRouter...")
                try:
                    report = openrouter.generate_response([
                        {"role": "system", "content": f"Bạn là chuyên gia phân tích chứng khoán, hãy viết báo cáo cho mã {symbol}"},
                        {"role": "user", "content": str(merged_data)}
                    ])
                    # Kiểm tra response của OpenRouter
                    if isinstance(report, dict):
                        if 'choices' in report and isinstance(report['choices'], list) and report['choices']:
                            try:
                                report = report['choices'][0]['message']['content']
                            except Exception as parse_e:
                                logger.error(f"Lỗi parse response OpenRouter: {parse_e}, response: {report}")
                                report = str(report)
                        else:
                            logger.error(f"OpenRouter trả về dict không có 'choices': {report}")
                            report = str(report)
                    elif isinstance(report, str):
                        pass  # OK
                    else:
                        logger.warning(f"OpenRouter trả về response không phải string: {report}")
                        report = str(report)
                except Exception as oe:
                    logger.error(f"Lỗi sinh báo cáo từ OpenRouter: {oe}")
                    report = f"Không thể sinh báo cáo từ OpenRouter: {oe}"
            return {"success": True, "report": report}
        except Exception as e:
            logger.error(f"Lỗi sinh báo cáo: {e}")
            return {"success": False, "error": str(e), "report": None}

    async def save_report(self, symbol: str, report: str, close_today: float = None, close_yesterday: float = None, timeframe: str = '1D') -> bool:
        # Import DBManager động
        try:
            from core.data.db import DBManager
        except ImportError as e:
            logger.error(f"Không thể import DBManager: {e}")
            return False
        db = self.db or DBManager()
        try:
            # Truyền đúng 5 tham số (symbol, report, close_today, close_yesterday, timeframe)
            await db.save_report_history(symbol, report, close_today, close_yesterday, timeframe)
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu báo cáo vào DB: {e}")
            return False

    async def send_report(self, report_text: str, chat_id: str = None, bot_token: str = None) -> bool:
        # Import notify động
        try:
            from core.telegram.notify import send_report_to_telegram
        except ImportError as e:
            logger.error(f"Không thể import send_report_to_telegram: {e}")
            return False
        try:
            result = send_report_to_telegram(report_text, chat_id=chat_id, bot_token=bot_token)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            logger.error(f"Lỗi gửi báo cáo Telegram: {e}")
            return False

    async def create_and_send_report(self, merged_data: Dict[str, Any], symbol: str, meta: dict = None) -> Dict[str, Any]:
        # Rút gọn merged_data trước khi truyền vào AI
        merged_data = optimize_merged_data_for_ai(merged_data)
        # Sinh báo cáo
        report_result = await self.generate_report(merged_data, symbol, meta=meta)
        if not report_result.get("success") or not report_result.get("report"):
            logger.error(f"Không thể sinh báo cáo cho {symbol}: {report_result.get('error', 'Không rõ lỗi')}")
            return {"success": False, "error": report_result.get("error", "Không thể sinh báo cáo"), "report": None}
        report = report_result["report"]
        # Kiểm tra nội dung báo cáo hợp lệ trước khi lưu
        if not report or not isinstance(report, str) or report.strip() == "" or "lỗi" in report.lower() or "error" in report.lower() or "không thể" in report.lower():
            logger.error(f"Báo cáo không hợp lệ, không lưu vào DB cho {symbol}: {report}")
            return {"success": False, "error": "Báo cáo không hợp lệ, không lưu vào DB", "report": report}
        # Lưu DB
        close_today = meta.get("close_today") if meta else None
        close_yesterday = meta.get("close_yesterday") if meta else None
        save_ok = await self.save_report(symbol, report, close_today, close_yesterday)
        if not save_ok:
            logger.error(f"Không thể lưu báo cáo vào DB cho {symbol}")
            return {"success": False, "error": "Không thể lưu báo cáo vào DB", "report": report}
        # Gửi Telegram
        send_ok = await self.send_report(report)
        if not send_ok:
            logger.error(f"Không thể gửi báo cáo Telegram cho {symbol}")
            return {"success": False, "error": "Không thể gửi báo cáo Telegram", "report": report, "saved": save_ok, "sent": send_ok}
        return {"success": True, "report": report, "saved": save_ok, "sent": send_ok} 