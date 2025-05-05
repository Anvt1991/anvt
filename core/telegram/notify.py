import os
import logging
import requests
import html
import time

logger = logging.getLogger("telegram_notify")

TELEGRAM_MAX_LENGTH = 4096


def send_report_to_telegram(report_text: str, chat_id: str = None, bot_token: str = None, parse_mode: str = "HTML", disable_notification: bool = False, max_retries: int = 3) -> bool:
    """
    Gửi báo cáo tới Telegram, hỗ trợ HTML/Markdown, tự động cắt nếu quá dài, retry, log chi tiết.
    """
    if not report_text or not isinstance(report_text, str) or not report_text.strip():
        logger.warning("Không gửi Telegram vì report_text rỗng hoặc None")
        return False
    bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        logger.warning("Thiếu bot_token hoặc chat_id khi gửi Telegram")
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    # Xử lý định dạng
    if parse_mode.upper() == "HTML":
        safe_report = f"<pre>{html.escape(report_text)}</pre>"
    else:
        safe_report = report_text
    # Cắt báo cáo nếu quá dài
    messages = []
    text = safe_report
    while len(text) > TELEGRAM_MAX_LENGTH:
        messages.append(text[:TELEGRAM_MAX_LENGTH])
        text = text[TELEGRAM_MAX_LENGTH:]
    if text:
        messages.append(text)
    success = True
    for idx, msg in enumerate(messages):
        for attempt in range(max_retries):
            payload = {
                "chat_id": chat_id,
                "text": msg,
                "parse_mode": parse_mode,
                "disable_notification": disable_notification
            }
            try:
                response = requests.post(url, data=payload, timeout=10)
                if response.status_code == 200:
                    logger.info(f"Đã gửi báo cáo Telegram thành công (phần {idx+1}/{len(messages)})")
                    break
                else:
                    logger.error(f"Telegram API error (phần {idx+1}): {response.text}")
                    if attempt == max_retries - 1:
                        success = False
                time.sleep(1)
            except Exception as e:
                logger.error(f"Lỗi gửi Telegram (phần {idx+1}): {e}")
                if attempt == max_retries - 1:
                    success = False
                time.sleep(1)
    return success 