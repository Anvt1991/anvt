import requests
import json

# 👉 Thay bằng token bot thật của bạn
BOT_TOKEN = "7780930655:AAEPJ77fbGwtDeCj-jCzVUuA-rZgGxPsuMM"

# Gọi getUpdates
url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

try:
    response = requests.get(url)
    response.raise_for_status()  # Kiểm tra lỗi HTTP

    data = response.json()

    if data.get("ok") and data.get("result"):
        for update in data["result"]:
            if "channel_post" in update:
                chat_id = update["channel_post"]["chat"]["id"]
                title = update["channel_post"]["chat"]["title"]
                print(f"📢 Channel Name: {title}")
                print(f"🆔 Channel ID: {chat_id}")
            else:
                print("⚠️ Không có bài đăng từ kênh trong update này.")
    else:
        print("⚠️ Không có dữ liệu hợp lệ hoặc bot chưa được thêm vào kênh.")

except requests.exceptions.RequestException as e:
    print(f"❌ Lỗi HTTP: {e}")
except Exception as ex:
    print(f"❌ Lỗi khác: {ex}")
