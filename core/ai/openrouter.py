import os
import requests
import logging

class OpenRouterHandler:
    def __init__(self, api_key=None, model=None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3-0324:free")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

    def generate_response(self, messages, model=None):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model or self.model,
            "messages": messages
        }
        try:
            response = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            resp_json = response.json()
            if isinstance(resp_json, dict) and 'choices' in resp_json and isinstance(resp_json['choices'], list) and resp_json['choices']:
                return resp_json['choices'][0]['message']['content']
            else:
                logging.error(f"OpenRouter trả về dict không có 'choices': {resp_json}")
                return str(resp_json)
        except Exception as e:
            logging.error(f"Lỗi gọi OpenRouter: {e}")
            try:
                logging.error(f"Response text: {response.text}")
            except:
                pass
            return f"Lỗi gọi OpenRouter: {e}" 