import logging
import asyncio
import json
import httpx
from typing import Dict, Any
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ai.base_ai_analyzer import BaseAIAnalyzer
from app.services.fundamental_analysis import deep_fundamental_analysis
from app.utils.config import OPENROUTER_API_KEY

logger = logging.getLogger(__name__)

class OpenRouterAnalyzer(BaseAIAnalyzer):
    """
    AI analyzer using OpenRouter API to generate reports and analysis.
    """
    
    def __init__(self):
        self.api_key = OPENROUTER_API_KEY
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.semaphore = asyncio.Semaphore(3)  # Limit concurrent API calls
        
        # Models available through OpenRouter (miễn phí)
        self.models = {
            "default": "deepseek/deepseek-chat-v3-0324:free",
            "premium": "deepseek/deepseek-coder:free",
            "balanced": "deepseek/deepseek-chat-v3-0324:free",
            "affordable": "mistralai/mistral-7b-instruct:free"
        }
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://stockbot.vn"  # Replace with your actual domain
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_content(self, prompt: str, model_key: str = "default") -> str:
        """Generate content using OpenRouter API."""
        # Use semaphore to limit concurrent API calls
        async with self.semaphore:
            try:
                model = self.models.get(model_key, self.models["default"])
                
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a professional stock market analyst specializing in Vietnamese market. Provide concise, accurate analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1500  # Giảm token output để tối ưu sử dụng quotas
                }
                
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        return content
                    else:
                        logger.error(f"Unexpected API response format: {result}")
                        return "Error: Unexpected API response format"
                
            except httpx.HTTPError as e:
                logger.error(f"HTTP error from OpenRouter API: {str(e)}")
                if hasattr(e, "response") and e.response is not None:
                    logger.error(f"Response status: {e.response.status_code}")
                    logger.error(f"Response content: {e.response.text}")
                raise
                
            except Exception as e:
                logger.error(f"Error generating content with OpenRouter: {str(e)}")
                raise
    
    async def analyze_technical_data(self, technical_data: Dict[str, Any]) -> str:
        """Analyze technical market data using AI."""
        try:
            # Extract key information for prompt
            symbol = technical_data.get('symbol', 'Unknown')
            
            # Build a prompt with relevant technical data
            prompt_parts = [
                f"Analyze the technical indicators for {symbol} and provide insights on the current market conditions.",
                "\nFocus on the following aspects:",
                "1. Current trend direction (bullish, bearish, or sideways)",
                "2. Key support and resistance levels",
                "3. Potential entry or exit points",
                "4. Risk assessment",
                "\nHere's the data for your analysis:\n"
            ]
            
            # Add last candle data for each timeframe
            last_candle_data = technical_data.get('last_candle', {})
            for timeframe, candle in last_candle_data.items():
                if candle:
                    prompt_parts.append(f"\n{timeframe} Timeframe:")
                    prompt_parts.append(f"- Close: {candle.get('close', 'N/A')}")
                    prompt_parts.append(f"- Change: {candle.get('change_percent', 'N/A'):.2f}%" if 'change_percent' in candle else "- Change: N/A")
                    
                    # Add key indicators
                    for indicator in ['rsi_14', 'macd', 'macd_signal', 'adx']:
                        if indicator in candle:
                            prompt_parts.append(f"- {indicator.upper()}: {candle[indicator]:.2f}" if isinstance(candle[indicator], (int, float)) else f"- {indicator.upper()}: {candle[indicator]}")
            
            # Add pattern information
            patterns = technical_data.get('patterns', {})
            if patterns:
                prompt_parts.append("\nDetected Patterns:")
                for timeframe, tf_patterns in patterns.items():
                    if tf_patterns:
                        prompt_parts.append(f"\n{timeframe} Timeframe Patterns:")
                        for pattern_name, pattern_info in tf_patterns.items():
                            if pattern_info.get('detected'):
                                direction = "Bullish" if pattern_info.get('bullish') else "Bearish"
                                prompt_parts.append(f"- {pattern_name}: {direction}")
            
            prompt = "\n".join(prompt_parts)
            
            # Generate analysis using LLM
            analysis = await self.generate_content(prompt, model_key="balanced")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing technical data: {str(e)}")
            return f"Error analyzing technical data: {str(e)}"
    
    async def generate_report(self, dfs: Dict[str, Any], symbol: str, 
                             fundamental_data: Dict[str, Any], 
                             outlier_reports: Dict[str, Any]) -> str:
        """Generate a comprehensive stock analysis report."""
        try:
            # Prepare technical data
            technical_data = {
                'symbol': symbol,
                'last_candle': {},
                'patterns': {}
            }
            
            # Extract last candle info for each timeframe
            for timeframe, df in dfs.items():
                if df is not None and not df.empty:
                    from app.services.technical_analysis import TechnicalAnalyzer
                    analyzer = TechnicalAnalyzer()
                    technical_data['last_candle'][timeframe] = analyzer.extract_last_candle_info(df)
            
            # Detect patterns
            from app.services.technical_analysis import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer()
            technical_data['patterns'] = analyzer.detect_patterns(dfs)
            
            # Get fundamental analysis
            fundamental_analysis = deep_fundamental_analysis(fundamental_data)
            
            # Get technical analysis from AI
            technical_analysis = await self.analyze_technical_data(technical_data)
            
            # Build comprehensive report
            report_parts = [
                f"# Báo cáo phân tích {symbol}",
                f"Ngày: {time.strftime('%Y-%m-%d')}",
                "\n## Phân tích kỹ thuật:",
                technical_analysis,
                "\n## Phân tích cơ bản:",
                fundamental_analysis
            ]
            
            # Add outlier information if available
            if outlier_reports:
                report_parts.append("\n## Cảnh báo giá trị bất thường:")
                for timeframe, report in outlier_reports.items():
                    if report and report != "Không có giá trị bất thường":
                        report_parts.append(f"\n### Timeframe {timeframe}:")
                        report_parts.append(report)
            
            # Add conclusion
            conclusion_prompt = f"""
            Based on the technical and fundamental analysis for {symbol}, provide a concise conclusion with:
            1. A summary of the current situation
            2. Potential short-term outlook
            3. Key risks to watch
            
            Technical Analysis: {technical_analysis}
            
            Fundamental Analysis: {fundamental_analysis}
            """
            
            conclusion = await self.generate_content(conclusion_prompt, model_key="balanced")
            report_parts.append("\n## Kết luận:")
            report_parts.append(conclusion)
            
            full_report = "\n\n".join(report_parts)
            return full_report
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return f"Error generating report: {str(e)}" 