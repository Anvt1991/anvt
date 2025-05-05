from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """
    Sinh khuyến nghị đầu tư dựa trên các score tổng hợp và volatility.
    """
    def __init__(self, config: dict):
        self.config = config

    def generate(self, ml_scores: Dict[str, float], tech_signals: Dict[str, Any], 
                 sentiment_data: Optional[Dict[str, Any]] = None, 
                 market_condition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate recommendation from scores
        
        Args:
            ml_scores: Machine learning scores
            tech_signals: Technical analysis signals
            sentiment_data: News sentiment data
            market_condition: Market condition data
            
        Returns:
            Dictionary with recommendation
        """
        try:
            # Extract scores
            if not isinstance(ml_scores, dict):
                return {"action": "HOLD", "confidence": 0, "reasoning": "Invalid ML scores format"}
            
            short_term_score = ml_scores.get("short_term", 0)
            medium_term_score = ml_scores.get("medium_term", 0)
            
            # Extract technical signals
            tech_score = 0
            if tech_signals and isinstance(tech_signals, dict):
                buy_signals = sum(1 for signal in tech_signals.values() if signal == "BUY")
                sell_signals = sum(1 for signal in tech_signals.values() if signal == "SELL")
                if buy_signals + sell_signals > 0:
                    tech_score = (buy_signals - sell_signals) / (buy_signals + sell_signals)
            
            # Extract sentiment score
            sentiment_score = 0
            if sentiment_data and isinstance(sentiment_data, dict):
                sentiment_score = sentiment_data.get("overall_score", 0)
            
            # Calculate a combined score with weights
            weight_ml_short = self.config.get("weights", {}).get("ml_short", 0.5)
            weight_ml_medium = self.config.get("weights", {}).get("ml_medium", 0.3)
            weight_tech = self.config.get("weights", {}).get("technical", 0.15)
            weight_sentiment = self.config.get("weights", {}).get("sentiment", 0.05)
            
            weighted_score = (
                weight_ml_short * short_term_score +
                weight_ml_medium * medium_term_score +
                weight_tech * tech_score +
                weight_sentiment * sentiment_score
            )
            
            # Determine action and confidence
            action = "HOLD"
            confidence = abs(weighted_score)
            
            if weighted_score > self.config.get("thresholds", {}).get("buy", 0.2):
                action = "BUY"
            elif weighted_score < self.config.get("thresholds", {}).get("sell", -0.2):
                action = "SELL"
            
            # Cap confidence at 1.0
            confidence = min(confidence, 1.0)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(action, weighted_score, ml_scores, tech_signals, sentiment_data)
            
            return {
                "action": action,
                "confidence": round(confidence, 2),
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return {"action": "HOLD", "confidence": 0, "reasoning": f"Error: {str(e)}"}

    def get_combined_recommendation(self, ml_scores: Dict[str, float], tech_signals: Dict[str, Any], 
                                    sentiment_data: Optional[Dict[str, Any]] = None, 
                                    market_condition: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate combined recommendation from all data sources
        Wrapper for generate method to ensure backward compatibility
        """
        return self.generate(ml_scores, tech_signals, sentiment_data, market_condition)
    
    def _generate_reasoning(self, action: str, weighted_score: float, 
                          ml_scores: Dict[str, float], 
                          tech_signals: Dict[str, Any], 
                          sentiment_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate reasoning text for recommendation
        
        Args:
            action: Recommendation action
            weighted_score: Combined weighted score
            ml_scores: ML model scores
            tech_signals: Technical analysis signals
            sentiment_data: News sentiment data
            
        Returns:
            Reasoning text
        """
        if not isinstance(ml_scores, dict) or not ml_scores:
            return "Không đủ dữ liệu để đưa ra phân tích chi tiết."
        
        # Extract values for reasoning
        short_term = ml_scores.get("short_term", 0)
        medium_term = ml_scores.get("medium_term", 0)
        
        # Count technical signals
        buy_signals = 0
        sell_signals = 0
        if tech_signals and isinstance(tech_signals, dict):
            buy_signals = sum(1 for signal in tech_signals.values() if signal == "BUY")
            sell_signals = sum(1 for signal in tech_signals.values() if signal == "SELL")
        
        # Sentiment 
        sentiment_description = "trung lập"
        if sentiment_data and isinstance(sentiment_data, dict):
            sentiment_score = sentiment_data.get("overall_score", 0)
            if sentiment_score > 0.2:
                sentiment_description = "tích cực"
            elif sentiment_score < -0.2:
                sentiment_description = "tiêu cực"
        
        # Calculate confidence for text
        confidence = min(abs(weighted_score), 1.0)
        
        # Generate reasoning based on action
        reasoning = ""
        if short_term > 0.2:
            reasoning += "ML ngắn hạn dự báo tăng. "
        elif short_term < -0.2:
            reasoning += "ML ngắn hạn dự báo giảm. "
            
        if medium_term > 0.2:
            reasoning += "ML trung hạn dự báo tăng. "
        elif medium_term < -0.2:
            reasoning += "ML trung hạn dự báo giảm. "
            
        if buy_signals > sell_signals:
            reasoning += f"Có {buy_signals} tín hiệu mua và {sell_signals} tín hiệu bán từ phân tích kỹ thuật. "
        elif sell_signals > buy_signals:
            reasoning += f"Có {sell_signals} tín hiệu bán và {buy_signals} tín hiệu mua từ phân tích kỹ thuật. "
            
        if sentiment_description != "trung lập":
            reasoning += f"Tin tức và thị trường {sentiment_description}. "
        
        if action == "BUY":
            reasoning = f"Mua với độ tin cậy {confidence:.2f}. {reasoning}"
        elif action == "SELL":
            reasoning = f"Bán với độ tin cậy {confidence:.2f}. {reasoning}"
        else:
            reasoning = f"Nắm giữ với độ tin cậy {confidence:.2f}. {reasoning}"
        
        return reasoning 