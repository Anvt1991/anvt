from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SignalAggregator:
    """
    Tổng hợp các tín hiệu: technical, sentiment, market, ML thành 1 score tổng hợp.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = self.config.get("predictor", {}).get("weights", {
            "technical": 0.30,
            "sentiment": 0.15,
            "ai": 0.45,
            "market": 0.10
        })

    def aggregate(self, ml_scores: Dict[str, float], tech_signals: Dict[str, Any] = None,
                 sentiment_data: Optional[Dict[str, Any]] = None,
                 market_condition: Optional[Dict[str, Any]] = None) -> float:
        """
        Aggregate different signals into a unified score
        
        Args:
            ml_scores: ML model scores, containing short_term, medium_term, etc.
            tech_signals: Technical analysis signals
            sentiment_data: News sentiment data
            market_condition: Market condition data
            
        Returns:
            Aggregated score between -1.0 and 1.0
        """
        # Extract ML scores
        if not isinstance(ml_scores, dict):
            logger.warning("Invalid ml_scores format in signal aggregator")
            return 0.0
        
        ml_short_term = ml_scores.get("short_term", 0.0)
        ml_medium_term = ml_scores.get("medium_term", 0.0)
        
        # Extract technical score
        tech_score = 0.0
        if tech_signals and isinstance(tech_signals, dict):
            buy_signals = sum(1 for signal in tech_signals.values() if signal == "BUY")
            sell_signals = sum(1 for signal in tech_signals.values() if signal == "SELL")
            if buy_signals + sell_signals > 0:
                tech_score = (buy_signals - sell_signals) / (buy_signals + sell_signals)
        
        # Extract sentiment score
        sentiment_score = 0.0
        if sentiment_data and isinstance(sentiment_data, dict):
            sentiment_score = sentiment_data.get("overall_score", 0.0)
        
        # Extract market score
        market_score = 0.0
        if market_condition and isinstance(market_condition, dict):
            market_score = market_condition.get("overall_score", 0.0)
        
        try:
            # Apply weights to each signal
            weight_ml_short = self.weights.get("ml_short", 0.4)
            weight_ml_medium = self.weights.get("ml_medium", 0.2)
            weight_tech = self.weights.get("technical", 0.2)
            weight_sentiment = self.weights.get("sentiment", 0.1)
            weight_market = self.weights.get("market", 0.1)
            
            # Calculate weighted average (normalize to ensure weights sum to 1.0)
            weights_sum = weight_ml_short + weight_ml_medium + weight_tech + weight_sentiment + weight_market
            if weights_sum == 0:
                weights_sum = 1.0
                
            weighted_score = (
                weight_ml_short * ml_short_term +
                weight_ml_medium * ml_medium_term +
                weight_tech * tech_score + 
                weight_sentiment * sentiment_score +
                weight_market * market_score
            ) / weights_sum
            
            # Ensure score is between -1 and 1
            return max(min(weighted_score, 1.0), -1.0)
        except Exception as e:
            logger.error(f"Error in signal aggregation: {str(e)}")
            return 0.0

    def aggregate_legacy(self, tech_score: float, sentiment_score: float, 
                       market_score: float, ml_score: float) -> float:
        """
        Legacy method for backward compatibility
        """
        return self.aggregate(
            ml_scores={"short_term": ml_score, "medium_term": ml_score},
            tech_signals={} if tech_score == 0 else {"legacy": "BUY" if tech_score > 0 else "SELL"},
            sentiment_data={"overall_score": sentiment_score} if sentiment_score != 0 else None,
            market_condition={"overall_score": market_score} if market_score != 0 else None
        ) 