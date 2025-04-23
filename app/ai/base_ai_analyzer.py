import abc
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class BaseAIAnalyzer(abc.ABC):
    """
    Abstract base class for AI analysis services.
    All AI analyzers should implement this interface.
    """
    
    @abc.abstractmethod
    async def generate_content(self, prompt: str) -> str:
        """
        Generate content based on the given prompt.
        
        Args:
            prompt: The prompt to send to the AI model.
            
        Returns:
            Generated text response from the AI model.
        """
        pass
    
    @abc.abstractmethod
    async def analyze_technical_data(self, technical_data: Dict[str, Any]) -> str:
        """
        Analyze technical market data and generate insights.
        
        Args:
            technical_data: Dictionary containing technical analysis data.
            
        Returns:
            Analysis and insights based on the technical data.
        """
        pass
    
    @abc.abstractmethod
    async def generate_report(self, dfs: Dict[str, Any], symbol: str, 
                             fundamental_data: Dict[str, Any], 
                             outlier_reports: Dict[str, Any]) -> str:
        """
        Generate a comprehensive analysis report for a symbol.
        
        Args:
            dfs: Dictionary of dataframes with technical indicators for different timeframes.
            symbol: The stock symbol being analyzed.
            fundamental_data: Dictionary of fundamental data for the symbol.
            outlier_reports: Dictionary of outlier detection reports.
            
        Returns:
            A comprehensive analysis report.
        """
        pass 