#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Quality Control Module for Stock Market Bot:
- Evaluates data quality based on 5 criteria: completeness, consistency, timeliness, validity, and accuracy
- Provides comprehensive quality reporting and alerting
- Implements automated quality checks and scoring
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union

# Create module-specific logger
logger = logging.getLogger(__name__)

class DataQualityControl:
    """
    Provides comprehensive data quality assessment for financial market data
    
    Evaluates data based on five key criteria:
    1. Completeness: Check for missing values, incomplete series, etc.
    2. Consistency: Ensure data follows expected patterns and relationships
    3. Timeliness: Verify data is up-to-date and covers the expected time periods
    4. Validity: Check that data values are within acceptable ranges and formats
    5. Accuracy: Assess data accuracy through cross-validation and anomaly detection
    """
    
    def __init__(self, redis_manager=None):
        """Initialize the data quality control"""
        self.redis_manager = redis_manager
        
        # Quality threshold levels
        self.quality_thresholds = {
            'poor': 0.6,     # Below this is considered poor quality
            'medium': 0.8,   # Below this is considered medium quality
            'good': 0.9      # Below this is considered good quality (above is excellent)
        }
        
        # Weight for each quality criteria (must sum to 1.0)
        self.quality_weights = {
            'completeness': 0.25,
            'consistency': 0.2,
            'timeliness': 0.2,
            'validity': 0.2,
            'accuracy': 0.15
        }
        
        # Default thresholds for specific checks
        self.default_thresholds = {
            'missing_data_threshold': 0.05,       # Max acceptable % of missing values
            'max_price_change': 0.2,              # Max acceptable price change between candles
            'max_gap_days': 5,                    # Max acceptable number of days between candles
            'volatility_threshold': 0.05,         # Threshold for excessive volatility
            'volume_consistency_threshold': 0.1,  # Threshold for volume consistency
            'timeliness_max_days': 3,             # Max days old for daily data
            'timeliness_max_hours': 1,            # Max hours old for hourly data
        }
        
    async def check_data_quality(self, df: pd.DataFrame, symbol: str, 
                                timeframe: str) -> Dict[str, Union[float, Dict, str]]:
        """
        Perform a comprehensive quality check on market data
        
        Parameters:
        -----------
        df : pandas DataFrame
            The market data to check
        symbol : str
            The symbol/ticker of the data
        timeframe : str
            The timeframe of the data ('1m', '1h', '1d', etc.)
            
        Returns:
        --------
        dict
            Quality assessment results with scores and details
        """
        if df is None or df.empty:
            return {
                'overall_score': 0.0,
                'quality_rating': 'Poor',
                'details': 'Empty or missing data',
                'scores': {k: 0.0 for k in self.quality_weights.keys()},
                'issues': ['No data available for quality assessment']
            }
        
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Ensure proper columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return {
                    'overall_score': 0.0,
                    'quality_rating': 'Poor',
                    'details': f'Missing required column: {col}',
                    'scores': {k: 0.0 for k in self.quality_weights.keys()},
                    'issues': [f'Missing required column: {col}']
                }
        
        # Run individual quality checks
        completeness_score, completeness_issues = self._check_completeness(df, timeframe)
        consistency_score, consistency_issues = self._check_consistency(df, timeframe)
        timeliness_score, timeliness_issues = self._check_timeliness(df, timeframe)
        validity_score, validity_issues = self._check_validity(df)
        accuracy_score, accuracy_issues = self._check_accuracy(df, symbol)
        
        # Compute weighted average score
        scores = {
            'completeness': completeness_score,
            'consistency': consistency_score,
            'timeliness': timeliness_score,
            'validity': validity_score,
            'accuracy': accuracy_score
        }
        
        overall_score = sum(scores[k] * self.quality_weights[k] for k in scores.keys())
        
        # Determine quality rating
        if overall_score < self.quality_thresholds['poor']:
            quality_rating = 'Poor'
        elif overall_score < self.quality_thresholds['medium']:
            quality_rating = 'Medium'
        elif overall_score < self.quality_thresholds['good']:
            quality_rating = 'Good'
        else:
            quality_rating = 'Excellent'
        
        # Compile all issues
        all_issues = (
            completeness_issues + 
            consistency_issues + 
            timeliness_issues + 
            validity_issues + 
            accuracy_issues
        )
        
        # Store quality report in Redis if available
        if self.redis_manager:
            report_key = f"quality_report_{symbol}_{timeframe}"
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'timeframe': timeframe,
                'overall_score': overall_score,
                'quality_rating': quality_rating,
                'scores': scores,
                'issues': all_issues
            }
            asyncio.create_task(self.redis_manager.set(report_key, report_data, expire=86400))
        
        # Return complete quality report
        return {
            'overall_score': overall_score,
            'quality_rating': quality_rating,
            'details': f"Data quality is {quality_rating.lower()} ({overall_score:.2f})",
            'scores': scores,
            'issues': all_issues
        }
    
    def _check_completeness(self, df: pd.DataFrame, timeframe: str) -> Tuple[float, List[str]]:
        """Check data completeness"""
        issues = []
        
        # Check for missing values
        missing_pct = df[['open', 'high', 'low', 'close', 'volume']].isna().mean().mean()
        if missing_pct > 0:
            issues.append(f"Missing values: {missing_pct:.2%} of data points are missing")
        
        # Check for expected time intervals based on timeframe
        if df.index.name != 'date' and df.index.name != 'Date':
            df = df.copy()
            df.index.name = 'date'
        
        # Expected frequency based on timeframe
        freq_map = {
            '1m': 'T', '5m': '5T', '15m': '15T', '30m': '30T',
            '1h': 'H', '4h': '4H', '1d': 'B', '1w': 'W', '1mo': 'M'
        }
        
        # Normalize timeframe
        normalized_tf = timeframe.lower().replace(' ', '')
        aliases = {
            '1hour': '1h', '1hr': '1h', '60m': '1h',
            '4hour': '4h', '4hr': '4h', '240m': '4h',
            '1day': '1d', 'daily': '1d', 'day': '1d',
            '1week': '1w', 'weekly': '1w', 'week': '1w',
            '1month': '1mo', 'monthly': '1mo', 'month': '1mo'
        }
        normalized_tf = aliases.get(normalized_tf, normalized_tf)
        
        # Check for gaps in time series
        if normalized_tf in freq_map:
            freq = freq_map[normalized_tf]
            
            # For business day frequency, only check on business days
            if freq == 'B':
                # Create expected index from min to max date, business days only
                idx = pd.bdate_range(start=df.index.min(), end=df.index.max())
                
                # Convert actual index to dates for proper comparison
                actual_dates = df.index.normalize().unique()
                
                # Find business days that are missing (weekdays only)
                missing_dates = idx.difference(actual_dates)
                missing_dates = [d for d in missing_dates if d.weekday() < 5]
                
                if len(missing_dates) > 0:
                    issues.append(f"Missing {len(missing_dates)} business days in the data")
                    
                # Calculate completeness based on percentage of business days covered
                business_days_expected = len(idx)
                business_days_actual = len(actual_dates)
                
                if business_days_expected > 0:
                    completeness_ratio = business_days_actual / business_days_expected
                else:
                    completeness_ratio = 1.0  # No expected business days (unlikely)
                
            else:
                # For intraday and other timeframes, check expected frequency
                # Note: this is more complex and would require market open hours
                # For simplicity, we'll check for large gaps
                
                df_sorted = df.sort_index()
                if len(df_sorted) > 1:
                    # Calculate time deltas between consecutive timestamps
                    time_deltas = df_sorted.index[1:] - df_sorted.index[:-1]
                    
                    # Expected maximum delta based on timeframe
                    max_delta_map = {
                        'T': timedelta(minutes=2),
                        '5T': timedelta(minutes=10),
                        '15T': timedelta(minutes=30),
                        '30T': timedelta(minutes=60),
                        'H': timedelta(hours=2),
                        '4H': timedelta(hours=8),
                        'W': timedelta(days=14),
                        'M': timedelta(days=45)
                    }
                    max_expected_delta = max_delta_map.get(freq, timedelta(days=3))
                    
                    # Find large gaps
                    large_gaps = time_deltas > max_expected_delta
                    if large_gaps.any():
                        num_gaps = large_gaps.sum()
                        issues.append(f"Found {num_gaps} large time gaps in the data")
                    
                    # Calculate completeness based on gaps
                    if len(time_deltas) > 0:
                        completeness_ratio = 1.0 - (large_gaps.sum() / len(time_deltas))
                    else:
                        completeness_ratio = 1.0  # Only one record
                else:
                    completeness_ratio = 0.5  # Too few records to assess properly
        
        else:
            # For unsupported timeframes, give a default score
            completeness_ratio = 0.7
            issues.append(f"Unsupported timeframe for completeness check: {timeframe}")
        
        # Additional penalty for missing values
        completeness_score = completeness_ratio * (1 - missing_pct)
        
        # Ensure the score is between 0 and 1
        completeness_score = max(0.0, min(1.0, completeness_score))
        
        if not issues:
            issues.append("Data is complete with no missing values or time gaps")
        
        return completeness_score, issues
    
    def _check_consistency(self, df: pd.DataFrame, timeframe: str) -> Tuple[float, List[str]]:
        """Check data consistency"""
        issues = []
        
        # Price relationship checks
        price_relation_errors = ((df['high'] < df['low']) | 
                                (df['close'] < df['low']) | 
                                (df['close'] > df['high']) |
                                (df['open'] < df['low']) |
                                (df['open'] > df['high']))
        
        num_price_errors = price_relation_errors.sum()
        if num_price_errors > 0:
            issues.append(f"Price relationship inconsistencies in {num_price_errors} rows")
        
        # Volume consistency: check for zero volumes during trading hours
        zero_volumes = (df['volume'] == 0).sum()
        if zero_volumes > 0:
            zero_volume_pct = zero_volumes / len(df)
            if zero_volume_pct > self.default_thresholds['volume_consistency_threshold']:
                issues.append(f"High proportion of zero volumes: {zero_volume_pct:.2%}")
        
        # Extreme price changes (potential data errors)
        df_sorted = df.sort_index()
        pct_changes = df_sorted['close'].pct_change().abs()
        extreme_changes = pct_changes > self.default_thresholds['max_price_change']
        num_extreme_changes = extreme_changes.sum()
        
        if num_extreme_changes > 0:
            extreme_change_pct = num_extreme_changes / len(df)
            if extreme_change_pct > 0.01:  # More than 1% of data points
                issues.append(f"Detected {num_extreme_changes} extreme price changes")
        
        # Calculate consistency score
        # Base score on various consistency factors
        if len(df) > 1:
            # Price relationship consistency
            price_relation_score = 1.0 - (num_price_errors / len(df))
            
            # Volume consistency
            volume_score = 1.0 - min(1.0, (zero_volume_pct / self.default_thresholds['volume_consistency_threshold']) 
                                    if zero_volumes > 0 else 0)
            
            # Price change consistency
            price_change_score = 1.0 - min(1.0, extreme_change_pct * 10)
            
            # Overall consistency score (weighted average)
            consistency_score = 0.5 * price_relation_score + 0.25 * volume_score + 0.25 * price_change_score
        else:
            consistency_score = 0.5  # Not enough data to fully assess
            issues.append("Insufficient data to fully assess consistency")
        
        # Ensure the score is between 0 and 1
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        if not issues:
            issues.append("Data is consistent with no price relationship errors or extreme changes")
        
        return consistency_score, issues
    
    def _check_timeliness(self, df: pd.DataFrame, timeframe: str) -> Tuple[float, List[str]]:
        """Check if data is up-to-date"""
        issues = []
        
        # Check if we have data from recent periods based on timeframe
        now = datetime.now()
        latest_data_time = df.index.max()
        
        # Convert to datetime if it's not already
        if not isinstance(latest_data_time, (datetime, pd.Timestamp)):
            issues.append("Index is not a datetime type, cannot assess timeliness")
            return 0.5, issues
        
        # Make timezone-aware if needed
        if latest_data_time.tzinfo is None:
            latest_data_time = latest_data_time.tz_localize('UTC')
            now = now.astimezone()
        
        # Time difference from now
        time_diff = now - latest_data_time
        
        # Normalize timeframe for comparison
        normalized_tf = timeframe.lower().replace(' ', '')
        
        # Expected maximum age of data based on timeframe
        if normalized_tf in ['1m', '5m', '15m', '30m']:
            max_age = timedelta(hours=1)
            if time_diff > max_age:
                issues.append(f"Data is outdated by {time_diff}, expected maximum age: {max_age}")
        
        elif normalized_tf in ['1h', '4h']:
            max_age = timedelta(hours=self.default_thresholds['timeliness_max_hours'])
            if time_diff > max_age:
                issues.append(f"Data is outdated by {time_diff}, expected maximum age: {max_age}")
        
        elif normalized_tf in ['1d', '1day', 'daily', 'day']:
            # For daily data, consider weekends and holidays
            max_age = timedelta(days=self.default_thresholds['timeliness_max_days'])
            
            # If today is Monday and latest data is from Friday, that's OK
            if now.weekday() == 0 and latest_data_time.weekday() == 4:
                max_age = timedelta(days=3)  # 3 days for Monday with Friday data
            
            # If today is Sunday and latest data is from Friday, that's OK
            elif now.weekday() == 6 and latest_data_time.weekday() == 4:
                max_age = timedelta(days=2)  # 2 days for Sunday with Friday data
            
            # If today is Saturday and latest data is from Friday, that's OK
            elif now.weekday() == 5 and latest_data_time.weekday() == 4:
                max_age = timedelta(days=1)  # 1 day for Saturday with Friday data
            
            if time_diff > max_age:
                issues.append(f"Daily data is outdated by {time_diff}, expected maximum age: {max_age}")
        
        elif normalized_tf in ['1w', '1week', 'weekly', 'week']:
            max_age = timedelta(days=10)  # More flexible for weekly data
            if time_diff > max_age:
                issues.append(f"Weekly data is outdated by {time_diff}, expected maximum age: {max_age}")
        
        elif normalized_tf in ['1mo', '1month', 'monthly', 'month']:
            max_age = timedelta(days=35)  # More flexible for monthly data
            if time_diff > max_age:
                issues.append(f"Monthly data is outdated by {time_diff}, expected maximum age: {max_age}")
        
        else:
            # Default case
            max_age = timedelta(days=5)
            if time_diff > max_age:
                issues.append(f"Data may be outdated by {time_diff}, timeframe: {timeframe}")
        
        # Calculate timeliness score based on age relative to max age
        if not issues:
            timeliness_score = 1.0
            issues.append("Data is up-to-date")
        else:
            # Score based on how far beyond max age we are
            age_ratio = time_diff.total_seconds() / max_age.total_seconds()
            timeliness_score = max(0.0, 1.0 - min(1.0, age_ratio - 1.0))
        
        return timeliness_score, issues
    
    def _check_validity(self, df: pd.DataFrame) -> Tuple[float, List[str]]:
        """Check if data values are valid and within expected ranges"""
        issues = []
        
        # Check for negative prices
        neg_prices = ((df['open'] < 0) | (df['high'] < 0) | 
                      (df['low'] < 0) | (df['close'] < 0)).sum()
        if neg_prices > 0:
            issues.append(f"Found {neg_prices} instances of negative prices")
        
        # Check for negative volumes
        neg_volumes = (df['volume'] < 0).sum()
        if neg_volumes > 0:
            issues.append(f"Found {neg_volumes} instances of negative volume")
        
        # Check for extremely high values (potential errors)
        # For prices
        price_mean = df['close'].mean()
        price_std = df['close'].std()
        extreme_high_threshold = price_mean + 10 * price_std
        extreme_prices = ((df['open'] > extreme_high_threshold) | 
                         (df['high'] > extreme_high_threshold) | 
                         (df['low'] > extreme_high_threshold) | 
                         (df['close'] > extreme_high_threshold)).sum()
        
        if extreme_prices > 0:
            issues.append(f"Found {extreme_prices} instances of potentially erroneous high prices")
        
        # For volumes
        volume_mean = df['volume'].mean()
        volume_std = df['volume'].std()
        extreme_volume_threshold = volume_mean + 10 * volume_std
        extreme_volumes = (df['volume'] > extreme_volume_threshold).sum()
        
        if extreme_volumes > 0:
            issues.append(f"Found {extreme_volumes} instances of potentially erroneous high volumes")
        
        # Calculate validity score
        if len(df) > 0:
            # Calculate various validity scores
            price_validity = 1.0 - (neg_prices / len(df))
            volume_validity = 1.0 - (neg_volumes / len(df))
            extreme_price_validity = 1.0 - (extreme_prices / len(df))
            extreme_volume_validity = 1.0 - (extreme_volumes / len(df))
            
            # Combine scores with weights
            validity_score = (0.4 * price_validity + 
                             0.2 * volume_validity + 
                             0.3 * extreme_price_validity + 
                             0.1 * extreme_volume_validity)
        else:
            validity_score = 0.0
            issues.append("No data available to check validity")
        
        # Ensure the score is between 0 and 1
        validity_score = max(0.0, min(1.0, validity_score))
        
        if not issues:
            issues.append("All data values are within valid ranges")
        
        return validity_score, issues
    
    def _check_accuracy(self, df: pd.DataFrame, symbol: str) -> Tuple[float, List[str]]:
        """
        Check data accuracy through cross-validation and anomaly detection
        
        This is the most complex check as true accuracy often requires external validation.
        We focus on internal consistency and reasonableness checks.
        """
        issues = []
        
        # Check for outliers in the data
        if 'close' in df.columns and len(df) > 3:
            # Z-score method for outlier detection
            z_scores = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
            outliers = z_scores > 3.0
            num_outliers = outliers.sum()
            
            if num_outliers > 0:
                outlier_pct = num_outliers / len(df)
                if outlier_pct > 0.05:  # More than 5% outliers is concerning
                    issues.append(f"High proportion of outliers detected: {outlier_pct:.2%}")
                else:
                    issues.append(f"Found {num_outliers} potential outliers in price data")
        
        # Volatility check - periods of unusual volatility might indicate data issues
        if len(df) > 5:
            # Calculate rolling volatility
            returns = df['close'].pct_change().dropna()
            rolling_std = returns.rolling(window=5).std().dropna()
            
            # Check for unusually high volatility
            high_vol_periods = rolling_std > self.default_thresholds['volatility_threshold']
            num_high_vol = high_vol_periods.sum()
            
            if num_high_vol > 0:
                high_vol_pct = num_high_vol / len(rolling_std)
                if high_vol_pct > 0.1:  # More than 10% high volatility periods
                    issues.append(f"Unusual volatility detected in {high_vol_pct:.2%} of the data")
        
        # Check for artificial patterns (e.g., repeating values)
        if len(df) > 10:
            # Check for consecutive identical values
            identical_closes = (df['close'].shift() == df['close']).sum()
            identical_close_pct = identical_closes / (len(df) - 1)
            
            if identical_close_pct > 0.2:  # More than 20% identical consecutive closes
                issues.append(f"Unusual pattern of identical consecutive closes: {identical_close_pct:.2%}")
        
        # Calculate accuracy score
        if len(df) > 5:
            # Base score starts high and gets reduced for issues
            accuracy_score = 1.0
            
            # Penalize for outliers
            if 'outlier_pct' in locals():
                accuracy_score -= min(0.5, outlier_pct * 5)
            
            # Penalize for high volatility
            if 'high_vol_pct' in locals():
                accuracy_score -= min(0.3, high_vol_pct * 2)
            
            # Penalize for artificial patterns
            if 'identical_close_pct' in locals():
                accuracy_score -= min(0.4, identical_close_pct * 1.5)
        
        else:
            accuracy_score = 0.6  # Not enough data to fully assess
            issues.append("Insufficient data to fully assess accuracy")
        
        # Ensure the score is between 0 and 1
        accuracy_score = max(0.0, min(1.0, accuracy_score))
        
        if not issues:
            issues.append("Data appears accurate with no significant outliers or anomalies")
        
        return accuracy_score, issues

    async def generate_quality_report(self, symbol: str, timeframe: str, df: pd.DataFrame) -> str:
        """Generate a human-readable quality report"""
        quality_data = await self.check_data_quality(df, symbol, timeframe)
        
        report = f"📊 Data Quality Report for {symbol} ({timeframe})\n"
        report += f"🏅 Overall Score: {quality_data['overall_score']:.2f} - {quality_data['quality_rating']}\n\n"
        
        report += "🔍 Quality Dimensions:\n"
        for dim, score in quality_data['scores'].items():
            report += f"• {dim.capitalize()}: {score:.2f}\n"
        
        report += "\n❗ Issues Detected:\n"
        if quality_data['issues']:
            for issue in quality_data['issues']:
                report += f"• {issue}\n"
        else:
            report += "• No issues detected\n"
        
        return report
    
    def optimize_memory_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize memory usage of DataFrame by changing data types"""
        if df is None or df.empty:
            return df
        
        df_optimized = df.copy()
        
        # Optimize numeric columns
        float_cols = df_optimized.select_dtypes(include=['float']).columns
        for col in float_cols:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        int_cols = df_optimized.select_dtypes(include=['int']).columns
        for col in int_cols:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        # Log memory savings
        original_memory = df.memory_usage(deep=True).sum()
        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        savings = 1 - (optimized_memory / original_memory)
        
        logger.info(f"Memory optimization: {original_memory/1024**2:.2f} MB → {optimized_memory/1024**2:.2f} MB ({savings:.2%} saved)")
        
        return df_optimized 