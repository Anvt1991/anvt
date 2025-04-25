#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Automation Manager Module for Stock Market Bot:
- Manages automated data tasks with scheduling
- Performs periodic data quality checks
- Handles incremental data updates
- Optimizes memory usage and cleans old cache
- Monitors data systems and sends alerts
"""

import os
import logging
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable, Set
import traceback

# Create module-specific logger
logger = logging.getLogger(__name__)

class DataAutomationManager:
    """
    Manages automated data tasks such as scheduled data loading, quality checks,
    and cache maintenance.
    """
    
    def __init__(
        self, 
        redis_manager, 
        enhanced_loader, 
        quality_control, 
        data_processor,
        logger=None
    ):
        """
        Initialize the automation manager with required components.
        
        Args:
            redis_manager: Redis manager from the main bot
            enhanced_loader: Instance of EnhancedDataLoader
            quality_control: Instance of DataQualityControl
            data_processor: Instance of AdvancedDataProcessor
            logger: Optional custom logger
        """
        self.redis_manager = redis_manager
        self.loader = enhanced_loader
        self.quality_control = quality_control
        self.processor = data_processor
        
        # Set up logging
        self.logger = logger or logging.getLogger("DataAutomation")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Tracked symbols and their timeframes
        self.tracked_symbols: Dict[str, List[str]] = {}
        
        # Task scheduling state
        self.is_running = False
        self.tasks = []
        self.last_run_times = {}
        
        # Default schedules (in seconds)
        self.default_schedules = {
            '1m': 60,          # Every minute
            '5m': 300,         # Every 5 minutes
            '15m': 900,        # Every 15 minutes
            '30m': 1800,       # Every 30 minutes
            '1h': 3600,        # Every hour
            '4h': 14400,       # Every 4 hours
            '1d': 86400,       # Every day
            '1w': 604800,      # Every week
            '1mo': 2592000     # Every month (30 days)
        }
        
        # Task schedules in seconds (when each type of task should run)
        self.data_load_schedule = self.default_schedules.copy()
        self.quality_check_schedule = {k: v * 3 for k, v in self.default_schedules.items()}  # Less frequent
        self.cache_cleanup_interval = 86400  # Once a day
        
        # Market hours (Vietnam time - UTC+7)
        self.market_open_hour = 9   # 9 AM
        self.market_close_hour = 15  # 3 PM
        
        # Alert thresholds
        self.quality_alert_threshold = 0.7  # Alert if quality score < 70%
        
        # Cache settings
        self.max_cache_age = {
            '1m': 60 * 60 * 24,       # 1 day for 1m data
            '5m': 60 * 60 * 24 * 3,    # 3 days for 5m data
            '15m': 60 * 60 * 24 * 7,   # 7 days for 15m data
            '30m': 60 * 60 * 24 * 7,   # 7 days for 30m data
            '1h': 60 * 60 * 24 * 14,   # 14 days for 1h data
            '4h': 60 * 60 * 24 * 30,   # 30 days for 4h data
            '1d': 60 * 60 * 24 * 90,   # 90 days for 1d data
            '1w': 60 * 60 * 24 * 180,  # 180 days for 1w data
            '1mo': 60 * 60 * 24 * 365, # 365 days for 1mo data
        }
    
    async def add_tracked_symbol(self, symbol: str, timeframes: List[str]) -> None:
        """
        Add a symbol to be tracked and automatically updated.
        
        Args:
            symbol: Stock symbol to track
            timeframes: List of timeframes to track for this symbol
        """
        self.tracked_symbols[symbol] = timeframes
        self.logger.info(f"Added symbol {symbol} to tracking with timeframes: {timeframes}")
        
        # Initialize last run times if needed
        for timeframe in timeframes:
            key = f"{symbol}_{timeframe}"
            if key not in self.last_run_times:
                # Set to a time that will trigger immediate loading
                self.last_run_times[key] = {
                    'data_load': time.time() - self.data_load_schedule[timeframe],
                    'quality_check': time.time() - self.quality_check_schedule[timeframe]
                }
    
    async def remove_tracked_symbol(self, symbol: str) -> None:
        """
        Remove a symbol from automatic tracking.
        
        Args:
            symbol: Stock symbol to remove from tracking
        """
        if symbol in self.tracked_symbols:
            timeframes = self.tracked_symbols.pop(symbol)
            self.logger.info(f"Removed symbol {symbol} from tracking")
            
            # Clean up last run times
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                if key in self.last_run_times:
                    del self.last_run_times[key]
    
    async def initialize(self) -> None:
        """Initialize automation and start background tasks."""
        self.logger.info("Initializing data automation")
        
        # Perform initial data load for all tracked symbols
        await self.load_all_tracked_data(force=True)
        
        # Start the main automation loop
        if not self.is_running:
            self.is_running = True
            self.tasks.append(asyncio.create_task(self._automation_loop()))
            self.tasks.append(asyncio.create_task(self._cache_maintenance_loop()))
            self.logger.info("Data automation tasks started")
    
    async def shutdown(self) -> None:
        """Stop all automation tasks."""
        self.logger.info("Shutting down data automation")
        self.is_running = False
        
        for task in self.tasks:
            task.cancel()
            
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
        self.logger.info("Data automation tasks stopped")
    
    async def _automation_loop(self) -> None:
        """Main automation loop that schedules data loading and quality checks."""
        self.logger.info("Starting main automation loop")
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check each symbol and timeframe
                for symbol, timeframes in self.tracked_symbols.items():
                    for timeframe in timeframes:
                        key = f"{symbol}_{timeframe}"
                        
                        # Check if it's time to load new data
                        if (current_time - self.last_run_times[key]['data_load'] >= 
                            self.data_load_schedule[timeframe]):
                            
                            # Only load data during market hours for intraday timeframes
                            if self._should_update_data(timeframe):
                                self.logger.debug(f"Scheduling data load for {symbol} {timeframe}")
                                asyncio.create_task(self._load_and_process_data(symbol, timeframe))
                                self.last_run_times[key]['data_load'] = current_time
                        
                        # Check if it's time to run quality check
                        if (current_time - self.last_run_times[key]['quality_check'] >= 
                            self.quality_check_schedule[timeframe]):
                            
                            self.logger.debug(f"Scheduling quality check for {symbol} {timeframe}")
                            asyncio.create_task(self._run_quality_check(symbol, timeframe))
                            self.last_run_times[key]['quality_check'] = current_time
                
                # Sleep to prevent CPU overuse
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in automation loop: {str(e)}")
                await asyncio.sleep(30)  # Longer sleep on error
    
    async def _cache_maintenance_loop(self) -> None:
        """Loop that handles cache cleanup and optimization."""
        self.logger.info("Starting cache maintenance loop")
        
        while self.is_running:
            try:
                # Run cache cleanup once a day
                await self._cleanup_old_cache()
                await asyncio.sleep(self.cache_cleanup_interval)
                
            except Exception as e:
                self.logger.error(f"Error in cache maintenance loop: {str(e)}")
                await asyncio.sleep(3600)  # Sleep for an hour on error
    
    def _should_update_data(self, timeframe: str) -> bool:
        """
        Determine if data should be updated based on market hours and timeframe.
        
        Args:
            timeframe: The timeframe to check
            
        Returns:
            bool: True if data should be updated, False otherwise
        """
        # For daily, weekly, monthly data, always update
        if timeframe in ['1d', '1w', '1mo']:
            return True
            
        # For intraday data, only update during market hours
        now = datetime.now()
        
        # Check if it's a weekday (0 = Monday, 6 = Sunday)
        if now.weekday() >= 5:  # Weekend
            return False
            
        # Check if it's during market hours
        if not (self.market_open_hour <= now.hour < self.market_close_hour):
            return False
            
        return True
    
    async def _load_and_process_data(self, symbol: str, timeframe: str) -> None:
        """
        Load and process data for a symbol and timeframe.
        
        Args:
            symbol: The stock symbol
            timeframe: The timeframe to load
        """
        try:
            self.logger.info(f"Loading data for {symbol} {timeframe}")
            
            # Determine appropriate number of candles based on timeframe
            num_candles = self._get_num_candles_for_timeframe(timeframe)
            
            # Load data with enhanced features
            df, report = await self.loader.load_data(
                symbol=symbol,
                timeframe=timeframe,
                num_candles=num_candles,
                detect_outliers=True
            )
            
            if df is not None and not df.empty:
                # Process the data and add derived features
                df_with_features = self.processor.create_derived_features(df)
                
                # Save the processed data to Redis
                key = f"processed:{symbol}:{timeframe}"
                await self.redis_manager.store_dataframe(key, df_with_features)
                
                # Log success
                self.logger.info(f"Successfully loaded and processed {symbol} {timeframe} data, {len(df)} candles")
                
                # Check if any outliers were detected or fixed
                if report and 'outliers_detected' in report and report['outliers_detected'] > 0:
                    self.logger.warning(
                        f"Data quality issue: {report['outliers_detected']} outliers detected in {symbol} {timeframe}"
                    )
            else:
                self.logger.error(f"Failed to load data for {symbol} {timeframe}: empty dataframe")
                
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} {timeframe}: {str(e)}")
    
    async def _run_quality_check(self, symbol: str, timeframe: str) -> None:
        """
        Run a quality check on loaded data.
        
        Args:
            symbol: The stock symbol
            timeframe: The timeframe to check
        """
        try:
            self.logger.info(f"Running quality check for {symbol} {timeframe}")
            
            # Retrieve the data from Redis
            key = f"processed:{symbol}:{timeframe}"
            df = await self.redis_manager.get_dataframe(key)
            
            if df is not None and not df.empty:
                # Run quality check
                quality_result = await self.quality_control.check_data_quality(df, symbol, timeframe)
                
                # Store quality report
                quality_key = f"quality:{symbol}:{timeframe}"
                await self.redis_manager.set_json(quality_key, quality_result)
                
                # Log quality score
                overall_score = quality_result.get('overall_score', 0)
                self.logger.info(f"Quality score for {symbol} {timeframe}: {overall_score:.2f}")
                
                # Generate alert if quality is below threshold
                if overall_score < self.quality_alert_threshold:
                    self.logger.warning(
                        f"Data quality alert: {symbol} {timeframe} has a low quality score of {overall_score:.2f}"
                    )
            else:
                self.logger.error(f"Cannot run quality check for {symbol} {timeframe}: data not found")
                
        except Exception as e:
            self.logger.error(f"Error running quality check for {symbol} {timeframe}: {str(e)}")
    
    async def _cleanup_old_cache(self) -> None:
        """Clean up old cache data that's no longer needed."""
        try:
            self.logger.info("Running cache cleanup")
            
            # Get all keys for stock data
            all_keys = await self.redis_manager.get_keys("processed:*")
            current_time = time.time()
            
            for key in all_keys:
                try:
                    # Parse key to get symbol and timeframe
                    parts = key.split(":")
                    if len(parts) < 3:
                        continue
                        
                    timeframe = parts[2]
                    
                    # Skip if timeframe not recognized
                    if timeframe not in self.max_cache_age:
                        continue
                    
                    # Get last updated time
                    last_updated = await self.redis_manager.get_meta(f"{key}:updated")
                    if not last_updated:
                        continue
                        
                    # Convert to timestamp if it's not already
                    if isinstance(last_updated, str):
                        try:
                            last_updated = float(last_updated)
                        except ValueError:
                            continue
                    
                    # Check if data is too old
                    if current_time - last_updated > self.max_cache_age[timeframe]:
                        self.logger.info(f"Removing old cache data for {key}")
                        await self.redis_manager.delete(key)
                        await self.redis_manager.delete(f"{key}:updated")
                
                except Exception as e:
                    self.logger.error(f"Error processing cache key {key}: {str(e)}")
            
            self.logger.info("Cache cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cache cleanup: {str(e)}")
    
    async def load_all_tracked_data(self, force: bool = False) -> None:
        """
        Load data for all tracked symbols and timeframes.
        
        Args:
            force: If True, load data regardless of schedule
        """
        self.logger.info("Loading all tracked data")
        
        tasks = []
        for symbol, timeframes in self.tracked_symbols.items():
            for timeframe in timeframes:
                if force or self._should_update_data(timeframe):
                    tasks.append(self._load_and_process_data(symbol, timeframe))
        
        if tasks:
            await asyncio.gather(*tasks)
            self.logger.info(f"Completed loading all tracked data ({len(tasks)} items)")
    
    def _get_num_candles_for_timeframe(self, timeframe: str) -> int:
        """
        Determine appropriate number of candles based on timeframe.
        
        Args:
            timeframe: The timeframe to check
            
        Returns:
            int: Number of candles to load
        """
        # Default values based on timeframe
        defaults = {
            '1m': 1000,
            '5m': 1000,
            '15m': 1000,
            '30m': 500,
            '1h': 500,
            '4h': 250,
            '1d': 250,
            '1w': 200,
            '1mo': 120
        }
        
        return defaults.get(timeframe, 250)  # Default to 250 if timeframe not found
    
    async def get_automation_status(self) -> Dict:
        """
        Get the current status of the automation system.
        
        Returns:
            Dict: Status information
        """
        status = {
            'is_running': self.is_running,
            'tracked_symbols': self.tracked_symbols,
            'last_run_times': {},
            'scheduled_tasks': len(self.tasks),
            'cache_stats': await self._get_cache_stats()
        }
        
        # Format last run times for readability
        for key, times in self.last_run_times.items():
            status['last_run_times'][key] = {
                'data_load': datetime.fromtimestamp(times['data_load']).strftime('%Y-%m-%d %H:%M:%S'),
                'quality_check': datetime.fromtimestamp(times['quality_check']).strftime('%Y-%m-%d %H:%M:%S')
            }
        
        return status
    
    async def _get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache.
        
        Returns:
            Dict: Cache statistics
        """
        try:
            # Count the number of processed items
            processed_keys = await self.redis_manager.get_keys("processed:*")
            
            # Group by timeframe
            timeframe_counts = {}
            for key in processed_keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    timeframe = parts[2]
                    timeframe_counts[timeframe] = timeframe_counts.get(timeframe, 0) + 1
            
            # Get total size estimate (rough calculation)
            total_size = 0
            for key in processed_keys[:min(10, len(processed_keys))]:  # Sample up to 10 keys
                df = await self.redis_manager.get_dataframe(key)
                if df is not None:
                    # Rough estimate of dataframe size in bytes
                    size = df.memory_usage(deep=True).sum()
                    total_size += size
            
            # Extrapolate for all keys
            if processed_keys and len(processed_keys) > 10:
                total_size = total_size * (len(processed_keys) / min(10, len(processed_keys)))
            
            # Convert to MB
            total_size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
            
            return {
                'total_items': len(processed_keys),
                'by_timeframe': timeframe_counts,
                'estimated_size_mb': round(total_size_mb, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {str(e)}")
            return {'error': str(e)} 