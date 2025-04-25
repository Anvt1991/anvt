#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Modules for Vietnam Stock Bot

This package provides enhanced data handling capabilities:
- Advanced data loading with fallback mechanisms
- Data quality control and validation
- Sophisticated data processing and feature engineering
- Automatic data management and scheduling
- Timestamp alignment and utility functions
"""

__version__ = '1.0.0'

from .data_loader_advanced import EnhancedDataLoader, DataSourceTracker
from .data_quality_control import DataQualityControl
from .advanced_data_processor import AdvancedDataProcessor
from .data_automation_manager import DataAutomationManager
from .timestamp_utils import (
    align_timestamps, normalize_timeframe, resample_data,
    ensure_timezone, get_trading_periods, fill_missing_periods,
    is_trading_time, merge_timeframes
)

__all__ = [
    'EnhancedDataLoader',
    'DataSourceTracker',
    'DataQualityControl',
    'AdvancedDataProcessor',
    'DataAutomationManager',
    'align_timestamps',
    'normalize_timeframe',
    'resample_data',
    'ensure_timezone',
    'get_trading_periods',
    'fill_missing_periods',
    'is_trading_time',
    'merge_timeframes'
] 