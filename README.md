# Enhanced Data Modules for Vietnam Stock Bot

This package provides enhanced data handling capabilities for the Vietnam Stock Bot, offering improved data loading, quality control, processing, and automation.

## Modules

### 1. Enhanced Data Loader (`data_loader_advanced.py`)
Advanced data loader with the following features:
- Support for multiple timeframes including 1h, 4h from Yahoo Finance
- Automatic fallback between data sources (Yahoo Finance and VNStock)
- Data source reliability tracking and intelligent source selection
- Smart caching with Redis
- Error handling with retry logic
- Automatic outlier detection
- Timestamp alignment

### 2. Data Quality Control (`data_quality_control.py`)
Quality assessment system that evaluates data based on:
- Completeness: Checks for missing data
- Consistency: Verifies data follows expected patterns
- Timeliness: Ensures data is up-to-date
- Validity: Validates data against expected ranges and formats
- Accuracy: Detects anomalies and outliers

Provides quality scoring, reporting, and alerting capabilities.

### 3. Advanced Data Processor (`advanced_data_processor.py`)
Sophisticated data processing with:
- Multiple methods for outlier detection (Z-score, IQR, Isolation Forest, DBSCAN)
- Various outlier handling strategies (winsorizing, replacement, removal)
- Advanced missing data imputation techniques
- Rich feature engineering for technical analysis
- Data normalization and memory optimization

### 4. Data Automation Manager (`data_automation_manager.py`)
Automation system that manages:
- Scheduled data loading via cron-like scheduling
- Incremental data updates
- Periodic quality checks
- Cache cleanup and optimization
- Alerting for data issues

### 5. Timestamp Utilities (`timestamp_utils.py`)
Utilities for handling time-related operations:
- Timestamp alignment for different timeframes
- Trading period calculations
- Timezone handling
- Missing period filling
- Multi-timeframe data merging

## Integration

The `integration_example.py` file demonstrates how to integrate these modules with the main bot.

## Usage

1. Import the modules:
```python
from data_loader_advanced import EnhancedDataLoader
from data_quality_control import DataQualityControl
from advanced_data_processor import AdvancedDataProcessor
from data_automation_manager import DataAutomationManager
from timestamp_utils import align_timestamps
```

2. Initialize components:
```python
# Initialize with Redis manager and VNStock client from main bot
redis_manager = RedisManager()  # From main bot
vnstock_client = Vnstock()     # From main bot

# Create enhanced loader
loader = EnhancedDataLoader(
    redis_manager=redis_manager,
    vnstock_client=vnstock_client
)

# Create quality control
quality_control = DataQualityControl(redis_manager=redis_manager)

# Create data processor
processor = AdvancedDataProcessor(redis_manager=redis_manager)

# Create automation manager
automation = DataAutomationManager(
    redis_manager=redis_manager,
    enhanced_loader=loader,
    quality_control=quality_control,
    data_processor=processor
)
```

3. Load and process data:
```python
# Load data with enhanced features
df, report = await loader.load_data(
    symbol='VCB',
    timeframe='1d',
    num_candles=100,
    detect_outliers=True
)

# Check quality
quality_result = await quality_control.check_data_quality(df, 'VCB', '1d')

# Process data
df_with_features = processor.create_derived_features(df)
```

4. Set up automation:
```python
# Add symbols to track
await automation.add_tracked_symbol('VCB', timeframes=['1d', '1w'])
await automation.add_tracked_symbol('VNINDEX', timeframes=['1d', '1w', '1mo'])

# Start automation
await automation.initialize()
```

## Requirements
- Python 3.7+
- Redis
- Pandas
- NumPy
- scikit-learn
- yfinance
- vnstock (from main bot)
- redis-manager (from main bot)

## Note
These modules are designed to work alongside the main bot without modifying its source code. 