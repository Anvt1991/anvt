#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Integration Example for Data Modules with Stock Bot

This file demonstrates how to integrate the enhanced data modules with the main bot.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime

# Import enhanced data modules
from data_loader_advanced import EnhancedDataLoader
from data_quality_control import DataQualityControl
from advanced_data_processor import AdvancedDataProcessor
from data_automation_manager import DataAutomationManager
from timestamp_utils import align_timestamps, normalize_timeframe

# You'll need to import the main bot's components
# This is just placeholder - update with your actual imports
from redis_manager import RedisManager  # Assumed to be from main bot
from vnstock import Vnstock  # Assumed to be from main bot

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_in_thread(func, *args, **kwargs):
    """Helper to run blocking functions in a thread pool"""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

async def initialize_components():
    """Initialize all data components"""
    
    # Initialize Redis Manager (from main bot)
    redis_manager = RedisManager()
    await redis_manager.connect()
    
    # Initialize VNStock client (from main bot)
    vnstock_client = Vnstock()
    
    # Initialize enhanced data components
    enhanced_loader = EnhancedDataLoader(
        redis_manager=redis_manager,
        vnstock_client=vnstock_client,
        run_in_thread=run_in_thread
    )
    await enhanced_loader.initialize()
    
    quality_control = DataQualityControl(redis_manager=redis_manager)
    
    data_processor = AdvancedDataProcessor(redis_manager=redis_manager)
    
    automation_manager = DataAutomationManager(
        redis_manager=redis_manager,
        enhanced_loader=enhanced_loader,
        quality_control=quality_control,
        data_processor=data_processor
    )
    await automation_manager.initialize()
    
    # Add symbols to track
    await automation_manager.add_tracked_symbol('VNINDEX', timeframes=['1d', '1w', '1mo'])
    await automation_manager.add_tracked_symbol('VN30', timeframes=['1d', '1w', '1mo'])
    await automation_manager.add_tracked_symbol('MSN', timeframes=['1h', '4h', '1d'])
    await automation_manager.add_tracked_symbol('VCB', timeframes=['1h', '4h', '1d'])
    
    # Return all components
    return {
        'redis_manager': redis_manager,
        'enhanced_loader': enhanced_loader,
        'quality_control': quality_control,
        'data_processor': data_processor,
        'automation_manager': automation_manager
    }

async def example_data_loading(components):
    """Example of enhanced data loading with quality control"""
    enhanced_loader = components['enhanced_loader']
    quality_control = components['quality_control']
    data_processor = components['data_processor']
    
    # Load data with fallback and quality check
    symbol = 'VCB'
    timeframe = '1d'
    num_candles = 100
    
    try:
        # Load data with enhanced loader
        df, report = await enhanced_loader.load_data(
            symbol=symbol,
            timeframe=timeframe,
            num_candles=num_candles,
            detect_outliers=True,
            outlier_method='zscore',
            use_cache=True
        )
        
        if df is not None and not df.empty:
            logger.info(f"Loaded {len(df)} candles for {symbol} ({timeframe})")
            logger.info(f"Data report: {report}")
            
            # Check data quality
            quality_result = await quality_control.check_data_quality(
                df=df,
                symbol=symbol,
                timeframe=timeframe
            )
            
            logger.info(f"Quality score: {quality_result['overall_score']:.2f} ({quality_result['quality_rating']})")
            logger.info(f"Quality details: {quality_result['scores']}")
            
            if quality_result['issues']:
                logger.info("Quality issues:")
                for issue in quality_result['issues']:
                    logger.info(f"- {issue}")
            
            # Process data
            # 1. Detect and handle outliers
            df_cleaned = data_processor.detect_outliers(df, method='zscore', column='close')
            df_cleaned = data_processor.handle_outliers(df_cleaned, method='winsorize')
            
            # 2. Create derived features
            df_with_features = data_processor.create_derived_features(df_cleaned, feature_set='all')
            logger.info(f"Added {len(df_with_features.columns) - len(df.columns)} derived features")
            
            # 3. Memory optimization
            df_optimized = data_processor.optimize_memory(df_with_features)
            
            return df_optimized
    except Exception as e:
        logger.error(f"Error in data loading example: {str(e)}")
        return None

async def example_multi_timeframe_analysis(components):
    """Example of loading and combining multiple timeframes"""
    enhanced_loader = components['enhanced_loader']
    
    symbol = 'VCB'
    timeframes = ['1h', '4h', '1d']
    
    try:
        # Load data for multiple timeframes
        results = await enhanced_loader.load_multiple_timeframes(
            symbol=symbol,
            timeframes=timeframes,
            num_candles=100
        )
        
        # Process the results
        dfs = {}
        for tf, (df, report) in results.items():
            if df is not None and not df.empty:
                logger.info(f"Loaded {len(df)} candles for {symbol} ({tf})")
                dfs[tf] = df
        
        # Merge timeframes for combined analysis
        if dfs:
            from timestamp_utils import merge_timeframes
            merged_df = merge_timeframes(dfs)
            logger.info(f"Merged dataframe has {len(merged_df)} rows and {len(merged_df.columns)} columns")
            return merged_df
    except Exception as e:
        logger.error(f"Error in multi-timeframe example: {str(e)}")
    
    return None

async def integrate_with_bot(components):
    """
    Example of how to integrate with the main bot's analyze_command
    
    This would replace or enhance the existing data loading in the bot
    """
    
    async def enhanced_analyze_command(update, context):
        try:
            args = context.args
            if not args:
                raise ValueError("Nhập mã chứng khoán (e.g., VNINDEX, SSI).")
            
            symbol = args[0].upper()
            num_candles = int(args[1]) if len(args) > 1 else 100
            
            if num_candles < 20:
                raise ValueError("Số nến phải lớn hơn hoặc bằng 20 để tính toán chỉ báo!")
            if num_candles > 500:
                raise ValueError("Tối đa 500 nến!")
            
            # Use enhanced loader instead of original
            enhanced_loader = components['enhanced_loader']
            quality_control = components['quality_control']
            data_processor = components['data_processor']
            
            timeframes = ['1D', '1W', '1M']
            dfs = {}
            quality_reports = {}
            
            # Load data for each timeframe
            for tf in timeframes:
                df, report = await enhanced_loader.load_data(
                    symbol, tf, num_candles, 
                    detect_outliers=True
                )
                
                if df is None or df.empty:
                    continue
                
                # Process data with enhanced features
                df = data_processor.create_derived_features(df, feature_set='all')
                dfs[tf] = df
                
                # Check quality
                quality_result = await quality_control.check_data_quality(df, symbol, tf)
                quality_reports[tf] = quality_result
            
            # Continue with existing bot logic to generate report...
            # This is just a placeholder - integrate with your actual bot code
            
            # Example: Get fundamental data
            fundamental_data = await enhanced_loader.get_fundamental_data(symbol)
            
            # Return data for the bot's AI analysis
            return dfs, quality_reports, fundamental_data
            
        except ValueError as e:
            logger.error(f"Error in analyze command: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in analyze command: {str(e)}")
            raise
    
    # This function would be used instead of the original analyze_command in your bot
    return enhanced_analyze_command

async def main():
    """Main function to demonstrate integration"""
    try:
        # Initialize all components
        logger.info("Initializing components...")
        components = await initialize_components()
        
        # Example 1: Enhanced data loading
        logger.info("\n--- Example 1: Enhanced Data Loading ---")
        df = await example_data_loading(components)
        
        # Example 2: Multi-timeframe analysis
        logger.info("\n--- Example 2: Multi-Timeframe Analysis ---")
        multi_tf_df = await example_multi_timeframe_analysis(components)
        
        # Example 3: Bot integration
        logger.info("\n--- Example 3: Bot Integration ---")
        enhanced_analyze = await integrate_with_bot(components)
        logger.info("Enhanced analyze command ready for integration")
        
        # Example 4: Automation management
        logger.info("\n--- Example 4: Automation Status ---")
        automation_manager = components['automation_manager']
        status_report = await automation_manager.get_status_report()
        logger.info(f"Automation tracking {len(status_report['tracked_symbols'])} symbols")
        logger.info(f"Tracked timeframes: {status_report['tracked_timeframes']}")
        logger.info(f"Running tasks: {status_report['running_tasks']}")
        
        # Keep running to observe automated tasks
        logger.info("\nRunning for 60 seconds to observe automated tasks...")
        await asyncio.sleep(60)
        
        logger.info("Integration example completed")
    
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup any resources
        if 'components' in locals() and 'redis_manager' in components:
            await components['redis_manager'].close()
        
        if 'components' in locals() and 'enhanced_loader' in components:
            await components['enhanced_loader'].close()

if __name__ == "__main__":
    asyncio.run(main()) 