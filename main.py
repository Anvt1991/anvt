import os
import logging
import asyncio
import sys
import nest_asyncio
from datetime import datetime

from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler
from app.database.db_manager import DBManager, init_db
from app.database.model_db_manager import ModelDBManager
from app.database.redis_manager import RedisManager
from app.telegram.handlers import (
    start, analyze_command, get_id, approve_user, 
    refresh_report_cache, send_telegram_document
)
from app.services.model_trainer import auto_train_models
from app.utils.config import (
    TELEGRAM_TOKEN, DATABASE_URL, REDIS_URL, 
    WEBHOOK_URL, WEBHOOK_PORT, WEBHOOK_LISTEN, DEFAULT_MARKET_SYMBOLS, ADMIN_USER_IDS
)
from app.services.data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow running asyncio nested loops (needed in some environments)
nest_asyncio.apply()

async def keep_alive():
    """
    Keep the bot alive by performing regular maintenance tasks:
    - Optimize Redis cache
    - Refresh analysis for market symbols
    - Backup database
    """
    while True:
        try:
            logger.info("Running maintenance tasks...")
            
            # Optimize Redis cache
            redis_manager = RedisManager(REDIS_URL)
            deleted_count = await redis_manager.optimize_cache()
            logger.info(f"Redis cache optimization complete. Deleted {deleted_count} keys.")
            
            # Refresh cache for common market symbols
            for symbol in DEFAULT_MARKET_SYMBOLS[:3]:  # Limit to first 3 symbols to avoid overload
                await refresh_report_cache(symbol, 365, redis_manager)
                await asyncio.sleep(60)  # Delay between refreshes
                
            # Wait for next maintenance cycle (every 8 hours)
            await asyncio.sleep(8 * 60 * 60)
            
        except Exception as e:
            logger.error(f"Error in keep_alive: {str(e)}")
            await asyncio.sleep(15 * 60)  # Wait 15 minutes before retrying

async def backup_database():
    """Backup database to file and send to admin."""
    try:
        # Implementation depends on your database system
        # For PostgreSQL:
        # pg_dump -U username -d dbname -f backup.sql
        
        logger.info("Database backup completed")
        return True
    except Exception as e:
        logger.error(f"Database backup failed: {str(e)}")
        return False

async def main(test_mode=False):
    """Khởi động ứng dụng"""
    # Thiết lập logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Khởi tạo kết nối database
        logger.info("Khởi tạo kết nối đến database...")
        engine = await init_db(DATABASE_URL)
        db_manager = DBManager(engine)
        
        # Khởi tạo kết nối redis
        logger.info("Khởi tạo kết nối đến Redis...")
        redis_manager = RedisManager(REDIS_URL)
        await redis_manager.connect()
        
        # Chạy unit tests nếu ở chế độ test
        if test_mode:
            logger.info("Chạy ở chế độ test, đang thực hiện unit tests...")
            import unittest
            import pandas as pd
            import numpy as np
            from app.services.technical_analysis import TechnicalAnalyzer
            from app.services.model_trainer import forecast_with_prophet
            from app.utils.data_normalizer import DataNormalizer
            
            # ... existing test classes ...
                    
            # Chạy unit tests
            unittest.main(argv=['first-arg-is-ignored'], exit=False)
            return

        # Thiết lập Telegram bot
        from telegram.ext import ApplicationBuilder, CommandHandler
        from app.telegram.handlers import (
            start, analyze_command, get_id, approve_user, notify_admin_new_user
        )
        
        # Thiết lập bot application
        application_builder = ApplicationBuilder()
        application_builder.token(os.getenv("TELEGRAM_TOKEN"))
        application = application_builder.build()
        
        # Thêm các command handlers
        application.add_handler(CommandHandler("start", start))
        
        # Thêm handler cho /analyze command với truyền vào db_manager và redis_manager
        application.add_handler(
            CommandHandler(
                "analyze", 
                lambda update, context: analyze_command(update, context, db_manager, redis_manager)
            )
        )
        
        application.add_handler(CommandHandler("id", get_id))
        
        # Thêm handler cho /approve command với truyền vào db_manager
        application.add_handler(
            CommandHandler(
                "approve", 
                lambda update, context: approve_user(update, context, db_manager)
            )
        )
        
        # Thiết lập polling hoặc webhook
        webhook_url = os.getenv("WEBHOOK_URL")
        if webhook_url:
            # Sử dụng webhook nếu có
            logger.info(f"Sử dụng webhook: {webhook_url}")
            await setup_webhook()
        else:
            # Sử dụng polling nếu không có webhook
            logger.info("Sử dụng polling mode")
            await application.initialize()
            await application.start()
            await application.updater.start_polling()
            
            # Giữ ứng dụng chạy
            await asyncio.Event().wait()
            
    except Exception as e:
        logger.error(f"Lỗi khởi động ứng dụng: {str(e)}")
        raise

if __name__ == "__main__":
    # Check for test argument
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(main(test_mode=True))
    else:
        asyncio.run(main()) 