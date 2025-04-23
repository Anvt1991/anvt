#!/usr/bin/env python
"""
Simple entry point script to run the stock analysis bot.
"""
import asyncio
import sys
import logging

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == "test":
                logger.info("Running in test mode...")
                from main import main
                asyncio.run(main(test_mode=True))
            elif sys.argv[1] == "train":
                logger.info("Running model training...")
                from app.services.model_trainer import auto_train_models
                from app.database.model_db_manager import ModelDBManager
                from app.database.db_manager import init_db
                from app.utils.config import DATABASE_URL
                
                async def train():
                    engine = await init_db(DATABASE_URL)
                    model_db_manager = ModelDBManager(engine)
                    from app.services.data_loader import DataLoader
                    data_loader = DataLoader()
                    await auto_train_models(data_loader, model_db_manager)
                    
                asyncio.run(train())
        else:
            logger.info("Starting bot in normal mode...")
            from main import main
            asyncio.run(main())
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1) 