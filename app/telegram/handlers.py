import logging
import os
import asyncio
from datetime import datetime
from telegram import Update
from telegram.ext import ContextTypes

from app.database.db_manager import DBManager
from app.database.redis_manager import RedisManager
from app.services.data_pipeline import DataPipeline
from app.utils.config import ADMIN_USER_IDS, REDIS_CACHE_EXPIRY, REDIS_URL
from app.ai.openrouter_analyzer import OpenRouterAnalyzer

logger = logging.getLogger(__name__)

async def notify_admin_new_user(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Notify admins about new user interaction."""
    user = update.effective_user
    user_info = f"ID: {user.id}, Username: @{user.username or 'None'}, Name: {user.first_name} {user.last_name or ''}"
    
    admin_message = f"🔔 New user interaction:\n{user_info}"
    for admin_id in ADMIN_USER_IDS:
        try:
            await context.bot.send_message(chat_id=admin_id, text=admin_message)
        except Exception as e:
            logger.error(f"Error notifying admin {admin_id}: {str(e)}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /start command."""
    user = update.effective_user
    
    welcome_message = (
        f"Xin chào {user.first_name}! 👋\n\n"
        "Tôi là Bot phân tích chứng khoán 📊\n\n"
        "Bạn có thể sử dụng các lệnh sau:\n"
        "/analyze [mã CK] - Phân tích một mã chứng khoán\n"
        "/id - Lấy ID của bạn\n\n"
        "Ví dụ: /analyze FPT"
    )
    
    await update.message.reply_text(welcome_message)
    
    # Notify admins about new user
    await notify_admin_new_user(update, context)

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE, db_manager, redis_manager):
    """Handle the /analyze command to analyze a stock symbol."""
    start_time = datetime.now()
    user = update.effective_user
    
    # Check if user is approved
    is_approved = await db_manager.is_user_approved(user.id)
    if not is_approved and str(user.id) not in ADMIN_USER_IDS:
        await update.message.reply_text(
            "⚠️ Bạn chưa được phê duyệt để sử dụng chức năng này.\n"
            "Vui lòng liên hệ admin để được hỗ trợ."
        )
        logger.warning(f"Unapproved user {user.id} tried to use /analyze command")
        return
    
    # Extract symbol from command arguments
    args = context.args
    if not args or len(args) < 1:
        await update.message.reply_text(
            "⚠️ Vui lòng nhập mã chứng khoán.\n"
            "Ví dụ: /analyze FPT"
        )
        return
    
    symbol = args[0].upper()
    wait_message = await update.message.reply_text(f"⏳ Đang phân tích {symbol}...")
    
    try:
        # Check if analysis is cached in Redis
        cache_key = f"analysis_{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
        cached_report = await redis_manager.get(cache_key)
        
        if cached_report:
            logger.info(f"Using cached report for {symbol}")
            await wait_message.edit_text(cached_report)
            return
        
        # No cache, need to generate new report
        # 1. Get market data
        data_pipeline = await create_data_pipeline(db_manager, redis_manager)
        processed_data = await data_pipeline.prepare_symbol_data(symbol)
        
        if processed_data['status'] == 'error' or not processed_data.get('data'):
            error_message = (
                f"❌ Không thể tải dữ liệu cho {symbol}.\n"
                "Vui lòng kiểm tra lại mã chứng khoán và thử lại sau."
            )
            if processed_data.get('errors'):
                error_message += f"\n\nLỗi: {'; '.join(processed_data['errors'])}"
            
            await wait_message.edit_text(error_message)
            return
        
        # 2. Check for outliers
        outlier_reports = {}
        for timeframe, df in processed_data['data'].items():
            if df is not None and not df.empty:
                from app.utils.data_normalizer import DataNormalizer
                df_outliers, report = DataNormalizer.detect_outliers(df)
                outlier_reports[timeframe] = report
        
        # 3. Generate report using AI
        ai_analyzer = OpenRouterAnalyzer()
        report = await ai_analyzer.generate_report(
            processed_data['data'],
            symbol,
            processed_data['fundamental'],
            outlier_reports
        )
        
        # Cache the report in Redis
        await redis_manager.set(cache_key, report, REDIS_CACHE_EXPIRY)
        
        # Save report to database for history
        last_candle_data = processed_data.get('last_candle', {}).get('1D', {})
        close_today = last_candle_data.get('close', 0.0)
        
        previous_day_data = processed_data.get('data', {}).get('1D', None)
        close_yesterday = 0.0
        if previous_day_data is not None and len(previous_day_data) > 1:
            close_yesterday = previous_day_data.iloc[-2]['close'] if 'close' in previous_day_data.columns else 0.0
        
        await db_manager.save_report_history(
            symbol=symbol,
            report=report,
            close_today=close_today,
            close_yesterday=close_yesterday
        )
        
        # Send report to user
        await wait_message.edit_text(report)
        
        # Log execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"Analysis for {symbol} completed in {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        await wait_message.edit_text(
            f"❌ Lỗi khi phân tích {symbol}.\n"
            f"Chi tiết: {str(e)}\n\n"
            "Vui lòng thử lại sau."
        )

async def refresh_report_cache(symbol: str, num_candles: int, redis_manager):
    """Refresh the cached report for a symbol."""
    try:
        logger.info(f"Refreshing cached report for {symbol}")
        
        # Create a cache key
        cache_key = f"analysis_{symbol}_{datetime.now().strftime('%Y-%m-%d')}"
        
        # Delete old cache if it exists
        await redis_manager.redis_client.delete(cache_key)
        
        # Generate new report
        data_pipeline = await create_data_pipeline(None, redis_manager)
        processed_data = await data_pipeline.prepare_symbol_data(symbol, num_candles=num_candles)
        
        if processed_data['status'] == 'error' or not processed_data.get('data'):
            logger.error(f"Failed to refresh cache for {symbol}: {processed_data.get('errors', ['Unknown error'])}")
            return False
        
        # Get outlier reports
        outlier_reports = {}
        for timeframe, df in processed_data['data'].items():
            if df is not None and not df.empty:
                from app.utils.data_normalizer import DataNormalizer
                df_outliers, report = DataNormalizer.detect_outliers(df)
                outlier_reports[timeframe] = report
        
        # Generate report using AI
        ai_analyzer = OpenRouterAnalyzer()
        report = await ai_analyzer.generate_report(
            processed_data['data'],
            symbol,
            processed_data['fundamental'],
            outlier_reports
        )
        
        # Cache the report in Redis
        await redis_manager.set(cache_key, report, REDIS_CACHE_EXPIRY)
        
        # Store in database
        last_candle_data = processed_data.get('last_candle', {}).get('1D', {})
        close_today = last_candle_data.get('close', 0.0)
        
        previous_day_data = processed_data.get('data', {}).get('1D', None)
        close_yesterday = 0.0
        if previous_day_data is not None and len(previous_day_data) > 1:
            close_yesterday = previous_day_data.iloc[-2]['close'] if 'close' in previous_day_data.columns else 0.0
        
        return True
        
    except Exception as e:
        logger.error(f"Error refreshing cache for {symbol}: {str(e)}")
        return False

async def get_id(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /id command to get user ID."""
    user = update.effective_user
    await update.message.reply_text(
        f"👤 Your ID: {user.id}\n"
        f"Username: @{user.username or 'None'}"
    )

async def approve_user(update: Update, context: ContextTypes.DEFAULT_TYPE, db_manager):
    """Handle the /approve command to approve a user."""
    user = update.effective_user
    
    # Check if the command is from an admin
    if str(user.id) not in ADMIN_USER_IDS:
        await update.message.reply_text("⚠️ You are not authorized to use this command.")
        return
    
    # Extract user ID from command arguments
    args = context.args
    if not args or len(args) < 1:
        await update.message.reply_text(
            "⚠️ Please provide a user ID to approve.\n"
            "Example: /approve 123456789"
        )
        return
    
    try:
        user_id_to_approve = args[0]
        
        # Add user to approved users
        success = await db_manager.add_approved_user(user_id_to_approve)
        
        if success:
            await update.message.reply_text(f"✅ User {user_id_to_approve} has been approved!")
            
            # Notify the approved user
            try:
                await context.bot.send_message(
                    chat_id=user_id_to_approve,
                    text="🎉 Your account has been approved! You can now use all the features of the bot."
                )
            except Exception as e:
                logger.error(f"Failed to notify approved user {user_id_to_approve}: {str(e)}")
        else:
            await update.message.reply_text(f"ℹ️ User {user_id_to_approve} is already approved.")
            
    except Exception as e:
        logger.error(f"Error approving user: {str(e)}")
        await update.message.reply_text(f"❌ Error approving user: {str(e)}")

async def send_telegram_document(bot, chat_id, file_path, caption=None):
    """Send a file as document in Telegram."""
    try:
        with open(file_path, 'rb') as document:
            await bot.send_document(
                chat_id=chat_id,
                document=document,
                caption=caption
            )
        return True
    except Exception as e:
        logger.error(f"Error sending document: {str(e)}")
        return False

async def create_data_pipeline(db_manager=None, redis_manager=None):
    """Khởi tạo DataPipeline với các managers cần thiết"""
    if redis_manager is None:
        # Nếu không có redis_manager được truyền vào, tạo một instance mới
        redis_manager = RedisManager(REDIS_URL)
        await redis_manager.connect()
        
    return DataPipeline(data_source="vnstock", db_manager=db_manager, redis_manager=redis_manager) 