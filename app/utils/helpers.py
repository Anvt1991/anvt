import asyncio
import logging
import pandas as pd
import concurrent.futures
from functools import wraps
from datetime import datetime

from app.utils.config import VN_INDICES

logger = logging.getLogger(__name__)

async def run_in_thread(func, *args, **kwargs):
    """Run a blocking function in a thread pool executor."""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(
            pool, lambda: func(*args, **kwargs)
        )

def is_index(symbol: str) -> bool:
    """Check if a symbol is an index."""
    return symbol.upper() in VN_INDICES

def standardize_data_for_db(data: dict) -> dict:
    """Standardize data types for database storage."""
    from app.utils.data_normalizer import DataNormalizer
    return DataNormalizer.standardize_for_db(data)

def filter_trading_days(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include trading days."""
    if df is None or df.empty:
        return df
        
    # Đảm bảo index là datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    
    # Lọc ngày thứ 7, chủ nhật
    filtered_df = df[~df.index.weekday.isin([5, 6])]  # 5=Sat, 6=Sun
    
    # Check if we're removing too many days (possible indicator of exchange-specific holidays)
    if len(filtered_df) < len(df) * 0.6:
        logger.warning("Lọc ngày giao dịch loại bỏ quá nhiều dữ liệu, có thể có vấn đề")
        
    return filtered_df 