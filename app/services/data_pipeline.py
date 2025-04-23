import logging
import asyncio
from datetime import datetime, timedelta

from app.services.data_loader import DataLoader
from app.services.technical_analysis import TechnicalAnalyzer
from app.utils.config import DEFAULT_TIMEFRAMES, DEFAULT_CANDLES, DEFAULT_MARKET_SYMBOLS
from app.utils.data_normalizer import DataNormalizer

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Pipeline to coordinate data loading, processing, and analysis
    for stock data across multiple timeframes and symbols.
    """
    
    def __init__(self, data_source: str = 'vnstock', db_manager=None, redis_manager=None):
        """Initialize with the specified data source and optional managers."""
        self.data_loader = DataLoader(source=data_source)
        self.technical_analyzer = TechnicalAnalyzer()
        self.db_manager = db_manager  # Quản lý database
        self.redis_manager = redis_manager  # Quản lý cache Redis
    
    async def prepare_symbol_data(self, symbol: str, timeframes: list = None, num_candles: int = DEFAULT_CANDLES) -> dict:
        """
        Prepare data for a single symbol across multiple timeframes:
        1. Thử tải dữ liệu từ cache Redis
        2. Nếu không có cache, thử tải từ database
        3. Nếu không có trong database, tải từ nguồn bên ngoài
        4. Tính toán chỉ báo kỹ thuật
        5. Phát hiện mẫu hình
        6. Tải dữ liệu cơ bản
        7. Lưu cache và database
        """
        if timeframes is None:
            timeframes = DEFAULT_TIMEFRAMES
        
        # Dictionary to store processed data for each timeframe
        result = {
            'symbol': symbol,
            'data': {},
            'status': 'success',
            'errors': [],
            'patterns': {},
            'last_candle': {},
            'fundamental': {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_source': {}  # Để theo dõi nguồn dữ liệu (cache, db, external)
        }
        
        # Load data for each timeframe
        raw_dataframes = {}
        tasks = []
        
        for tf in timeframes:
            tasks.append(self._load_timeframe_data_with_cache(symbol, tf, num_candles))
        
        # Execute all loading tasks concurrently
        timeframe_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, tf in enumerate(timeframes):
            if isinstance(timeframe_results[i], Exception):
                error_message = f"Error loading {tf} data: {str(timeframe_results[i])}"
                logger.error(error_message)
                result['errors'].append(error_message)
                continue
                
            df, message, source = timeframe_results[i]
            result['data_source'][tf] = source
            
            if df is not None and not df.empty:
                raw_dataframes[tf] = df
            else:
                result['errors'].append(f"No data available for {symbol} in {tf} timeframe: {message}")
        
        # If no data was loaded, return with error
        if not raw_dataframes:
            result['status'] = 'error'
            result['errors'].append(f"No data available for {symbol} in any timeframe")
            return result
        
        # Calculate technical indicators for each timeframe
        dataframes_with_indicators = self.technical_analyzer.calculate_multi_timeframe_indicators(raw_dataframes)
        
        # Store processed dataframes
        result['data'] = dataframes_with_indicators
        
        # Detect candlestick patterns
        result['patterns'] = self.technical_analyzer.detect_patterns(dataframes_with_indicators)
        
        # Extract last candle information for each timeframe
        for tf, df in dataframes_with_indicators.items():
            result['last_candle'][tf] = self.technical_analyzer.extract_last_candle_info(df)
        
        # Load fundamental data
        try:
            # Thử tải dữ liệu cơ bản từ database trước
            fundamental_data = await self._load_fundamental_data_with_cache(symbol)
            result['fundamental'] = fundamental_data
        except Exception as e:
            logger.error(f"Error loading fundamental data for {symbol}: {str(e)}")
            result['errors'].append(f"Error loading fundamental data: {str(e)}")
        
        return result
    
    async def _load_timeframe_data_with_cache(self, symbol, timeframe, num_candles):
        """
        Tải dữ liệu cho một timeframe cụ thể, ưu tiên theo thứ tự:
        1. Redis cache
        2. Database
        3. Nguồn bên ngoài (Yahoo, VNStock)
        
        Trả về: (DataFrame, message, source)
        """
        # 1. Thử từ Redis cache trước
        if self.redis_manager is not None:
            cache_key = f"data_{symbol}_{timeframe}"
            cached_data = await self.redis_manager.get(cache_key)
            if cached_data is not None:
                logger.info(f"Tải dữ liệu {symbol} ({timeframe}) từ Redis cache")
                return cached_data, "Data loaded from Redis cache", "redis"
        
        # 2. Nếu không có trong Redis, thử từ database
        if self.db_manager is not None:
            try:
                end_date = datetime.now()
                # Tính toán start_date dựa trên timeframe và số nến yêu cầu
                if timeframe == "1D":
                    start_date = end_date - timedelta(days=num_candles * 1.5)  # 50% buffer for weekends/holidays
                elif timeframe == "1W":
                    start_date = end_date - timedelta(days=num_candles * 7 * 1.5)
                elif timeframe == "1M":
                    start_date = end_date - timedelta(days=num_candles * 30 * 1.5)
                else:
                    start_date = end_date - timedelta(days=num_candles * 1.5)
                
                df = await self.db_manager.load_stock_data(symbol, timeframe, start_date, end_date)
                if df is not None and not df.empty:
                    # Chuẩn hóa dữ liệu
                    df = DataNormalizer.normalize_dataframe(df)
                    
                    # Lưu vào Redis cache
                    if self.redis_manager is not None:
                        await self.redis_manager.set(cache_key, df, 3600)  # 1 giờ
                        
                    logger.info(f"Tải dữ liệu {symbol} ({timeframe}) từ database")
                    return df, "Data loaded from database", "database"
            except Exception as e:
                logger.warning(f"Lỗi khi tải dữ liệu từ database: {str(e)}")
        
        # 3. Nếu không có trong database, tải từ nguồn bên ngoài
        try:
            df, message = await self.data_loader.load_data(symbol, timeframe, num_candles)
            
            if df is not None and not df.empty:
                # Lưu vào database nếu có
                if self.db_manager is not None:
                    try:
                        await self.db_manager.save_stock_data(df, symbol, timeframe, self.data_loader.source)
                    except Exception as e:
                        logger.error(f"Lỗi khi lưu dữ liệu vào database: {str(e)}")
                
                # Lưu vào Redis cache nếu có
                if self.redis_manager is not None:
                    cache_key = f"data_{symbol}_{timeframe}"
                    await self.redis_manager.set(cache_key, df, 3600)  # 1 giờ
                    
                logger.info(f"Tải dữ liệu {symbol} ({timeframe}) từ nguồn bên ngoài ({self.data_loader.source})")
                return df, message, self.data_loader.source
                
            return None, message, "external_failed"
            
        except Exception as e:
            logger.error(f"Error loading {timeframe} data for {symbol} from external source: {str(e)}")
            raise
    
    async def _load_fundamental_data_with_cache(self, symbol):
        """
        Tải dữ liệu cơ bản cho một mã chứng khoán, ưu tiên theo thứ tự:
        1. Redis cache
        2. Database
        3. Nguồn bên ngoài
        """
        # 1. Thử từ Redis cache trước
        if self.redis_manager is not None:
            cache_key = f"fundamental_{symbol}"
            cached_data = await self.redis_manager.get(cache_key)
            if cached_data is not None:
                logger.info(f"Tải dữ liệu cơ bản {symbol} từ Redis cache")
                return cached_data
        
        # 2. Nếu không có trong Redis, thử từ database nếu có
        if self.db_manager is not None:
            try:
                # TODO: Implement fundamental data loading from database when database manager is ready
                # fundamental_data = await self.db_manager.load_fundamental_data(symbol)
                # if fundamental_data is not None:
                #     # Lưu vào Redis cache
                #     if self.redis_manager is not None:
                #         await self.redis_manager.set(cache_key, fundamental_data, 86400)  # 24 giờ
                #     return fundamental_data
                pass
            except Exception as e:
                logger.warning(f"Lỗi khi tải dữ liệu cơ bản từ database: {str(e)}")
        
        # 3. Tải từ nguồn bên ngoài
        try:
            fundamental_data = await self.data_loader.get_fundamental_data(symbol)
            
            # Lưu vào database nếu có
            if self.db_manager is not None:
                try:
                    await self.db_manager.save_fundamental_data(fundamental_data)
                except Exception as e:
                    logger.error(f"Lỗi khi lưu dữ liệu cơ bản vào database: {str(e)}")
            
            # Lưu vào Redis cache nếu có
            if self.redis_manager is not None and fundamental_data:
                cache_key = f"fundamental_{symbol}"
                await self.redis_manager.set(cache_key, fundamental_data, 86400)  # 24 giờ
                
            return fundamental_data
            
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu cơ bản cho {symbol}: {str(e)}")
            return {}
    
    async def prepare_market_data(self, market_symbols: list = None) -> dict:
        """
        Prepare data for multiple market symbols.
        
        Args:
            market_symbols: List of symbols to process. If None, uses DEFAULT_MARKET_SYMBOLS.
            
        Returns:
            Dictionary with processed data for each symbol.
        """
        if market_symbols is None:
            market_symbols = DEFAULT_MARKET_SYMBOLS
        
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols': {},
            'status': 'success',
            'errors': []
        }
        
        tasks = []
        for symbol in market_symbols:
            tasks.append(self.prepare_symbol_data(symbol))
        
        # Execute all symbol processing tasks concurrently
        symbol_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, symbol in enumerate(market_symbols):
            if isinstance(symbol_results[i], Exception):
                error_message = f"Error processing {symbol}: {str(symbol_results[i])}"
                logger.error(error_message)
                result['errors'].append(error_message)
                continue
                
            result['symbols'][symbol] = symbol_results[i]
        
        # If all symbols failed, mark as error
        if len(result['errors']) == len(market_symbols):
            result['status'] = 'error'
        
        return result 