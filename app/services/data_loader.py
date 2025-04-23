import logging
import asyncio
import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential, reraise
from datetime import datetime, timedelta

from app.utils.helpers import run_in_thread, is_index
from app.utils.data_normalizer import DataNormalizer

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, source: str = 'vnstock'):
        self.source = source.lower()
    
    async def load_data(self, symbol: str, timeframe: str, num_candles: int) -> (pd.DataFrame, str):
        """Load historical price data for the given symbol and timeframe."""
        try:
            # Thêm .VN cho mã chứng khoán Việt Nam với Yahoo Finance
            if self.source == 'yahoo':
                if len(symbol) <= 3 and not is_index(symbol):
                    yahoo_symbol = f"{symbol}.VN"
                else:
                    yahoo_symbol = f"{symbol}.VN" if not is_index(symbol) else f"^{symbol}"
                    
                # Map timeframe to yahoo period
                period_map = {
                    "1D": "1d", 
                    "1W": "1wk",
                    "1M": "1mo"
                }
                
                yahoo_period = period_map.get(timeframe, "1d")
                
                # Tải dữ liệu từ Yahoo Finance
                df = await self._download_yahoo_data(yahoo_symbol, num_candles, yahoo_period)
                
                if df is None or df.empty:
                    logger.warning(f"Không tìm thấy dữ liệu cho {symbol} (Yahoo Finance)")
                    return None, f"Không tìm thấy dữ liệu Yahoo Finance cho {symbol}"
                
                # Chuẩn hóa dữ liệu
                df = DataNormalizer.normalize_dataframe(df)
                return df, f"Dữ liệu {symbol} tải từ Yahoo Finance"
                
            elif self.source == 'vnstock':
                try:
                    # Sử dụng vnstock để lấy dữ liệu
                    # Cần phải chạy trong một thread riêng vì vnstock không hỗ trợ async
                    df = await run_in_thread(self._fetch_vnstock_data, symbol, timeframe, num_candles)
                    
                    if df is None or df.empty:
                        logger.warning(f"Không tìm thấy dữ liệu cho {symbol} (VNStock)")
                        return None, f"Không tìm thấy dữ liệu VNStock cho {symbol}"
                    
                    # Chuẩn hóa dữ liệu
                    df = DataNormalizer.normalize_dataframe(df)
                    return df, f"Dữ liệu {symbol} tải từ VNStock"
                    
                except Exception as e:
                    logger.error(f"Lỗi khi tải dữ liệu từ VNStock: {str(e)}")
                    # Fallback to Yahoo Finance if VNStock fails
                    logger.info(f"Thử tải {symbol} từ Yahoo Finance...")
                    return await self.load_data(symbol, timeframe, num_candles)
            else:
                logger.error(f"Nguồn dữ liệu không được hỗ trợ: {self.source}")
                return None, f"Nguồn dữ liệu không được hỗ trợ: {self.source}"
                
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu cho {symbol}: {str(e)}")
            return None, f"Lỗi: {str(e)}"
    
    def _fetch_vnstock_data(self, symbol, timeframe, num_candles):
        """Fetch data from VNStock library (needs to be run in thread)"""
        try:
            # Import vnstock và sử dụng các hàm thích hợp dựa trên timeframe
            import vnstock
            
            # Tính toán start_date và end_date dựa trên num_candles và timeframe
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            if timeframe == "1D":
                # Tính toán ngày bắt đầu, thêm 50% để tính đến ngày nghỉ và ngày lễ
                start_date = (datetime.now() - timedelta(days=int(num_candles * 1.5))).strftime('%Y-%m-%d')
                df = vnstock.stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date)
            elif timeframe == "1W":
                # Với dữ liệu tuần, cần lấy đủ số tuần, thêm vào 50% để đảm bảo
                start_date = (datetime.now() - timedelta(days=int(num_candles * 7 * 1.5))).strftime('%Y-%m-%d')
                # Lấy dữ liệu ngày và resample thành tuần
                df_daily = vnstock.stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if df_daily is not None and not df_daily.empty:
                    # Chuyển đổi cột time thành datetime nếu chưa phải
                    if 'time' in df_daily.columns:
                        df_daily['time'] = pd.to_datetime(df_daily['time'])
                        df_daily.set_index('time', inplace=True)
                    # Resample thành dữ liệu tuần
                    df = df_daily.resample('W').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                else:
                    return None
            elif timeframe == "1M":
                # Với dữ liệu tháng, cần lấy đủ số tháng, thêm vào 50% để đảm bảo
                start_date = (datetime.now() - timedelta(days=int(num_candles * 30 * 1.5))).strftime('%Y-%m-%d')
                # Lấy dữ liệu ngày và resample thành tháng
                df_daily = vnstock.stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date)
                if df_daily is not None and not df_daily.empty:
                    # Chuyển đổi cột time thành datetime nếu chưa phải
                    if 'time' in df_daily.columns:
                        df_daily['time'] = pd.to_datetime(df_daily['time'])
                        df_daily.set_index('time', inplace=True)
                    # Resample thành dữ liệu tháng
                    df = df_daily.resample('M').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                else:
                    return None
            else:
                logger.warning(f"Timeframe không được hỗ trợ: {timeframe}, sử dụng 1D")
                start_date = (datetime.now() - timedelta(days=int(num_candles * 1.5))).strftime('%Y-%m-%d')
                df = vnstock.stock_historical_data(symbol=symbol, start_date=start_date, end_date=end_date)
            
            # Kiểm tra kết quả
            if df is None or df.empty:
                logger.warning(f"Không có dữ liệu cho {symbol} với timeframe {timeframe}")
                return None
                
            # Đảm bảo tên cột tuân theo quy ước
            if 'time' in df.columns:
                df.rename(columns={'time': 'date'}, inplace=True)
                
            return df
            
        except ImportError:
            logger.error("Không thể import thư viện vnstock. Vui lòng cài đặt: pip install vnstock")
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu từ VNStock: {str(e)}")
            return None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8), reraise=True)
    async def _download_yahoo_data(self, symbol: str, num_candles: int, period: str) -> pd.DataFrame:
        """Download data from Yahoo Finance."""
        try:
            # Calculate start date based on num_candles and period
            end_date = datetime.now()
            
            if period == "1d":
                start_date = end_date - timedelta(days=num_candles * 1.5)  # Account for weekends/holidays
            elif period == "1wk":
                start_date = end_date - timedelta(weeks=num_candles * 1.5)
            elif period == "1mo":
                start_date = end_date - timedelta(days=num_candles * 31 * 1.2)
            else:
                start_date = end_date - timedelta(days=num_candles * 1.5)
            
            # Format dates for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Use run_in_thread to execute the blocking yfinance download in a separate thread
            df = await run_in_thread(
                yf.download,
                symbol,
                start=start_str,
                end=end_str,
                interval=period,
                progress=False
            )
            
            if df.empty:
                logger.warning(f"Yfinance returned empty dataframe for {symbol}")
                return None
                
            return df
            
        except Exception as e:
            logger.error(f"Error downloading from Yahoo Finance: {str(e)}")
            raise
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def fetch_fundamental_data_vnstock(self, symbol: str) -> dict:
        """Fetch fundamental data for a given symbol using VNStock."""
        try:
            # Fetch data using vnstock (placeholder - implement actual calls)
            def fetch():
                # Placeholder for actual vnstock implementation
                # This should return fundamental data like P/E, market cap, etc.
                return {
                    "symbol": symbol,
                    "source": "vnstock",
                    "market_cap": None,
                    "pe_ratio": None,
                    "pb_ratio": None,
                    "dividend_yield": None,
                    "sector": None,
                    "industry": None,
                    # Add other fundamental metrics
                }
                
            result = await run_in_thread(fetch)
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu cơ bản từ VNStock: {str(e)}")
            return {"symbol": symbol, "error": str(e), "source": "vnstock"}
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def fetch_fundamental_data_yahoo(self, symbol: str) -> dict:
        """Fetch fundamental data for a given symbol using Yahoo Finance."""
        try:
            # Thêm .VN cho mã chứng khoán Việt Nam với Yahoo Finance
            if len(symbol) <= 3 and not is_index(symbol):
                yahoo_symbol = f"{symbol}.VN"
            else:
                yahoo_symbol = f"{symbol}.VN" if not is_index(symbol) else f"^{symbol}"
            
            def fetch():
                try:
                    ticker = yf.Ticker(yahoo_symbol)
                    info = ticker.info
                    
                    return {
                        "symbol": symbol,
                        "source": "yahoo",
                        "market_cap": info.get("marketCap"),
                        "pe_ratio": info.get("trailingPE"),
                        "pb_ratio": info.get("priceToBook"),
                        "dividend_yield": info.get("dividendYield"),
                        "sector": info.get("sector"),
                        "industry": info.get("industry"),
                        # Thêm các thông tin khác
                        "summary": info.get("longBusinessSummary"),
                        "website": info.get("website"),
                        "full_info": info
                    }
                except Exception as e:
                    logger.error(f"Lỗi trong quá trình lấy dữ liệu Yahoo Finance cho {symbol}: {str(e)}")
                    return {"symbol": symbol, "error": str(e), "source": "yahoo"}
            
            result = await run_in_thread(fetch)
            return result
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu cơ bản từ Yahoo Finance: {str(e)}")
            return {"symbol": symbol, "error": str(e), "source": "yahoo"}
            
    async def get_fundamental_data(self, symbol: str) -> dict:
        """Get fundamental data for a symbol using the best available source."""
        # Try VNStock first for Vietnamese stocks
        data = await self.fetch_fundamental_data_vnstock(symbol)
        
        # If VNStock doesn't have the data or returns an error, try Yahoo
        if "error" in data or all(v is None for k, v in data.items() if k not in ["symbol", "source"]):
            logger.info(f"VNStock không có đủ dữ liệu cơ bản cho {symbol}, thử với Yahoo Finance")
            data = await self.fetch_fundamental_data_yahoo(symbol)
        
        return data 