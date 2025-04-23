import logging
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from tenacity import retry, stop_after_attempt, wait_exponential

from app.models.database_models import ApprovedUser, ReportHistory, Base, StockData, StockFundamentalData
from app.utils.data_normalizer import DataNormalizer
import pandas as pd

logger = logging.getLogger(__name__)

async def init_db(database_url):
    engine = create_async_engine(database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine

class DBManager:
    def __init__(self, engine):
        self.engine = engine
        self.async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def is_user_approved(self, user_id) -> bool:
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    select(ApprovedUser).where(ApprovedUser.user_id == str(user_id))
                )
                user = result.scalars().first()
                return user is not None
            except Exception as e:
                logger.error(f"Lỗi khi kiểm tra user được phê duyệt: {str(e)}")
                return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def add_approved_user(self, user_id, approved_at=None) -> None:
        async with self.async_session() as session:
            try:
                # Kiểm tra xem user đã tồn tại chưa
                result = await session.execute(
                    select(ApprovedUser).where(ApprovedUser.user_id == str(user_id))
                )
                existing_user = result.scalars().first()
                
                if existing_user is None:
                    new_user = ApprovedUser(
                        user_id=str(user_id),
                        approved_at=approved_at or datetime.now()
                    )
                    session.add(new_user)
                    await session.commit()
                    logger.info(f"Đã thêm user mới vào danh sách phê duyệt: {user_id}")
                    return True
                return False
            except Exception as e:
                await session.rollback()
                logger.error(f"Lỗi khi thêm user vào danh sách phê duyệt: {str(e)}")
                return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_report_history(self, symbol: str) -> list:
        async with self.async_session() as session:
            try:
                # Lấy các báo cáo của 1 mã chứng khoán, sắp xếp theo thời gian giảm dần
                result = await session.execute(
                    select(ReportHistory)
                    .where(ReportHistory.symbol == symbol)
                    .order_by(ReportHistory.timestamp.desc())
                    .limit(10)
                )
                
                reports = result.scalars().all()
                
                report_list = []
                for report in reports:
                    report_list.append({
                        'symbol': report.symbol,
                        'date': report.date,
                        'report': report.report,
                        'close_today': report.close_today,
                        'close_yesterday': report.close_yesterday,
                        'timestamp': report.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                return report_list
            except Exception as e:
                logger.error(f"Lỗi khi lấy lịch sử báo cáo: {str(e)}")
                return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def save_report_history(self, symbol: str, report: str, close_today: float, close_yesterday: float) -> None:
        async with self.async_session() as session:
            try:
                # Kiểm tra xem đã có báo cáo cùng ngày chưa
                today_date = datetime.now().strftime('%Y-%m-%d')
                
                result = await session.execute(
                    select(ReportHistory)
                    .where(ReportHistory.symbol == symbol)
                    .where(ReportHistory.date == today_date)
                )
                
                existing_report = result.scalars().first()
                
                if existing_report:
                    # Cập nhật báo cáo hiện có
                    existing_report.report = report
                    existing_report.close_today = close_today
                    existing_report.close_yesterday = close_yesterday
                    existing_report.timestamp = datetime.now()
                else:
                    # Tạo báo cáo mới
                    new_report = ReportHistory(
                        symbol=symbol,
                        date=today_date,
                        report=report,
                        close_today=close_today,
                        close_yesterday=close_yesterday
                    )
                    session.add(new_report)
                
                await session.commit()
                logger.info(f"Đã lưu báo cáo cho {symbol} vào database")
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Lỗi khi lưu báo cáo: {str(e)}")
                return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def save_stock_data(self, df: pd.DataFrame, symbol: str, timeframe: str, source: str):
        """
        Lưu dữ liệu lịch sử cổ phiếu từ DataFrame vào database.
        
        Args:
            df: DataFrame chứa dữ liệu cổ phiếu đã được chuẩn hóa
            symbol: Mã cổ phiếu
            timeframe: Khung thời gian (1D, 1W, 1M)
            source: Nguồn dữ liệu (yahoo, vnstock, etc.)
        """
        if df is None or df.empty:
            logger.warning(f"Không có dữ liệu để lưu cho {symbol}")
            return False
            
        try:
            # Đảm bảo index là datetime
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                df.index = pd.to_datetime(df.index)
                
            # Chuẩn bị các bản ghi để insert
            records = []
            for idx, row in df.iterrows():
                # Tạo đối tượng StockData cho mỗi hàng
                stock_data = StockData(
                    symbol=symbol,
                    date=idx.to_pydatetime(),
                    timeframe=timeframe,
                    open=row.get('open'),
                    high=row.get('high'),
                    low=row.get('low'),
                    close=row.get('close'),
                    volume=row.get('volume'),
                    adj_close=row.get('adj_close'),
                    source=source
                )
                records.append(stock_data)
                
            # Lưu vào database
            async with self.async_session() as session:
                try:
                    # Kiểm tra xem dữ liệu đã tồn tại chưa để upsert
                    for record in records:
                        # Tìm bản ghi hiện có với cùng symbol, date, timeframe
                        result = await session.execute(
                            select(StockData).where(
                                StockData.symbol == record.symbol,
                                StockData.date == record.date,
                                StockData.timeframe == record.timeframe
                            )
                        )
                        existing_record = result.scalars().first()
                        
                        if existing_record:
                            # Cập nhật bản ghi hiện có
                            existing_record.open = record.open
                            existing_record.high = record.high
                            existing_record.low = record.low
                            existing_record.close = record.close
                            existing_record.volume = record.volume
                            existing_record.adj_close = record.adj_close
                            existing_record.source = record.source
                            existing_record.updated_at = datetime.now()
                        else:
                            # Thêm bản ghi mới
                            session.add(record)
                            
                    await session.commit()
                    logger.info(f"Đã lưu {len(records)} bản ghi cho {symbol} ({timeframe}) vào database")
                    return True
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Lỗi khi lưu dữ liệu cổ phiếu: {str(e)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn bị dữ liệu cổ phiếu để lưu: {str(e)}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_stock_data(self, symbol: str, timeframe: str, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Tải dữ liệu lịch sử cổ phiếu từ database và trả về dưới dạng DataFrame.
        
        Args:
            symbol: Mã cổ phiếu
            timeframe: Khung thời gian (1D, 1W, 1M)
            start_date: Ngày bắt đầu (mặc định là 1 năm trước)
            end_date: Ngày kết thúc (mặc định là ngày hiện tại)
        """
        try:
            # Thiết lập mặc định cho ngày bắt đầu và kết thúc
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                if timeframe == '1D':
                    start_date = end_date - timedelta(days=365)
                elif timeframe == '1W':
                    start_date = end_date - timedelta(days=365*2)
                elif timeframe == '1M':
                    start_date = end_date - timedelta(days=365*5)
                else:
                    start_date = end_date - timedelta(days=365)
                    
            async with self.async_session() as session:
                # Truy vấn dữ liệu
                result = await session.execute(
                    select(StockData).where(
                        StockData.symbol == symbol,
                        StockData.timeframe == timeframe,
                        StockData.date >= start_date,
                        StockData.date <= end_date
                    ).order_by(StockData.date)
                )
                
                records = result.scalars().all()
                
                if not records:
                    logger.warning(f"Không tìm thấy dữ liệu cho {symbol} với timeframe {timeframe}")
                    return None
                    
                # Chuyển đổi thành DataFrame
                data = []
                for record in records:
                    data.append({
                        'date': record.date,
                        'open': record.open,
                        'high': record.high,
                        'low': record.low,
                        'close': record.close,
                        'volume': record.volume,
                        'adj_close': record.adj_close
                    })
                
                df = pd.DataFrame(data)
                if not df.empty:
                    df.set_index('date', inplace=True)
                    df.sort_index(inplace=True)
                    
                return df
                
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu cổ phiếu: {str(e)}")
            return None
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def save_fundamental_data(self, fundamental_data: dict):
        """
        Lưu dữ liệu cơ bản của cổ phiếu vào database.
        
        Args:
            fundamental_data: Dictionary chứa dữ liệu cơ bản đã chuẩn hóa
        """
        if not fundamental_data or 'symbol' not in fundamental_data:
            logger.warning("Không có dữ liệu cơ bản hợp lệ để lưu")
            return False
            
        try:
            symbol = fundamental_data.get('symbol')
            source = fundamental_data.get('source', 'unknown')
            
            # Chuẩn hóa dữ liệu để lưu trữ an toàn trong DB
            fundamental_data = DataNormalizer.standardize_for_db(fundamental_data)
            
            # Tạo bản ghi mới
            fund_data = StockFundamentalData(
                symbol=symbol,
                date=datetime.now(),
                market_cap=fundamental_data.get('market_cap'),
                pe_ratio=fundamental_data.get('pe_ratio'),
                pb_ratio=fundamental_data.get('pb_ratio'),
                dividend_yield=fundamental_data.get('dividend_yield'),
                eps=fundamental_data.get('eps'),
                roe=fundamental_data.get('roe'),
                roa=fundamental_data.get('roa'),
                sector=fundamental_data.get('sector'),
                industry=fundamental_data.get('industry'),
                summary=fundamental_data.get('summary'),
                source=source
            )
            
            async with self.async_session() as session:
                try:
                    # Kiểm tra xem đã có dữ liệu trong ngày chưa
                    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                    tomorrow = today + timedelta(days=1)
                    
                    result = await session.execute(
                        select(StockFundamentalData).where(
                            StockFundamentalData.symbol == symbol,
                            StockFundamentalData.date >= today,
                            StockFundamentalData.date < tomorrow
                        )
                    )
                    
                    existing_record = result.scalars().first()
                    
                    if existing_record:
                        # Cập nhật bản ghi hiện có
                        existing_record.market_cap = fund_data.market_cap
                        existing_record.pe_ratio = fund_data.pe_ratio
                        existing_record.pb_ratio = fund_data.pb_ratio
                        existing_record.dividend_yield = fund_data.dividend_yield
                        existing_record.eps = fund_data.eps
                        existing_record.roe = fund_data.roe
                        existing_record.roa = fund_data.roa
                        existing_record.sector = fund_data.sector
                        existing_record.industry = fund_data.industry
                        existing_record.summary = fund_data.summary
                        existing_record.source = fund_data.source
                        existing_record.updated_at = datetime.now()
                    else:
                        # Thêm bản ghi mới
                        session.add(fund_data)
                        
                    await session.commit()
                    logger.info(f"Đã lưu dữ liệu cơ bản cho {symbol} vào database")
                    return True
                except Exception as e:
                    await session.rollback()
                    logger.error(f"Lỗi khi lưu dữ liệu cơ bản: {str(e)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Lỗi khi chuẩn bị dữ liệu cơ bản để lưu: {str(e)}")
            return False 