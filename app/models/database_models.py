from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, LargeBinary, Boolean, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ApprovedUser(Base):
    __tablename__ = 'approved_users'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, nullable=False)
    approved_at = Column(DateTime, default=datetime.now)

class ReportHistory(Base):
    __tablename__ = 'report_history'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    date = Column(String, nullable=False)
    report = Column(Text, nullable=False)
    close_today = Column(Float, nullable=False)
    close_yesterday = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    
    # Thêm index cho truy vấn tối ưu
    __table_args__ = (Index('idx_report_symbol_date', 'symbol', 'date'),)

class TrainedModel(Base):
    __tablename__ = 'trained_models'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    performance = Column(Float, nullable=True)
    
    # Thêm index cho truy vấn tối ưu
    __table_args__ = (Index('idx_model_symbol_type', 'symbol', 'model_type'),)

class StockData(Base):
    """Bảng lưu trữ dữ liệu lịch sử chứng khoán"""
    __tablename__ = 'stock_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    timeframe = Column(String(10), nullable=False)  # 1D, 1W, 1M
    
    # Dữ liệu giá và khối lượng
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    
    # Trường tùy chọn
    adj_close = Column(Float, nullable=True)
    
    # Siêu dữ liệu
    source = Column(String(50), nullable=False)  # yahoo, vnstock, etc.
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Indexes cho truy vấn hiệu quả
    __table_args__ = (
        Index('idx_stock_symbol_date_tf', 'symbol', 'date', 'timeframe', unique=True),
        Index('idx_stock_date', 'date'),
    )

class StockFundamentalData(Base):
    """Bảng lưu trữ dữ liệu cơ bản của cổ phiếu"""
    __tablename__ = 'stock_fundamental_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    
    # Dữ liệu cơ bản
    market_cap = Column(Float, nullable=True)
    pe_ratio = Column(Float, nullable=True)
    pb_ratio = Column(Float, nullable=True)
    dividend_yield = Column(Float, nullable=True)
    eps = Column(Float, nullable=True)
    roe = Column(Float, nullable=True)
    roa = Column(Float, nullable=True)
    
    # Thông tin ngành
    sector = Column(String(100), nullable=True)
    industry = Column(String(100), nullable=True)
    
    # Tóm tắt và mô tả
    summary = Column(Text, nullable=True)
    
    # Siêu dữ liệu
    source = Column(String(50), nullable=False)  # yahoo, vnstock, etc.
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Indexes
    __table_args__ = (
        Index('idx_fundamental_symbol_date', 'symbol', 'date', unique=True),
    ) 