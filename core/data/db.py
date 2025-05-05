#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module quản lý cơ sở dữ liệu cho hệ thống chatbot.
Triển khai dựa trên SQLite đồng bộ (sync).
"""

import json
import logging
import pickle
import sqlite3
from datetime import datetime
import pytz
import os
from typing import List, Dict, Tuple, Any, Optional, Union
import requests
import html
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, LargeBinary, select, func, text
import redis.asyncio as redis

# Logging setup
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Timezone configuration
TZ = pytz.timezone('Asia/Bangkok')

# Admin ID from environment for access control
ADMIN_ID = os.environ.get("ADMIN_ID", "1225226589")

# Database URLs
DATABASE_URL = os.getenv("DATABASE_URL")
REDIS_URL = os.getenv("REDIS_URL")

# SQLAlchemy setup
Base = declarative_base()
engine = create_async_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

# Redis setup
class RedisManager:
    def __init__(self):
        try:
            self.redis_client = redis.from_url(REDIS_URL)
            logger.info("Kết nối Redis thành công.")
        except Exception as e:
            logger.error(f"Lỗi kết nối Redis: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def set(self, key, value, expire):
        try:
            serialized_value = pickle.dumps(value)
            await self.redis_client.set(key, serialized_value, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis set: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get(self, key):
        try:
            data = await self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Lỗi Redis get: {str(e)}")
            return None

redis_manager = RedisManager()

# ORM Models
class ApprovedUser(Base):
    __tablename__ = 'approved_users'
    id = Column(Integer, primary_key=True)
    user_id = Column(String, unique=True, nullable=False)
    approved_at = Column(DateTime, default=datetime.now)
    last_active = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

class ReportHistory(Base):
    __tablename__ = 'report_history'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False, default='1D')
    date = Column(String, nullable=False)
    report = Column(Text, nullable=False)
    close_today = Column(Float, nullable=False)
    close_yesterday = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now)

class TrainedModel(Base):
    __tablename__ = 'trained_models'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    model_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    performance = Column(Float, nullable=True)
    version = Column(String, nullable=False, default="1.0")
    params = Column(Text, nullable=True)

# DBManager async
class DBManager:
    """
    Quản lý cơ sở dữ liệu PostgreSQL thông qua SQLAlchemy async
    """
    def __init__(self):
        self.Session = SessionLocal

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def is_user_approved(self, user_id) -> bool:
        try:
            async with self.Session() as session:
                query = select(ApprovedUser.user_id).filter_by(user_id=str(user_id))
                result = await session.execute(query)
                return result.scalar_one_or_none() is not None or str(user_id) == ADMIN_ID
        except Exception as e:
            logger.error(f"Lỗi kiểm tra người dùng: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def add_approved_user(self, user_id, approved_at=None, notes=None) -> None:
        try:
            is_approved = await self.is_user_approved(user_id)
            if not is_approved and str(user_id) != ADMIN_ID:
                async with self.Session() as session:
                    try:
                        insert_stmt = text("""
                            INSERT INTO approved_users (user_id, approved_at, notes)
                            VALUES (:user_id, :approved_at, :notes)
                        """)
                        await session.execute(
                            insert_stmt,
                            {
                                "user_id": str(user_id),
                                "approved_at": approved_at or datetime.now(),
                                "notes": notes
                            }
                        )
                        await session.commit()
                        logger.info(f"Thêm người dùng được phê duyệt: {user_id}")
                    except Exception as inner_e:
                        logger.error(f"Lỗi khi thêm người dùng bằng SQL trực tiếp: {str(inner_e)}")
                        await session.rollback()
                        try:
                            new_user = ApprovedUser(
                                user_id=str(user_id),
                                approved_at=approved_at or datetime.now(),
                                notes=notes
                            )
                            session.add(new_user)
                            await session.commit()
                            logger.info(f"Thêm người dùng được phê duyệt (qua ORM): {user_id}")
                        except Exception as orm_e:
                            logger.error(f"Lỗi khi thêm người dùng qua ORM: {str(orm_e)}")
                            await session.rollback()
                            raise
        except Exception as e:
            logger.error(f"Lỗi thêm người dùng: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def update_user_last_active(self, user_id) -> None:
        try:
            async with self.Session() as session:
                query = select(ApprovedUser.id).filter_by(user_id=str(user_id))
                result = await session.execute(query)
                user_id_exists = result.scalar_one_or_none()
                
                if user_id_exists:
                    try:
                        update_stmt = text("""
                            UPDATE approved_users 
                            SET last_active = :now 
                            WHERE user_id = :user_id
                        """)
                        await session.execute(
                            update_stmt, 
                            {"now": datetime.now(), "user_id": str(user_id)}
                        )
                        await session.commit()
                    except Exception as e:
                        logger.warning(f"Không thể cập nhật last_active (có thể cột chưa tồn tại): {str(e)}")
                        await session.rollback()
        except Exception as e:
            logger.error(f"Lỗi cập nhật trạng thái người dùng: {str(e)}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_report_history(self, symbol: str, timeframe: str = '1D', limit: int = 10) -> list:
        try:
            async with self.Session() as session:
                query = select(ReportHistory).filter_by(symbol=symbol, timeframe=timeframe).order_by(ReportHistory.id.desc()).limit(limit)
                reports = await session.execute(query)
                reports = reports.scalars().all()
                return [
                    {
                        "id": report.id,
                        "symbol": report.symbol,
                        "timeframe": report.timeframe,
                        "date": report.date,
                        "report": report.report,
                        "close_today": report.close_today,
                        "close_yesterday": report.close_yesterday,
                        "timestamp": report.timestamp.isoformat()
                    }
                    for report in reports
                ]
        except Exception as e:
            logger.error(f"Lỗi tải lịch sử báo cáo: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def save_report_history(self, symbol: str, report: str, close_today: float, close_yesterday: float, timeframe: str = '1D') -> None:
        try:
            async with self.Session() as session:
                date_str = datetime.now().strftime('%Y-%m-%d')
                new_report = ReportHistory(
                    symbol=symbol,
                    timeframe=timeframe,
                    date=date_str,
                    report=report,
                    close_today=close_today,
                    close_yesterday=close_yesterday
                )
                session.add(new_report)
                await session.commit()
                logger.info(f"Lưu báo cáo mới cho {symbol} ({timeframe})")
        except Exception as e:
            logger.error(f"Lỗi lưu báo cáo: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def store_trained_model(self, symbol: str, model_type: str, model, performance: float = None, version: str = "1.0", params: dict = None):
        try:
            model_blob = pickle.dumps(model)
            params_json = json.dumps(params) if params else None
            
            async with self.Session() as session:
                result = await session.execute(
                    select(TrainedModel).filter_by(
                        symbol=symbol, 
                        model_type=model_type, 
                        version=version
                    )
                )
                existing = result.scalar_one_or_none()
                
                if existing:
                    existing.model_blob = model_blob
                    existing.created_at = datetime.now()
                    existing.performance = performance
                    existing.params = params_json
                else:
                    new_model = TrainedModel(
                        symbol=symbol, 
                        model_type=model_type, 
                        model_blob=model_blob, 
                        performance=performance,
                        version=version,
                        params=params_json
                    )
                    session.add(new_model)
                
                await session.commit()
                logger.info(f"Lưu mô hình {model_type} v{version} cho {symbol} thành công với hiệu suất: {performance}")
                
        except Exception as e:
            logger.error(f"Lỗi lưu mô hình {model_type} cho {symbol}: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_trained_model(self, symbol: str, model_type: str, version: str = None):
        try:
            async with self.Session() as session:
                if version:
                    query = select(TrainedModel).filter_by(symbol=symbol, model_type=model_type, version=version)
                else:
                    subquery = select(
                        TrainedModel.symbol,
                        TrainedModel.model_type,
                        func.max(TrainedModel.created_at).label('max_date')
                    ).filter_by(symbol=symbol, model_type=model_type).group_by(TrainedModel.symbol, TrainedModel.model_type).subquery()
                    
                    query = select(TrainedModel).join(
                        subquery,
                        (TrainedModel.symbol == subquery.c.symbol) &
                        (TrainedModel.model_type == subquery.c.model_type) &
                        (TrainedModel.created_at == subquery.c.max_date)
                    )
                
                result = await session.execute(query)
                model_record = result.scalar_one_or_none()
                
                if model_record:
                    logger.info(f"Tải mô hình {model_type} v{model_record.version} cho {symbol} thành công")
                    return {
                        'model': pickle.loads(model_record.model_blob),
                        'performance': model_record.performance,
                        'version': model_record.version,
                        'created_at': model_record.created_at,
                        'params': json.loads(model_record.params) if model_record.params else None
                    }
                return None, None, None, None, None
                
        except Exception as e:
            logger.error(f"Lỗi tải mô hình {model_type} cho {symbol}: {str(e)}")
            return None, None, None, None, None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get_model_versions(self, symbol: str, model_type: str) -> list:
        try:
            async with self.Session() as session:
                query = select(
                    TrainedModel.version,
                    TrainedModel.created_at,
                    TrainedModel.performance
                ).filter_by(symbol=symbol, model_type=model_type).order_by(TrainedModel.created_at.desc())
                
                result = await session.execute(query)
                versions = result.all()
                
                return [
                    {
                        'version': v.version,
                        'created_at': v.created_at.isoformat(),
                        'performance': v.performance
                    }
                    for v in versions
                ]
        except Exception as e:
            logger.error(f"Lỗi lấy versions mô hình cho {symbol}: {str(e)}")
            return []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get_training_symbols(self) -> list:
        try:
            async with self.Session() as session:
                query = select(TrainedModel.symbol).distinct()
                result = await session.execute(query)
                symbols = result.scalars().all()
                return symbols
        except Exception as e:
            logger.error(f"Lỗi lấy danh sách symbol: {str(e)}")
            return []

# Singleton instance
_db_manager = DBManager()

def get_db_manager():
    return _db_manager

def get_redis_manager():
    return redis_manager

# ---------- HÀM HỖ TRỢ GLOBAL ----------
async def is_user_approved(user_id: str) -> bool:
    db = get_db_manager()
    return await db.is_user_approved(str(user_id))

# ---------- HÀM HỖ TRỢ GLOBAL ----------
# (Đã loại bỏ hàm sync is_user_approved) 