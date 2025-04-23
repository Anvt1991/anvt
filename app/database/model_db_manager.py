import logging
import pickle
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from tenacity import retry, stop_after_attempt, wait_exponential

from app.models.database_models import TrainedModel

logger = logging.getLogger(__name__)

class ModelDBManager:
    def __init__(self, engine):
        self.engine = engine
        self.async_session = sessionmaker(
            engine, expire_on_commit=False, class_=AsyncSession
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def store_trained_model(self, symbol: str, model_type: str, model, performance: float = None):
        async with self.async_session() as session:
            try:
                # Kiểm tra xem đã có model cùng loại và cùng symbol chưa
                result = await session.execute(
                    select(TrainedModel)
                    .where(TrainedModel.symbol == symbol)
                    .where(TrainedModel.model_type == model_type)
                )
                
                existing_model = result.scalars().first()
                
                # Serialize model thành bytes
                model_bytes = pickle.dumps(model)
                
                if existing_model:
                    # Cập nhật model hiện có
                    existing_model.model_blob = model_bytes
                    if performance is not None:
                        existing_model.performance = performance
                else:
                    # Tạo model mới
                    new_model = TrainedModel(
                        symbol=symbol,
                        model_type=model_type,
                        model_blob=model_bytes,
                        performance=performance
                    )
                    session.add(new_model)
                
                await session.commit()
                logger.info(f"Đã lưu model {model_type} cho {symbol} vào database")
                return True
            except Exception as e:
                await session.rollback()
                logger.error(f"Lỗi khi lưu model: {str(e)}")
                return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def load_trained_model(self, symbol: str, model_type: str):
        async with self.async_session() as session:
            try:
                result = await session.execute(
                    select(TrainedModel)
                    .where(TrainedModel.symbol == symbol)
                    .where(TrainedModel.model_type == model_type)
                )
                
                model_record = result.scalars().first()
                
                if model_record:
                    # Deserialize model từ bytes
                    model = pickle.loads(model_record.model_blob)
                    logger.info(f"Đã tải model {model_type} cho {symbol} từ database")
                    return model, model_record.performance
                else:
                    logger.warning(f"Không tìm thấy model {model_type} cho {symbol} trong database")
                    return None, None
            except Exception as e:
                logger.error(f"Lỗi khi tải model: {str(e)}")
                return None, None 