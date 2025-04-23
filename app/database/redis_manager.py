import pickle
import logging
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# Thay thế aioredis bằng redis.asyncio để tránh lỗi TimeoutError
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.redis_client = None
        
    async def connect(self):
        try:
            self.redis_client = redis.from_url(self.redis_url)
            # Kiểm tra kết nối đến Redis
            await self.redis_client.ping()
            logger.info("Kết nối Redis thành công.")
        except Exception as e:
            logger.error(f"Lỗi kết nối Redis: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def set(self, key, value, expire):
        try:
            if self.redis_client is None:
                await self.connect()
                
            serialized_value = pickle.dumps(value)
            await self.redis_client.set(key, serialized_value, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Lỗi Redis set: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def get(self, key):
        try:
            if self.redis_client is None:
                await self.connect()
                
            data = await self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"Lỗi Redis get: {str(e)}")
            return None
            
    async def optimize_cache(self):
        """Tối ưu bộ nhớ Redis bằng cách xóa cache cũ và không sử dụng"""
        try:
            if self.redis_client is None:
                await self.connect()
                
            # Lấy tất cả các key từ Redis
            all_keys = await self.redis_client.keys("*")
            current_time = datetime.now()
            deleted_count = 0
            
            # Ưu tiên xóa các loại cache khác nhau
            for key in all_keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                
                # Xóa cache quá cũ (7+ ngày)
                ttl = await self.redis_client.ttl(key)
                if ttl < 0:  # Không có TTL hoặc key không tồn tại
                    await self.redis_client.delete(key)
                    deleted_count += 1
                
                # Xóa cache tạm thời (các key tạm thời)
                if 'temp_' in key_str:
                    await self.redis_client.delete(key)
                    deleted_count += 1
                    
                # Xóa cache của các phiên phân tích cũ
                if 'analysis_' in key_str and '_' in key_str:
                    # Format thường là analysis_symbol_timestamp
                    try:
                        parts = key_str.split('_')
                        if len(parts) >= 3:
                            # Nếu có timestamp trong key
                            pass
                    except:
                        # Nếu không parse được, xóa luôn
                        await self.redis_client.delete(key)
                        deleted_count += 1
            
            logger.info(f"Đã tối ưu Redis cache: xóa {deleted_count} key.")
            return deleted_count
        except Exception as e:
            logger.error(f"Lỗi khi tối ưu Redis cache: {str(e)}")
            return 0 