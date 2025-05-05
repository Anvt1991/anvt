#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module chuyển tiếp sang load_data.py
(Giữ lại để đảm bảo tương thích với code cũ)
"""

import logging
from core.data.load_data import DataLoader
from core.data.data_validator import DataValidator

# Thiết lập logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Định nghĩa CacheManager để đảm bảo tương thích với code cũ
class CacheManager:
    """
    Lớp quản lý cache đơn giản
    """
    def __init__(self):
        self.cache = {}
    
    def get(self, key, default=None):
        return self.cache.get(key, default)
    
    def set(self, key, value):
        self.cache[key] = value
        
    def clear(self):
        self.cache.clear()

# Giữ lại StockDataLoader để tương thích ngược
class StockDataLoader(DataLoader):
    """
    Lớp tương thích ngược với code cũ
    """
    pass

# Log thông báo cho developer
logger.info("data_loader.py is now a wrapper for load_data.py. Consider updating your imports to use load_data.py directly.")