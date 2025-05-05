#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cấu hình logging cho toàn bộ ứng dụng
"""

import os
import logging
import logging.handlers
import sys
from datetime import datetime
import json
from typing import Dict, Any, Optional
import platform
from pathlib import Path

# Cấu hình thư mục logs
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Tên file log
LOG_FILENAME = os.path.join(LOG_DIR, "botchatai.log")
ERROR_FILENAME = os.path.join(LOG_DIR, "error.log")
DEBUG_FILENAME = os.path.join(LOG_DIR, "debug.log")
PIPELINE_FILENAME = os.path.join(LOG_DIR, "pipeline.log")
API_FILENAME = os.path.join(LOG_DIR, "api.log")

# Định dạng log
CONSOLE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
JSON_FORMAT = "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d %(funcName)s %(message)s"

class JsonFormatter(logging.Formatter):
    """
    Định dạng log dưới dạng JSON
    """
    def __init__(self, fmt=None, datefmt=None, style='%', ensure_ascii=False):
        super().__init__(fmt, datefmt, style)
        self.ensure_ascii = ensure_ascii
        
    def format(self, record):
        """
        Format log record as JSON
        """
        log_record = {}
        
        # Add standard log record attributes
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["message"] = record.getMessage()
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno
        log_record["thread_id"] = record.thread
        log_record["thread_name"] = record.threadName
        
        # Add exception info if exists
        if record.exc_info:
            log_record["exception"] = {
                "type": record.exc_info[0].__name__,
                "value": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
            
        # Add custom fields from record
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", "filename", 
                          "funcName", "id", "levelname", "levelno", "lineno", "module",
                          "msecs", "message", "msg", "name", "pathname", "process",
                          "processName", "relativeCreated", "stack_info", "thread", "threadName"]:
                log_record[key] = value
        
        # Add system info
        log_record["system"] = {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version()
        }
        
        # Convert to JSON
        return json.dumps(log_record, ensure_ascii=self.ensure_ascii)

def configure_logging(level: int = logging.INFO, 
                     console: bool = True, 
                     file: bool = True,
                     json_format: bool = False,
                     component: str = "main",
                     log_dir: Optional[str] = None) -> logging.Logger:
    """
    Cấu hình logging cho ứng dụng
    
    Args:
        level: Cấp độ log
        console: Bật log ra console
        file: Bật log ra file
        json_format: Sử dụng định dạng JSON
        component: Thành phần của ứng dụng
        log_dir: Thư mục chứa file log (nếu None thì sử dụng thư mục mặc định)
        
    Returns:
        Logger đã cấu hình
    """
    # Xác định thư mục log
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = LOG_DIR
        
    # Tạo logger
    logger = logging.getLogger(f"botchatai.{component}")
    logger.setLevel(level)
    
    # Xóa tất cả handlers hiện có
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Định dạng log
    if json_format:
        formatter = JsonFormatter()
    else:
        console_formatter = logging.Formatter(CONSOLE_FORMAT)
        file_formatter = logging.Formatter(FILE_FORMAT)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter if not json_format else formatter)
        logger.addHandler(console_handler)
    
    # File handlers
    if file:
        # Component-specific log files
        if component == "api":
            log_file = os.path.join(log_dir, "api.log")
        elif component == "pipeline":
            log_file = os.path.join(log_dir, "pipeline.log")
        elif component == "data":
            log_file = os.path.join(log_dir, "data.log")
        else:
            log_file = os.path.join(log_dir, "botchatai.log")
            
        # Main log file - keep 7 days of logs, max 10MB each
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=7
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter if not json_format else formatter)
        logger.addHandler(file_handler)
        
        # Error log file - only ERROR and above
        error_handler = logging.handlers.RotatingFileHandler(
            ERROR_FILENAME, maxBytes=10*1024*1024, backupCount=7
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter if not json_format else formatter)
        logger.addHandler(error_handler)
        
        # Debug log file - only if level is DEBUG
        if level <= logging.DEBUG:
            debug_handler = logging.handlers.RotatingFileHandler(
                DEBUG_FILENAME, maxBytes=10*1024*1024, backupCount=3
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(file_formatter if not json_format else formatter)
            logger.addHandler(debug_handler)
    
    return logger

def add_custom_field(logger: logging.Logger, name: str, value: Any):
    """
    Thêm trường tùy chỉnh vào logger
    
    Args:
        logger: Logger cần thêm trường
        name: Tên trường
        value: Giá trị
    """
    # Thêm dữ liệu tùy chỉnh vào tất cả log records
    class CustomAdapter(logging.LoggerAdapter):
        def process(self, msg, kwargs):
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra'][name] = value
            return msg, kwargs
    
    return CustomAdapter(logger, {})

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Lấy logger với cấu hình có sẵn
    
    Args:
        name: Tên logger
        level: Cấp độ log
        
    Returns:
        Logger đã cấu hình
    """
    # Xác định thành phần dựa trên tên
    if name.startswith("botchatai."):
        component = name.split(".")[1] if len(name.split(".")) > 1 else "main"
    else:
        component = name
        name = f"botchatai.{name}"
    
    # Lấy logger từ logging system
    logger = logging.getLogger(name)
    
    # Nếu logger chưa được cấu hình, cấu hình nó
    if not logger.handlers:
        logger = configure_logging(level=level, component=component)
    
    return logger

# Cấu hình mặc định khi module được import
def setup_default_logging():
    """Thiết lập cấu hình logging mặc định"""
    # Cấu hình root logger
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        configure_logging(level=logging.INFO)
    
    # Cấu hình logger cho các thành phần chính
    configure_logging(level=logging.INFO, component="api")
    configure_logging(level=logging.INFO, component="pipeline")
    configure_logging(level=logging.INFO, component="data")
    
    # Thiết lập các logger bên thứ 3 không quá nhiều log
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("pandas").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

# Tự động cấu hình khi module được import
setup_default_logging() 