#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto retrain ML models for all symbols on a schedule
"""

import os
import time
import logging
import fcntl
from apscheduler.schedulers.background import BackgroundScheduler
from core.model.enhanced_predictor import EnhancedPredictor
from core.data.data import DataLoader

# Cấu hình logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_symbols():
    """
    Lấy danh sách tất cả các mã cổ phiếu cần train lại.
    Có thể lấy từ database, file, hoặc hardcode tạm thời.
    """
    # Ví dụ: lấy từ DataLoader (bạn có thể thay đổi cho phù hợp)
    try:
        loader = DataLoader()
        symbols = loader.get_vn30_stocks()  # hoặc loader.get_all_symbols()
        return symbols
    except Exception as e:
        logger.error(f"Không lấy được danh sách mã: {e}")
        return []

def retrain_symbol(symbol, predictor, loader):
    lock_file = f"/tmp/train_lock_{symbol}.lock"
    with open(lock_file, "w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            data = loader.load_stock_data(symbol)
            if data is None or data.empty:
                logger.warning(f"Không có dữ liệu cho {symbol}, bỏ qua.")
                return
            result = predictor.train_model(symbol, data)
            if result.get("success"):
                logger.info(f"✓ Đã train lại model cho {symbol} ({result.get('best_model')})")
            else:
                logger.warning(f"Lỗi train lại model cho {symbol}: {result.get('error')}")
        except BlockingIOError:
            logger.info(f"Đã có tiến trình train {symbol} đang chạy, bỏ qua.")
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def retrain_all_models():
    logger.info("Bắt đầu train lại model cho tất cả các mã...")
    predictor = EnhancedPredictor()
    symbols = get_all_symbols()
    if not symbols:
        logger.warning("Không có mã nào để train lại.")
        return
    for symbol in symbols:
        try:
            loader = DataLoader()
            retrain_symbol(symbol, predictor, loader)
        except Exception as e:
            logger.error(f"Lỗi khi train lại {symbol}: {e}")
    logger.info("Hoàn tất train lại model cho tất cả các mã.")

def main():
    scheduler = BackgroundScheduler()
    # Đặt lịch mỗi tuần (hoặc thay đổi tuỳ ý)
    scheduler.add_job(retrain_all_models, 'interval', weeks=1, next_run_time=None)
    scheduler.start()
    logger.info("Auto retrain scheduler đã khởi động. Nhấn Ctrl+C để dừng.")
    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Đã dừng auto retrain scheduler.")

if __name__ == "__main__":
    main() 