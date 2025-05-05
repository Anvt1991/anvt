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

# Logging setup
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Timezone configuration
TZ = pytz.timezone('Asia/Bangkok')

# Admin ID from environment for access control
ADMIN_ID = os.environ.get("ADMIN_ID", "1225226589")

class DBManager:
    """
    Quản lý cơ sở dữ liệu sử dụng sqlite3 với migrations và theo dõi lịch sử.
    """
    VERSION = "1.1.0"  # Schema version
    DB_FILE = 'bot_sieucap_v19.db'
    
    def __init__(self, db_file=None):
        self.db_file = db_file or self.DB_FILE
        self.conn = None
        self.migrations = [
            self._migration_v1_0_0,
            self._migration_v1_1_0,
        ]
        self._setup_completed = False
        self.connect()
    
    def connect(self):
        """Kết nối đến cơ sở dữ liệu"""
        try:
            self.conn = sqlite3.connect(self.db_file)
            # Để nhận kết quả dưới dạng dict
            self.conn.row_factory = lambda cursor, row: {
                col[0]: row[idx] for idx, col in enumerate(cursor.description)
            }
            # Bật WAL mode để tối ưu hiệu suất đa luồng
            self.conn.execute("PRAGMA journal_mode = WAL")
            # Kiểm tra và áp dụng migrations nếu cần
            self._check_and_migrate()
            logger.info(f"Đã kết nối đến cơ sở dữ liệu {self.db_file}")
            self._setup_completed = True
            return self.conn
        except Exception as e:
            logger.error(f"Lỗi kết nối đến DB: {str(e)}")
            raise
    
    def _check_and_migrate(self):
        """Kiểm tra phiên bản schema và áp dụng migrations nếu cần"""
        try:
            # Kiểm tra bảng schema_version tồn tại chưa
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                # Nếu không tồn tại, tạo bảng schema_version và áp dụng tất cả migrations
                self.conn.execute("""
                    CREATE TABLE schema_version (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        version TEXT NOT NULL,
                        applied_at TEXT NOT NULL
                    )
                """)
                for migration in self.migrations:
                    migration(self)
                self.conn.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (self.VERSION, datetime.now(TZ).isoformat())
                )
                self.conn.commit()
                logger.info(f"Đã tạo schema mới phiên bản {self.VERSION}")
            else:
                # Nếu đã tồn tại, kiểm tra phiên bản hiện tại
                cursor = self.conn.execute("SELECT version FROM schema_version ORDER BY id DESC LIMIT 1")
                result = cursor.fetchone()
                current_version = result["version"] if result else "0.0.0"
                
                # Áp dụng migrations cần thiết
                applied = False
                for migration in self.migrations:
                    # Lấy phiên bản từ tên hàm migration (vd: _migration_v1_0_0 -> 1.0.0)
                    migration_version = migration.__name__.split('_v')[1].replace('_', '.')
                    if self._compare_versions(migration_version, current_version) > 0:
                        logger.info(f"Áp dụng migration {migration_version}")
                        migration(self)
                        self.conn.execute(
                            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                            (migration_version, datetime.now(TZ).isoformat())
                        )
                        applied = True
                
                if applied:
                    self.conn.commit()
                    logger.info(f"Đã cập nhật schema lên phiên bản {self.VERSION}")
                else:
                    logger.debug(f"Schema đã cập nhật (phiên bản hiện tại: {current_version})")
                    
        except Exception as e:
            logger.error(f"Lỗi kiểm tra/áp dụng migrations: {str(e)}")
            # Rollback trong trường hợp lỗi
            self.conn.rollback()
            raise
    
    def _compare_versions(self, version1, version2):
        """So sánh hai phiên bản semantically (1.0.0 > 0.9.0)"""
        v1_parts = list(map(int, version1.split('.')))
        v2_parts = list(map(int, version2.split('.')))
        
        for i in range(min(len(v1_parts), len(v2_parts))):
            if v1_parts[i] > v2_parts[i]:
                return 1
            elif v1_parts[i] < v2_parts[i]:
                return -1
        
        # Nếu các phần bằng nhau, so sánh độ dài
        if len(v1_parts) > len(v2_parts):
            return 1
        elif len(v1_parts) < len(v2_parts):
            return -1
        else:
            return 0
    
    def _migration_v1_0_0(self):
        """Migration ban đầu để tạo cấu trúc cơ sở dữ liệu"""
        self.conn.executescript("""
            -- Bảng người dùng đã được phê duyệt
            CREATE TABLE IF NOT EXISTS approved_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                approved_at TEXT NOT NULL,
                last_active TEXT,
                notes TEXT
            );
            
            -- Bảng lịch sử báo cáo
            CREATE TABLE IF NOT EXISTS report_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date TEXT NOT NULL,
                report TEXT NOT NULL,
                close_today REAL NOT NULL,
                close_yesterday REAL NOT NULL,
                timestamp TEXT NOT NULL,
                timeframe TEXT DEFAULT '1D'
            );
            
            -- Bảng mô hình đã huấn luyện (với version tracking)
            CREATE TABLE IF NOT EXISTS trained_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                model_type TEXT NOT NULL,
                model_blob BLOB NOT NULL,
                created_at TEXT NOT NULL,
                performance REAL,
                version TEXT NOT NULL,
                params TEXT,
                UNIQUE(symbol, model_type, version)
            );
            
            -- Bảng lịch sử hiệu suất mô hình
            CREATE TABLE IF NOT EXISTS model_performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                evaluated_at TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                FOREIGN KEY (model_id) REFERENCES trained_models (id)
            );
            
            -- Chỉ mục để cải thiện hiệu suất truy vấn
            CREATE INDEX IF NOT EXISTS idx_report_history_symbol ON report_history (symbol);
            CREATE INDEX IF NOT EXISTS idx_report_history_date ON report_history (date);
            CREATE INDEX IF NOT EXISTS idx_trained_models_symbol ON trained_models (symbol);
            CREATE INDEX IF NOT EXISTS idx_trained_models_model_type ON trained_models (model_type);
        """)
        self.conn.commit()
        logger.info("Đã áp dụng migration v1.0.0")
    
    def _migration_v1_1_0(self):
        """Migration để thêm bảng history_reports cho tương thích ngược"""
        self.conn.executescript("""
            -- Bảng history_reports (sử dụng trong ChatbotAI._pipeline_save_to_database)
            CREATE TABLE IF NOT EXISTS history_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                report TEXT NOT NULL,
                created_at TEXT NOT NULL,
                close_price REAL NOT NULL,
                previous_close REAL NOT NULL
            );
            
            -- Thêm chỉ mục cho truy vấn hiệu quả
            CREATE INDEX IF NOT EXISTS idx_history_reports_symbol ON history_reports (symbol);
            CREATE INDEX IF NOT EXISTS idx_history_reports_created_at ON history_reports (created_at);
            
            -- Hàm để đồng bộ dữ liệu giữa report_history và history_reports
            -- Trong tương lai, sẽ loại bỏ một trong hai bảng này
        """)
        self.conn.commit()
        logger.info("Đã áp dụng migration v1.1.0")
    
    def is_user_approved(self, user_id: str) -> bool:
        """Kiểm tra người dùng đã được phê duyệt chưa"""
        if not self._setup_completed:
            self.connect()
        
        if user_id == ADMIN_ID:
            return True
            
        cursor = self.conn.execute(
            "SELECT * FROM approved_users WHERE user_id = ?", (user_id,)
        )
        result = cursor.fetchone()
        return result is not None
    
    def add_approved_user(self, user_id: str, approved_at: str = None, notes: str = None):
        """Thêm người dùng vào danh sách được phê duyệt"""
        if not self._setup_completed:
            self.connect()
            
        is_approved = self.is_user_approved(user_id)
        if not is_approved and user_id != ADMIN_ID:
            approved_at = approved_at or datetime.now(TZ).isoformat()
            try:
                self.conn.execute(
                    "INSERT INTO approved_users (user_id, approved_at, notes) VALUES (?, ?, ?)",
                    (user_id, approved_at, notes)
                )
                self.conn.commit()
                logger.info(f"Thêm người dùng được phê duyệt: {user_id}")
            except Exception as e:
                logger.error(f"Lỗi thêm người dùng {user_id}: {str(e)}")
                self.conn.rollback()
    
    def update_user_last_active(self, user_id: str):
        """Cập nhật thời gian hoạt động gần nhất của người dùng"""
        if not self._setup_completed:
            self.connect()
            
        try:
            # Kiểm tra xem user có tồn tại trong hệ thống không
            cursor = self.conn.execute(
                "SELECT 1 FROM approved_users WHERE user_id = ?", (user_id,)
            )
            user_exists = cursor.fetchone()
                
            # Nếu user không tồn tại, không cần cập nhật
            if not user_exists:
                logger.info(f"Không cập nhật last_active: User {user_id} không tồn tại trong database")
                return
                
            # Kiểm tra xem cột last_active có tồn tại không
            cursor = self.conn.execute("PRAGMA table_info(approved_users)")
            columns = cursor.fetchall()
            column_names = [col["name"] for col in columns]
                
            if "last_active" in column_names:
                self.conn.execute(
                    "UPDATE approved_users SET last_active = ? WHERE user_id = ?",
                    (datetime.now(TZ).isoformat(), user_id)
                )
                self.conn.commit()
            else:
                logger.warning(f"Cột 'last_active' không tồn tại trong bảng approved_users")
        except Exception as e:
            logger.error(f"Lỗi cập nhật thời gian hoạt động người dùng {user_id}: {str(e)}")
            # Rollback nhưng không ảnh hưởng đến luồng chính
            try:
                self.conn.rollback()
            except:
                pass
    
    def load_report_history(self, symbol: str, timeframe: str = '1D', limit: int = 10) -> list:
        """Tải lịch sử báo cáo của một mã từ bảng report_history"""
        if not self._setup_completed:
            self.connect()
            
        cursor = self.conn.execute(
            "SELECT * FROM report_history WHERE symbol = ? AND timeframe = ? ORDER BY id DESC LIMIT ?",
            (symbol, timeframe, limit)
        )
        return cursor.fetchall()
    
    def save_report_history(self, symbol: str, report: str, close_today: float, close_yesterday: float, timeframe: str = '1D'):
        """Lưu báo cáo phân tích vào bảng report_history"""
        if not self._setup_completed:
            self.connect()
            
        date_str = datetime.now(TZ).strftime('%Y-%m-%d')
        timestamp = datetime.now(TZ).isoformat()
        try:
            self.conn.execute(
                """
                INSERT INTO report_history 
                (symbol, date, report, close_today, close_yesterday, timestamp, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, date_str, report, close_today, close_yesterday, timestamp, timeframe)
            )
            self.conn.commit()
            logger.info(f"Đã lưu báo cáo cho {symbol} vào report_history")
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu báo cáo: {str(e)}")
            self.conn.rollback()
            return False
    
    def load_history_report(self, symbol: str, limit: int = 1) -> list:
        """Tải báo cáo lịch sử từ bảng history_reports"""
        if not self._setup_completed:
            self.connect()
            
        cursor = self.conn.execute(
            "SELECT * FROM history_reports WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
            (symbol, limit)
        )
        return cursor.fetchall()
    
    def save_history_report(self, symbol: str, report: str, close_price: float, previous_close: float):
        """Lưu báo cáo vào bảng history_reports (được sử dụng bởi ChatbotAI._pipeline_save_to_database)"""
        if not self._setup_completed:
            self.connect()
            
        created_at = datetime.now(TZ).isoformat()
        try:
            self.conn.execute(
                """
                INSERT INTO history_reports 
                (symbol, report, created_at, close_price, previous_close) 
                VALUES (?, ?, ?, ?, ?)
                """,
                (symbol, report, created_at, close_price, previous_close)
            )
            self.conn.commit()
            logger.info(f"Đã lưu báo cáo cho {symbol} vào history_reports")
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu báo cáo vào history_reports: {str(e)}")
            self.conn.rollback()
            return False
    
    def store_trained_model(self, symbol: str, model_type: str, model, 
                          performance: float = None, version: str = "1.0",
                          params: dict = None):
        """Lưu mô hình đã huấn luyện với version tracking"""
        if not self._setup_completed:
            self.connect()
            
        model_blob = pickle.dumps(model)
        created_at = datetime.now(TZ).isoformat()
        params_json = json.dumps(params) if params else None
        
        try:
            # Lưu mô hình mới
            self.conn.execute(
                """
                INSERT INTO trained_models 
                (symbol, model_type, model_blob, created_at, performance, version, params)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (symbol, model_type, model_blob, created_at, performance, version, params_json)
            )
            self.conn.commit()
            
            # Lấy ID của mô hình vừa lưu
            cursor = self.conn.execute(
                "SELECT id FROM trained_models WHERE symbol = ? AND model_type = ? AND version = ?",
                (symbol, model_type, version)
            )
            model_id = cursor.fetchone()["id"]
                
            # Lưu hiệu suất vào lịch sử
            if performance is not None:
                self.conn.execute(
                    """
                    INSERT INTO model_performance_history
                    (model_id, evaluated_at, metric_name, metric_value)
                    VALUES (?, ?, ?, ?)
                    """,
                    (model_id, created_at, "accuracy", performance)
                )
                self.conn.commit()
            logger.info(f"Đã lưu mô hình {model_type} cho {symbol} phiên bản {version}")
            return True
        except Exception as e:
            logger.error(f"Lỗi lưu mô hình: {str(e)}")
            self.conn.rollback()
            return False
    
    def load_trained_model(self, symbol: str, model_type: str, version: str = None):
        """Tải mô hình đã huấn luyện, mặc định lấy version mới nhất"""
        if not self._setup_completed:
            self.connect()
            
        try:
            if version:
                # Lấy mô hình theo version cụ thể
                cursor = self.conn.execute(
                    """
                    SELECT id, model_blob, performance, version, params
                    FROM trained_models 
                    WHERE symbol = ? AND model_type = ? AND version = ?
                    """,
                    (symbol, model_type, version)
                )
                result = cursor.fetchone()
            else:
                # Lấy mô hình mới nhất
                cursor = self.conn.execute(
                    """
                    SELECT id, model_blob, performance, version, params
                    FROM trained_models 
                    WHERE symbol = ? AND model_type = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (symbol, model_type)
                )
                result = cursor.fetchone()
            
            if result:
                model_id = result["id"]
                model = pickle.loads(result["model_blob"])
                performance = result["performance"]
                version = result["version"]
                params = json.loads(result["params"]) if result["params"] else None
                
                # Lấy lịch sử hiệu suất
                cursor = self.conn.execute(
                    """
                    SELECT metric_name, metric_value, evaluated_at
                    FROM model_performance_history
                    WHERE model_id = ?
                    ORDER BY evaluated_at DESC
                    """,
                    (model_id,)
                )
                perf_history = cursor.fetchall()
                
                logger.info(f"Đã tải mô hình {model_type} cho {symbol} (phiên bản {version})")
                # Để tương thích với API cũ chỉ trả về (model, performance)
                return model, performance
            logger.warning(f"Không tìm thấy mô hình {model_type} cho {symbol}")
            return None, None
        except Exception as e:
            logger.error(f"Lỗi tải mô hình: {str(e)}")
            return None, None
    
    def get_model_versions(self, symbol: str, model_type: str) -> list:
        """Lấy danh sách các phiên bản của mô hình"""
        if not self._setup_completed:
            self.connect()
            
        cursor = self.conn.execute(
            """
            SELECT version, created_at, performance
            FROM trained_models
            WHERE symbol = ? AND model_type = ?
            ORDER BY id DESC
            """,
            (symbol, model_type)
        )
        return cursor.fetchall()
    
    def get_training_symbols(self) -> list:
        """Lấy danh sách các mã đã có trong lịch sử báo cáo"""
        if not self._setup_completed:
            self.connect()
            
        cursor = self.conn.execute(
            "SELECT DISTINCT symbol FROM report_history"
        )
        results = cursor.fetchall()
        return [row["symbol"] for row in results]
    
    def migrate_from_sync_db(self, sync_db_file: str = 'bot_sieucap_v18.db'):
        """Di chuyển dữ liệu từ cơ sở dữ liệu sqlite đồng bộ cũ"""
        if not self._setup_completed:
            self.connect()
            
        try:
            # Kết nối đến db cũ
            sync_conn = sqlite3.connect(sync_db_file)
            sync_conn.row_factory = sqlite3.Row
            
            # Di chuyển người dùng đã phê duyệt
            cursor = sync_conn.execute("SELECT * FROM approved_users")
            for row in cursor.fetchall():
                user_id = row['user_id']
                approved_at = row['approved_at']
                self.conn.execute(
                    "INSERT OR IGNORE INTO approved_users (user_id, approved_at) VALUES (?, ?)",
                    (user_id, approved_at)
                )
            
            # Di chuyển lịch sử báo cáo
            cursor = sync_conn.execute("SELECT * FROM report_history")
            for row in cursor.fetchall():
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO report_history 
                    (symbol, date, report, close_today, close_yesterday, timestamp, timeframe)
                    VALUES (?, ?, ?, ?, ?, ?, '1D')
                    """,
                    (row['symbol'], row['date'], row['report'], 
                     row['close_today'], row['close_yesterday'], row['timestamp'])
                )
            
            # Di chuyển mô hình đã huấn luyện
            cursor = sync_conn.execute("SELECT * FROM trained_models")
            for row in cursor.fetchall():
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO trained_models
                    (symbol, model_type, model_blob, created_at, performance, version, params)
                    VALUES (?, ?, ?, ?, ?, '1.0', NULL)
                    """,
                    (row['symbol'], row['model_type'], row['model_blob'], 
                     row['created_at'], row['performance'])
                )
            
            # Di chuyển dữ liệu từ history_reports nếu có
            try:
                cursor = sync_conn.execute("SELECT * FROM history_reports")
                for row in cursor.fetchall():
                    self.conn.execute(
                        """
                        INSERT OR IGNORE INTO history_reports
                        (symbol, report, created_at, close_price, previous_close)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (row['symbol'], row['report'], row['created_at'],
                         row['close_price'], row['previous_close'])
                    )
            except sqlite3.OperationalError:
                logger.info("Bảng history_reports không tồn tại trong DB cũ")
            
            self.conn.commit()
            sync_conn.close()
            logger.info("Di chuyển dữ liệu từ cơ sở dữ liệu cũ thành công")
            return True
        except Exception as e:
            logger.error(f"Lỗi di chuyển dữ liệu: {str(e)}")
            self.conn.rollback()
            return False
    
    def close(self):
        """Đóng kết nối cơ sở dữ liệu"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self._setup_completed = False
            logger.info("Đã đóng kết nối DB")

    def validate_report_data(self, report_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Kiểm tra tính hợp lệ của dữ liệu báo cáo trước khi lưu vào database
        
        Args:
            report_data: Dữ liệu báo cáo cần kiểm tra
            
        Returns:
            Tuple (valid, message) - valid là True nếu dữ liệu hợp lệ, message là thông báo lỗi nếu có
        """
        if not isinstance(report_data, dict):
            return False, "Dữ liệu báo cáo không phải là dict"
            
        # Kiểm tra các trường quan trọng trong báo cáo
        required_fields = ['symbol', 'timestamp']
        for field in required_fields:
            if field not in report_data:
                return False, f"Thiếu trường dữ liệu quan trọng: {field}"
        
        # Kiểm tra định dạng mã chứng khoán
        symbol = report_data.get('symbol', '')
        if not symbol or not isinstance(symbol, str) or len(symbol) > 10:
            return False, f"Mã chứng khoán không hợp lệ: {symbol}"
            
        # Kiểm tra pattern_analysis nếu có
        if 'pattern_analysis' in report_data:
            pattern_analysis = report_data['pattern_analysis']
            if not isinstance(pattern_analysis, dict):
                return False, "pattern_analysis không phải là dict"
                
            # Kiểm tra technical_patterns
            if 'technical_patterns' in pattern_analysis:
                patterns = pattern_analysis['technical_patterns']
                if not isinstance(patterns, list):
                    return False, "technical_patterns không phải là list"
                    
                # Kiểm tra từng pattern
                for i, pattern in enumerate(patterns):
                    if not isinstance(pattern, dict):
                        return False, f"Pattern #{i} không phải là dict"
                    if 'pattern_name' not in pattern:
                        return False, f"Pattern #{i} thiếu pattern_name"
            
            # Kiểm tra support_resistance
            if 'support_resistance' in pattern_analysis:
                sr = pattern_analysis['support_resistance']
                if not isinstance(sr, dict):
                    return False, "support_resistance không phải là dict"
                    
                # Kiểm tra support_levels và resistance_levels
                for field in ['support_levels', 'resistance_levels']:
                    if field in sr and not isinstance(sr[field], list):
                        return False, f"{field} không phải là list"
                        
                    # Kiểm tra xem các giá trị có phải là số không
                    if field in sr:
                        for i, level in enumerate(sr[field]):
                            if not isinstance(level, (int, float)) and not (isinstance(level, dict) and 'value' in level):
                                return False, f"{field}[{i}] không phải là số hoặc dict có 'value'"
        
        # Kiểm tra định dạng dữ liệu giá
        if 'last_price' in report_data and not isinstance(report_data['last_price'], (int, float)):
            return False, f"last_price không phải là số: {report_data['last_price']}"
            
        if 'previous_price' in report_data and not isinstance(report_data['previous_price'], (int, float)):
            return False, f"previous_price không phải là số: {report_data['previous_price']}"
            
        # Nếu mọi kiểm tra đều thành công
        return True, "Dữ liệu hợp lệ"

    def save_market_report(self, market_report_data: Dict[str, Any]) -> bool:
        """
        Lưu báo cáo phân tích thị trường (VNINDEX) vào cơ sở dữ liệu
        
        Args:
            market_report_data: Dict chứa dữ liệu báo cáo với các khóa:
                - report: Nội dung báo cáo
                - vnindex_value: Giá trị VNINDEX tại thời điểm tạo báo cáo
                - created_at: Thời gian tạo báo cáo (dạng ISO)
                - data: Dict chứa dữ liệu phân tích (tùy chọn)
                
        Returns:
            True nếu lưu thành công, False nếu lỗi
        """
        if not self._setup_completed:
            self.connect()
            
        # Kiểm tra dữ liệu đầu vào
        if not isinstance(market_report_data, dict):
            logger.error("market_report_data phải là dict")
            return False
            
        if "report" not in market_report_data or not isinstance(market_report_data["report"], str):
            logger.error("market_report_data phải có khóa 'report' là string")
            return False
            
        # Trích xuất dữ liệu
        report = market_report_data["report"]
        vnindex_value = market_report_data.get("vnindex_value")
        created_at = market_report_data.get("created_at", datetime.now(TZ).isoformat())
        
        # Chuyển data thành JSON nếu có
        data_json = None
        if "data" in market_report_data and market_report_data["data"]:
            try:
                data_json = json.dumps(market_report_data["data"], default=str)
            except Exception as e:
                logger.error(f"Lỗi khi chuyển đổi data sang JSON: {str(e)}")
        
        try:
            # Tạo bảng market_reports nếu chưa tồn tại
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS market_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report TEXT NOT NULL,
                vnindex_value REAL,
                created_at TEXT NOT NULL,
                data_json TEXT
            )
            """)
            self.conn.commit()
            
            # Thêm báo cáo vào cơ sở dữ liệu
            self.conn.execute("""
            INSERT INTO market_reports (report, vnindex_value, created_at, data_json)
            VALUES (?, ?, ?, ?)
            """, (report, vnindex_value, created_at, data_json))
            self.conn.commit()
            
            logger.info("Đã lưu báo cáo thị trường vào cơ sở dữ liệu")
            
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu báo cáo thị trường: {str(e)}")
            return False
            
    def load_market_reports(self, limit: int = 5) -> list:
        """
        Tải các báo cáo thị trường gần đây
        
        Args:
            limit: Số lượng báo cáo tối đa
            
        Returns:
            Danh sách báo cáo
        """
        if not self._setup_completed:
            self.connect()
            
        try:
            cursor = self.conn.execute("""
            SELECT id, report, vnindex_value, created_at, data_json
            FROM market_reports
            ORDER BY created_at DESC
            LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
                
            reports = []
            for row in rows:
                report_data = {
                    "id": row["id"],
                    "report": row["report"],
                    "vnindex_value": row["vnindex_value"],
                    "created_at": row["created_at"]
                }
                
                # Chuyển đổi data_json thành dict nếu có
                if row["data_json"]:
                    try:
                        report_data["data"] = json.loads(row["data_json"])
                    except:
                        report_data["data"] = None
                        
                reports.append(report_data)
                
            return reports
            
        except Exception as e:
            logger.error(f"Lỗi khi tải báo cáo thị trường: {str(e)}")
            return []

# Khởi tạo singleton instance để sử dụng trong ứng dụng
db = DBManager()

# ---------- HÀM HỖ TRỢ GLOBAL ----------
def is_user_approved(user_id: str) -> bool:
    """
    Wrapper function để kiểm tra người dùng đã được phê duyệt chưa
    Được sử dụng như một tiện ích toàn cục cho tương thích.
    """
    from core.data.db import db
    return db.is_user_approved(str(user_id)) 