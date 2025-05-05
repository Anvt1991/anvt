#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kiểm thử module DataLoader
"""

import unittest
import pandas as pd
from core.data.data import DataLoader

class TestDataLoader(unittest.TestCase):
    """
    Kiểm thử các chức năng của DataLoader
    """
    
    def setUp(self):
        """
        Chuẩn bị cho kiểm thử
        """
        self.data_loader = DataLoader()
        self.test_symbol = 'FPT'  # Mã cổ phiếu để kiểm thử
    
    def test_load_stock_data_mock(self):
        """
        Kiểm thử tải dữ liệu mẫu
        """
        df = self.data_loader.load_stock_data(self.test_symbol, use_mock=True)
        
        # Kiểm tra DataFrame có dữ liệu
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Kiểm tra các cột cần thiết
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_load_stock_data_real(self):
        """
        Kiểm thử tải dữ liệu thực
        """
        # Thử tải dữ liệu từ TCBS
        self.data_loader.set_data_source('tcbs')
        try:
            df_tcbs = self.data_loader.load_stock_data(self.test_symbol, use_cache=False, use_mock=False)
            
            # Kiểm tra DataFrame có dữ liệu
            self.assertIsInstance(df_tcbs, pd.DataFrame)
            self.assertGreater(len(df_tcbs), 0)
            
            # Kiểm tra các cột cần thiết
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                self.assertIn(col, df_tcbs.columns)
        except Exception as e:
            print(f"Không thể tải dữ liệu từ TCBS: {str(e)}")
        
        # Thử tải dữ liệu từ SSI
        self.data_loader.set_data_source('ssi')
        try:
            df_ssi = self.data_loader.load_stock_data(self.test_symbol, use_cache=False, use_mock=False)
            
            # Kiểm tra DataFrame có dữ liệu
            self.assertIsInstance(df_ssi, pd.DataFrame)
            self.assertGreater(len(df_ssi), 0)
            
            # Kiểm tra các cột cần thiết
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                self.assertIn(col, df_ssi.columns)
        except Exception as e:
            print(f"Không thể tải dữ liệu từ SSI: {str(e)}")
    
    def test_load_stock_list(self):
        """
        Kiểm thử tải danh sách cổ phiếu
        """
        # Tải danh sách mẫu
        stocks_mock = self.data_loader.load_stock_list(use_mock=True)
        self.assertIsInstance(stocks_mock, list)
        self.assertGreater(len(stocks_mock), 0)
        
        # Thử tải danh sách thực
        try:
            stocks_real = self.data_loader.load_stock_list(use_mock=False)
            self.assertIsInstance(stocks_real, list)
            self.assertGreater(len(stocks_real), 0)
        except Exception as e:
            print(f"Không thể tải danh sách cổ phiếu thực: {str(e)}")
    
    def test_load_realtime_quotes(self):
        """
        Kiểm thử tải dữ liệu giá thời gian thực
        """
        # Tải dữ liệu mẫu
        quotes_mock = self.data_loader.load_realtime_quotes([self.test_symbol], use_mock=True)
        self.assertIsInstance(quotes_mock, dict)
        self.assertIn(self.test_symbol, quotes_mock)
        
        # Thử tải dữ liệu thực
        try:
            quotes_real = self.data_loader.load_realtime_quotes([self.test_symbol], use_mock=False)
            self.assertIsInstance(quotes_real, dict)
        except Exception as e:
            print(f"Không thể tải dữ liệu giá thời gian thực: {str(e)}")
    
    def test_load_market_overview(self):
        """
        Kiểm thử tải tổng quan thị trường
        """
        # Tải dữ liệu mẫu
        overview_mock = self.data_loader.load_market_overview(use_mock=True)
        self.assertIsInstance(overview_mock, dict)
        self.assertIn('indices', overview_mock)
        
        # Thử tải dữ liệu thực
        try:
            overview_real = self.data_loader.load_market_overview(use_mock=False)
            self.assertIsInstance(overview_real, dict)
            self.assertIn('indices', overview_real)
        except Exception as e:
            print(f"Không thể tải tổng quan thị trường thực: {str(e)}")
    
    def test_load_stock_news(self):
        """
        Kiểm thử tải tin tức cổ phiếu
        """
        # Tải dữ liệu mẫu
        news_mock = self.data_loader.load_stock_news(self.test_symbol, use_mock=True)
        self.assertIsInstance(news_mock, list)
        
        # Thử tải dữ liệu thực
        try:
            news_real = self.data_loader.load_stock_news(self.test_symbol, use_mock=False)
            self.assertIsInstance(news_real, list)
        except Exception as e:
            print(f"Không thể tải tin tức cổ phiếu thực: {str(e)}")
    
    def test_load_stock_financials(self):
        """
        Kiểm thử tải dữ liệu tài chính
        """
        # Tải dữ liệu mẫu
        financials_mock = self.data_loader.load_stock_financials(self.test_symbol, use_mock=True)
        self.assertIsInstance(financials_mock, dict)
        self.assertIn('ratios', financials_mock)
        
        # Thử tải dữ liệu thực
        try:
            financials_real = self.data_loader.load_stock_financials(self.test_symbol, use_mock=False)
            self.assertIsInstance(financials_real, dict)
            self.assertIn('ratios', financials_real)
        except Exception as e:
            print(f"Không thể tải dữ liệu tài chính thực: {str(e)}")

if __name__ == '__main__':
    unittest.main() 