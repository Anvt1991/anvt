#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ví dụ sử dụng module DataLoader
"""

from core.data.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Khởi tạo DataLoader
    loader = DataLoader()
    
    # 1. Tải dữ liệu lịch sử cổ phiếu FPT
    print("1. Tải dữ liệu lịch sử cổ phiếu FPT:")
    df = loader.load_stock_data('FPT', timeframe='1D', num_candles=100, use_mock=True)
    print(df.head())
    print(f"Tổng số nến: {len(df)}")
    print("-" * 50)
    
    # 2. Tải dữ liệu thời gian thực
    print("2. Tải dữ liệu thời gian thực:")
    quotes = loader.load_realtime_quotes(['FPT', 'VNM', 'MWG'], use_mock=True)
    for symbol, data in quotes.items():
        print(f"{symbol}: {data['price']} ({data['percent_change']}%) - Khối lượng: {data['volume']}")
    print("-" * 50)
    
    # 3. Tải tổng quan thị trường
    print("3. Tải tổng quan thị trường:")
    market = loader.load_market_overview(use_mock=True)
    print("Chỉ số thị trường:")
    for index_name, index_data in market['indices'].items():
        print(f"  {index_name}: {index_data['value']} ({index_data['change']}%)")
    print("-" * 50)
    
    # 4. Tải tin tức
    print("4. Tải tin tức của FPT:")
    news = loader.load_stock_news('FPT', limit=3, use_mock=True)
    for item in news:
        print(f"  {item['title']} ({item['date']} - {item['source']})")
    print("-" * 50)
    
    # 5. Tải dữ liệu tài chính
    print("5. Tải dữ liệu tài chính của FPT:")
    financials = loader.load_stock_financials('FPT', use_mock=True)
    print("Các chỉ số tài chính:")
    for key, value in financials['ratios'].items():
        print(f"  {key}: {value}")
    print("-" * 50)
    
    # 6. Vẽ biểu đồ giá
    try:
        print("6. Vẽ biểu đồ giá cổ phiếu FPT:")
        plt.figure(figsize=(12, 6))
        plt.plot(df['date'], df['close'])
        plt.title('Biểu đồ giá cổ phiếu FPT')
        plt.xlabel('Ngày')
        plt.ylabel('Giá (VND)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('fpt_price_chart.png')
        print("  Đã vẽ biểu đồ và lưu vào file 'fpt_price_chart.png'")
    except Exception as e:
        print(f"  Lỗi khi vẽ biểu đồ: {str(e)}")
    
    print("\nĐã hoàn thành các ví dụ!")

if __name__ == "__main__":
    main() 