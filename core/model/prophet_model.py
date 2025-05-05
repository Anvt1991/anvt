import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional, Dict, Any
from prophet import Prophet
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ProphetModel:
    def __init__(self, model_dir: str = "models/prophet"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.model = None
        self.model_path = None

    def _get_model_path(self, symbol: str) -> str:
        return os.path.join(self.model_dir, f"{symbol}_prophet_model.joblib")

    def train(self, symbol: str, df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
        """
        Train Prophet model and return forecast DataFrame
        """
        logger.info(f"Training Prophet model for {symbol}")
        # Chuẩn hóa dữ liệu
        prophet_df = df.copy()
        if 'date' in prophet_df.columns:
            prophet_df['ds'] = pd.to_datetime(prophet_df['date'])
        elif prophet_df.index.name in ['date', 'datetime']:
            prophet_df['ds'] = pd.to_datetime(prophet_df.index)
        else:
            prophet_df['ds'] = pd.to_datetime(prophet_df.index)
        # Remove timezone info if present
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
        prophet_df['y'] = prophet_df['close']
        prophet_df = prophet_df[['ds', 'y']].dropna()
        # Train
        model = Prophet(daily_seasonality=True, yearly_seasonality=True, weekly_seasonality=True)
        model.fit(prophet_df)
        self.model = model
        self.model_path = self._get_model_path(symbol)
        joblib.dump(model, self.model_path)
        logger.info(f"Saved Prophet model for {symbol} to {self.model_path}")
        # Dự báo
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        return forecast_df

    def load_model(self, symbol: str) -> Optional[Prophet]:
        path = self._get_model_path(symbol)
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.model_path = path
            logger.info(f"Loaded Prophet model for {symbol} from {path}")
            return self.model
        logger.warning(f"No Prophet model found for {symbol} at {path}")
        return None

    def predict(self, symbol: str, df: pd.DataFrame, periods: int = 14) -> pd.DataFrame:
        """
        Dự báo với Prophet, tự động train nếu chưa có model
        """
        if not self.load_model(symbol):
            logger.info(f"No existing Prophet model for {symbol}, training new model...")
            return self.train(symbol, df, periods)
        # Chuẩn hóa dữ liệu
        prophet_df = df.copy()
        if 'date' in prophet_df.columns:
            prophet_df['ds'] = pd.to_datetime(prophet_df['date'])
        elif prophet_df.index.name in ['date', 'datetime']:
            prophet_df['ds'] = pd.to_datetime(prophet_df.index)
        else:
            prophet_df['ds'] = pd.to_datetime(prophet_df.index)
        # Remove timezone info if present
        prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
        prophet_df['y'] = prophet_df['close']
        prophet_df = prophet_df[['ds', 'y']].dropna()
        # Dự báo
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        return forecast_df

    def forecast_pipeline_result(self, symbol: str, df: pd.DataFrame, periods: int = 14) -> Dict[str, Any]:
        """
        Chuẩn hóa kết quả dự báo Prophet cho pipeline
        """
        forecast_df = self.predict(symbol, df, periods)
        return {
            "forecast_df": forecast_df,
            "model_type": "prophet",
            "version": "1.0",
            "timestamp": datetime.now().isoformat(),
            "metadata": {"symbol": symbol, "periods": periods}
        } 