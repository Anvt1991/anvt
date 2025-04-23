import logging
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from datetime import datetime, timedelta

from app.utils.helpers import run_in_thread, filter_trading_days
from app.utils.config import DEFAULT_MARKET_SYMBOLS

logger = logging.getLogger(__name__)

def prepare_data_for_prophet(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for Prophet forecasting model."""
    if df is None or df.empty or 'close' not in df.columns:
        return None
        
    # Prophet requires columns named 'ds' and 'y'
    prophet_df = pd.DataFrame()
    prophet_df['ds'] = df.index
    prophet_df['y'] = df['close']
    
    # Filter trading days (no weekends)
    prophet_df = filter_trading_days(prophet_df)
    
    return prophet_df

def get_vietnam_holidays(years) -> pd.DataFrame:
    """Get Vietnamese holidays for Prophet model."""
    holidays = []
    
    for year in years:
        # Tet Holiday (varies each year based on lunar calendar, approximate dates)
        tet_start = datetime(year, 1, 20)  # Approximate
        for i in range(7):
            holidays.append({'holiday': 'Tet_Holiday', 'ds': tet_start + timedelta(days=i), 'lower_window': 0, 'upper_window': 0})
        
        # Independence Day
        holidays.append({'holiday': 'Independence_Day', 'ds': datetime(year, 9, 2), 'lower_window': 0, 'upper_window': 0})
        
        # Labor Day
        holidays.append({'holiday': 'Labor_Day', 'ds': datetime(year, 5, 1), 'lower_window': 0, 'upper_window': 0})
        
        # Reunification Day
        holidays.append({'holiday': 'Reunification_Day', 'ds': datetime(year, 4, 30), 'lower_window': 0, 'upper_window': 0})
    
    return pd.DataFrame(holidays)

def forecast_with_prophet(df: pd.DataFrame, periods: int = 7) -> (pd.DataFrame, Prophet):
    """Use Prophet to forecast future prices."""
    # Prepare data
    prophet_df = prepare_data_for_prophet(df)
    if prophet_df is None or prophet_df.empty:
        return None, None
    
    # Define model parameters
    model = Prophet(
        changepoint_prior_scale=0.05,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    # Add holidays
    years = list(range(df.index.min().year, df.index.max().year + 2))
    holidays_df = get_vietnam_holidays(years)
    model.add_country_holidays(country_name='VN')
    model.add_holidays(holidays_df)
    
    # Fit model
    model.fit(prophet_df)
    
    # Make forecast
    future = model.make_future_dataframe(periods=periods, freq='D')
    future = filter_trading_days(future)  # Remove weekends
    
    forecast = model.predict(future)
    
    return forecast, model

def evaluate_prophet_performance(df: pd.DataFrame, forecast: pd.DataFrame) -> float:
    """Evaluate Prophet model performance using MAPE."""
    if df is None or df.empty or forecast is None or forecast.empty:
        return None
    
    # Merge actual and predicted data
    evaluation_df = pd.DataFrame()
    evaluation_df['ds'] = df.index
    evaluation_df['y_true'] = df['close']
    
    # Join with forecast
    evaluation_df = evaluation_df.merge(forecast[['ds', 'yhat']], on='ds', how='left')
    
    # Calculate MAPE
    valid_rows = ~evaluation_df['yhat'].isna()
    if valid_rows.sum() == 0:
        return None
        
    evaluation_df = evaluation_df[valid_rows]
    mape = mean_absolute_percentage_error(evaluation_df['y_true'], evaluation_df['yhat'])
    
    return mape

def predict_xgboost_signal(df: pd.DataFrame, features: list) -> (int, float):
    """Predict buy/sell signal using trained XGBoost model."""
    # This is a placeholder implementation
    # In real implementation, you would:
    # 1. Load a trained model
    # 2. Extract features from the latest data point
    # 3. Make a prediction
    # 4. Return the signal (1 for buy, -1 for sell, 0 for hold) and confidence
    
    # For now, return a random signal as placeholder
    import random
    signal = random.choice([-1, 0, 1])
    confidence = random.uniform(0.5, 0.9)
    
    return signal, confidence

def train_prophet_model(df: pd.DataFrame) -> (Prophet, float):
    """Train a Prophet model for time series forecasting."""
    try:
        # Prepare data
        prophet_df = prepare_data_for_prophet(df)
        if prophet_df is None or prophet_df.empty:
            logger.warning("Cannot prepare data for Prophet model")
            return None, None
        
        # Split data for evaluation
        train_size = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:train_size]
        test_df = prophet_df.iloc[train_size:]
        
        # Train Prophet model
        model = Prophet(
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        
        # Add Vietnam holidays
        years = list(range(df.index.min().year, df.index.max().year + 2))
        holidays_df = get_vietnam_holidays(years)
        model.add_holidays(holidays_df)
        
        model.fit(train_df)
        
        # Make predictions on test set
        future = model.make_future_dataframe(periods=len(test_df), freq='D')
        future = filter_trading_days(future)
        forecast = model.predict(future)
        
        # Evaluate model
        performance = evaluate_prophet_performance(
            pd.DataFrame({'close': test_df['y']}, index=test_df['ds']), 
            forecast
        )
        
        if performance is not None:
            logger.info(f"Prophet model performance (MAPE): {performance:.4f}")
        
        # Retrain on full dataset for final model
        full_model = Prophet(
            changepoint_prior_scale=0.05,
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        full_model.add_holidays(holidays_df)
        full_model.fit(prophet_df)
        
        return full_model, performance
        
    except Exception as e:
        logger.error(f"Error training Prophet model: {str(e)}")
        return None, None

def train_xgboost_model(df: pd.DataFrame, features: list) -> (xgb.XGBClassifier, float):
    """Train an XGBoost model for signal classification."""
    try:
        if df is None or df.empty:
            logger.warning("Empty DataFrame for XGBoost training")
            return None, None
            
        # Check if all required features exist in dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features for XGBoost training: {missing_features}")
            return None, None
        
        # Create target variable (simple example: 1 if price goes up next day, 0 otherwise)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 100:  # Not enough data for meaningful training
            logger.warning("Not enough data for XGBoost training")
            return None, None
        
        # Prepare features and target
        X = df[features]
        y = df['target']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"XGBoost model accuracy: {accuracy:.4f}")
        
        return model, accuracy
        
    except Exception as e:
        logger.error(f"Error training XGBoost model: {str(e)}")
        return None, None

async def get_training_symbols() -> list:
    """Get list of symbols for model training."""
    # In a real implementation, this could be fetched from a database or external source
    # For now, use a predefined list from config
    return DEFAULT_MARKET_SYMBOLS

async def train_models_for_symbol(symbol: str, data_loader, model_db_manager):
    """Train and store models for a specific symbol."""
    try:
        # Load data
        logger.info(f"Training models for {symbol}")
        
        # Load daily data for the symbol
        df_daily, message = await data_loader.load_data(symbol, "1D", 365 * 2)  # 2 years of data
        
        if df_daily is None or df_daily.empty:
            logger.warning(f"No data available for {symbol}, skipping training")
            return False
            
        # Train Prophet model
        prophet_model, prophet_performance = train_prophet_model(df_daily)
        
        if prophet_model:
            # Store trained model
            await model_db_manager.store_trained_model(
                symbol=symbol,
                model_type="prophet",
                model=prophet_model,
                performance=prophet_performance
            )
            logger.info(f"Prophet model for {symbol} trained and stored successfully")
        
        # Calculate technical indicators for XGBoost
        from app.services.technical_analysis import TechnicalAnalyzer
        analyzer = TechnicalAnalyzer()
        df_with_indicators = analyzer.calculate_indicators(df_daily)
        
        # Define features for XGBoost
        features = [
            'rsi_14', 'macd', 'macd_signal', 'macd_hist', 
            'stoch_k', 'stoch_d', 'adx',
            'sma_20', 'sma_50', 'ema_20', 'ema_50'
        ]
        
        # Train XGBoost model
        xgb_model, xgb_performance = train_xgboost_model(df_with_indicators, features)
        
        if xgb_model:
            # Store trained model
            await model_db_manager.store_trained_model(
                symbol=symbol,
                model_type="xgboost",
                model=xgb_model,
                performance=xgb_performance
            )
            logger.info(f"XGBoost model for {symbol} trained and stored successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Error training models for {symbol}: {str(e)}")
        return False

async def auto_train_models(data_loader, model_db_manager):
    """Automatically train models for all symbols."""
    symbols = await get_training_symbols()
    logger.info(f"Starting automatic model training for {len(symbols)} symbols")
    
    success_count = 0
    for symbol in symbols:
        try:
            success = await train_models_for_symbol(symbol, data_loader, model_db_manager)
            if success:
                success_count += 1
        except Exception as e:
            logger.error(f"Error in auto_train_models for {symbol}: {str(e)}")
    
    logger.info(f"Completed model training. Successful: {success_count}/{len(symbols)}")
    return success_count 