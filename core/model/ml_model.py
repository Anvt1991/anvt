#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Model for stock market prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import json
import datetime
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.impute import SimpleImputer
import time
import math
import pickle
import sklearn
from prophet import Prophet
import traceback

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictor:
    """
    ML Model for stock market prediction (Predictor only)
    """
    
    def __init__(self, model_type: str = "classifier", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML model
        
        Args:
            model_type: Type of model - 'classifier' for binary prediction or 'regressor' for continuous
            config: Configuration dictionary
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = None
        self.feature_names = []
        
        # Default configuration
        self.default_config = {
            "model_name": "random_forest",  # random_forest, xgboost, lightgbm
            "target_column": "target",
            "prediction_horizon": 5,  # Days to predict ahead
            "train_test_split": 0.8,  # Percentage of data to use for training
            "cv_folds": 5,  # Number of folds for cross-validation
            "random_state": 42,
            "hyperparameters": {
                "random_forest": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt"  # Sẽ được chuyển đổi để tương thích với cả phiên bản mới và cũ
                },
                "xgboost": {
                    "n_estimators": 100,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "gamma": 0
                },
                "prophet": {
                    "seasonality_mode": "additive",
                    "yearly_seasonality": True,
                    "weekly_seasonality": True,
                    "daily_seasonality": False
                }
            }
        }
        
        # Update config with default values for missing keys
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"MLPredictor initialized with model_type={model_type}, model_name={self.config.get('model_name', 'random_forest')}")
    
    def _convert_max_features(self, max_features_value):
        """
        Convert max_features value to appropriate type based on scikit-learn version
        
        Args:
            max_features_value: The max_features value to convert
            
        Returns:
            Converted max_features value
        """
        # Không cần chuyển đổi nữa, giữ nguyên chuỗi để tương thích với mọi phiên bản
        # Scikit-learn mới vẫn chấp nhận các chuỗi như 'sqrt', 'log2'
        logger.info(f"Sử dụng max_features với giá trị nguyên thủy: {max_features_value}")
        return max_features_value
    
    def _initialize_model(self):
        """Initialize model based on configuration"""
        model_name = self.config.get("model_name", "random_forest")
        # Save original model_type (classifier/regressor)
        original_model_type = self.model_type
        
        if model_name == "random_forest":
            hyperparams = self.config.get("hyperparameters", {}).get("random_forest", {})
            
            # Get max_features value
            max_features_value = hyperparams.get("max_features", "sqrt")
            
            # Convert max_features value appropriately
            converted_max_features = self._convert_max_features(max_features_value)
            
            # Log the conversion
            logger.info(f"Using max_features value: {max_features_value} (converted to: {converted_max_features})")
            
            # Other parameters
            n_estimators = hyperparams.get("n_estimators", 100)
            max_depth = hyperparams.get("max_depth", None)
            min_samples_split = hyperparams.get("min_samples_split", 2)
            min_samples_leaf = hyperparams.get("min_samples_leaf", 1)
            
            # Create the model
            if original_model_type == "classifier":
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=converted_max_features,
                    random_state=42
                )
            else:
                self.model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=converted_max_features,
                    random_state=42
                )
        elif model_name == "xgboost":
            hyperparams = self.config.get("hyperparameters", {}).get("xgboost", {})
            if original_model_type == "classifier":
                self.model = xgb.XGBClassifier(
                    n_estimators=hyperparams.get("n_estimators", 100),
                    max_depth=hyperparams.get("max_depth", 6),
                    learning_rate=hyperparams.get("learning_rate", 0.1),
                    subsample=hyperparams.get("subsample", 0.8),
                    colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
                    gamma=hyperparams.get("gamma", 0),
                    random_state=self.config["random_state"],
                    n_jobs=-1
                )
            else:
                self.model = xgb.XGBRegressor(
                    n_estimators=hyperparams.get("n_estimators", 100),
                    max_depth=hyperparams.get("max_depth", 6),
                    learning_rate=hyperparams.get("learning_rate", 0.1),
                    subsample=hyperparams.get("subsample", 0.8),
                    colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
                    gamma=hyperparams.get("gamma", 0),
                    random_state=self.config["random_state"],
                    n_jobs=-1
                )
        elif model_name == "prophet":
            hyperparams = self.config.get("hyperparameters", {}).get("prophet", {})
            self.model = Prophet(**hyperparams)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
    
    def load_model(self, filepath: str):
        """
        Load a model from file
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            bool: True if the model was loaded successfully
        """
        try:
            logger.info(f"Loading model from {filepath}")
            model_data = joblib.load(filepath)
            if isinstance(model_data, dict) and 'model' in model_data:
                self.model = model_data['model']
                if 'metadata' in model_data and isinstance(model_data['metadata'], dict):
                    metadata = model_data['metadata']
                    if 'feature_names' in metadata:
                        self.feature_names = metadata['feature_names']
                        logger.info(f"Loaded {len(self.feature_names)} feature names from model metadata")
                    else:
                        logger.warning("No feature_names found in model metadata!")
                        raise ValueError("Model metadata missing feature_names, which is required")
                    if 'config' in metadata:
                        self.config = metadata['config']
                else:
                    if 'feature_names' in model_data:
                        self.feature_names = model_data['feature_names']
                        logger.info(f"Loaded {len(self.feature_names)} feature names from legacy format")
                    else:
                        logger.warning("No feature_names found in legacy model format!")
                        raise ValueError("Model missing feature_names, which is required")
                    if 'config' in model_data:
                        self.config = model_data.get('config', {})
                    if 'model_type' in model_data:
                        self.model_type = model_data['model_type']
            else:
                self.model = model_data
                logger.warning("Loaded direct model object without metadata")
                if not hasattr(self, 'feature_names') or not self.feature_names:
                    raise ValueError("Loaded model has no feature_names, which is required")
            # Check feature_names validity
            expected_feature_count = None
            if hasattr(self.model, 'n_features_in_'):
                expected_feature_count = self.model.n_features_in_
                logger.info(f"Model mong đợi {expected_feature_count} features (từ n_features_in_)")
            elif hasattr(self.model, 'feature_importances_') and hasattr(self.model, 'n_features_'):
                expected_feature_count = self.model.n_features_
                logger.info(f"Model mong đợi {expected_feature_count} features (từ n_features_)")
            elif hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
                if hasattr(self.model.estimators_[0], 'n_features_'):
                    expected_feature_count = self.model.estimators_[0].n_features_
                    logger.info(f"Model mong đợi {expected_feature_count} features (từ estimators[0].n_features_)")
                elif hasattr(self.model.estimators_[0], 'n_features_in_'):
                    expected_feature_count = self.model.estimators_[0].n_features_in_
                    logger.info(f"Model mong đợi {expected_feature_count} features (từ estimators[0].n_features_in_)")
            elif hasattr(self.model, 'coef_') and len(getattr(self.model, 'coef_', [])) > 0:
                coef = self.model.coef_
                if isinstance(coef, np.ndarray):
                    if coef.ndim > 1:
                        expected_feature_count = coef.shape[1]
                    else:
                        expected_feature_count = coef.shape[0]
                    logger.info(f"Model mong đợi {expected_feature_count} features (từ coef_ shape)")
            # Warn if feature_names is missing or mismatched
            if expected_feature_count is not None:
                if not self.feature_names:
                    logger.warning(f"Model expects {expected_feature_count} features but feature_names is missing!")
                elif len(self.feature_names) != expected_feature_count:
                    logger.warning(f"CẢNH BÁO: Model mong đợi {expected_feature_count} features, nhưng feature_names có {len(self.feature_names)}!")
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
            
    def set_feature_names(self, feature_names: List[str]):
        """
        Set feature names for the model
        
        Args:
            feature_names: List of feature names
        """
        if not feature_names:
            logger.warning("Attempting to set empty feature_names")
            return
            
        self.feature_names = list(feature_names)
        logger.info(f"Set {len(self.feature_names)} feature names")

    def _prepare_features(self, input_data: Dict[str, Any]) -> list:
        """
        Prepare features for prediction from input data
        
        Args:
            input_data: Dictionary containing feature values
            
        Returns:
            List of feature values in the correct order for the model
        """
        try:
            if not self.feature_names:
                logger.error("Feature names not available")
                return None
                
            # Create a feature vector with the same order as training data
            features = []
            missing_features = []
            
            for feature in self.feature_names:
                if feature in input_data:
                    features.append(input_data[feature])
                else:
                    missing_features.append(feature)
                    features.append(0)  # Default value for missing features
            
            if missing_features:
                logger.warning(f"Missing features in input data: {missing_features}")
                
            return features
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction using the trained model
        
        Args:
            input_data: Dictionary containing feature values
            
        Returns:
            Dictionary with prediction results or error information
        """
        response = {
            "success": False,
            "error": None,
            "prediction": None,
            "probability": None,
            "features_used": None
        }
        
        try:
            # Check if model is initialized
            if not self.model:
                response["error"] = "Model not initialized"
                logger.error("Cannot predict: Model not initialized")
                return response
            # Check if feature_names is available
            if not self.feature_names:
                response["error"] = "Feature names list is empty - cannot predict"
                logger.error("Feature names list is empty - cannot predict")
                return response
                
            # Check if model is trained - multiple ways to check based on model type
            model_is_trained = False
            
            # For classifiers
            if hasattr(self.model, 'classes_'):
                model_is_trained = True
            # For tree-based models
            elif hasattr(self.model, 'estimators_') and len(self.model.estimators_) > 0:
                model_is_trained = True
            # For regressors that don't have classes_
            elif hasattr(self.model, 'feature_importances_'):
                model_is_trained = True
            # For sklearn models like Linear Regression
            elif hasattr(self.model, 'coef_'):
                model_is_trained = True
            # For other model types
            elif hasattr(self.model, 'predict'):
                # Try a simple predict to see if it works
                try:
                    # Create a dummy feature array matching our feature names
                    dummy_features = np.zeros((1, len(self.feature_names)))
                    self.model.predict(dummy_features)
                    model_is_trained = True
                except Exception as e:
                    logger.error(f"Error testing model with dummy prediction: {str(e)}")
                    model_is_trained = False
            
            if not model_is_trained:
                response["error"] = "Model not trained"
                logger.error("Cannot predict: Model not trained")
                return response
                
            # Prepare features
            features = self._prepare_features(input_data)
            if features is None:
                response["error"] = "Failed to prepare features"
                return response
                
            # Make prediction
            features_array = np.array([features])
            prediction = self.model.predict(features_array)[0]
            
            # Get probabilities if available
            probability = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_array)[0]
                max_prob_index = np.argmax(probabilities)
                probability = float(probabilities[max_prob_index])
            
            # Populate response
            response["success"] = True
            response["prediction"] = prediction
            response["probability"] = probability
            response["features_used"] = self.feature_names
            
            if self.config.get("model_name") == "prophet":
                try:
                    # input_data: dict with 'ds' (date)
                    df = pd.DataFrame([input_data])
                    forecast = self.model.predict(df)
                    response["yhat_lower"] = forecast['yhat_lower'].iloc[0]
                    response["yhat_upper"] = forecast['yhat_upper'].iloc[0]
                    response["ds"] = forecast['ds'].iloc[0]
                    response["model_type"] = "Prophet"
                    return response
                except Exception as e:
                    response["error"] = str(e)
                    return response
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            response["error"] = f"Prediction failed: {str(e)}"
            return response
    
    def batch_predict(self, input_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple rows of data
        """
        try:
            if not self.model:
                logger.warning("Cannot predict: Model not initialized. Sẽ train lại model nếu có thể.")
                return []
            if not self.feature_names:
                logger.warning("Feature names list is empty - cannot predict. Sẽ train lại model nếu có thể.")
                return []
            # Tự động khớp feature: điền 0 cho feature thiếu, chỉ lấy đúng các feature model cần
            feature_df = pd.DataFrame(index=input_df.index)
            for f in self.feature_names:
                if f not in input_df.columns:
                    feature_df[f] = 0
                else:
                    feature_df[f] = input_df[f]
            
            # Fill NaN/inf
            X = feature_df.fillna(0).replace([np.inf, -np.inf], 0)
            if X.empty:
                logger.warning("Feature matrix is empty after alignment.")
                return []
                
            # Kiểm tra thêm để tránh lỗi
            if len(X) == 0:
                logger.warning("Empty feature matrix after processing")
                return []
                
            try:
                predictions = self.model.predict(X)
                probabilities = None
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X)
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}. Falling back to default values.")
                return []
                
            results = []
            for i in range(len(predictions)):
                prob = None
                if probabilities is not None:
                    prob_row = probabilities[i]
                    max_prob_index = np.argmax(prob_row)
                    prob = float(prob_row[max_prob_index])
                results.append({
                    "success": True,
                    "prediction": predictions[i],
                    "probability": prob,
                    "features_used": self.feature_names
                })
            return results
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return []
    
    def evaluate(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            if self.model is None:
                logger.error("Model not trained yet")
                return {}
            
            # Make predictions
            if self.model_type == "classifier":
                y_pred = self.model.predict(X)
                y_prob = self.model.predict_proba(X)[:, 1]
                
                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(y, y_pred),
                    "precision": precision_score(y, y_pred),
                    "recall": recall_score(y, y_pred),
                    "f1": f1_score(y, y_pred),
                    "positive_rate": y_pred.mean()  # Percentage of positive predictions
                }
            else:
                y_pred = self.model.predict(X)
                
                # Calculate metrics
                metrics = {
                    "r2": r2_score(y, y_pred),
                    "mse": mean_squared_error(y, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                    "mae": np.mean(np.abs(y - y_pred)),
                    "mean_pred": y_pred.mean()
                }
            
            logger.info(f"Model evaluation: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance
        
        Returns:
            DataFrame with feature importance
        """
        try:
            if self.model is None or not hasattr(self.model, 'feature_importances_'):
                logger.error("Model not trained or doesn't support feature importances")
                return pd.DataFrame()
            
            # Get feature importance
            importances = self.model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None) -> None:
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save the plot
        """
        try:
            importance_df = self.feature_importance()
            
            if importance_df.empty:
                return
            
            # Get top N features
            top_features = importance_df.head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 8))
            bars = plt.barh(top_features['feature'], top_features['importance'])
            
            # Add labels
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Top {top_n} Feature Importance')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
    
    def predict_ml_score(self, symbol: str, data: pd.DataFrame, 
                        tech_analysis: Optional[Dict[str, Any]] = None,
                        news_sentiment: Optional[Dict[str, Any]] = None,
                        market_condition: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Dự đoán ML score cho các khung thời gian khác nhau.
        """
        try:
            logger.info(f"====== Bắt đầu tính điểm ML cho {symbol} ======")
            # Check if data is valid
            if data is None or not isinstance(data, pd.DataFrame):
                logger.error(f"Invalid data for ML prediction for {symbol}")
                return {"error": "Invalid data", "short_term": 0, "medium_term": 0, "long_term": 0}
                
            if data.empty:
                logger.error(f"Empty DataFrame for ML prediction for {symbol}")
                return {"error": "Empty DataFrame", "short_term": 0, "medium_term": 0, "long_term": 0}
                
            # Chuẩn bị features từ DataFrame
            features = data.copy()
            try:
                # Chuyển đổi dữ liệu sang numeric nếu cần
                for col in features.columns:
                    if features[col].dtype == 'object':
                        try:
                            features[col] = pd.to_numeric(features[col], errors='coerce')
                        except:
                            features[col] = np.nan
                
                # Xử lý giá trị NaN/Inf
                features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
            except Exception as prep_error:
                logger.error(f"Error preparing data for {symbol}: {prep_error}")
                return {"error": f"Data preparation error: {str(prep_error)}", "short_term": 0, "medium_term": 0, "long_term": 0}
                
            # Kiểm tra model
            model_path = os.path.join("models", f"{symbol}_model.joblib")
            if not self.model:
                try:
                    model_loaded = self.load_model(model_path)
                    if not model_loaded:
                        logger.warning(f"Failed to load model for {symbol}, sẽ train lại model.")
                        return self._fallback_train_and_predict(symbol, features)
                except Exception as e:
                    logger.warning(f"Error loading model from {model_path}: {str(e)}. Sẽ train lại model.")
                    return self._fallback_train_and_predict(symbol, features)
                    
            if not self.feature_names:
                logger.warning(f"Model for {symbol} has no feature names, sẽ train lại model.")
                return self._fallback_train_and_predict(symbol, features)
                
            # Tự động khớp feature: điền 0 cho feature thiếu, chỉ lấy đúng thứ tự model cần
            try:
                aligned_features = pd.DataFrame(index=features.index)
                for f in self.feature_names:
                    if f not in features.columns:
                        aligned_features[f] = 0
                    else:
                        aligned_features[f] = features[f]
                features = aligned_features
            except Exception as align_error:
                logger.error(f"Error aligning features for {symbol}: {align_error}")
                return {"error": f"Feature alignment error: {str(align_error)}", "short_term": 0, "medium_term": 0, "long_term": 0}
                
            # Dự đoán
            if len(features) == 0:
                logger.error(f"No data after feature preparation for {symbol}")
                return {"error": "No data after feature preparation", "short_term": 0, "medium_term": 0, "long_term": 0}
                
            try:
                if len(features) == 1:
                    feature_dict = features.iloc[0].to_dict()
                    prediction_result = self.predict(feature_dict)
                    prediction = prediction_result.get("prediction", 0)
                    probability = prediction_result.get("probability", 0)
                else:
                    batch_results = self.batch_predict(features)
                    predictions = [r.get("prediction", 0) for r in batch_results if r.get("success", False)]
                    probabilities = [r.get("probability", 0) for r in batch_results if r.get("success", False)]
                    
                    if not predictions:
                        logger.warning(f"No valid predictions for {symbol}")
                        prediction = 0
                        probability = 0
                    else:
                        prediction = np.mean(predictions)
                        probability = np.mean(probabilities) if probabilities else 0
            except Exception as pred_error:
                logger.error(f"Error during prediction for {symbol}: {pred_error}")
                return {"error": f"Prediction error: {str(pred_error)}", "short_term": 0, "medium_term": 0, "long_term": 0}
                
            # Tính score
            try:
                if self.model_type == 'classifier':
                    # Đảm bảo probability trong khoảng [0, 1]
                    probability = max(0, min(1, probability))
                    score = (probability - 0.5) * 2
                else:
                    score = prediction / 10
                    score = max(min(score, 1), -1)
                    
                # Đảm bảo score không có giá trị NaN hoặc inf
                if np.isnan(score) or np.isinf(score):
                    logger.warning(f"Invalid score value for {symbol}: {score}, using 0")
                    score = 0
            except Exception as score_error:
                logger.error(f"Error calculating score for {symbol}: {score_error}")
                return {"error": f"Score calculation error: {str(score_error)}", "short_term": 0, "medium_term": 0, "long_term": 0}
                
            # Áp dụng trọng số
            weights = {"short_term": 1.0, "medium_term": 0.8, "long_term": 0.6}
            result = {
                "short_term": score * weights["short_term"],
                "medium_term": score * weights["medium_term"],
                "long_term": score * weights["long_term"]
            }
            
            logger.info(f"ML score for {symbol}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ML prediction for {symbol}: {str(e)}. Sẽ train lại model nếu có thể.")
            try:
                return self._fallback_train_and_predict(symbol, data)
            except Exception as fallback_error:
                logger.critical(f"Fallback also failed for {symbol}: {str(fallback_error)}")
                return {"error": f"Complete failure: {str(e)}, fallback: {str(fallback_error)}", 
                        "short_term": 0, "medium_term": 0, "long_term": 0}

    def _fallback_train_and_predict(self, symbol: str, features: pd.DataFrame) -> Dict[str, float]:
        """
        Fallback: train lại model với dữ liệu hiện tại nếu không load được model hoặc feature_names mismatch.
        """
        try:
            logger.info(f"Fallback: training model for {symbol} do không load được model hoặc feature mismatch.")
            # Tạo target tạm thời (binary tăng/giảm)
            if 'close' in features.columns:
                features['future_close'] = features['close'].shift(-5)
                features['price_change'] = features['future_close'] - features['close']
                features['price_pct_change'] = features['price_change'] / features['close'] * 100
                features['target'] = (features['price_pct_change'] > 0).astype(int)
                features = features.dropna(subset=['target'])
                X = features.drop(columns=['target', 'future_close', 'price_change', 'price_pct_change'], errors='ignore')
                y = features['target']
                if len(X) > 10:
                    self.train(X, y)
                    self.save_model(os.path.join("models", f"{symbol}_model.joblib"))
                    # Predict lại
                    return self.predict_ml_score(symbol, X.tail(1))
            logger.error("Fallback train không thành công do thiếu dữ liệu hoặc cột close.")
            return {"error": "Fallback train failed", "short_term": 0, "medium_term": 0, "long_term": 0}
        except Exception as e:
            logger.error(f"Fallback train/predict error: {str(e)}")
            return {"error": f"Fallback train error: {str(e)}", "short_term": 0, "medium_term": 0, "long_term": 0}
    
    def save_model(self, filepath: str, extra_metadata: Optional[Dict[str, Any]] = None):
        """
        Save the model, feature_names, config, and extra metadata to file.
        """
        try:
            if not self.feature_names or len(self.feature_names) == 0:
                logger.error("Không thể lưu model vì feature_names rỗng! Model sẽ không được lưu.")
                return False
            metadata = {
                "feature_names": self.feature_names,
                "config": self.config,
                "model_type": self.model_type,
            }
            if extra_metadata:
                metadata.update(extra_metadata)
            joblib.dump({
                "model": self.model,
                "metadata": metadata
            }, filepath)
            logger.info(f"Model saved to {filepath} with {len(self.feature_names)} features")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def train(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        """
        Train the model directly
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dictionary with training results
        """
        try:
            if self.model is None:
                self._initialize_model()
            
            if X.empty or len(X) < 10:
                return {"error": "Not enough data for training"}
            
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Split data if needed
            train_size = int(len(X) * self.config.get("train_test_split", 0.8))
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size] if isinstance(y, pd.Series) else y[:train_size], y.iloc[train_size:] if isinstance(y, pd.Series) else y[train_size:]
            
            # Train the model
            logger.info(f"Training model with {len(X_train)} samples and {len(self.feature_names)} features")
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            metrics = self.evaluate(X_test, y_test)
            
            return {
                "success": True,
                "metrics": metrics,
                "feature_names": self.feature_names
            }
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {"error": f"Model training failed: {str(e)}"} 