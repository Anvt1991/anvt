import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Quản lý việc load, validate, và truy xuất config cho toàn bộ hệ thống AI.
    """
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        default_config = {
            "use_ml_model": True,
            "use_backtester": True,
            "model_dir": "models",
            "results_dir": "results",
            "feature_engineering": {},
            "ml_model": {
                "model_type": "classifier",
                "model_name": "random_forest",
                "hyperparameters": {
                    "random_forest": {
                        "n_estimators": 100,
                        "max_depth": 10,
                        "min_samples_split": 5,
                        "min_samples_leaf": 2,
                        "max_features": "sqrt"
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
            },
            "backtester": {},
            "predictor": {
                "weights": {
                    "technical": 0.30,
                    "sentiment": 0.15,
                    "ai": 0.45,
                    "market": 0.10
                }
            }
        }
        if config_path is None:
            config_path = "config/default_config.json"
        config = default_config
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    self._deep_update(config, file_config)
                logger.info(f"Configuration loaded from {config_path}")
            else:
                logger.warning(f"Config file {config_path} not found, using defaults")
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
        return config

    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v
        return d

    def get(self, key: str, default=None):
        return self.config.get(key, default) 