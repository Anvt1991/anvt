import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from core.model.ml_model import MLPredictor
from core.data.data_validator import DataValidator
from datetime import datetime, timedelta
import hashlib
from core.model.feature_generator import FeatureGenerator

logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Quản lý toàn bộ vòng đời model: train, save, load, optimize, evaluate, feature selection.
    """
    def __init__(self, config: Dict[str, Any], feature_generator):
        self.config = config
        self.feature_generator = feature_generator
        self.model_dir = self.config.get("model_dir", "models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.ml_predictor = MLPredictor(
            model_type=self.config.get("ml_model", {}).get("model_type", "classifier"),
            config=self.config.get("ml_model", {})
        )
        self.feature_importance_scores = None

    def train_model(self, symbol: str, data: pd.DataFrame,
                   tech_analysis: Optional[Dict[str, Any]] = None,
                   news_sentiment: Optional[Dict[str, Any]] = None,
                   market_condition: Optional[Dict[str, Any]] = None,
                   from_optimization: bool = False) -> Dict[str, Any]:
        """
        Train ML model for a stock
        """
        try:
            if self.ml_predictor is None:
                return {"error": "ML predictor not initialized"}

            # Kiểm tra xem dữ liệu có được truyền từ optimization hay không
            if data is None:
                # Trường hợp được gọi từ phương thức optimize, cần có dữ liệu đã xử lý sẵn
                model_info_path = self._get_model_info_path(symbol)
                opt_result_path = os.path.join(model_info_path, "optimization_results.json")
                if os.path.exists(opt_result_path):
                    with open(opt_result_path, "r") as f:
                        opt_data = json.load(f)
                        best_model = opt_data.get("best_model")
                        best_params = opt_data.get("best_params", {})
                        feature_names_opt = opt_data.get("feature_names", None)
                        if best_model and best_params:
                            # Lưu model_type trước khi khởi tạo lại model
                            original_model_type = self.ml_predictor.model_type
                            self.ml_predictor.config["model_name"] = best_model
                            self.ml_predictor.config["hyperparameters"][best_model].update(best_params)
                            # Đảm bảo model_type không bị ghi đè
                            self.ml_predictor.model_type = original_model_type
                            self.ml_predictor._initialize_model()
                            model_path = self._get_model_path(symbol)
                            # Load lại feature_names từ model cũ nếu có
                            if os.path.exists(model_path):
                                self.ml_predictor.load_model(model_path)
                            # Nếu vẫn không có, thử load từ features_predict
                            if not self.ml_predictor.feature_names or len(self.ml_predictor.feature_names) == 0:
                                features_predict_path = os.path.join(self.model_dir, f"{symbol}_features_predict.parquet")
                                if os.path.exists(features_predict_path):
                                    features_df = pd.read_parquet(features_predict_path)
                                    self.ml_predictor.set_feature_names(list(features_df.columns))
                            # Nếu vẫn không có, thử lấy từ optimization_results.json
                            if (not self.ml_predictor.feature_names or len(self.ml_predictor.feature_names) == 0) and feature_names_opt:
                                self.ml_predictor.set_feature_names(feature_names_opt)
                            # Nếu vẫn không có, raise error
                            if not self.ml_predictor.feature_names or len(self.ml_predictor.feature_names) == 0:
                                logger.error("Không tìm thấy feature_names để lưu cùng model khi train từ optimization!")
                                raise ValueError("Không tìm thấy feature_names để lưu cùng model khi train từ optimization!")
                            self.ml_predictor.save_model(model_path)
                            logger.info(f"Model {best_model} đã được cập nhật với tham số tối ưu và lưu thành công")
                            return {
                                "success": True,
                                "model_name": best_model,
                                "model_path": model_path
                            }
                return {"error": "Không có dữ liệu để train model"}

            # 1. Tự động kiểm tra & làm sạch dữ liệu
            data = DataValidator.normalize_dataframe(data)
            data, _ = DataValidator.detect_and_handle_outliers(data)

            # 2. Feature engineering
            features = self.feature_generator.prepare_features(
                data, tech_analysis, news_sentiment, market_condition
            )
            if features.empty:
                return {"error": "Failed to prepare features for training"}
            data_with_target = self.prepare_target(features)
            if data_with_target.empty:
                return {"error": "Failed to prepare target for training"}
            target_column = self.ml_predictor.config["target_column"]
            X_full = data_with_target.drop(columns=[target_column, 'future_close'], errors='ignore')
            y_full = data_with_target[target_column]
            mask = ~(X_full.isnull().any(axis=1) | y_full.isnull())
            X_full = X_full[mask]
            y_full = y_full[mask]
            X_full = X_full.dropna()
            y_full = y_full.loc[X_full.index]
            if len(X_full) < 40:
                return {"error": "Not enough data for train/test split"}

            # Khi train, loại bỏ các feature target phụ khỏi feature_names
            drop_features = ['price_change', 'price_pct_change', 'future_close']
            X_full = X_full.drop(columns=[col for col in drop_features if col in X_full.columns], errors='ignore')
            X_train = X_full.iloc[:int(len(X_full) * 0.8)]
            X_test = X_full.iloc[int(len(X_full) * 0.8):]
            y_train = y_full.iloc[:int(len(y_full) * 0.8)]
            y_test = y_full.iloc[int(len(y_full) * 0.8):]

            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.feature_selection import SelectFromModel, mutual_info_classif, mutual_info_regression
            from sklearn.model_selection import TimeSeriesSplit, cross_val_score
            import optuna
            model_type = self.ml_predictor.model_type
            logger.info(f"Feature selection bắt đầu với {X_train.shape[1]} features")
            correlation_threshold = 0.90
            corr_matrix = X_train.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            if len(to_drop) > 10:
                to_drop = sorted([(column, upper[column].max()) for column in to_drop if any(upper[column] > correlation_threshold)], 
                                key=lambda x: x[1], reverse=True)[:10]
                to_drop = [x[0] for x in to_drop]
            X_train_decorr = X_train.drop(columns=to_drop)
            X_test_decorr = X_test.drop(columns=to_drop)
            logger.info(f"Loại bỏ {len(to_drop)} features có tương quan cao > {correlation_threshold}")
            mi_threshold = 0.01
            if model_type == 'classifier':
                mi_scores = mutual_info_classif(X_train_decorr, y_train, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_train_decorr, y_train, random_state=42)
            mi_features = [f for f, score in zip(X_train_decorr.columns, mi_scores) if score > mi_threshold]
            if len(mi_features) < 5:
                if len(mi_features) == 0:
                    mi_features = list(X_train_decorr.columns)
                else:
                    mi_features = [col for _, col in sorted(zip(mi_scores, X_train_decorr.columns), reverse=True)[:5]]
            X_train_mi = X_train_decorr[mi_features]
            X_test_mi = X_test_decorr[mi_features]
            logger.info(f"Chọn {len(mi_features)} features dựa trên mutual information > {mi_threshold}")
            if model_type == 'classifier':
                selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                selector = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            selector.fit(X_train_mi, y_train)
            importances = selector.feature_importances_
            importance_threshold = max(0.01, np.percentile(importances, 25))
            feature_importance = sorted(zip(importances, X_train_mi.columns), reverse=True)
            important_features = [f for imp, f in feature_importance if imp > importance_threshold]
            if len(important_features) < 10:
                important_features = [f for _, f in sorted(zip(importances, X_train_mi.columns), reverse=True)[:10]]
            X_train = X_train_mi[important_features]
            X_test = X_test_mi[important_features]
            self.feature_importance_scores = {f: imp for imp, f in feature_importance}
            logger.info(f"Chọn {len(important_features)}/{X_train_mi.shape[1]} features dựa trên feature importance > {importance_threshold}")
            logger.info(f"Kết thúc feature selection với {X_train.shape[1]} features được chọn từ {X_full.shape[1]} features ban đầu")
            model_info_path = self._get_model_info_path(symbol)
            opt_result_path = os.path.join(model_info_path, "optimization_results.json")
            model_path = self._get_model_path(symbol)
            is_new_model = not os.path.exists(model_path)
            need_optimize = True
            if os.path.exists(opt_result_path):
                with open(opt_result_path, "r") as f:
                    opt_data = json.load(f)
                    last_optimized = opt_data.get("optimized_at")
                    if last_optimized:
                        try:
                            last_optimized_dt = datetime.fromisoformat(last_optimized)
                            if datetime.now() - last_optimized_dt < timedelta(days=90):
                                need_optimize = False
                        except Exception:
                            pass
                    else:
                        need_optimize = False
            if (is_new_model or need_optimize) and not from_optimization:
                self._optuna_optimize(symbol, X_train, y_train, X_test, y_test, important_features)
                return {"info": "Hyperparameter optimization triggered. Training will continue after optimization."}
            else:
                if os.path.exists(opt_result_path):
                    with open(opt_result_path, "r") as f:
                        opt_data = json.load(f)
                        best_params = opt_data.get("best_params")
                        if best_params:
                            model_name = self.ml_predictor.config["model_name"]
                            self.ml_predictor.config["hyperparameters"][model_name].update(best_params)
                            self.ml_predictor._initialize_model()
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = {}
            model_names = ['random_forest', 'xgboost']
            best_cv_score = -np.inf
            best_model_name = None
            for model_name in model_names:
                self.ml_predictor.config['model_name'] = model_name
                self.ml_predictor._initialize_model()
                model = self.ml_predictor.model
                try:
                    if model_type == 'classifier':
                        scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                                scoring='f1', n_jobs=-1)
                    else:
                        scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                                scoring='r2', n_jobs=-1)
                    cv_mean = np.mean(scores)
                    cv_std = np.std(scores)
                    cv_scores[model_name] = {'mean': cv_mean, 'std': cv_std, 'scores': scores.tolist()}
                    logger.info(f"CV Score cho {model_name}: {cv_mean:.4f} ± {cv_std:.4f}")
                    if cv_mean > best_cv_score:
                        best_cv_score = cv_mean
                        best_model_name = model_name
                except Exception as e:
                    logger.warning(f"Cross-validation cho {model_name} gặp lỗi: {e}")
                    cv_scores[model_name] = {'error': str(e)}
            best_metric = -np.inf
            best_model = None
            best_model_name = best_model_name
            best_feature_names = None
            best_metrics = None
            for model_name in model_names:
                if model_name in cv_scores and 'error' in cv_scores[model_name]:
                    logger.warning(f"Bỏ qua {model_name} do lỗi trong CV")
                    continue
                self.ml_predictor.config['model_name'] = model_name
                self.ml_predictor._initialize_model()
                model = self.ml_predictor.model
                if model_name == 'xgboost':
                    fit_params = {}
                    if model_type == 'classifier':
                        fit_params = {
                            'eval_set': [(X_test, y_test)],
                            'early_stopping_rounds': 20,
                            'verbose': False
                        }
                    try:
                        model.fit(X_train, y_train, **fit_params)
                    except Exception as e:
                        logger.warning(f"Early stopping failed for {model_name}: {e}")
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                self.ml_predictor.model = model
                self.ml_predictor.feature_names = important_features
                metrics = self.ml_predictor.evaluate(X_test, y_test)
                metric_value = metrics.get('f1') if model_type == 'classifier' else metrics.get('r2')
                if metric_value is not None and metric_value > best_metric:
                    best_metric = metric_value
                    best_model = model
                    best_model_name = model_name
                    best_feature_names = important_features.copy()
                    best_metrics = metrics
            if best_model is None:
                return {"error": "Không train được model phù hợp cho mã này"}
            self.ml_predictor.model = best_model
            self.ml_predictor.config['model_name'] = best_model_name
            self.ml_predictor.feature_names = best_feature_names
            # Kiểm tra feature_names trước khi lưu model
            if not self.ml_predictor.feature_names or len(self.ml_predictor.feature_names) == 0:
                logger.error("Không có feature_names để lưu cùng model! Không lưu model.")
                return {"error": "Không có feature_names để lưu cùng model"}
            # Đảm bảo set feature_names trước khi lưu
            if hasattr(self.ml_predictor, 'set_feature_names'):
                self.ml_predictor.set_feature_names(best_feature_names)
            else:
                logger.error("MLPredictor không có method set_feature_names!")
                return {"error": "MLPredictor missing set_feature_names method"}
            self.ml_predictor.save_model(model_path)
            if hasattr(self.feature_generator, 'scaler') and self.feature_generator.scaler is not None:
                scaler_path = self._get_scaler_path(symbol)
                self.feature_generator.save_scaler(scaler_path)
            self.ml_predictor.plot_feature_importance(save_path=os.path.join(self.model_dir, f"{symbol}_feature_importance.png"))
            logger.info(f"Model trained for {symbol} - Best model: {best_model_name} - Metrics: {best_metrics}")
            with open(os.path.join(model_info_path, "cv_scores.json"), "w") as f:
                json.dump(cv_scores, f, indent=4, default=str)
            # Lưu toàn bộ features cho train/audit
            features_train_path = os.path.join(self.model_dir, f"{symbol}_features_train.parquet")
            features.to_parquet(features_train_path)
            logger.info(f"Đã lưu toàn bộ features train vào {features_train_path}")
            # Lưu features tối giản cho predictor (chỉ các cột thực sự dùng cho model)
            features_predict_path = os.path.join(self.model_dir, f"{symbol}_features_predict.parquet")
            features_predict = features[important_features] if all(f in features.columns for f in important_features) else features
            features_predict.to_parquet(features_predict_path)
            logger.info(f"Đã lưu features predictor vào {features_predict_path}")
            # Tính hash SHA256 của file features_predict
            def file_sha256(path):
                h = hashlib.sha256()
                with open(path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        h.update(chunk)
                return h.hexdigest()
            # Generate hash for feature generator
            feature_generator_hash = self._get_feature_generator_hash()
            features_predict_hash = file_sha256(features_predict_path)
            logger.info(f"Hash SHA256 của features_predict: {features_predict_hash}")
            # Lưu hash này vào metadata của model
            if hasattr(self.ml_predictor, 'save_model'):
                self.ml_predictor.save_model(model_path, extra_metadata={
                    "features_predict_hash": features_predict_hash,
                    "feature_generator_hash": feature_generator_hash,
                    "version": "1.0",
                    "timestamp": datetime.now().isoformat()
                })
            else:
                logger.error("MLPredictor không có method save_model!")
                raise AttributeError("MLPredictor missing save_model method")
            return {
                "success": True,
                "metrics": best_metrics,
                "model_path": model_path,
                "feature_importance_path": os.path.join(self.model_dir, f"{symbol}_feature_importance.png"),
                "best_model": best_model_name,
                "cv_scores": cv_scores
            }
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {"error": f"Model training failed: {str(e)}"}

    def _train_simple_model(self, symbol: str, data: pd.DataFrame, 
                         tech_analysis: Optional[Dict[str, Any]] = None,
                         news_sentiment: Optional[Dict[str, Any]] = None,
                         market_condition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Train a simple ML model for the given stock
        """
        try:
            logger.info(f"Training simple model for {symbol}...")
            # Check if we have enough data
            min_required = 50
            if len(data) < 100:
                logger.warning(f"Not enough data to train model for {symbol}")
                return False
            # Try with decreasing prediction_horizon if not enough data
            prediction_horizon = self.config.get("prediction_horizon", 5)
            tried_horizons = []
            while prediction_horizon >= 1:
                tried_horizons.append(prediction_horizon)
                data['target'] = data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
                data.dropna(inplace=True)
                if len(data) >= min_required:
                    break
                prediction_horizon -= 1
                data = data.copy()  # reset for next try
                data['target'] = np.nan
            if len(data) < min_required:
                logger.warning(f"Not enough data after creating target for {symbol} even with horizons tried: {tried_horizons}")
                logger.error(f"Không đủ dữ liệu để train model cho {symbol} (cần tối thiểu {min_required} dòng sau khi tạo target)")
                return False
            # Prepare features
            features = self.feature_generator.prepare_features(
                data, tech_analysis, news_sentiment, market_condition
            )
            if features.empty:
                logger.error(f"Failed to prepare features for {symbol}")
                return False
            # Binary classification - positive or negative return
            data['target'] = (data['target'] > 0).astype(int)
            # Get feature matrix and target
            X = features.iloc[:-prediction_horizon]  # Remove last rows that don't have targets
            y = data['target'].iloc[prediction_horizon:len(X)+prediction_horizon]
            if len(X) != len(y):
                logger.error(f"Feature matrix and target dimensions mismatch: {len(X)} vs {len(y)}")
                return False
            model_path = self._get_model_path(symbol)
            # Configure ML predictor
            self.ml_predictor.set_params(
                model_type='random_forest',
                target_column='target',
                prediction_type='classification',
                prediction_horizon=prediction_horizon
            )
            # Fit model
            logger.info(f"Training model for {symbol} with {len(X)} samples and horizon={prediction_horizon}")
            self.ml_predictor.train(X, y)
            # Save model
            self.ml_predictor.save_model(model_path)
            logger.info(f"Simple model trained and saved for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error training simple model for {symbol}: {str(e)}")
            return False

    def _optuna_optimize(self, symbol: str, X_train, y_train, X_test, y_test, selected_features):
        """
        Tối ưu hóa hyperparameter với Optuna
        """
        import optuna
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        model_type = self.ml_predictor.model_type
        direction = "maximize" if model_type == 'classifier' else "minimize"
        metric = "f1" if model_type == 'classifier' else "rmse"
        if not optuna:
            return self._simple_optimize(symbol, X_train, y_train, X_test, y_test, selected_features)
        def objective_rf(trial):
            max_features_options = ['sqrt', 'log2', None]
            max_features_value = trial.suggest_categorical('max_features', max_features_options)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': max_features_value
            }
            if model_type == 'classifier':
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model.fit(X_tr, y_tr)
                if model_type == 'classifier':
                    y_pred = model.predict(X_val)
                    from sklearn.metrics import f1_score
                    score = f1_score(y_val, y_pred)
                else:
                    y_pred = model.predict(X_val)
                    from sklearn.metrics import r2_score
                    score = r2_score(y_val, y_pred)
                scores.append(score)
            return np.mean(scores)
        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 1)
            }
            try:
                import xgboost as xgb
                if model_type == 'classifier':
                    model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
                else:
                    model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    model.fit(X_tr, y_tr, 
                              eval_set=[(X_val, y_val)], 
                              early_stopping_rounds=20,
                              verbose=False)
                    if model_type == 'classifier':
                        y_pred = model.predict(X_val)
                        from sklearn.metrics import f1_score
                        score = f1_score(y_val, y_pred)
                    else:
                        y_pred = model.predict(X_val)
                        from sklearn.metrics import r2_score
                        score = r2_score(y_val, y_pred)
                    scores.append(score)
                return np.mean(scores)
            except Exception as e:
                logger.warning(f"Error in XGBoost tuning: {e}")
                return -np.inf if direction == 'maximize' else np.inf
        optimization_results = {}
        models = ['random_forest', 'xgboost']
        objectives = {
            'random_forest': objective_rf,
            'xgboost': objective_xgb,
        }
        for model_name in models:
            try:
                logger.info(f"Bắt đầu tối ưu hóa {model_name} với Optuna")
                study = optuna.create_study(direction=direction)
                study.optimize(objectives[model_name], n_trials=30)
                best_params = study.best_params
                best_value = study.best_value
                logger.info(f"Tối ưu hóa {model_name} hoàn tất - Best {metric}: {best_value}")
                logger.info(f"Best parameters cho {model_name}: {best_params}")
                optimization_results[model_name] = {
                    "best_params": best_params,
                    "best_value": best_value
                }
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {str(e)}")
        model_info_path = self._get_model_info_path(symbol)
        os.makedirs(model_info_path, exist_ok=True)
        best_model_name = None
        best_score = -np.inf
        for model_name, result in optimization_results.items():
            if 'best_value' in result and result['best_value'] > best_score:
                best_score = result['best_value']
                best_model_name = model_name
        combined_result = {
            "optimized_at": datetime.now().isoformat(),
            "best_model": best_model_name,
            "best_value": best_score,
            "best_params": optimization_results.get(best_model_name, {}).get("best_params", {}),
            "all_results": optimization_results,
            "method": "simple_grid_search",
            "feature_names": selected_features
        }
        with open(os.path.join(model_info_path, "optimization_results.json"), "w") as f:
            json.dump(combined_result, f, indent=4)
        if best_model_name:
            self.ml_predictor.config["model_name"] = best_model_name
            self.ml_predictor.config["hyperparameters"][best_model_name].update(
                optimization_results[best_model_name]["best_params"]
            )
            self.ml_predictor._initialize_model()
        return self.train_model(symbol, None, None, None, None, from_optimization=True)

    def _simple_optimize(self, symbol: str, X_train, y_train, X_test, y_test, selected_features):
        """
        Phương thức thay thế khi không có optuna - tối ưu hóa đơn giản với grid search
        """
        model_type = self.ml_predictor.model_type
        optimization_results = {}
        model_names = ['random_forest', 'xgboost']
        # 1. Random Forest
        try:
            logger.info(f"Tối ưu hóa RandomForest với Grid Search đơn giản")
            n_estimators_list = [100, 200, 300]
            max_depth_list = [10, 15, 20, None]
            max_features_list = ['sqrt', 'log2', None]
            best_score = -np.inf
            best_params = {}
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            tscv = TimeSeriesSplit(n_splits=3)
            for n_estimators in n_estimators_list:
                for max_depth in max_depth_list:
                    for max_features in max_features_list:
                        max_features_value = max_features
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'max_features': max_features_value,
                            'random_state': 42,
                            'n_jobs': -1
                        }
                        if model_type == 'classifier':
                            model = RandomForestClassifier(**params)
                        else:
                            model = RandomForestRegressor(**params)
                        scores = []
                        for train_idx, val_idx in tscv.split(X_train):
                            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                            model.fit(X_tr, y_tr)
                            if model_type == 'classifier':
                                y_pred = model.predict(X_val)
                                from sklearn.metrics import f1_score
                                score = f1_score(y_val, y_pred)
                            else:
                                y_pred = model.predict(X_val)
                                from sklearn.metrics import r2_score
                                score = r2_score(y_val, y_pred)
                            scores.append(score)
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = params
            optimization_results['random_forest'] = {
                'best_params': best_params,
                'best_value': best_score
            }
            logger.info(f"Tối ưu hóa RandomForest hoàn tất - Best score: {best_score}")
            logger.info(f"Best parameters cho RandomForest: {best_params}")
        except Exception as e:
            logger.error(f"Error optimizing RandomForest: {str(e)}")
        # 2. XGBoost (nếu có)
        try:
            import xgboost as xgb
            logger.info(f"Tối ưu hóa XGBoost với Grid Search đơn giản")
            n_estimators_list = [100, 200]
            max_depth_list = [3, 6, 9]
            learning_rate_list = [0.01, 0.1, 0.2]
            best_score = -np.inf
            best_params = {}
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=3)
            for n_estimators in n_estimators_list:
                for max_depth in max_depth_list:
                    for learning_rate in learning_rate_list:
                        params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'learning_rate': learning_rate,
                            'random_state': 42,
                            'n_jobs': -1
                        }
                        if model_type == 'classifier':
                            model = xgb.XGBClassifier(**params)
                        else:
                            model = xgb.XGBRegressor(**params)
                        scores = []
                        for train_idx, val_idx in tscv.split(X_train):
                            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                            try:
                                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
                            except:
                                model.fit(X_tr, y_tr)
                            if model_type == 'classifier':
                                y_pred = model.predict(X_val)
                                from sklearn.metrics import f1_score
                                score = f1_score(y_val, y_pred)
                            else:
                                y_pred = model.predict(X_val)
                                from sklearn.metrics import r2_score
                                score = r2_score(y_val, y_pred)
                            scores.append(score)
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = params
            optimization_results['xgboost'] = {
                'best_params': best_params,
                'best_value': best_score
            }
            logger.info(f"Tối ưu hóa XGBoost hoàn tất - Best score: {best_score}")
            logger.info(f"Best parameters cho XGBoost: {best_params}")
        except Exception as e:
            logger.error(f"Error optimizing XGBoost: {str(e)}")
        best_model_name = None
        best_score = -np.inf
        for model_name, result in optimization_results.items():
            if 'best_value' in result and result['best_value'] > best_score:
                best_score = result['best_value']
                best_model_name = model_name
        model_info_path = self._get_model_info_path(symbol)
        os.makedirs(model_info_path, exist_ok=True)
        combined_result = {
            "optimized_at": datetime.now().isoformat(),
            "best_model": best_model_name,
            "best_value": best_score,
            "best_params": optimization_results.get(best_model_name, {}).get("best_params", {}),
            "all_results": optimization_results,
            "method": "simple_grid_search",
            "feature_names": selected_features
        }
        with open(os.path.join(model_info_path, "optimization_results.json"), "w") as f:
            json.dump(combined_result, f, indent=4)
        if best_model_name:
            self.ml_predictor.config["model_name"] = best_model_name
            self.ml_predictor.config["hyperparameters"][best_model_name].update(
                optimization_results[best_model_name]["best_params"]
            )
            self.ml_predictor._initialize_model()
        return self.train_model(symbol, None, None, None, None, from_optimization=True)

    def _select_features(self, features: pd.DataFrame, expected_count: int) -> List[str]:
        """
        Chọn các features quan trọng nhất từ DataFrame
        """
        try:
            if features is None or features.empty:
                logger.error("Không thể chọn features từ dữ liệu rỗng")
                return []
            if len(features.columns) <= expected_count:
                logger.info(f"Số features hiện tại ({len(features.columns)}) ít hơn hoặc bằng số cần thiết ({expected_count})")
                return list(features.columns)
            # Phương pháp 1: Chọn features đầu tiên
            selected_features = list(features.columns[:expected_count])
            logger.info(f"Chọn {expected_count} features đầu tiên: {selected_features[:5]}...")
            # Phương pháp 2: Chọn features dựa trên phương sai
            try:
                variances = features.var()
                sorted_features = variances.sort_values(ascending=False).index.tolist()
                variance_selected = sorted_features[:expected_count]
                logger.info(f"Chọn {expected_count} features có phương sai cao nhất: {variance_selected[:5]}...")
                selected_features = variance_selected
            except Exception as e:
                logger.warning(f"Lỗi khi chọn features theo phương sai: {str(e)}")
            # Phương pháp 3: Nếu có thể, sử dụng RandomForest để chọn features quan trọng
            try:
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.feature_selection import SelectFromModel
                X = features.fillna(0)
                selector = RandomForestRegressor(n_estimators=50, random_state=42)
                if 'close' in X.columns:
                    y = X['close'].pct_change(5).shift(-5)
                    valid_idx = ~y.isna()
                    if sum(valid_idx) > expected_count:
                        X_valid = X[valid_idx]
                        y_valid = y[valid_idx]
                        selector.fit(X_valid, y_valid)
                        importances = selector.feature_importances_
                        indices = np.argsort(importances)[::-1]
                        rf_selected = [X.columns[indices[i]] for i in range(min(expected_count, len(indices)))]
                        logger.info(f"Chọn {len(rf_selected)} features quan trọng nhất dựa trên RandomForest: {rf_selected[:5]}...")
                        selected_features = rf_selected
            except Exception as e:
                logger.warning(f"Lỗi khi chọn features bằng RandomForest: {str(e)}")
            logger.info(f"Đã chọn {len(selected_features)} features từ tổng số {len(features.columns)} features")
            return selected_features
        except Exception as e:
            logger.error(f"Lỗi khi chọn features: {str(e)}")
            return list(features.columns[:expected_count])

    def _get_model_path(self, symbol: str) -> str:
        return os.path.join(self.model_dir, f"{symbol}_model.joblib")

    def _get_model_info_path(self, symbol: str) -> str:
        return os.path.join(self.model_dir, f"{symbol}_info")

    def _get_scaler_path(self, symbol: str) -> str:
        return os.path.join(self.model_dir, f"{symbol}_scaler.joblib")

    def _get_feature_generator_hash(self) -> str:
        """
        Tạo hash để định danh phiên bản của feature generator
        """
        try:
            # Tạo một string đại diện cho config của feature generator
            config_str = json.dumps(self.feature_generator.config, sort_keys=True)
            # Tạo hash từ config
            import hashlib
            m = hashlib.sha256()
            m.update(config_str.encode())
            return m.hexdigest()
        except Exception as e:
            logger.warning(f"Không thể tạo hash cho feature generator: {str(e)}")
            return "unknown"

    def ensure_model_synced(self, symbol: str, data: pd.DataFrame,
                            tech_analysis: Optional[Dict[str, Any]] = None,
                            news_sentiment: Optional[Dict[str, Any]] = None,
                            market_condition: Optional[Dict[str, Any]] = None) -> bool:
        """
        Đảm bảo model và scaler đã tồn tại và đồng bộ với feature hiện tại.
        Nếu thiếu hoặc không đồng bộ, tự động train lại.
        Trả về True nếu model đã sẵn sàng, False nếu lỗi.
        """
        if data is None or not isinstance(data, pd.DataFrame) or data.empty:
            logger.error(f"Invalid data for model sync check: {symbol}")
            return False
        
        model_path = self._get_model_path(symbol)
        scaler_path = self._get_scaler_path(symbol)
        model_exists = os.path.exists(model_path)
        scaler_exists = os.path.exists(scaler_path)
        loaded = False
        # Load model nếu tồn tại
        if model_exists and scaler_exists:
            try:
                loaded = self.ml_predictor.load_model(model_path)
                # Kiểm tra feature_names có khớp với data hiện tại không
                current_features = list(data.columns)
                model_features = getattr(self.ml_predictor, 'feature_names', None)
                if not model_features or set(model_features) != set(current_features):
                    logger.warning(f"Model feature_names mismatch. Retraining model for {symbol}...")
                    raise Exception("Feature mismatch")
                # Load scaler nếu có
                if hasattr(self.feature_generator, 'load_scaler'):
                    self.feature_generator.load_scaler(scaler_path)
                logger.info(f"Model and scaler for {symbol} are in sync.")
                return True
            except Exception as e:
                logger.warning(f"Model exists but failed to load or feature mismatch: {e}")
                loaded = False
        # Nếu chưa có model hoặc không đồng bộ, train lại
        try:
            logger.info(f"Training model for {symbol} (auto sync)...")
            # Đảm bảo dữ liệu không có cột target khi train
            target_cols = ['target', 'future_close', 'price_change', 'price_pct_change', 'y']
            data_for_training = data.copy()
            for col in target_cols:
                if col in data_for_training.columns:
                    logger.info(f"Removing target column {col} before training")
                    data_for_training = data_for_training.drop(columns=[col], errors='ignore')
                    
            result = self.train_model(symbol, data_for_training, tech_analysis, news_sentiment, market_condition)
            if "error" in result:
                logger.error(f"Error training model: {result['error']}")
                return False
            
            # Save scaler nếu có
            if hasattr(self.feature_generator, 'save_scaler'):
                self.feature_generator.save_scaler(scaler_path)
            logger.info(f"Model and scaler for {symbol} trained and saved.")
            return True
        except Exception as e:
            logger.error(f"Failed to train and sync model for {symbol}: {e}")
            return False

    def prepare_target(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa target cho train (binary, multiclass, regression)
        """
        try:
            data_with_target = features.copy()
            days_ahead = self.ml_predictor.config.get("days_ahead", 5)
            target_type = self.ml_predictor.config.get("target_type", "binary")
            target_column = self.ml_predictor.config.get("target_column", "target")
            data_with_target['future_close'] = data_with_target['close'].shift(-days_ahead)
            data_with_target['price_change'] = data_with_target['future_close'] - data_with_target['close']
            data_with_target['price_pct_change'] = data_with_target['price_change'] / data_with_target['close'] * 100
            if target_type == "binary":
                threshold = self.ml_predictor.config.get("threshold", 0)
                data_with_target[target_column] = (data_with_target['price_pct_change'] > threshold).astype(int)
            elif target_type == "multiclass":
                ranges = self.ml_predictor.config.get("class_ranges", [-5, -2, 2, 5])
                labels = list(range(len(ranges) + 1))
                data_with_target[target_column] = pd.cut(data_with_target['price_pct_change'], bins=[-float('inf')] + ranges + [float('inf')], labels=labels)
            elif target_type == "regression":
                data_with_target[target_column] = data_with_target['price_pct_change']
            else:
                logger.error(f"Invalid target type: {target_type}")
                return pd.DataFrame()
            data_with_target = data_with_target.dropna(subset=['future_close', target_column])
            return data_with_target
        except Exception as e:
            logger.error(f"Error preparing target: {str(e)}")
            return pd.DataFrame() 