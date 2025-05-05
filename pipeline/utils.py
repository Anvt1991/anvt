import time
import logging
from datetime import datetime
from typing import Dict, Any, Type, TypeVar, Optional, Tuple, List, Union
from dataclasses import is_dataclass, asdict, fields
import traceback
import os
import json
from pipeline.interfaces import PipelineData, StockData

logger = logging.getLogger(__name__)

T = TypeVar('T')

# Danh sách ngày nghỉ lễ (cập nhật đến hết 2025)
HOLIDAYS = {
    # 2024
    "2024-01-01",  # Tết Dương lịch
    "2024-02-08", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14", "2024-02-15",  # Tết Âm lịch 2024
    "2024-04-18",  # Giỗ tổ Hùng Vương
    "2024-04-29",  # Nghỉ bù Giỗ tổ Hùng Vương (thứ 2)
    "2024-04-30",  # 30/4
    "2024-05-01",  # 1/5
    "2024-09-02",  # 2/9
    "2024-09-03",  # Nghỉ bù Quốc khánh (thứ 3)
    # 2025
    "2025-01-01",  # Tết Dương lịch
    "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-01-31", "2025-02-03",  # Tết Âm lịch 2025 (29/12/2024 - 04/02/2025 dương lịch, nghỉ chính thức 27/1-3/2)
    "2025-04-10",  # Giỗ tổ Hùng Vương
    "2025-04-30",  # 30/4
    "2025-05-01",  # 1/5
    "2025-09-01",  # 2/9 (Nghỉ bù Quốc khánh)
    # ... thêm các ngày nghỉ lễ khác nếu có
}

def time_execution(func):
    """Decorator to measure execution time of a function"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            # Check if the first argument is a pipeline_data object
            if args and hasattr(args[0], 'execution_times'):
                pipeline_data = args[0]
                function_name = func.__name__
                pipeline_data.execution_times[function_name] = execution_time
                logger.info(f"Executed {function_name} in {execution_time:.2f} seconds")
            else:
                logger.info(f"Executed {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing {func.__name__} after {execution_time:.2f} seconds: {str(e)}")
            raise
    return wrapper

def handle_errors(func):
    """Decorator to handle errors in pipeline steps"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if the first argument is a pipeline_data object
            if args and hasattr(args[0], 'has_error'):
                pipeline_data = args[0]
                function_name = func.__name__
                error_message = f"Error in {function_name}: {str(e)}"
                pipeline_data.error = error_message
                pipeline_data.has_error = True
                
                # Add traceback for debugging
                tb = traceback.format_exc()
                if not hasattr(pipeline_data, 'error_details'):
                    pipeline_data.error_details = {}
                pipeline_data.error_details[function_name] = {
                    'error': str(e),
                    'traceback': tb,
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.error(error_message, exc_info=True)
                return pipeline_data
            else:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
    return wrapper

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten dict nhiều tầng thành dict 1 tầng với key dạng a.b.c"""
    if not isinstance(d, dict):
        return {parent_key: d}
        
    items = []
    for k, v in d.items():
        if not k:  # Skip empty keys
            continue
            
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def validate_pipeline_fields(data: dict, required_fields: List[str], context: str = "pipeline") -> List[str]:
    """Kiểm tra các trường dữ liệu cần thiết, log cảnh báo nếu thiếu"""
    if not isinstance(data, dict):
        logger.error(f"[{context}] Invalid data: expected dict, got {type(data)}")
        return required_fields  # All fields are missing if data is not a dict
        
    missing = [f for f in required_fields if f not in data or data[f] is None]
    if missing:
        logger.warning(f"[{context}] Thiếu các trường dữ liệu: {missing}")
    return missing

def dataclass_to_dict(obj: Any, flatten: bool = False, sep: str = '.') -> Dict[str, Any]:
    """Convert a dataclass instance to a dictionary, hỗ trợ flatten nếu cần"""
    if obj is None:
        return {}
    if not is_dataclass(obj):
        raise TypeError(f"Object of type {type(obj)} is not a dataclass")
    
    try:
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            if value is None:
                result[field.name] = None
                continue
                
            if is_dataclass(value):
                result[field.name] = dataclass_to_dict(value, flatten=flatten, sep=sep)
            elif isinstance(value, list) and value and is_dataclass(value[0]):
                result[field.name] = [dataclass_to_dict(item, flatten=flatten, sep=sep) if is_dataclass(item) else item for item in value]
            elif isinstance(value, dict) and any(is_dataclass(v) for v in value.values() if v is not None):
                result[field.name] = {k: dataclass_to_dict(v, flatten=flatten, sep=sep) if is_dataclass(v) else v for k, v in value.items()}
            elif isinstance(value, datetime):
                result[field.name] = value.isoformat()
            elif hasattr(value, 'to_dict'):  # Handle objects with to_dict method
                result[field.name] = value.to_dict()
            else:
                result[field.name] = value
                
        if flatten:
            return flatten_dict(result, sep=sep)
        return result
    except Exception as e:
        logger.error(f"Error converting dataclass to dict: {str(e)}")
        return {"error": f"Failed to convert dataclass to dict: {str(e)}"}

def dict_to_dataclass(data: Dict[str, Any], dataclass_type: Type[T]) -> Optional[T]:
    """Convert a dictionary to a dataclass instance"""
    if data is None:
        return None
    
    if not is_dataclass(dataclass_type):
        raise TypeError(f"Type {dataclass_type} is not a dataclass")
    
    try:
        # Create a copy of the data to avoid modifying the original
        processed_data = {}
        
        # Process each field in the dataclass
        for field in fields(dataclass_type):
            field_name = field.name
            field_type = field.type
            
            # Skip if field is not in data
            if field_name not in data:
                continue
                
            value = data[field_name]
            
            # Handle None values
            if value is None:
                processed_data[field_name] = None
                continue
                
            # Handle nested dataclasses
            if hasattr(field_type, "__origin__") and field_type.__origin__ is Optional:
                # Handle Optional types
                inner_type = field_type.__args__[0]
                if is_dataclass(inner_type) and isinstance(value, dict):
                    processed_data[field_name] = dict_to_dataclass(value, inner_type)
                else:
                    processed_data[field_name] = value
            elif is_dataclass(field_type) and isinstance(value, dict):
                processed_data[field_name] = dict_to_dataclass(value, field_type)
            # Handle lists of dataclasses
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is list:
                if len(field_type.__args__) > 0 and is_dataclass(field_type.__args__[0]) and isinstance(value, list):
                    item_type = field_type.__args__[0]
                    processed_data[field_name] = [dict_to_dataclass(item, item_type) if isinstance(item, dict) else item for item in value]
                else:
                    processed_data[field_name] = value
            # Handle dictionaries
            elif hasattr(field_type, "__origin__") and field_type.__origin__ is dict:
                if len(field_type.__args__) > 1 and is_dataclass(field_type.__args__[1]) and isinstance(value, dict):
                    value_type = field_type.__args__[1]
                    processed_data[field_name] = {k: dict_to_dataclass(v, value_type) if isinstance(v, dict) else v for k, v in value.items()}
                else:
                    processed_data[field_name] = value
            # Handle datetime objects
            elif field_type == datetime and isinstance(value, str):
                try:
                    processed_data[field_name] = datetime.fromisoformat(value)
                except ValueError:
                    # Fallback if fromisoformat fails
                    try:
                        processed_data[field_name] = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")
                    except ValueError:
                        # Final fallback
                        logger.warning(f"Could not parse datetime value '{value}' for field {field_name}, using current time")
                        processed_data[field_name] = datetime.now()
            else:
                processed_data[field_name] = value
        
        # Create and return the dataclass instance
        return dataclass_type(**processed_data)
    except Exception as e:
        logger.error(f"Error converting dict to dataclass {dataclass_type.__name__}: {str(e)}")
        # Return a minimal valid instance
        try:
            minimal_data = {}
            for field in fields(dataclass_type):
                if field.default is not fields.default:
                    # Field has a default value
                    continue
                if field.default_factory is not fields.default_factory:
                    # Field has a default factory
                    continue
                # Field is required, provide a minimal value
                if field.type == str:
                    minimal_data[field.name] = ""
                elif field.type == int:
                    minimal_data[field.name] = 0
                elif field.type == float:
                    minimal_data[field.name] = 0.0
                elif field.type == bool:
                    minimal_data[field.name] = False
                elif field.type == list or (hasattr(field.type, "__origin__") and field.type.__origin__ is list):
                    minimal_data[field.name] = []
                elif field.type == dict or (hasattr(field.type, "__origin__") and field.type.__origin__ is dict):
                    minimal_data[field.name] = {}
            return dataclass_type(**minimal_data)
        except Exception as nested_e:
            logger.error(f"Failed to create minimal dataclass instance: {str(nested_e)}")
            return None

def convert_legacy_to_pipeline_data(legacy_data: Dict[str, Any], from_pipeline_interfaces=None):
    """Convert legacy data format to PipelineData"""
    if from_pipeline_interfaces is None:
        try:
            from pipeline.interfaces import PipelineData, StockData
        except ImportError:
            logger.error("Failed to import PipelineData from pipeline.interfaces")
            return None
    else:
        PipelineData = from_pipeline_interfaces.PipelineData
        StockData = from_pipeline_interfaces.StockData
    
    try:
        symbol = legacy_data.get('symbol', '')
        if not symbol:
            logger.error("Missing required symbol in legacy_data")
            return None
            
        pipeline_data = PipelineData(
            symbol=symbol,
            period=legacy_data.get('period', '1y')
        )
        
        # Convert stock data if available
        if 'data' in legacy_data and legacy_data['data'] is not None:
            try:
                pipeline_data.stock_data = StockData(
                    symbol=symbol,
                    df=legacy_data.get('data'),
                    start_date=legacy_data.get('start_date', datetime.now()),
                    end_date=legacy_data.get('end_date', datetime.now()),
                    timeframe=legacy_data.get('timeframe', 'daily')
                )
            except Exception as e:
                logger.warning(f"Error converting stock data: {str(e)}")
        
        # Store the original legacy data in the result field for backward compatibility
        pipeline_data.result = legacy_data
        
        return pipeline_data
    except Exception as e:
        logger.error(f"Error converting legacy data to PipelineData: {str(e)}")
        return None

def ensure_valid_data(pipeline_data) -> bool:
    """Ensures that a pipeline has valid data to work with"""
    if pipeline_data is None:
        logger.error("Pipeline data is None")
        return False
        
    if not hasattr(pipeline_data, 'has_error'):
        logger.error("Pipeline data object does not have 'has_error' attribute")
        return False
        
    if pipeline_data.has_error:
        logger.error(f"Pipeline has error: {getattr(pipeline_data, 'error', 'Unknown error')}")
        return False
        
    if not hasattr(pipeline_data, 'stock_data'):
        logger.error("Pipeline data object does not have 'stock_data' attribute")
        return False
        
    if pipeline_data.stock_data is None:
        pipeline_data.error = "No stock data available"
        pipeline_data.has_error = True
        return False
        
    # Check if stock_data has the df attribute
    if not hasattr(pipeline_data.stock_data, 'df'):
        pipeline_data.error = "Stock data missing 'df' attribute"
        pipeline_data.has_error = True
        return False
        
    if pipeline_data.stock_data.df is None or (hasattr(pipeline_data.stock_data.df, 'empty') and pipeline_data.stock_data.df.empty):
        pipeline_data.error = "Stock data DataFrame is empty"
        pipeline_data.has_error = True
        return False
        
    # Check for validation results
    if hasattr(pipeline_data, 'validation_result') and pipeline_data.validation_result is not None:
        # Check if data is valid according to validation
        if hasattr(pipeline_data.validation_result, 'is_valid') and not pipeline_data.validation_result.is_valid:
            error_msg = getattr(pipeline_data.validation_result, 'error_message', None) or "Data validation failed"
            pipeline_data.error = error_msg
            pipeline_data.has_error = True
            return False
        
    return True

def is_trading_day(date: datetime.date) -> bool:
    """Kiểm tra một ngày có phải là ngày giao dịch chứng khoán VN không"""
    if date.weekday() >= 5:  # Saturday or Sunday
        return False
    if date.strftime("%Y-%m-%d") in HOLIDAYS:
        return False
    return True

def save_debug_info(pipeline_data, debug_dir: str = "debug_logs"):
    """Save pipeline data to a debug file for troubleshooting"""
    try:
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol = getattr(pipeline_data, 'symbol', 'unknown')
        filename = f"{debug_dir}/{symbol}_{timestamp}_debug.json"
        
        # Convert pipeline data to dict
        if is_dataclass(pipeline_data):
            data_dict = dataclass_to_dict(pipeline_data)
        else:
            # Try to convert using asdict if it's a dataclass-like object
            try:
                data_dict = asdict(pipeline_data)
            except (TypeError, ValueError):
                # Fallback to a simple dict with basic attributes
                data_dict = {
                    'symbol': getattr(pipeline_data, 'symbol', 'unknown'),
                    'error': getattr(pipeline_data, 'error', None),
                    'has_error': getattr(pipeline_data, 'has_error', False),
                    'execution_times': getattr(pipeline_data, 'execution_times', {}),
                }
                
                # Include error details if available
                if hasattr(pipeline_data, 'error_details'):
                    data_dict['error_details'] = pipeline_data.error_details
        
        # Remove DataFrame objects as they're not JSON serializable
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if k != 'df'}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
                
        data_dict = clean_for_json(data_dict)
        
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2, default=str)
            
        logger.info(f"Debug information saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving debug information: {str(e)}")
        return None 