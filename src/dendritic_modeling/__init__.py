# Set up the logging environment
from dendritic_modeling.logging_config import LoggerManager

logger_manager = LoggerManager()
logger = logger_manager.get_logger()