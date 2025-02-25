"""
logging_config.py
=================
This module contains the logging configuration for the dendritic_modeling 
package.
"""

import sys
import logging

LOGGER_NAME = 'dendritic_modeling'

class LoggerManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance
    
    def _initialize_logger(self):
        self.logger = logging.getLogger(LOGGER_NAME)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(levelname)s %(filename)s:%(lineno)d  %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        self.file_path = LOGGER_NAME + '.log'
        self._set_file_handler(self.file_path)

    def _set_file_handler(self, file_path):
        self.logger.handlers = [
            h for h in self.logger.handlers if not isinstance(h, logging.FileHandler)
        ]
        try:
            file_handler = logging.FileHandler(file_path)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s')
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        except Exception as e:
            self.logger.error(f'Failed to set file handler: {e}')

    def set_log_file(self, file_path):
        self.file_path = file_path
        self._set_file_handler(file_path)
        self.logger.info(f'Log file set to {file_path}.')

    def get_logger(self):
        return self.logger