import unittest
from unittest.mock import patch, MagicMock
from dendritic_modeling import LoggerManager

class TestLoggerManager(unittest.TestCase):

    def setUp(self):
        # Reset singleton instance before each test
        LoggerManager._instance = None

    def test_singleton_instance(self):
        """Test that only one instance of LoggerManager is created."""
        instance1 = LoggerManager().get_logger()
        instance2 = LoggerManager().get_logger()
        self.assertIs(instance1, instance2, "LoggerManager should be a singleton")

   

if __name__ == '__main__':
    unittest.main()
