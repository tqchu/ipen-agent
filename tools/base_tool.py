# tools/base_tool.py
import logging
from abc import ABC, abstractmethod

class BaseTool(ABC):
    def __init__(self, name: str = None):
        # Each tool uses its own logger; use module or class name if not provided
        tool_name = name or self.__class__.__name__
        self.logger = logging.getLogger(tool_name)

    @abstractmethod
    def run(self, action: "PentestAction", actions: list["PentestAction"], state) -> "Result":
        """Execute the action using the given state and return the result."""
        pass
