# src/Abstract_class.py
from abc import ABC, abstractmethod

class PipelineComponent(ABC):
    @abstractmethod
    def process(self, data):
        """Processes input data and returns output."""
        pass
