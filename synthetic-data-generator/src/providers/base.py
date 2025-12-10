"""Base provider interface."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseProvider(ABC):
    """Base class for AI API providers."""
    
    def __init__(self, api_key: str, model: str, **kwargs):
        """
        Initialize provider.
        
        Args:
            api_key: API key for the service
            model: Model name to use
            **kwargs: Additional provider-specific parameters
        """
        self.api_key = api_key
        self.model = model
        self._client = None
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate content from prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text or None if generation failed
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> bool:
        """
        Test if the API connection is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        pass

