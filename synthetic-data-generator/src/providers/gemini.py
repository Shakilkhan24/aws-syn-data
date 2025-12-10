"""Google Gemini API provider."""

from typing import Optional
from google import genai
from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Google Gemini API provider implementation."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", **kwargs):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google Gemini API key
            model: Model name (default: gemini-2.5-flash)
        """
        super().__init__(api_key, model, **kwargs)
        self._client = None
    
    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate content using Gemini API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Generated text or None if generation failed
        """
        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                **kwargs
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {e}")
    
    def test_connection(self) -> bool:
        """
        Test Gemini API connection.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            client = self._get_client()
            # Try a simple generation
            response = client.models.generate_content(
                model=self.model,
                contents="Test"
            )
            return response is not None
        except Exception:
            return False

