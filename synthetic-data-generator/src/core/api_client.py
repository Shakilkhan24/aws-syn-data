"""API client manager with key rotation and usage tracking."""

import time
from typing import List, Dict, Optional
from ..providers.base import BaseProvider
from ..providers.gemini import GeminiProvider
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class APIClientManager:
    """Manages multiple API clients with automatic rotation and usage tracking."""
    
    def __init__(self, provider: str, model: str, api_keys: List[str], **provider_kwargs):
        """
        Initialize API client manager.
        
        Args:
            provider: Provider name (e.g., 'gemini', 'openai')
            model: Model name to use
            api_keys: List of API keys
            **provider_kwargs: Additional provider-specific parameters
        """
        self.provider_name = provider
        self.model = model
        self.api_keys = api_keys
        self.provider_kwargs = provider_kwargs
        
        if not self.api_keys:
            raise ValueError("At least one API key is required")
        
        self.key_names = [f"KEY_{i+1}" for i in range(len(api_keys))]
        self.current_index = 0
        self.total_requests = 0
        self.key_usage_stats = {name: 0 for name in self.key_names}
        self.failed_keys = set()
        
        # Create provider instances
        self.providers: List[BaseProvider] = []
        for api_key in api_keys:
            provider = self._create_provider(api_key)
            self.providers.append(provider)
        
        logger.info(f"Initialized {self.provider_name} manager with {len(api_keys)} API keys")
        logger.info(f"Current key: {self.key_names[self.current_index]}")
    
    def _create_provider(self, api_key: str) -> BaseProvider:
        """
        Create a provider instance.
        
        Args:
            api_key: API key for the provider
        
        Returns:
            Provider instance
        """
        if self.provider_name.lower() == "gemini":
            return GeminiProvider(api_key, self.model, **self.provider_kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}")
    
    def get_current_provider(self) -> BaseProvider:
        """
        Get the currently active provider.
        
        Returns:
            Current provider instance
        """
        return self.providers[self.current_index]
    
    def get_current_key_info(self) -> Dict:
        """
        Get information about the current API key.
        
        Returns:
            Dictionary with key information
        """
        return {
            "name": self.key_names[self.current_index],
            "index": self.current_index,
            "usage_count": self.key_usage_stats[self.key_names[self.current_index]],
            "total_requests": self.total_requests
        }
    
    def generate(self, prompt: str, max_retries: int = None, **kwargs) -> Optional[str]:
        """
        Generate content using current provider, with automatic key rotation on failure.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum number of retries (default: number of available keys)
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text or None if all keys failed
        """
        if max_retries is None:
            max_retries = len(self.api_keys)
        
        retry_count = 0
        original_index = self.current_index
        
        while retry_count < max_retries:
            provider = self.get_current_provider()
            key_info = self.get_current_key_info()
            
            try:
                logger.debug(f"Using {key_info['name']} (attempt {retry_count + 1})")
                result = provider.generate(prompt, **kwargs)
                
                # Success - increment counters
                self.total_requests += 1
                self.key_usage_stats[key_info['name']] += 1
                
                return result
                
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error with {key_info['name']}: {e}")
                
                if retry_count < max_retries:
                    # Try switching to next key
                    if self.switch_key():
                        time.sleep(2)  # Brief delay before retry
                    else:
                        logger.error("No available keys remaining")
                        return None
                else:
                    logger.error(f"All retries exhausted. Last error: {e}")
                    return None
        
        return None
    
    def switch_key(self) -> bool:
        """
        Switch to the next available API key.
        
        Returns:
            True if switch was successful, False if no keys available
        """
        old_key_name = self.key_names[self.current_index]
        self.failed_keys.add(self.current_index)
        
        # Find next available key
        original_index = self.current_index
        attempts = 0
        
        while attempts < len(self.api_keys):
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            
            if self.current_index == original_index:
                # Cycled through all keys
                available = len(self.api_keys) - len(self.failed_keys)
                if available == 0:
                    logger.error("All API keys have been exhausted")
                    return False
                break
            
            if self.current_index not in self.failed_keys:
                break
            
            attempts += 1
        
        new_key_name = self.key_names[self.current_index]
        logger.info(f"Switched from {old_key_name} → {new_key_name}")
        logger.info(f"Available keys: {len(self.api_keys) - len(self.failed_keys)}/{len(self.api_keys)}")
        
        return True
    
    def print_usage_stats(self) -> None:
        """Print usage statistics for all API keys."""
        logger.info("\n" + "=" * 50)
        logger.info("API Key Usage Statistics")
        logger.info("=" * 50)
        for key_name in self.key_names:
            usage = self.key_usage_stats[key_name]
            idx = self.key_names.index(key_name)
            status = "❌ Failed" if idx in self.failed_keys else "✅ Active"
            current = "🎯 Current" if idx == self.current_index else ""
            logger.info(f"{key_name}: {usage} requests | {status} {current}")
        logger.info(f"Total requests: {self.total_requests}")
        logger.info("=" * 50)

