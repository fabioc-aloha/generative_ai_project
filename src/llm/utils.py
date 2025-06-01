"""
LLM Utilities Module

This module provides utility functions for working with Language Learning Models,
including token counting, rate limiting helpers, and model management utilities.

Author: Brij Kishore Pandey
"""

import time
import hashlib
from typing import Dict, List, Optional, Any, Union, Callable
from functools import wraps
import logging

from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .claude_client import ClaudeClient


logger = logging.getLogger(__name__)


class LLMManager:
    """
    Manager class for handling multiple LLM clients and providing unified access.
    
    This class allows you to manage multiple LLM providers and switch between them
    easily while maintaining consistent interfaces and configuration.
    
    Example:
        >>> manager = LLMManager()
        >>> manager.add_client("gpt", OpenAIClient(model_name="gpt-3.5-turbo"))
        >>> manager.add_client("claude", ClaudeClient(model_name="claude-3-sonnet"))
        >>> response = manager.generate("gpt", "Hello, world!")
    """
    
    def __init__(self):
        """Initialize the LLM manager."""
        self.clients: Dict[str, BaseLLMClient] = {}
        self.default_client: Optional[str] = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_client(self, name: str, client: BaseLLMClient) -> None:
        """
        Add a new LLM client to the manager.
        
        Args:
            name (str): Unique name for the client
            client (BaseLLMClient): LLM client instance
        """
        self.clients[name] = client
        if self.default_client is None:
            self.default_client = name
        
        self.logger.info(f"Added client '{name}': {client}")
    
    def remove_client(self, name: str) -> None:
        """
        Remove a client from the manager.
        
        Args:
            name (str): Name of the client to remove
            
        Raises:
            KeyError: If client doesn't exist
        """
        if name not in self.clients:
            raise KeyError(f"Client '{name}' not found")
        
        del self.clients[name]
        
        # Update default if needed
        if self.default_client == name:
            self.default_client = next(iter(self.clients.keys())) if self.clients else None
        
        self.logger.info(f"Removed client '{name}'")
    
    def set_default(self, name: str) -> None:
        """
        Set the default client to use.
        
        Args:
            name (str): Name of the client to set as default
            
        Raises:
            KeyError: If client doesn't exist
        """
        if name not in self.clients:
            raise KeyError(f"Client '{name}' not found")
        
        self.default_client = name
        self.logger.info(f"Set default client to '{name}'")
    
    def generate(
        self,
        prompt: str,
        client_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text using specified or default client.
        
        Args:
            prompt (str): Input prompt
            client_name (Optional[str]): Client to use (default: default_client)
            **kwargs: Additional parameters for generation
            
        Returns:
            str: Generated text
            
        Raises:
            ValueError: If no clients available or client not found
        """
        client_name = client_name or self.default_client
        
        if not client_name or client_name not in self.clients:
            raise ValueError(f"Client '{client_name}' not available")
        
        return self.clients[client_name].generate_text(prompt, **kwargs)
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        client_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion using specified or default client.
        
        Args:
            messages (List[Dict[str, str]]): Conversation history
            client_name (Optional[str]): Client to use (default: default_client)
            **kwargs: Additional parameters for completion
            
        Returns:
            str: Generated response
            
        Raises:
            ValueError: If no clients available or client not found
        """
        client_name = client_name or self.default_client
        
        if not client_name or client_name not in self.clients:
            raise ValueError(f"Client '{client_name}' not available")
        
        return self.clients[client_name].chat_completion(messages, **kwargs)
    
    def get_client_info(self, client_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific client or all clients.
        
        Args:
            client_name (Optional[str]): Client to get info for (default: all)
            
        Returns:
            Dict[str, Any]: Client information
        """
        if client_name:
            if client_name not in self.clients:
                raise KeyError(f"Client '{client_name}' not found")
            return self.clients[client_name].get_model_info()
        
        return {
            name: client.get_model_info()
            for name, client in self.clients.items()
        }
    
    def list_clients(self) -> List[str]:
        """
        List all available client names.
        
        Returns:
            List[str]: List of client names
        """
        return list(self.clients.keys())


def rate_limit(calls_per_minute: int = 60):
    """
    Decorator to rate limit function calls.
    
    Args:
        calls_per_minute (int): Maximum calls allowed per minute
        
    Returns:
        Callable: Decorated function with rate limiting
        
    Example:
        >>> @rate_limit(calls_per_minute=30)
        ... def api_call():
        ...     return "response"
    """
    min_interval = 60.0 / calls_per_minute
    last_called = {}
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use function name and first arg (usually self) as key
            key = f"{func.__name__}_{id(args[0]) if args else 'global'}"
            
            now = time.time()
            if key in last_called:
                elapsed = now - last_called[key]
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            last_called[key] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0
):
    """
    Decorator to retry function calls with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        base_delay (float): Initial delay between retries in seconds
        max_delay (float): Maximum delay between retries in seconds
        exponential_base (float): Base for exponential backoff
        
    Returns:
        Callable: Decorated function with retry logic
        
    Example:
        >>> @retry_with_backoff(max_retries=3, base_delay=1.0)
        ... def unreliable_api_call():
        ...     # This will retry up to 3 times with exponential backoff
        ...     return make_api_request()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Final retry failed for {func.__name__}: {str(e)}")
                        raise e
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def estimate_tokens(text: str, model_type: str = "gpt") -> int:
    """
    Estimate the number of tokens in a text string.
    
    Args:
        text (str): Text to estimate tokens for
        model_type (str): Type of model ("gpt", "claude", etc.)
        
    Returns:
        int: Estimated token count
        
    Note:
        This is a rough estimation. For precise counting, use model-specific tokenizers.
    """
    if model_type.lower() in ["gpt", "openai"]:
        # GPT models: ~4 characters per token
        return len(text) // 4
    elif model_type.lower() in ["claude", "anthropic"]:
        # Claude models: ~3.5 characters per token
        return int(len(text) / 3.5)
    else:
        # Generic estimation
        return len(text) // 4


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_name: str
) -> float:
    """
    Calculate estimated cost for LLM usage.
    
    Args:
        input_tokens (int): Number of input tokens
        output_tokens (int): Number of output tokens
        model_name (str): Name of the model
        
    Returns:
        float: Estimated cost in USD
        
    Note:
        Pricing is approximate and may change. Check provider documentation for current rates.
    """
    # Approximate pricing (per 1K tokens) as of 2024
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-32k": {"input": 0.06, "output": 0.12},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }
    
    # Find matching model pricing
    model_pricing = None
    for model_key, prices in pricing.items():
        if model_name.startswith(model_key):
            model_pricing = prices
            break
    
    if not model_pricing:
        logger.warning(f"Pricing not available for model: {model_name}")
        return 0.0
    
    input_cost = (input_tokens / 1000) * model_pricing["input"]
    output_cost = (output_tokens / 1000) * model_pricing["output"]
    
    return input_cost + output_cost


def create_client_from_config(config: Dict[str, Any]) -> BaseLLMClient:
    """
    Create an LLM client from configuration dictionary.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        BaseLLMClient: Configured LLM client
        
    Raises:
        ValueError: If provider is not supported or config is invalid
        
    Example:
        >>> config = {
        ...     "provider": "openai",
        ...     "model_name": "gpt-3.5-turbo",
        ...     "api_key": "your-key",
        ...     "max_tokens": 1000
        ... }
        >>> client = create_client_from_config(config)
    """
    provider = config.get("provider", "").lower()
    
    if provider in ["openai", "gpt"]:
        return OpenAIClient(**{k: v for k, v in config.items() if k != "provider"})
    elif provider in ["anthropic", "claude"]:
        return ClaudeClient(**{k: v for k, v in config.items() if k != "provider"})
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def generate_prompt_hash(prompt: str, params: Dict[str, Any] = None) -> str:
    """
    Generate a hash for a prompt and parameters for caching purposes.
    
    Args:
        prompt (str): The prompt text
        params (Dict[str, Any], optional): Additional parameters
        
    Returns:
        str: SHA-256 hash of the prompt and parameters
    """
    content = prompt
    if params:
        # Sort params for consistent hashing
        sorted_params = sorted(params.items())
        content += str(sorted_params)
    
    return hashlib.sha256(content.encode()).hexdigest()


def validate_message_format(messages: List[Dict[str, str]]) -> bool:
    """
    Validate the format of messages for chat completion.
    
    Args:
        messages (List[Dict[str, str]]): Messages to validate
        
    Returns:
        bool: True if format is valid, False otherwise
    """
    if not isinstance(messages, list) or not messages:
        return False
    
    valid_roles = {"system", "user", "assistant"}
    
    for message in messages:
        if not isinstance(message, dict):
            return False
        
        if "role" not in message or "content" not in message:
            return False
        
        if message["role"] not in valid_roles:
            return False
        
        if not isinstance(message["content"], str):
            return False
    
    return True


# Convenience functions for quick access
def quick_generate(
    prompt: str,
    provider: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> str:
    """
    Quick text generation with minimal setup.
    
    Args:
        prompt (str): Input prompt
        provider (str): Provider to use ("openai" or "claude")
        model_name (Optional[str]): Model name (uses default if not specified)
        **kwargs: Additional parameters
        
    Returns:
        str: Generated text
    """
    config = {"provider": provider, **kwargs}
    if model_name:
        config["model_name"] = model_name
    
    client = create_client_from_config(config)
    return client.generate_text(prompt, **kwargs)


def quick_chat(
    messages: List[Dict[str, str]],
    provider: str = "openai",
    model_name: Optional[str] = None,
    **kwargs
) -> str:
    """
    Quick chat completion with minimal setup.
    
    Args:
        messages (List[Dict[str, str]]): Conversation history
        provider (str): Provider to use ("openai" or "claude")
        model_name (Optional[str]): Model name (uses default if not specified)
        **kwargs: Additional parameters
        
    Returns:
        str: Generated response
    """
    if not validate_message_format(messages):
        raise ValueError("Invalid message format")
    
    config = {"provider": provider, **kwargs}
    if model_name:
        config["model_name"] = model_name
    
    client = create_client_from_config(config)
    return client.chat_completion(messages, **kwargs)