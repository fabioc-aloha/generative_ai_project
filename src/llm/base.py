"""
Base LLM Client Module

This module provides the abstract base class for all LLM client implementations.
It defines the common interface and shared functionality that all LLM providers
should implement.

Author: Brij Kishore Pandey
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime


class BaseLLMClient(ABC):
    """
    Abstract base class for all LLM client implementations.
    
    This class defines the common interface that all LLM providers must implement,
    including methods for text generation, chat completion, and configuration management.
    
    Attributes:
        model_name (str): Name of the model being used
        api_key (str): API key for authentication
        max_tokens (int): Maximum number of tokens for generation
        temperature (float): Sampling temperature for generation
        logger (logging.Logger): Logger instance for this client
    """
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the base LLM client.
        
        Args:
            model_name (str): Name of the model to use
            api_key (Optional[str]): API key for authentication
            max_tokens (int): Maximum tokens for generation (default: 1000)
            temperature (float): Sampling temperature (default: 0.7)
            **kwargs: Additional provider-specific parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Store additional parameters
        self.additional_params = kwargs
        
        # Initialize client-specific setup
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self) -> None:
        """
        Setup client-specific configuration and authentication.
        
        This method should be implemented by each provider to handle
        their specific setup requirements.
        """
        pass
    
    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text based on a given prompt.
        
        Args:
            prompt (str): Input prompt for text generation
            max_tokens (Optional[int]): Override default max_tokens
            temperature (Optional[float]): Override default temperature
            **kwargs: Additional generation parameters
            
        Returns:
            str: Generated text response
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a chat completion based on conversation history.
        
        Args:
            messages (List[Dict[str, str]]): List of message objects with 'role' and 'content'
            max_tokens (Optional[int]): Override default max_tokens
            temperature (Optional[float]): Override default temperature
            **kwargs: Additional completion parameters
            
        Returns:
            str: Generated response
            
        Raises:
            Exception: If completion fails
        """
        pass
    
    def validate_api_key(self) -> bool:
        """
        Validate that the API key is configured and valid.
        
        Returns:
            bool: True if API key is valid, False otherwise
        """
        if not self.api_key:
            self.logger.error("API key not configured")
            return False
        
        # Basic validation - subclasses can override for provider-specific validation
        return len(self.api_key.strip()) > 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration details
        """
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "provider": self.__class__.__name__,
            "additional_params": self.additional_params
        }
    
    def update_config(self, **kwargs) -> None:
        """
        Update model configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                self.logger.info(f"Updated {key} to {value}")
            else:
                self.additional_params[key] = value
                self.logger.info(f"Added additional parameter {key}: {value}")
    
    def _log_request(self, prompt: str, response: str, duration: float) -> None:
        """
        Log request details for monitoring and debugging.
        
        Args:
            prompt (str): Input prompt
            response (str): Generated response
            duration (float): Request duration in seconds
        """
        self.logger.info(
            f"Request completed - Model: {self.model_name}, "
            f"Duration: {duration:.2f}s, "
            f"Prompt length: {len(prompt)}, "
            f"Response length: {len(response)}"
        )
    
    def __str__(self) -> str:
        """String representation of the client."""
        return f"{self.__class__.__name__}(model={self.model_name})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the client."""
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"max_tokens={self.max_tokens}, "
            f"temperature={self.temperature})"
        )