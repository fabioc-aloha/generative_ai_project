"""
OpenAI Client Implementation

This module provides the OpenAI client implementation for the generative AI project.
It integrates with OpenAI's API to provide text generation and chat completion capabilities.

Author: Brij Kishore Pandey
"""

import os
import time
from typing import Dict, List, Optional, Any
import logging

from .base import BaseLLMClient


class OpenAIClient(BaseLLMClient):
    """
    OpenAI client implementation for GPT models.
    
    This client provides integration with OpenAI's API, supporting both text generation
    and chat completion endpoints. It includes built-in error handling, rate limiting,
    and comprehensive logging.
    
    Example:
        >>> client = OpenAIClient(
        ...     model_name="gpt-3.5-turbo",
        ...     api_key="your-api-key"
        ... )
        >>> response = client.generate_text("Hello, world!")
        >>> print(response)
    """
    
    # Default models for different use cases
    DEFAULT_MODELS = {
        "chat": "gpt-3.5-turbo",
        "text": "gpt-3.5-turbo-instruct",
        "completion": "gpt-4"
    }
    
    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        organization: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            model_name (str): OpenAI model name (default: gpt-3.5-turbo)
            api_key (Optional[str]): OpenAI API key (default: from env OPENAI_API_KEY)
            max_tokens (int): Maximum tokens for generation (default: 1000)
            temperature (float): Sampling temperature (default: 0.7)
            organization (Optional[str]): OpenAI organization ID
            **kwargs: Additional OpenAI-specific parameters
        """
        # Set API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        self.organization = organization
        
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def _setup_client(self) -> None:
        """
        Setup OpenAI client configuration and authentication.
        
        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not configured
        """
        try:
            import openai
            self.openai = openai
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        if not self.validate_api_key():
            raise ValueError(
                "OpenAI API key not configured. Set OPENAI_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        # Configure OpenAI client
        self.openai.api_key = self.api_key
        if self.organization:
            self.openai.organization = self.organization
        
        self.logger.info(f"OpenAI client initialized with model: {self.model_name}")
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI's completion endpoint.
        
        Args:
            prompt (str): Input prompt for text generation
            max_tokens (Optional[int]): Override default max_tokens
            temperature (Optional[float]): Override default temperature
            stop (Optional[List[str]]): Stop sequences for generation
            **kwargs: Additional OpenAI parameters
            
        Returns:
            str: Generated text response
            
        Raises:
            Exception: If generation fails
            
        Example:
            >>> client = OpenAIClient()
            >>> response = client.generate_text(
            ...     "Write a short story about a robot:",
            ...     max_tokens=200,
            ...     temperature=0.8
            ... )
        """
        start_time = time.time()
        
        try:
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            # Choose appropriate endpoint based on model
            if self.model_name.startswith("gpt-3.5-turbo") or self.model_name.startswith("gpt-4"):
                # Use chat completion for newer models
                messages = [{"role": "user", "content": prompt}]
                response = self.chat_completion(messages, max_tokens, temperature, **kwargs)
            else:
                # Use completion endpoint for older models
                response = self.openai.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    **kwargs
                )
                response = response.choices[0].text.strip()
            
            duration = time.time() - start_time
            self._log_request(prompt, response, duration)
            
            return response
            
        except Exception as e:
            self.logger.error(f"OpenAI text generation failed: {str(e)}")
            raise Exception(f"Text generation failed: {str(e)}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion using OpenAI's chat endpoint.
        
        Args:
            messages (List[Dict[str, str]]): Conversation history with roles and content
            max_tokens (Optional[int]): Override default max_tokens
            temperature (Optional[float]): Override default temperature
            stop (Optional[List[str]]): Stop sequences for generation
            **kwargs: Additional OpenAI parameters
            
        Returns:
            str: Generated response
            
        Raises:
            Exception: If completion fails
            
        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is machine learning?"}
            ... ]
            >>> response = client.chat_completion(messages)
        """
        start_time = time.time()
        
        try:
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            response = self.openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                **kwargs
            )
            
            response_text = response.choices[0].message.content.strip()
            duration = time.time() - start_time
            
            # Log with last user message for context
            last_user_msg = next(
                (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
                "No user message"
            )
            self._log_request(last_user_msg, response_text, duration)
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"OpenAI chat completion failed: {str(e)}")
            raise Exception(f"Chat completion failed: {str(e)}")
    
    def validate_api_key(self) -> bool:
        """
        Validate OpenAI API key by making a test request.
        
        Returns:
            bool: True if API key is valid, False otherwise
        """
        if not super().validate_api_key():
            return False
        
        try:
            # Test API key with a minimal request
            self.openai.api_key = self.api_key
            if self.organization:
                self.openai.organization = self.organization
            
            # Make a test request to verify the key
            self.openai.Model.list()
            return True
            
        except Exception as e:
            self.logger.error(f"OpenAI API key validation failed: {str(e)}")
            return False
    
    def list_available_models(self) -> List[str]:
        """
        List available OpenAI models for the current API key.
        
        Returns:
            List[str]: List of available model names
            
        Raises:
            Exception: If listing models fails
        """
        try:
            models = self.openai.Model.list()
            return [model.id for model in models.data]
        except Exception as e:
            self.logger.error(f"Failed to list OpenAI models: {str(e)}")
            raise Exception(f"Failed to list models: {str(e)}")
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for given text.
        
        Note: This is a rough estimation. For precise counting,
        use the tiktoken library with the specific model's tokenizer.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Estimated token count
        """
        # Rough estimation: ~4 characters per token
        return len(text) // 4
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current model configuration.
        
        Returns:
            Dict[str, Any]: Extended model configuration details
        """
        base_info = super().get_model_info()
        base_info.update({
            "organization": self.organization,
            "provider_specific": {
                "supports_chat": self.model_name.startswith(("gpt-3.5-turbo", "gpt-4")),
                "supports_completion": True,
                "context_window": self._get_context_window()
            }
        })
        return base_info
    
    def _get_context_window(self) -> int:
        """
        Get the context window size for the current model.
        
        Returns:
            int: Maximum context window size in tokens
        """
        context_windows = {
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "text-davinci-003": 4096,
            "text-davinci-002": 4096,
        }
        
        # Return exact match or closest match
        for model_key, window in context_windows.items():
            if self.model_name.startswith(model_key):
                return window
        
        # Default fallback
        return 4096