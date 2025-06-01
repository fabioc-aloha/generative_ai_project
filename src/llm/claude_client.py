"""
Claude Client Implementation

This module provides the Anthropic Claude client implementation for the generative AI project.
It integrates with Anthropic's API to provide text generation and chat completion capabilities.

Author: Brij Kishore Pandey
"""

import os
import time
from typing import Dict, List, Optional, Any
import logging

from .base import BaseLLMClient


class ClaudeClient(BaseLLMClient):
    """
    Anthropic Claude client implementation.
    
    This client provides integration with Anthropic's Claude API, supporting both text
    generation and chat completion. It includes built-in error handling, rate limiting,
    and comprehensive logging.
    
    Example:
        >>> client = ClaudeClient(
        ...     model_name="claude-3-sonnet-20240229",
        ...     api_key="your-anthropic-api-key"
        ... )
        >>> response = client.generate_text("Hello, world!")
        >>> print(response)
    """
    
    # Default models for different use cases
    DEFAULT_MODELS = {
        "chat": "claude-3-sonnet-20240229",
        "text": "claude-3-haiku-20240307",
        "reasoning": "claude-3-opus-20240229"
    }
    
    def __init__(
        self,
        model_name: str = "claude-3-sonnet-20240229",
        api_key: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Claude client.
        
        Args:
            model_name (str): Claude model name (default: claude-3-sonnet-20240229)
            api_key (Optional[str]): Anthropic API key (default: from env ANTHROPIC_API_KEY)
            max_tokens (int): Maximum tokens for generation (default: 1000)
            temperature (float): Sampling temperature (default: 0.7)
            **kwargs: Additional Claude-specific parameters
        """
        # Set API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def _setup_client(self) -> None:
        """
        Setup Claude client configuration and authentication.
        
        Raises:
            ImportError: If anthropic package is not installed
            ValueError: If API key is not configured
        """
        try:
            import anthropic
            self.anthropic = anthropic
        except ImportError:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )
        
        if not self.validate_api_key():
            raise ValueError(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment "
                "variable or pass api_key parameter."
            )
        
        # Configure Anthropic client
        self.client = self.anthropic.Anthropic(api_key=self.api_key)
        
        self.logger.info(f"Claude client initialized with model: {self.model_name}")
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text using Claude's messages API.
        
        Args:
            prompt (str): Input prompt for text generation
            max_tokens (Optional[int]): Override default max_tokens
            temperature (Optional[float]): Override default temperature
            system_prompt (Optional[str]): System prompt to guide behavior
            **kwargs: Additional Claude parameters
            
        Returns:
            str: Generated text response
            
        Raises:
            Exception: If generation fails
            
        Example:
            >>> client = ClaudeClient()
            >>> response = client.generate_text(
            ...     "Write a short story about a robot:",
            ...     max_tokens=200,
            ...     temperature=0.8,
            ...     system_prompt="You are a creative writer."
            ... )
        """
        start_time = time.time()
        
        try:
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt
            
            response = self.client.messages.create(**request_params)
            
            # Extract text from response
            response_text = response.content[0].text
            
            duration = time.time() - start_time
            self._log_request(prompt, response_text, duration)
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Claude text generation failed: {str(e)}")
            raise Exception(f"Text generation failed: {str(e)}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate chat completion using Claude's messages API.
        
        Args:
            messages (List[Dict[str, str]]): Conversation history with roles and content
            max_tokens (Optional[int]): Override default max_tokens
            temperature (Optional[float]): Override default temperature
            system_prompt (Optional[str]): System prompt to guide behavior
            **kwargs: Additional Claude parameters
            
        Returns:
            str: Generated response
            
        Raises:
            Exception: If completion fails
            
        Example:
            >>> messages = [
            ...     {"role": "user", "content": "What is machine learning?"},
            ...     {"role": "assistant", "content": "Machine learning is..."},
            ...     {"role": "user", "content": "Can you give an example?"}
            ... ]
            >>> response = client.chat_completion(
            ...     messages,
            ...     system_prompt="You are a helpful AI tutor."
            ... )
        """
        start_time = time.time()
        
        try:
            # Use provided parameters or defaults
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature
            
            # Filter out system messages for Claude (handled separately)
            claude_messages = []
            system_from_messages = None
            
            for msg in messages:
                if msg["role"] == "system":
                    system_from_messages = msg["content"]
                else:
                    claude_messages.append(msg)
            
            # Use system prompt from parameter or extracted from messages
            final_system = system_prompt or system_from_messages
            
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": claude_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            # Add system prompt if available
            if final_system:
                request_params["system"] = final_system
            
            response = self.client.messages.create(**request_params)
            
            # Extract text from response
            response_text = response.content[0].text
            
            duration = time.time() - start_time
            
            # Log with last user message for context
            last_user_msg = next(
                (msg["content"] for msg in reversed(claude_messages) if msg["role"] == "user"),
                "No user message"
            )
            self._log_request(last_user_msg, response_text, duration)
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Claude chat completion failed: {str(e)}")
            raise Exception(f"Chat completion failed: {str(e)}")
    
    def validate_api_key(self) -> bool:
        """
        Validate Anthropic API key format and basic connectivity.
        
        Returns:
            bool: True if API key appears valid, False otherwise
        """
        if not super().validate_api_key():
            return False
        
        # Check API key format (Anthropic keys start with 'sk-ant-')
        if not self.api_key.startswith('sk-ant-'):
            self.logger.error("Invalid Anthropic API key format")
            return False
        
        return True
    
    def get_token_count(self, text: str) -> int:
        """
        Estimate token count for given text.
        
        Note: This is a rough estimation. Claude uses a different tokenizer
        than OpenAI, so this is an approximation.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Estimated token count
        """
        # Rough estimation for Claude: ~3.5 characters per token
        return len(text) // 3.5
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current Claude model configuration.
        
        Returns:
            Dict[str, Any]: Extended model configuration details
        """
        base_info = super().get_model_info()
        base_info.update({
            "provider_specific": {
                "supports_system_prompts": True,
                "supports_function_calling": self._supports_function_calling(),
                "context_window": self._get_context_window(),
                "model_family": self._get_model_family()
            }
        })
        return base_info
    
    def _supports_function_calling(self) -> bool:
        """
        Check if the current model supports function calling.
        
        Returns:
            bool: True if model supports function calling
        """
        # Claude-3 models support function calling
        return self.model_name.startswith("claude-3")
    
    def _get_model_family(self) -> str:
        """
        Get the model family (e.g., claude-3, claude-2).
        
        Returns:
            str: Model family name
        """
        if self.model_name.startswith("claude-3"):
            if "opus" in self.model_name:
                return "claude-3-opus"
            elif "sonnet" in self.model_name:
                return "claude-3-sonnet"
            elif "haiku" in self.model_name:
                return "claude-3-haiku"
            else:
                return "claude-3"
        elif self.model_name.startswith("claude-2"):
            return "claude-2"
        else:
            return "unknown"
    
    def _get_context_window(self) -> int:
        """
        Get the context window size for the current model.
        
        Returns:
            int: Maximum context window size in tokens
        """
        # Claude-3 models have 200k context window
        if self.model_name.startswith("claude-3"):
            return 200000
        # Claude-2 models have 100k context window
        elif self.model_name.startswith("claude-2"):
            return 100000
        else:
            # Conservative default
            return 100000
    
    def function_calling(
        self,
        messages: List[Dict[str, str]],
        functions: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform function calling with Claude (if supported).
        
        Args:
            messages (List[Dict[str, str]]): Conversation history
            functions (List[Dict[str, Any]]): Available functions
            max_tokens (Optional[int]): Override default max_tokens
            temperature (Optional[float]): Override default temperature
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Response with potential function calls
            
        Raises:
            NotImplementedError: If model doesn't support function calling
        """
        if not self._supports_function_calling():
            raise NotImplementedError(
                f"Function calling not supported for model: {self.model_name}"
            )
        
        # Note: This is a placeholder for Claude's function calling implementation
        # The actual implementation would depend on Claude's specific API for tools/functions
        self.logger.warning(
            "Function calling implementation is model-specific and may require "
            "additional setup based on Claude's latest API features."
        )
        
        # For now, return a standard chat completion
        response = self.chat_completion(messages, max_tokens, temperature, **kwargs)
        return {"response": response, "function_calls": None}