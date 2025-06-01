"""
Token Counter Module

This module provides utilities for counting tokens in text for various LLM models.
It supports both approximate counting and exact counting using model-specific tokenizers.

Author: Brij Kishore Pandey
"""

import re
import json
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class TokenCount:
    """
    Container for token count information.
    
    Attributes:
        input_tokens (int): Number of input tokens
        output_tokens (int): Number of output tokens (if applicable)
        total_tokens (int): Total number of tokens
        model_name (str): Name of the model used for counting
        method (str): Method used for counting ("exact", "approximate")
        metadata (Dict[str, Any]): Additional metadata about the counting
    """
    input_tokens: int
    output_tokens: int = 0
    total_tokens: int = 0
    model_name: str = ""
    method: str = "approximate"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Calculate total tokens and initialize metadata."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens
        
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "method": self.method,
            "metadata": self.metadata
        }


class BaseTokenCounter(ABC):
    """
    Abstract base class for token counters.
    
    This class defines the interface that all token counters should implement.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the token counter.
        
        Args:
            model_name (str): Name of the model for token counting
        """
        self.model_name = model_name
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{model_name}")
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in the given text.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Number of tokens
        """
        pass
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> TokenCount:
        """
        Count tokens in a list of messages (for chat models).
        
        Args:
            messages (List[Dict[str, str]]): List of message objects
            
        Returns:
            TokenCount: Token count information
        """
        total_tokens = 0
        
        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "")
            
            # Count content tokens
            content_tokens = self.count_tokens(content)
            
            # Add overhead for role and message formatting
            # This is a rough estimate and varies by model
            overhead_tokens = len(role) // 4 + 3  # Role tokens + formatting
            
            total_tokens += content_tokens + overhead_tokens
        
        return TokenCount(
            input_tokens=total_tokens,
            model_name=self.model_name,
            method=self.get_method_name(),
            metadata={"message_count": len(messages)}
        )
    
    def count_prompt_and_completion(
        self,
        prompt: str,
        completion: str = ""
    ) -> TokenCount:
        """
        Count tokens for prompt and completion.
        
        Args:
            prompt (str): Input prompt
            completion (str): Generated completion
            
        Returns:
            TokenCount: Token count information
        """
        input_tokens = self.count_tokens(prompt)
        output_tokens = self.count_tokens(completion) if completion else 0
        
        return TokenCount(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model_name=self.model_name,
            method=self.get_method_name()
        )
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Get the name of the counting method."""
        pass


class ApproximateTokenCounter(BaseTokenCounter):
    """
    Approximate token counter that uses heuristics for fast token estimation.
    
    This counter provides quick estimates without requiring model-specific tokenizers.
    The accuracy varies by model and text type but is generally good enough for
    cost estimation and rate limiting.
    """
    
    # Token ratio estimates for different models (characters per token)
    MODEL_RATIOS = {
        # OpenAI models
        "gpt-4": 3.8,
        "gpt-3.5-turbo": 4.0,
        "text-davinci-003": 4.0,
        "text-davinci-002": 4.0,
        "text-curie-001": 4.2,
        "text-babbage-001": 4.5,
        "text-ada-001": 4.8,
        
        # Anthropic models
        "claude-3-opus": 3.5,
        "claude-3-sonnet": 3.5,
        "claude-3-haiku": 3.5,
        "claude-2": 3.5,
        "claude-instant": 3.5,
        
        # Default fallback
        "default": 4.0
    }
    
    def __init__(self, model_name: str):
        """
        Initialize the approximate token counter.
        
        Args:
            model_name (str): Name of the model
        """
        super().__init__(model_name)
        self.char_per_token = self._get_ratio_for_model(model_name)
    
    def _get_ratio_for_model(self, model_name: str) -> float:
        """Get the character-to-token ratio for the model."""
        # Try exact match first
        if model_name in self.MODEL_RATIOS:
            return self.MODEL_RATIOS[model_name]
        
        # Try partial matches
        for model_key, ratio in self.MODEL_RATIOS.items():
            if model_name.startswith(model_key):
                return ratio
        
        # Use default
        return self.MODEL_RATIOS["default"]
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using character-based approximation.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Estimated number of tokens
        """
        if not text:
            return 0
        
        # Basic character count
        char_count = len(text)
        
        # Adjust for whitespace (tokens often break on whitespace)
        word_count = len(text.split())
        
        # Use a hybrid approach: character-based with word-based adjustment
        char_based_estimate = char_count / self.char_per_token
        word_based_estimate = word_count * 1.3  # Average 1.3 tokens per word
        
        # Take the average of both estimates, with bias toward character-based
        estimated_tokens = (char_based_estimate * 0.7 + word_based_estimate * 0.3)
        
        return max(1, int(round(estimated_tokens)))
    
    def get_method_name(self) -> str:
        """Get the method name."""
        return "approximate"


class RegexTokenCounter(BaseTokenCounter):
    """
    Token counter that uses regex patterns to split text into token-like units.
    
    This provides better accuracy than simple character counting by attempting
    to mimic how tokenizers split text.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the regex token counter.
        
        Args:
            model_name (str): Name of the model
        """
        super().__init__(model_name)
        
        # Pattern for splitting text into token-like units
        # This is a simplified version of what actual tokenizers do
        self.token_pattern = re.compile(
            r"""
            \w+                     # Whole words
            |[^\w\s]                # Single punctuation
            |\s+                    # Whitespace (will be filtered out)
            """,
            re.VERBOSE
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using regex pattern matching.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Estimated number of tokens
        """
        if not text:
            return 0
        
        # Find all token-like patterns
        matches = self.token_pattern.findall(text)
        
        # Filter out whitespace-only matches
        tokens = [match for match in matches if not match.isspace()]
        
        # Adjust for subword tokenization (many models split words further)
        total_chars = sum(len(token) for token in tokens)
        if total_chars > 0:
            # Estimate subword splits: longer tokens are more likely to be split
            estimated_splits = sum(max(1, len(token) // 4) for token in tokens)
            return max(len(tokens), estimated_splits)
        
        return len(tokens)
    
    def get_method_name(self) -> str:
        """Get the method name."""
        return "regex"


class TiktokenCounter(BaseTokenCounter):
    """
    Exact token counter using the tiktoken library for OpenAI models.
    
    This provides precise token counts but requires the tiktoken library
    to be installed and only works with supported OpenAI models.
    """
    
    # Model to encoding mapping
    MODEL_ENCODINGS = {
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "text-davinci-003": "p50k_base",
        "text-davinci-002": "p50k_base",
        "text-curie-001": "r50k_base",
        "text-babbage-001": "r50k_base",
        "text-ada-001": "r50k_base",
    }
    
    def __init__(self, model_name: str):
        """
        Initialize the tiktoken counter.
        
        Args:
            model_name (str): Name of the OpenAI model
            
        Raises:
            ImportError: If tiktoken is not installed
            ValueError: If model is not supported
        """
        super().__init__(model_name)
        
        try:
            import tiktoken
            self.tiktoken = tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken library is required for exact token counting. "
                "Install with: pip install tiktoken"
            )
        
        # Get encoding for model
        encoding_name = self._get_encoding_for_model(model_name)
        if not encoding_name:
            raise ValueError(f"Model '{model_name}' not supported by tiktoken")
        
        self.encoding = tiktoken.get_encoding(encoding_name)
    
    def _get_encoding_for_model(self, model_name: str) -> Optional[str]:
        """Get the encoding name for the model."""
        # Try exact match
        if model_name in self.MODEL_ENCODINGS:
            return self.MODEL_ENCODINGS[model_name]
        
        # Try partial matches
        for model_key, encoding in self.MODEL_ENCODINGS.items():
            if model_name.startswith(model_key):
                return encoding
        
        return None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken encoding.
        
        Args:
            text (str): Text to count tokens for
            
        Returns:
            int: Exact number of tokens
        """
        if not text:
            return 0
        
        try:
            tokens = self.encoding.encode(text)
            return len(tokens)
        except Exception as e:
            self.logger.error(f"Token counting failed: {str(e)}")
            # Fallback to approximate counting
            fallback_counter = ApproximateTokenCounter(self.model_name)
            return fallback_counter.count_tokens(text)
    
    def count_message_tokens(self, messages: List[Dict[str, str]]) -> TokenCount:
        """
        Count tokens in messages using OpenAI's specific formatting.
        
        Args:
            messages (List[Dict[str, str]]): List of message objects
            
        Returns:
            TokenCount: Exact token count information
        """
        try:
            # OpenAI's method for counting chat tokens
            tokens_per_message = 3  # Message overhead
            tokens_per_name = 1  # Name field overhead
            
            num_tokens = 0
            
            for message in messages:
                num_tokens += tokens_per_message
                for key, value in message.items():
                    num_tokens += len(self.encoding.encode(value))
                    if key == "name":
                        num_tokens += tokens_per_name
            
            num_tokens += 3  # Assistant response priming tokens
            
            return TokenCount(
                input_tokens=num_tokens,
                model_name=self.model_name,
                method="exact",
                metadata={
                    "message_count": len(messages),
                    "encoding": self.encoding.name
                }
            )
            
        except Exception as e:
            self.logger.error(f"Message token counting failed: {str(e)}")
            return super().count_message_tokens(messages)
    
    def get_method_name(self) -> str:
        """Get the method name."""
        return "exact"


class TokenCounterManager:
    """
    Manager for different token counting strategies.
    
    This class automatically selects the best available token counter
    for a given model and provides fallback options.
    """
    
    def __init__(self):
        """Initialize the token counter manager."""
        self.counters: Dict[str, BaseTokenCounter] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_counter(
        self,
        model_name: str,
        prefer_exact: bool = True,
        counter_type: Optional[str] = None
    ) -> BaseTokenCounter:
        """
        Get the best available token counter for a model.
        
        Args:
            model_name (str): Name of the model
            prefer_exact (bool): Whether to prefer exact counting when available
            counter_type (Optional[str]): Force specific counter type
            
        Returns:
            BaseTokenCounter: Token counter instance
        """
        cache_key = f"{model_name}_{counter_type or 'auto'}"
        
        if cache_key in self.counters:
            return self.counters[cache_key]
        
        # Create appropriate counter
        if counter_type == "approximate":
            counter = ApproximateTokenCounter(model_name)
        elif counter_type == "regex":
            counter = RegexTokenCounter(model_name)
        elif counter_type == "tiktoken":
            counter = TiktokenCounter(model_name)
        else:
            # Auto-select best counter
            counter = self._create_best_counter(model_name, prefer_exact)
        
        self.counters[cache_key] = counter
        return counter
    
    def _create_best_counter(
        self,
        model_name: str,
        prefer_exact: bool
    ) -> BaseTokenCounter:
        """Create the best available counter for the model."""
        
        # Try exact counting for OpenAI models if preferred
        if prefer_exact and self._is_openai_model(model_name):
            try:
                return TiktokenCounter(model_name)
            except (ImportError, ValueError) as e:
                self.logger.warning(f"Exact counting not available: {str(e)}")
        
        # Use regex counter as default (better than approximate)
        try:
            return RegexTokenCounter(model_name)
        except Exception:
            # Fallback to approximate
            return ApproximateTokenCounter(model_name)
    
    def _is_openai_model(self, model_name: str) -> bool:
        """Check if the model is an OpenAI model."""
        openai_prefixes = ["gpt", "text-", "code-", "davinci", "curie", "babbage", "ada"]
        return any(model_name.startswith(prefix) for prefix in openai_prefixes)
    
    def count_tokens(
        self,
        text: str,
        model_name: str,
        **kwargs
    ) -> int:
        """
        Count tokens in text using the best available counter.
        
        Args:
            text (str): Text to count tokens for
            model_name (str): Name of the model
            **kwargs: Additional arguments for counter selection
            
        Returns:
            int: Number of tokens
        """
        counter = self.get_counter(model_name, **kwargs)
        return counter.count_tokens(text)
    
    def count_message_tokens(
        self,
        messages: List[Dict[str, str]],
        model_name: str,
        **kwargs
    ) -> TokenCount:
        """
        Count tokens in messages using the best available counter.
        
        Args:
            messages (List[Dict[str, str]]): List of message objects
            model_name (str): Name of the model
            **kwargs: Additional arguments for counter selection
            
        Returns:
            TokenCount: Token count information
        """
        counter = self.get_counter(model_name, **kwargs)
        return counter.count_message_tokens(messages)
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: str
    ) -> float:
        """
        Estimate cost based on token counts.
        
        Args:
            input_tokens (int): Number of input tokens
            output_tokens (int): Number of output tokens
            model_name (str): Name of the model
            
        Returns:
            float: Estimated cost in USD
        """
        # Pricing data (approximate, as of 2024)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        }
        
        # Find pricing for model
        model_pricing = None
        for model_key, prices in pricing.items():
            if model_name.startswith(model_key):
                model_pricing = prices
                break
        
        if not model_pricing:
            self.logger.warning(f"No pricing data for model: {model_name}")
            return 0.0
        
        # Calculate cost (pricing is per 1K tokens)
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        
        return input_cost + output_cost
    
    def clear_cache(self) -> None:
        """Clear the counter cache."""
        self.counters.clear()
        self.logger.info("Cleared token counter cache")


# Global token counter manager instance
token_manager = TokenCounterManager()


# Convenience functions
def count_tokens(
    text: str,
    model_name: str = "gpt-3.5-turbo",
    **kwargs
) -> int:
    """
    Count tokens in text (convenience function).
    
    Args:
        text (str): Text to count tokens for
        model_name (str): Name of the model (default: gpt-3.5-turbo)
        **kwargs: Additional arguments
        
    Returns:
        int: Number of tokens
    """
    return token_manager.count_tokens(text, model_name, **kwargs)


def count_message_tokens(
    messages: List[Dict[str, str]],
    model_name: str = "gpt-3.5-turbo",
    **kwargs
) -> TokenCount:
    """
    Count tokens in messages (convenience function).
    
    Args:
        messages (List[Dict[str, str]]): List of message objects
        model_name (str): Name of the model (default: gpt-3.5-turbo)
        **kwargs: Additional arguments
        
    Returns:
        TokenCount: Token count information
    """
    return token_manager.count_message_tokens(messages, model_name, **kwargs)


def estimate_cost(
    input_tokens: int,
    output_tokens: int = 0,
    model_name: str = "gpt-3.5-turbo"
) -> float:
    """
    Estimate cost based on token counts (convenience function).
    
    Args:
        input_tokens (int): Number of input tokens
        output_tokens (int): Number of output tokens (default: 0)
        model_name (str): Name of the model (default: gpt-3.5-turbo)
        
    Returns:
        float: Estimated cost in USD
    """
    return token_manager.estimate_cost(input_tokens, output_tokens, model_name)