"""
LLM Module

This module provides language model clients and utilities for the generative AI project.
It includes implementations for OpenAI GPT and Anthropic Claude models, along with
utilities for managing multiple clients, rate limiting, and cost calculation.

Author: Brij Kishore Pandey
"""

from .base import BaseLLMClient
from .openai_client import OpenAIClient
from .claude_client import ClaudeClient
from .utils import (
    LLMManager,
    rate_limit,
    retry_with_backoff,
    estimate_tokens,
    calculate_cost,
    create_client_from_config,
    generate_prompt_hash,
    validate_message_format,
    quick_generate,
    quick_chat
)

__all__ = [
    # Base classes
    "BaseLLMClient",
    
    # Client implementations
    "OpenAIClient",
    "ClaudeClient",
    
    # Manager and utilities
    "LLMManager",
    
    # Decorators
    "rate_limit",
    "retry_with_backoff",
    
    # Utility functions
    "estimate_tokens",
    "calculate_cost",
    "create_client_from_config",
    "generate_prompt_hash",
    "validate_message_format",
    
    # Quick access functions
    "quick_generate",
    "quick_chat",
]