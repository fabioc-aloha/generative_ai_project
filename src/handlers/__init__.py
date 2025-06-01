"""
Handlers Module

This module provides error handling and other handler utilities for the generative AI project.

Author: Brij Kishore Pandey
"""

from .error_handler import (
    ErrorSeverity,
    ErrorCategory,
    ErrorInfo,
    GenerativeAIError,
    APIError,
    AuthenticationError,
    RateLimitError,
    NetworkError,
    ValidationError,
    ConfigurationError,
    ResourceError,
    ErrorClassifier,
    ErrorHandler,
    handle_exceptions,
    global_error_handler,
    handle_error,
    get_error_stats
)

__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorCategory",
    
    # Data structures
    "ErrorInfo",
    
    # Exceptions
    "GenerativeAIError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "NetworkError",
    "ValidationError",
    "ConfigurationError",
    "ResourceError",
    
    # Classes
    "ErrorClassifier",
    "ErrorHandler",
    
    # Decorators
    "handle_exceptions",
    
    # Global instances and functions
    "global_error_handler",
    "handle_error",
    "get_error_stats",
]