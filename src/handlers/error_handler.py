"""
Error Handler Module

This module provides comprehensive error handling utilities for the generative AI project.
It includes custom exceptions, error classification, retry mechanisms, and user-friendly 
error messages.

Author: Brij Kishore Pandey
"""

import sys
import traceback
import time
from typing import Dict, List, Optional, Any, Callable, Type, Union
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import logging


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors."""
    API_ERROR = "api_error"
    AUTHENTICATION_ERROR = "authentication_error" 
    RATE_LIMIT_ERROR = "rate_limit_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_ERROR = "resource_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorInfo:
    """
    Container for detailed error information.
    
    Attributes:
        error_type (str): Type of the error
        message (str): Error message
        category (ErrorCategory): Error category
        severity (ErrorSeverity): Error severity
        retry_after (Optional[float]): Seconds to wait before retrying
        context (Dict[str, Any]): Additional context about the error
        suggestions (List[str]): Suggestions for resolving the error
        timestamp (float): When the error occurred
    """
    error_type: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    retry_after: Optional[float] = None
    context: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    timestamp: float = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.context is None:
            self.context = {}
        if self.suggestions is None:
            self.suggestions = []
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "retry_after": self.retry_after,
            "context": self.context,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp
        }


# Custom Exception Classes

class GenerativeAIError(Exception):
    """Base exception for all generative AI project errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        retry_after: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestions: Optional[List[str]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message (str): Error message
            category (ErrorCategory): Error category
            severity (ErrorSeverity): Error severity
            retry_after (Optional[float]): Seconds to wait before retrying
            context (Optional[Dict[str, Any]]): Additional error context
            suggestions (Optional[List[str]]): Suggestions for resolving the error
        """
        super().__init__(message)
        self.error_info = ErrorInfo(
            error_type=self.__class__.__name__,
            message=message,
            category=category,
            severity=severity,
            retry_after=retry_after,
            context=context or {},
            suggestions=suggestions or []
        )


class APIError(GenerativeAIError):
    """Error related to API calls."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'status_code': status_code,
            'response_body': response_body
        })
        kwargs['context'] = context
        kwargs.setdefault('category', ErrorCategory.API_ERROR)
        
        super().__init__(message, **kwargs)


class AuthenticationError(GenerativeAIError):
    """Error related to authentication."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.AUTHENTICATION_ERROR)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('suggestions', [
            "Check your API key configuration",
            "Verify the API key is valid and not expired",
            "Ensure the API key has the required permissions"
        ])
        
        super().__init__(message, **kwargs)


class RateLimitError(GenerativeAIError):
    """Error related to rate limiting."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        **kwargs
    ):
        kwargs.setdefault('category', ErrorCategory.RATE_LIMIT_ERROR)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('suggestions', [
            "Reduce the request rate",
            "Implement exponential backoff",
            "Consider upgrading your API plan for higher limits"
        ])
        
        if retry_after:
            kwargs['retry_after'] = retry_after
        
        super().__init__(message, **kwargs)


class NetworkError(GenerativeAIError):
    """Error related to network connectivity."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK_ERROR)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('suggestions', [
            "Check your internet connection",
            "Verify the API endpoint is accessible",
            "Try again after a short delay"
        ])
        
        super().__init__(message, **kwargs)


class ValidationError(GenerativeAIError):
    """Error related to input validation."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        context.update({
            'field': field,
            'value': value
        })
        kwargs['context'] = context
        kwargs.setdefault('category', ErrorCategory.VALIDATION_ERROR)
        kwargs.setdefault('severity', ErrorSeverity.LOW)
        
        super().__init__(message, **kwargs)


class ConfigurationError(GenerativeAIError):
    """Error related to configuration."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION_ERROR)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('suggestions', [
            "Check your configuration files",
            "Verify all required settings are present",
            "Refer to the documentation for configuration details"
        ])
        
        super().__init__(message, **kwargs)


class ResourceError(GenerativeAIError):
    """Error related to resource availability."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RESOURCE_ERROR)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        kwargs.setdefault('suggestions', [
            "Check available system resources",
            "Consider reducing the workload",
            "Try again later when resources may be available"
        ])
        
        super().__init__(message, **kwargs)


class ErrorClassifier:
    """
    Classifier for categorizing and analyzing errors.
    
    This class analyzes exceptions and provides structured error information
    with categories, severity levels, and suggestions for resolution.
    """
    
    # Error patterns for classification
    ERROR_PATTERNS = {
        ErrorCategory.AUTHENTICATION_ERROR: [
            "unauthorized", "invalid api key", "authentication failed",
            "forbidden", "access denied", "invalid credentials"
        ],
        ErrorCategory.RATE_LIMIT_ERROR: [
            "rate limit", "too many requests", "quota exceeded",
            "throttled", "rate exceeded"
        ],
        ErrorCategory.NETWORK_ERROR: [
            "connection error", "timeout", "network", "connection refused",
            "host unreachable", "dns", "ssl error"
        ],
        ErrorCategory.VALIDATION_ERROR: [
            "validation", "invalid input", "bad request", "malformed",
            "invalid parameter", "missing required"
        ],
        ErrorCategory.CONFIGURATION_ERROR: [
            "configuration", "config", "setting", "environment",
            "missing variable", "invalid configuration"
        ],
        ErrorCategory.RESOURCE_ERROR: [
            "out of memory", "disk space", "resource", "capacity",
            "limit exceeded", "insufficient"
        ]
    }
    
    def classify_error(self, error: Exception) -> ErrorInfo:
        """
        Classify an error and return structured information.
        
        Args:
            error (Exception): Exception to classify
            
        Returns:
            ErrorInfo: Structured error information
        """
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Check if it's already a GenerativeAIError
        if isinstance(error, GenerativeAIError):
            return error.error_info
        
        # Classify based on error type
        category = self._classify_by_type(error)
        
        # If type-based classification didn't work, use message patterns
        if category == ErrorCategory.UNKNOWN_ERROR:
            category = self._classify_by_message(error_message)
        
        # Determine severity
        severity = self._determine_severity(error, category)
        
        # Get suggestions
        suggestions = self._get_suggestions(category, error)
        
        # Check for retry information
        retry_after = self._extract_retry_after(error)
        
        return ErrorInfo(
            error_type=error_type,
            message=str(error),
            category=category,
            severity=severity,
            retry_after=retry_after,
            context=self._extract_context(error),
            suggestions=suggestions
        )
    
    def _classify_by_type(self, error: Exception) -> ErrorCategory:
        """Classify error based on exception type."""
        error_type = type(error).__name__.lower()
        
        if "auth" in error_type or "permission" in error_type:
            return ErrorCategory.AUTHENTICATION_ERROR
        elif "network" in error_type or "connection" in error_type:
            return ErrorCategory.NETWORK_ERROR
        elif "validation" in error_type or "value" in error_type:
            return ErrorCategory.VALIDATION_ERROR
        elif "config" in error_type:
            return ErrorCategory.CONFIGURATION_ERROR
        elif "memory" in error_type or "resource" in error_type:
            return ErrorCategory.RESOURCE_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _classify_by_message(self, message: str) -> ErrorCategory:
        """Classify error based on message content."""
        for category, patterns in self.ERROR_PATTERNS.items():
            if any(pattern in message for pattern in patterns):
                return category
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _determine_severity(
        self,
        error: Exception,
        category: ErrorCategory
    ) -> ErrorSeverity:
        """Determine error severity."""
        # High severity categories
        if category in [ErrorCategory.AUTHENTICATION_ERROR, ErrorCategory.CONFIGURATION_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity categories
        if category in [ErrorCategory.API_ERROR, ErrorCategory.RATE_LIMIT_ERROR, 
                       ErrorCategory.NETWORK_ERROR, ErrorCategory.RESOURCE_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity categories
        if category == ErrorCategory.VALIDATION_ERROR:
            return ErrorSeverity.LOW
        
        # Default to medium
        return ErrorSeverity.MEDIUM
    
    def _get_suggestions(self, category: ErrorCategory, error: Exception) -> List[str]:
        """Get suggestions for resolving the error."""
        suggestions_map = {
            ErrorCategory.AUTHENTICATION_ERROR: [
                "Verify your API key is correct and has proper permissions",
                "Check if the API key has expired",
                "Ensure you're using the correct authentication method"
            ],
            ErrorCategory.RATE_LIMIT_ERROR: [
                "Implement exponential backoff in your retry logic",
                "Reduce the frequency of API calls",
                "Consider upgrading to a higher tier plan"
            ],
            ErrorCategory.NETWORK_ERROR: [
                "Check your internet connection",
                "Verify the API endpoint is accessible",
                "Try again after a short delay"
            ],
            ErrorCategory.VALIDATION_ERROR: [
                "Check the format and values of your input parameters",
                "Refer to the API documentation for valid input formats",
                "Validate your data before making the request"
            ],
            ErrorCategory.CONFIGURATION_ERROR: [
                "Review your configuration files for missing or incorrect values",
                "Check environment variables",
                "Refer to the setup documentation"
            ],
            ErrorCategory.RESOURCE_ERROR: [
                "Check available system resources (memory, disk space)",
                "Consider reducing the size of your request",
                "Try again when system resources are available"
            ]
        }
        
        return suggestions_map.get(category, [
            "Check the error message for specific details",
            "Refer to the documentation for troubleshooting guidance",
            "Contact support if the issue persists"
        ])
    
    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract retry-after information from the error."""
        # Check if error has retry-after information
        if hasattr(error, 'retry_after'):
            return error.retry_after
        
        # Look for retry-after in error message
        message = str(error).lower()
        import re
        
        # Look for patterns like "retry after 60 seconds"
        match = re.search(r'retry after (\d+)', message)
        if match:
            return float(match.group(1))
        
        # Look for patterns like "wait 30 seconds"
        match = re.search(r'wait (\d+)', message)
        if match:
            return float(match.group(1))
        
        return None
    
    def _extract_context(self, error: Exception) -> Dict[str, Any]:
        """Extract additional context from the error."""
        context = {}
        
        # Add traceback information
        context['traceback'] = traceback.format_exc()
        
        # Add error attributes
        for attr in dir(error):
            if not attr.startswith('_') and attr not in ['args', 'with_traceback']:
                try:
                    value = getattr(error, attr)
                    if not callable(value):
                        context[attr] = value
                except Exception:
                    pass
        
        return context


class ErrorHandler:
    """
    Comprehensive error handler with logging, classification, and recovery.
    
    This class provides centralized error handling with automatic classification,
    logging, and suggestions for error resolution.
    """
    
    def __init__(self, logger_name: str = "error_handler"):
        """
        Initialize the error handler.
        
        Args:
            logger_name (str): Name of the logger to use
        """
        self.logger = logging.getLogger(logger_name)
        self.classifier = ErrorClassifier()
        self.error_count = 0
        self.error_history: List[ErrorInfo] = []
        self.max_history = 100
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_message: bool = True
    ) -> ErrorInfo:
        """
        Handle an error with classification, logging, and user messaging.
        
        Args:
            error (Exception): Exception to handle
            context (Optional[Dict[str, Any]]): Additional context
            user_message (bool): Whether to generate user-friendly message
            
        Returns:
            ErrorInfo: Structured error information
        """
        # Classify the error
        error_info = self.classifier.classify_error(error)
        
        # Add additional context
        if context:
            error_info.context.update(context)
        
        # Log the error
        self._log_error(error_info, error)
        
        # Track error
        self._track_error(error_info)
        
        # Generate user-friendly message if requested
        if user_message:
            error_info.context['user_message'] = self._generate_user_message(error_info)
        
        return error_info
    
    def _log_error(self, error_info: ErrorInfo, original_error: Exception) -> None:
        """Log the error with appropriate level."""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_info.severity, logging.ERROR)
        
        self.logger.log(
            log_level,
            f"Error handled: {error_info.error_type}",
            extra={
                'error_type': error_info.error_type,
                'category': error_info.category.value,
                'severity': error_info.severity.value,
                'message': error_info.message,
                'retry_after': error_info.retry_after,
                'suggestions': error_info.suggestions,
                'context': error_info.context
            },
            exc_info=original_error
        )
    
    def _track_error(self, error_info: ErrorInfo) -> None:
        """Track error in history."""
        self.error_count += 1
        self.error_history.append(error_info)
        
        # Limit history size
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
    
    def _generate_user_message(self, error_info: ErrorInfo) -> str:
        """Generate user-friendly error message."""
        message = f"An error occurred: {error_info.message}"
        
        if error_info.suggestions:
            message += "\n\nSuggestions:"
            for suggestion in error_info.suggestions[:3]:  # Limit to top 3
                message += f"\n• {suggestion}"
        
        if error_info.retry_after:
            message += f"\n\nPlease wait {error_info.retry_after} seconds before retrying."
        
        return message
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics.
        
        Returns:
            Dict[str, Any]: Error statistics
        """
        if not self.error_history:
            return {"total_errors": 0}
        
        category_counts = {}
        severity_counts = {}
        
        for error_info in self.error_history:
            category = error_info.category.value
            severity = error_info.severity.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "total_errors": self.error_count,
            "recent_errors": len(self.error_history),
            "category_breakdown": category_counts,
            "severity_breakdown": severity_counts,
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            "highest_severity": max(severity_counts.items(), key=lambda x: x[1])[0] if severity_counts else None
        }


def handle_exceptions(
    handler: Optional[ErrorHandler] = None,
    reraise: bool = False,
    context: Optional[Dict[str, Any]] = None
):
    """
    Decorator for automatic exception handling.
    
    Args:
        handler (Optional[ErrorHandler]): Error handler to use
        reraise (bool): Whether to reraise the exception after handling
        context (Optional[Dict[str, Any]]): Additional context
        
    Returns:
        Callable: Decorated function
        
    Example:
        >>> @handle_exceptions(reraise=False)
        ... def risky_function():
        ...     raise ValueError("Something went wrong")
    """
    if handler is None:
        handler = ErrorHandler()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                func_context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                
                if context:
                    func_context.update(context)
                
                error_info = handler.handle_error(e, func_context)
                
                if reraise:
                    raise
                
                return error_info
        
        return wrapper
    
    return decorator


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    user_message: bool = True
) -> ErrorInfo:
    """
    Handle an error using the global error handler (convenience function).
    
    Args:
        error (Exception): Exception to handle
        context (Optional[Dict[str, Any]]): Additional context
        user_message (bool): Whether to generate user-friendly message
        
    Returns:
        ErrorInfo: Structured error information
    """
    return global_error_handler.handle_error(error, context, user_message)


def get_error_stats() -> Dict[str, Any]:
    """
    Get error statistics from the global handler (convenience function).
    
    Returns:
        Dict[str, Any]: Error statistics
    """
    return global_error_handler.get_error_stats()