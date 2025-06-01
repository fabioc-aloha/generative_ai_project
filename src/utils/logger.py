"""
Logger Module

This module provides comprehensive logging utilities for the generative AI project.
It supports structured logging, multiple output formats, and integration with
various monitoring systems.

Author: Brij Kishore Pandey
"""

import os
import sys
import json
import logging
import logging.handlers
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import threading
from contextlib import contextmanager


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels for console output.
    
    This formatter makes log output more readable in terminals that support
    ANSI color codes.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        """Format the log record with colors."""
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        
        # Apply color to the entire message for better visibility
        original_msg = record.getMessage()
        record.msg = f"{log_color}{original_msg}{self.RESET}"
        record.args = ()
        
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    This formatter outputs log records as JSON objects, making them
    easier to parse and analyze with log aggregation tools.
    """
    
    def format(self, record):
        """Format the log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class LLMInteractionLogger:
    """
    Specialized logger for LLM interactions.
    
    This logger captures detailed information about LLM requests and responses,
    including tokens, costs, and performance metrics.
    """
    
    def __init__(self, logger_name: str = "llm_interactions"):
        """
        Initialize the LLM interaction logger.
        
        Args:
            logger_name (str): Name of the logger
        """
        self.logger = logging.getLogger(logger_name)
        self.session_id = self._generate_session_id()
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def log_request(
        self,
        model: str,
        prompt: str,
        parameters: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """
        Log an LLM request.
        
        Args:
            model (str): Model name
            prompt (str): Input prompt
            parameters (Dict[str, Any]): Request parameters
            user_id (Optional[str]): User identifier
            session_id (Optional[str]): Session identifier
            
        Returns:
            str: Request ID for correlation
        """
        import uuid
        request_id = str(uuid.uuid4())
        
        self.logger.info(
            "LLM Request",
            extra={
                'event_type': 'llm_request',
                'request_id': request_id,
                'session_id': session_id or self.session_id,
                'user_id': user_id,
                'model': model,
                'prompt_length': len(prompt),
                'prompt_preview': prompt[:100] + '...' if len(prompt) > 100 else prompt,
                'parameters': parameters,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        return request_id
    
    def log_response(
        self,
        request_id: str,
        response: str,
        model: str,
        tokens_used: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None,
        duration: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log an LLM response.
        
        Args:
            request_id (str): Request ID for correlation
            response (str): Generated response
            model (str): Model name
            tokens_used (Optional[Dict[str, int]]): Token usage information
            cost (Optional[float]): Request cost in USD
            duration (Optional[float]): Request duration in seconds
            error (Optional[str]): Error message if request failed
        """
        log_level = logging.ERROR if error else logging.INFO
        
        self.logger.log(
            log_level,
            "LLM Response",
            extra={
                'event_type': 'llm_response',
                'request_id': request_id,
                'model': model,
                'response_length': len(response) if response else 0,
                'response_preview': (response[:100] + '...' if len(response) > 100 else response) if response else None,
                'tokens_used': tokens_used,
                'cost': cost,
                'duration': duration,
                'error': error,
                'success': error is None,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_cache_hit(
        self,
        model: str,
        prompt_hash: str,
        response_preview: str
    ) -> None:
        """
        Log a cache hit.
        
        Args:
            model (str): Model name
            prompt_hash (str): Hash of the prompt
            response_preview (str): Preview of cached response
        """
        self.logger.info(
            "Cache Hit",
            extra={
                'event_type': 'cache_hit',
                'model': model,
                'prompt_hash': prompt_hash,
                'response_preview': response_preview[:100] + '...' if len(response_preview) > 100 else response_preview,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_rate_limit(
        self,
        model: str,
        wait_time: float,
        reason: str = "Rate limit exceeded"
    ) -> None:
        """
        Log a rate limit event.
        
        Args:
            model (str): Model name
            wait_time (float): Time to wait in seconds
            reason (str): Reason for rate limiting
        """
        self.logger.warning(
            "Rate Limited",
            extra={
                'event_type': 'rate_limit',
                'model': model,
                'wait_time': wait_time,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
        )


class LogAggregator:
    """
    Log aggregator for collecting and analyzing log data.
    
    This class provides utilities for aggregating log data and generating
    reports about system usage and performance.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize the log aggregator.
        
        Args:
            log_file (Optional[str]): Path to log file to analyze
        """
        self.log_file = log_file
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'rate_limits': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'models_used': {},
            'users_active': set(),
            'errors': {}
        }
    
    def parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single log line.
        
        Args:
            line (str): Log line to parse
            
        Returns:
            Optional[Dict[str, Any]]: Parsed log data or None if invalid
        """
        try:
            # Try to parse as JSON first
            return json.loads(line.strip())
        except json.JSONDecodeError:
            # Try to parse as standard log format
            try:
                # Basic parsing for standard format
                parts = line.split(' - ', 2)
                if len(parts) >= 3:
                    return {
                        'timestamp': parts[0],
                        'level': parts[1],
                        'message': parts[2]
                    }
            except Exception:
                pass
        
        return None
    
    def analyze_logs(self) -> Dict[str, Any]:
        """
        Analyze logs and generate statistics.
        
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not self.log_file or not os.path.exists(self.log_file):
            return {'error': 'Log file not found'}
        
        # Reset stats
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'cache_hits': 0,
            'rate_limits': 0,
            'total_tokens': 0,
            'total_cost': 0.0,
            'models_used': {},
            'users_active': set(),
            'errors': {}
        }
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    log_entry = self.parse_log_line(line)
                    if log_entry:
                        self._process_log_entry(log_entry)
        
        except Exception as e:
            return {'error': f'Failed to analyze logs: {str(e)}'}
        
        # Convert sets to lists for JSON serialization
        result = dict(self.stats)
        result['users_active'] = len(self.stats['users_active'])
        
        return result
    
    def _process_log_entry(self, entry: Dict[str, Any]) -> None:
        """Process a single log entry for statistics."""
        event_type = entry.get('event_type', '')
        
        if event_type == 'llm_request':
            self.stats['total_requests'] += 1
            
            model = entry.get('model', 'unknown')
            self.stats['models_used'][model] = self.stats['models_used'].get(model, 0) + 1
            
            user_id = entry.get('user_id')
            if user_id:
                self.stats['users_active'].add(user_id)
        
        elif event_type == 'llm_response':
            if entry.get('success', True):
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
                
                error = entry.get('error', 'Unknown error')
                self.stats['errors'][error] = self.stats['errors'].get(error, 0) + 1
            
            # Aggregate tokens and cost
            tokens_used = entry.get('tokens_used', {})
            if isinstance(tokens_used, dict):
                self.stats['total_tokens'] += sum(tokens_used.values())
            
            cost = entry.get('cost', 0)
            if cost:
                self.stats['total_cost'] += cost
        
        elif event_type == 'cache_hit':
            self.stats['cache_hits'] += 1
        
        elif event_type == 'rate_limit':
            self.stats['rate_limits'] += 1
    
    def generate_report(self) -> str:
        """
        Generate a human-readable report.
        
        Returns:
            str: Formatted report
        """
        stats = self.analyze_logs()
        
        if 'error' in stats:
            return f"Error generating report: {stats['error']}"
        
        report = []
        report.append("=== LLM Usage Report ===")
        report.append(f"Total Requests: {stats['total_requests']}")
        report.append(f"Successful: {stats['successful_requests']}")
        report.append(f"Failed: {stats['failed_requests']}")
        report.append(f"Cache Hits: {stats['cache_hits']}")
        report.append(f"Rate Limits: {stats['rate_limits']}")
        report.append(f"Active Users: {stats['users_active']}")
        report.append(f"Total Tokens: {stats['total_tokens']:,}")
        report.append(f"Total Cost: ${stats['total_cost']:.4f}")
        
        if stats['models_used']:
            report.append("\nModels Used:")
            for model, count in sorted(stats['models_used'].items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {model}: {count} requests")
        
        if stats['errors']:
            report.append("\nErrors:")
            for error, count in sorted(stats['errors'].items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {error}: {count} occurrences")
        
        return "\n".join(report)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    console_colors: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (Optional[str]): Path to log file
        json_format (bool): Whether to use JSON format for file logging
        console_colors (bool): Whether to use colors in console output
        max_file_size (int): Maximum size of log file before rotation
        backup_count (int): Number of backup files to keep
    """
    # Convert level string to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    
    if console_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        
        if json_format:
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Log setup completion
    logging.info(f"Logging configured - Level: {level}, File: {log_file}, JSON: {json_format}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)


@contextmanager
def log_context(**kwargs):
    """
    Context manager for adding context to all log messages within a block.
    
    Args:
        **kwargs: Context variables to add to log messages
        
    Example:
        >>> with log_context(user_id="123", session_id="abc"):
        ...     logger.info("User action performed")
    """
    # Store old factory
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **factory_kwargs):
        record = old_factory(*args, **factory_kwargs)
        # Add context to record
        for key, value in kwargs.items():
            setattr(record, key, value)
        return record
    
    # Set new factory
    logging.setLogRecordFactory(record_factory)
    
    try:
        yield
    finally:
        # Restore old factory
        logging.setLogRecordFactory(old_factory)


class PerformanceLogger:
    """
    Performance logging utility for measuring execution time.
    
    This class provides decorators and context managers for measuring
    and logging the performance of functions and code blocks.
    """
    
    def __init__(self, logger_name: str = "performance"):
        """
        Initialize the performance logger.
        
        Args:
            logger_name (str): Name of the logger
        """
        self.logger = logging.getLogger(logger_name)
    
    @contextmanager
    def measure(self, operation_name: str, **context):
        """
        Context manager for measuring operation time.
        
        Args:
            operation_name (str): Name of the operation
            **context: Additional context to log
            
        Example:
            >>> perf_logger = PerformanceLogger()
            >>> with perf_logger.measure("api_call", user_id="123"):
            ...     # Perform operation
            ...     time.sleep(1)
        """
        import time
        start_time = time.time()
        
        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            duration = time.time() - start_time
            
            self.logger.info(
                f"Operation completed: {operation_name}",
                extra={
                    'event_type': 'performance',
                    'operation': operation_name,
                    'duration': duration,
                    'success': success,
                    'error': error,
                    **context
                }
            )
    
    def log_function_performance(self, func_name: str = None):
        """
        Decorator for logging function performance.
        
        Args:
            func_name (str, optional): Custom name for the function
            
        Returns:
            Callable: Decorated function
            
        Example:
            >>> perf_logger = PerformanceLogger()
            >>> @perf_logger.log_function_performance()
            ... def slow_function():
            ...     time.sleep(1)
            ...     return "result"
        """
        def decorator(func):
            import time
            from functools import wraps
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                operation_name = func_name or f"{func.__module__}.{func.__name__}"
                
                with self.measure(operation_name):
                    return func(*args, **kwargs)
            
            return wrapper
        
        return decorator


# Global instances
llm_logger = LLMInteractionLogger()
perf_logger = PerformanceLogger()

# Default logging setup
if not logging.getLogger().handlers:
    setup_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE"),
        json_format=os.getenv("LOG_JSON", "false").lower() == "true"
    )