"""
Utils Module

This module provides utility functions and classes for the generative AI project,
including rate limiting, token counting, caching, and logging utilities.

Author: Brij Kishore Pandey
"""

from .rate_limiter import (
    RateLimiter,
    PerKeyRateLimiter,
    AdaptiveRateLimiter,
    rate_limit,
    DEFAULT_API_LIMITER,
    AGGRESSIVE_API_LIMITER,
    CONSERVATIVE_API_LIMITER
)

from .token_counter import (
    TokenCount,
    BaseTokenCounter,
    ApproximateTokenCounter,
    RegexTokenCounter,
    TiktokenCounter,
    TokenCounterManager,
    token_manager,
    count_tokens,
    count_message_tokens,
    estimate_cost
)

from .cache import (
    CacheEntry,
    BaseCache,
    MemoryCache,
    FileCache,
    SQLiteCache,
    LLMResponseCache,
    memory_cache,
    file_cache,
    llm_cache,
    cache_llm_response,
    get_cached_llm_response
)

from .logger import (
    ColoredFormatter,
    JsonFormatter,
    LLMInteractionLogger,
    LogAggregator,
    PerformanceLogger,
    setup_logging,
    get_logger,
    log_context,
    llm_logger,
    perf_logger
)

__all__ = [
    # Rate limiting
    "RateLimiter",
    "PerKeyRateLimiter", 
    "AdaptiveRateLimiter",
    "rate_limit",
    "DEFAULT_API_LIMITER",
    "AGGRESSIVE_API_LIMITER",
    "CONSERVATIVE_API_LIMITER",
    
    # Token counting
    "TokenCount",
    "BaseTokenCounter",
    "ApproximateTokenCounter",
    "RegexTokenCounter", 
    "TiktokenCounter",
    "TokenCounterManager",
    "token_manager",
    "count_tokens",
    "count_message_tokens",
    "estimate_cost",
    
    # Caching
    "CacheEntry",
    "BaseCache",
    "MemoryCache",
    "FileCache",
    "SQLiteCache",
    "LLMResponseCache",
    "memory_cache",
    "file_cache",
    "llm_cache",
    "cache_llm_response",
    "get_cached_llm_response",
    
    # Logging
    "ColoredFormatter",
    "JsonFormatter",
    "LLMInteractionLogger",
    "LogAggregator",
    "PerformanceLogger",
    "setup_logging",
    "get_logger",
    "log_context",
    "llm_logger",
    "perf_logger",
]