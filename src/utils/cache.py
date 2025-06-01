"""
Cache Module

This module provides caching utilities for storing and retrieving LLM responses
to avoid repeated API calls and improve performance. It supports multiple
cache backends and automatic cache management.

Author: Brij Kishore Pandey
"""

import os
import json
import hashlib
import pickle
import sqlite3
import time
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
import threading
import logging


logger = logging.getLogger(__name__)


class CacheEntry:
    """
    Represents a single cache entry with metadata.
    
    Attributes:
        key (str): Cache key
        value (Any): Cached value
        created_at (float): Timestamp when entry was created
        accessed_at (float): Timestamp when entry was last accessed
        ttl (Optional[float]): Time to live in seconds
        metadata (Dict[str, Any]): Additional metadata
    """
    
    def __init__(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a cache entry.
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
            ttl (Optional[float]): Time to live in seconds
            metadata (Optional[Dict[str, Any]]): Additional metadata
        """
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.accessed_at = self.created_at
        self.ttl = ttl
        self.metadata = metadata or {}
    
    def is_expired(self) -> bool:
        """
        Check if the cache entry has expired.
        
        Returns:
            bool: True if expired, False otherwise
        """
        if self.ttl is None:
            return False
        
        return time.time() > (self.created_at + self.ttl)
    
    def touch(self) -> None:
        """Update the accessed timestamp."""
        self.accessed_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert entry to dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "ttl": self.ttl,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """
        Create a cache entry from dictionary data.
        
        Args:
            data (Dict[str, Any]): Dictionary data
            
        Returns:
            CacheEntry: Created cache entry
        """
        entry = cls(
            key=data["key"],
            value=data["value"],
            ttl=data.get("ttl"),
            metadata=data.get("metadata", {})
        )
        entry.created_at = data.get("created_at", time.time())
        entry.accessed_at = data.get("accessed_at", entry.created_at)
        
        return entry


class BaseCache(ABC):
    """
    Abstract base class for cache implementations.
    
    This class defines the interface that all cache backends should implement.
    """
    
    def __init__(self, default_ttl: Optional[float] = None):
        """
        Initialize the base cache.
        
        Args:
            default_ttl (Optional[float]): Default time to live in seconds
        """
        self.default_ttl = default_ttl
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found/expired
        """
        pass
    
    @abstractmethod
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """
        Set a value in the cache.
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
            ttl (Optional[float]): Time to live in seconds
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if key was deleted, False if not found
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if key exists and is not expired
        """
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        """
        Get all cache keys.
        
        Returns:
            List[str]: List of cache keys
        """
        pass
    
    @abstractmethod
    def size(self) -> int:
        """
        Get the number of entries in the cache.
        
        Returns:
            int: Number of cache entries
        """
        pass
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from the cache.
        
        Returns:
            int: Number of entries removed
        """
        # Default implementation - override for efficiency
        removed = 0
        for key in self.keys():
            if not self.exists(key):  # This will check expiration
                self.delete(key)
                removed += 1
        
        return removed


class MemoryCache(BaseCache):
    """
    In-memory cache implementation using a dictionary.
    
    This cache stores entries in memory and is fast but doesn't persist
    across application restarts. It supports automatic cleanup of expired entries.
    """
    
    def __init__(
        self,
        default_ttl: Optional[float] = None,
        max_size: Optional[int] = None,
        cleanup_interval: float = 300.0  # 5 minutes
    ):
        """
        Initialize the memory cache.
        
        Args:
            default_ttl (Optional[float]): Default time to live in seconds
            max_size (Optional[int]): Maximum number of entries
            cleanup_interval (float): Interval for automatic cleanup in seconds
        """
        super().__init__(default_ttl)
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
    
    def _auto_cleanup(self) -> None:
        """Automatically cleanup expired entries if needed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self.cleanup_interval:
            self.cleanup_expired()
            self._last_cleanup = current_time
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is at maximum size."""
        if self.max_size is None or len(self._cache) < self.max_size:
            return
        
        # Simple LRU eviction: remove least recently accessed
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda item: item[1].accessed_at
        )
        
        entries_to_remove = len(self._cache) - self.max_size + 1
        for i in range(entries_to_remove):
            key = sorted_entries[i][0]
            del self._cache[key]
            self.logger.debug(f"Evicted cache entry: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the memory cache."""
        with self._lock:
            self._auto_cleanup()
            
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            if entry.is_expired():
                del self._cache[key]
                return None
            
            entry.touch()
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set a value in the memory cache."""
        with self._lock:
            self._auto_cleanup()
            self._evict_if_needed()
            
            effective_ttl = ttl if ttl is not None else self.default_ttl
            entry = CacheEntry(key, value, effective_ttl)
            
            self._cache[key] = entry
            self.logger.debug(f"Cached entry: {key}")
    
    def delete(self, key: str) -> bool:
        """Delete a value from the memory cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self.logger.debug(f"Deleted cache entry: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from the memory cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self.logger.info(f"Cleared {count} cache entries")
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the memory cache."""
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                return False
            
            return True
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            self._auto_cleanup()
            return list(self._cache.keys())
    
    def size(self) -> int:
        """Get the number of entries in the cache."""
        with self._lock:
            return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from the cache."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for entry in self._cache.values()
                if entry.is_expired()
            )
            
            return {
                "total_entries": total_entries,
                "active_entries": total_entries - expired_entries,
                "expired_entries": expired_entries,
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
                "last_cleanup": self._last_cleanup
            }


class FileCache(BaseCache):
    """
    File-based cache implementation using JSON files.
    
    This cache persists entries to disk and survives application restarts.
    It's slower than memory cache but provides persistence.
    """
    
    def __init__(
        self,
        cache_dir: str = "data/cache",
        default_ttl: Optional[float] = None,
        max_files: Optional[int] = None
    ):
        """
        Initialize the file cache.
        
        Args:
            cache_dir (str): Directory to store cache files
            default_ttl (Optional[float]): Default time to live in seconds
            max_files (Optional[int]): Maximum number of cache files
        """
        super().__init__(default_ttl)
        self.cache_dir = Path(cache_dir)
        self.max_files = max_files
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
    
    def _get_file_path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        # Create a safe filename from the key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json"
    
    def _load_entry(self, file_path: Path) -> Optional[CacheEntry]:
        """Load a cache entry from file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return CacheEntry.from_dict(data)
        
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            self.logger.warning(f"Failed to load cache entry from {file_path}: {e}")
            return None
    
    def _save_entry(self, entry: CacheEntry) -> None:
        """Save a cache entry to file."""
        file_path = self._get_file_path(entry.key)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save cache entry to {file_path}: {e}")
    
    def _evict_if_needed(self) -> None:
        """Evict files if cache is at maximum size."""
        if self.max_files is None:
            return
        
        cache_files = list(self.cache_dir.glob("*.json"))
        if len(cache_files) < self.max_files:
            return
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda f: f.stat().st_mtime)
        
        files_to_remove = len(cache_files) - self.max_files + 1
        for i in range(files_to_remove):
            try:
                cache_files[i].unlink()
                self.logger.debug(f"Evicted cache file: {cache_files[i]}")
            except Exception as e:
                self.logger.warning(f"Failed to evict {cache_files[i]}: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the file cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                return None
            
            entry = self._load_entry(file_path)
            if not entry:
                return None
            
            if entry.is_expired():
                try:
                    file_path.unlink()
                except Exception:
                    pass
                return None
            
            entry.touch()
            self._save_entry(entry)  # Update access time
            
            return entry.value
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set a value in the file cache."""
        with self._lock:
            self._evict_if_needed()
            
            effective_ttl = ttl if ttl is not None else self.default_ttl
            entry = CacheEntry(key, value, effective_ttl)
            
            self._save_entry(entry)
            self.logger.debug(f"Cached entry to file: {key}")
    
    def delete(self, key: str) -> bool:
        """Delete a value from the file cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if file_path.exists():
                try:
                    file_path.unlink()
                    self.logger.debug(f"Deleted cache file: {key}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to delete cache file: {e}")
            
            return False
    
    def clear(self) -> None:
        """Clear all entries from the file cache."""
        with self._lock:
            cache_files = list(self.cache_dir.glob("*.json"))
            count = 0
            
            for file_path in cache_files:
                try:
                    file_path.unlink()
                    count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete {file_path}: {e}")
            
            self.logger.info(f"Cleared {count} cache files")
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the file cache."""
        with self._lock:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                return False
            
            entry = self._load_entry(file_path)
            if not entry or entry.is_expired():
                try:
                    file_path.unlink()
                except Exception:
                    pass
                return False
            
            return True
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            keys = []
            
            for file_path in self.cache_dir.glob("*.json"):
                entry = self._load_entry(file_path)
                if entry and not entry.is_expired():
                    keys.append(entry.key)
                elif entry and entry.is_expired():
                    try:
                        file_path.unlink()
                    except Exception:
                        pass
            
            return keys
    
    def size(self) -> int:
        """Get the number of entries in the cache."""
        return len(self.keys())
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from the cache."""
        with self._lock:
            removed = 0
            
            for file_path in self.cache_dir.glob("*.json"):
                entry = self._load_entry(file_path)
                if entry and entry.is_expired():
                    try:
                        file_path.unlink()
                        removed += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to remove expired file {file_path}: {e}")
            
            if removed > 0:
                self.logger.info(f"Cleaned up {removed} expired cache files")
            
            return removed


class SQLiteCache(BaseCache):
    """
    SQLite-based cache implementation.
    
    This cache uses SQLite database for storage, providing better performance
    than file cache for large datasets while maintaining persistence.
    """
    
    def __init__(
        self,
        db_path: str = "data/cache/cache.db",
        default_ttl: Optional[float] = None,
        max_entries: Optional[int] = None
    ):
        """
        Initialize the SQLite cache.
        
        Args:
            db_path (str): Path to SQLite database file
            default_ttl (Optional[float]): Default time to live in seconds
            max_entries (Optional[int]): Maximum number of entries
        """
        super().__init__(default_ttl)
        self.db_path = Path(db_path)
        self.max_entries = max_entries
        
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        key TEXT PRIMARY KEY,
                        value BLOB,
                        created_at REAL,
                        accessed_at REAL,
                        ttl REAL,
                        metadata TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_accessed_at ON cache_entries(accessed_at)
                """)
                conn.commit()
            finally:
                conn.close()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self.db_path))
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize a value for storage."""
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize a value from storage."""
        return pickle.loads(data)
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is at maximum size."""
        if self.max_entries is None:
            return
        
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
            count = cursor.fetchone()[0]
            
            if count < self.max_entries:
                return
            
            # Remove oldest entries by access time
            entries_to_remove = count - self.max_entries + 1
            conn.execute("""
                DELETE FROM cache_entries 
                WHERE key IN (
                    SELECT key FROM cache_entries 
                    ORDER BY accessed_at ASC 
                    LIMIT ?
                )
            """, (entries_to_remove,))
            
            conn.commit()
            self.logger.debug(f"Evicted {entries_to_remove} cache entries")
            
        finally:
            conn.close()
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the SQLite cache."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT value, created_at, ttl FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                value_data, created_at, ttl = row
                
                # Check expiration
                if ttl is not None and time.time() > (created_at + ttl):
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return None
                
                # Update access time
                conn.execute(
                    "UPDATE cache_entries SET accessed_at = ? WHERE key = ?",
                    (time.time(), key)
                )
                conn.commit()
                
                return self._deserialize_value(value_data)
                
            finally:
                conn.close()
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None
    ) -> None:
        """Set a value in the SQLite cache."""
        with self._lock:
            self._evict_if_needed()
            
            effective_ttl = ttl if ttl is not None else self.default_ttl
            current_time = time.time()
            
            value_data = self._serialize_value(value)
            metadata = json.dumps({})
            
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (key, value, created_at, accessed_at, ttl, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (key, value_data, current_time, current_time, effective_ttl, metadata))
                
                conn.commit()
                self.logger.debug(f"Cached entry in SQLite: {key}")
                
            finally:
                conn.close()
    
    def delete(self, key: str) -> bool:
        """Delete a value from the SQLite cache."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
                
                deleted = cursor.rowcount > 0
                if deleted:
                    self.logger.debug(f"Deleted cache entry: {key}")
                
                return deleted
                
            finally:
                conn.close()
    
    def clear(self) -> None:
        """Clear all entries from the SQLite cache."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                count = cursor.fetchone()[0]
                
                conn.execute("DELETE FROM cache_entries")
                conn.commit()
                
                self.logger.info(f"Cleared {count} cache entries from SQLite")
                
            finally:
                conn.close()
    
    def exists(self, key: str) -> bool:
        """Check if a key exists in the SQLite cache."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT created_at, ttl FROM cache_entries WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return False
                
                created_at, ttl = row
                
                # Check expiration
                if ttl is not None and time.time() > (created_at + ttl):
                    conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                    conn.commit()
                    return False
                
                return True
                
            finally:
                conn.close()
    
    def keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            # First cleanup expired entries
            self.cleanup_expired()
            
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT key FROM cache_entries")
                return [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()
    
    def size(self) -> int:
        """Get the number of entries in the cache."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                return cursor.fetchone()[0]
            finally:
                conn.close()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from the cache."""
        with self._lock:
            current_time = time.time()
            
            conn = self._get_connection()
            try:
                cursor = conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE ttl IS NOT NULL AND (created_at + ttl) < ?
                """, (current_time,))
                
                conn.commit()
                removed = cursor.rowcount
                
                if removed > 0:
                    self.logger.info(f"Cleaned up {removed} expired SQLite cache entries")
                
                return removed
                
            finally:
                conn.close()


class LLMResponseCache:
    """
    Specialized cache for LLM responses with automatic key generation.
    
    This cache automatically generates cache keys based on model, prompt,
    and parameters, making it easy to cache LLM responses without manual
    key management.
    """
    
    def __init__(
        self,
        backend: BaseCache,
        include_model: bool = True,
        include_params: bool = True
    ):
        """
        Initialize the LLM response cache.
        
        Args:
            backend (BaseCache): Cache backend to use
            include_model (bool): Include model name in cache key
            include_params (bool): Include parameters in cache key
        """
        self.backend = backend
        self.include_model = include_model
        self.include_params = include_params
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _generate_key(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params
    ) -> str:
        """
        Generate a cache key for the given inputs.
        
        Args:
            prompt (str): Input prompt
            model_name (Optional[str]): Model name
            **params: Additional parameters
            
        Returns:
            str: Generated cache key
        """
        key_parts = [prompt]
        
        if self.include_model and model_name:
            key_parts.append(f"model:{model_name}")
        
        if self.include_params and params:
            # Sort parameters for consistent keys
            sorted_params = sorted(params.items())
            params_str = json.dumps(sorted_params, sort_keys=True)
            key_parts.append(f"params:{params_str}")
        
        # Create hash of combined key parts
        combined = "|".join(key_parts)
        key_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return f"llm_response:{key_hash}"
    
    def get_response(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params
    ) -> Optional[str]:
        """
        Get a cached response for the given inputs.
        
        Args:
            prompt (str): Input prompt
            model_name (Optional[str]): Model name
            **params: Additional parameters
            
        Returns:
            Optional[str]: Cached response or None if not found
        """
        key = self._generate_key(prompt, model_name, **params)
        response = self.backend.get(key)
        
        if response:
            self.logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
        else:
            self.logger.debug(f"Cache miss for prompt: {prompt[:50]}...")
        
        return response
    
    def cache_response(
        self,
        prompt: str,
        response: str,
        model_name: Optional[str] = None,
        ttl: Optional[float] = None,
        **params
    ) -> None:
        """
        Cache a response for the given inputs.
        
        Args:
            prompt (str): Input prompt
            response (str): Generated response
            model_name (Optional[str]): Model name
            ttl (Optional[float]): Time to live
            **params: Additional parameters
        """
        key = self._generate_key(prompt, model_name, **params)
        self.backend.set(key, response, ttl)
        
        self.logger.debug(f"Cached response for prompt: {prompt[:50]}...")
    
    def invalidate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **params
    ) -> bool:
        """
        Invalidate a cached response.
        
        Args:
            prompt (str): Input prompt
            model_name (Optional[str]): Model name
            **params: Additional parameters
            
        Returns:
            bool: True if entry was deleted
        """
        key = self._generate_key(prompt, model_name, **params)
        return self.backend.delete(key)
    
    def clear_all(self) -> None:
        """Clear all cached responses."""
        self.backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        stats = {
            "backend_type": self.backend.__class__.__name__,
            "total_entries": self.backend.size(),
            "include_model": self.include_model,
            "include_params": self.include_params
        }
        
        if hasattr(self.backend, 'get_stats'):
            stats.update(self.backend.get_stats())
        
        return stats


# Global cache instances
memory_cache = MemoryCache(default_ttl=3600)  # 1 hour default TTL
file_cache = FileCache(default_ttl=86400)      # 1 day default TTL

# LLM response cache using memory backend
llm_cache = LLMResponseCache(memory_cache)


# Convenience functions
def cache_llm_response(
    prompt: str,
    response: str,
    model_name: Optional[str] = None,
    ttl: Optional[float] = None,
    **params
) -> None:
    """
    Cache an LLM response (convenience function).
    
    Args:
        prompt (str): Input prompt
        response (str): Generated response
        model_name (Optional[str]): Model name
        ttl (Optional[float]): Time to live
        **params: Additional parameters
    """
    llm_cache.cache_response(prompt, response, model_name, ttl, **params)


def get_cached_llm_response(
    prompt: str,
    model_name: Optional[str] = None,
    **params
) -> Optional[str]:
    """
    Get a cached LLM response (convenience function).
    
    Args:
        prompt (str): Input prompt
        model_name (Optional[str]): Model name
        **params: Additional parameters
        
    Returns:
        Optional[str]: Cached response or None if not found
    """
    return llm_cache.get_response(prompt, model_name, **params)