#!/usr/bin/env python3
"""
Pickle-Safe Transport Operator Cache
====================================

CRITICAL FIX: Makes cache serializable by avoiding nested functions.

The original implementation used a local nested function which pickle
cannot serialize. This version uses a proper class-based approach.

Author: Chris
Date: November 2025
"""

import numpy as np
from typing import Optional
import hashlib


class TransportCache:
    """
    Pickle-safe transport operator cache.
    
    Uses LRU caching with hash-based keys for fast lookups.
    Can be serialized with pickle unlike nested function approach.
    """
    
    def __init__(self, system, max_size: int = 1000):
        """
        Initialize cache for a system.
        
        Args:
            system: MultiAgentSystem instance
            max_size: Maximum number of cached transport operators
        """
        self.system = system
        self.max_size = max_size
        
        # Cache storage: (i, j, phi_i_hash, phi_j_hash) -> Omega_ij
        self._cache = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.invalidations = 0
        
        # Hash cache for phi arrays (avoid recomputing hashes)
        self._phi_hashes = {}
    
    def _hash_array(self, arr: np.ndarray) -> str:
        """
        Fast hash of numpy array for cache key.
        
        Uses first/last/mean values rather than full array hash
        for speed while maintaining uniqueness.
        """
        # Use array's data pointer + shape as unique identifier
        # This is fast and works for same object
        arr_id = id(arr)
        
        if arr_id in self._phi_hashes:
            return self._phi_hashes[arr_id]
        
        # For new arrays, create hash from key statistics
        # This is much faster than hashing entire array
        if arr.size < 100:
            # Small arrays: hash everything
            hash_val = hashlib.md5(arr.tobytes()).hexdigest()[:16]
        else:
            # Large arrays: hash statistics (faster)
            stats = np.array([
                arr.flat[0], arr.flat[-1],  # First/last
                np.mean(arr), np.std(arr),   # Mean/std
                arr.shape[0], arr.size       # Shape info
            ])
            hash_val = hashlib.md5(stats.tobytes()).hexdigest()[:16]
        
        self._phi_hashes[arr_id] = hash_val
        return hash_val
    
    def get(self, i: int, j: int) -> Optional[np.ndarray]:
        """
        Get cached transport operator Ω_ij.
        
        Returns None if not in cache or if phi has changed.
        """
        agent_i = self.system.agents[i]
        agent_j = self.system.agents[j]
        
        # Create cache key from phi hashes
        phi_i_hash = self._hash_array(agent_i.gauge.phi)
        phi_j_hash = self._hash_array(agent_j.gauge.phi)
        
        key = (i, j, phi_i_hash, phi_j_hash)
        
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        else:
            self.misses += 1
            return None
    
    def put(self, i: int, j: int, Omega_ij: np.ndarray):
        """
        Store transport operator in cache.
        
        Implements simple LRU eviction if cache is full.
        """
        agent_i = self.system.agents[i]
        agent_j = self.system.agents[j]
        
        phi_i_hash = self._hash_array(agent_i.gauge.phi)
        phi_j_hash = self._hash_array(agent_j.gauge.phi)
        
        key = (i, j, phi_i_hash, phi_j_hash)
        
        # Evict oldest entry if cache full
        if len(self._cache) >= self.max_size:
            # Remove first key (simple FIFO, could use LRU)
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[key] = Omega_ij
    
    def invalidate(self):
        """
        Invalidate entire cache.
        
        Called after parameter updates since all phi values may have changed.
        """
        self._cache.clear()
        self._phi_hashes.clear()
        self.invalidations += 1
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'max_size': self.max_size,
            'invalidations': self.invalidations
        }
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"TransportCache(size={stats['size']}/{stats['max_size']}, "
                f"hit_rate={stats['hit_rate']:.1%})")


def add_cache_to_system(system, max_size: int = 1000) -> TransportCache:
    """
    Add pickle-safe transport cache to system.
    
    FIXED: Uses class-based cache instead of nested functions.
    
    Args:
        system: MultiAgentSystem
        max_size: Maximum cache size
    
    Returns:
        cache: TransportCache instance (for monitoring)
    """
    # Create cache
    cache = TransportCache(system, max_size=max_size)
    
    # Store cache on system
    system._transport_cache = cache
    
    # Replace compute_transport_ij with cached version
    # Store original method
    if not hasattr(system, '_original_compute_transport_ij'):
        system._original_compute_transport_ij = system.compute_transport_ij
    
    # Create cached version as instance method
    def cached_compute_transport_ij(i: int, j: int) -> np.ndarray:
        """Cached version of compute_transport_ij."""
        # Try cache first
        Omega_ij = cache.get(i, j)
        
        if Omega_ij is not None:
            return Omega_ij
        
        # Cache miss - compute and store
        Omega_ij = system._original_compute_transport_ij(i, j)
        cache.put(i, j, Omega_ij)
        
        return Omega_ij
    
    # Replace method
    system.compute_transport_ij = cached_compute_transport_ij
    
    return cache


def invalidate_cache_after_update(system):
    """
    Invalidate transport cache after parameter updates.
    
    Call this after updating phi values in training loop.
    """
    if hasattr(system, '_transport_cache'):
        system._transport_cache.invalidate()


def remove_cache_from_system(system):
    """
    Remove cache and restore original method.
    
    Call this before pickling if you still have issues.
    """
    if hasattr(system, '_original_compute_transport_ij'):
        system.compute_transport_ij = system._original_compute_transport_ij
        delattr(system, '_original_compute_transport_ij')
    
    if hasattr(system, '_transport_cache'):
        delattr(system, '_transport_cache')


# =============================================================================
# Pickle Support - Custom Serialization
# =============================================================================

def prepare_system_for_pickle(system):
    """
    Prepare system for pickling by temporarily removing cache.
    
    Returns cache object so it can be restored after pickling.
    
    Usage:
        cache = prepare_system_for_pickle(system)
        pickle.dump(system, f)
        restore_system_after_pickle(system, cache)
    """
    cache = None
    
    if hasattr(system, '_transport_cache'):
        cache = system._transport_cache
        # Store cache stats
        cache_stats = cache.get_stats()
        
        # Remove cache
        remove_cache_from_system(system)
        
        print(f"Removed cache for pickling (stats: {cache_stats})")
    
    return cache


def restore_system_after_pickle(system, cache: Optional[TransportCache] = None):
    """
    Restore cache after unpickling.
    
    Args:
        system: Unpickled system
        cache: Original cache (if available) or None to create fresh cache
    """
    if cache is not None:
        # Restore cache with same settings
        new_cache = add_cache_to_system(system, max_size=cache.max_size)
        print(f"Restored cache: {new_cache}")
    else:
        # Create fresh cache
        new_cache = add_cache_to_system(system, max_size=1000)
        print(f"Created fresh cache: {new_cache}")
    
    return new_cache


# =============================================================================
# Safe Pickle Wrappers
# =============================================================================

def safe_pickle_dump(system, filepath):
    """
    Safely pickle system by temporarily removing cache.
    
    Args:
        system: MultiAgentSystem with cache
        filepath: Path to save to
    """
    import pickle
    
    # Remove cache temporarily
    cache = prepare_system_for_pickle(system)
    
    try:
        # Pickle without cache
        with open(filepath, 'wb') as f:
            pickle.dump(system, f)
        print(f"✓ Saved system to {filepath}")
    finally:
        # Restore cache
        restore_system_after_pickle(system, cache)


def safe_pickle_load(filepath):
    """
    Load pickled system and add cache.
    
    Args:
        filepath: Path to load from
    
    Returns:
        system: System with cache restored
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        system = pickle.load(f)
    
    print(f"✓ Loaded system from {filepath}")
    
    # Add cache to loaded system
    cache = add_cache_to_system(system, max_size=1000)
    print(f"✓ Added cache: {cache}")
    
    return system

