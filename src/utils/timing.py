"""Timing utilities for profiling experiment runtime."""

import logging
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str, log_level: int = logging.INFO):
        self.name = name
        self.log_level = log_level
        self.start_time = None
        self.elapsed_ms = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        logger.log(self.log_level, f"[TIMING] {self.name}: {self.elapsed_ms:.0f}ms")


class TimingReport:
    """Collect and report timing statistics."""
    
    def __init__(self):
        self.timings: dict[str, list[float]] = {}
    
    def add(self, name: str, elapsed_ms: float):
        """Add a timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(elapsed_ms)
    
    def summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics."""
        summary = {}
        for name, times in self.timings.items():
            summary[name] = {
                "total_ms": sum(times),
                "mean_ms": sum(times) / len(times) if times else 0,
                "count": len(times),
                "min_ms": min(times) if times else 0,
                "max_ms": max(times) if times else 0,
            }
        return summary
    
    def log_summary(self):
        """Log summary statistics."""
        logger.info("=" * 80)
        logger.info("TIMING SUMMARY")
        logger.info("=" * 80)
        
        summary = self.summary()
        total_time = sum(s["total_ms"] for s in summary.values())
        
        # Sort by total time descending
        sorted_items = sorted(summary.items(), key=lambda x: x[1]["total_ms"], reverse=True)
        
        for name, stats in sorted_items:
            pct = (stats["total_ms"] / total_time * 100) if total_time > 0 else 0
            logger.info(
                f"{name:40s}: {stats['total_ms']:>8.0f}ms ({pct:>5.1f}%)  "
                f"[count={stats['count']:>3d}, mean={stats['mean_ms']:>7.0f}ms]"
            )
        
        logger.info("=" * 80)
        logger.info(f"{'TOTAL':40s}: {total_time:>8.0f}ms (100.0%)")
        logger.info("=" * 80)
    
    @contextmanager
    def time(self, name: str):
        """Context manager for timing a code block."""
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.add(name, elapsed_ms)
