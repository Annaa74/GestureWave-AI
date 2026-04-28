"""
GestureWave AI — Real-Time Gesture Event Log
Shared in-memory log that both the engine (main.py) and dashboard read from.
No database needed — events are stored in a thread-safe deque.
"""
import time
from collections import deque
from threading import Lock


class GestureEvent:
    """A single gesture event with timing."""
    __slots__ = ("timestamp", "gesture", "action", "duration_ms")

    def __init__(self, gesture: str, action: str = "", duration_ms: float = 0.0):
        self.timestamp = time.time()
        self.gesture = gesture
        self.action = action
        self.duration_ms = duration_ms

    def formatted(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        dur = f" ({self.duration_ms:.0f}ms)" if self.duration_ms > 0 else ""
        act = f" → {self.action}" if self.action else ""
        return f"[{ts}] {self.gesture}{act}{dur}"


class GestureLog:
    """Thread-safe gesture event logger with frequency tracking."""

    def __init__(self, maxlen: int = 500):
        self._events = deque(maxlen=maxlen)
        self._counts = {}
        self._lock = Lock()
        self._last_read_index = 0
        self._total_logged = 0

    def log(self, gesture: str, action: str = "", duration_ms: float = 0.0):
        """Log a gesture event. Thread-safe."""
        event = GestureEvent(gesture, action, duration_ms)
        with self._lock:
            self._events.append(event)
            self._counts[gesture] = self._counts.get(gesture, 0) + 1
            self._total_logged += 1

    def get_new_events(self) -> list:
        """Get events since last read (for dashboard polling). Thread-safe."""
        with self._lock:
            current_total = self._total_logged
            if current_total == self._last_read_index:
                return []
            # Calculate how many new events
            new_count = current_total - self._last_read_index
            self._last_read_index = current_total
            events = list(self._events)
            return events[-new_count:] if new_count <= len(events) else list(events)

    def recent(self, n: int = 30) -> list:
        """Get last N events. Thread-safe."""
        with self._lock:
            return list(self._events)[-n:]

    @property
    def counts(self) -> dict:
        """Get gesture frequency counts. Thread-safe."""
        with self._lock:
            return dict(self._counts)

    @property
    def total(self) -> int:
        """Total number of gestures logged."""
        with self._lock:
            return self._total_logged

    def clear(self):
        """Clear all events and counts. Thread-safe."""
        with self._lock:
            self._events.clear()
            self._counts.clear()
            self._total_logged = 0
            self._last_read_index = 0


# ── Global singleton ────────────────────────────────────────────────────────
gesture_log = GestureLog()

# Event logging system
