"""Elapsed-time pollution window using a manually linked FIFO, not deque.

add/advance cost O(1+e) for e expired nodes; each node expires once, so total
stream-maintenance cost is amortized O(1) per observation. Statistics scan costs
O(n) time and O(1) auxiliary space. Storage O(n), bounded by window/cadence.
Run demo: python -B -m DATA_STRUCTURES.sliding_window
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
if __package__:
    from .trace import emit
else:
    from trace import emit


@dataclass(slots=True)
class _Sample:
    timestamp: datetime
    value: float
    next: object = None


@dataclass(frozen=True)
class WindowStatistics:
    count: int
    expected_count: int
    coverage: float
    complete: bool
    mean: float | None
    minimum: float | None
    maximum: float | None
    change: float | None
    slope_per_hour: float | None
    trend: str | None


class SlidingWindow:
    """One scalar station/pollutant/unit series, timestamped on a fixed cadence.

    Supports 3h/6h/12h/24h and other positive durations divisible by cadence.
    Keep samples in (as_of-window, as_of]; the left edge expires exactly.
    Timestamp inputs must be aware datetimes, strictly increasing after the last
    add/advance watermark, aligned to the first sample's grid. Gaps are allowed
    and reduce coverage; no zero filling or inferred measurements. Values must
    be finite nonnegative ints/floats. Invalid operations raise ValueError without
    changing state. Instantiate separately for each comparable series.
    """

    def __init__(self, window_hours=3, sampling_minutes=60):
        if type(window_hours) not in (int, float) or type(sampling_minutes) not in (int, float):
            raise ValueError('window and cadence must be positive finite numbers')
        try:
            valid = math.isfinite(window_hours) and math.isfinite(sampling_minutes)
            self._duration = timedelta(hours=window_hours)
            self._cadence = timedelta(minutes=sampling_minutes)
        except (OverflowError, ValueError):
            raise ValueError('window or cadence is outside the supported range') from None
        if not valid or self._duration <= timedelta(0) or self._cadence <= timedelta(0):
            raise ValueError('window and cadence must be positive')
        if self._duration % self._cadence != timedelta(0):
            raise ValueError('window must contain a whole number of sampling intervals')
        self._expected = self._duration // self._cadence
        self._head = self._tail = None
        self._size = 0
        self._origin = self._watermark = None

    def __len__(self):
        return self._size

    def _time(self, timestamp):
        if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError('timestamp must be a timezone-aware datetime')
        timestamp = timestamp.astimezone(timezone.utc)
        if self._watermark is not None and timestamp <= self._watermark:
            raise ValueError('timestamps must advance strictly beyond the watermark')
        if self._origin is not None and (timestamp - self._origin) % self._cadence != timedelta(0):
            raise ValueError('timestamp is off the declared sampling grid')
        try:
            timestamp - self._duration
        except OverflowError:
            raise ValueError('window start is outside datetime range') from None
        return timestamp

    def _expire(self, timestamp):
        cutoff = timestamp - self._duration
        while self._head is not None and self._head.timestamp <= cutoff:
            emit('SlidingWindow', 'expire', timestamp=self._head.timestamp.isoformat(), value=self._head.value)
            self._head = self._head.next
            self._size -= 1
        if self._head is None:
            self._tail = None
        self._watermark = timestamp
        if self._origin is None:
            self._origin = timestamp

    def add(self, timestamp, value):
        """Add one observation; expire old samples, return None."""
        timestamp = self._time(timestamp)
        if type(value) not in (int, float):
            raise ValueError('value must be a finite nonnegative int or float')
        try:
            value = float(value)
        except OverflowError:
            raise ValueError('value exceeds supported numeric range') from None
        if not math.isfinite(value) or value < 0:
            raise ValueError('value must be finite and nonnegative')
        self._expire(timestamp)
        node = _Sample(timestamp, value)
        if self._tail is None:
            self._head = node
        else:
            self._tail.next = node
        self._tail = node
        self._size += 1
        emit('SlidingWindow', 'add', timestamp=timestamp.isoformat(), value=value, window_start=(timestamp-self._duration).isoformat(), window_end=timestamp.isoformat())

    def advance(self, timestamp):
        """Expire stale observations without inventing a new one; output None.

        Subsequent add timestamps must be later than this watermark.
        """
        self._expire(self._time(timestamp))
        emit('SlidingWindow', 'advance', window_start=(self._watermark-self._duration).isoformat(), window_end=self._watermark.isoformat())

    def statistics(self):
        """Return descriptive stats; slope is endpoint change per elapsed hour.

        Partial windows return descriptive numbers with complete=False. They are
        not regulatory AQI averages. Empty or single-point trends are None.
        """
        if self._size == 0:
            emit('SlidingWindow', 'result', count=0, mean=None, values_empty=True,
                 window_end=self._watermark.isoformat() if self._watermark else None,
                 window_start=(self._watermark-self._duration).isoformat() if self._watermark else None)
            return WindowStatistics(0, self._expected, 0, False, None, None, None, None, None, None)
        node = self._head
        low = high = node.value
        # Online mean avoids overflow from summing large nonnegative readings.
        mean = 0.0
        count = 0
        while node is not None:
            emit('SlidingWindow', 'value', timestamp=node.timestamp.isoformat(), value=node.value)
            count += 1
            mean += (node.value - mean) / count
            if node.value < low:
                low = node.value
            if node.value > high:
                high = node.value
            node = node.next
        change = slope = trend = None
        if self._size >= 2:
            change = self._tail.value - self._head.value
            hours = (self._tail.timestamp - self._head.timestamp).total_seconds() / 3600
            slope = change / hours
            if not math.isfinite(slope):
                raise ValueError('trend exceeds supported numeric range')
            trend = 'rising' if change > 0 else 'falling' if change < 0 else 'flat'
        emit('SlidingWindow', 'result', count=self._size, mean=mean, minimum=low, maximum=high, trend=trend,
             window_start=(self._watermark-self._duration).isoformat(), window_end=self._watermark.isoformat())
        return WindowStatistics(self._size, self._expected, self._size / self._expected,
                                self._size == self._expected, mean, low, high, change, slope, trend)


if __name__ == '__main__':
    window = SlidingWindow(3, 60)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    for i, value in enumerate((10, 20, 30, 40)):
        window.add(start + timedelta(hours=i), value)
    print('Synthetic 3h window:', window.statistics())
