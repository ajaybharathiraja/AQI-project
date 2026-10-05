"""Stable pollution-alert queue using this project's manually built MaxHeap.

No heapq or queue.PriorityQueue. Enqueue/dequeue amortized O(log n), peek O(1),
storage O(n). Equal-severity alerts leave in arrival order; no external alerts
are sent. Run demo: python -B -m DATA_STRUCTURES.priority_queue
"""
if __package__:
    from .trace import emit
    from .max_heap import MaxHeap
else:
    from trace import emit
    from max_heap import MaxHeap


class PriorityQueue:
    """Manage arbitrary alert payloads with explicit nonnegative numeric severity.

    This class supplies alert semantics over the custom heap, not over a Python
    built-in collection. Callers choose consistent severity meaning; no AQI
    calculation or category inference occurs here.
    """

    def __init__(self):
        self._heap = MaxHeap()

    def __len__(self):
        return len(self._heap)

    def is_empty(self):
        return self._heap.is_empty()

    def enqueue(self, alert, severity):
        """Input alert plus finite severity >=0; returns None, ValueError if invalid."""
        priority = MaxHeap._priority(severity)
        if priority < 0:
            raise ValueError('alert severity cannot be negative')
        self._heap.insert(priority, alert)
        emit('PriorityQueue', 'insert', priority=priority, size=len(self))

    def dequeue(self):
        """Return highest-severity HeapItem; IndexError when empty."""
        item = self._heap.extract_max()
        emit('PriorityQueue', 'extract', priority=item.priority, size=len(self))
        return item

    def peek(self):
        """Return next HeapItem without removal; IndexError when empty."""
        return self._heap.peek()


if __name__ == '__main__':
    alerts = PriorityQueue()
    alerts.enqueue('Synthetic lower-severity alert', 1)
    alerts.enqueue('Synthetic higher-severity alert', 3)
    alerts.enqueue('Synthetic second higher-severity alert', 3)
    while not alerts.is_empty():
        print(alerts.dequeue())
