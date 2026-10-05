"""Manual fixed-capacity ring buffer for recent observations.

Enqueue/dequeue/peek/is_empty/is_full: O(1) time and auxiliary space.
Backing space O(capacity); to_list takes O(n) time and output space.
Run demo: python -B -m DATA_STRUCTURES.circular_queue
"""


if __package__:
    from .trace import emit
else:
    from trace import emit


class CircularQueue:
    """FIFO of arbitrary values. Explicit overwrite mode evicts the oldest item.

    Invalid capacity/mode: ValueError. Full enqueue without overwrite:
    OverflowError. Empty peek/dequeue: IndexError. Payloads are not copied.
    """

    def __init__(self, capacity, *, overwrite=False):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('capacity must be a positive integer')
        if type(overwrite) is not bool:
            raise ValueError('overwrite must be boolean')
        self._buffer = [None] * capacity
        self._head = 0
        self._size = 0
        self._overwrite = overwrite

    @property
    def capacity(self):
        return len(self._buffer)

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def is_full(self):
        return self._size == self.capacity

    def enqueue(self, value):
        """Add value; return an evicted payload in overwrite mode, otherwise None.

        A returned None may also be an evicted None payload; occupancy is tracked
        by size, never by using None as an empty-cell test.
        """
        removed = None
        if self.is_full():
            if not self._overwrite:
                raise OverflowError('Circular queue is full')
            removed = self.dequeue()
        tail = (self._head + self._size) % self.capacity
        self._buffer[tail] = value
        self._size += 1
        emit('CircularQueue', 'enqueue', front=self._head, rear=tail, size=self._size, capacity=self.capacity)
        return removed

    def dequeue(self):
        """Remove/return the oldest value; empty queue raises IndexError."""
        value = self.peek()
        self._buffer[self._head] = None  # Release the payload reference.
        self._head = (self._head + 1) % self.capacity
        self._size -= 1
        emit('CircularQueue', 'dequeue', front=self._head, rear=(self._head+self._size-1)%self.capacity if self._size else None, size=self._size)
        return value

    def peek(self):
        """Return oldest value without removal."""
        if self.is_empty():
            raise IndexError('Circular queue is empty')
        return self._buffer[self._head]

    def to_list(self):
        """Return a shallow FIFO snapshot; does not modify the queue."""
        return [self._buffer[(self._head + i) % self.capacity] for i in range(self._size)]


if __name__ == '__main__':
    recent = CircularQueue(3, overwrite=True)
    for observation in (45, 62, 71, 88):
        recent.enqueue(observation)
    print('Synthetic recent readings:', recent.to_list())
    print('Oldest:', recent.dequeue())
