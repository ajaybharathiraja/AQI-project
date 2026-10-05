"""Array-backed binary max heap with manually implemented sift operations.

Insert/extract: O(log n) heap work, peek O(1), bottom-up heapify O(n).
List growth makes insert amortized O(log n), with occasional O(n) allocation.
Storage O(n), sift auxiliary space O(1); heapify uses O(n) temporary space.
Run demo: python -B -m DATA_STRUCTURES.max_heap
"""
from dataclasses import dataclass
import math
if __package__:
    from .trace import emit, heap_snapshot
else:
    from trace import emit, heap_snapshot


@dataclass(frozen=True)
class HeapItem:
    """Returned priority/payload pair; sequence makes equal priorities FIFO."""
    priority: float
    payload: object
    sequence: int


class MaxHeap:
    """Finite int/float priorities, larger first; arbitrary payloads never compared.

    Invalid priorities raise ValueError. Empty extraction/peek raises IndexError.
    Comparison assumes caller supplies one comparable metric/unit/window.
    """

    def __init__(self):
        self._data = []
        self._sequence = 0

    @staticmethod
    def _priority(value):
        if type(value) not in (int, float):
            raise ValueError('priority must be a finite int or float')
        try:
            priority = float(value)
        except OverflowError:
            raise ValueError('priority exceeds supported numeric range') from None
        if not math.isfinite(priority):
            raise ValueError('priority must be finite')
        return priority

    @staticmethod
    def _higher(a, b):
        emit('MaxHeap', 'compare', left_priority=a.priority, right_priority=b.priority, left_sequence=a.sequence, right_sequence=b.sequence)
        return a.priority > b.priority or (a.priority == b.priority and a.sequence < b.sequence)

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def _sift_up(self, index):
        while index > 0:
            parent = (index - 1) // 2
            if not self._higher(self._data[index], self._data[parent]):
                break
            self._data[index], self._data[parent] = self._data[parent], self._data[index]
            emit('MaxHeap', 'swap', first=index, second=parent)
            heap_snapshot(self._data)
            index = parent

    @classmethod
    def _sift_down(cls, data, index):
        while 2 * index + 1 < len(data):
            child = 2 * index + 1
            right = child + 1
            if right < len(data) and cls._higher(data[right], data[child]):
                child = right
            if not cls._higher(data[child], data[index]):
                break
            data[index], data[child] = data[child], data[index]
            emit('MaxHeap', 'swap', first=index, second=child)
            heap_snapshot(data)
            index = child

    def insert(self, priority, payload):
        """Add (numeric priority, payload); output None."""
        item = HeapItem(self._priority(priority), payload, self._sequence)
        self._sequence += 1
        self._data.append(item)
        emit('MaxHeap', 'insert', priority=item.priority, index=len(self._data)-1)
        heap_snapshot(self._data)
        self._sift_up(len(self._data) - 1)

    def heapify(self, pairs):
        """Replace contents from iterable of (priority,payload), O(n) bottom-up.

        Validate/build in a temporary array first, preserving old heap on failure.
        Equal priorities follow input order. This is not repeated insertion.
        """
        data = []
        sequence = self._sequence
        for priority, payload in pairs:
            data.append(HeapItem(self._priority(priority), payload, sequence))
            sequence += 1
        heap_snapshot(data)
        for index in range(len(data) // 2 - 1, -1, -1):
            emit('MaxHeap', 'heapify', root=index, size=len(data))
            self._sift_down(data, index)
        self._data = data
        self._sequence = sequence
        heap_snapshot(self._data)

    def peek(self):
        """Return the highest HeapItem without removal."""
        if self.is_empty():
            raise IndexError('Max heap is empty')
        return self._data[0]

    def extract_max(self):
        """Remove/return highest HeapItem; restore heap by sifting down."""
        result = self.peek()
        emit('MaxHeap', 'extract_max', priority=result.priority, size_before=len(self._data))
        last = self._data.pop()
        if self._data:
            self._data[0] = last
            heap_snapshot(self._data)
            self._sift_down(self._data, 0)
        heap_snapshot(self._data)
        return result


if __name__ == '__main__':
    ranking = MaxHeap()
    ranking.heapify([(75, 'Synthetic station A'), (200, 'Synthetic station B')])
    ranking.insert(150, 'Synthetic station C')
    while not ranking.is_empty():
        print(ranking.extract_max())
