"""Manual string-key hash table: polynomial hashing, chaining and rehashing.

Lists are bucket arrays only; no dict/set implements lookup. For key length k,
search/update/delete cost expected O(k+1), worst O(n*k) including key comparisons. Insert is expected
amortized O(k+1); resize rehashes all keys. Space O(capacity+n+stored key bytes).
Run demo: python -B -m DATA_STRUCTURES.hash_table
"""
from dataclasses import dataclass
if __package__:
    from .trace import emit
else:
    from trace import emit


@dataclass(slots=True)
class _Node:
    key: str
    value: object
    next: object = None


class HashTable:
    """Nonempty string keys -> arbitrary values (including None).

    insert rejects duplicates; search/update/delete raise KeyError for absence.
    update/delete return the previous value. items yields (key,value) pairs in
    unspecified bucket order, O(capacity+n); do not mutate during iteration.
    """

    def __init__(self, capacity=8):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self._buckets = [None] * capacity
        self._size = 0

    @staticmethod
    def _validate(key):
        if not isinstance(key, str) or not key.strip():
            raise ValueError('key must be a nonempty string')

    @staticmethod
    def _hash(key):
        # Explicit bounded polynomial hash; Python hash() is not used.
        value = 0
        for char in key:
            value = (31 * value + ord(char)) & 0xFFFFFFFFFFFFFFFF
        return value

    def _find(self, key):
        self._validate(key)
        hashed = self._hash(key)
        bucket = hashed % len(self._buckets)
        emit('HashTable', 'hash', key=key, hash=hashed, bucket=bucket, capacity=self.capacity)
        node = self._buckets[bucket]
        while node is not None:
            emit('HashTable', 'probe', key=key, bucket=bucket, candidate=node.key)
            if node.key == key:
                emit('HashTable', 'match', key=key, bucket=bucket)
                return node
            node = node.next
        emit('HashTable', 'miss', key=key, bucket=bucket)
        return None

    def __len__(self):
        return self._size

    def __contains__(self, key):
        return self._find(key) is not None

    @property
    def capacity(self):
        return len(self._buckets)

    def search(self, key):
        """Return the stored value, or raise KeyError (expected O(k+1))."""
        node = self._find(key)
        if node is None:
            raise KeyError(key)
        return node.value

    def _resize(self):
        old = self._buckets
        self._buckets = [None] * (2 * len(old))
        # Move existing collision-chain nodes; no built-in mapping is involved.
        for head in old:
            node = head
            while node is not None:
                following = node.next
                bucket = self._hash(node.key) % len(self._buckets)
                node.next = self._buckets[bucket]
                self._buckets[bucket] = node
                node = following

    def insert(self, key, value):
        """Insert a new key/value; duplicate keys raise KeyError."""
        if self._find(key) is not None:
            raise KeyError(f'Key already exists: {key}')
        if (self._size + 1) * 4 > len(self._buckets) * 3:
            self._resize()
        bucket = self._hash(key) % len(self._buckets)
        self._buckets[bucket] = _Node(key, value, self._buckets[bucket])
        self._size += 1
        emit('HashTable', 'insert', key=key, bucket=bucket, size=self._size)

    def update(self, key, value):
        """Replace an existing value and return its previous value."""
        node = self._find(key)
        if node is None:
            raise KeyError(key)
        old, node.value = node.value, value
        return old

    def delete(self, key):
        """Unlink any position in a collision chain; return removed value."""
        self._validate(key)
        bucket = self._hash(key) % len(self._buckets)
        previous = None
        node = self._buckets[bucket]
        while node is not None:
            if node.key == key:
                if previous is None:
                    self._buckets[bucket] = node.next
                else:
                    previous.next = node.next
                self._size -= 1
                return node.value
            previous, node = node, node.next
        raise KeyError(key)

    def items(self):
        """Yield all key/value pairs, O(capacity+n) time, O(1) extra space."""
        for node in self._buckets:
            while node is not None:
                yield node.key, node.value
                node = node.next


class DatasetIndex:
    """district -> station -> dataset ID -> metadata, using HashTable at all levels.

    Input labels and dataset IDs must be nonempty strings. Use relative file paths
    as dataset IDs, not presumed years. Metadata remains caller-owned. Operations
    perform three hash lookups; expected O(sum of key lengths), with collision and
    resize costs as above. Space O(districts+stations+datasets+metadata).
    """

    def __init__(self):
        self._districts = HashTable()

    def _table(self, district, station):
        HashTable._validate(district)
        HashTable._validate(station)
        return self._districts.search(district).search(station)

    def insert(self, district, station, dataset_id, metadata):
        """Insert metadata at the three-level key; never replace silently."""
        for key in (district, station, dataset_id):
            HashTable._validate(key)
        if district not in self._districts:
            self._districts.insert(district, HashTable())
        stations = self._districts.search(district)
        if station not in stations:
            stations.insert(station, HashTable())
        stations.search(station).insert(dataset_id, metadata)

    def search(self, district, station, dataset_id):
        """Return matching metadata; absent keys raise KeyError."""
        return self._table(district, station).search(dataset_id)

    def update(self, district, station, dataset_id, metadata):
        """Replace matching metadata and return the old value."""
        return self._table(district, station).update(dataset_id, metadata)

    def delete(self, district, station, dataset_id):
        """Remove one dataset and prune empty station/district tables."""
        datasets = self._table(district, station)
        old = datasets.delete(dataset_id)
        stations = self._districts.search(district)
        if len(datasets) == 0:
            stations.delete(station)
        if len(stations) == 0:
            self._districts.delete(district)
        return old


if __name__ == '__main__':
    index = DatasetIndex()
    index.insert('Example district', 'Example station', '2024.csv', 'synthetic metadata')
    print(index.search('Example district', 'Example station', '2024.csv'))
    index.update('Example district', 'Example station', '2024.csv', 'updated metadata')
    print(index.delete('Example district', 'Example station', '2024.csv'))
