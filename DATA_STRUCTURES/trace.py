"""Optional bounded execution diagnostics; never implements algorithm state.

Use capture() around normal calls. Context-local sinks isolate concurrent requests.
Disabled emits cost O(1); enabled storage is O(limit), with counted truncation.
Payloads are explicit scalar snapshots, never arbitrary observation objects.
"""
from contextlib import contextmanager
from contextvars import ContextVar

_sink = ContextVar('dsa_trace', default=None)


@contextmanager
def capture(enabled=True, limit=2000):
    if type(limit) is not int or not 1 <= limit <= 10000:
        raise ValueError('trace limit must be 1..10000')
    sink = {'operations': [], 'dropped': 0, 'limit': limit} if enabled else None
    token = _sink.set(sink)
    try:
        yield sink
    finally:
        _sink.reset(token)


def emit(structure, operation, **state):
    sink = _sink.get()
    if sink is None:
        return
    # A fixed bound prevents trace mode from growing with the archive size.
    if len(sink['operations']) >= sink.get('limit', 2000):
        sink['dropped'] += 1
        return
    sink['operations'].append({'structure': structure, 'operation': operation, **state})


def snapshot():
    sink = _sink.get()
    return {'operations': list(sink['operations']), 'dropped': sink['dropped']} if sink else {'operations': [], 'dropped': 0}


def heap_snapshot(data):
    """Bounded display-only state; no comparisons or algorithm decisions."""
    if _sink.get() is not None:
        emit('MaxHeap', 'state', values=[item.priority for item in data[:32]], size=len(data), truncated=len(data)>32)
