"""Iterative DFS with a manual linked stack and custom hash-table visited state.

Expected O(V+E) time with bounded IDs, O(V) auxiliary/output space. Adjacency
iterators prevent copying edges into stack frames. Collision-heavy lookups can
worsen time to O((V+E)*V). No recursion-depth limit.
Run demo: python -B -m DATA_STRUCTURES.dfs
"""
from dataclasses import dataclass

if __package__:
    from .trace import emit
    from .graph import Graph
    from .hash_table import HashTable
else:
    from trace import emit
    from graph import Graph
    from hash_table import HashTable


@dataclass(slots=True)
class _Frame:
    vertex: str
    neighbors: object
    next: object = None


@dataclass(frozen=True)
class DFSResult:
    order: list
    parents: HashTable


def dfs(graph, start):
    """Input Graph/start ID -> discovery order and DFS parent table.

    Follows adjacency insertion order within the reachable component. Parent
    paths are traversal paths, not necessarily shortest paths. Missing start:
    KeyError. Wrong graph type: TypeError. Graph mutation is not supported.
    """
    if not isinstance(graph, Graph):
        raise TypeError('graph must be a Graph')
    if not graph.has_vertex(start):
        raise KeyError(start)
    parents = HashTable()
    parents.insert(start, None)
    order = [start]
    stack = _Frame(start, graph.neighbors(start))
    emit('DFS', 'push', current=start, visited_add=start)
    while stack is not None:
        try:
            neighbor = next(stack.neighbors)
        except StopIteration:
            emit('DFS', 'pop', current=stack.vertex)
            stack = stack.next
            continue
        if neighbor not in parents:
            parents.insert(neighbor, stack.vertex)
            order.append(neighbor)
            stack = _Frame(neighbor, graph.neighbors(neighbor), stack)
            emit('DFS', 'push', current=neighbor, parent=stack.next.vertex, visited_add=neighbor)
    return DFSResult(order, parents)


if __name__ == '__main__':
    graph = Graph()
    for name in ('Synthetic A', 'Synthetic B', 'Synthetic C', 'Synthetic isolated'):
        graph.add_vertex(name)
    graph.add_edge('Synthetic A', 'Synthetic B', relationship='test_fixture', evidence='Synthetic demonstration')
    graph.add_edge('Synthetic B', 'Synthetic C', relationship='test_fixture', evidence='Synthetic demonstration')
    print(dfs(graph, 'Synthetic A').order)
