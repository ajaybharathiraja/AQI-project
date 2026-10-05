"""Manual BFS with custom CircularQueue and HashTable visited/parent state.

Expected O(V+E) time for bounded IDs, O(V) auxiliary/output space, including
queue capacity for all graph vertices. Collision-heavy hash lookups can worsen
time to O((V+E)*V). Run demo: python -B -m DATA_STRUCTURES.bfs
"""
from dataclasses import dataclass

if __package__:
    from .trace import emit
    from .graph import Graph
    from .hash_table import HashTable
    from .circular_queue import CircularQueue
else:
    from trace import emit
    from graph import Graph
    from hash_table import HashTable
    from circular_queue import CircularQueue


@dataclass(frozen=True)
class BFSResult:
    order: list
    parents: HashTable
    distances: HashTable


def bfs(graph, start):
    """Input Graph/start ID -> order, parent table, shortest unweighted hop table.

    Traverses only the start's reachable component, honors edge direction and
    handles cycles by marking on enqueue. Missing start: KeyError. Wrong graph
    type: TypeError. Distance weights, if any, are deliberately not optimized.
    """
    if not isinstance(graph, Graph):
        raise TypeError('graph must be a Graph')
    if not graph.has_vertex(start):
        raise KeyError(start)
    parents, distances = HashTable(), HashTable()
    parents.insert(start, None)
    distances.insert(start, 0)
    queue = CircularQueue(len(graph))
    queue.enqueue(start)
    emit('BFS', 'enqueue', node=start, visited_add=start, queue_size=len(queue))
    order = []
    while not queue.is_empty():
        current = queue.dequeue()
        order.append(current)
        emit('BFS', 'visit', current=current, queue_size=len(queue), visit_index=len(order)-1)
        for neighbor in graph.neighbors(current):
            if neighbor not in parents:
                parents.insert(neighbor, current)
                distances.insert(neighbor, distances.search(current) + 1)
                queue.enqueue(neighbor)
                emit('BFS', 'enqueue', node=neighbor, parent=current, visited_add=neighbor, queue_size=len(queue))
    return BFSResult(order, parents, distances)


if __name__ == '__main__':
    graph = Graph()
    for name in ('Synthetic A', 'Synthetic B', 'Synthetic C'):
        graph.add_vertex(name)
    graph.add_edge('Synthetic A', 'Synthetic B', relationship='test_fixture', evidence='Synthetic edge for algorithm demonstration')
    graph.add_edge('Synthetic B', 'Synthetic C', relationship='test_fixture', evidence='Synthetic edge for algorithm demonstration')
    result = bfs(graph, 'Synthetic A')
    print(result.order, list(result.distances.items()))
