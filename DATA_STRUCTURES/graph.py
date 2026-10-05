"""Monitoring graph: custom hash-table vertices and linked adjacency chains.

Expected vertex lookup O(key length+1). Add vertex amortized expected O(1) for
bounded IDs; add edge O(deg(u)+deg(v)) for duplicate checks (undirected).
Neighbor iteration O(deg(u)); graph space O(V+E). Hash collisions can worsen
lookup to O(V). Run demo: python -B -m DATA_STRUCTURES.graph
"""
from dataclasses import dataclass
import math

if __package__:
    from .hash_table import HashTable
else:
    from hash_table import HashTable


@dataclass(frozen=True)
class Edge:
    target: str
    relationship: str
    evidence: str
    distance_km: float | None = None


@dataclass(slots=True)
class _Link:
    edge: Edge
    next: object = None


@dataclass(slots=True)
class _Vertex:
    name: str
    latitude: float | None
    longitude: float | None
    coordinate_source: str | None
    head: object = None
    tail: object = None


class Graph:
    """Vertex IDs are strings; edges require caller-supplied relationship evidence.

    Labels alone never imply spatial proximity. add_spatial_edge computes an edge
    only from explicitly sourced coordinates inside a declared radius. Evidence
    truth is the caller's responsibility. Synthetic examples are not real data.
    Unknown vertices: KeyError; duplicate vertices: KeyError; invalid inputs,
    self-loops or duplicate edges: ValueError. Mutation during traversal is not
    supported. neighbors returns IDs in edge insertion order.
    """

    def __init__(self, *, directed=False):
        if type(directed) is not bool:
            raise ValueError('directed must be boolean')
        self._directed = directed
        self._vertices = HashTable()
        self._edge_count = 0

    def __len__(self):
        return len(self._vertices)

    @property
    def edge_count(self):
        return self._edge_count

    @staticmethod
    def _text(value, label):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{label} must be a nonempty string')

    def add_vertex(self, name, *, latitude=None, longitude=None, coordinate_source=None):
        """Insert isolated vertex; optionally supply paired sourced coordinates."""
        self._text(name, 'vertex ID')
        if latitude is not None or longitude is not None:
            if type(latitude) not in (int, float) or type(longitude) not in (int, float):
                raise ValueError('both coordinates must be finite numbers')
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError('latitude/longitude outside geographic range')
            self._text(coordinate_source, 'coordinate source')
        elif coordinate_source is not None:
            raise ValueError('coordinate source supplied without coordinates')
        self._vertices.insert(name, _Vertex(name, latitude, longitude, coordinate_source))

    def has_vertex(self, name):
        return name in self._vertices

    def vertices(self):
        """Yield all IDs in unspecified hash order, O(bucket capacity+V)."""
        for name, _ in self._vertices.items():
            yield name

    def edges_from(self, name):
        """Yield immutable Edge records; O(degree) time, O(1) auxiliary space."""
        node = self._vertices.search(name).head
        while node is not None:
            yield node.edge
            node = node.next

    def neighbors(self, name):
        """Yield neighboring IDs in insertion order without materializing a list."""
        for edge in self.edges_from(name):
            yield edge.target

    @staticmethod
    def _append(vertex, edge):
        node = _Link(edge)
        if vertex.tail is None:
            vertex.head = node
        else:
            vertex.tail.next = node
        vertex.tail = node

    def _add_edge(self, source, target, relationship, evidence, distance=None):
        first = self._vertices.search(source)
        second = self._vertices.search(target)
        self._text(relationship, 'relationship')
        self._text(evidence, 'evidence')
        if source == target:
            raise ValueError('self-loops are not supported')
        if any(v == target for v in self.neighbors(source)):
            raise ValueError('edge already exists')
        if not self._directed and any(v == source for v in self.neighbors(target)):
            raise ValueError('reverse edge already exists')
        self._append(first, Edge(target, relationship, evidence, distance))
        if not self._directed:
            self._append(second, Edge(source, relationship, evidence, distance))
        self._edge_count += 1

    def add_edge(self, source, target, *, relationship, evidence):
        """Add an explicitly documented non-distance relationship; returns None.

        'spatial_distance' is reserved for the coordinate-validated method.
        """
        if relationship == 'spatial_distance':
            raise ValueError('use add_spatial_edge for distance relationships')
        self._add_edge(source, target, relationship, evidence)

    def add_spatial_edge(self, source, target, *, max_distance_km):
        """Insert a within-radius edge; return computed great-circle km.

        Requires coordinate sources at both endpoints. Haversine uses spherical
        mean radius 6371.0088 km; distance is approximate, not a road/transport
        path. Missing coordinates or an exceeded threshold raises ValueError.
        """
        if type(max_distance_km) not in (int, float):
            raise ValueError('max_distance_km must be a finite positive number')
        try:
            valid = math.isfinite(max_distance_km) and max_distance_km > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError('max_distance_km must be a finite positive number')
        a, b = self._vertices.search(source), self._vertices.search(target)
        if a.latitude is None or b.latitude is None:
            raise ValueError('spatial edges require sourced coordinates at both endpoints')
        lat1, lat2 = math.radians(a.latitude), math.radians(b.latitude)
        dlat = lat2 - lat1
        dlon = math.radians(b.longitude - a.longitude)
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        h = min(1.0, max(0.0, h))  # Clamp floating-point roundoff near antipodes.
        distance = 6371.0088 * 2 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
        if distance > max_distance_km:
            raise ValueError('stations exceed the selected spatial radius')
        evidence = f'{a.coordinate_source}; {b.coordinate_source}; radius <= {max_distance_km} km'
        self._add_edge(source, target, 'spatial_distance', evidence, distance)
        return distance


if __name__ == '__main__':
    graph = Graph()
    graph.add_vertex('Synthetic A', latitude=0, longitude=0, coordinate_source='Synthetic fixture')
    graph.add_vertex('Synthetic B', latitude=0, longitude=1, coordinate_source='Synthetic fixture')
    print('Synthetic distance km:', graph.add_spatial_edge('Synthetic A', 'Synthetic B', max_distance_km=112))
    print(list(graph.edges_from('Synthetic A')))
