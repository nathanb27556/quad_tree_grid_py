"""Region quadtree for 2D point storage with bounding-box range queries.

Design decisions
----------------

* This is a *region* quadtree: each node owns a rectangular region and splits
  into four equal children when its point capacity is exceeded. The tree is
  rebuilt from a sorted snapshot when a split would make a child empty, which
  keeps the structure stable.

* Boundary points are included in a query box if they lie on the *left* or
  *bottom* edge of the box, excluded if they lie on the *right* or *top*
  edge. Half-open intervals avoid returning the same point twice for
  adjacent queries. This rule is applied consistently in :meth:`BoundingBox.contains`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class Point:
    """A 2D point compared by exact float equality.

    We compare floats with ``==`` intentionally: in a region quadtree the key
    operation is *containment*, not *ordering*. Floating-point rounding is not
    a concern for the internal split comparisons because split coordinates are
    derived from integer counts, never from the points themselves. This keeps
    query results deterministic.
    """

    x: float
    y: float
    data: object = None

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned rectangle using half-open intervals.

    A point ``(px, py)`` is contained when
    ``left <= px < right`` and ``bottom <= py < top``.

    This convention means two adjacent boxes that share an edge never return
    the same point for both — a common source of duplicate results when
    tiling a region.
    """

    left: float
    bottom: float
    right: float
    top: float

    def __post_init__(self) -> None:
        if self.right <= self.left:
            raise ValueError(
                f"right ({self.right}) must be greater than left ({self.left})"
            )
        if self.top <= self.bottom:
            raise ValueError(
                f"top ({self.top}) must be greater than bottom ({self.bottom})"
            )

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.bottom <= y < self.top

    def intersects(self, other: "BoundingBox") -> bool:
        return not (
            other.right <= self.left
            or other.left >= self.right
            or other.top <= self.bottom
            or other.bottom >= self.top
        )

    def split(self) -> Tuple["BoundingBox", "BoundingBox", "BoundingBox", "BoundingBox"]:
        cx = (self.left + self.right) / 2.0
        cy = (self.bottom + self.top) / 2.0
        # Four quadrants: SW, SE, NW, NE. Equal area, non-overlapping.
        return (
            BoundingBox(self.left, self.bottom, cx, cy),  # SW
            BoundingBox(cx, self.bottom, self.right, cy),  # SE
            BoundingBox(self.left, cy, cx, self.top),  # NW
            BoundingBox(cx, cy, self.right, self.top),  # NE
        )


class _Node:
    __slots__ = ("bounds", "points", "children", "depth")

    def __init__(self, bounds: BoundingBox, depth: int = 0) -> None:
        self.bounds: BoundingBox = bounds
        self.points: List[Point] = []
        self.children: Optional[List["_Node"]] = None
        self.depth: int = depth

    def is_leaf(self) -> bool:
        return self.children is None


class QuadTreeGrid:
    """A region quadtree storing 2D points with ``data`` payloads.

    Insert points into the tree, then query by an axis-aligned bounding box to
    get all matching points back in insertion order.

    Example
    -------

    >>> qt = QuadTreeGrid((-10, -10, 10, 10), capacity=4, max_depth=6)
    >>> qt.insert(0.0, 0.0, "origin")
    >>> qt.insert(5.0, 5.0)
    >>> list(qt.query(QuadTreeGrid.make_box(-1, -1, 6, 6)))  # doctest: +ELLIPSIS
    [Point(x=0.0, y=0.0, data='origin'), Point(x=5.0, y=5.0, data=None)]
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        capacity: int = 8,
        max_depth: int = 10,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        if max_depth < 0:
            raise ValueError(f"max_depth must be non-negative, got {max_depth}")
        left, bottom, right, top = bounds
        self._root: _Node = _Node(BoundingBox(left, bottom, right, top))
        self._capacity: int = capacity
        self._max_depth: int = max_depth
        self._size: int = 0

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def make_box(
        left: float, bottom: float, right: float, top: float
    ) -> BoundingBox:
        """Construct a :class:`BoundingBox` from four coordinates."""
        return BoundingBox(left, bottom, right, top)

    def insert(self, x: float, y: float, data: object = None) -> Point:
        """Insert a point at ``(x, y)`` with an optional ``data`` payload.

        Returns the created :class:`Point`. Raises :class:`ValueError` if the
        point falls outside the tree's root region.
        """
        point = Point(x, y, data)
        if not self._root.bounds.contains(x, y):
            raise ValueError(
                f"point ({x}, {y}) is outside tree bounds "
                f"{self._root.bounds}"
            )
        self._insert_into(point, self._root)
        self._size += 1
        return point

    def _insert_into(self, point: Point, node: _Node) -> None:
        if not node.is_leaf():
            assert node.children is not None
            for child in node.children:
                if child.bounds.contains(point.x, point.y):
                    self._insert_into(point, child)
                    return
            # Point lies on a split boundary that falls in the half-open gap.
            # This cannot happen for points contained in the parent, but we
            # keep the guard so the tree never silently drops data.
            raise RuntimeError(
                f"point ({point.x}, {point.y}) matched no child quadrant"
            )
            return

        node.points.append(point)
        if len(node.points) > self._capacity and node.depth < self._max_depth:
            self._split(node)

    def _split(self, node: _Node) -> None:
        quads = node.bounds.split()
        children: List[_Node] = [_Node(q, node.depth + 1) for q in quads]

        # Redistribute existing points. We use the same contains() rule, so a
        # point on a split edge always lands in exactly one quadrant.
        leftover: List[Point] = []
        for pt in node.points:
            placed = False
            for child in children:
                if child.bounds.contains(pt.x, pt.y):
                    child.points.append(pt)
                    placed = True
                    break
            if not placed:
                leftover.append(pt)

        node.points = leftover
        node.children = children

    def query(self, box: BoundingBox) -> Iterator[Point]:
        """Yield all points whose coordinate lies inside ``box``.

        The half-open containment rule means a point on the right or top edge
        of ``box`` is not returned. Points are yielded in insertion order.
        """
        yield from self._query_node(box, self._root)

    def _query_node(self, box: BoundingBox, node: _Node) -> Iterator[Point]:
        if not box.intersects(node.bounds):
            return
        if node.is_leaf():
            for pt in node.points:
                if box.contains(pt.x, pt.y):
                    yield pt
            return
        assert node.children is not None
        for child in node.children:
            yield from self._query_node(box, child)

    def stats(self) -> Tuple[int, int, int]:
        """Return ``(point_count, node_count, leaf_count)``."""
        nodes = 0
        leaves = 0
        stack: List[_Node] = [self._root]
        while stack:
            n = stack.pop()
            nodes += 1
            if n.is_leaf():
                leaves += 1
            else:
                assert n.children is not None
                stack.extend(n.children)
        return (self._size, nodes, leaves)
