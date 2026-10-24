# quad_tree_grid_py

A region quadtree for storing 2D point data and answering axis-aligned bounding-box range queries. Standard library only, no dependencies.

## Usage

```python
from quad_tree_grid_py import QuadTreeGrid, BoundingBox, Point

qt = QuadTreeGrid(bounds=(-10, -10, 10, 10), capacity=4, max_depth=6)
qt.insert(0.0, 0.0, data="origin")
qt.insert(5.0, 5.0)
qt.insert(-3.0, 2.0, data="northwest")

box = QuadTreeGrid.make_box(-5, -5, 6, 6)
for point in qt.query(box):
    print(point.x, point.y, point.data)
```

## Why this exists

When you need to find all points inside an axis-aligned rectangle and a flat list scan is too slow, a region quadtree subdivides space into quadrants so a query only descends into boxes that intersect the search region. The trade-off: memory overhead for the tree structure and more code than a linear scan. This library targets small-to-medium in-memory datasets where zero dependencies matter more than peak throughput.

## Edge cases

Containment uses half-open intervals: a point on the left or bottom edge of a query box is included; a point on the right or top edge is not. This prevents the same point being returned twice for adjacent queries. The same rule applies internally when the tree splits, so a point on a split boundary lands in exactly one quadrant.

The tree will not split beyond `max_depth`. When a leaf at maximum depth exceeds its capacity, points accumulate in that leaf — queries still return them, just without further spatial subdivision. Inserting a point outside the root bounds raises `ValueError`.

## Exports

- `QuadTreeGrid` — the tree class. Constructor: `QuadTreeGrid(bounds=(left, bottom, right, top), capacity=8, max_depth=10)`. Methods: `insert(x, y, data=None)`, `query(box)`, `stats()`, `make_box(...)` (static).
- `BoundingBox` — frozen rectangle with `contains(x, y)` and `intersects(other)`.
- `Point` — frozen dataclass with `x`, `y`, `data` fields; iterable as `(x, y)`.
