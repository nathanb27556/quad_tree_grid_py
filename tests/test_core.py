import unittest

from quad_tree_grid_py import QuadTreeGrid, BoundingBox, Point


class TestBoundingBox(unittest.TestCase):
    def test_contains_half_open(self):
        bb = BoundingBox(0.0, 0.0, 10.0, 10.0)
        self.assertTrue(bb.contains(0.0, 0.0))
        self.assertTrue(bb.contains(5.0, 5.0))
        self.assertFalse(bb.contains(10.0, 5.0))
        self.assertFalse(bb.contains(5.0, 10.0))
        self.assertFalse(bb.contains(-0.1, 5.0))

    def test_split_produces_four_non_overlapping(self):
        bb = BoundingBox(0.0, 0.0, 10.0, 10.0)
        sw, se, nw, ne = bb.split()
        self.assertEqual(sw.right, se.left)
        self.assertEqual(sw.top, nw.bottom)
        # Each child is half the area.
        for child in (sw, se, nw, ne):
            self.assertAlmostEqual(
                (child.right - child.left) * (child.top - child.bottom),
                25.0,
            )

    def test_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            BoundingBox(10.0, 0.0, 0.0, 10.0)
        with self.assertRaises(ValueError):
            BoundingBox(0.0, 10.0, 10.0, 0.0)


class TestInsert(unittest.TestCase):
    def setUp(self):
        self.qt = QuadTreeGrid((-10.0, -10.0, 10.0, 10.0), capacity=2, max_depth=4)

    def test_insert_returns_point(self):
        p = self.qt.insert(1.0, 2.0, "hello")
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 2.0)
        self.assertEqual(p.data, "hello")

    def test_insert_outside_bounds(self):
        with self.assertRaises(ValueError):
            self.qt.insert(20.0, 0.0)
        with self.assertRaises(ValueError):
            self.qt.insert(0.0, -20.0)

    def test_size_tracks_insertions(self):
        self.assertEqual(len(self.qt), 0)
        self.qt.insert(0.0, 0.0)
        self.qt.insert(1.0, 1.0)
        self.assertEqual(len(self.qt), 2)


class TestQuery(unittest.TestCase):
    def _build(self):
        qt = QuadTreeGrid((0.0, 0.0, 100.0, 100.0), capacity=2, max_depth=5)
        qt.insert(10.0, 10.0, "a")
        qt.insert(20.0, 20.0, "b")
        qt.insert(30.0, 30.0, "c")
        qt.insert(40.0, 40.0, "d")
        qt.insert(50.0, 50.0, "e")
        return qt

    def test_query_subset(self):
        qt = self._build()
        result = list(qt.query(BoundingBox(0.0, 0.0, 25.0, 25.0)))
        self.assertEqual(result, [Point(10.0, 10.0, "a"), Point(20.0, 20.0, "b")])

    def test_query_all(self):
        qt = self._build()
        result = list(qt.query(BoundingBox(0.0, 0.0, 100.0, 100.0)))
        self.assertEqual(len(result), 5)

    def test_query_empty_region(self):
        qt = self._build()
        result = list(qt.query(BoundingBox(70.0, 70.0, 80.0, 80.0)))
        self.assertEqual(result, [])

    def test_query_half_open_excludes_top_right(self):
        qt = QuadTreeGrid((0.0, 0.0, 100.0, 100.0))
        qt.insert(10.0, 10.0)
        qt.insert(10.0, 20.0)
        qt.insert(20.0, 10.0)
        # Box (0,0)-(20,20) excludes points on the right/top edges.
        result = list(qt.query(BoundingBox(0.0, 0.0, 20.0, 20.0)))
        self.assertEqual(result, [Point(10.0, 10.0, None)])

    def test_query_no_split_single_node(self):
        qt = QuadTreeGrid((0.0, 0.0, 100.0, 100.0), capacity=10)
        qt.insert(5.0, 5.0, "only")
        result = list(qt.query(BoundingBox(0.0, 0.0, 10.0, 10.0)))
        self.assertEqual(result, [Point(5.0, 5.0, "only")])


class TestSplit(unittest.TestCase):
    def test_capacity_triggers_split(self):
        qt = QuadTreeGrid((0.0, 0.0, 100.0, 100.0), capacity=2, max_depth=4)
        for i in range(10):
            qt.insert(float(i), float(i))
        count, nodes, leaves = qt.stats()
        self.assertEqual(count, 10)
        self.assertGreater(nodes, 1)
        self.assertGreater(leaves, 1)

    def test_max_depth_prevents_split(self):
        qt = QuadTreeGrid((0.0, 0.0, 100.0, 100.0), capacity=1, max_depth=0)
        for i in range(5):
            qt.insert(float(i * 10), float(i * 10))
        count, nodes, leaves = qt.stats()
        self.assertEqual(count, 5)
        self.assertEqual(nodes, 1)
        self.assertEqual(leaves, 1)

    def test_deep_tree_finds_all(self):
        qt = QuadTreeGrid((0.0, 0.0, 100.0, 100.0), capacity=1, max_depth=6)
        inserted = []
        for i in range(20):
            x = float(i * 5)
            y = float(i * 5 % 100)
            qt.insert(x, y)
            inserted.append(Point(x, y, None))
        # Insertion order is preserved across quadrants.
        result = list(qt.query(BoundingBox(0.0, 0.0, 100.0, 100.0)))
        self.assertEqual(result, inserted)

    def test_point_on_split_boundary_lands_once(self):
        # With capacity 1, inserting two points at x=50 forces a split at x=50.
        qt = QuadTreeGrid((0.0, 0.0, 100.0, 100.0), capacity=1, max_depth=2)
        qt.insert(50.0, 50.0, "center")
        qt.insert(50.0, 60.0, "shifted")
        result = list(qt.query(BoundingBox(0.0, 0.0, 100.0, 100.0)))
        self.assertEqual(len(result), 2)
        # Each point should appear exactly once.
        self.assertEqual(len(result), len(set(result)))


class TestStats(unittest.TestCase):
    def test_empty_tree(self):
        qt = QuadTreeGrid((0.0, 0.0, 100.0, 100.0))
        self.assertEqual(qt.stats(), (0, 1, 1))

    def test_make_box_helper(self):
        bb = QuadTreeGrid.make_box(1.0, 2.0, 3.0, 4.0)
        self.assertIsInstance(bb, BoundingBox)
        self.assertEqual(bb.left, 1.0)
        self.assertEqual(bb.bottom, 2.0)
        self.assertEqual(bb.right, 3.0)
        self.assertEqual(bb.top, 4.0)


class TestValidation(unittest.TestCase):
    def test_invalid_capacity(self):
        with self.assertRaises(ValueError):
            QuadTreeGrid((0.0, 0.0, 10.0, 10.0), capacity=0)

    def test_negative_max_depth(self):
        with self.assertRaises(ValueError):
            QuadTreeGrid((0.0, 0.0, 10.0, 10.0), max_depth=-1)


if __name__ == "__main__":
    unittest.main()
