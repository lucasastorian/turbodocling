"""Pure logic tests for LayoutPostprocessor invariants.

Tests focus on determinism, bbox containment, cell deduplication,
and overlap detection — without requiring GPU or model inference.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from processor.postprocess_service.layout_postprocessor import (
    UnionFind,
    IntervalTree,
    GridIndex,
    SpatialClusterIndex,
)


class TestUnionFind:

    def test_singleton_groups(self):
        uf = UnionFind([1, 2, 3])
        groups = uf.get_groups()
        assert len(groups) == 3

    def test_union_merges_groups(self):
        uf = UnionFind([1, 2, 3, 4])
        uf.union(1, 2)
        uf.union(3, 4)
        groups = uf.get_groups()
        assert len(groups) == 2

    def test_transitive_union(self):
        uf = UnionFind([1, 2, 3])
        uf.union(1, 2)
        uf.union(2, 3)
        groups = uf.get_groups()
        assert len(groups) == 1
        assert set(list(groups.values())[0]) == {1, 2, 3}

    def test_idempotent_union(self):
        uf = UnionFind([1, 2])
        uf.union(1, 2)
        uf.union(1, 2)
        uf.union(2, 1)
        groups = uf.get_groups()
        assert len(groups) == 1

    def test_find_with_path_compression(self):
        uf = UnionFind(range(100))
        for i in range(1, 100):
            uf.union(0, i)
        # After path compression, all should point to same root
        root = uf.find(0)
        for i in range(100):
            assert uf.find(i) == root


class TestIntervalTree:

    def test_empty_tree(self):
        tree = IntervalTree()
        assert tree.find_containing(5.0) == set()

    def test_single_interval_contains_point(self):
        tree = IntervalTree()
        tree.insert(0.0, 10.0, 1)
        assert tree.find_containing(5.0) == {1}

    def test_point_outside_interval(self):
        tree = IntervalTree()
        tree.insert(0.0, 10.0, 1)
        assert tree.find_containing(15.0) == set()

    def test_overlapping_intervals(self):
        tree = IntervalTree()
        tree.insert(0.0, 10.0, 1)
        tree.insert(5.0, 15.0, 2)
        result = tree.find_containing(7.0)
        assert result == {1, 2}

    def test_boundary_inclusion(self):
        tree = IntervalTree()
        tree.insert(0.0, 10.0, 1)
        assert 1 in tree.find_containing(0.0)
        assert 1 in tree.find_containing(10.0)


class TestGridIndex:

    def test_empty_grid(self):
        grid = GridIndex(100.0, 100.0, 10.0, 10.0)
        assert grid.candidates(0, 0, 5, 5) == set()

    def test_insert_and_find(self):
        grid = GridIndex(100.0, 100.0, 50.0, 50.0)
        grid.insert(1, 10.0, 10.0, 30.0, 30.0)
        cands = grid.candidates(15.0, 15.0, 25.0, 25.0)
        assert 1 in cands

    def test_non_overlapping_not_found(self):
        grid = GridIndex(100.0, 100.0, 50.0, 50.0)
        grid.insert(1, 10.0, 10.0, 20.0, 20.0)
        cands = grid.candidates(80.0, 80.0, 90.0, 90.0)
        assert 1 not in cands

    def test_multiple_candidates(self):
        grid = GridIndex(100.0, 100.0, 50.0, 50.0)
        grid.insert(1, 10.0, 10.0, 30.0, 30.0)
        grid.insert(2, 20.0, 20.0, 40.0, 40.0)
        cands = grid.candidates(15.0, 15.0, 35.0, 35.0)
        assert cands == {1, 2}

    def test_degenerate_zero_area_query(self):
        grid = GridIndex(100.0, 100.0, 10.0, 10.0)
        grid.insert(1, 10.0, 10.0, 20.0, 20.0)
        # Zero-area query should return empty
        assert grid.candidates(15.0, 15.0, 15.0, 15.0) == set()

    def test_tiny_bins(self):
        """Grid with very small bins should still work."""
        grid = GridIndex(100.0, 100.0, 1.0, 1.0)
        grid.insert(1, 50.0, 50.0, 51.0, 51.0)
        cands = grid.candidates(50.0, 50.0, 51.0, 51.0)
        assert 1 in cands


class TestSpatialClusterIndex:

    def _make_cluster(self, id, l, t, r, b, label=None, confidence=0.9):
        """Create a minimal cluster-like object for testing."""
        from docling.datamodel.base_models import BoundingBox, Cluster
        from docling_core.types.doc import DocItemLabel
        bbox = BoundingBox.model_construct(l=l, t=t, r=r, b=b)
        return Cluster(
            id=id,
            label=label or DocItemLabel.TEXT,
            bbox=bbox,
            confidence=confidence,
            cells=[],
        )

    def test_find_overlapping(self):
        c1 = self._make_cluster(1, 0, 0, 50, 50)
        c2 = self._make_cluster(2, 25, 25, 75, 75)
        idx = SpatialClusterIndex([c1, c2])
        cands = idx.find_candidates(c1.bbox)
        assert 2 in cands

    def test_remove_cluster(self):
        c1 = self._make_cluster(1, 0, 0, 50, 50)
        idx = SpatialClusterIndex([c1])
        idx.remove_cluster(c1)
        assert 1 not in idx.clusters_by_id

    def test_check_overlap_high_iou(self):
        c1 = self._make_cluster(1, 0, 0, 100, 100)
        c2 = self._make_cluster(2, 5, 5, 95, 95)
        idx = SpatialClusterIndex([c1, c2])
        assert idx.check_overlap(c1.bbox, c2.bbox, 0.5, 0.8)

    def test_check_overlap_no_overlap(self):
        c1 = self._make_cluster(1, 0, 0, 10, 10)
        c2 = self._make_cluster(2, 90, 90, 100, 100)
        idx = SpatialClusterIndex([c1, c2])
        assert not idx.check_overlap(c1.bbox, c2.bbox, 0.5, 0.8)
