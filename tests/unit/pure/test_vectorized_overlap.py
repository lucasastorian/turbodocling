"""Tests for vectorized overlap detection used in layout postprocessing."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from processor.postprocess_service.layout_postprocessor import (
    vectorized_overlap_check_with_arrays,
)


class TestVectorizedOverlapCheck:

    def _make_arrays(self, boxes_list):
        """Build packed arrays from list of (l, t, r, b) tuples."""
        boxes = np.array(boxes_list, dtype=np.float32)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes, areas

    def test_identical_boxes_full_overlap(self):
        boxes, areas = self._make_arrays([
            [0, 0, 100, 100],
            [0, 0, 100, 100],
        ])
        mask = vectorized_overlap_check_with_arrays(
            0, [1], boxes, areas, ovlp_thr=0.5, cont_thr=0.8
        )
        assert mask[0] == True

    def test_no_overlap(self):
        boxes, areas = self._make_arrays([
            [0, 0, 10, 10],
            [90, 90, 100, 100],
        ])
        mask = vectorized_overlap_check_with_arrays(
            0, [1], boxes, areas, ovlp_thr=0.5, cont_thr=0.8
        )
        assert mask[0] == False

    def test_partial_overlap_below_threshold(self):
        boxes, areas = self._make_arrays([
            [0, 0, 100, 100],
            [80, 80, 120, 120],  # small overlap corner
        ])
        mask = vectorized_overlap_check_with_arrays(
            0, [1], boxes, areas, ovlp_thr=0.5, cont_thr=0.8
        )
        assert mask[0] == False

    def test_containment_triggers(self):
        """Small box fully inside large box should trigger containment."""
        boxes, areas = self._make_arrays([
            [0, 0, 100, 100],   # large
            [30, 30, 40, 40],   # small, fully contained
        ])
        mask = vectorized_overlap_check_with_arrays(
            0, [1], boxes, areas, ovlp_thr=0.5, cont_thr=0.8
        )
        assert mask[0] == True

    def test_multiple_candidates(self):
        boxes, areas = self._make_arrays([
            [0, 0, 100, 100],     # query
            [0, 0, 100, 100],     # full overlap
            [200, 200, 300, 300], # no overlap
            [10, 10, 90, 90],     # contained
        ])
        mask = vectorized_overlap_check_with_arrays(
            0, [1, 2, 3], boxes, areas, ovlp_thr=0.5, cont_thr=0.8
        )
        assert mask[0] == True   # full overlap
        assert mask[1] == False  # no overlap
        assert mask[2] == True   # contained

    def test_empty_candidates(self):
        boxes, areas = self._make_arrays([[0, 0, 10, 10]])
        mask = vectorized_overlap_check_with_arrays(
            0, [], boxes, areas, ovlp_thr=0.5, cont_thr=0.8
        )
        assert len(mask) == 0

    def test_zero_area_box(self):
        """Zero-area boxes should not trigger overlap."""
        boxes, areas = self._make_arrays([
            [0, 0, 100, 100],
            [50, 50, 50, 50],  # zero area
        ])
        mask = vectorized_overlap_check_with_arrays(
            0, [1], boxes, areas, ovlp_thr=0.5, cont_thr=0.8
        )
        # Division by zero area should not crash
        assert isinstance(mask[0], (bool, np.bool_))

    def test_with_temp_arrays(self):
        """Pre-allocated temp arrays should produce same results."""
        boxes, areas = self._make_arrays([
            [0, 0, 100, 100],
            [0, 0, 100, 100],
            [200, 200, 300, 300],
        ])
        temp = tuple(np.empty(256, dtype=np.float32) for _ in range(9))

        mask_with = vectorized_overlap_check_with_arrays(
            0, [1, 2], boxes, areas, 0.5, 0.8, temp
        )
        mask_without = vectorized_overlap_check_with_arrays(
            0, [1, 2], boxes, areas, 0.5, 0.8, None
        )
        np.testing.assert_array_equal(mask_with, mask_without)
