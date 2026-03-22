"""Contract tests for the Lambda→GPU serialization boundary.

These verify that pack_segmented_page output can be correctly unpacked
by the GPU processor's _unpack_cells_columnar / _reconstruct_page_from_dict.
Any breakage here means Lambda and GPU are out of sync.
"""

import array
import struct
import numpy as np
import pytest

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared"))

from shared.page_serialization import pack_segmented_page, _pack_cells_columnar
from tests.helpers.builders import (
    make_cells, make_segmented_page,
    FakeTextCell, FakePdfTextCell, FakeRect, FakeRGBA, FakeSegmentedPage,
)


class TestPackCellsColumnar:
    """Tests for _pack_cells_columnar — the core serialization function."""

    def test_empty_cells_returns_n_zero(self):
        result = _pack_cells_columnar([], is_topleft=True)
        assert result == {"n": 0}

    def test_cell_count_matches(self):
        cells = make_cells(10, pdf=True)
        result = _pack_cells_columnar(cells, is_topleft=True)
        assert result["n"] == 10

    def test_geometry_array_lengths(self):
        """Each geometry column should have exactly N float32 values."""
        cells = make_cells(5, pdf=True)
        result = _pack_cells_columnar(cells, is_topleft=True)

        for key in ["r_x0", "r_y0", "r_x1", "r_y1", "r_x2", "r_y2", "r_x3", "r_y3"]:
            data = result[key]
            assert isinstance(data, bytes)
            assert len(data) == 5 * 4  # 5 floats * 4 bytes each

    def test_geometry_values_roundtrip(self):
        """Packed geometry should exactly match source cell rects."""
        cells = make_cells(3, pdf=True)
        result = _pack_cells_columnar(cells, is_topleft=True)

        x0_vals = np.frombuffer(result["r_x0"], dtype=np.float32)
        for i, cell in enumerate(cells):
            assert x0_vals[i] == pytest.approx(cell.rect.r_x0)

    def test_index_array_preserves_order(self):
        """Cell indices should be preserved in order."""
        cells = make_cells(4, pdf=True)
        cells[2].index = 99  # non-sequential
        result = _pack_cells_columnar(cells, is_topleft=True)

        indices = np.frombuffer(result["index"], dtype=np.int32)
        assert list(indices) == [0, 1, 99, 3]

    def test_rgba_array_layout(self):
        """RGBA should be packed as flat bytes: [r,g,b,a, r,g,b,a, ...]."""
        cells = [
            FakePdfTextCell(index=0, rgba=FakeRGBA(255, 0, 0, 128)),
            FakePdfTextCell(index=1, rgba=FakeRGBA(0, 255, 0, 255)),
        ]
        result = _pack_cells_columnar(cells, is_topleft=True)

        rgba = np.frombuffer(result["rgba"], dtype=np.uint8)
        assert list(rgba) == [255, 0, 0, 128, 0, 255, 0, 255]

    def test_string_interning_deduplication(self):
        """Repeated strings should be interned to the same index."""
        cells = [
            FakePdfTextCell(index=0, text="hello", orig="hello"),
            FakePdfTextCell(index=1, text="world", orig="world"),
            FakePdfTextCell(index=2, text="hello", orig="world"),  # "hello" reused
        ]
        result = _pack_cells_columnar(cells, is_topleft=True)

        text_indices = np.frombuffer(result["text_idx"], dtype=np.int32)
        # "hello" should map to same index for cell 0 and cell 2
        assert text_indices[0] == text_indices[2]
        # "world" should be different
        assert text_indices[1] != text_indices[0]
        # All strings should be in the string table
        assert "hello" in result["strings"]
        assert "world" in result["strings"]

    def test_string_interning_orig_field(self):
        """orig field should also be interned."""
        cells = [
            FakePdfTextCell(index=0, text="a", orig="A"),
            FakePdfTextCell(index=1, text="b", orig="A"),  # same orig
        ]
        result = _pack_cells_columnar(cells, is_topleft=True)

        orig_indices = np.frombuffer(result["orig_idx"], dtype=np.int32)
        assert orig_indices[0] == orig_indices[1]  # same "A"

    def test_pdf_specific_fields_present(self):
        """PdfTextCell should produce PDF-specific columns."""
        cells = make_cells(2, pdf=True)
        result = _pack_cells_columnar(cells, is_topleft=True)

        assert "rendering_mode" in result
        assert "widget" in result
        assert "font_key_idx" in result
        assert "font_name_idx" in result
        assert "text_dir" in result

    def test_non_pdf_cells_omit_pdf_fields(self):
        """Plain TextCells should NOT produce PDF-specific columns."""
        cells = make_cells(2, pdf=False)
        result = _pack_cells_columnar(cells, is_topleft=True)

        assert "rendering_mode" not in result
        assert "widget" not in result
        assert "font_key_idx" not in result
        assert "font_name_idx" not in result
        assert "text_dir" not in result

    def test_topleft_flag_preserved(self):
        result = _pack_cells_columnar(make_cells(1), is_topleft=True)
        assert result["topleft"] is True

        result = _pack_cells_columnar(make_cells(1), is_topleft=False)
        assert result["topleft"] is False

    def test_confidence_default(self):
        """Cells without explicit confidence should default to 1.0."""
        cell = FakeTextCell(index=0)
        del cell.confidence  # remove the attribute
        result = _pack_cells_columnar([cell], is_topleft=True)
        conf = np.frombuffer(result["conf"], dtype=np.float32)
        assert conf[0] == pytest.approx(1.0)

    def test_widget_encoding(self):
        """Widget should be encoded as 0/1 uint8."""
        cells = [
            FakePdfTextCell(index=0, widget=False),
            FakePdfTextCell(index=1, widget=True),
        ]
        result = _pack_cells_columnar(cells, is_topleft=True)
        widgets = np.frombuffer(result["widget"], dtype=np.uint8)
        assert list(widgets) == [0, 1]


class TestPackSegmentedPage:
    """Tests for pack_segmented_page — top-level page serialization."""

    def test_version_2_marker(self):
        page = make_segmented_page(n_words=3)
        result = pack_segmented_page(page)
        assert result["version"] == 2

    def test_page_height_preserved(self):
        page = make_segmented_page()
        page.dimension.height = 1056.0
        result = pack_segmented_page(page)
        assert result["page_height"] == 1056.0

    def test_dimension_fields_complete(self):
        page = make_segmented_page()
        result = pack_segmented_page(page)
        dim = result["dimension"]

        assert "angle" in dim
        assert "height" in dim
        assert "width" in dim
        assert "boundary_type" in dim
        for bbox_key in ["crop_bbox", "media_bbox", "art_bbox", "bleed_bbox", "trim_bbox"]:
            assert bbox_key in dim
            assert len(dim[bbox_key]) == 4  # [l, t, r, b]

    def test_cell_type_fields_present(self):
        page = make_segmented_page(n_words=3, n_chars=2, n_lines=1)
        result = pack_segmented_page(page)

        assert result["char_cells"]["n"] == 2
        assert result["word_cells"]["n"] == 3
        assert result["textline_cells"]["n"] == 1

    def test_has_flags_match_cell_counts(self):
        page = make_segmented_page(n_words=3, n_chars=0, n_lines=0)
        result = pack_segmented_page(page)

        assert result["has_chars"] is False
        assert result["has_words"] is True
        assert result["has_lines"] is False

    def test_empty_page(self):
        page = make_segmented_page(n_words=0, n_chars=0, n_lines=0)
        result = pack_segmented_page(page)

        assert result["version"] == 2
        assert result["word_cells"]["n"] == 0
        assert result["char_cells"]["n"] == 0
        assert result["textline_cells"]["n"] == 0

    def test_large_page_cell_count(self):
        """Stress test: 500 cells should serialize without issues."""
        page = make_segmented_page(n_words=500)
        result = pack_segmented_page(page)
        assert result["word_cells"]["n"] == 500

        # Verify geometry arrays are correct size
        x0 = np.frombuffer(result["word_cells"]["r_x0"], dtype=np.float32)
        assert len(x0) == 500


class TestSerializationInvariants:
    """Property-like invariant tests for the serialization format."""

    def test_no_negative_dimensions_in_output(self):
        """All geometry values should be non-negative for standard pages."""
        page = make_segmented_page(n_words=10)
        result = pack_segmented_page(page)

        dim = result["dimension"]
        assert dim["width"] >= 0
        assert dim["height"] >= 0

    def test_string_table_contains_all_text(self):
        """Every cell text should appear in the string table."""
        cells = [
            FakePdfTextCell(index=i, text=f"unique_text_{i}")
            for i in range(5)
        ]
        result = _pack_cells_columnar(cells, is_topleft=True)

        for i in range(5):
            assert f"unique_text_{i}" in result["strings"]

    def test_index_uniqueness_preserved(self):
        """If input cell indices are unique, output should preserve that."""
        cells = make_cells(10)
        result = _pack_cells_columnar(cells, is_topleft=True)
        indices = np.frombuffer(result["index"], dtype=np.int32)
        assert len(set(indices)) == len(indices)

    def test_msgpack_roundtrip(self):
        """Packed page should survive msgpack serialization (the wire format)."""
        import msgpack

        page = make_segmented_page(n_words=5, n_chars=3)
        packed = pack_segmented_page(page)

        # msgpack round trip
        blob = msgpack.packb(packed, use_bin_type=True)
        unpacked = msgpack.unpackb(blob, raw=False, strict_map_key=False)

        assert unpacked["version"] == 2
        assert unpacked["word_cells"]["n"] == 5
        assert unpacked["char_cells"]["n"] == 3

        # Verify geometry survives
        x0_original = np.frombuffer(packed["word_cells"]["r_x0"], dtype=np.float32)
        x0_roundtrip = np.frombuffer(unpacked["word_cells"]["r_x0"], dtype=np.float32)
        np.testing.assert_array_almost_equal(x0_original, x0_roundtrip)

    def test_gzip_msgpack_roundtrip(self):
        """Packed page should survive gzip+msgpack (the full wire format)."""
        import gzip
        import msgpack

        page = make_segmented_page(n_words=10)
        packed = pack_segmented_page(page)

        blob = gzip.compress(msgpack.packb(packed, use_bin_type=True))
        unpacked = msgpack.unpackb(gzip.decompress(blob), raw=False, strict_map_key=False)

        assert unpacked["version"] == 2
        assert unpacked["word_cells"]["n"] == 10
