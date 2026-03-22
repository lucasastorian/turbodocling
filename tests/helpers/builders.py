"""Test builders for creating minimal docling objects without full pydantic overhead."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FakeRect:
    r_x0: float = 0.0
    r_y0: float = 0.0
    r_x1: float = 100.0
    r_y1: float = 0.0
    r_x2: float = 100.0
    r_y2: float = 20.0
    r_x3: float = 0.0
    r_y3: float = 20.0


@dataclass
class FakeRGBA:
    r: int = 0
    g: int = 0
    b: int = 0
    a: int = 255


@dataclass
class FakeTextCell:
    """Minimal TextCell-like object for serialization tests."""
    index: int = 0
    text: str = "hello"
    orig: str = "hello"
    rect: FakeRect = field(default_factory=FakeRect)
    rgba: FakeRGBA = field(default_factory=FakeRGBA)
    confidence: float = 1.0


@dataclass
class FakePdfTextCell(FakeTextCell):
    """Minimal PdfTextCell-like object with PDF-specific fields."""
    font_name: str = "Arial"
    font_key: str = "F1"
    rendering_mode: int = 0
    widget: bool = False
    text_direction: str = "left_to_right"


@dataclass
class FakeBBox:
    l: float
    t: float
    r: float
    b: float


@dataclass
class FakeDimension:
    angle: float = 0.0
    height: float = 792.0
    width: float = 612.0
    boundary_type: str = "CropBox"
    crop_bbox: FakeBBox = field(default_factory=lambda: FakeBBox(0, 0, 612, 792))
    media_bbox: FakeBBox = field(default_factory=lambda: FakeBBox(0, 0, 612, 792))
    art_bbox: FakeBBox = field(default_factory=lambda: FakeBBox(0, 0, 612, 792))
    bleed_bbox: FakeBBox = field(default_factory=lambda: FakeBBox(0, 0, 612, 792))
    trim_bbox: FakeBBox = field(default_factory=lambda: FakeBBox(0, 0, 612, 792))


@dataclass
class FakeSegmentedPage:
    """Minimal SegmentedPdfPage-like object for serialization tests."""
    dimension: FakeDimension = field(default_factory=FakeDimension)
    char_cells: list = field(default_factory=list)
    word_cells: list = field(default_factory=list)
    textline_cells: list = field(default_factory=list)
    has_chars: bool = False
    has_words: bool = False
    has_lines: bool = False


def make_cells(n: int, pdf: bool = True) -> list:
    """Create n fake cells with distinct positions and text."""
    cls = FakePdfTextCell if pdf else FakeTextCell
    cells = []
    for i in range(n):
        cells.append(cls(
            index=i,
            text=f"word_{i}",
            orig=f"word_{i}",
            rect=FakeRect(
                r_x0=float(i * 50), r_y0=0.0,
                r_x1=float(i * 50 + 40), r_y1=0.0,
                r_x2=float(i * 50 + 40), r_y2=15.0,
                r_x3=float(i * 50), r_y3=15.0,
            ),
            rgba=FakeRGBA(0, 0, 0, 255),
            confidence=0.95,
        ))
    return cells


def make_segmented_page(n_words: int = 5, n_chars: int = 0, n_lines: int = 0, pdf: bool = True) -> FakeSegmentedPage:
    """Create a fake segmented page with specified cell counts."""
    return FakeSegmentedPage(
        dimension=FakeDimension(),
        char_cells=make_cells(n_chars, pdf=pdf),
        word_cells=make_cells(n_words, pdf=pdf),
        textline_cells=make_cells(n_lines, pdf=pdf),
        has_chars=n_chars > 0,
        has_words=n_words > 0,
        has_lines=n_lines > 0,
    )
