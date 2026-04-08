"""
Pure CPU preprocessing: parse PDF pages, render images, pack to columnar format.

This module extracts the compute-only logic from the Lambda preprocessing
pipeline so it can be used locally without any AWS dependencies. The output
format is byte-for-byte identical to what the Lambda produces.
"""

import io
import time
from typing import Any, Dict, List, Optional

import pypdfium2 as pdfium
from PIL import Image
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_parse.pdf_parser import DoclingPdfParser

from shared.page_serialization import pack_segmented_page

MAX_2X_PIXELS = 3200

# Per-process globals for multiprocessing workers
_PARSER: Optional[DoclingPdfParser] = None


def worker_init():
    """Initialize the C++ parser once per worker process."""
    global _PARSER
    _PARSER = DoclingPdfParser(loglevel="fatal")


def preprocess_pages(pdf_bytes: bytes, start_page: int, end_page: int) -> List[Dict[str, Any]]:
    """
    Preprocess a range of pages from a PDF.

    Produces the same page dicts as PreprocessingPipeline._preprocess_page_batch():
    each dict has keys: page_index, size, segmented, images, images_scale.

    Args:
        pdf_bytes: Raw PDF file bytes.
        start_page: First page (0-indexed, inclusive).
        end_page: Last page (0-indexed, inclusive).

    Returns:
        List of page dicts ready for reconstruct_page().
    """
    global _PARSER
    if _PARSER is None:
        _PARSER = DoclingPdfParser(loglevel="fatal")

    pdoc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    dp_doc = _PARSER.load(path_or_stream=io.BytesIO(pdf_bytes))

    try:
        pages = []
        for page_no in range(start_page, end_page + 1):
            page_dict = _preprocess_one_page(pdoc, dp_doc, page_no)
            pages.append(page_dict)
        return pages
    finally:
        dp_doc.unload()
        close = getattr(pdoc, "close", None)
        if callable(close):
            close()


def _preprocess_one_page(pdoc, dp_doc, page_no: int) -> Dict[str, Any]:
    """Preprocess a single page. Matches Lambda's _preprocess_page_batch logic exactly."""
    # Parse
    seg = dp_doc.get_page(page_no + 1, create_words=True, create_textlines=True)
    H = seg.dimension.height
    for cells in (seg.textline_cells, seg.char_cells, seg.word_cells):
        for tc in cells:
            tc.to_top_left_origin(H)

    # Render
    pdf_page = pdoc[page_no]
    size = Size(width=pdf_page.get_width(), height=pdf_page.get_height())

    long_side = max(size.width, size.height)
    scale_2x = min(2.0, MAX_2X_PIXELS / long_side)

    img1 = _get_page_image(pdf_page, size, scale=1.0)
    img2 = _get_page_image(pdf_page, size, scale=scale_2x)

    # Encode
    img1_webp = _convert_to_webp(img1)
    img2_webp = _convert_to_webp(img2)

    # Pack
    seg_packed = pack_segmented_page(seg)

    return {
        "page_index": page_no,
        "size": {"w": size.width, "h": size.height},
        "segmented": seg_packed,
        "images": {1: img1_webp.getvalue(), 2: img2_webp.getvalue()},
        "images_scale": scale_2x,
    }


def _get_page_image(page, page_size: Size, scale: float = 1.0) -> Image.Image:
    """Render a page image with 1.5x supersample, matching stock Docling's backend."""
    cropbox = BoundingBox(l=0, r=page_size.width, t=0, b=page_size.height, coord_origin=CoordOrigin.TOPLEFT)
    padbox = BoundingBox(l=0, r=0, t=0, b=0, coord_origin=CoordOrigin.BOTTOMLEFT)

    # CRITICAL: Render at 1.5x the target scale, then downsample to match stock
    # Docling's backend. The layout model is sensitive to sub-pixel rendering
    # differences — see preprocessing_pipeline.py for the full explanation.
    img = page.render(scale=scale * 1.5, rotation=0, crop=padbox.as_tuple()).to_pil()
    return img.resize(size=(round(cropbox.width * scale), round(cropbox.height * scale)))


def _convert_to_webp(img: Image.Image) -> io.BytesIO:
    """Convert to lossless webp with fast compression."""
    mode = "RGB" if img.mode not in ("L", "RGB", "RGBA") else img.mode
    if img.mode != mode:
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", lossless=True, method=1)
    buf.seek(0)
    return buf
