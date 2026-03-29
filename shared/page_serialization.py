"""
Columnar serialization for SegmentedPdfPage.

Used by both the Lambda preprocessing pipeline and the Hetzner preprocessor
to pack parsed PDF pages into a compact binary format for the GPU processor.
"""

import array
from typing import Any, Dict, List


def pack_segmented_page(seg: Any) -> Dict[str, Any]:
    is_topleft = True
    return {
        "version": 2,
        "dimension": _pack_dimension(seg.dimension),
        "page_height": seg.dimension.height,
        "char_cells": _pack_cells_columnar(seg.char_cells, is_topleft),
        "word_cells": _pack_cells_columnar(seg.word_cells, is_topleft),
        "textline_cells": _pack_cells_columnar(seg.textline_cells, is_topleft),
        "has_chars": seg.has_chars,
        "has_words": seg.has_words,
        "has_lines": seg.has_lines,
    }


def _pack_dimension(dim) -> Dict[str, Any]:
    return {
        "angle": dim.angle,
        "height": dim.height,
        "width": dim.width,
        "boundary_type": str(dim.boundary_type),
        "crop_bbox": [dim.crop_bbox.l, dim.crop_bbox.t, dim.crop_bbox.r, dim.crop_bbox.b],
        "media_bbox": [dim.media_bbox.l, dim.media_bbox.t, dim.media_bbox.r, dim.media_bbox.b],
        "art_bbox": [dim.art_bbox.l, dim.art_bbox.t, dim.art_bbox.r, dim.art_bbox.b],
        "bleed_bbox": [dim.bleed_bbox.l, dim.bleed_bbox.t, dim.bleed_bbox.r, dim.bleed_bbox.b],
        "trim_bbox": [dim.trim_bbox.l, dim.trim_bbox.t, dim.trim_bbox.r, dim.trim_bbox.b],
    }


def _pack_cells_columnar(cells: List, is_topleft: bool) -> Dict[str, Any]:
    """Pack cells into columnar arrays with string interning for compact serialization."""
    N = len(cells)
    if N == 0:
        return {"n": 0}

    r_x0 = array.array('f')
    r_y0 = array.array('f')
    r_x1 = array.array('f')
    r_y1 = array.array('f')
    r_x2 = array.array('f')
    r_y2 = array.array('f')
    r_x3 = array.array('f')
    r_y3 = array.array('f')

    index = array.array('i')
    rgba = array.array('B')
    conf = array.array('f')
    text_idx = array.array('i')
    orig_idx = array.array('i')

    is_pdf = hasattr(cells[0], 'font_name') if N > 0 else False
    if is_pdf:
        rendering_mode = array.array('b')
        widget = array.array('B')
        font_key_idx = array.array('i')
        font_name_idx = array.array('i')
        text_dir = array.array('B')

    strings: List[str] = []
    str_cache: Dict[str, int] = {}

    def intern(s: str) -> int:
        i = str_cache.get(s)
        if i is None:
            i = len(strings)
            str_cache[s] = i
            strings.append(s)
        return i

    for c in cells:
        r = c.rect
        r_x0.append(r.r_x0)
        r_y0.append(r.r_y0)
        r_x1.append(r.r_x1)
        r_y1.append(r.r_y1)
        r_x2.append(r.r_x2)
        r_y2.append(r.r_y2)
        r_x3.append(r.r_x3)
        r_y3.append(r.r_y3)

        index.append(c.index)
        rgba.extend([c.rgba.r, c.rgba.g, c.rgba.b, c.rgba.a])
        conf.append(getattr(c, 'confidence', 1.0))
        text_idx.append(intern(c.text))
        orig_idx.append(intern(getattr(c, 'orig', c.text)))

        if is_pdf:
            rendering_mode.append(int(getattr(c, 'rendering_mode', 0)))
            widget.append(1 if getattr(c, 'widget', False) else 0)
            font_key_idx.append(intern(getattr(c, 'font_key', '')))
            font_name_idx.append(intern(getattr(c, 'font_name', '')))
            td = getattr(c, 'text_direction', 'left_to_right')
            text_dir.append(0 if str(td) == 'left_to_right' else 1)

    result = {
        "n": N,
        "topleft": is_topleft,
        "r_x0": r_x0.tobytes(),
        "r_y0": r_y0.tobytes(),
        "r_x1": r_x1.tobytes(),
        "r_y1": r_y1.tobytes(),
        "r_x2": r_x2.tobytes(),
        "r_y2": r_y2.tobytes(),
        "r_x3": r_x3.tobytes(),
        "r_y3": r_y3.tobytes(),
        "index": index.tobytes(),
        "rgba": rgba.tobytes(),
        "conf": conf.tobytes(),
        "text_idx": text_idx.tobytes(),
        "orig_idx": orig_idx.tobytes(),
        "strings": strings,
    }

    if is_pdf:
        result["rendering_mode"] = rendering_mode.tobytes()
        result["widget"] = widget.tobytes()
        result["font_key_idx"] = font_key_idx.tobytes()
        result["font_name_idx"] = font_name_idx.tobytes()
        result["text_dir"] = text_dir.tobytes()

    return result
