"""Columnar storage for text cells with lazy materialization."""

from __future__ import annotations
import io
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Iterator, Tuple, Any
from docling_core.types.doc.base import BoundingBox, CoordOrigin


@dataclass(slots=True)
class _StringTable:
    """Deduplicated string storage with fast lookup."""
    items: List[str]
    
    def get(self, i: int) -> str:
        return self.items[i]
    
    def intern_many(self, seq: List[str]) -> np.ndarray:
        """Intern a sequence of strings, returning indices."""
        idx = np.empty(len(seq), np.int32)
        cache = {s: i for i, s in enumerate(self.items)}
        k = len(self.items)
        
        for j, s in enumerate(seq):
            i = cache.get(s)
            if i is None:
                i = k
                k += 1
                cache[s] = i
                self.items.append(s)
            idx[j] = i
        return idx


@dataclass(slots=True)
class _CellColumns:
    """Columnar storage for cell data."""
    # Geometry (4 corners)
    r_x0: np.ndarray  # float32
    r_y0: np.ndarray
    r_x1: np.ndarray
    r_y1: np.ndarray
    r_x2: np.ndarray
    r_y2: np.ndarray
    r_x3: np.ndarray
    r_y3: np.ndarray
    
    # Basic properties
    index: np.ndarray          # int32
    rgba: np.ndarray           # (N,4) uint8
    conf: Optional[np.ndarray] # float32
    from_ocr: Optional[np.ndarray] # bool
    text_idx: np.ndarray       # int32
    orig_idx: Optional[np.ndarray] # int32
    
    # PDF-specific columns (None for TextCell)
    rendering_mode: Optional[np.ndarray] # int8
    widget: Optional[np.ndarray]         # bool
    font_key_idx: Optional[np.ndarray]   # int32
    font_name_idx: Optional[np.ndarray]  # int32
    text_dir: Optional[np.ndarray]       # uint8


class CellStore:
    """Compact columnar storage for text cells with fast spatial queries."""
    
    __slots__ = ("cols", "strs", "page_height", "bottomleft_origin",
                 "_xmin", "_xmax", "_ymin", "_ymax")
    
    def __init__(self, cols: _CellColumns, strs: _StringTable,
                 page_height: float, *, bottomleft_origin: bool = True):
        self.cols = cols
        self.strs = strs
        self.page_height = page_height
        self.bottomleft_origin = bottomleft_origin
        
        # Cache AABBs for fast spatial queries
        self._xmin = np.minimum.reduce([cols.r_x0, cols.r_x1, cols.r_x2, cols.r_x3])
        self._xmax = np.maximum.reduce([cols.r_x0, cols.r_x1, cols.r_x2, cols.r_x3])
        self._ymin = np.minimum.reduce([cols.r_y0, cols.r_y1, cols.r_y2, cols.r_y3])
        self._ymax = np.maximum.reduce([cols.r_y0, cols.r_y1, cols.r_y2, cols.r_y3])
    
    def __len__(self) -> int:
        return self.cols.index.shape[0]
    
    def find_indices_in_bbox_ios(self, bbox: BoundingBox, ios: float = 0.8) -> np.ndarray:
        """Find cell indices that intersect bbox with given IoS threshold."""
        # Handle coordinate origin mismatch
        flip_needed = (bbox.coord_origin.name == "TOPLEFT") != self.bottomleft_origin
        if flip_needed:
            ph = self.page_height
            ymins = ph - self._ymax
            ymaxs = ph - self._ymin
        else:
            ymins = self._ymin
            ymaxs = self._ymax
        
        xmins = self._xmin
        xmaxs = self._xmax
        
        # Compute intersection over self (division-free)
        L, R, B, T = bbox.l, bbox.r, bbox.b, bbox.t
        iw = np.maximum(0.0, np.minimum(xmaxs, R) - np.maximum(xmins, L))
        ih = np.maximum(0.0, np.minimum(ymaxs, T) - np.maximum(ymins, B))
        inter = iw * ih
        area = (xmaxs - xmins) * (ymaxs - ymins)
        
        # Use > (not >=) to match original SegmentedPdfPage.get_cells_in_bbox behavior
        return np.flatnonzero(inter > ios * np.maximum(area, 1e-9))
    
    def get_text(self, i: int) -> str:
        """Get text for cell at index i."""
        return self.strs.get(int(self.cols.text_idx[i]))
    
    def get_orig(self, i: int) -> str:
        """Get original text for cell at index i."""
        if self.cols.orig_idx is not None:
            return self.strs.get(int(self.cols.orig_idx[i]))
        return self.get_text(i)
    
    def materialize(self, i: int, classes):
        """Materialize a single cell without building a LazyCellList."""
        TextCell, PdfTextCell, BoundingRectangle, CoordOrigin, TextDirection, ColorRGBA, PdfCellRenderingMode = classes
        c = self.cols
        
        # Build rectangle
        rect = BoundingRectangle.model_construct(
            r_x0=float(c.r_x0[i]), r_y0=float(c.r_y0[i]),
            r_x1=float(c.r_x1[i]), r_y1=float(c.r_y1[i]),
            r_x2=float(c.r_x2[i]), r_y2=float(c.r_y2[i]),
            r_x3=float(c.r_x3[i]), r_y3=float(c.r_y3[i]),
            coord_origin=CoordOrigin.BOTTOMLEFT if self.bottomleft_origin else CoordOrigin.TOPLEFT,
        )
        
        # Common properties
        text = self.get_text(i)
        orig = self.get_orig(i)
        index = int(c.index[i])
        r, g, b, a = map(int, c.rgba[i])
        conf = float(c.conf[i]) if c.conf is not None else 1.0
        from_ocr = bool(c.from_ocr[i]) if c.from_ocr is not None else False
        
        # Build proper ColorRGBA object
        rgba_obj = ColorRGBA.model_construct(r=r, g=g, b=b, a=a)
        
        # Detect if this is a PDF cell
        if c.font_name_idx is not None:
            font_key = self.strs.get(int(c.font_key_idx[i])) if c.font_key_idx is not None else ""
            font_name = self.strs.get(int(c.font_name_idx[i])) if c.font_name_idx is not None else ""
            widget = bool(c.widget[i]) if c.widget is not None else False
            
            # Proper enum construction with safety fallback
            rm_int = int(c.rendering_mode[i]) if c.rendering_mode is not None else 0
            try:
                rm = PdfCellRenderingMode(rm_int)
            except ValueError:
                rm = PdfCellRenderingMode.UNKNOWN
            
            dir_code = int(c.text_dir[i]) if c.text_dir is not None else 0
            text_dir = TextDirection.LEFT_TO_RIGHT if dir_code == 0 else TextDirection.RIGHT_TO_LEFT
            
            return PdfTextCell.model_construct(
                rect=rect, text=text, orig=orig, index=index,
                rgba=rgba_obj, confidence=conf, from_ocr=False,  # Force invariant
                font_key=font_key, font_name=font_name,
                widget=widget, rendering_mode=rm,
                text_direction=text_dir,
            )
        else:
            return TextCell.model_construct(
                rect=rect, text=text, orig=orig, index=index,
                rgba=rgba_obj, confidence=conf, from_ocr=from_ocr,
            )
    
    def get_cells_in_bbox(self, bbox: 'BoundingBox', ios: float, classes) -> list:
        """Get cells that intersect bbox with IoS > threshold.

        This is a vectorized replacement for SegmentedPdfPage.get_cells_in_bbox().
        Returns cells with rect.coord_origin converted to match bbox.coord_origin.

        Args:
            bbox: Bounding box to check against
            ios: Intersection over self threshold (cells with IoS > ios are returned)
            classes: Tuple of (TextCell, PdfTextCell, BoundingRectangle, CoordOrigin,
                     TextDirection, ColorRGBA, PdfCellRenderingMode)

        Returns:
            List of cells within the bounding box, with coordinates converted
        """
        # Fast vectorized spatial query
        indices = self.find_indices_in_bbox_ios(bbox, ios)

        if len(indices) == 0:
            return []

        TextCell, PdfTextCell, BoundingRectangle, CoordOrigin, TextDirection, ColorRGBA, PdfCellRenderingMode = classes

        # Determine if we need coordinate conversion
        store_origin = CoordOrigin.BOTTOMLEFT if self.bottomleft_origin else CoordOrigin.TOPLEFT
        need_conversion = store_origin != bbox.coord_origin

        cells = []
        for i in indices:
            cell = self.materialize(int(i), classes)

            # Convert coordinate origin if needed (matches original behavior)
            if need_conversion:
                if bbox.coord_origin == CoordOrigin.TOPLEFT:
                    cell.rect = cell.rect.to_top_left_origin(self.page_height)
                elif bbox.coord_origin == CoordOrigin.BOTTOMLEFT:
                    cell.rect = cell.rect.to_bottom_left_origin(self.page_height)

            cells.append(cell)

        return cells

    def __getstate__(self) -> dict:
        """Compact pickle serialization."""
        buf = io.BytesIO()
        arrays_to_save = {
            'r_x0': self.cols.r_x0, 'r_y0': self.cols.r_y0,
            'r_x1': self.cols.r_x1, 'r_y1': self.cols.r_y1,
            'r_x2': self.cols.r_x2, 'r_y2': self.cols.r_y2,
            'r_x3': self.cols.r_x3, 'r_y3': self.cols.r_y3,
            'index': self.cols.index, 'rgba': self.cols.rgba,
            'text_idx': self.cols.text_idx,
            'page_height': np.array([self.page_height], np.float32),
            'origin': np.array([1 if self.bottomleft_origin else 0], np.uint8),
            'xmin': self._xmin, 'xmax': self._xmax,
            'ymin': self._ymin, 'ymax': self._ymax,
        }
        
        # Add optional arrays if present
        if self.cols.conf is not None:
            arrays_to_save['conf'] = self.cols.conf
        if self.cols.from_ocr is not None:
            arrays_to_save['from_ocr'] = self.cols.from_ocr
        if self.cols.orig_idx is not None:
            arrays_to_save['orig_idx'] = self.cols.orig_idx
        if self.cols.rendering_mode is not None:
            arrays_to_save['rendering_mode'] = self.cols.rendering_mode
        if self.cols.widget is not None:
            arrays_to_save['widget'] = self.cols.widget
        if self.cols.font_key_idx is not None:
            arrays_to_save['font_key_idx'] = self.cols.font_key_idx
        if self.cols.font_name_idx is not None:
            arrays_to_save['font_name_idx'] = self.cols.font_name_idx
        if self.cols.text_dir is not None:
            arrays_to_save['text_dir'] = self.cols.text_dir
        
        np.savez_compressed(buf, **arrays_to_save)
        
        return {
            "npz": buf.getvalue(),
            "strings": self.strs.items
        }
    
    def __setstate__(self, state: dict) -> None:
        """Restore from pickle."""
        f = io.BytesIO(state["npz"])
        d = np.load(f, allow_pickle=False)
        
        self.cols = _CellColumns(
            r_x0=d["r_x0"], r_y0=d["r_y0"], r_x1=d["r_x1"], r_y1=d["r_y1"],
            r_x2=d["r_x2"], r_y2=d["r_y2"], r_x3=d["r_x3"], r_y3=d["r_y3"],
            index=d["index"], rgba=d["rgba"],
            conf=d.get("conf"), from_ocr=d.get("from_ocr"),
            text_idx=d["text_idx"], orig_idx=d.get("orig_idx"),
            rendering_mode=d.get("rendering_mode"), widget=d.get("widget"),
            font_key_idx=d.get("font_key_idx"), font_name_idx=d.get("font_name_idx"),
            text_dir=d.get("text_dir")
        )
        
        self.strs = _StringTable(items=list(state["strings"]))
        self.page_height = float(d["page_height"][0])
        self.bottomleft_origin = bool(int(d["origin"][0]))
        self._xmin, self._xmax = d["xmin"], d["xmax"]
        self._ymin, self._ymax = d["ymin"], d["ymax"]


class LazyCellList:
    """A list-like view over CellStore that yields TextCell/PdfTextCell on demand."""
    
    __slots__ = ("_store", "_classes")
    
    def __init__(self, store: CellStore, classes: Tuple[Any, ...]):
        self._store = store
        self._classes = classes  # (TextCell, PdfTextCell, BoundingRectangle, CoordOrigin, TextDirection)
    
    def __len__(self) -> int:
        return len(self._store)
    
    def __iter__(self) -> Iterator:
        for i in range(len(self._store)):
            yield self._materialize(i)
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = range(*key.indices(len(self)))
            return [self._materialize(i) for i in indices]
        elif isinstance(key, int):
            if key < 0:
                key += len(self._store)
            if key < 0 or key >= len(self._store):
                raise IndexError
            return self._materialize(key)
        else:
            raise TypeError("Invalid key type")
    
    def _materialize(self, i: int):
        """Materialize a single cell without validation overhead."""
        return self._store.materialize(i, self._classes)


def _pack_list_to_store(cells: List, page_height: float) -> CellStore:
    """Convert a list of TextCell/PdfTextCell to compact CellStore."""
    N = len(cells)
    if N == 0:
        # Empty store - use None for truly optional arrays
        empty_cols = _CellColumns(
            r_x0=np.array([], np.float32), r_y0=np.array([], np.float32),
            r_x1=np.array([], np.float32), r_y1=np.array([], np.float32),
            r_x2=np.array([], np.float32), r_y2=np.array([], np.float32),
            r_x3=np.array([], np.float32), r_y3=np.array([], np.float32),
            index=np.array([], np.int32), rgba=np.array([], np.uint8).reshape(0, 4),
            conf=None, from_ocr=None, text_idx=np.array([], np.int32), 
            orig_idx=None, rendering_mode=None, widget=None, 
            font_key_idx=None, font_name_idx=None, text_dir=None
        )
        return CellStore(empty_cols, _StringTable([]), page_height)
    
    # Preallocate arrays
    r_x0 = np.empty(N, np.float32)
    r_y0 = np.empty(N, np.float32)
    r_x1 = np.empty(N, np.float32)
    r_y1 = np.empty(N, np.float32)
    r_x2 = np.empty(N, np.float32)
    r_y2 = np.empty(N, np.float32)
    r_x3 = np.empty(N, np.float32)
    r_y3 = np.empty(N, np.float32)
    
    index = np.empty(N, np.int32)
    rgba = np.empty((N, 4), np.uint8)
    conf = np.empty(N, np.float32)
    from_ocr = np.empty(N, np.bool_)
    text_idx = np.empty(N, np.int32)
    orig_idx = np.empty(N, np.int32)
    
    # Detect if any cells are PDF cells
    is_pdf = any(hasattr(c, "font_name") for c in cells)
    if is_pdf:
        rendering_mode = np.empty(N, np.int8)
        widget = np.empty(N, np.bool_)
        font_key_idx = np.empty(N, np.int32)
        font_name_idx = np.empty(N, np.int32)
        text_dir = np.empty(N, np.uint8)
    else:
        rendering_mode = widget = font_key_idx = font_name_idx = text_dir = None
    
    # String table for deduplication
    st = _StringTable([])
    cache = {}
    
    def intern(s: str) -> int:
        i = cache.get(s)
        if i is None:
            i = len(st.items)
            cache[s] = i
            st.items.append(s)
        return i
    
    # Fill arrays
    for i, c in enumerate(cells):
        r_x0[i], r_y0[i], r_x1[i], r_y1[i] = c.rect.r_x0, c.rect.r_y0, c.rect.r_x1, c.rect.r_y1
        r_x2[i], r_y2[i], r_x3[i], r_y3[i] = c.rect.r_x2, c.rect.r_y2, c.rect.r_x3, c.rect.r_y3
        
        index[i] = c.index
        rgba[i] = [c.rgba.r, c.rgba.g, c.rgba.b, c.rgba.a]
        conf[i] = getattr(c, "confidence", 1.0)
        from_ocr[i] = getattr(c, "from_ocr", False)
        text_idx[i] = intern(c.text)
        orig_idx[i] = intern(getattr(c, "orig", c.text))
        
        # Handle PDF columns per row to support mixed cell types
        is_pdf_row = hasattr(c, "font_name")
        if is_pdf_row:
            rendering_mode[i] = getattr(c, "rendering_mode", 0)
            widget[i] = getattr(c, "widget", False)
            font_key_idx[i] = intern(getattr(c, "font_key", ""))
            font_name_idx[i] = intern(getattr(c, "font_name", ""))
            td = getattr(c, "text_direction", None)
            text_dir[i] = 0 if (td is None or str(td) == "left_to_right") else 1
        elif is_pdf:
            # Non-PDF cell in PDF array - fill with defaults to keep arrays consistent
            rendering_mode[i] = 0
            widget[i] = False
            font_key_idx[i] = intern("")
            font_name_idx[i] = intern("")
            text_dir[i] = 0
    
    # Ensure contiguous arrays with proper dtypes
    def ensure_contiguous(arr, dtype):
        return np.ascontiguousarray(arr, dtype=dtype)
    
    cols = _CellColumns(
        r_x0=ensure_contiguous(r_x0, np.float32), r_y0=ensure_contiguous(r_y0, np.float32),
        r_x1=ensure_contiguous(r_x1, np.float32), r_y1=ensure_contiguous(r_y1, np.float32),
        r_x2=ensure_contiguous(r_x2, np.float32), r_y2=ensure_contiguous(r_y2, np.float32),
        r_x3=ensure_contiguous(r_x3, np.float32), r_y3=ensure_contiguous(r_y3, np.float32),
        index=ensure_contiguous(index, np.int32), 
        rgba=ensure_contiguous(rgba, np.uint8),
        conf=ensure_contiguous(conf, np.float32) if conf is not None else None,
        from_ocr=ensure_contiguous(from_ocr, np.bool_) if from_ocr is not None else None,
        text_idx=ensure_contiguous(text_idx, np.int32),
        orig_idx=ensure_contiguous(orig_idx, np.int32) if orig_idx is not None else None,
        rendering_mode=ensure_contiguous(rendering_mode, np.int8) if rendering_mode is not None else None,
        widget=ensure_contiguous(widget, np.bool_) if widget is not None else None,
        font_key_idx=ensure_contiguous(font_key_idx, np.int32) if font_key_idx is not None else None,
        font_name_idx=ensure_contiguous(font_name_idx, np.int32) if font_name_idx is not None else None,
        text_dir=ensure_contiguous(text_dir, np.uint8) if text_dir is not None else None
    )
    
    return CellStore(cols, st, page_height, bottomleft_origin=True)
