import io
from typing import Dict

import numpy as np
from PIL import Image

from docling.datamodel.base_models import Page
from docling_core.types.doc.page import (
    SegmentedPdfPage, PdfPageGeometry, TextCell, PdfTextCell,
    BoundingRectangle, ColorRGBA, PdfCellRenderingMode, PdfPageBoundaryType,
)
from docling_core.types.doc import Size, BoundingBox, CoordOrigin, TextDirection

from processor.shared.cell_store import CellStore, _CellColumns, _StringTable, LazyCellList
from processor.shared.logging_config import get_logger

logger = get_logger(__name__)


def unpack_cells_columnar(cell_dict: Dict) -> CellStore:
    n = cell_dict.get('n', 0)
    if n == 0:
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
        return CellStore(empty_cols, _StringTable([]), 0.0)

    r_x0 = np.frombuffer(cell_dict['r_x0'], dtype=np.float32)
    r_y0 = np.frombuffer(cell_dict['r_y0'], dtype=np.float32)
    r_x1 = np.frombuffer(cell_dict['r_x1'], dtype=np.float32)
    r_y1 = np.frombuffer(cell_dict['r_y1'], dtype=np.float32)
    r_x2 = np.frombuffer(cell_dict['r_x2'], dtype=np.float32)
    r_y2 = np.frombuffer(cell_dict['r_y2'], dtype=np.float32)
    r_x3 = np.frombuffer(cell_dict['r_x3'], dtype=np.float32)
    r_y3 = np.frombuffer(cell_dict['r_y3'], dtype=np.float32)

    index = np.frombuffer(cell_dict['index'], dtype=np.int32)
    rgba_flat = np.frombuffer(cell_dict['rgba'], dtype=np.uint8)
    rgba = rgba_flat.reshape(-1, 4)
    conf = np.frombuffer(cell_dict['conf'], dtype=np.float32)
    text_idx = np.frombuffer(cell_dict['text_idx'], dtype=np.int32)
    orig_idx = np.frombuffer(cell_dict['orig_idx'], dtype=np.int32)

    rendering_mode = np.frombuffer(cell_dict['rendering_mode'], dtype=np.int8) if 'rendering_mode' in cell_dict else None
    widget = np.frombuffer(cell_dict['widget'], dtype=np.uint8).astype(np.bool_) if 'widget' in cell_dict else None
    font_key_idx = np.frombuffer(cell_dict['font_key_idx'], dtype=np.int32) if 'font_key_idx' in cell_dict else None
    font_name_idx = np.frombuffer(cell_dict['font_name_idx'], dtype=np.int32) if 'font_name_idx' in cell_dict else None
    text_dir = np.frombuffer(cell_dict['text_dir'], dtype=np.uint8) if 'text_dir' in cell_dict else None

    strings = cell_dict.get('strings', [])
    is_topleft = cell_dict.get('topleft', True)

    cols = _CellColumns(
        r_x0=r_x0, r_y0=r_y0, r_x1=r_x1, r_y1=r_y1,
        r_x2=r_x2, r_y2=r_y2, r_x3=r_x3, r_y3=r_y3,
        index=index, rgba=rgba, conf=conf, from_ocr=None,
        text_idx=text_idx, orig_idx=orig_idx,
        rendering_mode=rendering_mode, widget=widget,
        font_key_idx=font_key_idx, font_name_idx=font_name_idx,
        text_dir=text_dir
    )

    return CellStore(cols, _StringTable(list(strings)), 0.0, bottomleft_origin=not is_topleft)


def reconstruct_page(page_dict: Dict) -> Page:
    size_dict = page_dict['size']
    size = Size(width=size_dict['w'], height=size_dict['h'])

    segmented_dict = page_dict['segmented']

    if segmented_dict.get('version') == 2:
        dim_dict = segmented_dict['dimension']
        page_height = segmented_dict['page_height']

        # The packed page geometry preserves the original parser semantics:
        # page boxes are stored in bottom-left coordinates, even though text cells
        # are flipped to top-left before serialization.
        crop_bbox = BoundingBox(
            l=dim_dict['crop_bbox'][0], t=dim_dict['crop_bbox'][1],
            r=dim_dict['crop_bbox'][2], b=dim_dict['crop_bbox'][3],
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
        rect = BoundingRectangle(
            r_x0=crop_bbox.l, r_y0=crop_bbox.b,
            r_x1=crop_bbox.r, r_y1=crop_bbox.b,
            r_x2=crop_bbox.r, r_y2=crop_bbox.t,
            r_x3=crop_bbox.l, r_y3=crop_bbox.t,
            coord_origin=CoordOrigin.BOTTOMLEFT,
        )
        dimension = PdfPageGeometry.model_construct(
            angle=dim_dict['angle'],
            rect=rect,
            boundary_type=PdfPageBoundaryType(dim_dict['boundary_type']),
            crop_bbox=crop_bbox,
            media_bbox=BoundingBox(
                l=dim_dict['media_bbox'][0], t=dim_dict['media_bbox'][1],
                r=dim_dict['media_bbox'][2], b=dim_dict['media_bbox'][3],
                coord_origin=CoordOrigin.BOTTOMLEFT,
            ),
            art_bbox=BoundingBox(
                l=dim_dict['art_bbox'][0], t=dim_dict['art_bbox'][1],
                r=dim_dict['art_bbox'][2], b=dim_dict['art_bbox'][3],
                coord_origin=CoordOrigin.BOTTOMLEFT,
            ),
            bleed_bbox=BoundingBox(
                l=dim_dict['bleed_bbox'][0], t=dim_dict['bleed_bbox'][1],
                r=dim_dict['bleed_bbox'][2], b=dim_dict['bleed_bbox'][3],
                coord_origin=CoordOrigin.BOTTOMLEFT,
            ),
            trim_bbox=BoundingBox(
                l=dim_dict['trim_bbox'][0], t=dim_dict['trim_bbox'][1],
                r=dim_dict['trim_bbox'][2], b=dim_dict['trim_bbox'][3],
                coord_origin=CoordOrigin.BOTTOMLEFT,
            ),
        )

        classes = (TextCell, PdfTextCell, BoundingRectangle, CoordOrigin, TextDirection,
                   ColorRGBA, PdfCellRenderingMode)

        char_store = unpack_cells_columnar(segmented_dict['char_cells'])
        char_store.page_height = page_height
        word_store = unpack_cells_columnar(segmented_dict['word_cells'])
        word_store.page_height = page_height
        textline_store = unpack_cells_columnar(segmented_dict['textline_cells'])
        textline_store.page_height = page_height

        char_cells = LazyCellList(char_store, classes)
        word_cells = LazyCellList(word_store, classes)
        textline_cells = LazyCellList(textline_store, classes)

        parsed_page = SegmentedPdfPage.model_construct(
            dimension=dimension,
            char_cells=char_cells,
            word_cells=word_cells,
            textline_cells=textline_cells,
            has_chars=segmented_dict.get('has_chars', len(char_cells) > 0),
            has_words=segmented_dict.get('has_words', len(word_cells) > 0),
            has_lines=segmented_dict.get('has_lines', len(textline_cells) > 0),
        )

        parsed_page._char_store = char_store
        parsed_page._word_store = word_store
        parsed_page._line_store = textline_store

        logger.info(
            f"page {page_dict['page_index']}: columnar v2 - "
            f"chars={len(char_cells)} words={len(word_cells)} lines={len(textline_cells)}"
        )
    else:
        parsed_page = SegmentedPdfPage.model_validate(segmented_dict)

    image_cache = {}
    for scale_str, img_bytes in page_dict['images'].items():
        scale = float(scale_str)
        pil_image = Image.open(io.BytesIO(img_bytes))
        image_cache[scale] = pil_image

    page = Page(
        page_no=page_dict['page_index'],
        size=size,
        parsed_page=parsed_page,
    )
    page._image_cache = image_cache
    page._images_scale = page_dict.get('images_scale', 2.0)

    return page
