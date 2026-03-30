#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import cv2
import numpy as np


class TablePreprocessor:
    r"""
    Table predictions for the in-memory Docling API
    """

    @staticmethod
    def prepare_table_inputs(
            page_inputs: list[dict],
            table_bboxes_list: list[list[list[float]]],
    ):
        """
        Batched API with ORIGINAL semantics:
          1) Resize each *page image* to height=1024 (record scale_factor).
          2) Scale each table bbox by the same page scale_factor.
          3) Crop table from the *resized* page.
          4) Run a single batched predict over all tables.
          5) Apply the original row/col reindexing or rs_seq counting.
        Returns a flat list of dicts [{ "tf_responses": ..., "predict_details": ... }, ...]
        in page order, table order (stable).
        """

        all_table_images: list[np.ndarray] = []
        all_scaled_bboxes: list[list[float]] = []
        all_scale_factors: list[float] = []
        all_iocr_pages: list[dict] = []

        for page_input, page_tbl_bboxes in zip(page_inputs, table_bboxes_list):
            img = page_input["image"]
            H, W = img.shape[:2]
            scale_factor = 1024.0 / float(H)

            # Per-table tokens (if available); fall back to page-level tokens
            per_table_tokens = page_input.get("per_table_tokens", None)

            # Local aliases to reduce attribute lookups in the inner loop
            _resize = cv2.resize
            _append_img = all_table_images.append
            _append_bbox = all_scaled_bboxes.append
            _append_scale = all_scale_factors.append
            _append_page = all_iocr_pages.append

            for tbl_idx, (x1, y1, x2, y2) in enumerate(page_tbl_bboxes):
                # Integer crop coordinates in original page space
                ix1 = int(round(x1));
                iy1 = int(round(y1))
                ix2 = int(round(x2));
                iy2 = int(round(y2))

                # Clamp to image bounds (robustness)
                if ix1 < 0: ix1 = 0
                if iy1 < 0: iy1 = 0
                if ix2 > W: ix2 = W
                if iy2 > H: iy2 = H

                if ix2 <= ix1 or iy2 <= iy1:
                    continue  # skip degenerate boxes

                crop = img[iy1:iy2, ix1:ix2]
                if crop.size == 0:
                    continue

                # Target size in the "resized page" coordinates
                tw = max(1, int(round((x2 - x1) * scale_factor)))
                th = max(1, int(round((y2 - y1) * scale_factor)))

                resized_crop = _resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

                _append_img(resized_crop)
                _append_bbox([x1 * scale_factor, y1 * scale_factor, x2 * scale_factor, y2 * scale_factor])
                _append_scale(scale_factor)

                # Build per-table iocr_page with only this table's tokens
                if per_table_tokens is not None:
                    table_iocr = {
                        "width": page_input["width"],
                        "height": page_input["height"],
                        "tokens": per_table_tokens[tbl_idx],
                    }
                    _append_page(table_iocr)
                else:
                    _append_page(page_input)

        return {
            'iocr_pages': all_iocr_pages,
            'table_bboxes': all_scaled_bboxes,
            'table_images': all_table_images,
            'scale_factors': all_scale_factors,
        }
