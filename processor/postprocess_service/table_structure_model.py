from typing import Any, Dict, List

from docling_core.types.doc import BoundingBox, CoordOrigin, DocItemLabel, TableCell, TextDirection
from docling_core.types.doc.page import (
    BoundingRectangle,
    ColorRGBA,
    PdfCellRenderingMode,
    PdfTextCell,
    TextCell,
    TextCellUnit,
)
from docling.datamodel.base_models import Table, TableStructurePrediction, Page

from .table_preprocessor import TablePreprocessor

_CELL_CLASSES = (
    TextCell,
    PdfTextCell,
    BoundingRectangle,
    CoordOrigin,
    TextDirection,
    ColorRGBA,
    PdfCellRenderingMode,
)


class TableStructureModel:

    do_cell_matching: bool = True

    def __init__(self):
        self.table_preprocessor = TablePreprocessor()

    def preprocess(self, pages: List[Page]) -> Dict:

        page_inputs: List[dict] = []
        table_bboxes_list: List[List[List[float]]] = []  # per-page list of bboxes
        page_clusters_list: List[List[Any]] = []  # per-page list of clusters
        batched_page_indexes: List[int] = []  # map batch idx -> pages_list idx

        # Prepare per-page table inputs
        for page_idx, page in enumerate(pages):

            assert page.predictions.layout is not None
            assert page.size is not None

            # Always initialize predictions (like original)
            page.predictions.tablestructure = TableStructurePrediction()

            scale = page._images_scale
            in_tables = self._get_tables_from_page(page, scale)
            if not in_tables:
                continue

            page_input = {
                "width": page.size.width * scale,
                "height": page.size.height * scale,
                "image": page.get_image_np(scale=2.0),  # cache key is always 2.0
            }

            page_table_bboxes: List[List[float]] = []
            page_clusters: List[Any] = []

            # Build per-table token lists (matching stock behavior).
            # Each table gets only the tokens inside its own bbox,
            # NOT the aggregated page-level token universe.
            per_table_tokens: List[List[dict]] = []
            for table_cluster, tbl_box in in_tables:
                if self.do_cell_matching:
                    toks = self._get_table_tokens(page, table_cluster, scale=scale)
                else:
                    toks = []

                per_table_tokens.append(toks)
                page_table_bboxes.append(tbl_box)
                page_clusters.append(table_cluster)

            # Store per-table tokens on the page_input for downstream use
            page_input["per_table_tokens"] = per_table_tokens
            # Keep a dummy "tokens" key for backward compat (unused by fixed path)
            page_input["tokens"] = []

            page_inputs.append(page_input)
            table_bboxes_list.append(page_table_bboxes)
            page_clusters_list.append(page_clusters)
            batched_page_indexes.append(page_idx)

        preprocessed_inputs = self.table_preprocessor.prepare_table_inputs(
            page_inputs,
            table_bboxes_list,
        )

        return {
            'page_inputs': page_inputs,
            'table_bboxes_list': table_bboxes_list,
            'page_clusters_list': page_clusters_list,
            'batched_page_indexes': batched_page_indexes,
            'iocr_pages': preprocessed_inputs['iocr_pages'],
            'table_bboxes': preprocessed_inputs['table_bboxes'],
            'table_images': preprocessed_inputs['table_images'],
            'scale_factors': preprocessed_inputs['scale_factors']
        }

    def _get_tables_from_page(self, page: Page, scale: float = 2.0):
        """Return list of (cluster, scaled_bbox) for table-like clusters."""
        scl = scale
        return [
            (
                cluster,
                [
                    round(cluster.bbox.l) * scl,
                    round(cluster.bbox.t) * scl,
                    round(cluster.bbox.r) * scl,
                    round(cluster.bbox.b) * scl,
                ],
            )
            for cluster in page.predictions.layout.clusters
            if cluster.label in (DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX)
        ]

    def _get_table_tokens(self, page: Page, table_cluster, scale: float = 2.0):
        """
        Match stock semantics as closely as possible:
        query word cells directly from the segmented page for the table bbox,
        and only fall back to cluster cells when no word cells are returned.
        """
        sp = page.parsed_page
        tcells = None

        if sp is not None:
            word_store = getattr(sp, "_word_store", None)
            if word_store is not None and hasattr(word_store, "get_cells_in_bbox"):
                tcells = word_store.get_cells_in_bbox(
                    bbox=table_cluster.bbox,
                    ios=0.8,
                    classes=_CELL_CLASSES,
                )
            else:
                tcells = sp.get_cells_in_bbox(
                    cell_unit=TextCellUnit.WORD,
                    bbox=table_cluster.bbox,
                )

            if len(tcells) == 0:
                tcells = table_cluster.cells
        else:
            tcells = table_cluster.cells

        tokens = []
        sx = sy = scale
        for c in tcells:
            text = c.text.strip()
            if not text:
                continue
            bb = c.rect.to_bounding_box()
            tokens.append(
                {
                    "id": c.index,
                    "text": text,
                    "bbox": {
                        "l": bb.l * sx, "t": bb.t * sy,
                        "r": bb.r * sx, "b": bb.b * sy
                    },
                }
            )
        return tokens

    def _process_table_output(self, page: Page, table_cluster: Any, table_out: Dict) -> Table:
        """
        Convert predictor output to Table while preserving original semantics:
        - When not matching, attach text via backend.get_text_in_rect
        - Always rescale bbox back to page coords (1/scale)
        """
        table_cells = []

        # Original behavior: always attach text when not matching
        attach_text = not self.do_cell_matching

        tf_responses = table_out.get("tf_responses", ())
        _BoundingBox_validate = BoundingBox.model_validate
        _TableCell_validate = TableCell.model_validate
        _scale = 1.0 / page._images_scale

        for element in tf_responses:
            if attach_text:
                bb = _BoundingBox_validate(element["bbox"]).scaled(_scale)
                element["bbox"]["token"] = page.get_text_in_rect(bbox=bb)

            tc = _TableCell_validate(element)
            if tc.bbox is not None:
                tc.bbox = tc.bbox.scaled(_scale)
            table_cells.append(tc)

        pd = table_out.get("predict_details", {})
        num_rows = pd.get("num_rows", 0)
        num_cols = pd.get("num_cols", 0)
        otsl_seq = pd.get("prediction", {}).get("rs_seq", [])

        return Table(
            otsl_seq=otsl_seq,
            table_cells=table_cells,
            num_rows=num_rows,
            num_cols=num_cols,
            id=table_cluster.id,
            page_no=page.page_no,
            cluster=table_cluster,
            label=table_cluster.label,
        )
