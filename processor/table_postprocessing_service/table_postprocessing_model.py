import logging
import os
import re
from collections.abc import Iterable
from typing import Any, Dict, List
from itertools import groupby

from docling.datamodel.base_models import Table, Page
from docling_core.types.doc import BoundingBox, TableCell

from docling.datamodel.pipeline_options import TableFormerMode
from processor.table_postprocessing_service.matching_post_processor import MatchingPostProcessor
from processor.table_postprocessing_service.tf_cell_matcher import CellMatcher

logger = logging.getLogger(__name__)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def otsl_sqr_chk(rs_list):
    rs_list_split = [
        list(group) for k, group in groupby(rs_list, lambda x: x == "nl") if not k
    ]
    isSquare = True
    if len(rs_list_split) > 0:
        init_tag_len = len(rs_list_split[0]) + 1

        for ind, ln in enumerate(rs_list_split):
            ln.append("nl")
            if len(ln) != init_tag_len:
                isSquare = False

    return isSquare


class TablePostprocessingModel:
    # Minimum tokens in an unmatched group to qualify as a missing row.
    _RECONCILE_MIN_TOKENS = 2
    # Minimum distinct columns spanned to qualify as a missing row.
    _RECONCILE_MIN_COLS = 2
    # If too much of the table is unmatched, audit only: the table is likely structurally broken.
    _RECONCILE_MAX_UNMATCHED_RATIO = 0.35
    # Keep the fallback narrow; lots of orphan groups usually means a bad table, not one missing row.
    _RECONCILE_MAX_GROUPS = 3
    # Numeric values are often split into "$" and "5,125" style tokens.
    _VALUE_LIKE_RE = re.compile(r"^(?:[$€£¥]|[-–—]?\(?\d[\d,]*(?:\.\d+)?\)?%?)$")

    def __init__(self):
        self.do_cell_matching = True
        self.mode = TableFormerMode.FAST

        self._cell_matcher = CellMatcher()
        self._post_processor = MatchingPostProcessor()


    def postprocess(self,
                    all_predictions,
                    pages: List[Page],
                    iocr_pages,
                    table_bboxes,
                    scale_factors,
                    page_clusters_list,
                    batched_page_indexes,
                    correct_overlapping_cells: bool = False,
                    sort_row_col_indexes: bool = True) -> List[Page]:
        outputs: list[tuple[list[dict], dict]] = []

        for i, prediction in enumerate(all_predictions):
            iocr_page = iocr_pages[i]
            scaled_bbox = table_bboxes[i]
            scale_factor = scale_factors[i]

            otsl_sqr_chk(rs_list=prediction["rs_seq"])

            sync, corrected_bboxes = self._check_bbox_sync(prediction)
            if not sync:
                prediction["bboxes"] = corrected_bboxes

            tbl_bbox_for_match = [
                scaled_bbox[0] / scale_factor,
                scaled_bbox[1] / scale_factor,
                scaled_bbox[2] / scale_factor,
                scaled_bbox[3] / scale_factor,
            ]

            # Matching (faithful to original)
            if len(prediction["bboxes"]) > 0:
                matching_details = self._cell_matcher.match_cells(
                    iocr_page, tbl_bbox_for_match, prediction
                )
                if len(iocr_page.get("tokens", [])) > 0:
                    matching_details = self._post_processor.process(
                        matching_details, correct_overlapping_cells,
                    )
                    # Reconciliation: catch any tokens the full pipeline failed to match.
                    # A malformed fallback row is always better than silent token loss.
                    matching_details = self._reconcile_unmatched_tokens(matching_details)

                docling_output = self._generate_tf_response(
                    matching_details["table_cells"], matching_details["matches"]
                )

                docling_output.sort(key=lambda item: item["cell_id"])
                matching_details["docling_responses"] = docling_output
                tf_output = self._merge_tf_output(docling_output, matching_details["pdf_cells"])

                outputs.append((tf_output, matching_details))
            else:
                # Empty prediction — append empty result to maintain 1:1 alignment
                outputs.append(([], {"prediction": prediction}))

        batched_results = outputs

        multi_tf_output: list[dict] = []

        for (tf_responses, predict_details) in batched_results:
            # predict_details here is the "matching_details" dict; we now augment it
            if sort_row_col_indexes:
                # Remap predicted start_row/col IDs to contiguous 0..K-1 indexes (original behavior)
                start_cols = []
                start_rows = []
                for c in tf_responses:
                    sc = c["start_col_offset_idx"]
                    sr = c["start_row_offset_idx"]
                    if sc not in start_cols:
                        start_cols.append(sc)
                    if sr not in start_rows:
                        start_rows.append(sr)
                start_cols.sort()
                start_rows.sort()
                col_remap = {v: i for i, v in enumerate(start_cols)}
                row_remap = {v: i for i, v in enumerate(start_rows)}

                max_end_c = 0
                max_end_r = 0
                for c in tf_responses:
                    c["start_col_offset_idx"] = col_remap[c["start_col_offset_idx"]]
                    c["end_col_offset_idx"] = c["start_col_offset_idx"] + c["col_span"]
                    if c["end_col_offset_idx"] > max_end_c:
                        max_end_c = c["end_col_offset_idx"]

                    c["start_row_offset_idx"] = row_remap[c["start_row_offset_idx"]]
                    c["end_row_offset_idx"] = c["start_row_offset_idx"] + c["row_span"]
                    if c["end_row_offset_idx"] > max_end_r:
                        max_end_r = c["end_row_offset_idx"]

                predict_details["num_cols"] = max_end_c
                predict_details["num_rows"] = max_end_r
            else:
                # Fallback identical to standard: infer from rs_seq when not compacting
                rs_seq = predict_details["prediction"]["rs_seq"] if "prediction" in predict_details else \
                    predict_details.get("prediction", {}).get("rs_seq", None)
                if rs_seq is None:
                    # If not present (e.g., legacy matching_details), keep conservative defaults
                    predict_details["num_cols"] = 0
                    predict_details["num_rows"] = 0
                else:
                    predict_details["num_cols"] = rs_seq.index("nl")
                    predict_details["num_rows"] = rs_seq.count("nl")

            multi_tf_output.append({
                "tf_responses": tf_responses,
                "predict_details": predict_details,
            })

        result_idx = 0
        for i, page_batch_idx in enumerate(batched_page_indexes):
            page = pages[page_batch_idx]
            clusters = page_clusters_list[i]
            n_tables = len(clusters)

            # Use processed multi_tf_output, not raw all_predictions
            page_outputs = multi_tf_output[result_idx: result_idx + n_tables]
            result_idx += n_tables

            for output, table_cluster in zip(page_outputs, clusters):
                table = self._process_table_output(page, table_cluster, output)
                page.predictions.tablestructure.table_map[table_cluster.id] = table

        return pages

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

    @classmethod
    def _token_is_value_like(cls, text: str) -> bool:
        return bool(cls._VALUE_LIKE_RE.match(text.strip()))

    @classmethod
    def _reconcile_unmatched_tokens(cls, matching_details):
        """
        Post-postprocessing audit: find PDF tokens that survived the full
        matching + orphan-recovery pipeline without being assigned to any
        table cell.

        Only synthesizes a fallback row when unmatched tokens look like a
        coherent missing row (>= 2 tokens, spanning >= 2 columns, no
        existing row already at the same y-band, and at least one value-like
        token outside the label column). Otherwise just logs the audit for
        investigation.

        Enabled by default. Set TURBODOCLING_RECONCILE_ROWS=0 to disable
        recovery (audit logging still runs).

        Uses the same cell schema as the stock orphan path in
        MatchingPostProcessor._pick_orphan_cells().
        """
        from collections import defaultdict

        table_cells = matching_details["table_cells"]
        matches = matching_details["matches"]
        pdf_cells = matching_details["pdf_cells"]

        if not pdf_cells or not table_cells:
            return matching_details

        # 1. Audit: which tokens are unmatched?
        matched_ids = set(matches.keys())
        unmatched = [pc for pc in pdf_cells
                     if str(pc["id"]) not in matched_ids
                     and pc.get("text", "").strip()]

        if not unmatched:
            return matching_details

        enabled = os.environ.get("TURBODOCLING_RECONCILE_ROWS", "1") == "1"

        unmatched_ratio = len(unmatched) / len(pdf_cells)
        if unmatched_ratio > cls._RECONCILE_MAX_UNMATCHED_RATIO:
            logger.warning(
                "RECONCILE_AUDIT: %d/%d unmatched tokens (ratio=%.2f) - audit only",
                len(unmatched),
                len(pdf_cells),
                unmatched_ratio,
            )
            return matching_details

        # 2. Build column boundaries and row y-bands from existing cells
        col_x = defaultdict(lambda: [float('inf'), float('-inf')])
        row_ys = defaultdict(list)
        row_bands = defaultdict(lambda: [float('inf'), float('-inf')])
        for tc in table_cells:
            bb = tc["bbox"]
            cx = col_x[tc["column_id"]]
            if bb[0] < cx[0]:
                cx[0] = bb[0]
            if bb[2] > cx[1]:
                cx[1] = bb[2]
            row_ys[tc["row_id"]].append((bb[1] + bb[3]) / 2.0)
            band = row_bands[tc["row_id"]]
            if bb[1] < band[0]:
                band[0] = bb[1]
            if bb[3] > band[1]:
                band[1] = bb[3]

        row_y_avg = {rid: sum(v) / len(v) for rid, v in row_ys.items()}
        sorted_avg_ys = sorted(row_y_avg.values())
        leftmost_col = min(col_x.keys())

        # Estimate row spacing for clustering threshold
        if len(sorted_avg_ys) >= 2:
            threshold = (sorted_avg_ys[-1] - sorted_avg_ys[0]) / (len(sorted_avg_ys) - 1) * 0.6
        else:
            threshold = 15.0

        # 3. Cluster unmatched tokens into groups by y-proximity
        unmatched.sort(key=lambda pc: (pc["bbox"][1] + pc["bbox"][3]) / 2.0)

        groups = [[unmatched[0]]]
        for pc in unmatched[1:]:
            prev_y = (groups[-1][-1]["bbox"][1] + groups[-1][-1]["bbox"][3]) / 2.0
            curr_y = (pc["bbox"][1] + pc["bbox"][3]) / 2.0
            if curr_y - prev_y < threshold:
                groups[-1].append(pc)
            else:
                groups.append([pc])

        if len(groups) > cls._RECONCILE_MAX_GROUPS:
            logger.warning(
                "RECONCILE_AUDIT: %d unmatched groups - audit only", len(groups)
            )
            return matching_details

        # 4. Filter: only groups that look like coherent missing rows
        max_row = max(tc["row_id"] for tc in table_cells)
        max_cell = max(tc["cell_id"] for tc in table_cells)
        new_matches = dict(matches)
        recovered = 0

        for group in groups:
            texts = [pc.get("text", "") for pc in group]
            group_y = sum((pc["bbox"][1] + pc["bbox"][3]) / 2.0 for pc in group) / len(group)
            group_top = min(pc["bbox"][1] for pc in group)
            group_bottom = max(pc["bbox"][3] for pc in group)

            # Check: does an existing row already occupy this y-band?
            y_margin = min(4.0, threshold * 0.25)
            y_occupied = any(
                not (group_bottom < band[0] - y_margin or group_top > band[1] + y_margin)
                for band in row_bands.values()
            )

            # Check: how many distinct columns does this group span?
            col_assignments = set()
            token_col_map = []
            for pc in group:
                pc_l, pc_r = pc["bbox"][0], pc["bbox"][2]
                best_col = 0
                best_overlap = 0.0
                for col_id, (cl, cr) in col_x.items():
                    overlap = min(pc_r, cr) - max(pc_l, cl)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_col = col_id
                # Only count if there's actual positive overlap
                if best_overlap > 0:
                    col_assignments.add(best_col)
                token_col_map.append((pc, best_col, best_overlap))

            positive_tokens = [(pc, col, overlap) for pc, col, overlap in token_col_map if overlap > 0]
            value_like_cols = {
                col for pc, col, _ in positive_tokens
                if col != leftmost_col and cls._token_is_value_like(pc.get("text", ""))
            }
            has_label_col = leftmost_col in col_assignments

            is_row_candidate = (
                len(positive_tokens) >= cls._RECONCILE_MIN_TOKENS
                and len(col_assignments) >= cls._RECONCILE_MIN_COLS
                and has_label_col
                and bool(value_like_cols)
                and not y_occupied
            )

            if not is_row_candidate or not enabled:
                # Audit only — log but don't modify output
                logger.warning(
                    "RECONCILE_AUDIT: %d unmatched tokens (row_candidate=%s, enabled=%s, "
                    "cols=%s, value_cols=%s, y_occupied=%s): %s",
                    len(group),
                    is_row_candidate,
                    enabled,
                    sorted(col_assignments),
                    sorted(value_like_cols),
                    y_occupied,
                    texts,
                )
                continue

            # 5. Synthesize fallback row
            max_row += 1
            cell_id_by_col = {}
            cell_ref_by_col = {}
            for pc, col, _ in positive_tokens:
                if col not in cell_id_by_col:
                    max_cell += 1
                    new_cell = {
                        "bbox": pc["bbox"][:],
                        "cell_id": max_cell,
                        "column_id": col,
                        "label": "body",
                        "row_id": max_row,
                        "cell_class": 2,
                    }
                    table_cells.append(new_cell)
                    cell_id_by_col[col] = max_cell
                    cell_ref_by_col[col] = new_cell
                else:
                    cell = cell_ref_by_col[col]
                    bbox = cell["bbox"]
                    pc_bbox = pc["bbox"]
                    cell["bbox"] = [
                        min(bbox[0], pc_bbox[0]),
                        min(bbox[1], pc_bbox[1]),
                        max(bbox[2], pc_bbox[2]),
                        max(bbox[3], pc_bbox[3]),
                    ]

                new_matches[str(pc["id"])] = [
                    {"post": 1.0, "table_cell_id": cell_id_by_col[col]}
                ]
                recovered += 1

            logger.warning(
                "RECONCILE: %d unmatched tokens -> fallback row %d across cols=%s: %s",
                len(group),
                max_row,
                sorted(cell_id_by_col.keys()),
                texts,
            )

        if recovered > 0:
            matching_details["table_cells"] = table_cells
            matching_details["matches"] = new_matches

        return matching_details

    def _remove_bbox_span_desync(self, prediction):
        # Delete 1 extra bbox after span tag
        index_to_delete_from = 0
        indexes_to_delete = []
        newbboxes = []
        for html_elem in prediction["html_seq"]:
            if html_elem == "<td>":
                index_to_delete_from += 1
            if html_elem == ">":
                index_to_delete_from += 1
                indexes_to_delete.append(index_to_delete_from)

        newbboxes = self._deletebbox(prediction["bboxes"], indexes_to_delete)
        return newbboxes

    def _check_bbox_sync(self, prediction):
        bboxes = []
        match = False
        # count bboxes
        count_bbox = len(prediction["bboxes"])
        # count td tags
        count_td = 0
        for html_elem in prediction["html_seq"]:
            if html_elem == "<td>" or html_elem == ">":
                count_td += 1
            if html_elem in ["fcel", "ecel", "ched", "rhed", "srow"]:
                count_td += 1
        if count_bbox != count_td:
            bboxes = self._remove_bbox_span_desync(prediction)
        else:
            bboxes = prediction["bboxes"]
            match = True
        return match, bboxes

    def _deletebbox(self, listofbboxes, index):
        newlist = []
        for i in range(len(listofbboxes)):
            bbox = listofbboxes[i]
            if i not in index:
                newlist.append(bbox)
        return newlist

    def _generate_tf_response(self, table_cells, matches) -> List:
        r"""
        Convert the matching details to the expected output for Docling

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id",
                                                  "bbox", "label", "class"
        matches : dictionary of lists of table_cells
            A dictionary which is indexed by the pdf_cell_id as key and the value is a list
            of the table_cells that fall inside that pdf cell

        Returns
        -------
        docling_output : string
            json response formatted according to Docling api expectations
        """

        # format output to look similar to tests/examples/tf_gte_output_2.json
        table_cell_by_id = {tc["cell_id"]: tc for tc in table_cells}
        tf_cell_list = []
        for pdf_cell_id, pdf_cell_matches in matches.items():
            tf_cell = {
                "bbox": {},  # b,l,r,t,token
                "row_span": 1,
                "col_span": 1,
                "start_row_offset_idx": -1,
                "end_row_offset_idx": -1,
                "start_col_offset_idx": -1,
                "end_col_offset_idx": -1,
                "indentation_level": 0,
                # return text cell bboxes additionally to the matched index
                "text_cell_bboxes": [{}],  # b,l,r,t,token
                "column_header": False,
                "row_header": False,
                "row_section": False,
            }
            tf_cell["cell_id"] = int(pdf_cell_id)

            row_ids = set()
            column_ids = set()
            labels = set()

            for match in pdf_cell_matches:
                tm = match["table_cell_id"]
                table_cell = table_cell_by_id.get(tm)
                if table_cell is not None:
                    row_ids.add(table_cell["row_id"])
                    column_ids.add(table_cell["column_id"])
                    labels.add(table_cell["label"])

                    if table_cell["label"] is not None:
                        if table_cell["label"] in ["ched"]:
                            tf_cell["column_header"] = True
                        if table_cell["label"] in ["rhed"]:
                            tf_cell["row_header"] = True
                        if table_cell["label"] in ["srow"]:
                            tf_cell["row_section"] = True

                    tf_cell["start_col_offset_idx"] = table_cell["column_id"]
                    tf_cell["end_col_offset_idx"] = table_cell["column_id"] + 1
                    tf_cell["start_row_offset_idx"] = table_cell["row_id"]
                    tf_cell["end_row_offset_idx"] = table_cell["row_id"] + 1

                    if "colspan_val" in table_cell:
                        tf_cell["col_span"] = table_cell["colspan_val"]
                        tf_cell["start_col_offset_idx"] = table_cell["column_id"]
                        off_idx = table_cell["column_id"] + tf_cell["col_span"]
                        tf_cell["end_col_offset_idx"] = off_idx
                    if "rowspan_val" in table_cell:
                        tf_cell["row_span"] = table_cell["rowspan_val"]
                        tf_cell["start_row_offset_idx"] = table_cell["row_id"]
                        tf_cell["end_row_offset_idx"] = (
                            table_cell["row_id"] + tf_cell["row_span"]
                        )
                    if "bbox" in table_cell:
                        table_match_bbox = table_cell["bbox"]
                        tf_bbox = {
                            "b": table_match_bbox[3],
                            "l": table_match_bbox[0],
                            "r": table_match_bbox[2],
                            "t": table_match_bbox[1],
                        }
                        tf_cell["bbox"] = tf_bbox

            tf_cell["row_ids"] = list(row_ids)
            tf_cell["column_ids"] = list(column_ids)
            tf_cell["label"] = "None"
            l_labels = list(labels)
            if len(l_labels) > 0:
                tf_cell["label"] = l_labels[0]
            tf_cell_list.append(tf_cell)
        return tf_cell_list

    def _merge_tf_output(self, docling_output, pdf_cells):
        tf_output = []
        tf_cells_map = {}
        max_row_idx = 0
        pdf_cell_by_id = {pc["id"]: pc for pc in pdf_cells}

        for docling_item in docling_output:
            r_idx = str(docling_item["start_row_offset_idx"])
            c_idx = str(docling_item["start_col_offset_idx"])
            cell_key = c_idx + "_" + r_idx
            if cell_key in tf_cells_map:
                pdf_cell = pdf_cell_by_id.get(docling_item["cell_id"])
                if pdf_cell is not None:
                    text_cell_bbox = {
                        "b": pdf_cell["bbox"][3],
                        "l": pdf_cell["bbox"][0],
                        "r": pdf_cell["bbox"][2],
                        "t": pdf_cell["bbox"][1],
                        "token": pdf_cell["text"],
                    }
                    tf_cells_map[cell_key]["text_cell_bboxes"].append(
                        text_cell_bbox
                    )
            else:
                tf_cells_map[cell_key] = {
                    "bbox": docling_item["bbox"],
                    "row_span": docling_item["row_span"],
                    "col_span": docling_item["col_span"],
                    "start_row_offset_idx": docling_item["start_row_offset_idx"],
                    "end_row_offset_idx": docling_item["end_row_offset_idx"],
                    "start_col_offset_idx": docling_item["start_col_offset_idx"],
                    "end_col_offset_idx": docling_item["end_col_offset_idx"],
                    "indentation_level": docling_item["indentation_level"],
                    "text_cell_bboxes": [],
                    "column_header": docling_item["column_header"],
                    "row_header": docling_item["row_header"],
                    "row_section": docling_item["row_section"],
                }

                if docling_item["start_row_offset_idx"] > max_row_idx:
                    max_row_idx = docling_item["start_row_offset_idx"]

                pdf_cell = pdf_cell_by_id.get(docling_item["cell_id"])
                if pdf_cell is not None:
                    text_cell_bbox = {
                        "b": pdf_cell["bbox"][3],
                        "l": pdf_cell["bbox"][0],
                        "r": pdf_cell["bbox"][2],
                        "t": pdf_cell["bbox"][1],
                        "token": pdf_cell["text"],
                    }
                    tf_cells_map[cell_key]["text_cell_bboxes"].append(
                        text_cell_bbox
                    )

        for k in tf_cells_map:
            tf_output.append(tf_cells_map[k])
        return tf_output
