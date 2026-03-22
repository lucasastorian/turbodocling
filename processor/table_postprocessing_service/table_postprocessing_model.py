from collections.abc import Iterable
from typing import Any, Dict, List
from itertools import groupby

from docling.datamodel.base_models import Table, Page
from docling_core.types.doc import BoundingBox, TableCell

from docling.datamodel.pipeline_options import TableFormerMode
from processor.table_postprocessing_service.matching_post_processor import MatchingPostProcessor
from processor.table_postprocessing_service.tf_cell_matcher import CellMatcher


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

    def __init__(self):
        self.do_cell_matching = True
        self.mode = TableFormerMode.FAST

        self._cell_matcher = CellMatcher()
        self._post_processor = MatchingPostProcessor()

        self.scale = 2.0

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
        - Always rescale bbox back to page coords (1/self.scale)
        """
        table_cells = []

        # Original behavior: always attach text when not matching
        attach_text = not self.do_cell_matching

        tf_responses = table_out.get("tf_responses", ())
        _BoundingBox_validate = BoundingBox.model_validate
        _TableCell_validate = TableCell.model_validate
        _scale = 1.0 / self.scale

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
