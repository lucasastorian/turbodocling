#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import contextlib
import glob
import json
import logging
import os
import threading
from itertools import groupby
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_model

import docling_ibm_models.tableformer.common as c
import docling_ibm_models.tableformer.data_management.transforms as T
import docling_ibm_models.tableformer.settings as s
import docling_ibm_models.tableformer.utils.utils as u
from docling_ibm_models.tableformer.otsl import otsl_to_html
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

from processor.gpu_service.tablemodel04_rs import TableModel04_rs
from processor.shared.timers import _CPUTimer, _CudaTimer

LOG_LEVEL = logging.INFO

logger = s.get_custom_logger(__name__, LOG_LEVEL)

_model_init_lock = threading.Lock()


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


def otsl_sqr_chk(rs_list, logdebug):
    rs_list_split = [
        list(group) for k, group in groupby(rs_list, lambda x: x == "nl") if not k
    ]
    isSquare = True
    if len(rs_list_split) > 0:
        init_tag_len = len(rs_list_split[0]) + 1

        totcelnum = rs_list.count("fcel") + rs_list.count("ecel")
        if logdebug:
            logger.debug("Total number of cells = {}".format(totcelnum))

        for ind, ln in enumerate(rs_list_split):
            ln.append("nl")
            if logdebug:
                logger.debug("{}".format(ln))
            if len(ln) != init_tag_len:
                isSquare = False
        if isSquare:
            if logdebug:
                logger.debug(
                    "{}*OK* Table is square! *OK*{}".format(
                        bcolors.OKGREEN, bcolors.ENDC
                    )
                )
        else:
            if logdebug:
                err_name = "{}***** ERR ******{}"
                logger.debug(err_name.format(bcolors.FAIL, bcolors.ENDC))
                logger.debug(
                    "{}*ERR* Table is not square! *ERR*{}".format(
                        bcolors.FAIL, bcolors.ENDC
                    )
                )
    return isSquare


class TFPredictor:
    r"""
    Table predictions for the in-memory Docling API
    """

    def __init__(self, config, device: str = "cpu", num_threads: int = 4):
        r"""
        Parameters
        ----------
        config : dict Parameters configuration
        device: (Optional) torch device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'

        Raises
        ------
        ValueError
        When the model cannot be found
        """
        self._device = torch.device(device)
        self._log().info("Running on device: {}".format(device))

        self._config = config
        self.enable_post_process = True

        self._padding = config["predict"].get("padding", False)
        self._padding_size = config["predict"].get("padding_size", 10)

        self._init_word_map()

        # Set the number of threads
        if device == "cpu":
            self._num_threads = num_threads
            torch.set_num_threads(self._num_threads)

        # Load the model
        self._model = self._load_model()
        self._model.eval()
        self._prof = config["predict"].get("profiling", False)
        self._profiling_agg_window = config["predict"].get("profiling_agg_window", None)
        if self._profiling_agg_window is not None:
            AggProfiler(self._profiling_agg_window)
        else:
            AggProfiler()

    def _init_word_map(self):
        self._prepared_data_dir = c.safe_get_parameter(
            self._config, ["dataset", "prepared_data_dir"], required=False
        )

        if self._prepared_data_dir is None:
            self._word_map = c.safe_get_parameter(
                self._config, ["dataset_wordmap"], required=True
            )
        else:
            data_name = c.safe_get_parameter(
                self._config, ["dataset", "name"], required=True
            )
            word_map_fn = c.get_prepared_data_filename("WORDMAP", data_name)

            # Load word_map
            with open(os.path.join(self._prepared_data_dir, word_map_fn), "r") as f:
                self._log().debug("Load WORDMAP from: {}".format(word_map_fn))
                self._word_map = json.load(f)

        self._init_data = {"word_map": self._word_map}
        # Prepare a reversed index for the word map
        self._rev_word_map = {v: k for k, v in self._word_map["word_map_tag"].items()}

    def get_init_data(self):
        r"""
        Return the initialization data
        """
        return self._init_data

    def get_model(self):
        r"""
        Return the loaded model
        """
        return self._model

    def _load_model(self):
        r"""
        Load the proper model
        """

        self._model_type = self._config["model"]["type"]

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            model = TableModel04_rs(self._config, self._init_data, self._device)

            self._remove_padding = False
            if self._model_type == "TableModel02":
                self._remove_padding = True

            # Load model from safetensors
            save_dir = self._config["model"]["save_dir"]
            models_fn = sorted(glob.glob(f"{save_dir}/tableformer_*.safetensors"))
            if not models_fn:
                err_msg = "Not able to find a model file for {}".format(
                    self._model_type
                )
                self._log().error(err_msg)
                raise ValueError(err_msg)
            model_fn = models_fn[
                0
            ]  # Take the first tableformer safetensors file inside the save_dir
            missing, unexpected = load_model(model, model_fn, device=str(self._device))
            if missing or unexpected:
                err_msg = "Not able to load the model weights for {}".format(
                    self._model_type
                )
                self._log().error(err_msg)
                raise ValueError(err_msg)

            # Setup model for inference (bf16 conversion, eval mode) AFTER loading weights
            model.setup_for_inference()
            self._log().info("Model configured for optimized inference (bf16 transformers, fp32 encoder)")

            # Prepare the encoder for inference AFTER loading weights
            # This handles fusion, compilation, and graph capture in correct order
            if hasattr(model, '_encoder') and hasattr(model._encoder, 'prepare_for_inference'):
                # prepare_for_inference returns the encoder (may be compiled wrapper)
                model._encoder = model._encoder.prepare_for_inference(device=self._device)
                self._log().info("Encoder prepared for inference after weight loading")

        return model

    def get_device(self):
        return self._device

    def get_model_type(self):
        return self._model_type

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def predict(
            self,
            iocr_pages: list[dict],
            table_bboxes: list[list[float]],
            table_images: list[np.ndarray],
            scale_factors: list[float],
            eval_res_preds=None,
    ):
        """
        Faithful batched variant of the original single-table `predict`:
          - Preprocess table crops with the same Normalize+Resize pipeline
          - Call model once on a stacked batch
          - For each item: build prediction dict (bboxes/classes/tag_seq/rs_seq/html_seq), depad if needed
          - Check bbox/tag sync
          - Match cells (or dummy), optional post-process
          - Generate Docling responses, sort, and merge into tf_output
        Returns: list of (tf_output, matching_details) in the same order as inputs.
        """
        assert len(iocr_pages) == len(table_bboxes) == len(table_images) == len(scale_factors), \
            "All batched inputs must be same length"

        AggProfiler().start_agg(self._prof)

        is_cuda = str(self._device).startswith('cuda')
        timer = _CudaTimer() if is_cuda else _CPUTimer()

        with timer.time_section('image_preprocessing'):
            resized_size = self._config["dataset"]["resized_image"]
            mean = self._config["dataset"]["image_normalization"]["mean"]
            std = self._config["dataset"]["image_normalization"]["std"]
            image_batch = self._batch_preprocess_images(table_images=table_images, resized_size=resized_size,
                                                        mean=mean, std=std)

        with timer.time_section('model_inference'):
            max_steps = self._config["predict"]["max_steps"]
            beam_size = self._config["predict"]["beam_size"]

            all_predictions: list[dict] = []

            with torch.no_grad():
                if eval_res_preds is not None:
                    for ev in eval_res_preds:
                        pred = {
                            "bboxes": ev.get("bboxes", []),
                            "tag_seq": ev.get("tag_seq", []),
                        }
                        pred["rs_seq"] = self._get_html_tags(pred["tag_seq"])
                        pred["html_seq"] = otsl_to_html(pred["rs_seq"], False)
                        all_predictions.append(pred)
                else:
                    model_result = self._model.predict(image_batch, max_steps, beam_size)

                    def _normalize_model_batch_outputs(model_result):
                        # Already a per-item list of tuples
                        if isinstance(model_result, (list, tuple)) and model_result and isinstance(model_result[0],
                                                                                                   (list, tuple)):
                            return list(model_result)
                        # Tuple of batched outputs
                        if isinstance(model_result, (list, tuple)) and len(model_result) in (2, 3):
                            if len(model_result) == 3:
                                seq_batch, class_batch, coord_batch = model_result
                            else:
                                seq_batch, class_batch, coord_batch = model_result[0], None, None
                            if not isinstance(seq_batch, (list, tuple)):
                                seq_batch = list(seq_batch)
                            out = []
                            for i in range(len(seq_batch)):
                                oc = class_batch[i] if class_batch is not None else None
                                od = coord_batch[i] if coord_batch is not None else None
                                out.append((seq_batch[i], oc, od))
                            return out
                        # Fallback: single item replicated?
                        return [(model_result, None, None)]

                    triples = _normalize_model_batch_outputs(model_result)

                    # Build original-format prediction dicts per item
                    for (seq, outputs_class, outputs_coord) in triples:
                        pred = {}
                        # bboxes
                        if outputs_coord is not None:
                            if torch.is_tensor(outputs_coord) and outputs_coord.numel() == 0:
                                pred["bboxes"] = []
                            else:
                                bbox_xyxy = u.box_cxcywh_to_xyxy(outputs_coord)
                                pred["bboxes"] = bbox_xyxy.tolist() if torch.is_tensor(bbox_xyxy) else bbox_xyxy
                        else:
                            pred["bboxes"] = []
                        # classes
                        if outputs_class is not None:
                            if torch.is_tensor(outputs_class) and outputs_class.numel() > 0:
                                pred["classes"] = torch.argmax(outputs_class, dim=1).tolist()
                            elif isinstance(outputs_class, (list, tuple)):
                                pred["classes"] = list(outputs_class)
                            else:
                                pred["classes"] = []
                        else:
                            pred["classes"] = []
                        # tag seq (+ optional depadding)
                        if self._remove_padding:
                            seq, _ = u.remove_padding(seq)
                        pred["tag_seq"] = seq
                        pred["rs_seq"] = self._get_html_tags(seq)
                        pred["html_seq"] = otsl_to_html(pred["rs_seq"], False)

                        all_predictions.append(pred)

        return all_predictions

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
        self._log().debug(
            "======================= PREDICTED BBOXES: {}".format(count_bbox)
        )
        self._log().debug(
            "=======================  PREDICTED CELLS: {}".format(count_td)
        )
        if count_bbox != count_td:
            bboxes = self._remove_bbox_span_desync(prediction)
        else:
            bboxes = prediction["bboxes"]
            match = True
        return match, bboxes

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
                # remove element from bboxes
                self._log().debug(
                    "========= DELETE BBOX INDEX: {}".format(index_to_delete_from)
                )
                indexes_to_delete.append(index_to_delete_from)

        newbboxes = self._deletebbox(prediction["bboxes"], indexes_to_delete)
        return newbboxes

    def _deletebbox(self, listofbboxes, index):
        newlist = []
        for i in range(len(listofbboxes)):
            bbox = listofbboxes[i]
            if i not in index:
                newlist.append(bbox)
        return newlist

    def _merge_tf_output(self, docling_output, pdf_cells):
        tf_output = []
        tf_cells_map = {}
        max_row_idx = 0

        for docling_item in docling_output:
            r_idx = str(docling_item["start_row_offset_idx"])
            c_idx = str(docling_item["start_col_offset_idx"])
            cell_key = c_idx + "_" + r_idx
            if cell_key in tf_cells_map:
                for pdf_cell in pdf_cells:
                    if pdf_cell["id"] == docling_item["cell_id"]:
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

                for pdf_cell in pdf_cells:
                    if pdf_cell["id"] == docling_item["cell_id"]:
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

    def _generate_tf_response(self, table_cells, matches):
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
                tcl = list(
                    filter(lambda table_cell: table_cell["cell_id"] == tm, table_cells)
                )
                if len(tcl) > 0:
                    table_cell = tcl[0]
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

    def _prepare_image(self, mat_image):
        r"""
        Rescale the image and prepare a batch of 1 with the image as as tensor

        Parameters
        ----------
        mat_image: cv2.Mat
            The image as an openCV Mat object

        Returns
        -------
        tensor (batch_size, image_channels, resized_image, resized_image)
        """
        normalize = T.Normalize(
            mean=self._config["dataset"]["image_normalization"]["mean"],
            std=self._config["dataset"]["image_normalization"]["std"],
        )
        resized_size = self._config["dataset"]["resized_image"]
        resize = T.Resize([resized_size, resized_size])

        img, _ = normalize(mat_image, None)
        img, _ = resize(img, None)

        img = img.transpose(2, 1, 0)  # (channels, width, height)
        img = torch.FloatTensor(img / 255.0)
        image_batch = img.unsqueeze(dim=0)
        image_batch = image_batch.to(device=self._device)
        return image_batch

    def _get_html_tags(self, seq):
        r"""
        Convert indices to actual html tags

        """
        # Map the tag indices back to actual tags (without start, end)
        html_tags = [self._rev_word_map[ind] for ind in seq[1:-1]]

        return html_tags

    def _batch_preprocess_images(self, table_images: List[np.ndarray], resized_size: int,  mean, std, dtype=torch.float32, use_stream=True):
        """Batch preprocess images"""
        if not table_images:
            return torch.empty((0, 3, int(resized_size), int(resized_size)), device=self._device, dtype=dtype)

            # --- bucket by (H, W, C) so each bucket stacks cleanly ---
        buckets = {}  # (H, W, C) -> [indices]
        shapes = []
        for i, img in enumerate(table_images):
            if img.ndim == 2:
                # promote grayscale to HxWx1 to avoid surprises
                img = img[..., None]
                table_images[i] = img
            if img.dtype != np.uint8:
                table_images[i] = img.astype(np.uint8, copy=False)
            h, w, c = img.shape
            key = (h, w, c)
            buckets.setdefault(key, []).append(i)
            shapes.append(key)

        # sanity: channel count should be consistent for your pipeline
        first_c = shapes[0][2]
        if any(c != first_c for (_, _, c) in shapes):
            raise ValueError(f"Inconsistent channel counts detected: {[s[2] for s in shapes]}")

        N, C, S = len(table_images), first_c, int(resized_size)
        out = torch.empty((N, C, S, S), device=self._device, dtype=dtype)

        # prepare normalization buffers once
        mean_t = torch.tensor(mean, device=self._device, dtype=dtype).view(1, 1, 1, C)
        std_t = torch.tensor(std, device=self._device, dtype=dtype).view(1, 1, 1, C)

        # CUDA streams only on CUDA devices
        is_cuda = self._device.type == "cuda"
        use_cuda_stream = use_stream and is_cuda

        if use_cuda_stream:
            stream = torch.cuda.Stream(device=self._device)
            stream_ctx = torch.cuda.stream(stream)
        else:
            stream_ctx = contextlib.nullcontext()

        with stream_ctx:
            for (h, w, c), idxs in buckets.items():
                # stack this bucket on CPU (identical shapes)
                nhwc = np.stack([table_images[i] for i in idxs], axis=0)  # [B,H,W,C], uint8
                cpu = torch.from_numpy(nhwc)
                if is_cuda:
                    cpu = cpu.pin_memory()

                # H2D once, normalize, resize on GPU
                t = cpu.to(device=self._device, dtype=dtype, non_blocking=is_cuda)  # [B,H,W,C]
                t = t / 255.0
                t = (t - mean_t) / std_t

                # NCHW for interpolate
                t = t.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)  # [B,C,H,W]
                t = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)

                # your layout quirk: (C, W, H). Swap the last two dims.
                t = t.permute(0, 1, 3, 2).contiguous()  # [B,C,W,H]

                # place back in original order
                out[torch.as_tensor(idxs, device=self._device, dtype=torch.long)] = t

        if use_cuda_stream:
            torch.cuda.current_stream(device=self._device).wait_stream(stream)

        return out  # (N, C, W, H), float32 on device
