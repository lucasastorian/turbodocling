# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT

import logging
import os
import threading
from collections.abc import Iterable
from typing import Dict, List, Set, Union, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

from docling_ibm_models.layoutmodel.labels import LayoutLabels
from processor.shared.timers import _CPUTimer, _CudaTimer

from processor.gpu_service.gpu_preprocess import GPUPreprocessor

_log = logging.getLogger(__name__)
_model_init_lock = threading.Lock()

# ---- Static shape constants (graph-friendly) ----
FIXED_BS = int(os.getenv("LAYOUT_FIXED_BS", "32"))
FIXED_H, FIXED_W = 640, 640


# Recommended allocator setting (set in the environment before import/process start):
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


class LayoutPredictor:
    """
    Document layout prediction with GPU-accelerated preprocessing,
    Inductor compile (dynamic=False), and CUDA Graph replay on fixed shapes.
    """

    def __init__(
            self,
            artifact_path: str,
            device: str = "cuda",
            num_threads: int = 4,
            base_threshold: float = 0.3,
            blacklist_classes: Optional[Set[str]] = None,
            use_gpu_preprocess: bool = True,
    ):
        self._black_classes = blacklist_classes if blacklist_classes is not None else set()
        self._labels = LayoutLabels()
        self._threshold = base_threshold
        self._compat_mode = os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes")

        self._device = torch.device(device)
        self._num_threads = num_threads
        if self._device.type == "cpu":
            torch.set_num_threads(self._num_threads)

        # ---- Load configs & processor ----
        self._processor_config = os.path.join(artifact_path, "preprocessor_config.json")
        self._model_config = os.path.join(artifact_path, "config.json")
        self._st_fn = os.path.join(artifact_path, "model.safetensors")
        if not os.path.isfile(self._st_fn):
            raise FileNotFoundError(f"Missing safe tensors file: {self._st_fn}")
        if not os.path.isfile(self._processor_config):
            raise FileNotFoundError(f"Missing processor config file: {self._processor_config}")
        if not os.path.isfile(self._model_config):
            raise FileNotFoundError(f"Missing model config file: {self._model_config}")

        self._image_preprocessor = RTDetrImageProcessor.from_json_file(self._processor_config)
        self._image_postprocessor = RTDetrImageProcessor.from_json_file(self._processor_config)

        # ---- Optional GPU preprocessor (must emit 640x640) ----
        self._use_gpu_preprocess = use_gpu_preprocess and self._device.type == "cuda"
        self._gpu_preprocessor = None

        if self._use_gpu_preprocess:
            self._gpu_preprocessor = GPUPreprocessor(
                size={"height": FIXED_H, "width": FIXED_W},
                do_pad=True,
                pad_size={"height": FIXED_H, "width": FIXED_W},
                do_rescale=self._image_preprocessor.do_rescale,
                rescale_factor=self._image_preprocessor.rescale_factor,
                do_normalize=self._image_preprocessor.do_normalize,
                mean=tuple(self._image_preprocessor.image_mean),
                std=tuple(self._image_preprocessor.image_std),
                device=str(self._device),
                dtype=torch.float32,
            )

        with _model_init_lock:
            self._model = AutoModelForObjectDetection.from_pretrained(
                artifact_path, config=self._model_config, device_map=self._device
            )
            self._model = self._model.to(memory_format=torch.contiguous_format)
            self._model.eval()

            if self._device.type == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = False

                self._model.to(memory_format=torch.channels_last)

                if os.getenv("TORCH_COMPILE", "").lower() in ("1", "true", "yes"):
                    # Inductor compile settings
                    from torch._inductor import config as ind_cfg
                    ind_cfg.triton.cudagraphs = True
                    ind_cfg.max_autotune_gemm = False
                    ind_cfg.coordinate_descent_tuning = False

                    self._model = torch.compile(
                        self._model,
                        dynamic=False,
                        fullgraph=True,
                        mode="reduce-overhead"
                    )
                    _log.info("Torch compile enabled via TORCH_COMPILE environment variable")

                self._static_in = torch.empty(
                    FIXED_BS, 3, FIXED_H, FIXED_W, device=self._device,
                    memory_format=torch.channels_last, dtype=torch.float32
                )

                # ---- Warm-up: 1st = compile, 2nd = graph capture ----
                with torch.inference_mode():
                    for _ in range(2):
                        _ = self._model(pixel_values=self._static_in)
                torch.cuda.synchronize()

            elif self._device.type == "mps":
                # MPS warmup (no CUDA graphs or TF32)
                self._static_in = torch.empty(
                    FIXED_BS, 3, FIXED_H, FIXED_W, device=self._device, dtype=torch.float32
                )
                with torch.inference_mode():
                    _ = self._model(pixel_values=self._static_in)
                torch.mps.synchronize()

            else:
                # CPU fallback - no special optimizations
                self._static_in = torch.empty(
                    FIXED_BS, 3, FIXED_H, FIXED_W, device=self._device, dtype=torch.float32
                )
                with torch.inference_mode():
                    _ = self._model(pixel_values=self._static_in)

        self._model_name = type(self._model).__name__

        if self._model_name == "RTDetrForObjectDetection":
            self._classes_map = self._labels.shifted_canonical_categories()
            self._label_offset = 1
        else:
            self._classes_map = self._labels.canonical_categories()
            self._label_offset = 0

    def _create_timer(self):
        return _CudaTimer() if self._device.type == "cuda" else _CPUTimer()

    def _store_timings(self, timer):
        # Accumulate timings across batches instead of overwriting
        if not hasattr(self, '_t_preprocess_ms'):
            self._t_preprocess_ms = 0.0
            self._t_predict_ms = 0.0
            self._t_postprocess_ms = 0.0

        self._t_preprocess_ms += timer.get_time('preprocess')
        self._t_predict_ms += timer.get_time('predict')
        self._t_postprocess_ms += timer.get_time('postprocess')

    def _stable_sort_result(self, res: Dict[str, Tensor]) -> Dict[str, Tensor]:
        boxes, scores, labels = res["boxes"], res["scores"], res["labels"]
        key = torch.stack([
            labels.to(torch.int64),
            (-scores).to(torch.float32),
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3],
        ], dim=1)
        idx = torch.arange(key.size(0), device=key.device)
        for col in range(key.size(1) - 1, -1, -1):
            vals = key[idx, col]
            order = torch.argsort(vals, stable=True)
            idx = idx[order]
        return {"boxes": boxes[idx], "scores": scores[idx], "labels": labels[idx]}

    @torch.inference_mode()
    def _preprocess_batch(self, pil_images: List[Image.Image], b: int) -> torch.Tensor:
        """
        Preprocess only the first b images (real images), zero out padding.
        Returns device tensor [FIXED_BS,3,640,640] with stable address for CUDA graphs.
        """
        # Preprocess only b real images (not the padded duplicates)
        if self._use_gpu_preprocess and self._gpu_preprocessor is not None:
            pre = self._gpu_preprocessor.preprocess_batch(pil_images[:b])  # only b images
            pv = pre["pixel_values"].contiguous(memory_format=torch.channels_last)
            self._static_in[:b].copy_(pv, non_blocking=True)
        else:
            inputs = self._image_preprocessor(images=pil_images[:b], return_tensors="pt")  # only b images
            pv = inputs.pixel_values
            # pin_memory() only works on CUDA, not MPS
            if self._device.type == "cuda":
                pv = pv.pin_memory()
            self._static_in[:b].copy_(pv.to(self._static_in.dtype, non_blocking=True), non_blocking=True)

        # Zero the padded part (fast set; fixed address, fixed shape)
        if b < FIXED_BS:
            self._static_in[b:].zero_()
        return self._static_in

    def _to_pil_rgb(self, images: List[Union[Image.Image, np.ndarray]]) -> List[Image.Image]:
        out = []
        for img in images:
            if isinstance(img, Image.Image):
                out.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                out.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError("Unsupported input image format")
        return out

    def _slice_model_output(self, outputs, b: int, full_bs: int):
        """Preserve the exact HF ModelOutput subclass while slicing batch dimension."""
        data = {}
        for k, v in outputs.items():  # ModelOutput is dict-like
            if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == full_bs:
                data[k] = v[:b]
            else:
                data[k] = v
        return outputs.__class__(**data)

    def _postprocess(
            self,
            outputs,
            target_sizes: torch.Tensor,
    ) -> List[Dict[str, Tensor]]:
        threshold = max(0.0, self._threshold - 1e-4) if self._compat_mode else self._threshold
        res = self._image_postprocessor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )

        if self._compat_mode:
            res = [self._stable_sort_result(r) for r in res]
        # Move to CPU in bulk to avoid later syncs

        out = []

        for r in res:
            out.append({
                "boxes": r["boxes"].detach().to("cpu"),
                "scores": r["scores"].detach().to("cpu"),
                "labels": r["labels"].detach().to("cpu"),
            })
        return out

    @torch.inference_mode()
    def predict_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> List[List[dict]]:
        """
        Runs in fixed-size chunks of 64 to preserve CUDA Graph replay.
        Returns list of predictions per original image (order preserved).
        """
        if not images:
            return []

        # Reset accumulated timings for this batch
        self._t_preprocess_ms = 0.0
        self._t_predict_ms = 0.0
        self._t_postprocess_ms = 0.0

        # Convert all to PIL RGB once
        pil_all = self._to_pil_rgb(images)
        N = len(pil_all)
        results_all: List[List[dict]] = []

        # Process in chunks of FIXED_BS
        num_batches = (N + FIXED_BS - 1) // FIXED_BS  # Calculate total number of batches
        for batch_idx, start in enumerate(range(0, N, FIXED_BS)):
            end = min(start + FIXED_BS, N)
            chunk = pil_all[start:end]
            b = len(chunk)

            # Pad chunk to FIXED_BS by repeating the last image
            if b < FIXED_BS:
                chunk = chunk + [chunk[-1]] * (FIXED_BS - b)

            timer = self._create_timer()

            with timer.time_section('preprocess'):
                # Prepare static device input with stable address/shape
                device_input = self._preprocess_batch(chunk, b)  # only preprocess b real images
                # Sanity: enforce memory format + shape
                assert device_input.shape == (FIXED_BS, 3, FIXED_H, FIXED_W)
                # channels_last only enforced for CUDA
                if self._device.type == "cuda":
                    assert device_input.is_contiguous(memory_format=torch.channels_last)

            with timer.time_section('predict'):
                outputs = self._model(pixel_values=device_input)

            # target_sizes is (H, W) per *real* image
            with timer.time_section('postprocess'):
                outputs_b = self._slice_model_output(outputs, b=b, full_bs=FIXED_BS)
                target_sizes = torch.tensor([img.size[::-1] for img in chunk[:b]], dtype=torch.long)
                res_cpu = self._postprocess(outputs_b, target_sizes=target_sizes)

            timer.finalize()
            self._store_timings(timer)

            for img, res in zip(chunk[:b], res_cpu[:b]):
                w, h = img.size
                preds = []
                for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"]):
                    s = float(score.item())
                    lid = int(label_id.item()) + self._label_offset
                    lbl = self._classes_map[lid]
                    if lbl in self._black_classes:
                        continue
                    x1, y1, x2, y2 = [float(bi.item()) for bi in box]
                    l = min(w, max(0.0, x1));
                    t = min(h, max(0.0, y1))
                    r = min(w, max(0.0, x2));
                    btm = min(h, max(0.0, y2))
                    preds.append({"l": l, "t": t, "r": r, "b": btm, "label": lbl, "confidence": s})
                results_all.append(preds)

        return results_all

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        res = self.predict_batch([orig_img])
        if res:
            for pred in res[0]:
                yield pred
