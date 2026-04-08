import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device

from processor.gpu_service.tf_predictor import TFPredictor


class TableInferenceModel:
    _model_repo_folder = "ds4sd--docling-models"
    _model_path = "model_artifacts/tableformer"

    def __init__(
            self,
            enabled: bool,
            artifacts_path: Optional[Path],
            options: TableStructureOptions,
            accelerator_options: AcceleratorOptions,
    ):
        self.options = options
        self.do_cell_matching = self.options.do_cell_matching
        self.mode = self.options.mode

        self.enabled = enabled
        if self.enabled:
            if artifacts_path is None:
                artifacts_path = self.download_models() / self._model_path
            else:
                # will become the default in the future
                if (artifacts_path / self._model_repo_folder).exists():
                    artifacts_path = (
                            artifacts_path / self._model_repo_folder / self._model_path
                    )
                elif (artifacts_path / self._model_path).exists():
                    warnings.warn(
                        "The usage of artifacts_path containing directly "
                        f"{self._model_path} is deprecated. Please point "
                        "the artifacts_path to the parent containing "
                        f"the {self._model_repo_folder} folder.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    artifacts_path = artifacts_path / self._model_path

            if self.mode == TableFormerMode.ACCURATE:
                artifacts_path = artifacts_path / "accurate"
            else:
                artifacts_path = artifacts_path / "fast"

            # Third Party
            import docling_ibm_models.tableformer.common as c

            device = decide_device(accelerator_options.device)

            # Stock Docling disables MPS for TableFormer. We've verified parity
            # on NVIDIA 10-Q/10-K — MPS produces identical output and is ~1.8x faster.
            # Set TURBODOCLING_TABLE_MPS=0 to force CPU fallback.
            if device == AcceleratorDevice.MPS.value:
                import os
                if os.environ.get("TURBODOCLING_TABLE_MPS", "1") == "0":
                    device = AcceleratorDevice.CPU.value

            self.tm_config = c.read_config(f"{artifacts_path}/tm_config.json")
            self.tm_config["model"]["save_dir"] = artifacts_path
            self.tm_model_type = self.tm_config["model"]["type"]

            # Initialize the actual TF predictor
            self.tf_predictor = TFPredictor(
                self.tm_config, device, accelerator_options.num_threads
            )
            self.scale = 2.0  # Scale up table input images to ~144 dpi

    @staticmethod
    def download_models(
            local_dir: Optional[Path] = None, force: bool = False, progress: bool = False
    ) -> Path:
        return download_hf_model(
            repo_id="ds4sd/docling-models",
            revision="v2.2.0",
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def predict(self, iocr_pages: list[dict],
                table_bboxes: list[list[float]],
                table_images: list[np.ndarray],
                scale_factors: list[float]) -> List[Dict]:

        return self.tf_predictor.predict(
            iocr_pages=iocr_pages,
            table_bboxes=table_bboxes,
            table_images=table_images,
            scale_factors=scale_factors,
            eval_res_preds=None,
        )
