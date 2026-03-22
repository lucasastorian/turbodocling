import logging
import warnings
import numpy as np
from PIL import Image
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel
from docling_core.types.doc import DocItemLabel

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2, LayoutModelConfig
from docling.datamodel.pipeline_options import LayoutOptions
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device

from processor.gpu_service.layout_predictor import LayoutPredictor
from docling.datamodel.base_models import Cluster, Page

_log = logging.getLogger(__name__)


class LayoutPrediction(BaseModel):
    clusters: List[Cluster] = []


class LayoutModel:
    TEXT_ELEM_LABELS = [
        DocItemLabel.TEXT,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.CODE,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.FORMULA,
    ]
    PAGE_HEADER_LABELS = [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]

    TABLE_LABELS = [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
    FIGURE_LABEL = DocItemLabel.PICTURE
    FORMULA_LABEL = DocItemLabel.FORMULA
    CONTAINER_LABELS = [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: LayoutOptions,
    ):
        self.options = options

        device = decide_device(accelerator_options.device)
        layout_model_config = options.model_spec
        model_repo_folder = layout_model_config.model_repo_folder
        model_path = layout_model_config.model_path

        if artifacts_path is None:
            artifacts_path = (
                self.download_models(layout_model_config=layout_model_config)
                / model_path
            )
        else:
            if (artifacts_path / model_repo_folder).exists():
                artifacts_path = artifacts_path / model_repo_folder / model_path
            elif (artifacts_path / model_path).exists():
                warnings.warn(
                    "The usage of artifacts_path containing directly "
                    f"{model_path} is deprecated. Please point "
                    "the artifacts_path to the parent containing "
                    f"the {model_repo_folder} folder.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                artifacts_path = artifacts_path / model_path

        self.layout_predictor = LayoutPredictor(
            artifact_path=str(artifacts_path),
            device=device,
            num_threads=accelerator_options.num_threads,
            use_gpu_preprocess=True,
        )

    @staticmethod
    def download_models(
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
        layout_model_config: LayoutModelConfig = DOCLING_LAYOUT_V2,
    ) -> Path:
        return download_hf_model(
            repo_id=layout_model_config.repo_id,
            revision=layout_model_config.revision,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def __call__(self, page_batch: Iterable[Page]) -> List[List[dict]]:
        pages = list(page_batch)

        valid_pages = []
        valid_page_images: List[Union[Image.Image, np.ndarray]] = []

        for page in pages:

            assert page.size is not None
            page_image = page.get_image(scale=1.0)
            assert page_image is not None

            valid_pages.append(page)
            valid_page_images.append(page_image)

        batch_predictions = []
        if valid_page_images:
            batch_predictions = self.layout_predictor.predict_batch(
                valid_page_images
            )

        return batch_predictions
