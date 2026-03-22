import logging
import os
from pathlib import Path
from typing import Optional

_log = logging.getLogger(__name__)


def download_hf_model(
    repo_id: str,
    local_dir: Optional[Path] = None,
    force: bool = False,
    progress: bool = False,
    revision: Optional[str] = None,
) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    if not progress:
        disable_progress_bars()

    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    cache_dir = os.environ.get("HF_HUB_CACHE")

    kwargs = dict(
        repo_id=repo_id,
        force_download=force,
        revision=revision,
    )
    if offline and cache_dir:
        kwargs["cache_dir"] = cache_dir
        kwargs["local_files_only"] = True
    elif local_dir is not None:
        kwargs["local_dir"] = local_dir

    download_path = snapshot_download(**kwargs)
    return Path(download_path)


class HuggingFaceModelDownloadMixin:
    @staticmethod
    def download_models(
        repo_id: str,
        local_dir: Optional[Path] = None,
        force: bool = False,
        progress: bool = False,
    ) -> Path:
        return download_hf_model(
            repo_id=repo_id, local_dir=local_dir, force=force, progress=progress
        )
