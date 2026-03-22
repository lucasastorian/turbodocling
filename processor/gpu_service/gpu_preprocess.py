import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from typing import List, Dict, Optional, Union, Tuple
from PIL import Image
import contextlib


def enable_strict_determinism():
    """
    Enable strict determinism for reproducible results.
    Disables TF32 and enables deterministic CUDA algorithms.
    
    IMPORTANT: Call this BEFORE any CUDA context is created (i.e., before the 
    first tensor hits GPU). Recommended usage in CI/compatibility mode:
    
    ```python
    if os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes"):
        from fork.layout.gpu_preprocess import enable_strict_determinism
        enable_strict_determinism()
    ```
    
    Call once at process start for CI/compatibility runs.
    """
    # Disable TF32 (Ampere+ GPUs use this by default and it changes scores slightly)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Enable deterministic kernels; disable autotuner
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    # Set workspace config for deterministic algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class GPUPreprocessor(nn.Module):
    """
    Optimized V2: Uses channels_last memory format for better performance.
    """
    
    def __init__(
        self,
        size: Dict[str, int],
        do_pad: bool = False,
        pad_size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1/255.0,
        do_normalize: bool = False,  # RT-DETR typically doesn't normalize
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
        return_channels_last: bool = False,  # Set True if model uses channels_last
    ):
        super().__init__()
        self.size = size
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.device = torch.device(device)  # Ensure proper device object
        self.dtype = dtype
        self.return_channels_last = return_channels_last
        
        # Enable compatibility mode if environment variable is set
        self._compat_mode = os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes")
        
        # Pre-compute mean/std for NHWC format (only used if do_normalize=True)
        self.register_buffer('mean', torch.tensor(mean, dtype=dtype, device=device).view(1, 1, 1, 3))
        self.register_buffer('std', torch.tensor(std, dtype=dtype, device=device).view(1, 1, 1, 3))
        
        # Create resize operation
        self.resize_op = self._make_resize(size)
        
        # Pre-allocated pinned staging tensor for large batches (avoid allocator churn)
        self._staging_cache = {}  # key: (B_bucket, H, W, C, dtype) -> pinned tensor
        
        # Persistent CUDA streams and events (avoid creation overhead per call)
        if self.device.type == "cuda":
            self._stream_h2d = torch.cuda.Stream()
            self._stream_compute = torch.cuda.Stream()
            self._h2d_event = torch.cuda.Event()
            
            # Warmup: initialize CUDA context and JIT compile kernels to avoid first-call overhead
            with torch.cuda.stream(self._stream_compute):
                dummy = torch.ones(1, 224, 224, 3, device=self.device, dtype=self.dtype)
                dummy.mul_(1.0)
                dummy_nchw = dummy.permute(0, 3, 1, 2).contiguous()
                dummy_resized = F.resize(dummy_nchw, [224, 224], 
                                       interpolation=T.InterpolationMode.BILINEAR, antialias=False)
                dummy_result = dummy_resized.contiguous().permute(0, 2, 3, 1).contiguous()
            torch.cuda.synchronize()  # Ensure warmup completes
        else:
            self._stream_h2d = None
            self._stream_compute = None
            self._h2d_event = None
    
    def _make_resize(self, size_dict: Dict[str, int]):
        """Create appropriate resize operation based on size specification."""
        # V2 only supports fixed size for optimal performance
        assert "height" in size_dict and "width" in size_dict, "V2 supports fixed size only"
        
        target_h = size_dict["height"]
        target_w = size_dict["width"]
        
        def resize_exact(x):
            # x is NHWC - force contiguity to avoid F.resize slow path
            x_nchw = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW, ensure contiguous
            resized = F.resize(
                x_nchw,
                [target_h, target_w],
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=False
            )
            return resized.contiguous().permute(0, 2, 3, 1).contiguous()  # Back to NHWC, force contiguous
        
        return resize_exact
    
    def _get_staging_tensor(self, shape, dtype):
        """Get pinned staging tensor with bucket-based allocation to avoid churn."""
        B, H, W, C = shape
        # Round batch size up to next power of 2 or common size
        B_bucket = max(8, 1 << (B-1).bit_length()) if B > 0 else 8  # 8, 16, 32, 64...
        bucket_key = (B_bucket, H, W, C, dtype)
        
        if bucket_key not in self._staging_cache:
            self._staging_cache[bucket_key] = torch.empty(
                (B_bucket, H, W, C), dtype=dtype, pin_memory=True
            )
        return self._staging_cache[bucket_key][:B]  # View into larger buffer
    
    @torch.no_grad()
    def _normalize_numpy_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize arbitrary numpy input into contiguous uint8 HWC RGB."""
        if img.ndim == 2:
            arr = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                arr = np.repeat(img, 3, axis=2)
            elif img.shape[2] >= 3:
                # Drop alpha or extra channels to match RGB model input.
                arr = img[:, :, :3]
            else:
                raise ValueError(f"Unsupported channel dimension: {img.shape}")
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8, copy=False)
        return np.ascontiguousarray(arr)

    @torch.no_grad()
    def _preprocess_batch_array(self, batch_np: np.ndarray) -> Dict[str, torch.Tensor]:
        """Preprocess one already-stacked NHWC uint8 batch."""
        # Use persistent streams (created in __init__)
        stream_h2d = self._stream_h2d
        stream_compute = self._stream_compute
        h2d_event = self._h2d_event

        # Create pinned tensor using staging buffer to avoid allocator churn
        batch_tensor = torch.from_numpy(batch_np)
        if self.device.type == "cuda":
            # Get staging tensor from bucket cache (already NHWC)
            staging_tensor = self._get_staging_tensor(batch_tensor.shape, batch_tensor.dtype)

            # Copy to staging tensor
            staging_tensor.copy_(batch_tensor)
            batch_pinned = staging_tensor
        else:
            batch_pinned = batch_tensor

        if stream_h2d:
            with torch.cuda.stream(stream_h2d):
                batch_gpu = batch_pinned.to(self.device, non_blocking=True)
                # Keep in NHWC format (already contiguous from numpy)
                # Record event when H2D transfer completes
                if h2d_event:
                    h2d_event.record()
        else:
            batch_gpu = batch_pinned.to(self.device)
        
        # Make compute stream wait on H2D completion using event (more efficient than wait_stream)
        if stream_compute and h2d_event:
            stream_compute.wait_event(h2d_event)
        
        with (torch.cuda.stream(stream_compute) if stream_compute else contextlib.nullcontext()):
            # Convert to float and rescale in one step if possible
            if self.do_rescale and batch_gpu.dtype == torch.uint8:
                # Fuse conversion + rescaling: uint8 -> float32 with scaling
                batch_float = batch_gpu.to(self.dtype) * self.rescale_factor
            else:
                batch_float = batch_gpu.to(self.dtype)
                if self.do_rescale:
                    batch_float.mul_(self.rescale_factor)
            
            # Resize (internally converts to NCHW and back)
            batch_resized = self.resize_op(batch_float)
            
            # Normalize only if do_normalize=True
            if self.do_normalize:
                batch_normalized = (batch_resized - self.mean) / self.std
            else:
                batch_normalized = batch_resized
            
            # Convert to final format based on model preference
            if self.return_channels_last:
                # Keep NHWC format with channels_last memory layout for optimal model performance
                batch_final = batch_normalized.contiguous(memory_format=torch.channels_last)
            else:
                # Convert to NCHW for standard model input
                batch_final = batch_normalized.permute(0, 3, 1, 2).contiguous()
        
        # Handle padding
        if self.do_pad:
            batch_padded, pixel_mask = self._apply_padding(batch_final)
            result = {
                'pixel_values': batch_padded,
                'pixel_mask': pixel_mask
            }
        else:
            result = {
                'pixel_values': batch_final
            }
        
        # Ensure compute is done before returning (optional sync)
        if stream_compute:
            torch.cuda.current_stream().wait_stream(stream_compute)

        return result

    @torch.no_grad()
    def preprocess_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess batch with optimized memory layout.
        Handles mixed input dimensions by grouping pages with identical shapes.
        """
        if not images:
            return {'pixel_values': torch.empty(0, 3, 0, 0, device=self.device, dtype=self.dtype)}

        # Convert to numpy arrays in HWC RGB format
        np_images: List[np.ndarray] = []
        for img in images:
            if isinstance(img, Image.Image):
                np_img = np.array(img.convert('RGB'), dtype=np.uint8)
            elif isinstance(img, np.ndarray):
                np_img = self._normalize_numpy_image(img)
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            np_images.append(np_img)

        # Fast path: all pages share the same shape.
        shape_groups: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, img in enumerate(np_images):
            shape_groups.setdefault(tuple(img.shape), []).append(idx)

        if len(shape_groups) == 1:
            batch_np = np.stack(np_images, axis=0)
            return self._preprocess_batch_array(batch_np)

        # Fallback path: mixed page sizes (e.g., SEC filings with inserts/landscape pages).
        total = len(np_images)
        pixel_values: Optional[torch.Tensor] = None
        pixel_mask: Optional[torch.Tensor] = None

        for indices in shape_groups.values():
            group_np = np.stack([np_images[i] for i in indices], axis=0)
            group_result = self._preprocess_batch_array(group_np)
            group_values = group_result["pixel_values"]

            if pixel_values is None:
                pixel_values = torch.empty(
                    (total, *group_values.shape[1:]),
                    dtype=group_values.dtype,
                    device=group_values.device,
                )
                if "pixel_mask" in group_result:
                    group_mask = group_result["pixel_mask"]
                    pixel_mask = torch.empty(
                        (total, *group_mask.shape[1:]),
                        dtype=group_mask.dtype,
                        device=group_mask.device,
                    )

            idx_tensor = torch.tensor(indices, device=group_values.device, dtype=torch.long)
            pixel_values.index_copy_(0, idx_tensor, group_values)

            if pixel_mask is not None and "pixel_mask" in group_result:
                pixel_mask.index_copy_(0, idx_tensor, group_result["pixel_mask"])

        result = {"pixel_values": pixel_values}
        if pixel_mask is not None:
            result["pixel_mask"] = pixel_mask
        return result
    
    def _apply_padding(
        self,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply padding to batch (NCHW format)."""
        B, C, H, W = batch.shape
        
        if self.pad_size is not None:
            target_h = self.pad_size["height"]
            target_w = self.pad_size["width"]
        else:
            target_h = H
            target_w = W
        
        if target_h == H and target_w == W:
            pixel_mask = torch.ones((B, H, W), dtype=torch.int64, device=self.device)
            return batch, pixel_mask
        
        # Use F.pad for single-kernel efficiency: (left, right, top, bottom)
        pad_right = target_w - W
        pad_bottom = target_h - H
        padded = F.pad(batch, (0, pad_right, 0, pad_bottom), mode='constant', value=0.0)
        
        # Create pixel mask efficiently using F.pad
        mask_template = torch.ones((B, H, W), dtype=torch.int64, device=self.device)
        pixel_mask = F.pad(mask_template, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        
        return padded, pixel_mask
