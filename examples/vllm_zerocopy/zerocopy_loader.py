# SPDX-License-Identifier: Apache-2.0
"""Zero-copy model loader: network -> CUDA mapped memory -> GPU tensors.

Downloads safetensors shards in parallel via hf_transfer and yields tensors
directly from CUDA mapped host memory, bypassing disk entirely.

Usage:
    vllm serve model_id --load-format hf_zerocopy
"""
import ctypes
import json
import struct
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from huggingface_hub import HfApi, hf_hub_url
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)

DTYPE_MAP = {
    "F64": (torch.float64, 8),
    "F32": (torch.float32, 4),
    "F16": (torch.float16, 2),
    "BF16": (torch.bfloat16, 2),
    "I64": (torch.int64, 8),
    "I32": (torch.int32, 4),
    "I16": (torch.int16, 2),
    "I8": (torch.int8, 1),
    "U8": (torch.uint8, 1),
}

_libcudart = None


def _get_cudart():
    global _libcudart
    if _libcudart is None:
        _libcudart = ctypes.CDLL("libcudart.so")
    return _libcudart


def _cuda_host_alloc(size: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    ret = _get_cudart().cudaHostAlloc(
        ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(0x03)
    )
    if ret != 0:
        raise RuntimeError(f"cudaHostAlloc failed with error {ret}")
    return ptr


def _cuda_host_free(ptr: ctypes.c_void_p):
    _get_cudart().cudaFreeHost(ptr)


def _get_hf_token() -> str | None:
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def _get_shard_info(
    model_name: str, revision: str | None
) -> list[tuple[str, int]]:
    """Get safetensors shard filenames and sizes from HF API (single call)."""
    api = HfApi()
    repo_info = api.model_info(model_name, revision=revision, files_metadata=True)
    all_files = {s.rfilename: s.size for s in repo_info.siblings}
    st_files = {k: v for k, v in all_files.items() if k.endswith(".safetensors")}

    if "model.safetensors.index.json" in all_files:
        index_path = api.hf_hub_download(
            model_name, "model.safetensors.index.json", revision=revision
        )
        with open(index_path) as f:
            referenced = set(json.load(f).get("weight_map", {}).values())
        if referenced:
            st_files = {k: v for k, v in st_files.items() if k in referenced}

    return sorted(st_files.items())


def _resolve_cdn_url(hub_url: str, headers: dict | None) -> str:
    """Resolve a HF Hub URL to its CDN URL by following the redirect."""
    req = urllib.request.Request(hub_url, method="HEAD")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req) as resp:
        return resp.url


def _resolve_cdn_urls_parallel(
    hub_urls: list[str], headers: dict | None
) -> list[str]:
    """Resolve all HF Hub URLs to CDN URLs in parallel."""
    cdn_urls = [None] * len(hub_urls)

    with ThreadPoolExecutor(max_workers=len(hub_urls)) as pool:
        futures = {
            pool.submit(_resolve_cdn_url, url, headers): i
            for i, url in enumerate(hub_urls)
        }
        for future in as_completed(futures):
            idx = futures[future]
            cdn_urls[idx] = future.result()

    return cdn_urls


def _parse_safetensors_header(buf_addr: int, buf_size: int) -> tuple[dict, int]:
    header_len = struct.unpack(
        "<Q", (ctypes.c_char * 8).from_address(buf_addr).raw
    )[0]
    if header_len > buf_size - 8:
        raise ValueError(
            f"Safetensors header ({header_len}B) exceeds buffer ({buf_size}B)"
        )
    header_json = (
        (ctypes.c_char * header_len).from_address(buf_addr + 8).raw.decode("utf-8")
    )
    return json.loads(header_json), 8 + header_len


class _ShardResult:
    """Holds the result of a background shard download."""

    __slots__ = ("host_ptr", "file_size", "error", "download_time", "done")

    def __init__(self):
        self.host_ptr: ctypes.c_void_p | None = None
        self.file_size: int = 0
        self.error: Exception | None = None
        self.download_time: float = 0.0
        self.done = threading.Event()


def _download_shard(
    cdn_url: str,
    file_size: int,
    max_connections: int,
    chunk_size: int,
    result: _ShardResult,
    host_ptr: ctypes.c_void_p | None = None,
):
    """Download a shard into CUDA mapped memory. Runs in a background thread.

    Uses download_into_buffer_direct with pre-resolved CDN URL,
    skipping the redirect discovery request.

    If host_ptr is provided, reuses it. Otherwise allocates new mapped memory.
    """
    try:
        import hf_transfer

        if host_ptr is None:
            host_ptr = _cuda_host_alloc(file_size)
        result.host_ptr = host_ptr
        result.file_size = file_size

        t0 = time.perf_counter()
        hf_transfer.download_into_buffer_direct(
            cdn_url,
            host_ptr.value,
            file_size,
            max_connections,
            chunk_size,
        )
        result.download_time = time.perf_counter() - t0
    except Exception as e:
        result.error = e
    finally:
        result.done.set()


def _yield_tensors_from_shard(host_ptr, file_size):
    """Parse safetensors header and yield (name, tensor) from mapped memory."""
    metadata, data_offset = _parse_safetensors_header(host_ptr.value, file_size)
    base = host_ptr.value + data_offset

    for name, info in metadata.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        if dtype_str not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        torch_dtype, _ = DTYPE_MAP[dtype_str]
        start, end = info["data_offsets"]
        tensor = torch.frombuffer(
            (ctypes.c_char * (end - start)).from_address(base + start),
            dtype=torch_dtype,
        ).reshape(info["shape"])
        yield name, tensor


class ZeroCopyModelLoader(BaseModelLoader):
    """Model loader that downloads directly to CUDA mapped memory.

    Bypasses disk entirely. Uses hf_transfer for parallel HTTP/1.1 downloads
    into CUDA mapped host memory, then yields tensors that the GPU can
    access directly via PCIe.

    Key optimizations:
    - Pre-resolved CDN URLs: all redirects resolved in parallel upfront
    - Direct download: skips redirect discovery per shard (no bytes=0-0 probe)
    - Prefetch: first shard download starts before model architecture init
    - Background: shard N+1 downloads while shard N tensors are consumed
    - TCP tuning: nodelay + connection pool reuse
    - GIL released: download runs in background, Python threads can work
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config
        self.chunk_size = extra.get("chunk_size", 50 * 1024 * 1024)
        self.max_connections = extra.get("max_connections", 16)

        # Prefetch state (populated by _start_prefetch, consumed by load_weights)
        self._shards: list[tuple[str, int]] | None = None
        self._cdn_urls: list[str] | None = None
        self._first_shard: _ShardResult | None = None

    def download_model(self, model_config: ModelConfig) -> None:
        pass

    def _start_prefetch(self, model_config: ModelConfig):
        """Resolve shard info, CDN URLs, alloc memory, and start downloading.

        Called BEFORE initialize_model() so download overlaps with model init.

        Timeline:
          [shard_info 90ms]
          [CDN resolve ~50ms] + [cudaHostAlloc ~2s] (parallel)
          [download starts] → overlaps with initialize_model()
        """
        t0 = time.perf_counter()
        token = _get_hf_token()
        headers = {"authorization": f"Bearer {token}"} if token else None

        self._shards = _get_shard_info(model_config.model, model_config.revision)
        hub_urls = [
            hf_hub_url(model_config.model, fname, revision=model_config.revision)
            for fname, _ in self._shards
        ]

        total_gb = sum(s for _, s in self._shards) / 1e9
        logger.info(
            "Zero-copy: %d shard(s), %.2f GB total. Resolving CDN URLs + alloc...",
            len(self._shards),
            total_gb,
        )

        if not self._shards:
            return

        # Start CUDA alloc for first shard in background (takes ~2s for large shards)
        alloc_result: dict = {}
        alloc_thread = threading.Thread(
            target=lambda: alloc_result.__setitem__(
                "ptr", _cuda_host_alloc(self._shards[0][1])
            ),
            daemon=True,
        )
        alloc_thread.start()

        # Resolve all CDN URLs in parallel (overlaps with alloc, ~50ms)
        self._cdn_urls = _resolve_cdn_urls_parallel(hub_urls, headers)

        dt_resolve = time.perf_counter() - t0
        logger.info("Zero-copy: CDN URLs resolved in %.2fs", dt_resolve)

        # Wait for alloc to finish
        alloc_thread.join()
        first_ptr = alloc_result["ptr"]

        dt_alloc = time.perf_counter() - t0
        logger.info("Zero-copy: first buffer allocated in %.2fs", dt_alloc)

        # Start download with pre-allocated buffer (overlaps with model init)
        self._first_shard = _ShardResult()
        t = threading.Thread(
            target=_download_shard,
            args=(
                self._cdn_urls[0],
                self._shards[0][1],
                self.max_connections,
                self.chunk_size,
                self._first_shard,
                first_ptr,
            ),
            daemon=True,
        )
        t.start()

        dt = time.perf_counter() - t0
        logger.info("Zero-copy: prefetch setup in %.2fs", dt)

    @instrument(span_name="Load weights (zerocopy)")
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        assert self._shards is not None, "_start_prefetch must be called first"

        t0 = time.perf_counter()

        def weights_iter():
            pending = self._first_shard

            for i, (filename, file_size) in enumerate(self._shards):
                # Wait for current shard
                assert pending is not None
                pending.done.wait()
                if pending.error:
                    raise pending.error

                logger.info(
                    "Downloaded %s (%.2f GB) in %.2fs (%.2f GB/s)",
                    filename,
                    pending.file_size / 1e9,
                    pending.download_time,
                    pending.file_size / pending.download_time / 1e9
                    if pending.download_time > 0
                    else 0,
                )

                # Start next shard in background
                next_pending = None
                if i + 1 < len(self._shards):
                    next_pending = _ShardResult()
                    t = threading.Thread(
                        target=_download_shard,
                        args=(
                            self._cdn_urls[i + 1],
                            self._shards[i + 1][1],
                            self.max_connections,
                            self.chunk_size,
                            next_pending,
                        ),
                        daemon=True,
                    )
                    t.start()

                # Yield tensors, then free shard memory
                try:
                    yield from _yield_tensors_from_shard(
                        pending.host_ptr, pending.file_size
                    )
                finally:
                    _cuda_host_free(pending.host_ptr)

                pending = next_pending

        loaded_weights = model.load_weights(weights_iter())
        dt = time.perf_counter() - t0
        logger.info("Zero-copy weight loading completed in %.2fs", dt)

        if model_config.quantization is None and loaded_weights is not None:
            weights_to_load = {name for name, _ in model.named_parameters()}
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                raise ValueError(
                    "Following weights were not initialized from "
                    f"checkpoint: {weights_not_loaded}"
                )

    @instrument(span_name="Load model (zerocopy)")
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        """Override load_model to start download BEFORE model architecture init.

        Standard flow:  initialize_model (1s) -> download (4s) -> load (3s)
        Our flow:       start_download -----> initialize_model (1s) -> load (~6s)
                        (download runs in background during model init)
        """
        from vllm.platforms import current_platform

        device_config = vllm_config.device_config
        load_config = vllm_config.load_config
        load_device = (
            device_config.device if load_config.device is None else load_config.device
        )
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            # Start download in background FIRST
            self._start_prefetch(model_config)

            # Then initialize model architecture (download runs concurrently)
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )

            # Now load weights (first shard may already be downloaded)
            self.load_weights(model, model_config)

            if current_platform.is_cuda():
                peak_memory = torch.cuda.max_memory_allocated()
                logger.debug_once(
                    "Peak GPU memory after loading weights: %s GiB",
                    format_gib(peak_memory),
                    scope="local",
                )

            process_weights_after_loading(model, model_config, target_device)

        return model.eval()
