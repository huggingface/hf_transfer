# vLLM Zero-Copy Model Loader

A vLLM model loader that downloads safetensors weights directly from HuggingFace CDN into CUDA mapped host memory, bypassing disk entirely.

## How it works

1. Resolves CDN URLs for all safetensors shards in parallel
2. Allocates CUDA mapped host memory (`cudaHostAlloc` with `cudaHostAllocMapped | cudaHostAllocPortable`)
3. Downloads shards via `hf_transfer.download_into_buffer_direct` (parallel HTTP/1.1 streams)
4. Parses safetensors headers in-place and yields `torch.frombuffer` tensors directly from mapped memory
5. GPU reads tensor data over PCIe without any `cudaMemcpy`

Key optimizations:
- **Prefetch**: first shard download starts before model architecture initialization
- **Pipelining**: shard N+1 downloads while shard N's tensors are being consumed
- **Pre-resolved CDN URLs**: skips per-shard redirect discovery
- **GIL release**: Rust download runs in background, Python threads can work concurrently

## Requirements

- `hf_transfer` built from the `feat/download-to-memory` branch
- CUDA GPU with `libcudart.so` available
- `vllm` (tested with v0.8+)

## Installation into vLLM

Copy the loader and register it:

```python
# In vllm/model_executor/model_loader/__init__.py:

# 1. Add import
from vllm.model_executor.model_loader.zerocopy_loader import ZeroCopyModelLoader

# 2. Add "hf_zerocopy" to the LoadFormat Literal type

# 3. Add to _LOAD_FORMAT_TO_MODEL_LOADER dict:
"hf_zerocopy": ZeroCopyModelLoader,
```

Then copy `zerocopy_loader.py` into `vllm/model_executor/model_loader/`.

## Usage

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --load-format hf_zerocopy --enforce-eager
```

Or programmatically:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    load_format="hf_zerocopy",
    enforce_eager=True,
)
output = llm.generate(["Hello!"], SamplingParams(max_tokens=64))
```

## Benchmarks

Measured with cold caches (no local files, OS page cache dropped):

| Model | Instance | Standard | Zerocopy | Speedup |
|-------|----------|----------|----------|---------|
| SmolLM2-1.7B | g6.xlarge (L4, 10 Gbps) | 28.8s | 15.7s | 1.83x |
| Qwen2.5-7B | g5.4xlarge (A10G, 25 Gbps) | 66.5s | 33.0s | 2.02x |

Raw download throughput: 1.17 GB/s on 10 Gbps, 1.67 GB/s on 25 Gbps.

## Configuration

Extra config can be passed via `--model-loader-extra-config`:

```bash
vllm serve model_id \
  --load-format hf_zerocopy \
  --model-loader-extra-config '{"chunk_size": 52428800, "max_connections": 16}'
```

- `chunk_size`: HTTP range request size in bytes (default: 50 MB)
- `max_connections`: parallel HTTP connections per shard (default: 16)
