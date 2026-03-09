"""Measure raw download speed for Qwen2.5-7B on 25 Gbps instance."""
import subprocess

code = '''
import time, ctypes, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_url, get_token
import hf_transfer

MODEL = "Qwen/Qwen2.5-7B-Instruct"

token = get_token()
headers = {"authorization": f"Bearer {token}"} if token else None

# Get shard info
api = HfApi()
repo_info = api.model_info(MODEL, revision=None, files_metadata=True)
all_files = {s.rfilename: s.size for s in repo_info.siblings}
st_files = {k: v for k, v in all_files.items() if k.endswith(".safetensors")}

import json
if "model.safetensors.index.json" in all_files:
    idx_path = api.hf_hub_download(MODEL, "model.safetensors.index.json")
    with open(idx_path) as f:
        referenced = set(json.load(f).get("weight_map", {}).values())
    if referenced:
        st_files = {k: v for k, v in st_files.items() if k in referenced}

shards = sorted(st_files.items())
total_size = sum(s for _, s in shards)
print(f"Model: {MODEL}")
print(f"Shards: {len(shards)}")
for name, size in shards:
    print(f"  {name}: {size/1e9:.2f} GB")
print(f"Total: {total_size/1e9:.2f} GB")

# Resolve CDN URLs
hub_urls = [hf_hub_url(MODEL, fname) for fname, _ in shards]
def resolve_cdn(url):
    req = urllib.request.Request(url, method="HEAD")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req) as resp:
        return resp.url

cdn_urls = [None] * len(hub_urls)
with ThreadPoolExecutor(max_workers=len(hub_urls)) as pool:
    futures = {pool.submit(resolve_cdn, url): i for i, url in enumerate(hub_urls)}
    for future in as_completed(futures):
        cdn_urls[futures[future]] = future.result()

# CUDA alloc + download each shard
libcudart = ctypes.CDLL("libcudart.so")
def cuda_host_alloc(size):
    ptr = ctypes.c_void_p()
    ret = libcudart.cudaHostAlloc(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_uint(0x03))
    assert ret == 0, f"cudaHostAlloc failed: {ret}"
    return ptr

t0_total = time.perf_counter()
for i, (fname, fsize) in enumerate(shards):
    ptr = cuda_host_alloc(fsize)
    t0 = time.perf_counter()
    hf_transfer.download_into_buffer_direct(cdn_urls[i], ptr.value, fsize, 16, 50*1024*1024)
    dt = time.perf_counter() - t0
    speed = fsize / dt / 1e9
    print(f"  Shard {i}: {fname} ({fsize/1e9:.2f} GB) in {dt:.2f}s = {speed:.2f} GB/s")
    libcudart.cudaFreeHost(ptr)

dt_total = time.perf_counter() - t0_total
total_speed = total_size / dt_total / 1e9
print(f"Total download: {total_size/1e9:.2f} GB in {dt_total:.2f}s = {total_speed:.2f} GB/s")
'''

proc = subprocess.run(["python3", "-c", code], capture_output=True, text=True, timeout=300)
print(proc.stdout.strip())
if proc.returncode != 0:
    for line in proc.stderr.strip().split("\n")[-5:]:
        print(f"  ERR: {line}")
