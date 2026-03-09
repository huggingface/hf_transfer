"""Time to First Token benchmark: standard vs zero-copy pipeline. 10 iterations each."""
import time, os, ctypes, json, struct, shutil, gc
import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import hf_transfer

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B"
URL = "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/resolve/main/model.safetensors"
CHUNK = 10 * 1024 * 1024
CONNS = 16
TOKEN = os.environ.get("HF_TOKEN", "")
HEADERS = {"authorization": f"Bearer {TOKEN}"} if TOKEN else None
PROMPT = "The meaning of life is"
ITERS = 10

DTYPE_MAP = {
    "F32": (torch.float32, 4), "F16": (torch.float16, 2), "BF16": (torch.bfloat16, 2),
    "I64": (torch.int64, 8), "I32": (torch.int32, 4), "I16": (torch.int16, 2),
    "I8": (torch.int8, 1), "U8": (torch.uint8, 1),
}

def get_file_size():
    import urllib.request
    req = urllib.request.Request(URL, method="HEAD")
    if TOKEN:
        req.add_header("Authorization", f"Bearer {TOKEN}")
    return int(urllib.request.urlopen(req).headers["Content-Length"])

FSIZE = None
TOKENIZER = None

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def ttft_normal(cold=True):
    global TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = TOKENIZER(PROMPT, return_tensors="pt")

    if cold:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

    cleanup()
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_new_tokens=1,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    torch.cuda.synchronize()
    t_total = time.time() - t0

    del model
    cleanup()
    return t_total


def ttft_zerocopy():
    global TOKENIZER, FSIZE
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = TOKENIZER(PROMPT, return_tensors="pt")
    if FSIZE is None:
        FSIZE = get_file_size()

    libcudart = ctypes.CDLL("libcudart.so")
    host_ptr = ctypes.c_void_p()
    ret = libcudart.cudaHostAlloc(ctypes.byref(host_ptr), ctypes.c_size_t(FSIZE), ctypes.c_uint(0x03))
    assert ret == 0

    cleanup()
    t0 = time.time()

    # Download into mapped memory
    hf_transfer.download_into_buffer(URL, host_ptr.value, FSIZE, CONNS, CHUNK, headers=HEADERS)

    # Parse safetensors + create GPU tensors
    header_len = struct.unpack("<Q", (ctypes.c_char * 8).from_address(host_ptr.value).raw)[0]
    header_json = (ctypes.c_char * header_len).from_address(host_ptr.value + 8).raw.decode("utf-8")
    metadata = json.loads(header_json)
    data_offset = 8 + header_len

    state_dict = {}
    for name, info in metadata.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        torch_dtype, _ = DTYPE_MAP[info["dtype"]]
        cpu_tensor = torch.frombuffer(
            (ctypes.c_char * (end - start)).from_address(host_ptr.value + data_offset + start),
            dtype=torch_dtype
        ).reshape(info["shape"])
        state_dict[name] = cpu_tensor.cuda(non_blocking=True)
    torch.cuda.synchronize()

    # Load model on cuda (to materialize non-persistent buffers like rotary embeddings)
    config = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("cuda"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)
    model.load_state_dict(state_dict, assign=True, strict=False)
    model.tie_weights()
    model.eval()
    del state_dict

    # Free mapped memory (data is now in GPU memory via state_dict assign)
    libcudart.cudaFreeHost(host_ptr)

    # Generate
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"].cuda(),
            attention_mask=inputs["attention_mask"].cuda(),
            max_new_tokens=1,
            pad_token_id=TOKENIZER.eos_token_id,
        )
    torch.cuda.synchronize()
    t_total = time.time() - t0

    del model
    cleanup()
    return t_total


if __name__ == "__main__":
    print(f"=== Time to First Token ({ITERS} iterations) ===")
    print(f"Model: {MODEL_ID} | GPU: {torch.cuda.get_device_name(0)}")
    print(f"Prompt: '{PROMPT}'")
    print()

    print(f"1. NORMAL (cold start, includes download)")
    normal_times = []
    for i in range(ITERS):
        t = ttft_normal(cold=True)
        normal_times.append(t)
        print(f"   [{i+1:2d}/{ITERS}] {t:.2f}s")
    print()

    print(f"2. ZERO-COPY (mapped memory, no disk)")
    zc_times = []
    for i in range(ITERS):
        t = ttft_zerocopy()
        zc_times.append(t)
        print(f"   [{i+1:2d}/{ITERS}] {t:.2f}s")
    print()

    import statistics
    def s(times):
        return statistics.mean(times), statistics.median(times), min(times), max(times), statistics.stdev(times)

    mn, medn, minn, maxn, sdn = s(normal_times)
    mz, medz, minz, maxz, sdz = s(zc_times)

    print(f"{'='*50}")
    print(f"{'':20s} {'Normal':>10s}  {'Zero-copy':>10s}")
    print(f"{'Mean':20s} {mn:>9.2f}s  {mz:>9.2f}s")
    print(f"{'Median':20s} {medn:>9.2f}s  {medz:>9.2f}s")
    print(f"{'Min':20s} {minn:>9.2f}s  {minz:>9.2f}s")
    print(f"{'Max':20s} {maxn:>9.2f}s  {maxz:>9.2f}s")
    print(f"{'Stdev':20s} {sdn:>9.2f}s  {sdz:>9.2f}s")
    print()
    print(f"Speedup (mean):   {mn/mz:.1f}x")
    print(f"Speedup (median): {medn/medz:.1f}x")
