import subprocess, time, os

MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "What is the capital of France?"
N_ITER = 3
SEP = "=" * 60


def clear_caches():
    subprocess.run(
        ["rm", "-rf", os.path.expanduser("~/.cache/huggingface/hub")],
        capture_output=True,
    )
    subprocess.run(
        ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
        capture_output=True,
    )


def bench_ttft(load_format, label):
    times = []
    load_times = []
    for i in range(N_ITER):
        clear_caches()
        time.sleep(2)

        code = (
            "import torch, time\n"
            "from vllm import LLM, SamplingParams\n"
            "t_start = time.perf_counter()\n"
            f'llm = LLM(model="{MODEL}", load_format="{load_format}", '
            "enforce_eager=True, gpu_memory_utilization=0.9)\n"
            "t_loaded = time.perf_counter()\n"
            f'output = llm.generate(["{PROMPT}"], SamplingParams(max_tokens=1))\n'
            "t_first = time.perf_counter()\n"
            'print(f"TTFT_LOAD={t_loaded - t_start:.3f}")\n'
            'print(f"TTFT_GEN={t_first - t_loaded:.3f}")\n'
            'print(f"TTFT_TOTAL={t_first - t_start:.3f}")\n'
        )

        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=600,
        )

        load_time = gen_time = total = None
        for line in proc.stdout.split("\n"):
            if line.startswith("TTFT_LOAD="):
                load_time = float(line.split("=")[1])
            elif line.startswith("TTFT_GEN="):
                gen_time = float(line.split("=")[1])
            elif line.startswith("TTFT_TOTAL="):
                total = float(line.split("=")[1])

        if total is not None:
            times.append(total)
            load_times.append(load_time)
            print(
                f"  [{label}] iter {i+1}/{N_ITER}: "
                f"load={load_time:.2f}s gen={gen_time:.2f}s total={total:.2f}s"
            )
        else:
            print(f"  [{label}] iter {i+1}/{N_ITER}: FAILED (exit={proc.returncode})")
            for line in proc.stderr.strip().split("\n")[-8:]:
                print(f"    {line}")

    if times:
        times.sort()
        load_times.sort()
        median = times[len(times) // 2]
        load_median = load_times[len(load_times) // 2]
        print(
            f"\n  [{label}] Results: median={median:.2f}s "
            f"(load={load_median:.2f}s) "
            f"min={min(times):.2f}s max={max(times):.2f}s "
            f"({len(times)}/{N_ITER} ok)\n"
        )
    return times


print(SEP)
print(f"vLLM TTFT Benchmark - {MODEL}")
print(f"Instance: g5.4xlarge (A10G 24GB, 25 Gbps)")
print(SEP)

print(f"\n--- Standard vLLM (load_format=auto) ---")
std_times = bench_ttft("auto", "standard")

print(f"\n--- Zero-copy vLLM (load_format=hf_zerocopy) ---")
zc_times = bench_ttft("hf_zerocopy", "zerocopy")

if std_times and zc_times:
    std_med = sorted(std_times)[len(std_times) // 2]
    zc_med = sorted(zc_times)[len(zc_times) // 2]
    print(f"\n{SEP}")
    print("SUMMARY:")
    print(f"  Model: {MODEL}")
    print(f"  Standard median TTFT: {std_med:.2f}s")
    print(f"  Zerocopy median TTFT: {zc_med:.2f}s")
    print(f"  Speedup: {std_med/zc_med:.2f}x")
    print(f"  Saved: {std_med - zc_med:.2f}s")
    print(SEP)
