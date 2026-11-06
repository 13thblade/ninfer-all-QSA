"""32K-context A/B: Qwen3.8-27B-QSA dense attention vs the QSA sparse branch.

One cold ninfer-serve per arm, the same deterministic prompt and seed, greedy,
no prefix reuse, and the artifact's own MTP drafts in the `_mtp` arms. Server-side
request-log timings are the source of truth (not client wall time). On Windows the
server resolves its CUDA runtime DLLs from NINFER_BENCH_DLL_DIR; keep the exe in
its build tree, since moving it breaks DLL resolution with an empty log.

    set NINFER_BENCH_MODEL=E:\\LLMs\\Qwen3.8-27B-QSA.ninfer
    set NINFER_BENCH_DLL_DIR=F:\\ninfer-all-workspace\\ninfer-all-nvfp4\\build-merge\\apps
    python tools/bench/run_qsa27b_32k_ab.py
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = Path(os.environ.get("NINFER_BENCH_SERVER", ROOT / "build-msvc/apps/ninfer-serve.exe"))
MODEL = Path(os.environ.get("NINFER_BENCH_MODEL", r"E:\LLMs\Qwen3.8-27B-QSA.ninfer"))
DLL_DIR = os.environ.get("NINFER_BENCH_DLL_DIR", "")
PORT = int(os.environ.get("NINFER_BENCH_PORT", "8095"))
MAX_CONTEXT = int(os.environ.get("NINFER_BENCH_MAX_CONTEXT", "32768"))
OUTPUT_TOKENS = int(os.environ.get("NINFER_BENCH_OUTPUT_TOKENS", "512"))
PROMPT_CHARS = int(os.environ.get("NINFER_BENCH_PROMPT_CHARS", "152000"))
DRAFT_TOKENS = os.environ.get("NINFER_BENCH_DRAFT_TOKENS", "3")
SEED = 57004
OUT = ROOT / "benchmark_results" / f"qsa27b_32k_{datetime.now():%Y%m%d_%H%M%S}"

env = os.environ.copy()
if DLL_DIR:
    env["PATH"] = DLL_DIR + os.pathsep + env.get("PATH", "")

PARAGRAPH = (
    "A reliable GPU inference service separates admission control, prompt ingestion, decode "
    "scheduling, memory accounting, observability, and failure recovery. It measures latency "
    "and throughput independently, bounds concurrency, avoids hidden cache reuse, and records "
    "enough metadata to reproduce every result. Engineers validate correctness before tuning "
    "kernels and preserve memory headroom for transient workspaces. "
)
OPENERS = ("Measurement", "Scheduling", "Storage", "Verification", "Transport", "Recovery",
           "Admission", "Profiling", "Placement", "Accounting")


def build_prompt(chars: int) -> str:
    rng = random.Random(SEED)
    chunks: list[str] = []
    total, i = 0, 0
    while total < chars:
        piece = (f"{rng.choice(OPENERS)} note {i:05d}: " + PARAGRAPH +
                 f" Token budget {rng.randrange(1000, 99999)}; "
                 f"window {rng.randrange(64, 8192)}; lane {rng.randrange(1, 9)}.\n")
        chunks.append(piece)
        total += len(piece)
        i += 1
    return "".join(chunks)[:chars]


PROMPT = build_prompt(PROMPT_CHARS)
ARMS = {
    "dense_plain": [],
    "dense_mtp": ["--spec", "mtp", "--draft-tokens", DRAFT_TOKENS, "--ngram-draft-tokens", "0"],
    "qsa_plain": ["--qwen38-27b-qsa"],
    "qsa_mtp": ["--qwen38-27b-qsa", "--spec", "mtp", "--draft-tokens", DRAFT_TOKENS,
                "--ngram-draft-tokens", "0"],
}


def post(path: str, payload: dict, timeout: float = 1800.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def wait_ready(proc) -> None:
    deadline = time.monotonic() + 600
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup: {proc.returncode}")
        if time.monotonic() > deadline:
            raise TimeoutError("server health timeout")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return
        except Exception:
            time.sleep(0.5)


def run_arm(name: str, extra: list[str], tokenize: bool) -> dict:
    arm = OUT / name
    arm.mkdir(parents=True, exist_ok=True)
    cmd = [str(SERVER), str(MODEL), "--host", "127.0.0.1", "--port", str(PORT),
           "--max-context", str(MAX_CONTEXT), "--kv-capacity", str(MAX_CONTEXT),
           "--max-concurrency", "1", "--kv-dtype", "int8", "--greedy", "--seed", str(SEED),
           "--no-prefix-reuse", "--request-log-jsonl", str(arm / "requests.jsonl")] + extra
    (arm / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    with (arm / "stdout.log").open("w", encoding="utf-8") as out, \
         (arm / "stderr.log").open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            wait_ready(proc)
            if tokenize:
                tokens = len(post("/tokenize", {"content": PROMPT,
                                                "parse_special": False})["tokens"])
                print(f"PROMPT_TOKENS={tokens}", flush=True)
                if tokens + OUTPUT_TOKENS + 64 > MAX_CONTEXT:
                    raise RuntimeError(f"prompt {tokens} tokens leaves no decode room")
            post("/v1/completions", {"prompt": "Warm up the inference path.", "max_tokens": 8,
                                     "temperature": 0, "seed": SEED, "cache_prompt": False})
            started = time.perf_counter()
            response = post("/v1/completions", {
                "prompt": PROMPT, "max_tokens": OUTPUT_TOKENS, "temperature": 0,
                "seed": SEED, "cache_prompt": False, "ignore_eos": True})
            wall = time.perf_counter() - started
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=20)
    (arm / "response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    done = [json.loads(line) for line in (arm / "requests.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()
        and json.loads(line).get("event") == "request_done"]
    record = done[-1]
    result, timings, spec = record["result"], record["timings_seconds"], record["speculative"]
    computed, generated = result["computed_prefill_tokens"], result["completion_tokens"]
    summary = {
        "arm": name,
        "prompt_tokens": result["prompt_tokens"],
        "prefix_cache_hit_tokens": result["prefix_cache_hit_tokens"],
        "computed_prefill_tokens": computed,
        "completion_tokens": generated,
        "prefill_seconds": timings["prefill"],
        "decode_seconds": timings["decode"],
        "ttft_ms": timings["ttft"] * 1000.0,
        "wall_seconds": wall,
        "prefill_tokens_per_second": computed / timings["prefill"],
        "decode_tokens_per_second": generated / timings["decode"],
        "speculative": {k: spec.get(k) for k in
                        ("backend", "draft_window", "rounds", "drafted_tokens",
                         "accepted_tokens", "fallback_steps")},
        "text": response.get("choices", [{}])[0].get("text", ""),
    }
    (arm / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items() if k != "text"}, indent=2), encoding="utf-8")
    print(f"{name:12s} prefill {computed:6d} tok / {timings['prefill']:7.3f}s "
          f"= {summary['prefill_tokens_per_second']:8.1f} tok/s | "
          f"decode {generated:4d} tok / {timings['decode']:7.3f}s "
          f"= {summary['decode_tokens_per_second']:8.1f} tok/s | "
          f"accept {spec.get('accepted_tokens', 0)}/{spec.get('drafted_tokens', 0)}", flush=True)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "configuration.json").write_text(json.dumps({
        "server": str(SERVER), "model": str(MODEL), "max_context": MAX_CONTEXT,
        "output_tokens": OUTPUT_TOKENS, "prompt_characters": len(PROMPT),
        "seed": SEED, "draft_tokens": int(DRAFT_TOKENS), "port": PORT,
    }, indent=2), encoding="utf-8")
    (OUT / "prompt.txt").write_text(PROMPT, encoding="utf-8")
    results = {name: run_arm(name, extra, i == 0) for i, (name, extra) in enumerate(ARMS.items())}
    print("\n== greedy output equality (same prompt, same seed) ==")
    base = results["dense_plain"]["text"]
    for name, result in results.items():
        print(f"  dense_plain == {name:12s}: {result['text'] == base}")
    print(f"  qsa_plain == qsa_mtp: {results['qsa_plain']['text'] == results['qsa_mtp']['text']}")
    print("\n== paired ratios vs dense_plain ==")
    for name, result in results.items():
        print(f"  {name:12s} prefill "
              f"{result['prefill_tokens_per_second'] / results['dense_plain']['prefill_tokens_per_second']:.3f}x  "
              f"decode {result['decode_tokens_per_second'] / results['dense_plain']['decode_tokens_per_second']:.3f}x")
    (OUT / "results.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in results.items()},
        indent=2), encoding="utf-8")
    print(f"Results saved to: {OUT}")


if __name__ == "__main__":
    main()
