"""QSA-27B streamed-KV benchmark with EXTERNAL total-PCIe-bandwidth sampling.

Two phases, each starting exactly one server at a time (one test on the GPU):

  perf32k : 32K context, resident vs streamed(16K/32K hot pool), one request
            each. Compares prefill/decode throughput with the historical
            in-engine baseline (~1560 tok/s prefill, ~100 tok/s MTP decode).
  pcie512k: 512K context with YaRN, same session window over several dialogue
            rounds (context cache is off under QSA, so every round re-prefills
            the growing history -- exactly the PCIe pressure case). NVML's
            hardware counters sample ALL link traffic, not just kv_stream.

Usage:
  python tools/bench/run_qsa27b_kvstream_bench.py --phase perf32k
  python tools/bench/run_qsa27b_kvstream_bench.py --phase pcie512k --rounds 3
  python tools/bench/run_qsa27b_kvstream_bench.py --phase all --rounds 3

The PCIe sampler lives in pcie_bandwidth_monitor.py in this directory.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcie_bandwidth_monitor import PcieMonitor, summarize_windows  # noqa: E402

DLL_DIR = r"F:\ninfer-all-workspace\ninfer-all-nvfp4\build-merge\apps"
EXE = r"F:\ninfer-all-QSA\build-msvc\apps\ninfer-serve.exe"
MODEL = r"E:\LLMs\Qwen3.8-27B-QSA.ninfer"
OUT = Path(r"F:\temp_scripts\qsa_kvstream_bench")
OUT.mkdir(parents=True, exist_ok=True)
PORT = 8101
SEED = 57004
BASELINE = {"prefill_tok_s": 1564.4, "decode_tok_s": 97.6}

env = os.environ.copy()
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


def build_text(chars: int, seed: int) -> str:
    rng = random.Random(seed)
    chunks, total, i = [], 0, 0
    while total < chars:
        piece = (f"{rng.choice(OPENERS)} note {i:05d}: " + PARAGRAPH +
                 f" Token budget {rng.randrange(1000, 99999)}; "
                 f"window {rng.randrange(64, 8192)}; lane {rng.randrange(1, 9)}.\n")
        chunks.append(piece)
        total += len(piece)
        i += 1
    return "".join(chunks)[:chars]


def post(path: str, payload: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def vram_mib() -> int:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=30)
        return int(raw.strip().splitlines()[0])
    except Exception:
        return -1


def throughput(timings: dict) -> tuple[float, float, float]:
    """Return (prefill tok/s, decode tok/s, TTFT ms) from the serve timings block."""
    prefill = float(timings.get("prompt_per_second", timings.get("prefill_per_second", 0.0)) or 0.0)
    decode = float(timings.get("predicted_per_second", timings.get("decode_per_second", 0.0)) or 0.0)
    ttft_ms = float(timings.get("prompt_ms", timings.get("ttft_ms", 0.0)) or 0.0)
    return prefill, decode, ttft_ms


def run_arm(name: str, *, hot: int | None, max_context: int, yarn: float | None,
            prompt_chars: int, rounds: int, growth_chars: int,
            output_tokens: int, sample_pcie: bool) -> dict:
    arm = OUT / name
    arm.mkdir(parents=True, exist_ok=True)
    cmd = [EXE, MODEL, "--host", "127.0.0.1", "--port", str(PORT),
           "--max-context", str(max_context), "--kv-dtype", "int8",
           "--max-concurrency", "1", "--greedy", "--seed", str(SEED),
           "--no-prefix-reuse", "--qwen38-27b-qsa",
           "--spec", "mtp", "--draft-tokens", "3", "--ngram-draft-tokens", "0"]
    if yarn is not None:
        cmd += ["--rope-yarn-factor", str(yarn)]
    if hot is not None:
        cmd += ["--kv-resident", str(hot)]
    (arm / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")

    result: dict = {"arm": name, "hot": hot, "max_context": max_context,
                    "yarn": yarn, "rounds": [], "ready": False}
    monitor = None
    with (arm / "stdout.log").open("w", encoding="utf-8") as out, \
         (arm / "stderr.log").open("w", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, env=env,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 900
            started = time.monotonic()
            next_report = 15.0
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    result["exit"] = proc.returncode
                    break
                elapsed = time.monotonic() - started
                if elapsed >= next_report:
                    print(f"  [{name}] starting {elapsed:.0f}s ...", flush=True)
                    next_report += 15.0
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
                        if json.load(r).get("status") == "ok":
                            result["ready"] = True
                            break
                except Exception:
                    time.sleep(0.5)
            if not result["ready"]:
                return result
            time.sleep(2)
            result["vram_mib_after_load"] = vram_mib()

            if sample_pcie:
                monitor = PcieMonitor(arm / "pcie.csv", interval=0.1)
                monitor.start()

            history = ""
            windows = []
            for turn in range(rounds):
                seed = SEED + turn * 977
                history += build_text(prompt_chars if turn == 0 else growth_chars, seed)
                prompt = history + (
                    "\nUser: Summarise the recurring engineering concerns above in two sentences.\n"
                    "Assistant: ")
                tokenized = post("/tokenize", {"content": prompt, "parse_special": False})
                prompt_tokens = len(tokenized["tokens"])
                if prompt_tokens + output_tokens + 64 > max_context:
                    print(f"  [{name}] turn {turn + 1}: {prompt_tokens} tokens leaves no room, stop",
                          flush=True)
                    break
                payload = {"prompt": prompt, "max_tokens": output_tokens, "temperature": 0,
                           "seed": SEED, "cache_prompt": False, "ignore_eos": True}
                window = {"name": f"{name}_turn{turn + 1}",
                          "start": time.time()}
                response = post("/v1/completions", payload)
                window["end"] = time.time()
                windows.append(window)
                timings = response.get("timings", {})
                prefill, decode, ttft = throughput(timings)
                text = response.get("choices", [{}])[0].get("text", "")
                # Keep the dialogue going in the same session window.
                history += "\nUser: ...\nAssistant: " + text[:200] + "\n"
                row = {"turn": turn + 1, "prompt_tokens": prompt_tokens,
                       "prefill_tok_s": round(prefill, 2), "decode_tok_s": round(decode, 2),
                       "ttft_ms": round(ttft, 2), "wall_s": round(window["end"] - window["start"], 3),
                       "timings": timings, "text_len": len(text)}
                result["rounds"].append(row)
                print(f"  [{name}] turn {turn + 1}: prompt={prompt_tokens} tok  "
                      f"prefill={prefill:.1f} tok/s  decode={decode:.1f} tok/s  "
                      f"TTFT={ttft:.0f} ms  wall={row['wall_s']:.1f}s", flush=True)
                (arm / f"turn{turn + 1}.json").write_text(
                    json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8")
            if monitor is not None:
                monitor.stop()
                if monitor.error:
                    result["pcie_error"] = monitor.error
                else:
                    result["pcie"] = summarize_windows(arm / "pcie.csv", windows)
                    (arm / "pcie_report.json").write_text(
                        json.dumps(result["pcie"], indent=2), encoding="utf-8")
        finally:
            if monitor is not None:
                monitor.stop()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
    tail = (arm / "stderr.log").read_text(encoding="utf-8", errors="replace").splitlines()[-4:]
    result["log_tail"] = tail
    (arm / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["perf32k", "pcie512k", "all"], default="all")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prompt-chars", type=int, default=152000,
                        help="~30.5K tokens; the first turn's context")
    parser.add_argument("--growth-chars", type=int, default=24000,
                        help="~4.8K tokens appended per dialogue round")
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--hot", type=int, nargs="*", default=[16384, 32768])
    args = parser.parse_args()

    all_results: dict = {"baseline_32k": BASELINE, "arms": []}

    if args.phase in ("perf32k", "all"):
        # Phase A: 32K, same shape as the historical 32K benchmark, to check that
        # streaming does not materially regress prefill / MTP decode.
        for label, hot in [("resident_32k", None)] + [(f"stream{h // 1024}k_32k", h)
                                                      for h in args.hot]:
            print(f"=== perf32k / {label} ===", flush=True)
            r = run_arm(label, hot=hot, max_context=32768, yarn=None,
                        prompt_chars=args.prompt_chars, rounds=1, growth_chars=0,
                        output_tokens=args.output_tokens, sample_pcie=False)
            all_results["arms"].append(r)
            if r["rounds"]:
                row = r["rounds"][0]
                row["prefill_vs_baseline_pct"] = round(
                    100.0 * (row["prefill_tok_s"] / BASELINE["prefill_tok_s"] - 1.0), 2)
                row["decode_vs_baseline_pct"] = round(
                    100.0 * (row["decode_tok_s"] / BASELINE["decode_tok_s"] - 1.0), 2)
                print(f"  vs baseline: prefill {row['prefill_vs_baseline_pct']:+.2f}%  "
                      f"decode {row['decode_vs_baseline_pct']:+.2f}%", flush=True)

    if args.phase in ("pcie512k", "all"):
        # Phase B: 512K with YaRN; several turns in one session window, total
        # link bandwidth sampled outside the engine by NVML.
        for hot in args.hot:
            print(f"=== pcie512k / hot={hot} ===", flush=True)
            r = run_arm(f"stream{hot // 1024}k_512k", hot=hot, max_context=524288, yarn=2.0,
                        prompt_chars=args.prompt_chars, rounds=args.rounds,
                        growth_chars=args.growth_chars, output_tokens=args.output_tokens,
                        sample_pcie=True)
            all_results["arms"].append(r)
            if "pcie" in r:
                cap = r["pcie"]["link"]
                print(f"  link: gen{max(cap['gen_current'], cap['gen_max'])} x{cap['width']} "
                      f"~{cap['capacity_gbps']:.2f} GB/s one-way", flush=True)
                for w in r["pcie"]["windows"]:
                    print(f"  {w['name']}: TX peak {w.get('tx_peak_gbps')} GB/s, "
                          f"RX peak {w.get('rx_peak_gbps')} GB/s, "
                          f"link util {w.get('peak_link_utilisation_pct')}%", flush=True)

    path = OUT / f"report_{args.phase}.json"
    path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nREPORT -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
