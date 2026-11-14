"""Standalone TOTAL PCIe bandwidth monitor (no engine instrumentation).

Why this exists
---------------
Instrumenting kv_stream inside the engine only counts the blocks that one
operator copies. It cannot show whether the PCIe link itself is saturated,
because expert weights, staging prefetch, weight uploads, vision tensors and
any other H2D/D2H traffic share the same link. Instead this reads NVML's
hardware PCIe throughput counters (``nvmlDeviceGetPcieThroughput``), which
report ALL traffic on the link in KB/s since the previous read. It runs in a
separate thread/process from the inference engine and needs no engine changes.

Usage
-----
Standalone sampling until interrupted:

    python tools/bench/pcie_bandwidth_monitor.py --out pcie.csv --interval 0.1

Summarize an existing CSV over request windows (``[{"name":..., "start":...,
"end":...}]``, unix seconds):

    python tools/bench/pcie_bandwidth_monitor.py --csv pcie.csv \
        --windows windows.json --report report.json

Python API for a benchmark driver::

    from pcie_bandwidth_monitor import PcieMonitor, summarize_windows, link_capacity
    mon = PcieMonitor("pcie.csv", interval=0.1)
    mon.start()
    t0 = time.time(); ...; t1 = time.time()
    mon.stop()
    report = summarize_windows("pcie.csv", [{"name": "round1", "start": t0, "end": t1}])

Notes
-----
* ``nvmlDeviceGetPcieThroughput`` returns KB/s averaged since the last call,
  so it must be polled periodically; a single read at the end is meaningless.
* Counter 0 = TX (host->device), 1 = RX (device->host). Values are integrated
  over the window to obtain transferred bytes.
* Link capacity is derived from the current PCIe generation/width. At rest the
  GPU downclocks the link, so the script reports both the current and maximum
  generation and uses the maximum for the utilisation figure.
"""
from __future__ import annotations

import argparse
import ctypes
import csv
import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

TX_BYTES = 0
RX_BYTES = 1

# Per-lane unidirectional payload bandwidth, GB/s (after 8b/10b or 128b/130b).
_LANE_GBPS = {1: 0.250, 2: 0.500, 3: 0.984, 4: 1.969, 5: 3.938, 6: 7.563}


class NvmlPcie:
    """Minimal ctypes binding to nvml.dll; no pynvml dependency."""

    def __init__(self, index: int = 0) -> None:
        self.lib = ctypes.WinDLL("nvml.dll") if hasattr(ctypes, "WinDLL") else ctypes.CDLL("libnvidia-ml.so.1")
        init = self._symbol("nvmlInit_v2", "nvmlInit")
        init.restype = ctypes.c_int
        init.argtypes = []
        self._check(init(), "nvmlInit")
        get = self._symbol("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex")
        get.restype = ctypes.c_int
        get.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        handle = ctypes.c_void_p()
        self._check(get(index, ctypes.byref(handle)), "nvmlDeviceGetHandleByIndex")
        self.handle = handle
        self.throughput_fn = self.lib.nvmlDeviceGetPcieThroughput
        self.throughput_fn.restype = ctypes.c_int
        self.throughput_fn.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                       ctypes.POINTER(ctypes.c_uint)]

    def _symbol(self, *names: str):
        for name in names:
            if hasattr(self.lib, name):
                return getattr(self.lib, name)
        raise RuntimeError(f"none of {names} exported by NVML")

    @staticmethod
    def _check(code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what} failed with NVML code {code}")

    def read(self) -> tuple[float, float]:
        """Return (TX, RX) in GB/s for this polling interval."""
        out = []
        for counter in (TX_BYTES, RX_BYTES):
            value = ctypes.c_uint()
            code = self.throughput_fn(self.handle, counter, ctypes.byref(value))
            if code != 0:
                raise RuntimeError(f"nvmlDeviceGetPcieThroughput({counter}) failed: {code}")
            out.append(value.value * 1000.0 / 1e9)  # KB/s -> GB/s
        return out[0], out[1]


def link_capacity() -> dict[str, float]:
    """Current/max PCIe generation, width, and unidirectional capacity."""
    query = "pcie.link.gen.gpucurrent,pcie.link.gen.gpumax,pcie.link.width.current"
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True, timeout=15).strip().splitlines()[0]
        gen, gen_max, width = (int(x.strip()) for x in raw.split(","))
    except Exception:
        gen, gen_max, width = 0, 0, 16
    return {"gen_current": gen, "gen_max": gen_max, "width": width,
            "capacity_gbps": _LANE_GBPS.get(gen_max, 0.0) * width}


class PcieMonitor:
    """Background sampler writing `t,tx_gbps,rx_gbps` rows to a CSV."""

    def __init__(self, path: str | Path, interval: float = 0.1, index: int = 0) -> None:
        self.path = Path(path)
        self.interval = interval
        self.index = index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            nvml = NvmlPcie(self.index)
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(["t", "tx_gbps", "rx_gbps"])
                while not self._stop.is_set():
                    t = time.time()
                    tx, rx = nvml.read()
                    writer.writerow([f"{t:.6f}", f"{tx:.6f}", f"{rx:.6f}"])
                    fh.flush()
                    self._stop.wait(max(0.0, self.interval - (time.time() - t)))
        except Exception as exc:  # sampling must never break the benchmark
            self.error = repr(exc)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_exc):
        self.stop()


def load_samples(path: str | Path) -> list[tuple[float, float, float]]:
    rows: list[tuple[float, float, float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append((float(row["t"]), float(row["tx_gbps"]), float(row["rx_gbps"])))
            except (ValueError, KeyError):
                continue
    return rows


def summarize_windows(path: str | Path,
                      windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Integrate TX/RX over each [start, end] window and compare with link capacity."""
    samples = load_samples(path)
    cap = link_capacity()
    reports = []
    for w in windows:
        start, end = float(w["start"]), float(w["end"])
        inside = [s for s in samples if start <= s[0] <= end]
        if not inside:
            reports.append({"name": w.get("name"), "samples": 0})
            continue
        duration = max(end - start, 1e-9)
        tx = [s[1] for s in inside]
        rx = [s[2] for s in inside]
        # Trapezoidal integration of the sampled rate over the window.
        def integral(values: list[float]) -> float:
            if len(values) < 2:
                return sum(values) * duration
            step = duration / (len(values) - 1)
            return (sum(values) - 0.5 * (values[0] + values[-1])) * step * 1e9

        peak = max(max(tx), max(rx))
        reports.append({
            "name": w.get("name"),
            "samples": len(inside),
            "window_seconds": round(duration, 3),
            "tx_peak_gbps": round(max(tx), 3),
            "tx_mean_gbps": round(sum(tx) / len(tx), 3),
            "rx_peak_gbps": round(max(rx), 3),
            "rx_mean_gbps": round(sum(rx) / len(rx), 3),
            "tx_total_bytes": int(integral(tx)),
            "rx_total_bytes": int(integral(rx)),
            "peak_link_utilisation_pct": round(
                100.0 * peak / cap["capacity_gbps"], 2) if cap["capacity_gbps"] else None,
        })
    return {"link": cap, "windows": reports}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", "--csv", dest="csv", default="pcie_samples.csv")
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--windows", help="JSON file with request windows")
    parser.add_argument("--report", help="where to write the summary JSON")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="0 = sample until Ctrl+C")
    args = parser.parse_args()

    if args.windows:
        windows = json.loads(Path(args.windows).read_text(encoding="utf-8"))
        result = summarize_windows(args.csv, windows)
        text = json.dumps(result, indent=2)
        print(text)
        if args.report:
            Path(args.report).write_text(text, encoding="utf-8")
        return 0

    print(f"Sampling TOTAL PCIe throughput every {args.interval}s -> {args.csv}", flush=True)
    print(f"Link: {json.dumps(link_capacity())}", flush=True)
    mon = PcieMonitor(args.csv, args.interval, args.index)
    mon.start()
    started = time.time()
    try:
        while not args.duration or time.time() - started < args.duration:
            time.sleep(0.5)
            if mon.error:
                print("sampler error:", mon.error, flush=True)
                break
            print(f"  {time.time() - started:.0f}s", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        mon.stop()
    samples = load_samples(args.csv)
    if samples:
        tx = [s[1] for s in samples]
        rx = [s[2] for s in samples]
        result = {"link": link_capacity(), "samples": len(samples),
                  "tx_peak_gbps": round(max(tx), 3), "tx_mean_gbps": round(sum(tx) / len(tx), 3),
                  "rx_peak_gbps": round(max(rx), 3), "rx_mean_gbps": round(sum(rx) / len(rx), 3)}
        print(json.dumps(result, indent=2), flush=True)
        if args.report:
            Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
