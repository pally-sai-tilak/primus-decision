"""Measure CPU latency, throughput and memory of Primus Decision 0.1 on the official test split.

    python examples/measure_latency.py --parquet test-00000-of-00001.parquet [--threads 2] [--n-single 100]
                                       [--batch-repeats 10] [--out latency.json]

The parquet file is the official test split of LocalLLaMA/typed-decisions at the pinned revision (the gold columns are
never read; only the states and questions are needed). One way to fetch it:

    huggingface-cli download LocalLLaMA/typed-decisions all/test-00000-of-00001.parquet --repo-type dataset \
        --revision ea9306458d6e9563628369a3d1e72e362fb381d2 --local-dir typed-decisions

Method (the one behind the latency figures in EXPERIMENTS.md): resident memory after importing the runtime, load time
of both members and both featurizers, the first inference, the warm latency of one case at a time over distinct
test cases (p50 / p95), the per-case time of batches of 32 cases, the wall time of the whole split, and the peak
resident memory of the process. Run it alone on an idle machine; the figures depend on the machine and its load, so
expect a spread between runs (the recorded runs are listed in EXPERIMENTS.md). Requires pandas and pyarrow in addition
to requirements.txt. Nothing here changes the model or its answers.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rss_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return float("nan")


def peak_mb() -> float:
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parquet", required=True, help="official test split parquet file")
    ap.add_argument("--model", default=str(ROOT / "model"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--n-single", type=int, default=100, help="distinct cases timed one at a time")
    ap.add_argument("--batch-repeats", type=int, default=10, help="batches of 32 cases timed")
    ap.add_argument("--out", default=None, help="write the record as JSON here (default: print only)")
    a = ap.parse_args()
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rss_start = rss_mb()
    import numpy as np
    import pandas as pd
    import torch
    torch.set_num_threads(a.threads)
    from primus_decision.ensemble import Ensemble
    from primus_decision.predict import request_to_case
    rss_imported = rss_mb()

    df = pd.read_parquet(a.parquet, columns=["id", "workflow", "state", "questions"])
    cases = [request_to_case(json.loads(r.state), json.loads(r.questions), r.workflow) for r in df.itertuples()]
    n_dec = sum(len(c.decisions) for c in cases)
    if len(cases) < 2:
        sys.exit("the parquet file holds fewer than two cases")

    t0 = time.perf_counter(); model = Ensemble.load(a.model); t_load = time.perf_counter() - t0
    rss_loaded = rss_mb()
    t0 = time.perf_counter(); model.predict_cases([cases[0]]); t_first = time.perf_counter() - t0
    single = []
    for c in cases[1:1 + a.n_single]:
        t0 = time.perf_counter(); model.predict_cases([c]); single.append(time.perf_counter() - t0)
    single_ms = np.array(single) * 1000
    batches = []
    span = max(len(cases) - 32, 1)
    for i in range(a.batch_repeats):
        chunk = cases[(i * 32) % span:(i * 32) % span + 32]
        t0 = time.perf_counter(); model.predict_cases(chunk); batches.append((time.perf_counter() - t0) / len(chunk))
    batch_ms = np.array(batches) * 1000
    t0 = time.perf_counter(); model.predict_cases(cases); t_full = time.perf_counter() - t0

    rec = {
        "what": "Primus Decision 0.1 CPU latency, throughput and memory (examples/measure_latency.py)",
        "run_utc": started,
        "machine": {"cpu": cpu_name(), "vcpus": os.cpu_count(), "os": platform.platform(), "python": platform.python_version(),
                    "torch": torch.__version__, "torch_threads": a.threads, "load_avg_1min_at_end": os.getloadavg()[0] if hasattr(os, "getloadavg") else None},
        "input": {"parquet": Path(a.parquet).name, "cases": len(cases), "decisions": n_dec},
        "memory_mb": {"rss_at_start": round(rss_start, 1), "rss_after_import": round(rss_imported, 1), "rss_after_load": round(rss_loaded, 1), "peak_rss": round(peak_mb(), 1)},
        "timing": {"load_s": round(t_load, 3), "first_inference_one_case_ms": round(t_first * 1000, 1),
                   "warm_single_case_ms": {"n": int(len(single_ms)), "p50": round(float(np.percentile(single_ms, 50)), 1), "p90": round(float(np.percentile(single_ms, 90)), 1),
                                           "p95": round(float(np.percentile(single_ms, 95)), 1), "mean": round(float(single_ms.mean()), 1), "min": round(float(single_ms.min()), 1), "max": round(float(single_ms.max()), 1)},
                   "batch_32_per_case_ms": {"repeats": int(len(batch_ms)), "mean": round(float(batch_ms.mean()), 1), "min": round(float(batch_ms.min()), 1), "max": round(float(batch_ms.max()), 1)},
                   "full_split": {"seconds": round(t_full, 2), "ms_per_case": round(t_full * 1000 / len(cases), 2), "cases_per_s": round(len(cases) / t_full, 1), "decisions_per_s": round(n_dec / t_full, 1)}},
    }
    text = json.dumps(rec, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
