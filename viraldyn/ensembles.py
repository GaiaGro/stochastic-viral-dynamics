"""Run one seeded realisation per seed across a process pool and record provenance."""

import json
import multiprocessing as mp
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import numba
import numpy as np


def run_seeds(fn, seeds, processes=None):
    """Map `fn(seed)` over `seeds`. `fn` must be a module-level function or a
    functools.partial of one, so that it can be sent to the workers. Numba code is
    compiled in the parent before forking, so call `fn` once first if it is cold."""
    seeds = list(seeds)
    processes = processes or os.cpu_count()
    if processes == 1 or len(seeds) < 2 * processes:
        return [fn(s) for s in seeds]
    with mp.get_context("fork" if hasattr(os, "fork") else "spawn").Pool(processes) as pool:
        return pool.map(fn, seeds, chunksize=max(1, len(seeds) // (8 * processes)))


def provenance(started, **extra):
    """Metadata written next to each results CSV."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "viraldyn", "scripts"], text=True).strip())
    except Exception:
        commit, dirty = None, None
    return {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - started, 1),
        "git_commit": commit,
        "code_dirty": dirty,
        "numba": numba.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        **extra,
    }


def write_meta(path, meta):
    with open(path, "w") as f:
        json.dump(meta, f, indent=1)
