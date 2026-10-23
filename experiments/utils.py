"""
experiments/utils.py
=====================
Shared utilities for experiment runners:
  - Result writing (with config + git hash)
  - Config loading
  - Reproducibility helpers
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional


def get_git_hash() -> str:
    """Return the current git commit hash (short). Returns 'no-git' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "no-git"


def write_results(
    results: Dict[str, Any],
    output_dir: Path,
    name: str,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write results to a JSON file in output_dir, annotated with git hash and timestamp.

    Parameters
    ----------
    results : dict
    output_dir : Path
    name : str
        Filename stem (without .json)
    config : dict, optional
        Experiment configuration to embed for reproducibility.

    Returns
    -------
    output_path : Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "git_hash": get_git_hash(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config or {},
        "results": results,
    }

    out_path = output_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    print(f"[results] Written to {out_path}")
    return out_path


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML or JSON config file."""
    import yaml
    config_path = Path(config_path)
    with open(config_path) as f:
        if config_path.suffix in (".yml", ".yaml"):
            return yaml.safe_load(f)
        return json.load(f)


def set_global_seed(seed: int) -> None:
    """Set seeds for all relevant random number generators."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def mean_std(values: list) -> tuple:
    """Return (mean, std) for a list of floats."""
    import numpy as np
    arr = np.array([v for v in values if v is not None], dtype=np.float64)
    if len(arr) == 0:
        return None, None
    return float(arr.mean()), float(arr.std())


def format_results_table(rows: list, columns: list) -> str:
    """Format a list of dicts as a plain-text table."""
    if not rows:
        return "(empty)"
    widths = [max(len(str(col)), max(len(str(r.get(col, ""))) for r in rows)) for col in columns]
    header = " | ".join(str(c).ljust(w) for c, w in zip(columns, widths))
    sep = "-+-".join("-" * w for w in widths)
    lines = [header, sep]
    for row in rows:
        lines.append(" | ".join(str(row.get(c, "")).ljust(w) for c, w in zip(columns, widths)))
    return "\n".join(lines)
