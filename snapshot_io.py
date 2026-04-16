"""Helper methods for saving/loading dataset snapshots (CSV)

AI tools (Cursor/LLM assistance) were used to 
draft/refactor the utility code in this file (snapshot_io.py).
The final version has been reviewed, tested and integrated by the authors.

"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

def load_snapshot(path: str | Path, *, delete_after: bool = False) -> pd.DataFrame:
    """Load a CSV snapshot (expects a 'time' column) and optionally delete it."""
    snapshot_path = Path(path)
    df = pd.read_csv(snapshot_path, parse_dates=["time"])
    print(f"Loaded snapshot: {snapshot_path}")
    if delete_after:
        snapshot_path.unlink(missing_ok=True)
        print(f"Deleted snapshot: {snapshot_path}")
    return df


def save_snapshot(df: pd.DataFrame, *, snapshots_dir: str | Path, prefix: str = "mt5_snapshot_") -> Path:
    """Save a DataFrame to a timestamped CSV snapshot and return the path."""
    out_dir = Path(snapshots_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_path = out_dir / f"{prefix}{ts}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved snapshot: {out_path}")
    return out_path
