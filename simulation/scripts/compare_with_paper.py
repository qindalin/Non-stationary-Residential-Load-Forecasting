#!/usr/bin/env python3
"""Compare reproduction runs (cleaned code) against the paper's main-table numbers."""
import json
from pathlib import Path

PAPER = {
    "lstm_18c_repro":     dict(mae=0.3634, mse=0.7052, epe=0.0291, sync=182.7, up=268.93, down=44.90),
    "mlp_18c_repro":      dict(mae=0.3693, mse=0.6875, epe=0.0519, sync=182.7, up=295.31, down=346.59),
    "cnn_18c_repro":      dict(mae=0.3036, mse=0.5479, epe=0.0193, sync=182.6, up=58.58,  down=9.45),
    "patchtst_18c_repro": dict(mae=0.3929, mse=0.7808, epe=0.0758, sync=182.7, up=113156.96, down=15968.62),
}

root = Path("results/co_adaptation_repro")
print(f"{'run':<20} {'metric':<6} {'paper':>12} {'repro':>12} {'rel.diff':>9}")
for name, ref in PAPER.items():
    summary_file = root / name / "run_summary.json"
    if not summary_file.exists():
        print(f"{name:<20} {'--':<6} not finished yet")
        continue
    s = json.loads(summary_file.read_text())
    got = dict(
        mae=s["avg_ensemble_mae"], mse=s["avg_ensemble_mse"], epe=s["avg_ensemble_epe"],
        sync=s["avg_sync_count"], up=s["total_upload_mb"], down=s["total_download_mb"],
    )
    for k in ["mae", "mse", "epe", "sync", "up", "down"]:
        p, g = ref[k], got[k]
        rel = (g - p) / p * 100.0
        print(f"{name:<20} {k:<6} {p:>12.4f} {g:>12.4f} {rel:>+8.2f}%")
    print()
