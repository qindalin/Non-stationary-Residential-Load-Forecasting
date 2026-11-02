from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
from pathlib import Path
from typing import Dict, List

from shared.protocol.schemas import MetricsReport


@dataclass
class MetricsService:
    root_dir: Path
    reports: Dict[str, List[MetricsReport]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.reports_jsonl_path = self.root_dir / "metrics_reports.jsonl"
        self.reports_csv_path = self.root_dir / "metrics_reports.csv"
        self.summary_json_path = self.root_dir / "metrics_summary.json"
        self.summary_csv_path = self.root_dir / "metrics_summary.csv"
        self._csv_header = [
            "client_id",
            "timestep",
            "mae",
            "mse",
            "abs_error_sum",
            "sq_error_sum",
            "sample_count",
            "upload_bytes",
            "download_bytes",
            "inference_latency_ms",
            "metadata",
        ]

        if self.reports_jsonl_path.exists():
            self._load_existing_reports()
        if not self.reports_csv_path.exists():
            with self.reports_csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._csv_header)
                writer.writeheader()

    def record(self, report: MetricsReport) -> None:
        self.reports.setdefault(report.client_id, []).append(report)
        self._append_jsonl(report)
        self._append_csv(report)
        self._write_summary_files()

    def latest(self, client_id: str) -> MetricsReport | None:
        payload = self.reports.get(client_id, [])
        if not payload:
            return None
        return payload[-1]

    def summary(self) -> Dict[str, object]:
        clients: Dict[str, Dict[str, float | int | None]] = {}
        for client_id, payload in self.reports.items():
            mae_values = [entry.mae for entry in payload if entry.mae is not None]
            mse_values = [entry.mse for entry in payload if entry.mse is not None]
            abs_error_total = sum(entry.abs_error_sum or 0.0 for entry in payload)
            sq_error_total = sum(entry.sq_error_sum or 0.0 for entry in payload)
            sample_count_total = sum(entry.sample_count for entry in payload)
            edge_mae_values = [entry.metadata.get("edge_mae") for entry in payload if entry.metadata.get("edge_mae") is not None]
            edge_mse_values = [entry.metadata.get("edge_mse") for entry in payload if entry.metadata.get("edge_mse") is not None]
            edge_abs_error_total = sum(float(entry.metadata.get("edge_abs_error_sum", 0.0)) for entry in payload)
            edge_sq_error_total = sum(float(entry.metadata.get("edge_sq_error_sum", 0.0)) for entry in payload)
            cloud_mae_values = [entry.metadata.get("cloud_mae") for entry in payload if entry.metadata.get("cloud_mae") is not None]
            cloud_mse_values = [entry.metadata.get("cloud_mse") for entry in payload if entry.metadata.get("cloud_mse") is not None]
            cloud_abs_error_total = sum(float(entry.metadata.get("cloud_abs_error_sum", 0.0)) for entry in payload)
            cloud_sq_error_total = sum(float(entry.metadata.get("cloud_sq_error_sum", 0.0)) for entry in payload)
            latest = payload[-1]
            clients[client_id] = {
                "reports": len(payload),
                "latest_timestep": latest.timestep,
                "latest_mae": latest.mae,
                "latest_mse": latest.mse,
                "avg_mae": sum(mae_values) / len(mae_values) if mae_values else None,
                "avg_mse": sum(mse_values) / len(mse_values) if mse_values else None,
                "exact_mae": abs_error_total / sample_count_total if sample_count_total else None,
                "exact_mse": sq_error_total / sample_count_total if sample_count_total else None,
                "avg_edge_mae": sum(edge_mae_values) / len(edge_mae_values) if edge_mae_values else None,
                "avg_edge_mse": sum(edge_mse_values) / len(edge_mse_values) if edge_mse_values else None,
                "exact_edge_mae": edge_abs_error_total / sample_count_total if sample_count_total else None,
                "exact_edge_mse": edge_sq_error_total / sample_count_total if sample_count_total else None,
                "avg_cloud_mae": sum(cloud_mae_values) / len(cloud_mae_values) if cloud_mae_values else None,
                "avg_cloud_mse": sum(cloud_mse_values) / len(cloud_mse_values) if cloud_mse_values else None,
                "exact_cloud_mae": cloud_abs_error_total / sample_count_total if sample_count_total else None,
                "exact_cloud_mse": cloud_sq_error_total / sample_count_total if sample_count_total else None,
                "abs_error_sum": abs_error_total,
                "sq_error_sum": sq_error_total,
                "sample_count": sample_count_total,
                "total_upload_bytes": latest.upload_bytes,
                "total_download_bytes": latest.download_bytes,
            }

        all_mae = [entry.mae for payload in self.reports.values() for entry in payload if entry.mae is not None]
        all_mse = [entry.mse for payload in self.reports.values() for entry in payload if entry.mse is not None]
        exact_maes = [payload["exact_mae"] for payload in clients.values() if payload["exact_mae"] is not None]
        exact_mses = [payload["exact_mse"] for payload in clients.values() if payload["exact_mse"] is not None]
        exact_edge_maes = [payload["exact_edge_mae"] for payload in clients.values() if payload["exact_edge_mae"] is not None]
        exact_edge_mses = [payload["exact_edge_mse"] for payload in clients.values() if payload["exact_edge_mse"] is not None]
        exact_cloud_maes = [payload["exact_cloud_mae"] for payload in clients.values() if payload["exact_cloud_mae"] is not None]
        exact_cloud_mses = [payload["exact_cloud_mse"] for payload in clients.values() if payload["exact_cloud_mse"] is not None]
        return {
            "clients": clients,
            "global": {
                "client_count": len(clients),
                "report_count": sum(len(payload) for payload in self.reports.values()),
                "avg_mae": sum(all_mae) / len(all_mae) if all_mae else None,
                "avg_mse": sum(all_mse) / len(all_mse) if all_mse else None,
                "avg_exact_mae": sum(exact_maes) / len(exact_maes) if exact_maes else None,
                "avg_exact_mse": sum(exact_mses) / len(exact_mses) if exact_mses else None,
                "avg_exact_edge_mae": sum(exact_edge_maes) / len(exact_edge_maes) if exact_edge_maes else None,
                "avg_exact_edge_mse": sum(exact_edge_mses) / len(exact_edge_mses) if exact_edge_mses else None,
                "avg_exact_cloud_mae": sum(exact_cloud_maes) / len(exact_cloud_maes) if exact_cloud_maes else None,
                "avg_exact_cloud_mse": sum(exact_cloud_mses) / len(exact_cloud_mses) if exact_cloud_mses else None,
            },
        }

    def _load_existing_reports(self) -> None:
        with self.reports_jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                report = MetricsReport(**payload)
                self.reports.setdefault(report.client_id, []).append(report)

    def _append_jsonl(self, report: MetricsReport) -> None:
        with self.reports_jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report.model_dump(), ensure_ascii=True) + "\n")

    def _append_csv(self, report: MetricsReport) -> None:
        with self.reports_csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_header)
            writer.writerow(
                {
                    "client_id": report.client_id,
                    "timestep": report.timestep,
                    "mae": report.mae,
                    "mse": report.mse,
                    "abs_error_sum": report.abs_error_sum,
                    "sq_error_sum": report.sq_error_sum,
                    "sample_count": report.sample_count,
                    "upload_bytes": report.upload_bytes,
                    "download_bytes": report.download_bytes,
                    "inference_latency_ms": report.inference_latency_ms,
                    "metadata": json.dumps(report.metadata, ensure_ascii=True, sort_keys=True),
                }
            )

    def _write_summary_files(self) -> None:
        summary = self.summary()
        self.summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        with self.summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "client_id",
                    "reports",
                    "latest_timestep",
                    "latest_mae",
                    "latest_mse",
                    "avg_mae",
                    "avg_mse",
                    "exact_mae",
                    "exact_mse",
                    "avg_edge_mae",
                    "avg_edge_mse",
                    "exact_edge_mae",
                    "exact_edge_mse",
                    "avg_cloud_mae",
                    "avg_cloud_mse",
                    "exact_cloud_mae",
                    "exact_cloud_mse",
                    "abs_error_sum",
                    "sq_error_sum",
                    "sample_count",
                    "total_upload_bytes",
                    "total_download_bytes",
                ],
            )
            writer.writeheader()
            for client_id, payload in summary["clients"].items():
                writer.writerow({"client_id": client_id, **payload})