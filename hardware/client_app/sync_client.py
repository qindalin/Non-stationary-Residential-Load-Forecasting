from __future__ import annotations

from typing import Any, Dict

import requests

from shared.protocol.schemas import (
    HeartbeatRequest,
    MetricsReport,
    ModelUpdatePayload,
    RegisterRequest,
    RegisterResponse,
    SyncDecision,
    SyncDownloadResponse,
    SyncRequest,
)


class SyncClient:
    def __init__(self, server_url: str, timeout_seconds: float = 10.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.server_url}{path}", json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def register(self, request: RegisterRequest) -> RegisterResponse:
        payload = self._post("/register", request.model_dump())
        return RegisterResponse(**payload)

    def heartbeat(self, request: HeartbeatRequest) -> None:
        self._post("/heartbeat", request.model_dump())

    def request_sync(self, request: SyncRequest) -> SyncDecision:
        payload = self._post("/sync/request", request.model_dump())
        return SyncDecision(**payload)

    def upload_update(self, request: ModelUpdatePayload) -> Dict[str, Any]:
        return self._post("/sync/upload", request.model_dump())

    def download_snapshot(self, client_id: str) -> SyncDownloadResponse:
        response = requests.get(f"{self.server_url}/sync/download/{client_id}", timeout=self.timeout_seconds)
        response.raise_for_status()
        return SyncDownloadResponse(**response.json())

    def report_metrics(self, report: MetricsReport) -> None:
        self._post("/metrics/report", report.model_dump())