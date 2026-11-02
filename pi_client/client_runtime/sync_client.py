from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class SyncClient:
    def __init__(self, server_url: str, timeout_seconds: float = 10.0) -> None:
        self.server_url = server_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = requests.post(f"{self.server_url}{path}", json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def register(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/register", payload)

    def heartbeat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/heartbeat", payload)

    def request_sync(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/sync/request", payload)

    def upload_update(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/sync/upload", payload)

    def download_staged(self, client_id: str) -> Dict[str, Any]:
        response = requests.get(f"{self.server_url}/sync/download/{client_id}", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def fetch_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        response = requests.get(f"{self.server_url}/snapshot/{snapshot_id}", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def report_metrics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("/metrics/report", payload)

    def status(self) -> Dict[str, Any]:
        response = requests.get(f"{self.server_url}/status", timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()