from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from server_app.coordinator import Coordinator
from shared.protocol.schemas import (
    HeartbeatRequest,
    MetricsReport,
    ModelUpdatePayload,
    RegisterRequest,
    RegisterResponse,
    StatusResponse,
    SyncDecision,
    SyncDownloadResponse,
    SyncRequest,
)


def create_app(
    snapshot_dir: str | Path,
    sync_threshold: float,
    sync_budget_clients: int,
    model_config: dict,
    server_lambda: float,
    stale_gamma: float,
    post_sync_broadcast: bool,
    initial_state_path: str | None,
) -> FastAPI:
    coordinator = Coordinator(
        snapshot_dir=Path(snapshot_dir),
        model_config=model_config,
        sync_threshold=sync_threshold,
        sync_budget_clients=sync_budget_clients,
        server_lambda=server_lambda,
        stale_gamma=stale_gamma,
        post_sync_broadcast=post_sync_broadcast,
        initial_state_path=initial_state_path,
    )
    app = FastAPI(title="Online Federated Hardware Server")
    app.state.coordinator = coordinator

    @app.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        payload = app.state.coordinator.status()
        return StatusResponse(**payload)

    @app.post("/register", response_model=RegisterResponse)
    def register(request: RegisterRequest) -> RegisterResponse:
        return app.state.coordinator.register(request)

    @app.post("/heartbeat")
    def heartbeat(request: HeartbeatRequest) -> dict[str, str]:
        try:
            app.state.coordinator.heartbeat(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.post("/sync/request", response_model=SyncDecision)
    def sync_request(request: SyncRequest) -> SyncDecision:
        return app.state.coordinator.evaluate_sync_request(request)

    @app.post("/sync/upload")
    def sync_upload(payload: ModelUpdatePayload) -> dict[str, object]:
        return app.state.coordinator.accept_update(payload)

    @app.get("/sync/download/{client_id}", response_model=SyncDownloadResponse)
    def sync_download(client_id: str) -> SyncDownloadResponse:
        return app.state.coordinator.download_for_client(client_id)

    @app.get("/snapshot/{snapshot_id}", response_model=SyncDownloadResponse)
    def snapshot(snapshot_id: str) -> SyncDownloadResponse:
        return app.state.coordinator.fetch_snapshot(snapshot_id)

    @app.post("/metrics/report")
    def metrics_report(report: MetricsReport) -> dict[str, str]:
        app.state.coordinator.report_metrics(report)
        return {"status": "recorded"}

    @app.get("/metrics/summary")
    def metrics_summary() -> dict[str, object]:
        return app.state.coordinator.metrics_summary()

    return app