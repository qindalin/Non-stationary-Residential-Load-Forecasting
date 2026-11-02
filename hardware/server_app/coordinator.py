from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from server_app.registry import ClientRegistry
from server_app.global_model_manager import GlobalModelManager
from server_app.services.aggregation_service import AggregationService
from server_app.services.dispatch_service import DispatchService
from server_app.services.metrics_service import MetricsService
from server_app.snapshot_store import SnapshotStore
from shared.models.factory import ModelConfig
from shared.protocol.serialization import decode_state_dict, encode_state_dict
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
from shared.strategy.scheduler import ClientPriority, select_clients


@dataclass
class Coordinator:
    snapshot_dir: Path
    model_config: ModelConfig
    sync_threshold: float = 0.80
    sync_budget_clients: int = 4
    server_lambda: float = 0.5
    stale_gamma: float = 1.0
    post_sync_broadcast: bool = True
    initial_state_path: str | None = None
    registry: ClientRegistry = field(default_factory=ClientRegistry)
    aggregation_service: AggregationService = field(default_factory=AggregationService)
    dispatch_service: DispatchService = field(default_factory=DispatchService)
    metrics_service: MetricsService | None = None

    def __post_init__(self) -> None:
        self.snapshot_store = SnapshotStore(root_dir=self.snapshot_dir)
        if self.metrics_service is None:
            self.metrics_service = MetricsService(root_dir=self.snapshot_dir.parent / "metrics")
        self.model_manager = GlobalModelManager(
            model_config=self.model_config,
            server_lambda=self.server_lambda,
            stale_gamma=self.stale_gamma,
            initial_state_path=self.initial_state_path,
        )
        self.pending_priorities: Dict[int, Dict[str, ClientPriority]] = {}
        self.update_buffer: Dict[int, List[Dict[str, object]]] = {}
        initial_snapshot_id = "snapshot_init"
        self.snapshot_store.save(
            snapshot_id=initial_snapshot_id,
            timestep=0,
            state_dict_b64=encode_state_dict(self.model_manager.get_state()),
            metadata={"utility": 0.0, "aggregated_clients": 0.0},
        )

    def register(self, request: RegisterRequest) -> RegisterResponse:
        self.registry.register(request.client_id, request.device)
        return RegisterResponse(
            accepted=True,
            server_time=datetime.now(timezone.utc).isoformat(),
            message="client registered",
        )

    def heartbeat(self, request: HeartbeatRequest) -> None:
        self.registry.heartbeat(request)

    def evaluate_sync_request(self, request: SyncRequest) -> SyncDecision:
        timestep_bucket = self.pending_priorities.setdefault(request.timestep, {})
        timestep_bucket[request.client_id] = ClientPriority(
            client_id=request.client_id,
            urgency=request.urgency,
            drift_score=request.drift_score,
        )
        selected = select_clients(timestep_bucket.values(), self.sync_budget_clients, self.sync_threshold)
        latest = self.snapshot_store.latest()
        accepted = request.client_id in selected
        snapshot_id = latest.snapshot_id if accepted and latest is not None else None
        return self.dispatch_service.build_decision(
            accepted=accepted,
            reason="selected" if accepted else "below threshold or outside budget",
            snapshot_id=snapshot_id,
            upload_required=accepted,
        )

    def accept_update(self, payload: ModelUpdatePayload) -> Dict[str, object]:
        decoded_payload = decode_state_dict(payload.state_dict_b64)
        update_record = {
            "client_name": payload.client_id,
            "delta_state": decoded_payload,
            "drift_score": payload.drift_score,
            "cloud_utility": payload.cloud_utility,
            "volatility": payload.volatility,
            "staleness": payload.staleness,
        }
        buffered = self.update_buffer.setdefault(payload.timestep, [])
        buffered.append(update_record)
        weights = self.aggregation_service.compute_weights(buffered)
        new_state = self.model_manager.aggregate_updates(buffered, weights)
        snapshot_id = f"snapshot_t{payload.timestep}_{len(self.snapshot_store.metadata):05d}"
        metadata = {
            "drift_score": payload.drift_score,
            "cloud_utility": payload.cloud_utility,
            "volatility": payload.volatility,
            "staleness": payload.staleness,
            "utility": payload.cloud_utility,
            "aggregated_clients": float(len(buffered)),
            "timestep": float(payload.timestep),
        }
        self.snapshot_store.save(snapshot_id, payload.timestep, encode_state_dict(new_state), metadata)
        recipients = list(self.registry.clients.keys()) if self.post_sync_broadcast else [payload.client_id]
        for client_id in recipients:
            self.dispatch_service.stage_download(client_id, snapshot_id)
        self.update_buffer[payload.timestep] = []
        return {
            "snapshot_id": snapshot_id,
            "aggregation_weights": weights,
        }

    def download_for_client(self, client_id: str) -> SyncDownloadResponse:
        snapshot_id = self.dispatch_service.pop_download(client_id)
        if snapshot_id is None:
            return SyncDownloadResponse(available=False)
        snapshot_metadata = self.snapshot_store.metadata_for(snapshot_id)
        return SyncDownloadResponse(
            available=True,
            snapshot_id=snapshot_id,
            state_dict_b64=self.snapshot_store.load_encoded(snapshot_id),
            metadata={"source": "server", "timestep": snapshot_metadata.timestep if snapshot_metadata else 0},
        )

    def fetch_snapshot(self, snapshot_id: str) -> SyncDownloadResponse:
        snapshot_metadata = self.snapshot_store.metadata_for(snapshot_id)
        if snapshot_metadata is None:
            return SyncDownloadResponse(available=False)
        return SyncDownloadResponse(
            available=True,
            snapshot_id=snapshot_id,
            state_dict_b64=self.snapshot_store.load_encoded(snapshot_id),
            metadata={"source": "server", "timestep": snapshot_metadata.timestep, **snapshot_metadata.metadata},
        )

    def report_metrics(self, report: MetricsReport) -> None:
        self.metrics_service.record(report)

    def status(self) -> Dict[str, object]:
        latest = self.snapshot_store.latest()
        return {
            "server": "online-federated-hardware",
            "registered_clients": self.registry.size(),
            "latest_snapshot_id": latest.snapshot_id if latest else None,
        }

    def metrics_summary(self) -> Dict[str, object]:
        return self.metrics_service.summary()