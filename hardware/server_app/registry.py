from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

from shared.protocol.schemas import DeviceInfo, HeartbeatRequest


@dataclass
class ClientRecord:
    client_id: str
    device: DeviceInfo
    registered_at: str
    last_seen_at: str
    last_timestep: int = 0
    latest_mae: Optional[float] = None
    latest_mse: Optional[float] = None


@dataclass
class ClientRegistry:
    clients: Dict[str, ClientRecord] = field(default_factory=dict)

    def register(self, client_id: str, device: DeviceInfo) -> ClientRecord:
        now = datetime.now(timezone.utc).isoformat()
        record = ClientRecord(
            client_id=client_id,
            device=device,
            registered_at=now,
            last_seen_at=now,
        )
        self.clients[client_id] = record
        return record

    def heartbeat(self, request: HeartbeatRequest) -> ClientRecord:
        if request.client_id not in self.clients:
            raise KeyError(f"Unknown client: {request.client_id}")
        record = self.clients[request.client_id]
        record.last_seen_at = datetime.now(timezone.utc).isoformat()
        record.last_timestep = request.timestep
        record.latest_mae = request.latest_mae
        record.latest_mse = request.latest_mse
        return record

    def get(self, client_id: str) -> Optional[ClientRecord]:
        return self.clients.get(client_id)

    def size(self) -> int:
        return len(self.clients)