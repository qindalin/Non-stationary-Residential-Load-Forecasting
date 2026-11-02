from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from shared.protocol.schemas import SyncDecision


@dataclass
class DispatchService:
    pending_downloads: Dict[str, str] = field(default_factory=dict)

    def stage_download(self, client_id: str, snapshot_id: str) -> None:
        self.pending_downloads[client_id] = snapshot_id

    def pop_download(self, client_id: str) -> Optional[str]:
        return self.pending_downloads.pop(client_id, None)

    def build_decision(self, accepted: bool, reason: str, snapshot_id: str | None, upload_required: bool) -> SyncDecision:
        return SyncDecision(
            accepted=accepted,
            reason=reason,
            snapshot_id=snapshot_id,
            upload_required=upload_required,
            download_available=snapshot_id is not None,
        )