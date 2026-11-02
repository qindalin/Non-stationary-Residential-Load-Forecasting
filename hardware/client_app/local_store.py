from __future__ import annotations

import json
from pathlib import Path

from shared.common.config import ensure_dir

from client_app.runtime_state import RuntimeState


class LocalStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = ensure_dir(root_dir)
        self.state_path = self.root_dir / "runtime_state.json"

    def load_state(self, client_id: str) -> RuntimeState:
        if not self.state_path.exists():
            return RuntimeState(client_id=client_id)
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        return RuntimeState.from_dict(payload)

    def save_state(self, state: RuntimeState) -> None:
        self.state_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")