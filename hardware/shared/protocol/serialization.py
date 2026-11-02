from __future__ import annotations

import base64
import io
from typing import Dict


def encode_state_dict(state_dict: Dict[str, object]) -> str:
    import torch

    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_state_dict(state_dict_b64: str) -> Dict[str, object]:
    import torch

    raw = base64.b64decode(state_dict_b64.encode("utf-8"))
    buffer = io.BytesIO(raw)
    payload = torch.load(buffer, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Decoded payload is not a state_dict-like mapping")
    return payload