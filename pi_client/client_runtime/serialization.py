from __future__ import annotations

import base64
import io
from typing import Dict

import torch


def encode_state_dict(state_dict: Dict[str, object]) -> str:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_state_dict(state_dict_b64: str) -> Dict[str, object]:
    raw = base64.b64decode(state_dict_b64.encode("utf-8"))
    payload = torch.load(io.BytesIO(raw), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Decoded payload is not a mapping")
    return payload