"""Atomic checkpoint and schema-validation helpers for ablation runs."""

import hashlib
import os
from pathlib import Path
from typing import Callable

from .checkpoint_contract import missing_required_keys


def atomic_save(payload, path: Path, serializer: Callable | None = None) -> str:
    """Atomically save bytes or a serialized payload and write its checksum."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if serializer is None:
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("serializer is required for non-byte payloads")
        data = bytes(payload)
    else:
        data = serializer(payload)
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("serializer must return bytes")
        data = bytes(data)

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)

    digest = hashlib.sha256(data).hexdigest()
    checksum_path = path.with_name(path.name + ".sha256")
    checksum_path.write_text(f"{digest}  {path.name}\n")
    return digest


def validate_payload(payload: dict, *, expected_config: dict, use_exid: bool) -> dict:
    """Validate schema and exact expected configuration before resume."""

    missing = missing_required_keys(payload, use_exid=use_exid)
    if missing:
        raise ValueError(f"checkpoint missing required keys: {list(missing)}")
    saved_config = payload.get("config", {})
    mismatches = {
        key: (saved_config.get(key), value)
        for key, value in expected_config.items()
        if saved_config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint configuration mismatch: {mismatches}")
    return payload
