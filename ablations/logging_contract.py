"""Filesystem contracts for isolated ablation runs."""

import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @classmethod
    def create(
        cls,
        base_dir: Path,
        *,
        dataset: str,
        variant: str,
        seed: int,
        config_hash: str,
    ) -> "RunLayout":
        root = Path(base_dir) / dataset / variant / str(seed) / config_hash
        layout = cls(root)
        for directory in (
            layout.root,
            layout.logs,
            layout.metrics,
            layout.checkpoints,
            layout.reports,
            layout.exports,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return layout


def record_failure(layout: RunLayout, *, phase: str, last_completed_epoch: int) -> None:
    """Persist the active exception without exposing environment secrets."""

    payload = {
        "phase": phase,
        "last_completed_epoch": int(last_completed_epoch),
        "traceback": "".join(traceback.format_exception(*sys.exc_info())),
        "exit_status": "failed",
    }
    (layout.reports / "failure.json").write_text(json.dumps(payload, indent=2))


def append_jsonl(path: Path, record: dict) -> None:
    """Append one structured event and flush it to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
