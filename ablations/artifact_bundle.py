"""Safe export and verification of ablation resume bundles.

The bundle is deliberately framework-independent: checkpoint tensor
validation remains the responsibility of the Torch checkpoint adapter.  This
module provides durable file/hash/path guarantees around that checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath

from .logging_contract import RunLayout


BUNDLE_MANIFEST = "bundle_manifest.json"
REQUIRED_FILES = ("config.json", "checkpoints/checkpoint_latest.pt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe bundle path: {name!r}")
    return path.as_posix()


def _bundle_files(layout: RunLayout) -> list[Path]:
    root = Path(layout.root)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative.startswith("exports/") or path.name.endswith(".tmp"):
            continue
        files.append(path)
    return files


def create_resume_bundle(layout: RunLayout, output: Path | None = None) -> Path:
    """Create an atomic zip bundle containing safe run state and hashes."""

    layout = RunLayout(Path(layout.root))
    root = Path(layout.root)
    files = _bundle_files(layout)
    names = {path.relative_to(root).as_posix() for path in files}
    missing = [name for name in REQUIRED_FILES if name not in names]
    if missing:
        raise FileNotFoundError(f"resume bundle missing required files: {missing}")

    bundle = Path(output) if output is not None else layout.exports / "resume_bundle.zip"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        _safe_name(path.relative_to(root).as_posix()): _sha256(path)
        for path in files
    }
    manifest = {
        "format_version": 1,
        "required_files": list(REQUIRED_FILES),
        "files": entries,
    }
    temporary = bundle.with_name(bundle.name + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr(BUNDLE_MANIFEST, json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(temporary, bundle)
    return bundle


def verify_resume_bundle(bundle: Path) -> dict:
    """Verify zip paths, manifest, required files, and every file hash."""

    bundle = Path(bundle)
    with zipfile.ZipFile(bundle) as archive:
        names = [_safe_name(name) for name in archive.namelist()]
        if len(names) != len(set(names)):
            raise ValueError("resume bundle contains duplicate paths")
        if BUNDLE_MANIFEST not in names:
            raise ValueError("resume bundle has no manifest")
        manifest = json.loads(archive.read(BUNDLE_MANIFEST))
        entries = manifest.get("files", {})
        for required in manifest.get("required_files", REQUIRED_FILES):
            if required not in entries or required not in names:
                raise ValueError(f"resume bundle missing required file: {required}")
        for name, expected in entries.items():
            _safe_name(name)
            if name not in names:
                raise ValueError(f"manifest references missing file: {name}")
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise ValueError(f"bundle checksum mismatch: {name}")
    return manifest


def extract_verified_bundle(bundle: Path, destination: Path) -> dict:
    """Verify then extract a bundle without permitting path traversal."""

    manifest = verify_resume_bundle(bundle)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        for name in manifest["files"]:
            relative = Path(*PurePosixPath(name).parts)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return manifest
