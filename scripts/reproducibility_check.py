#!/usr/bin/env python3
"""Preflight checks for deterministic LendingClub EDA runs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


REQUIRED_PACKAGES = {
    "numpy": "1.26.4",
    "pandas": "2.2.2",
    "matplotlib": "3.8.4",
    "seaborn": "0.13.2",
    "pyarrow": "15.0.2",
    "jupyter": "1.0.0",
    "ipykernel": "6.29.4",
    "nbconvert": "7.16.4",
}


def project_root() -> Path:
    env_root = os.environ.get("PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gzip_header(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="", errors="replace") as handle:
        return next(csv.reader(handle))


def check_python() -> list[str]:
    messages = []
    version = sys.version_info
    if (version.major, version.minor) != (3, 11):
        messages.append(
            "Python version mismatch: expected 3.11.x, "
            f"found {platform.python_version()}."
        )
    return messages


def check_packages() -> list[str]:
    messages = []
    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            messages.append(f"Missing package: {package}=={expected}.")
            continue
        if actual != expected:
            messages.append(
                f"Package version mismatch for {package}: "
                f"expected {expected}, found {actual}."
            )
    return messages


def check_manifest(root: Path, verify_hashes: bool) -> list[str]:
    messages = []
    manifest_path = root / "data_manifest.json"
    if not manifest_path.exists():
        return [f"Missing data manifest: {manifest_path}"]

    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["files"]:
        data_path = (root / entry["path"]).resolve()
        if not data_path.exists():
            messages.append(f"Missing {entry['name']} data file: {data_path}")
            continue
        if data_path.stat().st_size == 0:
            messages.append(f"Empty {entry['name']} data file: {data_path}")
            continue

        actual_header = read_gzip_header(data_path)
        if actual_header != entry["header"]:
            messages.append(
                f"Header mismatch for {entry['name']}: "
                f"expected {len(entry['header'])} columns, "
                f"found {len(actual_header)} columns."
            )

        if verify_hashes:
            actual_hash = sha256(data_path)
            if actual_hash != entry["sha256"]:
                messages.append(
                    f"SHA256 mismatch for {entry['name']}: "
                    f"expected {entry['sha256']}, found {actual_hash}."
                )
    return messages


def check_notebook_policy(root: Path) -> list[str]:
    messages = []
    notebook = root / "EDA" / "LendingClub_Executive_EDA.ipynb"
    if not notebook.exists():
        return [f"Missing main EDA notebook: {notebook}"]

    text = notebook.read_text(errors="replace")
    required_snippets = [
        "RANDOM_STATE = 42",
        "sample_rows: int = 250_000",
        "chunk_size: int = 250_000",
        "ACCEPTED_KEEP_COLS",
        "REJECTED_KEEP_COLS",
        "LEAKAGE_EXACT",
    ]
    for snippet in required_snippets:
        if snippet not in text:
            messages.append(f"Notebook reproducibility marker missing: {snippet}")
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check environment, data, and notebook reproducibility markers."
    )
    parser.add_argument(
        "--skip-package-check",
        action="store_true",
        help="Skip installed package version checks.",
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip full SHA256 checks for large data archives.",
    )
    args = parser.parse_args()

    root = project_root()
    failures = []
    failures.extend(check_python())
    if not args.skip_package_check:
        failures.extend(check_packages())
    failures.extend(check_manifest(root, verify_hashes=not args.skip_hash_check))
    failures.extend(check_notebook_policy(root))

    if failures:
        print("Reproducibility preflight: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Reproducibility preflight: PASS")
    print(f"Project root: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
