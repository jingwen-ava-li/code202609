"""Content-addressed implementation metadata for reproducible checkpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path

import numpy as np
import torch


IMPLEMENTATION_REVISION = (
    'stage2_fixed_trajectory_stability_2026-09-01_v10_hand_balanced_pilot')
METRIC_SCHEMA_VERSION = 14
SOURCE_FILES = (
    'models.py',
    'tasks.py',
    'metrics.py',
    'inference_evaluation.py',
    'sweep_bilateral_with_cc.py',
    'run_inference.py',
    'provenance.py',
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_implementation_metadata(root: Path | None = None) -> dict:
    """Describe the exact source and runtime used for a new result artifact."""
    root = Path(root or Path(__file__).resolve().parent)
    return {
        'implementation_revision': IMPLEMENTATION_REVISION,
        'metric_schema_version': METRIC_SCHEMA_VERSION,
        'source_sha256': {
            name: _file_sha256(root / name)
            for name in SOURCE_FILES
            if (root / name).exists()
        },
        'runtime': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'numpy': np.__version__,
            'motornet': _package_version('motornet'),
        },
    }


def compare_implementation_metadata(saved: dict | None, current: dict) -> dict:
    """Return source/runtime mismatches without silently accepting them."""
    if not saved:
        return {
            'comparable': False,
            'legacy_missing_provenance': True,
            'source_mismatches': [],
            'runtime_mismatches': [],
        }
    source_mismatches = sorted(
        name for name, digest in saved.get('source_sha256', {}).items()
        if current.get('source_sha256', {}).get(name) != digest
    )
    runtime_mismatches = sorted(
        name for name, version in saved.get('runtime', {}).items()
        if current.get('runtime', {}).get(name) != version
    )
    return {
        'comparable': not source_mismatches,
        'legacy_missing_provenance': False,
        'source_mismatches': source_mismatches,
        'runtime_mismatches': runtime_mismatches,
        'saved_revision': saved.get('implementation_revision'),
        'current_revision': current.get('implementation_revision'),
    }
