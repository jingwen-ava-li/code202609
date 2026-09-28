"""Extend the shared-objective, hand-balanced pilot to the full grid.

The three targeted constraint cells already contain five seeds each.  This
launcher preserves those 15 runs and trains only the other 61 factorial cells
for seeds 0--4 (305 runs).  New runs reuse the pilot's canonical provenance so
the completed folder remains one source-matched experiment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from itertools import product
from pathlib import Path

import sweep_bilateral_with_cc as sweep
from provenance import collect_implementation_metadata
from run_shared_balanced_targeted_pilot import (
    CONDITION_NAME,
    DEFAULT_EXP_NAME,
    DEFAULT_OUT_ROOT,
    PILOT_CELLS,
    SEEDS,
    _condition_spec,
)


DEFAULT_CANONICAL_ROOT = Path(DEFAULT_OUT_ROOT) / DEFAULT_EXP_NAME
SCIENTIFIC_SOURCE_FILES = (
    "models.py",
    "tasks.py",
    "metrics.py",
    "inference_evaluation.py",
    "sweep_bilateral_with_cc.py",
    "run_inference.py",
    "provenance.py",
)
PILOT_LAUNCH_FILES = (
    "run_shared_balanced_targeted_pilot.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_canonical_metadata(result_root: Path) -> dict:
    """Validate the 15-run pilot and return its canonical provenance."""
    expected = {
        (delay, energy, noise, seed)
        for (delay, energy, noise), seed in product(PILOT_CELLS, SEEDS)
    }
    observed = set()
    pilot_metadata = []
    pilot_paths = []

    for path in sorted((result_root / "runs").glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        config = record.get("config", {})
        if config.get("condition_name") != CONDITION_NAME:
            continue
        cell = (
            int(config.get("conduction_delay_steps", -1)),
            float(config.get("lambda_energy", -1.0)),
            float(config.get("noise_gain", -1.0)),
            int(record.get("seed", -1)),
        )
        if cell not in expected:
            continue
        if cell in observed:
            raise RuntimeError(f"duplicate shared-balanced pilot cell {cell}")

        required_config = {
            "objective_routing": "shared",
            "functional_roles_imposed": False,
            "task4_role_assignment_mode": "exact_balanced",
            "task4_primary_role_scope": "both_matched_roles",
            "cc_mode": "full",
        }
        mismatches = {
            key: (config.get(key), value)
            for key, value in required_config.items()
            if config.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"pilot config mismatch in {path}: {mismatches}")
        if abs(float(config.get("task4_role_bias", -1.0)) - 0.5) > 1e-12:
            raise RuntimeError(f"pilot role bias is not 0.5 in {path}")
        if abs(float(config.get("contra_fraction", -1.0)) - 0.8) > 1e-12:
            raise RuntimeError(f"pilot routing fraction is not 0.8 in {path}")

        run_id = record.get("run_id")
        checkpoint = result_root / "models" / f"{run_id}.pth"
        marker = result_root / "markers" / f"{run_id}.FINISHED"
        if not checkpoint.is_file() or not marker.is_file():
            raise RuntimeError(f"pilot artifacts missing for {run_id}")
        if marker.read_text().strip() not in ("", run_id):
            raise RuntimeError(f"pilot marker identity mismatch for {run_id}")
        if record.get("checkpoint_sha256") != _sha256(checkpoint):
            raise RuntimeError(f"pilot checkpoint hash mismatch for {run_id}")

        metadata = record.get("implementation_metadata")
        if not metadata or not metadata.get("source_sha256"):
            raise RuntimeError(f"pilot provenance missing in {path}")
        observed.add(cell)
        pilot_metadata.append(metadata)
        pilot_paths.append(path)

    if observed != expected:
        raise RuntimeError(
            "shared-balanced pilot inventory mismatch: "
            f"missing={sorted(expected - observed)}"
        )

    canonical = pilot_metadata[0]
    inconsistent = [
        path.name
        for path, metadata in zip(pilot_paths, pilot_metadata)
        if metadata.get("implementation_revision")
        != canonical.get("implementation_revision")
        or metadata.get("metric_schema_version")
        != canonical.get("metric_schema_version")
        or metadata.get("source_sha256") != canonical.get("source_sha256")
    ]
    if inconsistent:
        raise RuntimeError(
            "pilot records do not share one source signature: "
            + ", ".join(inconsistent[:10])
        )

    current = collect_implementation_metadata()
    if current.get("metric_schema_version") != canonical.get("metric_schema_version"):
        raise RuntimeError("metric schema changed since the pilot")
    canonical_sources = canonical["source_sha256"]
    current_sources = current.get("source_sha256", {})
    scientific_mismatches = [
        name
        for name in SCIENTIFIC_SOURCE_FILES
        if canonical_sources.get(name) != current_sources.get(name)
    ]
    if scientific_mismatches:
        raise RuntimeError(
            "scientific source changed since the shared-balanced pilot: "
            + ", ".join(scientific_mismatches)
        )

    source_root = Path(__file__).resolve().parent
    pilot_launch_mismatches = [
        name
        for name in PILOT_LAUNCH_FILES
        if not (source_root / name).is_file()
        or canonical_sources.get(name) != _sha256(source_root / name)
    ]
    if pilot_launch_mismatches:
        raise RuntimeError(
            "pilot launch definition changed or is missing: "
            + ", ".join(pilot_launch_mismatches)
        )

    return copy.deepcopy(canonical)


def _print_plan() -> None:
    remaining_cells = [
        (
            int(config["conduction_delay_steps"]),
            float(config["lambda_energy"]),
            float(config["noise_gain"]),
        )
        for config in sweep.REMAINING_FULLGRID_CONFIGS
    ]
    print(f"condition={CONDITION_NAME}")
    print("objective_routing=shared")
    print("hand_task_assignment=exact_balanced_50_50")
    print("contra_fraction=0.8; cc_mode=full")
    print(f"existing_pilot_runs={len(PILOT_CELLS) * len(SEEDS)}")
    print(f"remaining_cells={len(remaining_cells)}")
    print(f"remaining_unique_runs={len(remaining_cells) * len(SEEDS)}")
    print(f"final_unique_runs={64 * len(SEEDS)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total", type=int, default=32)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument(
        "--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    if args.plan_only:
        _print_plan()
        return

    sweep.CONDITION_SPECS[CONDITION_NAME] = _condition_spec()
    canonical_metadata = _load_canonical_metadata(args.canonical_root)
    sweep.collect_implementation_metadata = lambda: copy.deepcopy(
        canonical_metadata
    )
    sweep.run_sweep(
        mode="remaining_fullgrid",
        seed_values=list(SEEDS),
        part_idx=args.part,
        total_parts=args.total,
        exp_name=args.exp_name,
        out_root=args.out_root,
        overrides={
            "device": args.device,
            "cc_mode": "full",
            "contra_fraction": 0.8,
            "task4_role_bias": 0.5,
        },
        conditions=[CONDITION_NAME],
    )


if __name__ == "__main__":
    main()
