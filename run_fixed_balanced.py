"""Run the full-grid fixed-balanced condition with pilot-matched provenance.

The completed 15-run pilot predates the folder reorganisation and its source
manifest includes historical Slurm wrappers.  This launcher verifies that all
scientific sources are unchanged, then reuses the pilot's canonical source
signature for the remaining 305 runs.  Moving scheduler wrappers into the
archive therefore cannot split one matched experiment into two source blocks.
"""

from __future__ import annotations

import argparse
import copy
import json
from itertools import product
from pathlib import Path

import sweep_bilateral_with_cc as sweep
from provenance import collect_implementation_metadata


DEFAULT_OUT_ROOT = "M3results_fixed_role"
DEFAULT_EXP_NAME = "fixed_balanced_p08_fullcc_3constraints_seed01234_0901"
DEFAULT_CANONICAL_ROOT = (
    Path(DEFAULT_OUT_ROOT)
    / "fixed_balanced_p08_fullcc_3constraints_seed01234_0901"
)
PILOT_CELLS = (
    (0, 0.0, 0.0),
    (10, 1e-4, 0.1),
    (20, 1e-3, 0.1),
)
SEEDS = tuple(range(5))


def _load_canonical_metadata(result_root: Path) -> dict:
    """Validate the 15-run pilot and return compatible current metadata."""
    records = sorted((result_root / "runs").glob("*.json"))
    if len(records) < 15:
        raise RuntimeError(
            f"expected at least 15 records under {result_root / 'runs'}, "
            f"found {len(records)}"
        )

    expected = {
        (delay, energy, noise, seed)
        for (delay, energy, noise), seed in product(PILOT_CELLS, SEEDS)
    }
    observed = set()
    saved_metadata = []
    pilot_paths = []
    for path in records:
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        config = record.get("config", {})
        cell = (
            int(config.get("conduction_delay_steps", -1)),
            float(config.get("lambda_energy", -1.0)),
            float(config.get("noise_gain", -1.0)),
            int(record.get("seed", -1)),
        )
        if config.get("condition_name") != "fixed_balanced":
            raise RuntimeError(f"non-fixed-balanced pilot record: {path}")
        if cell not in expected:
            continue
        if cell in observed:
            raise RuntimeError(f"duplicate pilot cell {cell}")
        observed.add(cell)
        metadata = record.get("implementation_metadata")
        if not metadata or not metadata.get("source_sha256"):
            raise RuntimeError(f"missing source provenance in {path}")
        saved_metadata.append(metadata)
        pilot_paths.append(path)

    if observed != expected:
        raise RuntimeError(
            "pilot inventory mismatch: "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )

    canonical = saved_metadata[0]
    inconsistent = [
        path.name
        for path, metadata in zip(pilot_paths, saved_metadata)
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
    if (
        current.get("implementation_revision")
        != canonical.get("implementation_revision")
        or current.get("metric_schema_version")
        != canonical.get("metric_schema_version")
    ):
        raise RuntimeError(
            "implementation revision or metric schema changed since the pilot"
        )

    canonical_sources = canonical["source_sha256"]
    current_sources = current.get("source_sha256", {})
    mismatches = sorted(
        name
        for name in set(canonical_sources) | set(current_sources)
        if canonical_sources.get(name) != current_sources.get(name)
    )
    scientific_mismatches = [
        name for name in mismatches if not name.endswith(".slurm")
    ]
    if scientific_mismatches:
        raise RuntimeError(
            "scientific source changed since the fixed-balanced pilot: "
            + ", ".join(scientific_mismatches)
        )
    if mismatches:
        print(
            "[canonical provenance] scheduler-file relocation only: "
            + ", ".join(mismatches)
        )

    compatible = copy.deepcopy(current)
    compatible["source_sha256"] = copy.deepcopy(canonical_sources)
    return compatible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, required=True)
    parser.add_argument("--total", type=int, default=32)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument(
        "--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT
    )
    args = parser.parse_args()

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
            "device": "cuda",
            "cc_mode": "full",
            "contra_fraction": 0.8,
            "task4_role_bias": 0.5,
        },
        conditions=["fixed_balanced"],
    )


if __name__ == "__main__":
    main()
