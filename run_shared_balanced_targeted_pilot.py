"""Launch the missing shared-loss, hand-balanced Task4 control.

This condition completes the 2x2 design crossing objective assignment
(differential versus shared) with physical hand-task assignment (fixed versus
exactly balanced).  It deliberately reuses the frozen model, task, metric,
training and inference implementations.  Only the condition configuration is
new.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from itertools import product
from pathlib import Path

import sweep_bilateral_with_cc as sweep
from provenance import collect_implementation_metadata


CONDITION_NAME = "shared_balanced"
DEFAULT_OUT_ROOT = "M3results_fixed_role"
DEFAULT_EXP_NAME = "shared_balanced_p08_fullcc_3constraints_seed01234_0910"
PILOT_CELLS = (
    (0, 0.0, 0.0),
    (10, 1e-4, 0.1),
    (20, 1e-3, 0.1),
)
SEEDS = tuple(range(5))
IMPLEMENTATION_REVISION = (
    "stage2_fixed_trajectory_stability_2026-09-10_v11_"
    "shared_balanced_targeted_pilot"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _condition_spec() -> dict:
    """Return the exact shared-objective, balanced-assignment condition."""
    spec = copy.deepcopy(sweep.CONDITION_SPECS["shared_fullcc"])
    spec.update(
        {
            "condition_name": CONDITION_NAME,
            "task4_role_bias": 0.5,
            "task4_role_assignment_mode": "exact_balanced",
            "task4_primary_role_scope": "both_matched_roles",
            "hand_role_training": "exact_half_each_hand_tracks_and_holds",
        }
    )
    return spec


def _implementation_metadata() -> dict:
    """Include this public launcher in the checkpoint provenance."""
    root = Path(__file__).resolve().parent
    metadata = copy.deepcopy(collect_implementation_metadata(root))
    metadata["implementation_revision"] = IMPLEMENTATION_REVISION
    for name in (Path(__file__).name,):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required provenance source missing: {path}")
        metadata["source_sha256"][name] = _sha256(path)
    return metadata


def _print_plan() -> None:
    planned = list(product(PILOT_CELLS, SEEDS))
    print(f"condition={CONDITION_NAME}")
    print("objective_routing=shared")
    print("hand_task_assignment=exact_balanced_50_50")
    print("contra_fraction=0.8; cc_mode=full")
    print(f"planned_unique_runs={len(planned)}")
    for (delay, energy, noise), seed in planned:
        print(f"d={delay}, lambda={energy:g}, k={noise:g}, seed={seed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--total", type=int, default=15)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--exp-name", default=DEFAULT_EXP_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    if args.plan_only:
        _print_plan()
        return

    sweep.CONDITION_SPECS[CONDITION_NAME] = _condition_spec()
    metadata = _implementation_metadata()
    sweep.collect_implementation_metadata = lambda: copy.deepcopy(metadata)
    sweep.run_sweep(
        mode="targeted_pilot",
        seed_values=list(SEEDS),
        part_idx=args.part,
        total_parts=args.total,
        exp_name=args.exp_name,
        out_root=args.out_root,
        overrides={
            "device": args.device,
            "cc_mode": "full",
            "contra_fraction": 0.8,
        },
        conditions=[CONDITION_NAME],
    )


if __name__ == "__main__":
    main()
