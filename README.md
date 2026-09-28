# Bilateral Modular Motor Control

This repository contains the training and evaluation code for a simulated
bimanual motor-control study. Two recurrent controllers jointly operate two
six-muscle arms in a simultaneous tracking and holding task. The experiments
cross controller-specific versus shared objectives with fixed versus balanced
hand-task assignments.

## Code structure

- `models.py`: bilateral and monolithic recurrent controllers, delayed
  inter-controller communication, motor routing, and graded interventions.
- `tasks.py`: MotorNet-based bimanual arm environments and the simultaneous
  tracking-and-holding task.
- `metrics.py`: behavioural, energetic, routing, and functional
  specialisation metrics.
- `inference_evaluation.py`: matched post-training evaluation and intervention
  routines.
- `sweep_bilateral_with_cc.py`: main training and factorial-sweep entry point.
- `run_inference.py`: checkpoint reconstruction and manual evaluation entry
  point.
- `run_fixed_balanced_remaining61.py`: fixed-objective balanced-assignment
  continuation used for the full factorial grid.
- `run_shared_balanced_targeted_pilot.py`: shared-objective
  balanced-assignment pilot launcher.
- `run_shared_balanced_remaining61.py`: shared-objective balanced-assignment
  continuation used for the full factorial grid.
- `provenance.py`: source and runtime metadata recorded with each run.

## Environment

The experiments used Python 3.10.19, PyTorch 2.5.1 with CUDA 12.1, NumPy
2.2.5, and MotorNet 0.2.0. TensorBoard is required for training logs.

Example installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 numpy==2.2.5 motornet==0.2.0 tensorboard
```

## Quick check

The following command runs a short CPU smoke test:

```bash
python sweep_bilateral_with_cc.py \
  --mode pilot \
  --conditions fixed_roles \
  --seed-values 0 \
  --device cpu \
  --smoke \
  --out-root outputs \
  --exp-name smoke_test
```

## Training conditions

The main entry point exposes three conditions directly:

- `fixed_roles`: controller-specific objectives with fixed hand-task
  assignment.
- `shared_fullcc`: shared objective with fixed hand-task assignment.
- `fixed_balanced`: controller-specific objectives with exactly balanced
  hand-task assignment.

For the shared-objective balanced-assignment condition, run
`run_shared_balanced_targeted_pilot.py` followed by
`run_shared_balanced_remaining61.py`. The corresponding fixed-objective
continuation is provided in `run_fixed_balanced_remaining61.py`.

Use `--help` on each entry point for partitioning, device, seed, and output
options. Full factorial experiments are designed to be partitioned across
independent scheduler jobs. Every completed run writes a checkpoint, a JSON
record, and a completion marker atomically.

## Data and checkpoints

No trained checkpoints, generated results, logs, or experimental data are
included in this repository.
