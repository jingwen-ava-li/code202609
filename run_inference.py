"""Manually rerun the same causal analysis performed after training."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch

from inference_evaluation import create_envs, run_post_training_evaluation, summarise_evaluation
from models import BilateralNetwork, MonolithicNetwork
from provenance import collect_implementation_metadata, compare_implementation_metadata


def _checkpoint_files(explicit, directories):
    files = [Path(path) for path in explicit]
    for directory in directories:
        files.extend(sorted(Path(directory).glob('*.pth')))
    unique = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or 'state_dict' not in payload or 'config' not in payload:
        raise ValueError(
            f'{path} is not a structured checkpoint with state_dict and config; '
            'legacy 48-dimensional checkpoints are intentionally not inferred as new models')
    return payload


def _build_model(config, device):
    reach, hold = create_envs(config, device)
    if reach.obs_dim_total != hold.obs_dim_total:
        raise RuntimeError('checkpoint task observations do not share one input size')
    family = config.get('model_family', 'bilateral')
    if family == 'bilateral':
        return BilateralNetwork(
            input_dim=reach.obs_dim_total,
            hidden_size=config['hidden_size'],
            output_dim=config['output_size'],
            conduction_delay_steps=config['conduction_delay_steps'],
            delay_semantics=config.get('delay_semantics', 'legacy_extra_step'),
            noise_gain=config['noise_gain'],
            contra_fraction=config['contra_fraction'],
            routing_normalization=config['routing_normalization'],
            cc_mode=config['cc_mode'],
            cc_bottleneck_dim=config['cc_bottleneck_dim'],
            shared_bias_trainable=config.get('shared_bias_trainable', True),
            device=str(device),
        ).to(device)
    if family == 'monolithic':
        return MonolithicNetwork(
            input_dim=reach.obs_dim_total,
            hidden_size=int(config.get(
                'monolithic_hidden_size', config['hidden_size'])),
            output_dim=config['output_size'],
            conduction_delay_steps=config['conduction_delay_steps'],
            noise_gain=config['noise_gain'],
            device=str(device),
        ).to(device)
    raise ValueError(f'unknown checkpoint model_family: {family!r}')


def _apply_task_overrides(config, args):
    config = deepcopy(config)
    if args.reach_radius is not None:
        config['reach_radius'] = args.reach_radius
    if args.external_force_std is not None:
        # Used only when perturbation_mode=iid.
        config['external_force_std'] = args.external_force_std
    if args.pulse_magnitude_range is not None:
        config['pulse_magnitude_range'] = tuple(args.pulse_magnitude_range)
    if args.eval_batch_size is not None:
        config['eval_batch_size'] = args.eval_batch_size
    if args.functional_specialisation_gains is not None:
        config['functional_specialisation_gains'] = tuple(
            args.functional_specialisation_gains)
    # Whole-module lesion is the gain=0 endpoint of every controller gain
    # curve, including reruns of checkpoints saved before this convention.
    config['functional_specialisation_gains'] = tuple(sorted({
        0.0,
        *(float(gain) for gain in config.get(
            'functional_specialisation_gains', (0.95, 0.975, 0.99))),
    }))
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Rerun lesion, graded, pathway-specific and phase-specific inference')
    parser.add_argument('--checkpoint', action='append', default=[])
    parser.add_argument('--model-dir', action='append', default=[])
    parser.add_argument('--bilateral-dir', action='append', default=[],
                        help='Backward-compatible alias for --model-dir')
    parser.add_argument('--out-dir', default='results_inference')
    parser.add_argument('--device', default=None)
    parser.add_argument('--analysis-profile', choices=['lesion', 'core'], default='core')
    parser.add_argument('--eval-seed', type=int, default=None)
    parser.add_argument('--eval-batch-size', type=int, default=None)
    parser.add_argument('--reach-radius', type=float, default=None)
    parser.add_argument('--external-force-std', type=float, default=None)
    parser.add_argument('--pulse-magnitude-range', type=float, nargs=2, default=None,
                        metavar=('MIN', 'MAX'))
    parser.add_argument('--functional-specialisation-gains', type=float, nargs='+', default=None,
                        metavar='GAIN')
    parser.add_argument('--allow-implementation-mismatch', action='store_true',
                        help='Explicitly allow a provenance-tracked checkpoint to run under changed source')
    args = parser.parse_args()

    files = _checkpoint_files(args.checkpoint, args.model_dir + args.bilateral_dir)
    if not files:
        parser.error('provide at least one --checkpoint or --model-dir')
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    index, failures = [], []
    current_implementation = collect_implementation_metadata()

    for number, path in enumerate(files, start=1):
        print(f'[{number}/{len(files)}] {path}')
        try:
            checkpoint = _load_checkpoint(path, device)
            architecture_config = deepcopy(checkpoint['config'])
            # Schema-v2 checkpoints were trained before the delay off-by-one fix.
            architecture_config.setdefault('delay_semantics', 'legacy_extra_step')
            provenance_check = compare_implementation_metadata(
                checkpoint.get('implementation_metadata'), current_implementation)
            if (checkpoint.get('schema_version', 2) >= 3
                    and not provenance_check['comparable']
                    and not args.allow_implementation_mismatch):
                raise RuntimeError(
                    'checkpoint source does not match the current implementation: '
                    f"{provenance_check['source_mismatches']}; pass "
                    '--allow-implementation-mismatch only for an intentional rerun')
            evaluation_config = _apply_task_overrides(architecture_config, args)
            model = _build_model(architecture_config, device)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            analysis = run_post_training_evaluation(
                model,
                evaluation_config,
                device,
                profile=args.analysis_profile,
                eval_seed=args.eval_seed,
            )
            summary = summarise_evaluation(analysis)
            record = {
                'schema_version': 6,
                'checkpoint': str(path.resolve()),
                'run_id': checkpoint.get('run_id', path.stem),
                'seed': checkpoint.get('seed'),
                'training_config': architecture_config,
                'evaluation_config': evaluation_config,
                'checkpoint_implementation_metadata': checkpoint.get('implementation_metadata'),
                'evaluation_implementation_metadata': current_implementation,
                'implementation_compatibility': provenance_check,
                **summary,
            }
            output = out_dir / f'{path.stem}_evaluation_{timestamp}.json'
            with open(output, 'w') as handle:
                json.dump(record, handle, indent=2)
            index.append({
                'checkpoint': str(path.resolve()),
                'output': str(output.resolve()),
                'reach_success': summary['reach_success'],
                'hold_success': summary['hold_success'],
                'DDI': summary['DDI'],
                'CI_raw': summary['CI_raw'],
                'CI_routed': summary['CI_routed'],
                'FSI': summary.get('FSI'),
                'role_aligned_FSI': summary.get('role_aligned_FSI'),
                'functional_role_score': summary.get('functional_role_score'),
                'absolute_FSI': summary.get('absolute_FSI'),
                'reciprocal_FSI': summary.get('reciprocal_FSI'),
                'expected_double_dissociation_gain_fraction': summary.get(
                    'expected_double_dissociation_gain_fraction'),
                'FSI_valid_gain_count': summary.get('FSI_valid_gain_count', 0),
                'CC_dependency_role_aligned_score': summary.get(
                    'CC_dependency_role_aligned_score'),
                'whole_module_dependency_role_aligned_score': summary.get(
                    'whole_module_dependency_role_aligned_score'),
                'trajectory_controller_via_stability_hand_fraction': summary.get(
                    'trajectory_controller_via_stability_hand_fraction'),
                'stability_controller_via_trajectory_hand_fraction': summary.get(
                    'stability_controller_via_trajectory_hand_fraction'),
                'component_probe_FSI': summary.get('component_probe_FSI'),
                'lesion_error_changes': summary['lesion_error_changes'],
                'component_probe_reach_lesion_error_changes': summary.get(
                    'component_probe_reach_lesion_error_changes'),
                'task4_gain_table': summary.get('task4_gain_table'),
                'task4_lesion_error_changes': summary.get(
                    'task4_lesion_error_changes'),
                'transfer_probes': summary.get('transfer_probes'),
            })
            print(
                f"  success: reach={summary['reach_success']:.3f} "
                f"hold={summary['hold_success']:.3f} both={summary['both_success']:.3f}")
            if summary.get('controller_interventions_applicable', True):
                print(
                    f"  controller specialisation: "
                    f"role_score="
                    f"{summary.get('functional_role_score', float('nan')):.3f} "
                    f"absolute_separation="
                    f"{summary.get('absolute_FSI', float('nan')):.3f} "
                    f"double_dissociation="
                    f"{summary.get('expected_double_dissociation_gain_fraction', float('nan')):.3f} "
                    f"valid_gains={summary.get('FSI_valid_gain_count', 0)}")
                print(
                    f"  pathway diagnostics: "
                    f"CC_aligned={summary.get('CC_dependency_role_aligned_score')} "
                    f"whole_module_aligned="
                    f"{summary.get('whole_module_dependency_role_aligned_score')} "
                    f"T_via_hold_hand="
                    f"{summary.get('trajectory_controller_via_stability_hand_fraction')} "
                    f"S_via_track_hand="
                    f"{summary.get('stability_controller_via_trajectory_hand_fraction')}")
                lesion = summary['lesion_error_changes']
                print(
                    f"  lesion delta-error: "
                    f"L(reach={lesion['lesion_left']['reach_error_change']:+.6f}, "
                    f"hold={lesion['lesion_left']['hold_error_change']:+.6f}) "
                    f"R(reach={lesion['lesion_right']['reach_error_change']:+.6f}, "
                    f"hold={lesion['lesion_right']['hold_error_change']:+.6f})")
            for lesion_name, values in summary.get(
                    'component_probe_reach_lesion_error_changes', {}).items():
                assigned = values.get('assigned_function_hand_metrics')
                if not assigned:
                    continue
                print(
                    f"  reach probe {lesion_name} "
                    f"({values['assigned_functional_role']} controller, "
                    f"{values['assigned_function_hand']} hand): "
                    f"trajectory_delta="
                    f"{assigned['trajectory_deviation']['change']:+.6f} "
                    f"acquisition_delta="
                    f"{assigned['endpoint_acquisition_error']['change']:+.6f} "
                    f"final_endpoint_delta="
                    f"{assigned['final_endpoint_error_descriptive']['change']:+.6f}")
            task4 = summary.get('task4_evaluation', {})
            if task4:
                assigned_key = evaluation_config.get(
                    'fixed_role_key', 'left_hold_right_track')
                print(
                    f"  fixed-role competence: assigned="
                    f"{task4[assigned_key]['both_success']:.3f} "
                    f"role-swap transfer="
                    f"{task4['left_track_right_hold']['both_success']:.3f}")
            for lesion_name, roles in summary.get(
                    'task4_lesion_error_changes', {}).items():
                canonical = roles['left_hold_right_track']
                reversed_role = roles['left_track_right_hold']
                print(
                    f"  task4 {lesion_name} delta-error: "
                    f"canonical(track={canonical['tracking_error_change']:+.6f}, "
                    f"hold={canonical['holding_error_change']:+.6f}) "
                    f"reversed(track={reversed_role['tracking_error_change']:+.6f}, "
                    f"hold={reversed_role['holding_error_change']:+.6f})")
            for row in summary.get('task4_gain_table', []):
                if row['path'] not in {'readout', 'whole_module'}:
                    continue
                print(
                    f"  task4 gain path={row['path']} gain={row['gain']:.3f} "
                    f"T(track={row['trajectory_controller_tracking_delta_m']:+.6f}, "
                    f"stable={row['trajectory_controller_stability_delta_m']:+.6f}, "
                    f"both={row['trajectory_controller_both_success']:.3f}) "
                    f"S(track={row['stability_controller_tracking_delta_m']:+.6f}, "
                    f"stable={row['stability_controller_stability_delta_m']:+.6f}, "
                    f"both={row['stability_controller_both_success']:.3f}) "
                    f"valid={row['valid_for_specialisation']} "
                    f"DD={row['expected_double_dissociation']}")
            for probe_name, probe in summary.get('transfer_probes', {}).items():
                intact_probe = probe['intact']
                print(
                    f"  {probe_name}: intact "
                    f"trajectory={intact_probe['tracking_mean_error']:.6f} "
                    f"stability={intact_probe['holding_mean_error']:.6f} "
                    f"both={intact_probe['both_success']:.3f}")
                for row in probe.get('gain_table', []):
                    if row['path'] not in {'readout', 'whole_module'}:
                        continue
                    print(
                        f"    path={row['path']} gain={row['gain']:.3f} "
                        f"T(track={row['trajectory_controller_tracking_delta_m']:+.6f}, "
                        f"stable={row['trajectory_controller_stability_delta_m']:+.6f}, "
                        f"both={row['trajectory_controller_both_success']:.3f}) "
                        f"S(track={row['stability_controller_tracking_delta_m']:+.6f}, "
                        f"stable={row['stability_controller_stability_delta_m']:+.6f}, "
                        f"both={row['stability_controller_both_success']:.3f}) "
                        f"valid={row['valid_for_specialisation']} "
                        f"DD={row['expected_double_dissociation']}")
            if summary.get('controller_interventions_applicable', True):
                print(
                    f"  contra-sanity only: DDI={summary['DDI']:.3f} "
                    f"CIraw={summary['CI_raw']:.3f} CIrouted={summary['CI_routed']:.3f}")
            metrics = summary['performance_metrics']
            print(
                f"  perf: endpoint={metrics['endpoint_error']:.6f} "
                f"trajectory={metrics['trajectory_error']:.6f} "
                f"move_var={metrics['movement_variance']:.6f} "
                f"hold_var={metrics['hold_variance']:.6e} "
                f"energy={metrics['energetic_cost']:.6f} CCI={metrics['CCI']:.3f}")
        except Exception as error:
            failures.append({'checkpoint': str(path), 'error': repr(error)})
            print(f'  [failed] {error}')
            import traceback
            traceback.print_exc()

    with open(out_dir / f'inference_index_{timestamp}.json', 'w') as handle:
        json.dump({'completed': index, 'failures': failures}, handle, indent=2)
    if failures:
        raise RuntimeError(f'{len(failures)} checkpoint evaluations failed')


if __name__ == '__main__':
    main()
