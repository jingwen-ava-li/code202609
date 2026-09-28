"""Shared inference execution for automatic evaluation and manual reruns."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Optional

import numpy as np
import torch

from metrics import MotorMetrics as MM
from tasks import (
    BimanualArms,
    BimanualCentreOutReaching,
    BimanualCoupledTrackHold,
    BimanualPosturalHolding,
)


def set_seed(seed: int) -> None:
    """Seed task sampling and signal-dependent action noise."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_fingerprint(*tensors: torch.Tensor) -> str:
    """Return a compact identity for sampled targets or perturbations."""
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def select_half_batch(tensor, half_batch, side, batch_dim=0):
    """Select trials whose active or disturbed hand is ``side``."""
    index = [slice(None)] * tensor.dim()
    index[batch_dim] = slice(0, half_batch) if side == 'left' else slice(half_batch, None)
    return tensor[tuple(index)]


def build_hold_perturbation_mask(batch_size, device):
    """Disturb left hands in the first half and right hands in the second."""
    half = batch_size // 2
    mask = torch.zeros(batch_size, 2, device=device)
    mask[:half, 0] = 1.0
    mask[half:, 1] = 1.0
    return mask


def configure_unimanual_reach(env, info):
    """Assign one reaching hand and one rest hand per trial, then refresh obs."""
    batch_size = info['start_pos_left'].shape[0]
    half = batch_size // 2
    targets_left = env.targets_left.clone()
    targets_right = env.targets_right.clone()
    targets_right[:half] = info['start_pos_right'][:half]
    targets_left[half:] = info['start_pos_left'][half:]
    env.set_current_targets(targets_left=targets_left, targets_right=targets_right)
    if not torch.allclose(env.current_targets_right[:half], info['start_pos_right'][:half]):
        raise RuntimeError('first-half rest-hand target was not assigned before control')
    if not torch.allclose(env.current_targets_left[half:], info['start_pos_left'][half:]):
        raise RuntimeError('second-half rest-hand target was not assigned before control')
    # reset() built an observation with two reach targets; refresh only after
    # replacing the rest-hand targets so the first control step is correct.
    return env.get_observation(), targets_left, targets_right


def create_envs(config: dict, device):
    """Construct matched reach and hold environments from saved configuration."""
    indicator = bool(config.get('include_task_indicator', False))
    timestep = config.get('timestep', 0.01)
    reach = BimanualCentreOutReaching(
        bimanual_arms=BimanualArms(timestep=timestep),
        device=device,
        reach_radius=config.get('reach_radius', 0.08),
        include_task_indicator=indicator,
        timestep=timestep,
    )
    hold = BimanualPosturalHolding(
        bimanual_arms=BimanualArms(timestep=timestep),
        device=device,
        hold_duration_ms=config.get('hold_duration_ms', 400),
        external_force_std=config.get('external_force_std', 0.5),
        perturbation_mode=config.get('perturbation_mode', 'pulse'),
        pulse_onset_range=config.get('pulse_onset_range', (5, 10)),
        pulse_duration_range=config.get('pulse_duration_range', (10, 15)),
        pulse_magnitude_range=config.get('pulse_magnitude_range', (0.5, 1.5)),
        recovery_min_steps=config.get('recovery_min_steps', 10),
        include_task_indicator=indicator,
        timestep=timestep,
    )
    if reach.obs_dim_total != hold.obs_dim_total:
        raise RuntimeError('reach and hold observations must have the same dimension')
    return reach, hold


def create_task4_env(config: dict, device, role_bias=None):
    """Construct Task4 from saved mechanics, optionally fixing its role assignment."""
    timestep = config.get('timestep', 0.01)
    role_assignment_mode = (
        config.get('task4_role_assignment_mode', 'bernoulli')
        if role_bias is None else 'bernoulli')
    return BimanualCoupledTrackHold(
        bimanual_arms=BimanualArms(timestep=timestep),
        device=device,
        duration_ms=config.get('task4_duration_ms', 1000),
        role_bias=(
            config.get('task4_role_bias', 0.8) if role_bias is None else role_bias),
        role_assignment_mode=role_assignment_mode,
        trajectory_amplitude_range=config.get(
            'task4_trajectory_amplitude_range', (0.04, 0.06)),
        trajectory_frequency_range=config.get(
            'task4_trajectory_frequency_range', (0.75, 1.25)),
        spring_stiffness=config.get('task4_spring_stiffness', 20.0),
        damping_coefficient=config.get('task4_damping_coefficient', 1.0),
        max_coupling_force=config.get('task4_max_coupling_force', 3.0),
        # Missing means a historical checkpoint whose Task4 had no
        # independent holding perturbation.
        hold_perturbation_mode=config.get(
            'task4_hold_perturbation_mode', 'none'),
        hold_pulse_onset_range=config.get(
            'task4_hold_pulse_onset_range', (5, 10)),
        hold_pulse_duration_range=config.get(
            'task4_hold_pulse_duration_range', (10, 15)),
        hold_pulse_magnitude_range=config.get(
            'task4_hold_pulse_magnitude_range', (0.5, 1.5)),
        hold_recovery_min_steps=config.get(
            'task4_hold_recovery_min_steps', 10),
        include_task_indicator=bool(config.get('include_task_indicator', False)),
        timestep=timestep,
    )


def create_fixed_target_probe_env(config: dict, device, probe: str):
    """Construct one simultaneous centre-out/stability transfer probe."""
    timestep = float(config.get('timestep', 0.01))
    if probe == 'trajectory_reach_stability_hold':
        duration_ms = int(config.get(
            'transfer_probe_reach_hold_duration_ms', 500))
        pulse_onset = config.get('task4_hold_pulse_onset_range', (5, 10))
        pulse_duration = config.get(
            'task4_hold_pulse_duration_range', (10, 15))
        recovery_steps = int(config.get('task4_hold_recovery_min_steps', 10))
    elif probe == 'bilateral_reach_then_hold':
        duration_ms = int(config.get(
            'transfer_probe_bilateral_duration_ms', 800))
        pulse_onset = config.get(
            'transfer_probe_bilateral_pulse_onset_range', (55, 60))
        pulse_duration = config.get(
            'transfer_probe_bilateral_pulse_duration_range', (8, 12))
        recovery_steps = int(config.get(
            'transfer_probe_bilateral_recovery_min_steps', 10))
    else:
        raise ValueError(f'unknown fixed-target probe: {probe}')
    return BimanualPosturalHolding(
        bimanual_arms=BimanualArms(timestep=timestep),
        device=device,
        hold_duration_ms=duration_ms,
        external_force_std=0.0,
        perturbation_mode='pulse',
        pulse_onset_range=pulse_onset,
        pulse_duration_range=pulse_duration,
        pulse_magnitude_range=config.get(
            'task4_hold_pulse_magnitude_range', (0.5, 1.5)),
        recovery_min_steps=recovery_steps,
        include_task_indicator=bool(config.get('include_task_indicator', False)),
        timestep=timestep,
    )


def _sample_mirrored_reach_targets(info: dict, radius: float, device):
    """Sample the established eight-direction mirrored centre-out targets."""
    batch_size = info['start_pos_left'].shape[0]
    index = torch.randint(0, 8, (batch_size,), device=device)
    angle_right = index.float() * (2.0 * np.pi / 8.0)
    angle_left = np.pi - angle_right
    target_left = info['start_pos_left'] + radius * torch.stack(
        [torch.cos(angle_left), torch.sin(angle_left)], dim=-1)
    target_right = info['start_pos_right'] + radius * torch.stack(
        [torch.cos(angle_right), torch.sin(angle_right)], dim=-1)
    return target_left, target_right


def _gain_label(gain: float) -> str:
    text = f'{float(gain):.6f}'.rstrip('0').rstrip('.')
    if '.' not in text:
        text += '.00'
    else:
        decimals = len(text.split('.', 1)[1])
        text += '0' * max(0, 2 - decimals)
    return text.replace('-', 'm').replace('.', 'p')


def _functional_gains(config: dict) -> tuple[float, ...]:
    """Return configured functional gains with whole-module failure at zero."""
    gains = {
        0.0,
        *(float(value) for value in config.get(
            'functional_specialisation_gains', (0.95, 0.975, 0.99))),
    }
    if any(not 0.0 <= gain < 1.0 for gain in gains):
        raise ValueError('functional specialisation gains must be in [0, 1)')
    return tuple(sorted(gains))


def build_intervention_conditions(config: dict, profile: str = 'core') -> list[dict]:
    """Build intact, lesion, graded, pathway and phase intervention conditions."""
    conditions = [{'name': 'intact'}]
    for side in ('left', 'right'):
        conditions.append({'name': f'lesion_{side}', 'side': side, 'gain': 0.0, 'path': 'both'})

    if profile == 'lesion':
        return conditions

    functional_gains = set(_functional_gains(config))
    graded_gains = {
        float(value) for value in config.get(
            'intervention_gains', (0.25, 0.5, 0.75))
    } | functional_gains
    for gain in sorted(graded_gains):
        if not 0.0 <= gain <= 1.0:
            raise ValueError('intervention gains must be in [0, 1]')
        # Whole-module gain zero is already evaluated once under the backwards-
        # compatible lesion name, then aliased into the gain curve downstream.
        if gain == 0.0:
            continue
        for side in ('left', 'right'):
            conditions.append({
                'name': f'graded_both_{side}_g{_gain_label(float(gain))}',
                'side': side,
                'gain': float(gain),
                'path': 'both',
            })

    # Functional sensitivity separates motor-output, communication and
    # whole-module effects. Hand-specific motor masks expose cross-hand
    # strategies inside the simultaneous track-and-stabilise task.
    for gain in sorted(functional_gains):
        for path in ('readout', 'cc'):
            for side in ('left', 'right'):
                conditions.append({
                    'name': f'graded_{path}_{side}_g{_gain_label(gain)}',
                    'side': side,
                    'gain': gain,
                    'path': path,
                })
        for side in ('left', 'right'):
            for hand in ('left', 'right'):
                conditions.append({
                    'name': (
                        f'graded_readout_{side}_to_{hand}_hand_'
                        f'g{_gain_label(gain)}'),
                    'side': side,
                    'hand': hand,
                    'gain': gain,
                    'path': 'readout',
                })

    for path in ('cc', 'readout'):
        for side in ('left', 'right'):
            conditions.append({
                'name': f'{path}_only_{side}_g0p00',
                'side': side,
                'gain': 0.0,
                'path': path,
            })

    phase_gain = float(config.get('phase_intervention_gain', 0.0))
    for phase in ('reach_pre_hit', 'hold_pulse', 'hold_recovery'):
        for side in ('left', 'right'):
            conditions.append({
                'name': f'{phase}_{side}_g{_gain_label(phase_gain)}',
                'side': side,
                'gain': phase_gain,
                'path': 'both',
                'phase': phase,
            })
    return conditions


def _alias_whole_module_zero(conditions: dict) -> None:
    """Expose each whole-module lesion as the gain-zero curve endpoint."""
    for side in ('left', 'right'):
        lesion_name = f'lesion_{side}'
        if lesion_name in conditions:
            conditions[f'graded_both_{side}_g0p00'] = conditions[lesion_name]


def _intervention(
        spec: dict, batch_size: int, device, active=None, output_dim=None):
    """Convert a condition and phase mask into the model's granular API."""
    if spec.get('name') == 'intact':
        return None
    if active is None:
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
    gain = torch.ones(batch_size, device=device)
    gain[active] = float(spec['gain'])
    masks = {}
    side = spec['side']
    if spec.get('path', 'both') in {'both', 'cc'}:
        masks[f'cc_{side}'] = gain
    if spec.get('path', 'both') in {'both', 'readout'}:
        hand = spec.get('hand')
        if hand is None:
            masks[f'readout_{side}'] = gain
        else:
            if output_dim is None or output_dim % 2 != 0:
                raise ValueError(
                    'hand-specific readout intervention requires an even output_dim')
            readout_gain = torch.ones(
                batch_size, output_dim, device=device)
            half = output_dim // 2
            hand_slice = slice(0, half) if hand == 'left' else slice(half, output_dim)
            if hand not in {'left', 'right'}:
                raise ValueError("intervention hand must be 'left' or 'right'")
            readout_gain[:, hand_slice] = gain.unsqueeze(-1)
            masks[f'readout_{side}'] = readout_gain
    return masks


def _rollout_reach(model, env, config, device, spec):
    batch_size = int(config.get('eval_batch_size', config.get('batch_size', 128)))
    half = batch_size // 2
    max_steps = int(config.get('max_episode_steps', 50))
    tolerance = float(config.get('target_tolerance', 0.02))
    dt = float(config.get('timestep', 0.01))

    obs, info = env.reset(batch_size=batch_size)
    obs, targets_l, targets_r = configure_unimanual_reach(env, info)
    starts_l, starts_r = info['start_pos_left'], info['start_pos_right']
    model.reset_buffers(batch_size)
    hidden = None
    hit_so_far = torch.zeros(batch_size, dtype=torch.bool, device=device)

    pos_l, pos_r, vel_l, vel_r, err_l, err_r = [], [], [], [], [], []
    plant_actions, deterministic_actions = [], []
    routed_l, routed_r, raw_l, raw_r = [], [], [], []
    with torch.no_grad():
        for _ in range(max_steps):
            phase = spec.get('phase')
            active = ~hit_so_far if phase == 'reach_pre_hit' else None
            intervention = None if phase and phase != 'reach_pre_hit' else _intervention(
                spec, batch_size, device, active=active,
                output_dim=int(config['output_size']))
            out = model(obs.unsqueeze(1), hidden_state=hidden, intervention=intervention)
            action = out[0].squeeze(1).clamp(0.0, 1.0)
            deterministic_action = out[4].squeeze(1)
            hidden = (out[1][0][:, -1], out[1][1][:, -1])
            routed_l.append(out[2][0].squeeze(1)); routed_r.append(out[2][1].squeeze(1))
            raw_l.append(out[3][0].squeeze(1)); raw_r.append(out[3][1].squeeze(1))
            obs, _, terminated, truncated, step_info = env.step(action)
            pl, pr = step_info['pos_world_left'], step_info['pos_world_right']
            el = torch.norm(pl - targets_l, dim=-1)
            er = torch.norm(pr - targets_r, dim=-1)
            active_error = torch.cat([el[:half], er[half:]])
            hit_so_far |= active_error <= tolerance
            pos_l.append(pl); pos_r.append(pr); err_l.append(el); err_r.append(er)
            plant_actions.append(action); deterministic_actions.append(deterministic_action)
            vel_l.append(step_info['states_left']['cartesian'][:, 2:4])
            vel_r.append(step_info['states_right']['cartesian'][:, 2:4])
            if bool(torch.as_tensor(terminated).all()) or bool(torch.as_tensor(truncated).all()):
                break

    def side_metrics(positions, velocities, errors, starts, targets, side):
        positions = select_half_batch(torch.stack(positions), half, side, batch_dim=1)
        velocities = select_half_batch(torch.stack(velocities), half, side, batch_dim=1)
        errors = select_half_batch(torch.stack(errors), half, side, batch_dim=1)
        starts = select_half_batch(starts, half, side)
        targets = select_half_batch(targets, half, side)
        hit_metrics = MM.compute_first_hit(errors, tolerance, dt)
        hit = hit_metrics['hit']
        first = hit_metrics['first_index']
        time = hit_metrics['time']
        metrics = MM.compute_reach_trajectory_metrics(
            positions, velocities, starts, targets, first)
        metrics.update({
            'success': hit.float().mean().item(),
            'min_distance': errors.min(dim=0).values.mean().item(),
            'final_endpoint_error': errors[-1].mean().item(),
            'time_to_target_success': time[hit].mean().item() if hit.any() else float(max_steps * dt),
            'time_to_target_censored': time.mean().item(),
        })
        return metrics, hit

    left, hit_l = side_metrics(pos_l, vel_l, err_l, starts_l, targets_l, 'left')
    right, hit_r = side_metrics(pos_r, vel_r, err_r, starts_r, targets_r, 'right')
    return {
        'left': left,
        'right': right,
        'energy': MM.compute_energy_cost(torch.stack(deterministic_actions)),
        'executed_energy': MM.compute_energy_cost(torch.stack(plant_actions)),
        'rollout_steps': len(plant_actions),
        'trial_fingerprint': _tensor_fingerprint(targets_l, targets_r),
        'success_per_trial': torch.cat([hit_l, hit_r]),
        'routed': (torch.stack(routed_l), torch.stack(routed_r)),
        'raw': (torch.stack(raw_l), torch.stack(raw_r)),
    }


def _rollout_hold(model, env, config, device, spec):
    batch_size = int(config.get('eval_batch_size', config.get('batch_size', 128)))
    half = batch_size // 2
    max_steps = env.hold_steps
    tolerance = float(config.get('hold_tolerance', 0.01))
    report_tolerances = tuple(
        float(value) for value in config.get(
            'hold_report_tolerances', (0.005, 0.01, 0.02)))
    dt = float(config.get('timestep', 0.01))
    disturbance = build_hold_perturbation_mask(batch_size, device)

    obs, info = env.reset(batch_size=batch_size)
    targets_l, targets_r = info['start_pos_left'], info['start_pos_right']
    env.set_current_targets(targets_left=targets_l, targets_right=targets_r)
    model.reset_buffers(batch_size)
    hidden = None
    pos_l, pos_r, err_l, err_r = [], [], [], []
    plant_actions, deterministic_actions = [], []
    pulse_masks, recovery_masks = [], []
    routed_l, routed_r, raw_l, raw_r = [], [], [], []

    with torch.no_grad():
        for _ in range(max_steps):
            phases = env.get_phase_masks(next_step=True)
            phase = spec.get('phase')
            if phase == 'hold_pulse':
                active = phases['pulse']
            elif phase == 'hold_recovery':
                active = phases['recovery']
            else:
                active = None
            intervention = None if phase == 'reach_pre_hit' else _intervention(
                spec, batch_size, device, active=active,
                output_dim=int(config['output_size']))
            out = model(obs.unsqueeze(1), hidden_state=hidden, intervention=intervention)
            action = out[0].squeeze(1).clamp(0.0, 1.0)
            deterministic_action = out[4].squeeze(1)
            hidden = (out[1][0][:, -1], out[1][1][:, -1])
            routed_l.append(out[2][0].squeeze(1)); routed_r.append(out[2][1].squeeze(1))
            raw_l.append(out[3][0].squeeze(1)); raw_r.append(out[3][1].squeeze(1))
            obs, _, terminated, truncated, step_info = env.step(action, perturbation_mask=disturbance)
            pl, pr = step_info['pos_world_left'], step_info['pos_world_right']
            pos_l.append(pl); pos_r.append(pr)
            plant_actions.append(action); deterministic_actions.append(deterministic_action)
            err_l.append(torch.norm(pl - targets_l, dim=-1)); err_r.append(torch.norm(pr - targets_r, dim=-1))
            pulse_masks.append(step_info['perturbation_phase']['pulse'])
            recovery_masks.append(step_info['perturbation_phase']['recovery'])
            if bool(torch.as_tensor(terminated).all()) or bool(torch.as_tensor(truncated).all()):
                break

    if len(plant_actions) != env.hold_steps:
        raise RuntimeError(
            f'hold inference ended after {len(plant_actions)} steps; expected {env.hold_steps}')

    pulse = torch.stack(pulse_masks); recovery = torch.stack(recovery_masks)
    # Left-arm local x is mirrored in the shared world frame; force directions
    # must use the same transform when computing directional overshoot.
    force_world_left = env.pulse_force_left.clone()
    force_world_left[:, 0] = -force_world_left[:, 0]
    left = MM.compute_hold_recovery_metrics(
        select_half_batch(torch.stack(pos_l), half, 'left', 1),
        select_half_batch(torch.stack(err_l), half, 'left', 1),
        select_half_batch(targets_l, half, 'left'),
        select_half_batch(force_world_left, half, 'left'),
        select_half_batch(recovery, half, 'left', 1),
        select_half_batch(pulse, half, 'left', 1), tolerance, dt,
        report_tolerances=report_tolerances)
    right = MM.compute_hold_recovery_metrics(
        select_half_batch(torch.stack(pos_r), half, 'right', 1),
        select_half_batch(torch.stack(err_r), half, 'right', 1),
        select_half_batch(targets_r, half, 'right'),
        select_half_batch(env.pulse_force_right, half, 'right'),
        select_half_batch(recovery, half, 'right', 1),
        select_half_batch(pulse, half, 'right', 1), tolerance, dt,
        report_tolerances=report_tolerances)
    deterministic_action_tensor = torch.stack(deterministic_actions)
    plant_action_tensor = torch.stack(plant_actions)
    cci = MM.compute_assigned_hand_cocontraction(
        deterministic_action_tensor, half_batch=half)
    return {
        'left': left,
        'right': right,
        'energy': MM.compute_energy_cost(deterministic_action_tensor),
        'executed_energy': MM.compute_energy_cost(plant_action_tensor),
        'CCI': cci,
        'rollout_steps': len(plant_actions),
        'trial_fingerprint': _tensor_fingerprint(
            env.pulse_onset, env.pulse_end,
            env.pulse_force_left, env.pulse_force_right, disturbance),
        'success_per_trial': torch.cat([left.pop('success_per_trial'), right.pop('success_per_trial')]),
        'routed': (torch.stack(routed_l), torch.stack(routed_r)),
        'raw': (torch.stack(raw_l), torch.stack(raw_r)),
    }


def _rollout_fixed_target_transfer_probe(
        model, env, config: dict, device, probe: str, spec=None) -> dict:
    """Evaluate assigned trajectory/stability functions on centre-out probes."""
    spec = spec or {'name': 'intact'}
    batch_size = int(config.get('eval_batch_size', config.get('batch_size', 128)))
    trajectory_hand = config.get('trajectory_hand', 'right')
    stability_hand = config.get('stability_hand', 'left')
    if {trajectory_hand, stability_hand} != {'left', 'right'}:
        raise ValueError('transfer probes require opposite trajectory/stability hands')
    reach_tolerance = float(config.get('target_tolerance', 0.02))
    stability_tolerance = float(config.get('task4_hold_tolerance', 0.01))
    dt = float(config.get('timestep', 0.01))
    movement_steps = min(
        int(config.get('max_episode_steps', 50)), env.hold_steps)

    obs, reset_info = env.reset(batch_size=batch_size)
    sampled_left, sampled_right = _sample_mirrored_reach_targets(
        reset_info, float(config.get('reach_radius', 0.08)), device)
    starts = {
        'left': reset_info['start_pos_left'],
        'right': reset_info['start_pos_right'],
    }
    if probe == 'trajectory_reach_stability_hold':
        targets = {
            trajectory_hand: (
                sampled_left if trajectory_hand == 'left' else sampled_right),
            stability_hand: starts[stability_hand],
        }
    elif probe == 'bilateral_reach_then_hold':
        targets = {'left': sampled_left, 'right': sampled_right}
    else:
        raise ValueError(f'unknown transfer probe: {probe}')
    env.set_current_targets(
        targets_left=targets['left'], targets_right=targets['right'])
    obs = env.get_observation()

    perturbation_mask = torch.zeros(batch_size, 2, device=device)
    perturbation_mask[:, 0 if stability_hand == 'left' else 1] = 1.0
    model.reset_buffers(batch_size)
    hidden = None
    positions = {'left': [], 'right': []}
    velocities = {'left': [], 'right': []}
    errors = {'left': [], 'right': []}
    pulse_masks, recovery_masks = [], []
    deterministic_actions, plant_actions = [], []
    with torch.no_grad():
        for _ in range(env.hold_steps):
            intervention = _intervention(
                spec, batch_size, device,
                output_dim=int(config['output_size']))
            out = model(
                obs.unsqueeze(1), hidden_state=hidden,
                intervention=intervention)
            action = out[0].squeeze(1).clamp(0.0, 1.0)
            deterministic_actions.append(out[4].squeeze(1))
            plant_actions.append(action)
            hidden = (out[1][0][:, -1], out[1][1][:, -1])
            obs, _, terminated, truncated, info = env.step(
                action, perturbation_mask=perturbation_mask)
            for hand in ('left', 'right'):
                positions[hand].append(info[f'pos_world_{hand}'])
                velocities[hand].append(
                    info[f'states_{hand}']['cartesian'][:, 2:4])
                errors[hand].append(torch.norm(
                    info[f'pos_world_{hand}'] - targets[hand], dim=-1))
            pulse_masks.append(info['perturbation_phase']['pulse'])
            recovery_masks.append(info['perturbation_phase']['recovery'])
            if bool(torch.as_tensor(terminated).all()) or bool(
                    torch.as_tensor(truncated).all()):
                break
    if len(deterministic_actions) != env.hold_steps:
        raise RuntimeError(
            f'{probe} ended after {len(deterministic_actions)} steps; '
            f'expected {env.hold_steps}')

    position = {hand: torch.stack(value) for hand, value in positions.items()}
    velocity = {hand: torch.stack(value) for hand, value in velocities.items()}
    error = {hand: torch.stack(value) for hand, value in errors.items()}
    pulse = torch.stack(pulse_masks)
    recovery = torch.stack(recovery_masks)
    stability_window = pulse | recovery

    trajectory_error = error[trajectory_hand][:movement_steps]
    hit_metrics = MM.compute_first_hit(trajectory_error, reach_tolerance, dt)
    trajectory_metrics = MM.compute_reach_trajectory_metrics(
        position[trajectory_hand][:movement_steps],
        velocity[trajectory_hand][:movement_steps],
        starts[trajectory_hand], targets[trajectory_hand],
        hit_metrics['first_index'])
    trajectory_hit = hit_metrics['hit']

    stability_error = error[stability_hand]
    stability_position = position[stability_hand]
    force_world = (
        env.pulse_force_left.clone()
        if stability_hand == 'left' else env.pulse_force_right.clone())
    if stability_hand == 'left':
        force_world[:, 0] = -force_world[:, 0]
    stability_metrics = MM.compute_hold_recovery_metrics(
        stability_position,
        stability_error,
        targets[stability_hand],
        force_world,
        recovery,
        pulse,
        stability_tolerance,
        dt,
        report_tolerances=tuple(config.get(
            'hold_report_tolerances', (0.005, 0.01, 0.02))),
    )
    stability_count = stability_window.sum(dim=0).clamp_min(1)
    stability_window_mean_per_trial = (
        (stability_error * stability_window).sum(dim=0) / stability_count)
    stability_success = stability_metrics.pop('success_per_trial')

    pre_pulse = ~pulse & ~recovery
    stability_arrived_before_pulse = (
        (stability_error <= reach_tolerance) & pre_pulse).any(dim=0)
    if probe == 'bilateral_reach_then_hold':
        stability_success = stability_success & stability_arrived_before_pulse
    else:
        # The stability hand begins at its assigned target in Probe 1.
        stability_arrived_before_pulse = torch.ones_like(stability_success)

    both_success = trajectory_hit & stability_success
    deterministic_action = torch.stack(deterministic_actions)
    plant_action = torch.stack(plant_actions)
    return {
        'probe': probe,
        'role': config.get('fixed_role_key', 'left_hold_right_track'),
        'trajectory_hand': trajectory_hand,
        'stability_hand': stability_hand,
        # These compatibility names let the established fixed-role causal
        # sensitivity calculation operate on the new functional metrics.
        'tracking_mean_error': trajectory_metrics['trajectory_deviation'],
        'holding_mean_error': stability_window_mean_per_trial.mean().item(),
        'trajectory_deviation': trajectory_metrics['trajectory_deviation'],
        'trajectory_initial_direction_error_rad': trajectory_metrics[
            'initial_direction_error_rad'],
        'trajectory_path_curvature': trajectory_metrics['path_curvature'],
        'trajectory_min_distance': trajectory_error.min(dim=0).values.mean().item(),
        'trajectory_time_to_target_s': hit_metrics['time'].mean().item(),
        'trajectory_success': trajectory_hit.float().mean().item(),
        'stability_mean_error': stability_window_mean_per_trial.mean().item(),
        'stability_peak_error': stability_metrics['peak_displacement'],
        'stability_residual_error': stability_metrics['residual_error'],
        'stability_settling_time_s': stability_metrics['settling_time'],
        'stability_post_pulse_variance': stability_metrics['post_pulse_variance'],
        'stability_overshoot': stability_metrics['overshoot'],
        'stability_arrival_before_pulse': (
            stability_arrived_before_pulse.float().mean().item()),
        'stability_success': stability_success.float().mean().item(),
        'both_success': both_success.float().mean().item(),
        'coupling_force_cap_fraction': 0.0,
        'energy': MM.compute_energy_cost(deterministic_action),
        'executed_energy': MM.compute_energy_cost(plant_action),
        'rollout_steps': len(deterministic_actions),
        'movement_window_steps': movement_steps,
        'stability_window': 'pulse_and_recovery',
        'mechanics_fingerprint': _tensor_fingerprint(
            starts['left'], starts['right'], targets['left'], targets['right'],
            env.pulse_onset, env.pulse_end,
            env.pulse_force_left, env.pulse_force_right,
            perturbation_mask),
    }


def _rollout_task4(model, env, config, device, spec=None):
    """Evaluate simultaneous tracking and pulse-rejection for one fixed role."""
    spec = spec or {'name': 'intact'}
    batch_size = int(config.get('eval_batch_size', config.get('batch_size', 128)))
    track_tolerance = float(config.get('task4_track_tolerance', 0.02))
    hold_tolerance = float(config.get('task4_hold_tolerance', 0.01))
    obs, reset_info = env.reset(batch_size=batch_size)
    track_right = reset_info['track_right']
    if not (bool(track_right.all()) or bool((~track_right).all())):
        raise RuntimeError('Task4 evaluation requires a fixed role assignment')

    model.reset_buffers(batch_size)
    hidden = None
    track_errors, hold_errors, hold_positions = [], [], []
    pulse_masks, recovery_masks = [], []
    deterministic_actions, coupling_forces, holding_forces = [], [], []
    with torch.no_grad():
        for _ in range(env.task4_steps):
            phase = spec.get('phase')
            if phase is None:
                active = None
            else:
                phases = env.get_phase_masks(next_step=True)
                if phase == 'task4_pre_pulse':
                    active = ~(phases['pulse'] | phases['recovery'])
                elif phase in {'task4_pulse', 'hold_pulse'}:
                    active = phases['pulse']
                elif phase in {'task4_recovery', 'hold_recovery'}:
                    active = phases['recovery']
                else:
                    raise ValueError(f'unknown Task4 intervention phase: {phase}')
            intervention = _intervention(
                spec, batch_size, device, active=active,
                output_dim=int(config['output_size']))
            out = model(
                obs.unsqueeze(1), hidden_state=hidden,
                intervention=intervention)
            action = out[0].squeeze(1).clamp(0.0, 1.0)
            deterministic_actions.append(out[4].squeeze(1))
            hidden = (out[1][0][:, -1], out[1][1][:, -1])
            obs, _, terminated, truncated, info = env.step(action)
            track_errors.append(torch.where(
                track_right, info['error_right'], info['error_left']))
            hold_errors.append(torch.where(
                track_right, info['error_left'], info['error_right']))
            hold_positions.append(torch.where(
                track_right.unsqueeze(-1),
                info['pos_world_left'], info['pos_world_right']))
            pulse_masks.append(info['perturbation_phase']['pulse'])
            recovery_masks.append(info['perturbation_phase']['recovery'])
            coupling_forces.append(info['coupling_force_left_world'])
            holding_forces.append(torch.where(
                track_right.unsqueeze(-1),
                info['holding_force_left_world'],
                info['holding_force_right_world']))
            if bool(torch.as_tensor(terminated).all()) or bool(torch.as_tensor(truncated).all()):
                break
    if len(track_errors) != env.task4_steps:
        raise RuntimeError(
            f'Task4 inference ended after {len(track_errors)} steps; '
            f'expected {env.task4_steps}')

    track = torch.stack(track_errors)
    hold = torch.stack(hold_errors)
    hold_position = torch.stack(hold_positions)
    pulse = torch.stack(pulse_masks)
    recovery = torch.stack(recovery_masks)
    stability_window = pulse | recovery
    coupling_force = torch.stack(coupling_forces)
    holding_force = torch.stack(holding_forces)
    force_cap = env.max_coupling_force
    track_mean_per_trial = track.mean(dim=0)
    stability_count = stability_window.sum(dim=0).clamp_min(1)
    hold_mean_per_trial = (
        (hold * stability_window).sum(dim=0) / stability_count)
    hold_target = torch.where(
        track_right.unsqueeze(-1),
        reset_info['start_pos_left'], reset_info['start_pos_right'])
    hold_force_direction = torch.where(
        track_right.unsqueeze(-1),
        reset_info['pulse_force_left'], reset_info['pulse_force_right'])
    recovery_metrics = MM.compute_hold_recovery_metrics(
        hold_position,
        hold,
        hold_target,
        hold_force_direction,
        recovery,
        pulse,
        hold_tolerance,
        float(config.get('timestep', 0.01)),
        report_tolerances=tuple(config.get(
            'hold_report_tolerances', (0.005, 0.01, 0.02))),
    )
    both_success = (
        (track_mean_per_trial <= track_tolerance)
        & (hold_mean_per_trial <= hold_tolerance))
    holding_peak_per_trial = hold.masked_fill(
        ~stability_window, float('-inf')).max(dim=0).values
    return {
        'role': 'left_hold_right_track' if bool(track_right.all()) else 'left_track_right_hold',
        'tracking_mean_error': track.mean().item(),
        'tracking_rmse': torch.sqrt(track.pow(2).mean()).item(),
        # Primary stability measures start at perturbation onset. Pre-pulse
        # quiet holding cannot dilute the force-rejection demand.
        'holding_mean_error': hold_mean_per_trial.mean().item(),
        'holding_whole_episode_mean_error': hold.mean().item(),
        'holding_pulse_mean_error': (
            hold[pulse].mean().item() if bool(pulse.any()) else 0.0),
        'holding_recovery_mean_error': (
            hold[recovery].mean().item() if bool(recovery.any()) else 0.0),
        'holding_residual_error': recovery_metrics['residual_error'],
        'holding_settling_time_s': recovery_metrics['settling_time'],
        'holding_post_pulse_variance': recovery_metrics['post_pulse_variance'],
        'holding_overshoot': recovery_metrics['overshoot'],
        'holding_peak_error': holding_peak_per_trial.mean().item(),
        'tracking_success': (track_mean_per_trial <= track_tolerance).float().mean().item(),
        'holding_success': (hold_mean_per_trial <= hold_tolerance).float().mean().item(),
        'both_success': both_success.float().mean().item(),
        'coupling_force_rms': torch.sqrt(
            coupling_force.pow(2).sum(dim=-1).mean()).item(),
        'coupling_force_cap_fraction': (
            (torch.norm(coupling_force, dim=-1) >= force_cap * (1.0 - 1e-5))
            .float().mean().item() if force_cap is not None else 0.0),
        'holding_perturbation_force_rms': torch.sqrt(
            holding_force.pow(2).sum(dim=-1).mean()).item(),
        'stability_metric_window': 'pulse_and_recovery',
        'energy': MM.compute_energy_cost(torch.stack(deterministic_actions)),
        'rollout_steps': len(track_errors),
        'trial_fingerprint': _tensor_fingerprint(
            reset_info['trajectory_amplitude'],
            reset_info['trajectory_frequency'],
            reset_info['trajectory_direction'],
            reset_info['pulse_onset'],
            reset_info['pulse_end'],
            reset_info['pulse_force_left'],
            reset_info['pulse_force_right'],
            reset_info['track_right']),
        'mechanics_fingerprint': _tensor_fingerprint(
            reset_info['start_pos_left'],
            reset_info['start_pos_right'],
            reset_info['trajectory_amplitude'],
            reset_info['trajectory_frequency'],
            reset_info['trajectory_direction'],
            reset_info['pulse_onset'],
            reset_info['pulse_end'],
            reset_info['pulse_force_left'],
            reset_info['pulse_force_right']),
    }


def evaluate_task4_fixed_roles(
        model, config: dict, device, eval_seed: int, spec=None) -> dict:
    """Replay canonical and reversed Task4 roles with matched mechanics."""
    was_training = model.training
    model.eval()
    evaluations = {}
    for name, role_bias in (
            ('left_hold_right_track', 1.0),
            ('left_track_right_hold', 0.0)):
        set_seed(eval_seed)
        evaluations[name] = _rollout_task4(
            model,
            create_task4_env(config, device, role_bias=role_bias),
            config,
            device,
            spec=spec,
        )
    if (evaluations['left_hold_right_track']['mechanics_fingerprint']
            != evaluations['left_track_right_hold']['mechanics_fingerprint']):
        raise RuntimeError('fixed-role Task4 evaluations did not receive matched mechanics')
    if was_training:
        model.train()
    return evaluations


def _primary_functional_specs(config: dict, profile: str) -> list[dict]:
    """Return intact and controller-level conditions, excluding hand masks."""
    requested_gains = set(_functional_gains(config))
    selected = []
    for spec in build_intervention_conditions(config, profile=profile):
        name = spec['name']
        is_controller_gain = (
            name.startswith(('graded_both_', 'graded_readout_', 'graded_cc_'))
            and '_to_' not in name
            and float(spec.get('gain', -1.0)) in requested_gains)
        if name in {'intact', 'lesion_left', 'lesion_right'} or is_controller_gain:
            selected.append(spec)
    return selected


def evaluate_fixed_target_transfer_probe(
        model, config: dict, device, eval_seed: int, probe: str,
        profile: str = 'core') -> dict:
    """Evaluate one transfer probe over matched controller gain conditions."""
    assigned_role = config.get('fixed_role_key', 'left_hold_right_track')
    conditions = {}
    matched_fingerprint = None
    for spec in _primary_functional_specs(config, profile):
        set_seed(eval_seed)
        result = _rollout_fixed_target_transfer_probe(
            model,
            create_fixed_target_probe_env(config, device, probe),
            config,
            device,
            probe,
            spec=spec,
        )
        fingerprint = result['mechanics_fingerprint']
        if matched_fingerprint is None:
            matched_fingerprint = fingerprint
        elif fingerprint != matched_fingerprint:
            raise RuntimeError(
                f'{probe} condition {spec["name"]} did not receive matched trials')
        conditions[spec['name']] = {assigned_role: result}
    _alias_whole_module_zero(conditions)
    functional = MM.compute_fixed_role_functional_specialisation(
        conditions,
        assigned_role=assigned_role,
        trajectory_controller=config.get('trajectory_controller', 'left'),
        stability_controller=config.get('stability_controller', 'right'),
        gains=_functional_gains(config),
        trajectory_scale=float(config.get('task4_track_tolerance', 0.02)),
        stability_scale=float(config.get('task4_hold_tolerance', 0.01)),
        trajectory_hand=config.get('trajectory_hand', 'right'),
        stability_hand=config.get('stability_hand', 'left'),
        min_intact_both_success=float(config.get(
            'functional_min_intact_both_success', 0.8)),
        min_competence_ratio=float(config.get(
            'functional_min_competence_ratio', 0.8)),
        max_force_cap_fraction=float(config.get(
            'functional_max_force_cap_fraction', 0.01)),
        min_effect_magnitude=float(config.get(
            'functional_min_effect_magnitude', 0.05)),
    )
    return {
        'probe': probe,
        'status': 'post_training_out_of_distribution_transfer_probe',
        'assigned_role': assigned_role,
        'intact': conditions['intact'][assigned_role],
        'conditions': conditions,
        'matched_mechanics_fingerprint': matched_fingerprint,
        'functional_specialisation': functional,
        'gain_table': _compact_functional_gain_table(functional),
    }


def _compact_functional_gain_table(functional: dict) -> list[dict]:
    """Flatten each controller-by-gain causal matrix for JSON/table export."""
    rows = []
    trajectory_controller = functional.get('trajectory_controller', 'left')
    stability_controller = functional.get('stability_controller', 'right')
    for path, path_values in functional.get('paths', {}).items():
        for label, entry in path_values.get('by_gain', {}).items():
            trajectory = entry['controllers'][trajectory_controller]
            stability = entry['controllers'][stability_controller]
            rows.append({
                'path': path,
                'gain_label': label,
                'gain': entry['gain'],
                'trajectory_controller': trajectory_controller,
                'stability_controller': stability_controller,
                'trajectory_controller_tracking_delta_m': trajectory[
                    'delta_trajectory'],
                'trajectory_controller_stability_delta_m': trajectory[
                    'delta_stability'],
                'stability_controller_tracking_delta_m': stability[
                    'delta_trajectory'],
                'stability_controller_stability_delta_m': stability[
                    'delta_stability'],
                'trajectory_controller_tracking_delta_normalised': trajectory[
                    'delta_trajectory_normalised'],
                'trajectory_controller_stability_delta_normalised': trajectory[
                    'delta_stability_normalised'],
                'stability_controller_tracking_delta_normalised': stability[
                    'delta_trajectory_normalised'],
                'stability_controller_stability_delta_normalised': stability[
                    'delta_stability_normalised'],
                'trajectory_controller_both_success': trajectory['both_success'],
                'stability_controller_both_success': stability['both_success'],
                'trajectory_controller_valid_operating_regime': trajectory[
                    'valid_operating_regime'],
                'stability_controller_valid_operating_regime': stability[
                    'valid_operating_regime'],
                'trajectory_controller_exclusion_reasons': trajectory[
                    'exclusion_reasons'],
                'stability_controller_exclusion_reasons': stability[
                    'exclusion_reasons'],
                'valid_for_specialisation': entry['valid_for_specialisation'],
                'expected_double_dissociation': entry[
                    'expected_double_dissociation'],
                'role_aligned_FSI': entry['role_aligned_FSI'],
            })
    return rows


def _flatten_condition(reach: dict, hold: dict) -> dict:
    output = {}
    for side in ('left', 'right'):
        for key, value in reach[side].items():
            output[f'reach_{key}_{side}'] = value
        for key, value in hold[side].items():
            output[f'hold_{key}_{side}'] = value
    output.update({
        'energy_reach': reach['energy'],
        'energy_hold': hold['energy'],
        'energy_total': reach['energy'] + hold['energy'],
        'executed_energy_reach': reach['executed_energy'],
        'executed_energy_hold': hold['executed_energy'],
        'executed_energy_total': reach['executed_energy'] + hold['executed_energy'],
        'CCI': hold['CCI']['CCI'],
        'CCI_left': hold['CCI']['CCI_left'],
        'CCI_right': hold['CCI']['CCI_right'],
        'reach_success': reach['success_per_trial'].float().mean().item(),
        'hold_success': hold['success_per_trial'].float().mean().item(),
        'both_success': (reach['success_per_trial'] & hold['success_per_trial']).float().mean().item(),
    })
    return output


def _run_performance_only_evaluation(
        model, config: dict, device, eval_seed: int) -> dict:
    """Evaluate a monolithic controller without inventing module interventions."""
    env_reach, env_hold = create_envs(config, device)
    set_seed(eval_seed)
    reach = _rollout_reach(
        model, env_reach, config, device, {'name': 'intact'})
    set_seed(eval_seed + 1)
    hold = _rollout_hold(
        model, env_hold, config, device, {'name': 'intact'})
    intact = _flatten_condition(reach, hold)
    fingerprints = {
        'reach': reach['trial_fingerprint'],
        'hold': hold['trial_fingerprint'],
    }

    task4_evaluation = evaluate_task4_fixed_roles(
        model, config, device, eval_seed=eval_seed + 2,
        spec={'name': 'intact'})
    transfer_probes = {}
    for offset, probe in enumerate((
            'trajectory_reach_stability_hold',
            'bilateral_reach_then_hold'), start=3):
        set_seed(eval_seed + offset)
        probe_result = _rollout_fixed_target_transfer_probe(
            model,
            create_fixed_target_probe_env(config, device, probe),
            config,
            device,
            probe,
            spec={'name': 'intact'},
        )
        transfer_probes[probe] = {
            'probe': probe,
            'intact': probe_result,
            'conditions': {'intact': probe_result},
            'gain_table': [],
            'functional_specialisation': {
                'available': False,
                'reason': 'single_controller_no_module_interventions',
            },
        }

    return {
        'schema_version': 8,
        'analysis_profile': 'performance_only',
        'eval_seed': eval_seed,
        'model_family': config.get('model_family', 'monolithic'),
        'controller_interventions_applicable': False,
        'metric_definitions': {
            'reach_success': 'first hit within the rollout',
            'reach_threshold_m': float(config.get(
                'target_tolerance', MM.TAU_REACH)),
            'hold_success': 'mean error over the last up-to-five recovery steps',
            'hold_threshold_m': float(config.get(
                'hold_tolerance', MM.TAU_HOLD)),
            'both_success': 'trial-index-matched reach AND hold success',
            'energy': 'sum of squared deterministic pre-noise muscle commands',
            'executed_energy': 'diagnostic sum of squared noisy/clipped plant commands',
            'movement_variance_scale': float(config.get(
                'movement_variance_scale', 0.1)),
            'controller_causal_metrics': 'not applicable to a single controller',
        },
        'conditions': {'intact': intact},
        'matched_trial_fingerprints': fingerprints,
        'causal_effects': {},
        'lesion_error_changes': {},
        'component_probe_reach_lesion_error_changes': {},
        'task4_evaluation': task4_evaluation,
        'task4_conditions': {'intact': task4_evaluation},
        'task4_gain_table': [],
        'matched_task4_fingerprints': {
            role: values['mechanics_fingerprint']
            for role, values in task4_evaluation.items()
        },
        'task4_lesion_error_changes': {},
        'transfer_probes': transfer_probes,
        'DDI': None,
        'CI_routed': None,
        'CI_raw': None,
        'graded_functional_specialisation': {
            'available': False,
            'reason': 'single_controller_no_module_interventions',
        },
        'task4_graded_functional_specialisation': {
            'available': False,
            'reason': 'single_controller_no_module_interventions',
        },
        'component_probe_graded_functional_specialisation': {
            'available': False,
            'reason': 'single_controller_no_module_interventions',
        },
    }


def run_post_training_evaluation(model, config: dict, device, profile: Optional[str] = None,
                                 eval_seed: Optional[int] = None) -> dict:
    """Run the exact analysis used automatically and by manual checkpoint reruns."""
    model.eval()
    profile = profile or config.get('analysis_profile', 'core')
    eval_seed = int(config.get('eval_seed', 4242) if eval_seed is None else eval_seed)
    if not bool(getattr(model, 'supports_controller_interventions', True)):
        return _run_performance_only_evaluation(
            model, config, device, eval_seed)
    conditions = {}
    intact_contributions = None
    matched_trial_fingerprints = None

    for spec in build_intervention_conditions(config, profile=profile):
        # Pathway-resolved and controller-to-hand perturbations are defined for
        # the simultaneous Task4 endpoint. The separate reach/hold component probes
        # retain their historical intervention inventory.
        if spec['name'].startswith(('graded_readout_', 'graded_cc_')):
            continue
        # Give reach and hold independent, condition-matched stochastic streams.
        env_reach, env_hold = create_envs(config, device)
        set_seed(eval_seed)
        reach = _rollout_reach(model, env_reach, config, device, spec)
        set_seed(eval_seed + 1)
        hold = _rollout_hold(model, env_hold, config, device, spec)
        trial_fingerprints = {
            'reach': reach['trial_fingerprint'],
            'hold': hold['trial_fingerprint'],
        }
        if matched_trial_fingerprints is None:
            matched_trial_fingerprints = trial_fingerprints
        elif trial_fingerprints != matched_trial_fingerprints:
            raise RuntimeError(
                f"condition {spec['name']} did not receive matched stochastic trials")
        conditions[spec['name']] = _flatten_condition(reach, hold)
        if spec['name'] == 'intact':
            intact_contributions = {
                'routed': (
                    torch.cat([reach['routed'][0], hold['routed'][0]], dim=0),
                    torch.cat([reach['routed'][1], hold['routed'][1]], dim=0),
                ),
                'raw': (
                    torch.cat([reach['raw'][0], hold['raw'][0]], dim=0),
                    torch.cat([reach['raw'][1], hold['raw'][1]], dim=0),
                ),
            }

    # The legacy lesion condition is the exact whole-module gain-zero endpoint.
    # Alias it into the graded curve without repeating the stochastic rollout.
    _alias_whole_module_zero(conditions)

    intact = conditions['intact']
    lesion_l, lesion_r = conditions['lesion_left'], conditions['lesion_right']
    tau_reach = float(config.get('target_tolerance', MM.TAU_REACH))
    tau_hold = float(config.get('hold_tolerance', MM.TAU_HOLD))

    def combo(result, side):
        return 0.5 * (
            result[f'reach_min_distance_{side}'] / tau_reach
            + result[f'hold_mean_error_{side}'] / tau_hold)

    ddi = MM.compute_ddi(
        combo(intact, 'left'), combo(intact, 'right'),
        combo(lesion_l, 'left'), combo(lesion_l, 'right'),
        combo(lesion_r, 'left'), combo(lesion_r, 'right'))
    ci_routed = MM.compute_contralateral_index(*intact_contributions['routed'])
    ci_raw = MM.compute_contralateral_index(*intact_contributions['raw'])
    component_probe_gains = tuple(
        gain for gain in _functional_gains(config) if gain > 0.0)
    component_probe_functional_specialisation = MM.compute_graded_functional_specialisation(
        conditions,
        # Gain zero is retained in the raw whole-module curve but excluded from
        # this legacy ungated mean, where it would dominate the summary.
        gains=component_probe_gains,
        metric_pairs=config.get('functional_specialisation_metric_pairs'),
    )

    def paired_metric(result, metric):
        return 0.5 * (
            float(result[f'{metric}_left']) + float(result[f'{metric}_right']))

    # Use the same primary errors as the FSI trajectory/residual pair. Positive
    # delta means lesion-induced worsening; absolute post-lesion values are
    # retained so a large percentage is not over-read when intact error is tiny.
    intact_reach_error = paired_metric(intact, 'reach_trajectory_deviation')
    intact_hold_error = paired_metric(intact, 'hold_residual_error')
    lesion_error_changes = {}
    for side, lesion in (('left', lesion_l), ('right', lesion_r)):
        reach_error = paired_metric(lesion, 'reach_trajectory_deviation')
        hold_error = paired_metric(lesion, 'hold_residual_error')
        lesion_error_changes[f'lesion_{side}'] = {
            'reach_error_metric': 'trajectory_deviation_m',
            'hold_error_metric': 'residual_error_m',
            'reach_error_intact': intact_reach_error,
            'reach_error_post_lesion': reach_error,
            'reach_error_change': reach_error - intact_reach_error,
            'reach_error_relative_change': (
                (reach_error - intact_reach_error) / max(abs(intact_reach_error), 1e-12)),
            'hold_error_intact': intact_hold_error,
            'hold_error_post_lesion': hold_error,
            'hold_error_change': hold_error - intact_hold_error,
            'hold_error_relative_change': (
                (hold_error - intact_hold_error) / max(abs(intact_hold_error), 1e-12)),
        }

    # Preserve the hand-resolved reach decomposition. The component reach probe
    # is target acquisition without a terminal dwell, so minimum distance is the
    # primary endpoint-acquisition metric. Final-step endpoint error is retained
    # as a descriptive transfer diagnostic rather than treated as trained
    # endpoint-stability competence.
    reach_metric_specs = {
        'trajectory_deviation': 'reach_trajectory_deviation',
        'endpoint_acquisition_error': 'reach_min_distance',
        'final_endpoint_error_descriptive': 'reach_final_endpoint_error',
        'initial_direction_error_rad': 'reach_initial_direction_error_rad',
    }
    controller_roles = {
        config.get('trajectory_controller', 'left'): 'trajectory',
        config.get('stability_controller', 'right'): 'stability',
    }
    role_hands = {
        'trajectory': config.get('trajectory_hand', 'right'),
        'stability': config.get('stability_hand', 'left'),
    }
    component_probe_reach_lesion_error_changes = {}
    for side, lesion in (('left', lesion_l), ('right', lesion_r)):
        role = controller_roles.get(side, 'unassigned')
        hand_metrics = {}
        for hand in ('left', 'right'):
            metric_values = {}
            for output_name, condition_name in reach_metric_specs.items():
                intact_value = float(intact[f'{condition_name}_{hand}'])
                lesion_value = float(lesion[f'{condition_name}_{hand}'])
                change = lesion_value - intact_value
                metric_values[output_name] = {
                    'intact': intact_value,
                    'post_lesion': lesion_value,
                    'change': change,
                    'relative_change': (
                        change / max(abs(intact_value), 1e-12)),
                }
            hand_metrics[hand] = metric_values

        mean_across_hands = {}
        for output_name in reach_metric_specs:
            mean_across_hands[output_name] = {
                field: 0.5 * sum(
                    hand_metrics[hand][output_name][field]
                    for hand in ('left', 'right'))
                for field in ('intact', 'post_lesion', 'change', 'relative_change')
            }
        assigned_hand = role_hands.get(role)
        component_probe_reach_lesion_error_changes[f'lesion_{side}'] = {
            'controller': side,
            'assigned_functional_role': role,
            'assigned_function_hand': assigned_hand,
            'by_reaching_hand': hand_metrics,
            'mean_across_hands': mean_across_hands,
            'assigned_function_hand_metrics': (
                hand_metrics.get(assigned_hand) if assigned_hand else None),
        }

    # Fixed-role competence and specialisation use the trained assignment.
    # The role-swapped rollout remains an out-of-assignment transfer diagnostic.
    requested_fsi_gains = set(_functional_gains(config))
    task4_specs = []
    for spec in build_intervention_conditions(config, profile=profile):
        is_primary_graded = (
            spec['name'].startswith((
                'graded_both_', 'graded_readout_', 'graded_cc_'))
            and float(spec.get('gain', -1.0)) in requested_fsi_gains)
        if spec['name'] in {'intact', 'lesion_left', 'lesion_right'} or is_primary_graded:
            task4_specs.append(spec)

    task4_conditions = {}
    matched_task4_fingerprints = None
    for spec in task4_specs:
        role_evaluation = evaluate_task4_fixed_roles(
            model, config, device, eval_seed=eval_seed + 2, spec=spec)
        fingerprints = {
            role: values['mechanics_fingerprint']
            for role, values in role_evaluation.items()
        }
        if matched_task4_fingerprints is None:
            matched_task4_fingerprints = fingerprints
        elif fingerprints != matched_task4_fingerprints:
            raise RuntimeError(
                f"Task4 condition {spec['name']} did not receive matched mechanics")
        task4_conditions[spec['name']] = role_evaluation

    _alias_whole_module_zero(task4_conditions)

    task4_evaluation = task4_conditions['intact']
    functional_kwargs = {
        'trajectory_controller': config.get('trajectory_controller', 'left'),
        'stability_controller': config.get('stability_controller', 'right'),
        'gains': _functional_gains(config),
        'trajectory_scale': float(config.get('task4_track_tolerance', 0.02)),
        'stability_scale': float(config.get('task4_hold_tolerance', 0.01)),
        'min_intact_both_success': float(config.get(
            'functional_min_intact_both_success', 0.8)),
        'min_competence_ratio': float(config.get(
            'functional_min_competence_ratio', 0.8)),
        'max_force_cap_fraction': float(config.get(
            'functional_max_force_cap_fraction', 0.01)),
        'min_effect_magnitude': float(config.get(
            'functional_min_effect_magnitude', 0.05)),
    }
    if config.get('task4_primary_role_scope') == 'both_matched_roles':
        task4_functional_specialisation = (
            MM.compute_hand_balanced_functional_specialisation(
                task4_conditions, **functional_kwargs))
    else:
        task4_functional_specialisation = (
            MM.compute_fixed_role_functional_specialisation(
                task4_conditions,
                assigned_role=config.get(
                    'fixed_role_key', 'left_hold_right_track'),
                trajectory_hand=config.get('trajectory_hand', 'right'),
                stability_hand=config.get('stability_hand', 'left'),
                **functional_kwargs,
            ))
    task4_lesion_error_changes = {}
    for side in ('left', 'right'):
        lesion_name = f'lesion_{side}'
        task4_lesion_error_changes[lesion_name] = {}
        for role, intact_role in task4_evaluation.items():
            lesion_role = task4_conditions[lesion_name][role]
            task4_lesion_error_changes[lesion_name][role] = {
                'tracking_error_intact': intact_role['tracking_mean_error'],
                'tracking_error_post_lesion': lesion_role['tracking_mean_error'],
                'tracking_error_change': (
                    lesion_role['tracking_mean_error']
                    - intact_role['tracking_mean_error']),
                'holding_error_intact': intact_role['holding_mean_error'],
                'holding_error_post_lesion': lesion_role['holding_mean_error'],
                'holding_error_change': (
                    lesion_role['holding_mean_error']
                    - intact_role['holding_mean_error']),
            }

    transfer_probes = {
        'trajectory_reach_stability_hold': (
            evaluate_fixed_target_transfer_probe(
                model, config, device, eval_seed=eval_seed + 3,
                probe='trajectory_reach_stability_hold', profile=profile)),
        'bilateral_reach_then_hold': (
            evaluate_fixed_target_transfer_probe(
                model, config, device, eval_seed=eval_seed + 4,
                probe='bilateral_reach_then_hold', profile=profile)),
    }
    return {
        'schema_version': 8,
        'analysis_profile': profile,
        'eval_seed': eval_seed,
        'model_family': config.get('model_family', 'bilateral'),
        'controller_interventions_applicable': True,
        'metric_definitions': {
            'reach_success': 'first hit within the rollout',
            'reach_threshold_m': tau_reach,
            'hold_success': 'mean error over the last up-to-five recovery steps',
            'hold_threshold_m': tau_hold,
            'hold_report_thresholds_m': list(config.get(
                'hold_report_tolerances', (0.005, 0.01, 0.02))),
            'hold_pulse_magnitude_range_n': list(config.get(
                'pulse_magnitude_range', (0.5, 1.5))),
            'hold_rollout_steps': int(config.get('hold_duration_ms', 400) / 1000.0
                                      / float(config.get('timestep', 0.01))),
            'hold_rollout_duration_ms': int(config.get('hold_duration_ms', 400)),
            'both_success': 'trial-index-matched reach AND hold success',
            'energy': 'sum of squared deterministic pre-noise muscle commands',
            'executed_energy': 'diagnostic sum of squared noisy/clipped plant commands',
            'CCI': 'deterministic-command CCI for the assigned disturbed hand only',
            'CI': 'final-model intact held-out reach+hold contribution asymmetry',
            'matched_stochastic_trials': True,
            'delay_semantics': config.get('delay_semantics', 'legacy_extra_step'),
            'movement_variance_scale': float(config.get('movement_variance_scale', 0.1)),
            'endpoint_error_role': 'descriptive only; not used by revised reach success or loss',
            'component_probe_reach_endpoint_metrics': {
                'endpoint_acquisition_error': (
                    'minimum target distance during the target-acquisition rollout'),
                'final_endpoint_error_descriptive': (
                    'target distance at the final rollout step; descriptive only '
                    'because the probe has no trained terminal-dwell phase'),
            },
            'lesion_error_change': (
                'post-lesion minus intact; reach uses trajectory deviation and '
                'hold uses residual recovery error; positive values are worse'),
            'component_probe_status': (
                'reach and hold are post-training component probes; primary training '
                'competence is measured in the assigned fixed functional role'),
            'transfer_probe_1': (
                'assigned trajectory hand performs centre-out reach while the '
                'assigned stability hand rejects a matched force pulse'),
            'transfer_probe_2': (
                'both hands perform matched centre-out reaches; the assigned '
                'stability hand is perturbed after the 50-step movement window'),
            'whole_module_gain_zero': (
                'path=both and gain=0; exact alias of the historical whole-module '
                'lesion with no duplicate rollout'),
            'task4_role_bias_training': float(config.get('task4_role_bias', 1.0)),
            'task4_role_assignment_mode': config.get(
                'task4_role_assignment_mode', 'bernoulli'),
            'task4_tracking_metric_window': 'whole episode',
            'task4_stability_metric_window': (
                'holding-hand mean error over pulse plus recovery; pre-pulse '
                'quiet holding is excluded from loss, competence and FSI'),
            'task4_hold_perturbation_mode': config.get(
                'task4_hold_perturbation_mode', 'none'),
            'task4_hold_pulse_onset_range_steps': list(config.get(
                'task4_hold_pulse_onset_range', (5, 10))),
            'task4_hold_pulse_duration_range_steps': list(config.get(
                'task4_hold_pulse_duration_range', (10, 15))),
            'task4_hold_pulse_magnitude_range_n': list(config.get(
                'task4_hold_pulse_magnitude_range', (0.5, 1.5))),
            'task4_mechanical_coupling': {
                'spring_stiffness_n_per_m': float(config.get(
                    'task4_spring_stiffness', 20.0)),
                'damping_coefficient_n_s_per_m': float(config.get(
                    'task4_damping_coefficient', 1.0)),
                'max_coupling_force_n': config.get(
                    'task4_max_coupling_force', 3.0),
            },
            'functional_gradient_routing': config.get(
                'functional_gradient_routing', 'none'),
            'trajectory_controller': config.get('trajectory_controller', 'left'),
            'stability_controller': config.get('stability_controller', 'right'),
            'cc_parameter_ownership': config.get(
                'cc_parameter_ownership', 'unspecified'),
            'shared_bias_trainable': bool(
                config.get('shared_bias_trainable', True)),
            'energy_gradient_routing': config.get(
                'energy_gradient_routing', 'global_undifferentiated'),
            'fixed_role_name': config.get(
                'fixed_role_name', 'right_trajectory_left_stability'),
            'task4_role_evaluation': (
                'both matched physical hand roles are equally primary'
                if config.get('task4_primary_role_scope') == 'both_matched_roles'
                else 'assigned role is primary; swapped role is a transfer diagnostic'),
            'task4_primary_FSI': (
                'signed tolerance-normalised local motor-readout sensitivity in '
                'both matched physical hand roles with validity and expected '
                'direction required separately in each role; CC-only and combined '
                'module sensitivity are reported separately'
                if config.get('task4_primary_role_scope') == 'both_matched_roles'
                else 'signed tolerance-normalised local motor-readout sensitivity '
                'in the assigned Trajectory/Stability role; CC-only and combined '
                'module sensitivity are reported separately'),
            'reciprocal_FSI': (
                'readout FSI retained only when both assigned sensitivities have '
                'the expected positive double-dissociation direction and all '
                'operating-regime gates pass'),
            'task4_functional_sensitivity_gates': {
                'minimum_intact_both_success': float(config.get(
                    'functional_min_intact_both_success', 0.8)),
                'minimum_both_success_fraction_of_intact': float(config.get(
                    'functional_min_competence_ratio', 0.8)),
                'maximum_force_cap_fraction': float(config.get(
                    'functional_max_force_cap_fraction', 0.01)),
                'minimum_controller_effect_magnitude': float(config.get(
                    'functional_min_effect_magnitude', 0.05)),
            },
            'task4_function_hand_identifiability': (
                'functional roles are trained equally on both physical hands; '
                'demand and physical-hand selectivity are reported separately'
                if config.get('task4_primary_role_scope') == 'both_matched_roles'
                else 'the single trained hand-role assignment confounds function '
                'with physical hand; controller-by-hand readout sensitivity is '
                'reported to expose indirect cross-hand strategies'),
            'component_probe_FSI_status': (
                'secondary out-of-distribution reach/hold diagnostic; not the primary '
                'specialisation endpoint for Task4-trained controllers'),
        },
        'conditions': conditions,
        'matched_trial_fingerprints': matched_trial_fingerprints,
        'causal_effects': MM.compute_causal_effects(conditions),
        'lesion_error_changes': lesion_error_changes,
        'component_probe_reach_lesion_error_changes': (
            component_probe_reach_lesion_error_changes),
        'task4_evaluation': task4_evaluation,
        'task4_conditions': task4_conditions,
        'task4_gain_table': _compact_functional_gain_table(
            task4_functional_specialisation),
        'matched_task4_fingerprints': matched_task4_fingerprints,
        'task4_lesion_error_changes': task4_lesion_error_changes,
        'transfer_probes': transfer_probes,
        'DDI': ddi,
        'CI_routed': {key: value.item() for key, value in ci_routed.items()},
        'CI_raw': {key: value.item() for key, value in ci_raw.items()},
        # The compatibility key now denotes the in-distribution Task4 endpoint.
        'graded_functional_specialisation': task4_functional_specialisation,
        'task4_graded_functional_specialisation': task4_functional_specialisation,
        'component_probe_graded_functional_specialisation': (
            component_probe_functional_specialisation),
    }


def summarise_evaluation(analysis: dict) -> dict:
    """Create compact sweep fields while preserving the full causal analysis."""
    intact = analysis['conditions']['intact']
    definitions = analysis.get('metric_definitions', {})

    def paired_mean(metric):
        return 0.5 * (
            float(intact[f'{metric}_left']) + float(intact[f'{metric}_right']))

    performance_metrics = {
        'success_rate': intact['both_success'],
        'reach_success': intact['reach_success'],
        'hold_success': intact['hold_success'],
        'endpoint_error': paired_mean('reach_final_endpoint_error'),
        'acquisition_error': paired_mean('reach_min_distance'),
        'trajectory_error': paired_mean('reach_trajectory_deviation'),
        'movement_variance': (
            paired_mean('reach_movement_variance')
            * float(definitions.get('movement_variance_scale', 0.1))
        ),
        'hold_variance': paired_mean('hold_position_variance'),
        'hold_post_pulse_variance': paired_mean('hold_post_pulse_variance'),
        'hold_mean_error': paired_mean('hold_mean_error'),
        'hold_peak_displacement': paired_mean('hold_peak_displacement'),
        'hold_residual_error': paired_mean('hold_residual_error'),
        'time_to_target': paired_mean('reach_time_to_target_censored'),
        'energetic_cost': intact['energy_total'],
        'energy_reach': intact['energy_reach'],
        'energy_hold': intact['energy_hold'],
        'executed_energy_total': intact['executed_energy_total'],
        'executed_energy_reach': intact['executed_energy_reach'],
        'executed_energy_hold': intact['executed_energy_hold'],
        'CCI': intact['CCI'],
    }
    summary = deepcopy(intact)
    interventions_applicable = bool(
        analysis.get('controller_interventions_applicable', True))
    summary.update({
        'model_family': analysis.get('model_family', 'bilateral'),
        'controller_interventions_applicable': interventions_applicable,
        'DDI': (
            analysis['DDI']['average'] if interventions_applicable else None),
        'CI': (
            analysis['CI_routed']['contralateral_index_total']
            if interventions_applicable else None),
        'CI_routed': (
            analysis['CI_routed']['contralateral_index_total']
            if interventions_applicable else None),
        'CI_raw': (
            analysis['CI_raw']['contralateral_index_total']
            if interventions_applicable else None),
        'performance_metrics': performance_metrics,
        # Compatibility aliases for the submitted-paper table decomposition.
        'Success_rate': performance_metrics['success_rate'],
        'Final_err_int': performance_metrics['endpoint_error'],
        'Acquisition_err_int': performance_metrics['acquisition_error'],
        'Err_traj_int': performance_metrics['trajectory_error'],
        'Var_move_int': performance_metrics['movement_variance'],
        'Var_hold_int': performance_metrics['hold_variance'],
        'Hold_err_int': performance_metrics['hold_mean_error'],
        'Ene_int': performance_metrics['energetic_cost'],
        'Ene_reach': performance_metrics['energy_reach'],
        'Ene_hold': performance_metrics['energy_hold'],
        'analysis': analysis,
        'lesion_error_changes': deepcopy(analysis['lesion_error_changes']),
        'component_probe_reach_lesion_error_changes': deepcopy(
            analysis.get('component_probe_reach_lesion_error_changes', {})),
        'task4_evaluation': deepcopy(analysis.get('task4_evaluation', {})),
        'task4_gain_table': deepcopy(analysis.get('task4_gain_table', [])),
        'task4_lesion_error_changes': deepcopy(
            analysis.get('task4_lesion_error_changes', {})),
        'transfer_probes': deepcopy(analysis.get('transfer_probes', {})),
    })
    for side in ('left', 'right') if interventions_applicable else ():
        lesion = analysis['lesion_error_changes'][f'lesion_{side}']
        summary.update({
            f'lesion_{side}_reach_error_post': lesion['reach_error_post_lesion'],
            f'lesion_{side}_reach_error_change': lesion['reach_error_change'],
            f'lesion_{side}_hold_error_post': lesion['hold_error_post_lesion'],
            f'lesion_{side}_hold_error_change': lesion['hold_error_change'],
        })
    functional = analysis.get('graded_functional_specialisation', {})
    if functional:
        hand_pathways = functional.get('readout_hand_pathways', {})
        summary.update({
            'functional_pathway_sensitivity': deepcopy(
                functional.get('paths', {})),
            'functional_readout_hand_pathways': deepcopy(
                functional.get('readout_hand_pathways', {})),
            'functional_single_role_hand_confound': functional.get(
                'single_role_function_hand_confound'),
            'trajectory_controller_via_stability_hand_fraction': (
                hand_pathways.get(
                    'mean_trajectory_controller_via_stability_hand_fraction')),
            'stability_controller_via_trajectory_hand_fraction': (
                hand_pathways.get(
                    'mean_stability_controller_via_trajectory_hand_fraction')),
        })
    if functional.get('available'):
        summary.update({
            'FSI': functional['mean_FSI'],
            'graded_FSI': functional['mean_FSI'],
            'absolute_FSI': functional['mean_absolute_FSI'],
            'role_aligned_FSI': functional['mean_role_aligned_FSI'],
            'functional_role_score': functional['mean_role_aligned_FSI'],
            'FSI_interpretation': (
                'absolute controller separation only; use functional_role_score '
                'and expected_double_dissociation for the assigned-role claim'),
            'reciprocal_FSI': functional['mean_reciprocal_FSI'],
            'FSI_reciprocal_gain_fraction': functional['reciprocal_gain_fraction'],
            'expected_double_dissociation_gain_fraction': (
                functional['expected_double_dissociation_gain_fraction']),
            'FSI_evaluated_gain_count': functional['evaluated_gain_count'],
            'FSI_valid_gain_count': functional['valid_gain_count'],
            'FSI_excluded_gain_count': functional['excluded_gain_count'],
            'FSI_reciprocal_gain_count': functional['reciprocal_gain_count'],
            'FSI_reciprocal_role_consistent': functional['reciprocal_role_consistent'],
            'FSI_scope': (
                'in_distribution_fixed_role_readout_local_sensitivity'),
            'trajectory_controller': functional['trajectory_controller'],
            'stability_controller': functional['stability_controller'],
        })
    for path, prefix in (
            ('cc', 'CC_dependency'),
            ('whole_module', 'whole_module_dependency')):
        path_values = functional.get('paths', {}).get(path, {})
        summary.update({
            f'{prefix}_available': path_values.get('available', False),
            f'{prefix}_valid_gain_count': path_values.get(
                'valid_gain_count', 0),
            f'{prefix}_role_aligned_score': path_values.get(
                'mean_role_aligned_FSI'),
            f'{prefix}_expected_double_dissociation_gain_fraction': (
                path_values.get('expected_double_dissociation_gain_fraction')),
        })
    component = analysis.get('component_probe_graded_functional_specialisation', {})
    if component.get('available'):
        primary = component['primary']
        summary.update({
            'component_probe_FSI': primary['mean_FSI'],
            'component_probe_reciprocal_FSI': primary['mean_reciprocal_FSI'],
            'component_probe_FSI_reciprocal_gain_fraction': (
                primary['reciprocal_gain_fraction']),
        })
    return summary
