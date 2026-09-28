"""Train matched fixed-role and architecture/objective baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from inference_evaluation import (
    create_task4_env,
    evaluate_task4_fixed_roles,
    run_post_training_evaluation,
    summarise_evaluation,
)
from models import BilateralNetwork, MonolithicNetwork
from provenance import collect_implementation_metadata


BASE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_size': 128,
    'eval_batch_size': 128,
    'n_epochs': 2000,
    'learning_rate': 0.001,
    'hidden_size': 64,
    # Width 137 gives 77,691 trainable parameters for the 46-D/12-output
    # monolithic controller, within 0.5% of the 77,312 trainable parameters in
    # the hidden-64 full-CC bilateral controller.
    'monolithic_hidden_size': 137,
    'output_size': 12,
    'max_episode_steps': 50,
    'movement_variance_scale': 0.1,
    'tbptt_k': 10,
    'log_every': 100,
    'timestep': 0.01,
    # Primary architecture: fixed 80:20 contralateral-to-ipsilateral routing.
    'contra_fraction': 0.8,
    'routing_normalization': 'rms',
    'cc_mode': 'full',
    'cc_bottleneck_dim': 8,
    # Positive delta reads h[t-delta]; zero delay reads latest causal h[t-1].
    'delay_semantics': 'causal_exact',
    'include_task_indicator': False,
    'energy_command': 'deterministic_pre_noise',
    'cci_scope': 'assigned_disturbed_hand',
    # Fixed functional roles. Both controllers remain active in one simultaneous
    # rollout, while each functional loss updates only its assigned controller.
    'training_regime': 'fixed_controller_function_task4',
    'condition_name': 'fixed_roles',
    'model_family': 'bilateral',
    'objective_routing': 'fixed_functional',
    'functional_roles_imposed': True,
    'functional_gradient_routing': 'exclusive_controller_parameter_groups',
    'cc_parameter_ownership': 'receiver_controller',
    'shared_bias_trainable': False,
    'energy_gradient_routing': 'all_controller_parameter_groups',
    'task4_duration_ms': 1000,
    'task4_role_bias': 1.0,
    'task4_role_assignment_mode': 'bernoulli',
    'task4_primary_role_scope': 'assigned_only',
    'fixed_role_key': 'left_hold_right_track',
    'fixed_role_name': 'right_trajectory_left_stability',
    'trajectory_hand': 'right',
    'stability_hand': 'left',
    'trajectory_controller': 'left',
    'stability_controller': 'right',
    'task4_trajectory_amplitude_range': (0.04, 0.06),
    'task4_trajectory_frequency_range': (0.75, 1.25),
    # Primary positive control removes cross-hand mechanics. The holding hand
    # receives the same pulse family used by the established hold probe.
    'task4_spring_stiffness': 0.0,
    'task4_damping_coefficient': 0.0,
    'task4_max_coupling_force': None,
    'task4_hold_perturbation_mode': 'pulse',
    'task4_hold_pulse_onset_range': (5, 10),
    'task4_hold_pulse_duration_range': (10, 15),
    'task4_hold_pulse_magnitude_range': (0.5, 1.5),
    'task4_hold_recovery_min_steps': 10,
    'task4_huber_delta': 0.01,
    'task4_tracking_weight': 1.0,
    'task4_holding_weight': 1.0,
    'task4_track_tolerance': 0.02,
    'task4_hold_tolerance': 0.01,
    # Reach is target acquisition, not endpoint dwelling.
    'reach_radius': 0.08,
    'target_tolerance': 0.02,
    'reach_dense_shaping_weight': 0.0,
    # Hold uses a fixed random pulse followed by an unforced recovery period.
    'hold_duration_ms': 400,
    # Canonical recovery success is 10 mm; 5/10/20-mm rates are all reported.
    'hold_tolerance': 0.01,
    'hold_report_tolerances': (0.005, 0.01, 0.02),
    'hold_huber_delta': 0.01,
    'external_force_std': 0.5,
    'perturbation_mode': 'pulse',
    'pulse_onset_range': (5, 10),
    'pulse_duration_range': (10, 15),
    'pulse_magnitude_range': (0.5, 1.5),
    'recovery_min_steps': 10,
    # Post-training transfer probes. Probe 1 replaces rhythmic tracking with a
    # centre-out reach while preserving the assigned perturbed hold. Probe 2
    # gives both hands matched reaches and perturbs the stability hand only
    # after the movement window.
    'transfer_probe_reach_hold_duration_ms': 500,
    'transfer_probe_bilateral_duration_ms': 800,
    'transfer_probe_bilateral_pulse_onset_range': (55, 60),
    'transfer_probe_bilateral_pulse_duration_range': (8, 12),
    'transfer_probe_bilateral_recovery_min_steps': 10,
    # Post-training causal analysis. It never enters optimisation.
    'analysis_profile': 'core',
    'eval_seed': 4242,
    'intervention_gains': (0.25, 0.5, 0.75),
    # Gain zero is the whole-module lesion endpoint. Near-intact values estimate
    # local causal slopes; wider values can be supplied during manual inference.
    'functional_specialisation_gains': (0.0, 0.95, 0.975, 0.99),
    'functional_min_intact_both_success': 0.8,
    'functional_min_competence_ratio': 0.8,
    'functional_max_force_cap_fraction': 0.01,
    'functional_min_effect_magnitude': 0.05,
    'functional_specialisation_metric_pairs': {
        'trajectory_residual': (
            'reach_trajectory_deviation', 'hold_residual_error'),
        'trajectory_peak': (
            'reach_trajectory_deviation', 'hold_peak_displacement'),
        'initial_peak': (
            'reach_initial_direction_error_rad', 'hold_peak_displacement'),
    },
    'phase_intervention_gain': 0.0,
    # Model selection uses only the assigned Trajectory/Stability role. The
    # swapped hand-role rollout is retained after training as a transfer probe.
    'task4_loss_normalization': 'none_separate_controller_objectives',
    'task4_validation_every': 50,
    'task4_validation_batch_size': 128,
    'task4_validation_seed': 31415,
    'task4_validation_selection': 'worst_case',
    'early_stop_start': 300,
    'early_stop_patience': 8,
    'early_stop_min_delta': 1e-3,
    # Start patience after both assigned functional components are competent.
    'early_stop_requires_competence': True,
    'early_stop_competence_threshold': 1.0,
}


PILOT_CONFIGS = (
    {'conduction_delay_steps': 0, 'lambda_energy': 0.0, 'noise_gain': 0.0},
)
TARGETED_PILOT_CONFIGS = (
    {'conduction_delay_steps': 0, 'lambda_energy': 0.0, 'noise_gain': 0.0},
    {'conduction_delay_steps': 10, 'lambda_energy': 1e-4, 'noise_gain': 0.1},
    {'conduction_delay_steps': 20, 'lambda_energy': 1e-3, 'noise_gain': 0.1},
)
# Compatibility alias for the targeted three-cell inventory.
FULL_CONFIGS = TARGETED_PILOT_CONFIGS
FULL_FACTORIAL_LEVELS = {
    'conduction_delay_steps': (0, 5, 10, 20),
    'lambda_energy': (0.0, 1e-5, 1e-4, 1e-3),
    'noise_gain': (0.0, 0.05, 0.1, 0.2),
}
FULL_FACTORIAL_CONFIGS = tuple(
    {
        'conduction_delay_steps': delay,
        'lambda_energy': energy,
        'noise_gain': noise,
    }
    for delay in FULL_FACTORIAL_LEVELS['conduction_delay_steps']
    for energy in FULL_FACTORIAL_LEVELS['lambda_energy']
    for noise in FULL_FACTORIAL_LEVELS['noise_gain']
)


def _constraint_key(config: dict) -> tuple[int, float, float]:
    return (
        int(config['conduction_delay_steps']),
        float(config['lambda_energy']),
        float(config['noise_gain']),
    )


COMPLETED_TARGET_CONFIG_KEYS = frozenset(
    _constraint_key(config) for config in FULL_CONFIGS)
REMAINING_FULLGRID_CONFIGS = tuple(
    config for config in FULL_FACTORIAL_CONFIGS
    if _constraint_key(config) not in COMPLETED_TARGET_CONFIG_KEYS
)
if len(FULL_FACTORIAL_CONFIGS) != 64 or len(REMAINING_FULLGRID_CONFIGS) != 61:
    raise RuntimeError('fixed-role factorial inventory must contain 64 total and 61 remaining cells')
SEED_POOL = list(range(10))


CONDITION_SPECS = {
    # The configured CC mode is retained so existing full-CC/bottleneck
    # positive-control commands continue to work.
    'fixed_roles': {
        'condition_name': 'fixed_roles',
        'model_family': 'bilateral',
        'objective_routing': 'fixed_functional',
        'functional_roles_imposed': True,
        'training_regime': 'fixed_controller_function_task4',
        'functional_gradient_routing': 'exclusive_controller_parameter_groups',
        'task4_loss_normalization': 'none_separate_controller_objectives',
        'energy_gradient_routing': 'all_controller_parameter_groups',
        'communication_delay_applicable': True,
        'contralateral_routing_applicable': True,
    },
    'fixed_balanced': {
        'condition_name': 'fixed_balanced',
        'model_family': 'bilateral',
        'objective_routing': 'fixed_functional',
        'functional_roles_imposed': True,
        'training_regime': 'fixed_controller_function_task4',
        'functional_gradient_routing': 'exclusive_controller_parameter_groups',
        'task4_loss_normalization': 'none_separate_controller_objectives',
        'energy_gradient_routing': 'all_controller_parameter_groups',
        'communication_delay_applicable': True,
        'contralateral_routing_applicable': True,
        'task4_role_bias': 0.5,
        'task4_role_assignment_mode': 'exact_balanced',
        'task4_primary_role_scope': 'both_matched_roles',
        'hand_role_training': 'exact_half_each_hand_tracks_and_holds',
    },
    'shared_fullcc': {
        'condition_name': 'shared_fullcc',
        'model_family': 'bilateral',
        'objective_routing': 'shared',
        'functional_roles_imposed': False,
        'training_regime': 'shared_objective_task4',
        'functional_gradient_routing': 'shared_total_objective',
        'task4_loss_normalization': 'none_shared_objective',
        'energy_gradient_routing': 'shared_total_objective',
        'communication_delay_applicable': True,
        'contralateral_routing_applicable': True,
        'cc_mode': 'full',
    },
    'fixed_nocc': {
        'condition_name': 'fixed_nocc',
        'model_family': 'bilateral',
        'objective_routing': 'fixed_functional',
        'functional_roles_imposed': True,
        'training_regime': 'fixed_controller_function_task4',
        'functional_gradient_routing': 'exclusive_controller_parameter_groups',
        'task4_loss_normalization': 'none_separate_controller_objectives',
        'energy_gradient_routing': 'all_controller_parameter_groups',
        'communication_delay_applicable': False,
        'contralateral_routing_applicable': True,
        'cc_mode': 'none',
    },
    'monolithic_shared': {
        'condition_name': 'monolithic_shared',
        'model_family': 'monolithic',
        'objective_routing': 'shared',
        'functional_roles_imposed': False,
        'training_regime': 'shared_objective_task4',
        'functional_gradient_routing': 'not_applicable_single_controller',
        'task4_loss_normalization': 'none_shared_objective',
        'energy_gradient_routing': 'shared_total_objective',
        'communication_delay_applicable': False,
        'contralateral_routing_applicable': False,
        'cc_parameter_ownership': 'not_applicable_single_controller',
        'cc_mode': 'none',
    },
}


def _set_train_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _detach_hidden(hidden_states, step, tbptt_k):
    hidden = (hidden_states[0][:, -1], hidden_states[1][:, -1])
    if (step + 1) % tbptt_k == 0:
        hidden = (hidden[0].detach(), hidden[1].detach())
    return hidden


def _prepare_functional_parameter_groups(model, config: dict) -> dict:
    """Validate the exclusive Trajectory/Stability parameter ownership."""
    trajectory_side = config.get('trajectory_controller')
    stability_side = config.get('stability_controller')
    if {trajectory_side, stability_side} != {'left', 'right'}:
        raise ValueError(
            'trajectory_controller and stability_controller must assign the '
            'two distinct bilateral controllers')
    if config.get('cc_parameter_ownership') != 'receiver_controller':
        raise ValueError('fixed functional routing requires receiver-owned CC parameters')
    if bool(config.get('shared_bias_trainable', True)):
        raise ValueError('shared motor bias must be frozen in fixed functional training')

    groups = model.functional_parameter_groups()
    trajectory = tuple(groups[trajectory_side])
    stability = tuple(groups[stability_side])
    shared = tuple(groups['shared'])
    trajectory_ids = {id(parameter) for parameter in trajectory}
    stability_ids = {id(parameter) for parameter in stability}
    if trajectory_ids & stability_ids:
        raise RuntimeError('Trajectory and Stability parameter groups overlap')
    if any(parameter.requires_grad for parameter in shared):
        raise RuntimeError('an unassigned shared parameter remains trainable')

    owned_ids = trajectory_ids | stability_ids
    trainable_ids = {
        id(parameter) for parameter in model.parameters()
        if parameter.requires_grad
    }
    if owned_ids != trainable_ids:
        raise RuntimeError(
            'functional parameter groups do not exactly cover all trainable parameters')
    return {
        'trajectory': trajectory,
        'stability': stability,
        'shared_frozen': shared,
    }


def _assign_functional_gradients(
        trajectory_objective,
        stability_objective,
        parameter_groups: dict,
) -> None:
    """Route each functional objective only to its assigned controller."""
    trajectory_parameters = parameter_groups['trajectory']
    stability_parameters = parameter_groups['stability']
    trajectory_gradients = torch.autograd.grad(
        trajectory_objective,
        trajectory_parameters,
        retain_graph=True,
        allow_unused=False,
    )
    stability_gradients = torch.autograd.grad(
        stability_objective,
        stability_parameters,
        retain_graph=False,
        allow_unused=False,
    )
    for parameter, gradient in zip(trajectory_parameters, trajectory_gradients):
        parameter.grad = gradient
    for parameter, gradient in zip(stability_parameters, stability_gradients):
        parameter.grad = gradient


def _parameter_gradient_norm(parameters) -> float:
    """Return the pre-clipping L2 norm for one controller parameter group."""
    squared_norm = sum(
        parameter.grad.detach().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    )
    if not torch.is_tensor(squared_norm):
        return 0.0
    return float(torch.sqrt(squared_norm))


def _build_training_model(config: dict, input_size: int, device):
    """Construct the controller family declared by the experiment condition."""
    family = config.get('model_family', 'bilateral')
    if family == 'bilateral':
        return BilateralNetwork(
            input_dim=input_size,
            hidden_size=config['hidden_size'],
            output_dim=config['output_size'],
            conduction_delay_steps=config['conduction_delay_steps'],
            delay_semantics=config.get('delay_semantics', 'causal_exact'),
            noise_gain=config['noise_gain'],
            contra_fraction=config['contra_fraction'],
            routing_normalization=config['routing_normalization'],
            cc_mode=config['cc_mode'],
            cc_bottleneck_dim=config['cc_bottleneck_dim'],
            shared_bias_trainable=config.get('shared_bias_trainable', False),
            device=str(device),
        ).to(device)
    if family == 'monolithic':
        return MonolithicNetwork(
            input_dim=input_size,
            hidden_size=int(config['monolithic_hidden_size']),
            output_dim=config['output_size'],
            conduction_delay_steps=config['conduction_delay_steps'],
            noise_gain=config['noise_gain'],
            device=str(device),
        ).to(device)
    raise ValueError(f'unknown model_family: {family!r}')


def _parameter_inventory(model) -> dict:
    """Return nominal and effective-active counts for fairness audits."""
    trainable = int(sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad))
    effective_active = trainable
    inactive_reason = None
    if (getattr(model, 'model_family', None) == 'bilateral'
            and getattr(model, 'cc_mode', None) == 'none'):
        # BilateralNetwork retains the same GRU input tensor layout as the
        # connected model. Its hidden-sized cross-input block is identically
        # zero without CC, so the corresponding input columns never receive a
        # gradient and must not be counted as an active path.
        hidden = int(model.hidden_size)
        inactive_cross_input_weights = 2 * 3 * hidden * hidden
        effective_active -= inactive_cross_input_weights
        inactive_reason = 'zero_CC_cross_input_columns_excluded'
    return {
        'total': int(sum(parameter.numel() for parameter in model.parameters())),
        'trainable': trainable,
        'effective_active': effective_active,
        'effective_active_definition': inactive_reason or 'all_trainable_parameters',
    }


def _prepare_optimisation(model, config: dict) -> tuple[optim.Optimizer, dict | None]:
    """Create either exclusive functional routing or a shared objective update."""
    routing = config.get('objective_routing', 'fixed_functional')
    if routing == 'fixed_functional':
        if config.get('model_family') != 'bilateral':
            raise ValueError('fixed functional routing requires a bilateral model')
        groups = _prepare_functional_parameter_groups(model, config)
        optimizer = optim.Adam(
            [
                {
                    'params': groups['trajectory'],
                    'functional_role': 'trajectory',
                },
                {
                    'params': groups['stability'],
                    'functional_role': 'stability',
                },
            ],
            lr=config['learning_rate'],
        )
        return optimizer, groups
    if routing == 'shared':
        parameters = tuple(
            parameter for parameter in model.parameters()
            if parameter.requires_grad)
        if not parameters:
            raise RuntimeError('shared-objective model has no trainable parameters')
        return optim.Adam(parameters, lr=config['learning_rate']), None
    raise ValueError(f'unknown objective_routing: {routing!r}')


def _task4_validation_scores(roles: dict, config: dict) -> dict:
    """Score assigned-only or hand-balanced trajectory/stability competence."""
    track_tolerance = float(config['task4_track_tolerance'])
    hold_tolerance = float(config['task4_hold_tolerance'])
    assigned_role = config['fixed_role_key']
    role_scope = config.get('task4_primary_role_scope', 'assigned_only')
    if role_scope == 'both_matched_roles':
        required_roles = (
            'left_hold_right_track', 'left_track_right_hold')
        missing = [role for role in required_roles if role not in roles]
        if missing:
            raise KeyError(f'balanced validation roles unavailable: {missing}')
        components = {}
        for role in required_roles:
            values = roles[role]
            components[f'{role}_trajectory'] = (
                values['tracking_mean_error'] / track_tolerance)
            components[f'{role}_stability'] = (
                values['holding_mean_error'] / hold_tolerance)
        assigned_role_name = 'both_matched_hand_roles'
    elif role_scope == 'assigned_only':
        if assigned_role not in roles:
            raise KeyError(
                f'assigned validation role {assigned_role!r} is unavailable')
        values = roles[assigned_role]
        components = {
            'trajectory': values['tracking_mean_error'] / track_tolerance,
            'stability': values['holding_mean_error'] / hold_tolerance,
        }
        assigned_role_name = config['fixed_role_name']
    else:
        raise ValueError(f'unknown task4_primary_role_scope: {role_scope!r}')
    selection_method = config.get('task4_validation_selection', 'worst_case')
    if selection_method != 'worst_case':
        raise ValueError(
            '0817_fixed_role_assign requires '
            'task4_validation_selection=worst_case')
    return {
        'selection_method': selection_method,
        'role_scope': role_scope,
        'assigned_role_key': assigned_role,
        'assigned_role_name': assigned_role_name,
        'selection_score': float(max(components.values())),
        'worst_case_score': float(max(components.values())),
        'mean_score': float(np.mean(list(components.values()))),
        'worst_component': max(components, key=components.get),
        'normalised_components': components,
    }


def _task4_validation(model, config: dict, device) -> tuple[dict, dict]:
    """Evaluate held-out fixed roles without perturbing the training RNG."""
    cpu_rng = torch.get_rng_state()
    numpy_rng = np.random.get_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    validation_config = config.copy()
    validation_config['eval_batch_size'] = int(
        config.get('task4_validation_batch_size', 128))
    try:
        roles = evaluate_task4_fixed_roles(
            model,
            validation_config,
            device,
            eval_seed=int(config.get('task4_validation_seed', 31415)),
        )
    finally:
        torch.set_rng_state(cpu_rng)
        np.random.set_state(numpy_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)

    return _task4_validation_scores(roles, config), roles


def _update_early_stopping(
        validation_score: float,
        epoch: int,
        config: dict,
        patience_reference_score: float,
        stale_checks: int,
        competence_epoch: int | None,
) -> dict:
    """Advance competence-gated patience without stopping an unlearned run."""
    threshold = float(config.get('early_stop_competence_threshold', 1.0))
    requires_competence = bool(
        config.get('early_stop_requires_competence', True))
    newly_competent = competence_epoch is None and validation_score <= threshold
    if newly_competent:
        competence_epoch = epoch

    patience_active = not requires_competence or competence_epoch is not None
    if not patience_active:
        # Pre-competence progress remains eligible for best-checkpoint
        # selection, but must not consume post-competence patience.
        return {
            'patience_reference_score': float('inf'),
            'stale_checks': 0,
            'competence_epoch': competence_epoch,
            'patience_active': False,
            'should_stop': False,
        }

    if newly_competent and requires_competence:
        patience_reference_score = validation_score
        stale_checks = 0
    else:
        significantly_improved = (
            validation_score
            < patience_reference_score - float(config['early_stop_min_delta']))
        if significantly_improved:
            patience_reference_score = validation_score
            stale_checks = 0
        else:
            stale_checks += 1

    should_stop = (
        epoch >= int(config['early_stop_start'])
        and stale_checks >= int(config['early_stop_patience']))
    return {
        'patience_reference_score': patience_reference_score,
        'stale_checks': stale_checks,
        'competence_epoch': competence_epoch,
        'patience_active': True,
        'should_stop': should_stop,
    }


def train_single_config(config: dict, writer=None) -> dict:
    """Optimise one simultaneous Task4 condition with matched selection."""
    device = torch.device(config['device'])
    batch_size = int(config['batch_size'])
    if config.get('training_regime') not in {
            'fixed_controller_function_task4', 'shared_objective_task4'}:
        raise ValueError(
            '0817_fixed_role_assign requires '
            'a fixed-functional or shared Task4 training regime')
    role_assignment_mode = config.get(
        'task4_role_assignment_mode', 'bernoulli')
    role_bias = float(config.get('task4_role_bias', -1.0))
    if role_assignment_mode == 'exact_balanced':
        if not np.isclose(role_bias, 0.5):
            raise ValueError('exact-balanced training requires task4_role_bias=0.5')
        if batch_size % 2 != 0:
            raise ValueError('exact-balanced training requires an even batch size')
    elif role_assignment_mode == 'bernoulli':
        if not np.isclose(role_bias, 1.0):
            raise ValueError('assigned-only training requires task4_role_bias=1.0')
    else:
        raise ValueError(
            f'unknown task4_role_assignment_mode: {role_assignment_mode!r}')
    env_task4 = create_task4_env(config, device)
    input_size = env_task4.obs_dim_total

    model = _build_training_model(config, input_size, device)
    parameter_inventory = _parameter_inventory(model)
    optimizer, parameter_groups = _prepare_optimisation(model, config)
    best_validation_score = float('inf')
    patience_reference_score = float('inf')
    best_validation = None
    best_validation_scores = None
    best_state = None
    best_epoch = None
    stale_checks = 0
    competence_epoch = None
    early_stop_triggered = False
    training_stop_reason = 'max_epochs'
    validation_history = []
    last_trace = {}

    for epoch in range(int(config['n_epochs'])):
        model.train()
        optimizer.zero_grad()

        obs, task4_info = env_task4.reset(batch_size=batch_size)
        track_right = task4_info['track_right']
        canonical_count = int(track_right.sum().item())
        reversed_count = int((~track_right).sum().item())
        if role_assignment_mode == 'exact_balanced':
            if canonical_count != batch_size // 2 or reversed_count != batch_size // 2:
                raise RuntimeError(
                    'hand-balanced batch did not contain exact half canonical '
                    'and half reversed trials')
        elif not bool(track_right.all()):
            raise RuntimeError('assigned-only batch contained a reversed hand role')
        model.reset_buffers(batch_size)
        hidden = None
        track_errors, hold_errors, stability_masks = [], [], []
        deterministic_actions = []
        coupling_force_norms, holding_force_norms = [], []
        for step in range(env_task4.task4_steps):
            out = model(obs.unsqueeze(1), hidden_state=hidden)
            action = out[0].squeeze(1).clamp(0.0, 1.0)
            deterministic_action = out[4].squeeze(1)
            hidden = _detach_hidden(out[1], step, int(config['tbptt_k']))
            deterministic_actions.append(deterministic_action)
            obs, _, terminated, truncated, info = env_task4.step(action)
            track_errors.append(torch.where(
                track_right, info['error_right'], info['error_left']))
            hold_errors.append(torch.where(
                track_right, info['error_left'], info['error_right']))
            stability_masks.append(
                info['perturbation_phase']['pulse']
                | info['perturbation_phase']['recovery'])
            coupling_force_norms.append(torch.norm(
                info['coupling_force_left_world'], dim=-1))
            holding_force_norms.append(torch.norm(torch.where(
                track_right.unsqueeze(-1),
                info['holding_force_left_world'],
                info['holding_force_right_world']), dim=-1))
            if bool(torch.as_tensor(terminated).all()) or bool(torch.as_tensor(truncated).all()):
                break
        if len(deterministic_actions) != env_task4.task4_steps:
            raise RuntimeError(
                f'Task4 training ended after {len(deterministic_actions)} steps; '
                f'expected {env_task4.task4_steps}')

        track_errors = torch.stack(track_errors)
        hold_errors = torch.stack(hold_errors)
        stability_masks = torch.stack(stability_masks)
        delta = float(config['task4_huber_delta'])
        tracking_loss = F.huber_loss(
            track_errors, torch.zeros_like(track_errors), reduction='mean', delta=delta)
        hold_error_loss = F.huber_loss(
            hold_errors, torch.zeros_like(hold_errors), reduction='none', delta=delta)
        hold_position_loss = hold_error_loss[stability_masks].mean()
        task4_energy = torch.stack(deterministic_actions).pow(2).sum(dim=-1).mean()
        weighted_tracking_loss = (
            float(config['task4_tracking_weight']) * tracking_loss)
        weighted_stability_loss = (
            float(config['task4_holding_weight']) * hold_position_loss)
        weighted_energy_loss = float(config['lambda_energy']) * task4_energy
        trajectory_objective = weighted_tracking_loss + weighted_energy_loss
        stability_objective = weighted_stability_loss + weighted_energy_loss
        # Fixed-functional training routes the two objectives separately.
        # Shared baselines backpropagate this combined scalar through every
        # trainable parameter. Energy enters once in either update rule.
        task4_loss = (
            weighted_tracking_loss
            + weighted_stability_loss
            + weighted_energy_loss)
        total_loss = task4_loss

        if config.get('objective_routing') == 'fixed_functional':
            _assign_functional_gradients(
                trajectory_objective,
                stability_objective,
                parameter_groups,
            )
            trajectory_grad_norm = _parameter_gradient_norm(
                parameter_groups['trajectory'])
            stability_grad_norm = _parameter_gradient_norm(
                parameter_groups['stability'])
        else:
            total_loss.backward()
            trajectory_grad_norm = None
            stability_grad_norm = None
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        track_mean_per_trial = track_errors.mean(dim=0)
        stability_count = stability_masks.sum(dim=0).clamp_min(1)
        hold_mean_per_trial = (
            (hold_errors * stability_masks).sum(dim=0) / stability_count)
        track_success = (
            track_mean_per_trial <= float(config['task4_track_tolerance'])).float().mean()
        hold_success = (
            hold_mean_per_trial <= float(config['task4_hold_tolerance'])).float().mean()
        both_success = (
            (track_mean_per_trial <= float(config['task4_track_tolerance']))
            & (hold_mean_per_trial <= float(config['task4_hold_tolerance']))
        ).float().mean()
        last_trace = {
            'total_loss': total_loss.item(),
            'task4_loss': task4_loss.item(),
            'task4_tracking_loss': tracking_loss.item(),
            'task4_holding_loss': hold_position_loss.item(),
            'trajectory_controller_objective': trajectory_objective.item(),
            'stability_controller_objective': stability_objective.item(),
            'task4_tracking_mean_error': track_errors.mean().item(),
            'task4_holding_mean_error': hold_mean_per_trial.mean().item(),
            'task4_holding_whole_episode_mean_error': hold_errors.mean().item(),
            'trajectory_mean_error': track_errors.mean().item(),
            'stability_mean_error': hold_mean_per_trial.mean().item(),
            'task4_tracking_success': track_success.item(),
            'task4_holding_success': hold_success.item(),
            'task4_both_success': both_success.item(),
            'task4_canonical_role_fraction': track_right.float().mean().item(),
            'task4_canonical_role_count': canonical_count,
            'task4_reversed_role_count': reversed_count,
            'task4_coupling_force_rms': torch.sqrt(
                torch.stack(coupling_force_norms).pow(2).mean()).item(),
            'task4_coupling_force_cap_fraction': (
                (torch.stack(coupling_force_norms)
                 >= float(config['task4_max_coupling_force']) * (1.0 - 1e-5))
                .float().mean().item()
                if config.get('task4_max_coupling_force') is not None else 0.0),
            'task4_holding_perturbation_force_rms': torch.sqrt(
                torch.stack(holding_force_norms).pow(2).mean()).item(),
            'task4_energy_deterministic': task4_energy.item(),
            'grad_norm': float(grad_norm),
            'trajectory_controller_grad_norm': trajectory_grad_norm,
            'stability_controller_grad_norm': stability_grad_norm,
            'shared_objective_grad_norm': (
                float(grad_norm)
                if config.get('objective_routing') == 'shared' else None),
        }
        log_every = int(config.get('log_every', 100))
        if writer is not None and epoch % log_every == 0:
            for key, value in last_trace.items():
                if value is not None:
                    writer.add_scalar(f'train/{key}', value, epoch)
        if epoch % log_every == 0 or epoch == int(config['n_epochs']) - 1:
            print(
                f"    epoch={epoch:04d} total={last_trace['total_loss']:.4f} "
                f"task4_both={last_trace['task4_both_success']:.3f} "
                f"trajectory_err={last_trace['trajectory_mean_error']:.5f} "
                f"stability_err={last_trace['stability_mean_error']:.5f} "
                f"grad={last_trace['grad_norm']:.3f}",
                flush=True,
            )

        validation_every = int(config.get('task4_validation_every', 50))
        should_validate = (
            epoch % validation_every == 0
            or epoch == int(config['n_epochs']) - 1)
        if should_validate:
            validation_scores, validation_roles = _task4_validation(
                model, config, device)
            validation_score = validation_scores['selection_score']
            validation_record = {
                'epoch': epoch,
                # Compatibility alias: score is the configured selection score.
                'score': validation_score,
                **validation_scores,
                'roles': validation_roles,
            }
            validation_history.append(validation_record)
            if writer is not None:
                writer.add_scalar('validation/worst_case_normalised_error',
                                  validation_scores['worst_case_score'], epoch)
                writer.add_scalar('validation/mean_normalised_error',
                                  validation_scores['mean_score'], epoch)
                for role, values in validation_roles.items():
                    writer.add_scalar(
                        f'validation/{role}_tracking_mean_error',
                        values['tracking_mean_error'], epoch)
                    writer.add_scalar(
                        f'validation/{role}_holding_mean_error',
                        values['holding_mean_error'], epoch)
                    writer.add_scalar(
                        f'validation/{role}_both_success',
                        values['both_success'], epoch)
            if validation_score < best_validation_score:
                best_validation_score = validation_score
                best_validation = validation_roles
                best_validation_scores = validation_scores
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            early_stop_state = _update_early_stopping(
                validation_score=validation_score,
                epoch=epoch,
                config=config,
                patience_reference_score=patience_reference_score,
                stale_checks=stale_checks,
                competence_epoch=competence_epoch,
            )
            patience_reference_score = early_stop_state[
                'patience_reference_score']
            stale_checks = early_stop_state['stale_checks']
            competence_epoch = early_stop_state['competence_epoch']
            validation_record.update({
                'competence_achieved_by_epoch': competence_epoch is not None,
                'competence_epoch': competence_epoch,
                'early_stop_patience_active': early_stop_state['patience_active'],
                'early_stop_stale_checks': stale_checks,
            })
            patience_status = (
                f"active@{competence_epoch}"
                if early_stop_state['patience_active']
                else 'waiting-for-competence')
            print(
                f"    validation epoch={epoch:04d} "
                f"worst={validation_scores['worst_case_score']:.4f} "
                f"mean={validation_scores['mean_score']:.4f} "
                f"best={best_validation_score:.4f}@{best_epoch} "
                f"patience={patience_status} stale={stale_checks}",
                flush=True,
            )
            if early_stop_state['should_stop']:
                early_stop_triggered = True
                training_stop_reason = 'competence_gated_early_stop'
                print(
                    f'  Early stopping at epoch {epoch}; restoring '
                    f'validation-best epoch {best_epoch}')
                break

    if best_state is None:
        raise RuntimeError('no Task4 validation checkpoint was selected')
    model.load_state_dict(best_state, strict=True)

    analysis = run_post_training_evaluation(model, config, device)
    result = summarise_evaluation(analysis)
    result['training_trace'] = last_trace
    result['model_parameter_counts'] = parameter_inventory
    result['epochs_completed'] = epoch + 1
    result['checkpoint_selection'] = {
        'method': (
            'fixed_balanced_four_role_function_worst_normalised_error'
            if config.get('task4_primary_role_scope') == 'both_matched_roles'
            else 'fixed_assigned_role_worst_function_normalised_error'
            if config.get('objective_routing') == 'fixed_functional'
            else 'canonical_hand_role_worst_demand_normalised_error'),
        'best_epoch': best_epoch,
        'best_score': best_validation_score,
        'best_worst_case_score': best_validation_scores['worst_case_score'],
        'best_mean_score': best_validation_scores['mean_score'],
        'best_worst_component': best_validation_scores['worst_component'],
        'best_normalised_components': (
            best_validation_scores['normalised_components']),
        'selected_validation': best_validation,
        'final_training_epoch': epoch,
        'restored_best_state': True,
        'validation_every': int(config.get('task4_validation_every', 50)),
        'validation_seed': int(config.get('task4_validation_seed', 31415)),
        'validation_batch_size': int(
            config.get('task4_validation_batch_size', 128)),
        'early_stop_start': int(config['early_stop_start']),
        'early_stop_patience': int(config['early_stop_patience']),
        'early_stop_min_delta': float(config['early_stop_min_delta']),
        'early_stop_requires_competence': bool(
            config.get('early_stop_requires_competence', True)),
        'competence_threshold': float(
            config.get('early_stop_competence_threshold', 1.0)),
        'competence_achieved': competence_epoch is not None,
        'competence_epoch': competence_epoch,
        'early_stop_triggered': early_stop_triggered,
        'training_stop_reason': training_stop_reason,
        'stale_checks_at_stop': stale_checks,
    }
    result['validation_history'] = validation_history
    result['_model'] = model
    return result


def _config_id(config: dict) -> str:
    p = str(config['contra_fraction']).replace('.', 'p')
    cc = config['cc_mode']
    if cc == 'bottleneck':
        cc = f"bn{config['cc_bottleneck_dim']}"
    return (
        f"cond={config.get('condition_name', 'fixed_roles')}"
        f"_model={config.get('model_family', 'bilateral')}"
        f"_p={p}_cc={cc}_d={config['conduction_delay_steps']}"
        f"_l={config['lambda_energy']}_k={config['noise_gain']}"
        f"_Tctrl=L_Sctrl=R_hands=TrightSleft"
    )


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            'w', dir=path.parent, prefix=f'.{path.name}.',
            suffix='.tmp', delete=False) as handle:
        json.dump(payload, handle, indent=2)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checkpoint(path: Path, payload: dict) -> None:
    """Atomically persist and read back one structured checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f'.{path.name}.',
            suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        loaded = torch.load(temporary, map_location='cpu', weights_only=False)
        if not isinstance(loaded, dict) or not loaded.get('state_dict'):
            raise RuntimeError(
                f'checkpoint validation failed before commit: {temporary}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_completion_marker(path: Path, run_id: str) -> None:
    """Commit the completion marker only after both durable artifacts validate."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
            'w', dir=path.parent, prefix=f'.{path.name}.',
            suffix='.tmp', delete=False) as handle:
        handle.write(f'{run_id}\n')
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _normalise_json(value):
    """Normalise tuples/lists before comparing JSON and checkpoint metadata."""
    return json.loads(json.dumps(value, sort_keys=True))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _validate_run_artifacts(
        root: Path,
        run_id: str,
        seed: int,
        config: dict,
        implementation_metadata: dict,
) -> dict:
    """Validate one checkpoint/result pair and return its per-run record."""
    run_path = root / 'runs' / f'{run_id}.json'
    checkpoint_path = root / 'models' / f'{run_id}.pth'
    if not run_path.exists() or not checkpoint_path.exists():
        raise RuntimeError(
            f'incomplete run {run_id}: checkpoint/result pair is not both present')

    with open(run_path) as handle:
        record = json.load(handle)
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not checkpoint.get('state_dict'):
        raise RuntimeError(f'{checkpoint_path} is not a readable structured checkpoint')

    expected_config = _normalise_json(config)
    expected_source = implementation_metadata.get('source_sha256')
    for artifact_name, artifact in (
            ('result', record), ('checkpoint', checkpoint)):
        if artifact.get('schema_version') != 6:
            raise RuntimeError(
                f'{artifact_name} schema mismatch for {run_id}')
        if artifact.get('run_id') != run_id or artifact.get('seed') != seed:
            raise RuntimeError(
                f'{artifact_name} identity mismatch for {run_id}')
        if _normalise_json(artifact.get('config')) != expected_config:
            raise RuntimeError(
                f'{artifact_name} configuration mismatch for {run_id}')
        if (artifact.get('implementation_metadata', {}).get('source_sha256')
                != expected_source):
            raise RuntimeError(
                f'{artifact_name} source-provenance mismatch for {run_id}')

    checkpoint_keys = set(checkpoint['state_dict'])
    if not checkpoint_keys:
        raise RuntimeError(f'checkpoint state_dict is empty for {run_id}')
    if record.get('checkpoint_sha256') != _file_sha256(checkpoint_path):
        raise RuntimeError(
            f'checkpoint/result content hash mismatch for {run_id}')
    return record


def _balanced_partition_bounds(
        expected_total_runs: int, total_parts: int, part_idx: int,
) -> tuple[int, int, int, int]:
    """Return non-overlapping balanced [start, end) bounds for one partition."""
    if total_parts <= 0:
        raise ValueError('total must be positive')
    if total_parts > expected_total_runs:
        raise ValueError(
            f'total={total_parts} exceeds the {expected_total_runs} planned '
            'runs and would create empty partitions')
    if not 0 <= part_idx < total_parts:
        raise ValueError('part must satisfy 0 <= part < total')
    base_runs, remainder = divmod(expected_total_runs, total_parts)
    partition_runs = base_runs + int(part_idx < remainder)
    start = part_idx * base_runs + min(part_idx, remainder)
    return start, start + partition_runs, base_runs, remainder


def run_sweep(mode='pilot', n_seeds=None, seed_values=None, part_idx=0, total_parts=1,
              exp_name=None, out_root='results_fixed_role', overrides=None, smoke=False,
              conditions=None):
    """Run matched conditions over a declared Task4 constraint inventory."""
    mode_configs = {
        'pilot': PILOT_CONFIGS,
        'targeted_pilot': TARGETED_PILOT_CONFIGS,
        'full': FULL_CONFIGS,
        'remaining_fullgrid': REMAINING_FULLGRID_CONFIGS,
    }
    if mode not in mode_configs:
        raise ValueError(
            "mode must be 'pilot', 'targeted_pilot', 'full', or "
            "'remaining_fullgrid'")
    run_configs = list(mode_configs[mode])
    config_base = BASE_CONFIG.copy()
    config_base['contra_fraction'] = 0.8
    if overrides:
        config_base.update({key: value for key, value in overrides.items() if value is not None})
    if smoke:
        config_base.update({
            'batch_size': 8,
            'eval_batch_size': 8,
            'task4_validation_batch_size': 8,
            'task4_validation_every': 1,
            'n_epochs': 2,
            # Keep FSI and lesion-error reporting together even in the smoke.
            'analysis_profile': 'core',
            'early_stop_start': 99,
        })
        run_configs = run_configs[:1]

    conditions = list(conditions or ('fixed_roles',))
    unknown_conditions = sorted(set(conditions) - set(CONDITION_SPECS))
    if unknown_conditions:
        raise ValueError(
            f'unknown conditions {unknown_conditions}; choose from '
            f'{sorted(CONDITION_SPECS)}')
    if len(set(conditions)) != len(conditions):
        raise ValueError('conditions must not contain duplicates')

    if seed_values is None:
        default_seed_counts = {
            'pilot': 3,
            'targeted_pilot': 5,
        }
        default_seed_count = default_seed_counts.get(mode, 10)
        count = n_seeds if n_seeds is not None else default_seed_count
        seeds = SEED_POOL[:count]
    else:
        seeds = list(seed_values)
    if not seeds or any(seed not in SEED_POOL for seed in seeds):
        raise ValueError(f'seeds must be a non-empty subset of {SEED_POOL}')

    run_plan = [
        (condition, run_config, seed)
        for condition in conditions
        for run_config in run_configs
        for seed in seeds
    ]
    expected_total_runs = len(run_plan)
    # Balance non-divisible inventories without creating an empty final task.
    # The first ``remainder`` partitions receive one extra run.
    start, end, base_runs, remainder = _balanced_partition_bounds(
        expected_total_runs, total_parts, part_idx)
    run_plan = run_plan[start:end]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = exp_name or f'{mode}_{timestamp}'
    root = Path(out_root) / exp_name
    implementation_metadata = collect_implementation_metadata()
    print(f'=== {mode.upper()} matched Task4 condition sweep ===')
    print(f'conditions={conditions}; p={config_base["contra_fraction"]}; '
          f'role={config_base["fixed_role_name"]}; '
          f'partition={part_idx}/{total_parts}; '
          f'total_runs={expected_total_runs}; '
          f'partition_runs={len(run_plan)} ({start}:{end})')
    summaries, failures = [], []
    trained_this_process = 0
    recovered_without_training = 0
    skipped_verified = 0

    for condition, run_config, seed in run_plan:
        config = config_base.copy()
        config.update(CONDITION_SPECS[condition])
        config.update(run_config)
        run_id = f'{_config_id(config)}_seed{seed}'
        marker = root / 'markers' / f'{run_id}.FINISHED'
        run_path = root / 'runs' / f'{run_id}.json'
        checkpoint_path = root / 'models' / f'{run_id}.pth'
        if marker.exists():
            marker_identity = marker.read_text().strip()
            if marker_identity and marker_identity != run_id:
                raise RuntimeError(
                    f'completion marker identity mismatch for {run_id}')
            existing = _validate_run_artifacts(
                root, run_id, seed, config, implementation_metadata)
            summaries.append({
                key: value for key, value in existing.items()
                if key != 'analysis'
            })
            skipped_verified += 1
            print(f'  [skip] {run_id}')
            continue

        # A timeout may occur after both files commit but before the marker.
        # Recover that validated pair without retraining. A one-file remnant is
        # ambiguous and is never silently overwritten.
        if run_path.exists() or checkpoint_path.exists():
            if not (run_path.exists() and checkpoint_path.exists()):
                raise RuntimeError(
                    f'unmarked partial run {run_id}: preserve the output and '
                    'diagnose the missing checkpoint/result before resubmission')
            existing = _validate_run_artifacts(
                root, run_id, seed, config, implementation_metadata)
            _write_completion_marker(marker, run_id)
            summaries.append({
                key: value for key, value in existing.items()
                if key != 'analysis'
            })
            recovered_without_training += 1
            print(f'  [recover] {run_id}')
            continue

        print(f'  [run] {run_id}')
        _set_train_seed(seed)
        writer = SummaryWriter(root / 'tensorboard' / run_id)
        writer.add_text('config', json.dumps(config, indent=2))
        try:
            result = train_single_config(config, writer=writer)
            model = result.pop('_model')
            checkpoint = {
                'schema_version': 6,
                'model_type': (
                    f"{config['model_family']}_{config['objective_routing']}"),
                'state_dict': model.state_dict(),
                'config': config,
                'seed': seed,
                'run_id': run_id,
                'implementation_metadata': implementation_metadata,
            }
            _write_checkpoint(checkpoint_path, checkpoint)

            record = {
                'schema_version': 6,
                'config': config,
                'seed': seed,
                'run_id': run_id,
                'implementation_metadata': implementation_metadata,
                'checkpoint_sha256': _file_sha256(checkpoint_path),
                **result,
            }
            _write_json(run_path, record)
            validated_record = _validate_run_artifacts(
                root, run_id, seed, config, implementation_metadata)
            compact = {
                key: value for key, value in validated_record.items()
                if key != 'analysis'
            }
            summaries.append(compact)
            _write_completion_marker(marker, run_id)
            trained_this_process += 1
            print(
                f"    success: reach={result['reach_success']:.3f} "
                f"hold={result['hold_success']:.3f} both={result['both_success']:.3f}")
            if result.get('controller_interventions_applicable', True):
                print(
                    f"    controller specialisation: role_score="
                    f"{result.get('functional_role_score', float('nan')):.3f} "
                    f"absolute_separation="
                    f"{result.get('absolute_FSI', float('nan')):.3f} "
                    f"double_dissociation="
                    f"{result.get('expected_double_dissociation_gain_fraction', float('nan')):.3f} "
                    f"valid_gains={result.get('FSI_valid_gain_count', 0)}")
                print(
                    f"    pathway diagnostics: "
                    f"CC_aligned={result.get('CC_dependency_role_aligned_score')} "
                    f"whole_module_aligned="
                    f"{result.get('whole_module_dependency_role_aligned_score')} "
                    f"T_via_hold_hand="
                    f"{result.get('trajectory_controller_via_stability_hand_fraction')} "
                    f"S_via_track_hand="
                    f"{result.get('stability_controller_via_trajectory_hand_fraction')}")
                lesion = result['lesion_error_changes']
                print(
                    f"    lesion delta-error: "
                    f"L(reach={lesion['lesion_left']['reach_error_change']:+.6f}, "
                    f"hold={lesion['lesion_left']['hold_error_change']:+.6f}) "
                    f"R(reach={lesion['lesion_right']['reach_error_change']:+.6f}, "
                    f"hold={lesion['lesion_right']['hold_error_change']:+.6f})")
            for lesion_name, values in result.get(
                    'component_probe_reach_lesion_error_changes', {}).items():
                assigned_metrics = values.get('assigned_function_hand_metrics')
                if not assigned_metrics:
                    continue
                print(
                    f"    reach probe {lesion_name} "
                    f"({values['assigned_functional_role']} controller, "
                    f"{values['assigned_function_hand']} hand): "
                    f"trajectory_delta="
                    f"{assigned_metrics['trajectory_deviation']['change']:+.6f} "
                    f"acquisition_delta="
                    f"{assigned_metrics['endpoint_acquisition_error']['change']:+.6f} "
                    f"final_endpoint_delta="
                    f"{assigned_metrics['final_endpoint_error_descriptive']['change']:+.6f}")
            task4 = result['task4_evaluation']
            assigned = task4[config['fixed_role_key']]
            swapped = task4['left_track_right_hold']
            print(
                f"    fixed-role competence: assigned={assigned['both_success']:.3f} "
                f"role-swap transfer={swapped['both_success']:.3f}")
            task4_lesion = result['task4_lesion_error_changes']
            for lesion_name, roles in task4_lesion.items():
                canonical = roles['left_hold_right_track']
                reversed_role = roles['left_track_right_hold']
                print(
                    f"    task4 {lesion_name} delta-error: "
                    f"canonical(track={canonical['tracking_error_change']:+.6f}, "
                    f"hold={canonical['holding_error_change']:+.6f}) "
                    f"reversed(track={reversed_role['tracking_error_change']:+.6f}, "
                    f"hold={reversed_role['holding_error_change']:+.6f})")
            for row in result.get('task4_gain_table', []):
                if row['path'] not in {'readout', 'whole_module'}:
                    continue
                print(
                    f"    task4 gain path={row['path']} gain={row['gain']:.3f} "
                    f"T(track={row['trajectory_controller_tracking_delta_m']:+.6f}, "
                    f"stable={row['trajectory_controller_stability_delta_m']:+.6f}, "
                    f"both={row['trajectory_controller_both_success']:.3f}) "
                    f"S(track={row['stability_controller_tracking_delta_m']:+.6f}, "
                    f"stable={row['stability_controller_stability_delta_m']:+.6f}, "
                    f"both={row['stability_controller_both_success']:.3f}) "
                    f"valid={row['valid_for_specialisation']} "
                    f"DD={row['expected_double_dissociation']}")
            for probe_name, probe in result.get('transfer_probes', {}).items():
                intact_probe = probe['intact']
                print(
                    f"    {probe_name}: intact "
                    f"trajectory={intact_probe['tracking_mean_error']:.6f} "
                    f"stability={intact_probe['holding_mean_error']:.6f} "
                    f"both={intact_probe['both_success']:.3f}")
                for row in probe.get('gain_table', []):
                    if row['path'] not in {'readout', 'whole_module'}:
                        continue
                    print(
                        f"      path={row['path']} gain={row['gain']:.3f} "
                        f"T(track={row['trajectory_controller_tracking_delta_m']:+.6f}, "
                        f"stable={row['trajectory_controller_stability_delta_m']:+.6f}, "
                        f"both={row['trajectory_controller_both_success']:.3f}) "
                        f"S(track={row['stability_controller_tracking_delta_m']:+.6f}, "
                        f"stable={row['stability_controller_stability_delta_m']:+.6f}, "
                        f"both={row['stability_controller_both_success']:.3f}) "
                        f"valid={row['valid_for_specialisation']} "
                        f"DD={row['expected_double_dissociation']}")
            if result.get('controller_interventions_applicable', True):
                print(
                    f"    contra-sanity only: DDI={result['DDI']:.3f} "
                    f"CIraw={result['CI_raw']:.3f} CIrouted={result['CI_routed']:.3f}")
            metrics = result['performance_metrics']
            print(
                f"    perf: endpoint={metrics['endpoint_error']:.6f} "
                f"trajectory={metrics['trajectory_error']:.6f} "
                f"move_var={metrics['movement_variance']:.6f} "
                f"hold_var={metrics['hold_variance']:.6e} "
                f"energy={metrics['energetic_cost']:.6f} CCI={metrics['CCI']:.3f}")
        except Exception as error:
            failures.append({'run_id': run_id, 'error': repr(error)})
            print(f'  [failed] {run_id}: {error}')
            import traceback
            traceback.print_exc()
        finally:
            writer.close()

    _write_json(root / 'parts' / f'part_{part_idx:03d}.json', {
        'mode': mode,
        'command': sys.argv,
        'base_config': config_base,
        'implementation_metadata': implementation_metadata,
        'run_configs': run_configs,
        'conditions': conditions,
        'seeds': seeds,
        'part': part_idx,
        'total_parts': total_parts,
        'expected_total_runs': expected_total_runs,
        'partition_base_runs': base_runs,
        'partition_extra_partitions': remainder,
        'partition_start': start,
        'partition_end': end,
        'expected_partition_runs': len(run_plan),
        'completed_this_process': trained_this_process,
        'recovered_without_training': recovered_without_training,
        'skipped_verified': skipped_verified,
        'completed_partition_runs': len(summaries),
        'summary_source': 'validated_per_run_records',
        'summaries': summaries,
        'failures': failures,
    })
    if failures:
        raise RuntimeError(f'{len(failures)} runs failed; inspect {root / "parts"}')


def _parse_seeds(value):
    return [int(item) for item in value.split(',') if item.strip()]


def _parse_conditions(value):
    return [item.strip() for item in value.split(',') if item.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=[
            'pilot', 'targeted_pilot', 'full', 'remaining_fullgrid'],
        default='pilot')
    parser.add_argument(
        '--conditions', type=_parse_conditions, default=['fixed_roles'],
        help=(
            'Comma-separated condition names: fixed_roles, fixed_balanced, '
            'shared_fullcc, fixed_nocc, monolithic_shared'))
    parser.add_argument('--seeds', type=int, default=None)
    parser.add_argument('--seed-values', type=_parse_seeds, default=None)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--total', type=int, default=1)
    parser.add_argument('--exp-name', default=None)
    parser.add_argument('--out-root', default='results_task4')
    parser.add_argument('--device', default=None)
    parser.add_argument('--cc-mode', choices=['full', 'bottleneck', 'none'], default=None)
    parser.add_argument('--cc-bottleneck-dim', type=int, default=None)
    parser.add_argument('--contra-fraction', type=float, default=None,
                        help='Explicit diagnostic override; all sweep modes default to 0.8')
    parser.add_argument('--task4-role-bias', type=float, default=None,
                        help='P(left stabilises/right follows trajectory); fixed-role default is 1.0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_sweep(
        mode=args.mode,
        n_seeds=args.seeds,
        seed_values=args.seed_values,
        part_idx=args.part,
        total_parts=args.total,
        exp_name=args.exp_name,
        out_root=args.out_root,
        overrides={
            'device': args.device,
            'cc_mode': args.cc_mode,
            'cc_bottleneck_dim': args.cc_bottleneck_dim,
            'contra_fraction': args.contra_fraction,
            'task4_role_bias': args.task4_role_bias,
        },
        smoke=args.smoke,
        conditions=args.conditions,
    )


if __name__ == '__main__':
    main()
