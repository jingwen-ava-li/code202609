from typing import Dict

import numpy as np
import torch


def _gain_label(gain: float) -> str:
    """Stable filename/JSON label without collapsing three-decimal gains."""
    text = f'{float(gain):.6f}'.rstrip('0').rstrip('.')
    if '.' not in text:
        text += '.00'
    else:
        decimals = len(text.split('.', 1)[1])
        text += '0' * max(0, 2 - decimals)
    return f'g{text.replace("-", "m").replace(".", "p")}'


class MotorMetrics:
    """Compute performance, actuation, contralaterality and causal-role metrics."""

    TAU_REACH = 0.02
    TAU_HOLD = 0.01

    @staticmethod
    def compute_contralateral_index(
        c_l: torch.Tensor, c_r: torch.Tensor, n_muscles_per_arm: int = 6
    ) -> Dict[str, torch.Tensor]:
        """Compute the contribution-based contralaterality index (CI)."""
        if c_l.dim() == 2:
            c_l = c_l.unsqueeze(1)
            c_r = c_r.unsqueeze(1)

        eps = 1e-8
        left_idx = list(range(n_muscles_per_arm))
        right_idx = list(range(n_muscles_per_arm, c_l.shape[-1]))

        c_l_to_left = torch.abs(c_l[..., left_idx]).mean()
        c_l_to_right = torch.abs(c_l[..., right_idx]).mean()
        c_r_to_left = torch.abs(c_r[..., left_idx]).mean()
        c_r_to_right = torch.abs(c_r[..., right_idx]).mean()

        ci_left = (c_l_to_right - c_l_to_left) / (c_l_to_right + c_l_to_left + eps)
        ci_right = (c_r_to_left - c_r_to_right) / (c_r_to_left + c_r_to_right + eps)
        ci_total = (ci_left + ci_right) / 2
        return {
            "contralateral_index_left": ci_left,
            "contralateral_index_right": ci_right,
            "contralateral_index_total": ci_total,
        }

    @staticmethod
    def compute_contralaterality_summary(
        model, c_l_tensor: torch.Tensor, c_r_tensor: torch.Tensor
    ) -> Dict[str, float]:
        """Return contribution-based contralaterality fields for sweep scripts."""
        del model
        metrics = MotorMetrics.compute_contralateral_index(c_l_tensor, c_r_tensor)
        return {"contralateral_index": metrics["contralateral_index_total"].item()}

    @staticmethod
    def compute_cocontraction_index(
        actions: torch.Tensor, muscle_pairs: list = None, n_muscles_per_arm: int = 6
    ) -> Dict[str, float]:
        """Compute CCI across antagonist muscle pairs, time, and trials."""
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)
        if actions.dim() != 3:
            raise ValueError('actions must have shape [time, batch, muscles]')

        n_muscles_total = actions.shape[-1]
        if muscle_pairs is None:
            if n_muscles_total == 2 * n_muscles_per_arm:
                muscle_pairs = [
                    (offset + i, offset + i + 1)
                    for offset in (0, n_muscles_per_arm)
                    for i in range(0, n_muscles_per_arm, 2)
                ]
            else:
                muscle_pairs = [(i, i + 1) for i in range(0, n_muscles_total, 2)]

        per_pair = []
        for idx1, idx2 in muscle_pairs:
            a1 = actions[..., idx1]
            a2 = actions[..., idx2]
            per_pair.append((2.0 * torch.minimum(a1, a2) / (a1 + a2 + 1e-8)).mean().item())

        cci_total = float(np.mean(per_pair)) if per_pair else 0.0
        n_pairs_per_arm = n_muscles_per_arm // 2
        if n_muscles_total == 2 * n_muscles_per_arm:
            cci_left = float(np.mean(per_pair[:n_pairs_per_arm]))
            cci_right = float(np.mean(per_pair[n_pairs_per_arm:]))
        else:
            cci_left = cci_right = cci_total
        return {"CCI": cci_total, "CCI_left": cci_left, "CCI_right": cci_right}

    @staticmethod
    def compute_assigned_hand_cocontraction(
        actions: torch.Tensor, half_batch: int, n_muscles_per_arm: int = 6
    ) -> Dict[str, float]:
        """Compute hold CCI only for the disturbed hand in each trial.

        First-half trials use the left-arm muscles and second-half trials use
        the right-arm muscles, matching the hold perturbation assignment.
        """
        if actions.dim() != 3:
            raise ValueError('actions must have shape [time, batch, muscles]')
        if actions.shape[1] != 2 * half_batch:
            raise ValueError('half_batch does not match the action batch dimension')
        if actions.shape[-1] != 2 * n_muscles_per_arm:
            raise ValueError('assigned-hand CCI requires equal left/right muscle blocks')

        left_actions = actions[:, :half_batch, :n_muscles_per_arm]
        right_actions = actions[:, half_batch:, n_muscles_per_arm:]
        left = MotorMetrics.compute_cocontraction_index(
            left_actions, n_muscles_per_arm=n_muscles_per_arm)
        right = MotorMetrics.compute_cocontraction_index(
            right_actions, n_muscles_per_arm=n_muscles_per_arm)
        return {
            'CCI': 0.5 * (left['CCI'] + right['CCI']),
            'CCI_left': left['CCI'],
            'CCI_right': right['CCI'],
        }

    @staticmethod
    def compute_first_hit(errors: torch.Tensor, tolerance: float, dt: float) -> dict:
        """Compute per-trial target acquisition and the inclusive pre-hit mask."""
        hit_matrix = errors <= tolerance
        hit = hit_matrix.any(dim=0)
        first = hit_matrix.float().argmax(dim=0)
        first = torch.where(hit, first, torch.full_like(first, errors.shape[0] - 1))
        steps = torch.arange(errors.shape[0], device=errors.device).unsqueeze(1)
        return {
            'hit': hit,
            'first_index': first,
            'time': (first.float() + 1.0) * dt,
            'pre_hit_mask': steps <= first.unsqueeze(0),
        }

    @staticmethod
    def compute_reach_trajectory_metrics(
        positions: torch.Tensor,
        velocities: torch.Tensor,
        starts: torch.Tensor,
        targets: torch.Tensor,
        first_hit_indices: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute initial direction and trajectory metrics before first hit."""
        direction = targets - starts
        direction_unit = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
        initial_errors = []
        deviations = []
        curvatures = []
        velocity_samples = []
        for trial in range(positions.shape[1]):
            end = int(first_hit_indices[trial].item()) + 1
            path = positions[:end, trial]
            velocity_samples.append(velocities[:end, trial])

            initial_idx = min(end - 1, 4)
            movement = path[initial_idx] - starts[trial]
            cos_angle = torch.dot(movement, direction_unit[trial]) / (torch.norm(movement) + 1e-8)
            initial_errors.append(torch.acos(torch.clamp(cos_angle, -1.0, 1.0)))

            offset = path - starts[trial]
            deviation = (
                offset[:, 0] * direction_unit[trial, 1]
                - offset[:, 1] * direction_unit[trial, 0]
            )
            deviations.append(torch.abs(deviation).mean())

            path_length = (
                torch.norm(path[1:] - path[:-1], dim=-1).sum()
                if path.shape[0] > 1
                else torch.tensor(0.0, device=path.device)
            )
            straight = torch.norm(path[-1] - starts[trial])
            curvatures.append(path_length / (straight + 1e-8) - 1.0)

        velocity_values = torch.cat(velocity_samples, dim=0)
        movement_variance = (
            velocity_values.var(dim=0).sum().item()
            if velocity_values.shape[0] > 1 else 0.0
        )
        return {
            'initial_direction_error_rad': torch.stack(initial_errors).mean().item(),
            'trajectory_deviation': torch.stack(deviations).mean().item(),
            'path_curvature': torch.stack(curvatures).mean().item(),
            'movement_variance': movement_variance,
        }

    @staticmethod
    def compute_hold_recovery_metrics(
        positions: torch.Tensor,
        errors: torch.Tensor,
        targets: torch.Tensor,
        force_directions: torch.Tensor,
        recovery_mask: torch.Tensor,
        pulse_mask: torch.Tensor,
        tolerance: float,
        dt: float,
        report_tolerances=(0.005, 0.01, 0.02),
    ) -> dict:
        """Compute disturbed-hand pulse and recovery metrics per trial."""
        residuals, settling, variances, overshoots = [], [], [], []
        for trial in range(errors.shape[1]):
            recovery_indices = torch.where(recovery_mask[:, trial])[0]
            if recovery_indices.numel() == 0:
                recovery_indices = torch.tensor(
                    [errors.shape[0] - 1], device=errors.device)
            recovery_errors = errors[recovery_indices, trial]
            residuals.append(
                recovery_errors[-min(5, recovery_errors.numel()):].mean())

            settling_time = recovery_errors.numel() * dt
            for offset in range(recovery_errors.numel()):
                if torch.all(recovery_errors[offset:] <= tolerance):
                    settling_time = (offset + 1) * dt
                    break
            settling.append(torch.as_tensor(settling_time, device=errors.device))

            recovery_positions = positions[recovery_indices, trial]
            variances.append(
                recovery_positions.var(dim=0).sum()
                if recovery_positions.shape[0] > 1
                else torch.tensor(0.0, device=errors.device)
            )
            force = force_directions[trial]
            force_unit = force / (torch.norm(force) + 1e-8)
            displacement = recovery_positions - targets[trial]
            overshoots.append(
                torch.relu(-(displacement * force_unit).sum(dim=-1)).max())

        residual = torch.stack(residuals)
        result = {
            'mean_error': errors.mean().item(),
            'integrated_error': (errors.sum(dim=0) * dt).mean().item(),
            'peak_displacement': errors.max(dim=0).values.mean().item(),
            'pulse_error': errors[pulse_mask].mean().item() if pulse_mask.any() else 0.0,
            'residual_error': residual.mean().item(),
            'settling_time': torch.stack(settling).mean().item(),
            'position_variance': positions.var(dim=0).sum(dim=-1).mean().item(),
            'post_pulse_variance': torch.stack(variances).mean().item(),
            'overshoot': torch.stack(overshoots).mean().item(),
            'success': (residual <= tolerance).float().mean().item(),
            'success_per_trial': residual <= tolerance,
        }
        for report_tolerance in report_tolerances:
            millimetres = int(round(float(report_tolerance) * 1000.0))
            result[f'success_{millimetres}mm'] = (
                residual <= float(report_tolerance)).float().mean().item()
        return result

    @staticmethod
    def compute_causal_effects(conditions: dict) -> dict:
        """Report signed impairment relative to intact without assigning roles."""
        intact = conditions['intact']
        lower_better = [
            'reach_min_distance_left', 'reach_min_distance_right',
            'reach_final_endpoint_error_left', 'reach_final_endpoint_error_right',
            'reach_initial_direction_error_rad_left', 'reach_initial_direction_error_rad_right',
            'reach_trajectory_deviation_left', 'reach_trajectory_deviation_right',
            'reach_movement_variance_left', 'reach_movement_variance_right',
            'hold_mean_error_left', 'hold_mean_error_right',
            'hold_peak_displacement_left', 'hold_peak_displacement_right',
            'hold_residual_error_left', 'hold_residual_error_right',
            'hold_position_variance_left', 'hold_position_variance_right',
            'hold_post_pulse_variance_left', 'hold_post_pulse_variance_right',
        ]
        effects = {}
        for name, values in conditions.items():
            if name == 'intact':
                continue
            effect = {key: values[key] - intact[key] for key in lower_better}
            for key in ('reach_success', 'hold_success', 'both_success'):
                effect[key] = intact[key] - values[key]
            effects[name] = effect
        return effects

    @staticmethod
    def compute_graded_functional_specialisation(
        conditions: dict,
        gains=(0.7, 0.8, 0.9),
        metric_pairs=None,
        epsilon: float = 1e-8,
    ) -> dict:
        """Measure controller demand preferences under mild graded interventions.

        This adapts the historical FSI: each controller receives a signed
        trajectory-versus-stability preference (SI), and FSI is half the
        distance between the two preferences. Metrics are already restricted
        upstream to the active reaching hand and disturbed holding hand.
        """
        if metric_pairs is None:
            metric_pairs = {
                'trajectory_residual': (
                    'reach_trajectory_deviation', 'hold_residual_error'),
                'trajectory_peak': (
                    'reach_trajectory_deviation', 'hold_peak_displacement'),
                'initial_peak': (
                    'reach_initial_direction_error_rad', 'hold_peak_displacement'),
            }

        def paired_mean(condition, metric):
            return 0.5 * (
                float(condition[f'{metric}_left'])
                + float(condition[f'{metric}_right'])
            )

        intact = conditions['intact']
        requested_gains = sorted({float(gain) for gain in gains})
        output = {
            'schema_version': 1,
            'available': False,
            'gains': requested_gains,
            'pairs': {},
            'primary_pair': 'trajectory_residual',
        }

        for pair_name, metrics in metric_pairs.items():
            reach_metric, hold_metric = metrics
            intact_reach = paired_mean(intact, reach_metric)
            intact_hold = paired_mean(intact, hold_metric)
            by_gain = {}

            for gain in requested_gains:
                label = _gain_label(gain)
                names = {
                    side: f'graded_both_{side}_{label}'
                    for side in ('left', 'right')
                }
                if any(name not in conditions for name in names.values()):
                    continue

                entry = {
                    'gain': gain,
                    'intact_reach': intact_reach,
                    'intact_hold': intact_hold,
                }
                preferences = {}
                for side, condition_name in names.items():
                    condition = conditions[condition_name]
                    delta_reach = max(
                        0.0, paired_mean(condition, reach_metric) - intact_reach)
                    delta_hold = max(
                        0.0, paired_mean(condition, hold_metric) - intact_hold)
                    norm_reach = delta_reach / (abs(intact_reach) + epsilon)
                    norm_hold = delta_hold / (abs(intact_hold) + epsilon)
                    preference = (
                        (norm_reach - norm_hold)
                        / (norm_reach + norm_hold + epsilon)
                    )
                    preferences[side] = preference
                    entry.update({
                        f'delta_reach_{side}': delta_reach,
                        f'delta_hold_{side}': delta_hold,
                        f'delta_reach_norm_{side}': norm_reach,
                        f'delta_hold_norm_{side}': norm_hold,
                        f'effect_magnitude_{side}': norm_reach + norm_hold,
                        f'SI_{side}': preference,
                    })

                fsi = abs(preferences['left'] - preferences['right']) / 2.0
                reciprocal = preferences['left'] * preferences['right'] < 0.0
                entry.update({
                    'FSI': fsi,
                    'reciprocal_sign': reciprocal,
                    'reciprocal_FSI': fsi if reciprocal else 0.0,
                })
                by_gain[label] = entry

            entries = list(by_gain.values())
            reciprocal_entries = [entry for entry in entries if entry['reciprocal_sign']]
            left_signs = {
                1 if entry['SI_left'] > 0 else -1
                for entry in reciprocal_entries
            }
            pair_output = {
                'reach_metric': reach_metric,
                'hold_metric': hold_metric,
                'by_gain': by_gain,
                'mean_FSI': (
                    float(np.mean([entry['FSI'] for entry in entries]))
                    if entries else None
                ),
                'mean_reciprocal_FSI': (
                    float(np.mean([entry['reciprocal_FSI'] for entry in entries]))
                    if entries else None
                ),
                'reciprocal_gain_fraction': (
                    len(reciprocal_entries) / len(entries) if entries else None
                ),
                'evaluated_gain_count': len(entries),
                'reciprocal_gain_count': len(reciprocal_entries),
                'reciprocal_role_consistent': (
                    len(reciprocal_entries) >= 2 and len(left_signs) == 1
                ),
            }
            output['pairs'][pair_name] = pair_output
            output['available'] = output['available'] or bool(entries)

        if output['primary_pair'] not in output['pairs'] and output['pairs']:
            output['primary_pair'] = next(iter(output['pairs']))
        primary = output['pairs'].get(output['primary_pair'], {})
        output['primary'] = {
            key: primary.get(key)
            for key in (
                'reach_metric', 'hold_metric', 'mean_FSI',
                'mean_reciprocal_FSI', 'reciprocal_gain_fraction',
                'evaluated_gain_count', 'reciprocal_gain_count',
                'reciprocal_role_consistent',
            )
        }
        return output

    @staticmethod
    def compute_task4_graded_functional_specialisation(
        conditions: dict,
        gains=(0.7, 0.8, 0.9),
        epsilon: float = 1e-8,
    ) -> dict:
        """Measure tracking-versus-holding preferences within simultaneous Task4.

        Each controller is mildly attenuated while both fixed hand-role
        assignments are replayed with identical mechanics.  Tracking and
        holding impairments are normalised to the role-matched intact error,
        then averaged across roles before computing controller preference.
        """
        roles = ('left_hold_right_track', 'left_track_right_hold')

        requested_gains = sorted({float(gain) for gain in gains})
        output = {
            'schema_version': 2,
            'available': False,
            'gains': requested_gains,
            'tracking_metric': 'tracking_mean_error_m',
            'holding_metric': 'holding_mean_error_m',
            'demand_SI_sign_convention': (
                'positive=tracking-selective; negative=holding-selective'),
            'reciprocal_definition': (
                'left and right demand SI have opposite signs at the same gain'),
            'hand_SI_sign_convention': (
                'positive=right-hand-selective; negative=left-hand-selective'),
            'role_aggregation': (
                'mean threshold-normalised impairment across matched fixed roles'),
            'by_gain': {},
        }
        intact = conditions.get('intact')
        if not intact or any(role not in intact for role in roles):
            return output

        for gain in requested_gains:
            label = _gain_label(gain)
            names = {
                side: f'graded_both_{side}_{label}'
                for side in ('left', 'right')
            }
            if any(name not in conditions for name in names.values()):
                continue

            entry = {'gain': gain, 'controllers': {}}
            preferences = {}
            for side, condition_name in names.items():
                per_role = {}
                tracking_norm, holding_norm = [], []
                tracking_raw, holding_raw = [], []
                for role in roles:
                    intact_role = intact[role]
                    intervened_role = conditions[condition_name][role]
                    delta_tracking = max(
                        0.0,
                        float(intervened_role['tracking_mean_error'])
                        - float(intact_role['tracking_mean_error']),
                    )
                    delta_holding = max(
                        0.0,
                        float(intervened_role['holding_mean_error'])
                        - float(intact_role['holding_mean_error']),
                    )
                    norm_tracking = delta_tracking / (
                        abs(float(intact_role['tracking_mean_error'])) + epsilon)
                    norm_holding = delta_holding / (
                        abs(float(intact_role['holding_mean_error'])) + epsilon)
                    tracking_raw.append(delta_tracking)
                    holding_raw.append(delta_holding)
                    tracking_norm.append(norm_tracking)
                    holding_norm.append(norm_holding)
                    per_role[role] = {
                        'delta_tracking': delta_tracking,
                        'delta_holding': delta_holding,
                        'delta_tracking_normalised': norm_tracking,
                        'delta_holding_normalised': norm_holding,
                    }

                mean_tracking = float(np.mean(tracking_norm))
                mean_holding = float(np.mean(holding_norm))
                preference = (
                    (mean_tracking - mean_holding)
                    / (mean_tracking + mean_holding + epsilon)
                )
                canonical = per_role['left_hold_right_track']
                reversed_role = per_role['left_track_right_hold']
                # Canonical maps right hand -> tracking and left hand -> holding;
                # reversed maps left hand -> tracking and right hand -> holding.
                # This orthogonal decomposition prevents anatomical hand
                # laterality from being mislabelled demand specialisation.
                right_hand_raw = float(np.mean([
                    canonical['delta_tracking'],
                    reversed_role['delta_holding'],
                ]))
                left_hand_raw = float(np.mean([
                    canonical['delta_holding'],
                    reversed_role['delta_tracking'],
                ]))
                right_hand_norm = float(np.mean([
                    canonical['delta_tracking_normalised'],
                    reversed_role['delta_holding_normalised'],
                ]))
                left_hand_norm = float(np.mean([
                    canonical['delta_holding_normalised'],
                    reversed_role['delta_tracking_normalised'],
                ]))
                hand_preference = (
                    (right_hand_norm - left_hand_norm)
                    / (right_hand_norm + left_hand_norm + epsilon)
                )
                preferences[side] = preference
                entry['controllers'][side] = {
                    'per_role': per_role,
                    'delta_tracking_mean': float(np.mean(tracking_raw)),
                    'delta_holding_mean': float(np.mean(holding_raw)),
                    'delta_tracking_normalised_mean': mean_tracking,
                    'delta_holding_normalised_mean': mean_holding,
                    'effect_magnitude': mean_tracking + mean_holding,
                    'SI': preference,
                    'delta_right_hand_mean': right_hand_raw,
                    'delta_left_hand_mean': left_hand_raw,
                    'delta_right_hand_normalised_mean': right_hand_norm,
                    'delta_left_hand_normalised_mean': left_hand_norm,
                    'hand_effect_magnitude': right_hand_norm + left_hand_norm,
                    'hand_SI_right_minus_left': hand_preference,
                    'dominant_selectivity_axis': (
                        'hand' if abs(hand_preference) > abs(preference)
                        else 'demand'),
                }

            fsi = abs(preferences['left'] - preferences['right']) / 2.0
            reciprocal = preferences['left'] * preferences['right'] < 0.0
            hand_preferences = {
                side: entry['controllers'][side]['hand_SI_right_minus_left']
                for side in ('left', 'right')
            }
            hand_separation = abs(
                hand_preferences['left'] - hand_preferences['right']) / 2.0
            opposite_hand_preference = (
                hand_preferences['left'] * hand_preferences['right'] < 0.0)
            entry.update({
                'FSI': fsi,
                'reciprocal_sign': reciprocal,
                'reciprocal_FSI': fsi if reciprocal else 0.0,
                'hand_separation_index': hand_separation,
                'opposite_hand_preference': opposite_hand_preference,
            })
            output['by_gain'][label] = entry

        entries = list(output['by_gain'].values())
        reciprocal_entries = [entry for entry in entries if entry['reciprocal_sign']]
        left_signs = {
            1 if entry['controllers']['left']['SI'] > 0 else -1
            for entry in reciprocal_entries
        }
        opposite_hand_entries = [
            entry for entry in entries if entry['opposite_hand_preference']]
        controller_entries = [
            entry['controllers'][side]
            for entry in entries for side in ('left', 'right')]
        output.update({
            'available': bool(entries),
            'mean_FSI': (
                float(np.mean([entry['FSI'] for entry in entries]))
                if entries else None),
            'mean_reciprocal_FSI': (
                float(np.mean([entry['reciprocal_FSI'] for entry in entries]))
                if entries else None),
            'reciprocal_gain_fraction': (
                len(reciprocal_entries) / len(entries) if entries else None),
            'evaluated_gain_count': len(entries),
            'reciprocal_gain_count': len(reciprocal_entries),
            'reciprocal_role_consistent': (
                len(reciprocal_entries) >= 2 and len(left_signs) == 1),
            'hand_selectivity': {
                'mean_abs_hand_SI_left': (
                    float(np.mean([
                        abs(entry['controllers']['left']['hand_SI_right_minus_left'])
                        for entry in entries])) if entries else None),
                'mean_abs_hand_SI_right': (
                    float(np.mean([
                        abs(entry['controllers']['right']['hand_SI_right_minus_left'])
                        for entry in entries])) if entries else None),
                'mean_hand_separation_index': (
                    float(np.mean([
                        entry['hand_separation_index'] for entry in entries]))
                    if entries else None),
                'opposite_hand_preference_gain_fraction': (
                    len(opposite_hand_entries) / len(entries) if entries else None),
                'hand_dominant_controller_gain_fraction': (
                    sum(
                        controller['dominant_selectivity_axis'] == 'hand'
                        for controller in controller_entries)
                    / len(controller_entries) if controller_entries else None),
            },
        })
        return output

    @staticmethod
    def compute_fixed_role_functional_specialisation(
        conditions: dict,
        assigned_role: str,
        trajectory_controller: str,
        stability_controller: str,
        gains=(0.95, 0.975, 0.99),
        trajectory_scale: float = 0.02,
        stability_scale: float = 0.01,
        trajectory_hand: str = 'right',
        stability_hand: str = 'left',
        min_intact_both_success: float = 0.8,
        min_competence_ratio: float = 0.8,
        max_force_cap_fraction: float = 0.01,
        min_effect_magnitude: float = 0.05,
        epsilon: float = 1e-8,
    ) -> dict:
        """Measure fixed-role specialisation with pathway-resolved perturbations.

        Readout-only local sensitivity is the primary motor-function endpoint.
        CC-only and combined attenuation are reported separately as
        communication and whole-module dependence. Signed error changes are
        converted to slopes per unit attenuation and are only aggregated while
        the intervention remains in a competent, unsaturated operating regime.
        """
        if trajectory_controller == stability_controller:
            raise ValueError('trajectory and stability controllers must differ')
        if {trajectory_controller, stability_controller} != {'left', 'right'}:
            raise ValueError('fixed-role controllers must be left and right')
        if {trajectory_hand, stability_hand} != {'left', 'right'}:
            raise ValueError('fixed-role functional hands must be left and right')
        if trajectory_scale <= 0.0 or stability_scale <= 0.0:
            raise ValueError('functional normalisation scales must be positive')
        if not 0.0 <= min_intact_both_success <= 1.0:
            raise ValueError('min_intact_both_success must be in [0, 1]')
        if not 0.0 <= min_competence_ratio <= 1.0:
            raise ValueError('min_competence_ratio must be in [0, 1]')
        if not 0.0 <= max_force_cap_fraction <= 1.0:
            raise ValueError('max_force_cap_fraction must be in [0, 1]')
        if min_effect_magnitude < 0.0:
            raise ValueError('min_effect_magnitude must be non-negative')

        requested_gains = sorted({float(gain) for gain in gains})
        output = {
            'schema_version': 2,
            'available': False,
            'assigned_role': assigned_role,
            'trajectory_controller': trajectory_controller,
            'stability_controller': stability_controller,
            'trajectory_hand': trajectory_hand,
            'stability_hand': stability_hand,
            'trajectory_metric': 'tracking_mean_error_m',
            'stability_metric': 'holding_mean_error_m',
            'trajectory_normalisation_m': float(trajectory_scale),
            'stability_normalisation_m': float(stability_scale),
            'primary_path': 'readout',
            'path_interpretation': {
                'readout': 'motor-output functional sensitivity',
                'cc': 'communication dependence',
                'whole_module': 'combined communication and motor dependence',
            },
            'sensitivity_definition': (
                '(intervened_error-intact_error)/(tolerance*(1-gain))'),
            'signed_effects': True,
            'SI_sign_convention': (
                'positive=trajectory-sensitive; negative=stability-sensitive'),
            'role_aligned_sign_convention': (
                'positive=assigned Trajectory/Stability controller roles'),
            'validity_gates': {
                'minimum_intact_both_success': float(
                    min_intact_both_success),
                'minimum_both_success_fraction_of_intact': float(
                    min_competence_ratio),
                'maximum_force_cap_fraction': float(max_force_cap_fraction),
                'minimum_controller_effect_magnitude': float(
                    min_effect_magnitude),
            },
            'gains': requested_gains,
            'paths': {},
            'readout_hand_pathways': {},
            'single_role_function_hand_confound': True,
        }
        intact_conditions = conditions.get('intact', {})
        if assigned_role not in intact_conditions:
            return output
        intact = intact_conditions[assigned_role]

        intact_both = float(intact.get('both_success', 0.0))
        competence_floor = intact_both * min_competence_ratio

        def controller_sensitivity(intervened, gain):
            attenuation = 1.0 - gain
            if attenuation <= 0.0:
                raise ValueError('functional intervention gains must be below 1')
            delta_trajectory = (
                float(intervened['tracking_mean_error'])
                - float(intact['tracking_mean_error']))
            delta_stability = (
                float(intervened['holding_mean_error'])
                - float(intact['holding_mean_error']))
            norm_trajectory = delta_trajectory / trajectory_scale
            norm_stability = delta_stability / stability_scale
            sensitivity_trajectory = norm_trajectory / attenuation
            sensitivity_stability = norm_stability / attenuation
            effect_magnitude = (
                abs(sensitivity_trajectory) + abs(sensitivity_stability))
            preference = (
                (sensitivity_trajectory - sensitivity_stability)
                / (effect_magnitude + epsilon))
            exclusion_reasons = []
            if intact_both < min_intact_both_success:
                exclusion_reasons.append('intact_incompetent')
            if float(intervened.get('both_success', 0.0)) < competence_floor:
                exclusion_reasons.append('competence_collapse')
            if float(intervened.get(
                    'coupling_force_cap_fraction', 0.0)) > max_force_cap_fraction:
                exclusion_reasons.append('force_cap_saturation')
            if effect_magnitude < min_effect_magnitude:
                exclusion_reasons.append('insufficient_effect')
            return {
                'delta_trajectory': delta_trajectory,
                'delta_stability': delta_stability,
                'delta_trajectory_normalised': norm_trajectory,
                'delta_stability_normalised': norm_stability,
                'trajectory_local_sensitivity': sensitivity_trajectory,
                'stability_local_sensitivity': sensitivity_stability,
                'effect_magnitude': effect_magnitude,
                'SI': preference,
                'both_success': float(intervened.get('both_success', 0.0)),
                'coupling_force_cap_fraction': float(intervened.get(
                    'coupling_force_cap_fraction', 0.0)),
                'valid_operating_regime': not exclusion_reasons,
                'exclusion_reasons': exclusion_reasons,
            }

        def compute_path(path_name, condition_prefix):
            path_output = {'by_gain': {}}
            for gain in requested_gains:
                label = _gain_label(gain)
                names = {
                    side: f'{condition_prefix}_{side}_{label}'
                    for side in ('left', 'right')
                }
                if any(name not in conditions for name in names.values()):
                    continue
                entry = {'gain': gain, 'controllers': {}}
                for side, name in names.items():
                    role_results = conditions[name]
                    if assigned_role not in role_results:
                        continue
                    entry['controllers'][side] = controller_sensitivity(
                        role_results[assigned_role], gain)
                if set(entry['controllers']) != {'left', 'right'}:
                    continue

                trajectory_values = entry['controllers'][trajectory_controller]
                stability_values = entry['controllers'][stability_controller]
                trajectory_si = trajectory_values['SI']
                stability_si = stability_values['SI']
                matrix = {
                    'trajectory_controller': {
                        'trajectory': trajectory_values[
                            'trajectory_local_sensitivity'],
                        'stability': trajectory_values[
                            'stability_local_sensitivity'],
                    },
                    'stability_controller': {
                        'trajectory': stability_values[
                            'trajectory_local_sensitivity'],
                        'stability': stability_values[
                            'stability_local_sensitivity'],
                    },
                }
                numerator = (
                    matrix['trajectory_controller']['trajectory']
                    - matrix['trajectory_controller']['stability']
                    + matrix['stability_controller']['stability']
                    - matrix['stability_controller']['trajectory'])
                denominator = sum(
                    abs(value)
                    for controller in matrix.values()
                    for value in controller.values())
                role_aligned = numerator / (denominator + epsilon)
                fsi = abs(trajectory_si - stability_si) / 2.0
                valid = all(
                    values['valid_operating_regime']
                    for values in entry['controllers'].values())
                expected = (
                    valid
                    and matrix['trajectory_controller']['trajectory'] > 0.0
                    and matrix['trajectory_controller']['trajectory']
                    > matrix['trajectory_controller']['stability']
                    and matrix['stability_controller']['stability'] > 0.0
                    and matrix['stability_controller']['stability']
                    > matrix['stability_controller']['trajectory'])
                entry.update({
                    'causal_sensitivity_matrix': matrix,
                    'FSI': fsi,
                    'role_aligned_FSI': role_aligned,
                    'expected_double_dissociation': expected,
                    'reciprocal_sign': trajectory_si * stability_si < 0.0,
                    'reciprocal_FSI': fsi if expected else 0.0,
                    'valid_for_specialisation': valid,
                })
                path_output['by_gain'][label] = entry

            entries = list(path_output['by_gain'].values())
            valid_entries = [
                entry for entry in entries
                if entry['valid_for_specialisation']]
            aligned_entries = [
                entry for entry in valid_entries
                if entry['expected_double_dissociation']]
            path_output.update({
                'path': path_name,
                'conditions_available': bool(entries),
                'available': bool(valid_entries),
                'mean_FSI': (
                    float(np.mean([entry['FSI'] for entry in valid_entries]))
                    if valid_entries else None),
                'mean_absolute_FSI': (
                    float(np.mean([entry['FSI'] for entry in valid_entries]))
                    if valid_entries else None),
                'mean_role_aligned_FSI': (
                    float(np.mean([
                        entry['role_aligned_FSI'] for entry in valid_entries]))
                    if valid_entries else None),
                'mean_reciprocal_FSI': (
                    float(np.mean([
                        entry['reciprocal_FSI'] for entry in valid_entries]))
                    if valid_entries else None),
                'reciprocal_gain_fraction': (
                    len(aligned_entries) / len(valid_entries)
                    if valid_entries else None),
                'expected_double_dissociation_gain_fraction': (
                    len(aligned_entries) / len(valid_entries)
                    if valid_entries else None),
                'evaluated_gain_count': len(entries),
                'valid_gain_count': len(valid_entries),
                'excluded_gain_count': len(entries) - len(valid_entries),
                'reciprocal_gain_count': len(aligned_entries),
                'reciprocal_role_consistent': (
                    len(valid_entries) >= 2
                    and len(aligned_entries) == len(valid_entries)),
            })
            return path_output

        output['paths'] = {
            'readout': compute_path('readout', 'graded_readout'),
            'cc': compute_path('cc', 'graded_cc'),
            'whole_module': compute_path('whole_module', 'graded_both'),
        }

        # Resolve which motor half each controller uses for each function. This
        # diagnostic reveals indirect strategies such as improving tracking by
        # moving the nominal holding hand.
        hand_pathways = {'by_gain': {}}
        for gain in requested_gains:
            label = _gain_label(gain)
            entry = {'gain': gain, 'controllers': {}}
            complete = True
            for side in ('left', 'right'):
                entry['controllers'][side] = {}
                for hand in ('left', 'right'):
                    name = f'graded_readout_{side}_to_{hand}_hand_{label}'
                    if (name not in conditions
                            or assigned_role not in conditions[name]):
                        complete = False
                        break
                    entry['controllers'][side][hand] = controller_sensitivity(
                        conditions[name][assigned_role], gain)
                if not complete:
                    break
            if not complete:
                continue

            t_paths = entry['controllers'][trajectory_controller]
            s_paths = entry['controllers'][stability_controller]
            t_direct = abs(t_paths[trajectory_hand][
                'trajectory_local_sensitivity'])
            t_cross = abs(t_paths[stability_hand][
                'trajectory_local_sensitivity'])
            s_direct = abs(s_paths[stability_hand][
                'stability_local_sensitivity'])
            s_cross = abs(s_paths[trajectory_hand][
                'stability_local_sensitivity'])
            t_valid = all(
                t_paths[hand]['valid_operating_regime']
                for hand in (trajectory_hand, stability_hand))
            s_valid = all(
                s_paths[hand]['valid_operating_regime']
                for hand in (trajectory_hand, stability_hand))
            entry['cross_hand_strategy'] = {
                'trajectory_controller_via_stability_hand_fraction': (
                    t_cross / (t_direct + t_cross + epsilon)),
                'trajectory_controller_fraction_valid': t_valid,
                'stability_controller_via_trajectory_hand_fraction': (
                    s_cross / (s_direct + s_cross + epsilon)),
                'stability_controller_fraction_valid': s_valid,
            }
            hand_pathways['by_gain'][label] = entry

        hand_entries = list(hand_pathways['by_gain'].values())
        valid_trajectory_entries = [
            entry for entry in hand_entries
            if entry['cross_hand_strategy'][
                'trajectory_controller_fraction_valid']]
        valid_stability_entries = [
            entry for entry in hand_entries
            if entry['cross_hand_strategy'][
                'stability_controller_fraction_valid']]
        hand_pathways.update({
            'available': bool(hand_entries),
            'trajectory_controller_valid_gain_count': len(
                valid_trajectory_entries),
            'stability_controller_valid_gain_count': len(
                valid_stability_entries),
            'mean_trajectory_controller_via_stability_hand_fraction': (
                float(np.mean([
                    entry['cross_hand_strategy'][
                        'trajectory_controller_via_stability_hand_fraction']
                    for entry in valid_trajectory_entries]))
                if valid_trajectory_entries else None),
            'mean_stability_controller_via_trajectory_hand_fraction': (
                float(np.mean([
                    entry['cross_hand_strategy'][
                        'stability_controller_via_trajectory_hand_fraction']
                    for entry in valid_stability_entries]))
                if valid_stability_entries else None),
        })
        output['readout_hand_pathways'] = hand_pathways

        primary = output['paths']['readout']
        output['available'] = primary['available']
        output['by_gain'] = primary['by_gain']
        for key in (
                'mean_FSI', 'mean_absolute_FSI', 'mean_role_aligned_FSI',
                'mean_reciprocal_FSI',
                'reciprocal_gain_fraction',
                'expected_double_dissociation_gain_fraction',
                'evaluated_gain_count', 'valid_gain_count',
                'excluded_gain_count', 'reciprocal_gain_count',
                'reciprocal_role_consistent'):
            output[key] = primary[key]
        return output

    @staticmethod
    def compute_hand_balanced_functional_specialisation(
        conditions: dict,
        trajectory_controller: str,
        stability_controller: str,
        gains=(0.95, 0.975, 0.99),
        trajectory_scale: float = 0.02,
        stability_scale: float = 0.01,
        min_intact_both_success: float = 0.8,
        min_competence_ratio: float = 0.8,
        max_force_cap_fraction: float = 0.01,
        min_effect_magnitude: float = 0.05,
        epsilon: float = 1e-8,
    ) -> dict:
        """Aggregate fixed-functional sensitivity across both physical hand roles.

        Each role first passes through the established signed, tolerance-scaled
        fixed-role metric. A gain is valid only when both controllers remain in
        the valid operating regime in both matched hand-role evaluations. The
        expected double dissociation must also hold separately in both roles,
        preventing a favourable average from hiding a role-specific reversal.
        """
        role_hands = {
            'left_hold_right_track': ('right', 'left'),
            'left_track_right_hold': ('left', 'right'),
        }
        requested_gains = sorted({float(gain) for gain in gains})
        per_role = {
            role: MotorMetrics.compute_fixed_role_functional_specialisation(
                conditions,
                assigned_role=role,
                trajectory_controller=trajectory_controller,
                stability_controller=stability_controller,
                gains=requested_gains,
                trajectory_scale=trajectory_scale,
                stability_scale=stability_scale,
                trajectory_hand=hands[0],
                stability_hand=hands[1],
                min_intact_both_success=min_intact_both_success,
                min_competence_ratio=min_competence_ratio,
                max_force_cap_fraction=max_force_cap_fraction,
                min_effect_magnitude=min_effect_magnitude,
                epsilon=epsilon,
            )
            for role, hands in role_hands.items()
        }
        output = {
            'schema_version': 3,
            'available': False,
            'assigned_role': 'both_matched_hand_roles',
            'evaluated_roles': list(role_hands),
            'trajectory_controller': trajectory_controller,
            'stability_controller': stability_controller,
            'trajectory_hand': 'balanced_across_left_and_right',
            'stability_hand': 'balanced_across_left_and_right',
            'trajectory_metric': 'tracking_mean_error_m',
            'stability_metric': 'holding_mean_error_m',
            'trajectory_normalisation_m': float(trajectory_scale),
            'stability_normalisation_m': float(stability_scale),
            'primary_path': 'readout',
            'sensitivity_definition': (
                '(intervened_error-intact_error)/(tolerance*(1-gain))'),
            'role_aggregation': (
                'equal mean across matched canonical and reversed hand roles; '
                'validity and expected direction required separately in both'),
            'single_role_function_hand_confound': False,
            'validity_gates': next(iter(per_role.values()))['validity_gates'],
            'gains': requested_gains,
            'per_role': per_role,
            'paths': {},
            'readout_hand_pathways': {
                'available': False,
                'reason': (
                    'hand-balanced demand/hand decomposition is reported per gain '
                    'under each causal path'),
            },
        }

        def mean_field(role_entries, controller, field):
            return float(np.mean([
                entry['controllers'][controller][field]
                for entry in role_entries.values()
            ]))

        def aggregate_path(path_name):
            path_output = {'path': path_name, 'by_gain': {}}
            for gain in requested_gains:
                label = _gain_label(gain)
                role_entries = {}
                for role, role_output in per_role.items():
                    entry = role_output.get('paths', {}).get(
                        path_name, {}).get('by_gain', {}).get(label)
                    if entry is None:
                        break
                    role_entries[role] = entry
                if set(role_entries) != set(role_hands):
                    continue

                controllers = {}
                for controller in ('left', 'right'):
                    exclusion_reasons = sorted({
                        f'{role}:{reason}'
                        for role, role_entry in role_entries.items()
                        for reason in role_entry['controllers'][controller][
                            'exclusion_reasons']
                    })
                    sensitivity_trajectory = mean_field(
                        role_entries, controller, 'trajectory_local_sensitivity')
                    sensitivity_stability = mean_field(
                        role_entries, controller, 'stability_local_sensitivity')
                    effect_magnitude = abs(
                        sensitivity_trajectory) + abs(sensitivity_stability)
                    controllers[controller] = {
                        'delta_trajectory': mean_field(
                            role_entries, controller, 'delta_trajectory'),
                        'delta_stability': mean_field(
                            role_entries, controller, 'delta_stability'),
                        'delta_trajectory_normalised': mean_field(
                            role_entries, controller, 'delta_trajectory_normalised'),
                        'delta_stability_normalised': mean_field(
                            role_entries, controller, 'delta_stability_normalised'),
                        'trajectory_local_sensitivity': sensitivity_trajectory,
                        'stability_local_sensitivity': sensitivity_stability,
                        'effect_magnitude': effect_magnitude,
                        'SI': (
                            (sensitivity_trajectory - sensitivity_stability)
                            / (effect_magnitude + epsilon)),
                        'both_success': min(
                            float(entry['controllers'][controller]['both_success'])
                            for entry in role_entries.values()),
                        'coupling_force_cap_fraction': max(
                            float(entry['controllers'][controller][
                                'coupling_force_cap_fraction'])
                            for entry in role_entries.values()),
                        'valid_operating_regime': not exclusion_reasons,
                        'exclusion_reasons': exclusion_reasons,
                        'per_role': {
                            role: role_entry['controllers'][controller]
                            for role, role_entry in role_entries.items()
                        },
                    }

                trajectory = controllers[trajectory_controller]
                stability = controllers[stability_controller]
                matrix = {
                    'trajectory_controller': {
                        'trajectory': trajectory['trajectory_local_sensitivity'],
                        'stability': trajectory['stability_local_sensitivity'],
                    },
                    'stability_controller': {
                        'trajectory': stability['trajectory_local_sensitivity'],
                        'stability': stability['stability_local_sensitivity'],
                    },
                }
                numerator = (
                    matrix['trajectory_controller']['trajectory']
                    - matrix['trajectory_controller']['stability']
                    + matrix['stability_controller']['stability']
                    - matrix['stability_controller']['trajectory'])
                denominator = sum(
                    abs(value) for values in matrix.values()
                    for value in values.values())
                valid = all(
                    values['valid_operating_regime']
                    for values in controllers.values())
                per_role_expected = {
                    role: bool(entry['expected_double_dissociation'])
                    for role, entry in role_entries.items()
                }
                expected = (
                    valid
                    and all(per_role_expected.values())
                    and matrix['trajectory_controller']['trajectory'] > 0.0
                    and matrix['trajectory_controller']['trajectory']
                    > matrix['trajectory_controller']['stability']
                    and matrix['stability_controller']['stability'] > 0.0
                    and matrix['stability_controller']['stability']
                    > matrix['stability_controller']['trajectory'])
                fsi = abs(trajectory['SI'] - stability['SI']) / 2.0

                canonical = role_entries['left_hold_right_track']['controllers']
                reversed_role = role_entries['left_track_right_hold']['controllers']
                hand_preferences = {}
                demand_preferences = {}
                for controller in ('left', 'right'):
                    right_hand = float(np.mean([
                        canonical[controller]['trajectory_local_sensitivity'],
                        reversed_role[controller]['stability_local_sensitivity'],
                    ]))
                    left_hand = float(np.mean([
                        canonical[controller]['stability_local_sensitivity'],
                        reversed_role[controller]['trajectory_local_sensitivity'],
                    ]))
                    hand_preferences[controller] = (
                        (right_hand - left_hand)
                        / (abs(right_hand) + abs(left_hand) + epsilon))
                    demand_preferences[controller] = controllers[controller]['SI']
                hand_separation = abs(
                    hand_preferences['left'] - hand_preferences['right']) / 2.0
                demand_dominant_fraction = float(np.mean([
                    abs(demand_preferences[controller])
                    >= abs(hand_preferences[controller])
                    for controller in ('left', 'right')
                ]))
                path_output['by_gain'][label] = {
                    'gain': gain,
                    'controllers': controllers,
                    'causal_sensitivity_matrix': matrix,
                    'FSI': fsi,
                    'role_aligned_FSI': numerator / (denominator + epsilon),
                    'expected_double_dissociation': expected,
                    'expected_double_dissociation_by_role': per_role_expected,
                    'reciprocal_sign': trajectory['SI'] * stability['SI'] < 0.0,
                    'reciprocal_FSI': fsi if expected else 0.0,
                    'valid_for_specialisation': valid,
                    'hand_selectivity': {
                        'controller_hand_SI': hand_preferences,
                        'hand_separation_index': hand_separation,
                        'demand_dominant_controller_fraction': (
                            demand_dominant_fraction),
                    },
                }

            entries = list(path_output['by_gain'].values())
            valid_entries = [
                entry for entry in entries
                if entry['valid_for_specialisation']]
            aligned_entries = [
                entry for entry in valid_entries
                if entry['expected_double_dissociation']]
            path_output.update({
                'conditions_available': bool(entries),
                'available': bool(valid_entries),
                'mean_FSI': (
                    float(np.mean([entry['FSI'] for entry in valid_entries]))
                    if valid_entries else None),
                'mean_absolute_FSI': (
                    float(np.mean([entry['FSI'] for entry in valid_entries]))
                    if valid_entries else None),
                'mean_role_aligned_FSI': (
                    float(np.mean([
                        entry['role_aligned_FSI'] for entry in valid_entries]))
                    if valid_entries else None),
                'mean_reciprocal_FSI': (
                    float(np.mean([
                        entry['reciprocal_FSI'] for entry in valid_entries]))
                    if valid_entries else None),
                'reciprocal_gain_fraction': (
                    len(aligned_entries) / len(valid_entries)
                    if valid_entries else None),
                'expected_double_dissociation_gain_fraction': (
                    len(aligned_entries) / len(valid_entries)
                    if valid_entries else None),
                'evaluated_gain_count': len(entries),
                'valid_gain_count': len(valid_entries),
                'excluded_gain_count': len(entries) - len(valid_entries),
                'reciprocal_gain_count': len(aligned_entries),
                'reciprocal_role_consistent': (
                    len(valid_entries) >= 2
                    and len(aligned_entries) == len(valid_entries)),
                'hand_selectivity': {
                    'mean_hand_separation_index': (
                        float(np.mean([
                            entry['hand_selectivity']['hand_separation_index']
                            for entry in valid_entries]))
                        if valid_entries else None),
                    'mean_demand_dominant_controller_fraction': (
                        float(np.mean([
                            entry['hand_selectivity'][
                                'demand_dominant_controller_fraction']
                            for entry in valid_entries]))
                        if valid_entries else None),
                },
            })
            return path_output

        output['paths'] = {
            path: aggregate_path(path)
            for path in ('readout', 'cc', 'whole_module')
        }
        primary = output['paths']['readout']
        output['available'] = primary['available']
        output['by_gain'] = primary['by_gain']
        output['hand_selectivity'] = primary['hand_selectivity']
        for key in (
                'mean_FSI', 'mean_absolute_FSI', 'mean_role_aligned_FSI',
                'mean_reciprocal_FSI', 'reciprocal_gain_fraction',
                'expected_double_dissociation_gain_fraction',
                'evaluated_gain_count', 'valid_gain_count',
                'excluded_gain_count', 'reciprocal_gain_count',
                'reciprocal_role_consistent'):
            output[key] = primary[key]
        return output

    @staticmethod
    def compute_velocity_variance(velocities: torch.Tensor, k: float = 1.0) -> float:
        """Compute scaled velocity variance during reaching."""
        if velocities.size(0) < 2:
            return 0.0
        return k * velocities.var(dim=0).sum(dim=-1).mean().item()

    @staticmethod
    def compute_trajectory_error(
        positions: torch.Tensor, starts: torch.Tensor, targets: torch.Tensor
    ) -> float:
        """Compute mean perpendicular deviation from the straight reach path."""
        direction = targets - starts
        direction = direction / (torch.norm(direction, dim=-1, keepdim=True) + 1e-8)
        offset = positions - starts.unsqueeze(0)
        deviation = offset[..., 0] * direction[..., 1] - offset[..., 1] * direction[..., 0]
        return torch.abs(deviation).mean().item()

    @staticmethod
    def compute_holding_variance(positions: torch.Tensor) -> float:
        """Compute endpoint-position variance during holding."""
        if positions.size(0) < 2:
            return 0.0
        return positions.var(dim=0).sum(dim=-1).mean().item()

    @staticmethod
    def compute_energy_cost(actions: torch.Tensor) -> float:
        """Compute deterministic-command energy summed over muscles and averaged."""
        return actions.pow(2).sum(dim=-1).mean().item()

    @staticmethod
    def compute_ddi(
        intact_left: float,
        intact_right: float,
        lesion_l_left: float,
        lesion_l_right: float,
        lesion_r_left: float,
        lesion_r_right: float,
    ) -> dict:
        """Compute the double-dissociation index (DDI)."""
        contra_l = lesion_l_right - intact_right
        ipsi_l = lesion_l_left - intact_left
        contra_r = lesion_r_left - intact_left
        ipsi_r = lesion_r_right - intact_right
        ddi_l = (contra_l - ipsi_l) / (abs(contra_l) + abs(ipsi_l) + 1e-8)
        ddi_r = (contra_r - ipsi_r) / (abs(contra_r) + abs(ipsi_r) + 1e-8)
        return {"left_hem": ddi_l, "right_hem": ddi_r, "average": (ddi_l + ddi_r) / 2}

    @staticmethod
    def assemble_sweep_results(
        final_ddi: dict,
        final_contralaterality: dict,
        lesion_results: dict,
    ) -> dict:
        """Assemble evaluation outputs for a trained two-module controller."""
        intact = lesion_results["intact"]

        def avg(condition, key):
            """Average a paired metric across the two arms."""
            return (condition[f"{key}_left"] + condition[f"{key}_right"]) / 2.0

        result = {
            "Final_err_int": avg(intact, "final_err"),
            "final_err_left": intact["final_err_left"],
            "final_err_right": intact["final_err_right"],
            "Err_traj_int": avg(intact, "err_traj"),
            "Var_move_int": avg(intact, "var_move"),
            "Var_hold_int": avg(intact, "var_hold"),
            "Hold_err_int": avg(intact, "hold_err"),
            "hold_err_left": intact["hold_err_left"],
            "hold_err_right": intact["hold_err_right"],
            "DDI": final_ddi["average"],
            "CI": final_contralaterality["contralateral_index"],
            "Ene_int": intact["energy_cost"],
            "Ene_reach": intact["ene_reach"],
            "Ene_hold": intact["ene_hold"],
            "CCI": intact["CCI"],
            "CCI_left": intact["CCI_left"],
            "CCI_right": intact["CCI_right"],
        }
        result.update({k: v for k, v in intact.items() if "success_" in k})
        return result
