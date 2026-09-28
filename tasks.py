import torch
import numpy as np
import motornet
# ===== Bimanual arm geometry =====
class BimanualArms:
    """
    Pair of symmetric MotorNet arm effectors in a shared workspace.

    The left effector is mirrored across the x-axis and each shoulder is
    translated by half the configured separation. This shared geometry is
    used by both the reaching and postural-holding tasks.
    """
    def __init__(self, timestep=0.01, arm_separation=0.4):
        """
        Construct paired arm effectors and their world-coordinate offsets.
        
        Args:
            timestep: Simulator time step in seconds.
            arm_separation: Distance between shoulder origins in metres.
        """
        self.timestep = timestep
        self.arm_separation = arm_separation
        
        # Give each arm its own muscle dynamics and effector state.
        muscle_left = motornet.muscle.RigidTendonHillMuscle()
        muscle_right = motornet.muscle.RigidTendonHillMuscle()
        
        self.effector_left = motornet.effector.RigidTendonArm26(
            muscle=muscle_left, 
            timestep=timestep
        )
        self.effector_right = motornet.effector.RigidTendonArm26(
            muscle=muscle_right, 
            timestep=timestep
        )
        
        # Symmetric arms share dimensions, with six muscles per arm.
        self.n_muscles_per_arm = self.effector_left.n_muscles
        self.n_muscles_total = self.n_muscles_per_arm * 2
        self.dof_per_arm = self.effector_left.dof
        self.dof_total = self.dof_per_arm * 2
        
        # Shoulder positions define the world-coordinate reference frame.
        self.shoulder_left = torch.tensor([-arm_separation / 2, 0.0])
        self.shoulder_right = torch.tensor([arm_separation / 2, 0.0])
        
        # Actions contain left-arm muscles followed by right-arm muscles.
        self.muscle_mapping = {
            'left_arm': list(range(0, self.n_muscles_per_arm)),
            'right_arm': list(range(self.n_muscles_per_arm, self.n_muscles_total)),
        }
    
    def get_effector_left(self):
        """Return the left-arm MotorNet effector."""
        return self.effector_left
    
    def get_effector_right(self):
        """Return the right-arm MotorNet effector."""
        return self.effector_right
    
    def transform_left_to_world(self, pos_local):
        """
        Transform left-arm local kinematics into world coordinates.

        Position and x-velocity are mirrored before applying the shoulder
        translation.
        
        Args:
            pos_local: Position or kinematic state of shape [batch, 2] or
                [batch, 4].
            
        Returns:
            The state in the common bimanual workspace.
        """
        if pos_local.shape[-1] == 2:
            # The two-dimensional representation stores position alone.
            pos_world = pos_local.clone()
            pos_world[:, 0] = -pos_local[:, 0] + self.shoulder_left[0].to(pos_local.device)
            pos_world[:, 1] = pos_local[:, 1] + self.shoulder_left[1].to(pos_local.device)
        else:
            # The four-dimensional representation stores [x, y, vx, vy].
            pos_world = pos_local.clone()
            pos_world[:, 0] = -pos_local[:, 0] + self.shoulder_left[0].to(pos_local.device)
            pos_world[:, 1] = pos_local[:, 1] + self.shoulder_left[1].to(pos_local.device)
            pos_world[:, 2] = -pos_local[:, 2]
        return pos_world
    
    def transform_right_to_world(self, pos_local):
        """
        Transform right-arm local kinematics into world coordinates.
        
        Args:
            pos_local: Position or kinematic state of shape [batch, 2] or
                [batch, 4].
            
        Returns:
            The state in the common bimanual workspace.
        """
        if pos_local.shape[-1] == 2:
            pos_world = pos_local.clone()
            pos_world[:, 0] = pos_local[:, 0] + self.shoulder_right[0].to(pos_local.device)
            pos_world[:, 1] = pos_local[:, 1] + self.shoulder_right[1].to(pos_local.device)
        else:
            pos_world = pos_local.clone()
            pos_world[:, 0] = pos_local[:, 0] + self.shoulder_right[0].to(pos_local.device)
            pos_world[:, 1] = pos_local[:, 1] + self.shoulder_right[1].to(pos_local.device)
        return pos_world

    def transform_left_force_world_to_local(self, force_world):
        """Express a world-frame endpoint force in the mirrored left-arm frame."""
        force_local = force_world.clone()
        force_local[:, 0] = -force_world[:, 0]
        return force_local

    def transform_right_force_world_to_local(self, force_world):
        """Express a world-frame endpoint force in the right-arm local frame."""
        return force_world.clone()

# ===== Bimanual centre-out reaching task =====
class BimanualCentreOutReaching:
    """
    Bimanual centre-out reaching environment.

    Both arms begin at their workspace centres. Each episode samples one of
    eight directions and initialises mirrored targets for the two arms.
    Either target can subsequently be replaced to represent independent
    unimanual or coordinated bimanual reaching objectives.
    """
    def __init__(self, 
                 bimanual_arms=None, 
                 device='cpu', 
                 reach_radius=0.08,
                 include_task_indicator=False,
                 timestep=0.01):
        """
        Initialise the reaching environment and its observation layout.
        
        Args:
            bimanual_arms: Shared bimanual geometry. A new instance is
                created when omitted.
            device: Torch device used for environment tensors.
            reach_radius: Distance from start position to each reach target.
            timestep: Simulator time step in seconds.
        """
        self._device = device
        self.reach_radius = reach_radius
        self.include_task_indicator = include_task_indicator
        self.timestep = timestep
        
        # Targets are sampled from eight evenly spaced movement directions.
        self.n_targets = 8
        self.target_angles = torch.tensor([i * 2 * np.pi / self.n_targets 
                                          for i in range(self.n_targets)])
        
        if bimanual_arms is None:
            bimanual_arms = BimanualArms(timestep=timestep)
        self.bimanual_arms = bimanual_arms
        
        self.effector_left = bimanual_arms.get_effector_left()
        self.effector_right = bimanual_arms.get_effector_right()
        
        # MotorNet evolves one effector per environment instance.
        self._env_left = _SingleArmEnvWrapper(
            self.effector_left, 
            self._compute_center_joint_angles(),
            device=device
        )
        self._env_right = _SingleArmEnvWrapper(
            self.effector_right, 
            self._compute_center_joint_angles(),
            device=device
        )
        
        # Episode state stores sampled targets and currently configured targets.
        self.targets_left = None
        self.targets_right = None
        self.start_pos_left = None
        self.start_pos_right = None
        
        # Each arm contributes 16 sensor values and 5 target-relative values.
        # Coordination adds distance, relative position and target symmetry.
        self.obs_dim_per_arm = 21 + int(include_task_indicator)
        self.obs_dim_coordination = 4
        self.obs_dim_total = self.obs_dim_per_arm * 2 + self.obs_dim_coordination
        
        # The action activates six muscles for each arm.
        self.action_dim = 12
        self._last_obs_left = self._last_obs_right = None
        self._last_info_left = self._last_info_right = None
        
    @property
    def device(self):
        """Return the Torch device used for task tensors."""
        return self._device
    
    @property
    def dt(self):
        """Return the simulator time step in seconds."""
        return self.timestep
    
    def set_current_targets(self, targets_left=None, targets_right=None):
        """
        Set the targets currently exposed to control and error computation.

        Left and right targets can be specified independently for tasks in
        which one hand acts while the other rests, or both hands act together
        in a bimanual coordination task.
        
        Args:
            targets_left: Left-hand target positions of shape [batch, 2].
            targets_right: Right-hand target positions of shape [batch, 2].
        """
        if targets_left is not None:
            self.current_targets_left = targets_left
        if targets_right is not None:
            self.current_targets_right = targets_right

    def get_observation(self):
        """Rebuild the current observation after callers change targets."""
        if self._last_obs_left is None:
            raise RuntimeError('reset must be called before get_observation')
        return self._get_bimanual_obs(
            self._last_obs_left,
            self._last_obs_right,
            self._last_info_left,
            self._last_info_right,
        )
    
    def _compute_center_joint_angles(self):
        """Return joint angles that place an effector at its start position."""
        shoulder_angle = np.deg2rad(45)
        elbow_angle = np.deg2rad(90)
        return torch.tensor([[shoulder_angle, elbow_angle]], dtype=torch.float32)
    
    def reset(self, batch_size=1, options=None):
        """
        Reset both effectors and sample mirrored reach targets.
        
        Args:
            batch_size: Number of simulated trials.
            options: Reserved environment reset options.
            
        Returns:
            Observation tensor and task state dictionary.
        """
        # Begin both arms at their central start posture.
        obs_left, info_left = self._env_left.reset(batch_size=batch_size)
        obs_right, info_right = self._env_right.reset(batch_size=batch_size)
        self._last_obs_left, self._last_obs_right = obs_left, obs_right
        self._last_info_left, self._last_info_right = info_left, info_right
        
        # Express endpoints in the common workspace.
        start_local_left = info_left['states']['cartesian'][:, :2]
        start_local_right = info_right['states']['cartesian'][:, :2]
        
        self.start_pos_left = self.bimanual_arms.transform_left_to_world(start_local_left).to(self.device)
        self.start_pos_right = self.bimanual_arms.transform_right_to_world(start_local_right).to(self.device)
        
        # Random direction sampling provides the eight-direction reach set.
        target_indices = torch.randint(0, self.n_targets, (batch_size,), device=self.device)
        target_angles_device = self.target_angles.to(self.device)
        target_angles_batch = target_angles_device[target_indices]
        
        # Reaching targets form a mirrored bimanual pair.
        angles_right = target_angles_batch
        angles_left = np.pi - target_angles_batch
        
        # Convert the sampled direction into Cartesian target offsets.
        delta_x_left = self.reach_radius * torch.cos(angles_left)
        delta_y_left = self.reach_radius * torch.sin(angles_left)
        delta_x_right = self.reach_radius * torch.cos(angles_right)
        delta_y_right = self.reach_radius * torch.sin(angles_right)
        
        self.targets_left = self.start_pos_left + torch.stack([delta_x_left, delta_y_left], dim=-1).to(self.device)
        self.targets_right = self.start_pos_right + torch.stack([delta_x_right, delta_y_right], dim=-1).to(self.device)
        
        # Use the sampled pair until a caller specifies alternative targets.
        self.current_targets_left = self.targets_left.clone().to(self.device)
        self.current_targets_right = self.targets_right.clone().to(self.device)
        
        obs = self._get_bimanual_obs(obs_left.to(self.device), obs_right.to(self.device), info_left, info_right)
        
        info = {
            'states_left': info_left['states'],
            'states_right': info_right['states'],
            'targets_left': self.targets_left,
            'targets_right': self.targets_right,
            'current_targets_left': self.current_targets_left,
            'current_targets_right': self.current_targets_right,
            'start_pos_left': self.start_pos_left,
            'start_pos_right': self.start_pos_right,
            'pos_world_left': self.start_pos_left,
            'pos_world_right': self.start_pos_right,
        }
        
        return obs, info
    
    def step(self, action):
        """
        Advance both effectors by one control step.
        
        Args:
            action: Muscle activations of shape [batch, 12], ordered as six
                left-arm commands followed by six right-arm commands.
                   
        Returns:
            obs, reward, terminated, truncated, info
        """
        batch_size = action.shape[0]
        
        # Split the combined motor command across the two effectors.
        action_left = action[:, :6]
        action_right = action[:, 6:]
        
        obs_left, reward_left, term_left, trunc_left, info_left = self._env_left.step(action_left)
        obs_right, reward_right, term_right, trunc_right, info_right = self._env_right.step(action_right)
        self._last_obs_left, self._last_obs_right = obs_left, obs_right
        self._last_info_left, self._last_info_right = info_left, info_right
        
        # Express endpoint positions in the shared workspace.
        pos_local_left = info_left['states']['cartesian'][:, :2]
        pos_local_right = info_right['states']['cartesian'][:, :2]
        pos_world_left = self.bimanual_arms.transform_left_to_world(pos_local_left).to(self.device)
        pos_world_right = self.bimanual_arms.transform_right_to_world(pos_local_right).to(self.device)
        
        # Error follows the currently configured targets.
        error_left = torch.norm(pos_world_left - self.current_targets_left, dim=-1)
        error_right = torch.norm(pos_world_right - self.current_targets_right, dim=-1)
        
        obs = self._get_bimanual_obs(obs_left.to(self.device), obs_right.to(self.device), info_left, info_right)
        
        # MotorNet may omit rewards, so task rewards default to zero.
        if reward_left is None:
            reward_left = torch.zeros(batch_size, device=self.device)
        if reward_right is None:
            reward_right = torch.zeros(batch_size, device=self.device)
        reward = reward_left + reward_right
        
        terminated = term_left | term_right
        truncated = trunc_left | trunc_right
        
        info = {
            'states_left': info_left['states'],
            'states_right': info_right['states'],
            'targets_left': self.targets_left,
            'targets_right': self.targets_right,
            'current_targets_left': self.current_targets_left,
            'current_targets_right': self.current_targets_right,
            'error_left': error_left,
            'error_right': error_right,
            'error_total': (error_left + error_right) / 2,
            'pos_world_left': pos_world_left,
            'pos_world_right': pos_world_right,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_bimanual_obs(self, obs_left, obs_right, info_left, info_right):
        """
        Construct the 46-dimensional bimanual reach observation.

        Each arm supplies MotorNet sensory input and five target-relative
        features: displacement, direction and distance. Four shared features
        describe inter-hand distance, relative position and target symmetry.
        """
        pos_local_left = info_left['states']['cartesian'][:, :2]
        pos_local_right = info_right['states']['cartesian'][:, :2]
        pos_world_left = self.bimanual_arms.transform_left_to_world(pos_local_left).to(self.device)
        pos_world_right = self.bimanual_arms.transform_right_to_world(pos_local_right).to(self.device)
        
        # Target features reflect the currently configured targets.
        if self.current_targets_left is not None:
            relative_pos_left = self.current_targets_left - pos_world_left
            distance_left = torch.norm(relative_pos_left, dim=-1, keepdim=True)
            direction_left = relative_pos_left / (distance_left + 1e-8)
            target_info_left = torch.cat([relative_pos_left, direction_left, distance_left], dim=-1)
        else:
            target_info_left = torch.zeros(obs_left.shape[0], 5, device=self.device)
        
        if self.current_targets_right is not None:
            relative_pos_right = self.current_targets_right - pos_world_right
            distance_right = torch.norm(relative_pos_right, dim=-1, keepdim=True)
            direction_right = relative_pos_right / (distance_right + 1e-8)
            target_info_right = torch.cat([relative_pos_right, direction_right, distance_right], dim=-1)
        else:
            target_info_right = torch.zeros(obs_right.shape[0], 5, device=self.device)
        
        obs_left_full = torch.cat([obs_left.to(self.device), target_info_left], dim=-1)
        obs_right_full = torch.cat([obs_right.to(self.device), target_info_right], dim=-1)
        if self.include_task_indicator:
            indicator = torch.zeros(obs_left.shape[0], 1, device=self.device)
            obs_left_full = torch.cat([obs_left_full, indicator], dim=-1)
            obs_right_full = torch.cat([obs_right_full, indicator], dim=-1)
        
        # Shared features provide the controller with bimanual geometry.
        hand_distance = torch.norm(pos_world_right - pos_world_left, dim=-1, keepdim=True)
        relative_hand_pos = pos_world_right - pos_world_left
        
        # Symmetry is zero for mirror-equivalent current targets.
        if self.current_targets_left is not None and self.current_targets_right is not None:
            target_midpoint = (self.current_targets_left + self.current_targets_right) / 2
            symmetry_indicator = torch.norm(target_midpoint - (self.start_pos_left.to(self.device) + self.start_pos_right.to(self.device)) / 2, 
                                           dim=-1, keepdim=True) / (self.reach_radius + 1e-8)
        else:
            symmetry_indicator = torch.zeros(obs_left.shape[0], 1, device=self.device)
        
        coordination_info = torch.cat([hand_distance, relative_hand_pos, symmetry_indicator], dim=-1)
        
        obs = torch.cat([obs_left_full, obs_right_full, coordination_info], dim=-1)
        
        # Fail early if upstream MotorNet observation dimensions change.
        assert obs.shape[-1] == self.obs_dim_total, (
            f"Obs dim mismatch: got {obs.shape[-1]}, expected {self.obs_dim_total}. "
            f"Check MotorNet base_obs dim or target_info calculation."
        )
        
        return obs
    
class _SingleArmEnvWrapper(motornet.environment.Environment):
    """
    Minimal MotorNet environment wrapper for one arm effector.

    The bimanual tasks coordinate two instances and can pass Cartesian loads
    to each physical arm independently.
    """
    def __init__(self, effector, q_init, device='cpu'):
        """Create a single-effector environment on the requested device."""
        self._device = device
        super().__init__(effector=effector, q_init=q_init)
        self.to(device)

    def reset(self, batch_size=1, options=None):
        """Reset one arm for a batch of trials."""
        opts = {} if options is None else options.copy()
        opts['batch_size'] = batch_size

        if self.q_init is not None and not torch.is_tensor(self.q_init):
            opts['joint_state'] = torch.tensor(self.q_init, dtype=torch.float32)

        obs, info = super().reset(seed=None, options=opts)
        return obs, info

    def step(self, action, endpoint_load=None):
        """
        Advance one effector with optional endpoint loading.

        Args:
            action: Muscle activation signal of shape [batch, n_muscles].
            endpoint_load: Optional Cartesian force of shape [batch, 2].

        Returns:
            MotorNet observation, reward, termination flags and state details.
        """
        if endpoint_load is not None:
            return super().step(action, endpoint_load=endpoint_load)
        else:
            return super().step(action)


# ===== Bimanual postural-holding task =====
class BimanualPosturalHolding:
    """
    Bimanual postural-holding environment under configurable endpoint loads.

    Both hands initially hold their starting positions. Target positions can
    be set independently, and endpoint loads can be applied independently to
    either hand or to both hands through a perturbation mask.
    """

    def __init__(
        self,
        bimanual_arms=None,
        device='cpu',
        hold_duration_ms=400,
        external_force_std=0.5,
        perturbation_mode='pulse',
        pulse_onset_range=(5, 10),
        pulse_duration_range=(10, 15),
        pulse_magnitude_range=(0.5, 1.5),
        recovery_min_steps=10,
        include_task_indicator=False,
        timestep=0.01,
    ):
        """
        Initialise the holding environment and perturbation process.

        Args:
            bimanual_arms: Shared bimanual geometry.
            device: Torch device used for environment tensors.
            hold_duration_ms: Duration of a holding episode in milliseconds.
            external_force_std: Standard deviation of endpoint force in N.
            timestep: Simulator time step in seconds.
        """
        self._device = device
        self.hold_duration_ms = hold_duration_ms
        self.external_force_std = external_force_std
        self.perturbation_mode = perturbation_mode
        self.pulse_onset_range = tuple(pulse_onset_range)
        self.pulse_duration_range = tuple(pulse_duration_range)
        self.pulse_magnitude_range = tuple(pulse_magnitude_range)
        self.recovery_min_steps = int(recovery_min_steps)
        self.include_task_indicator = include_task_indicator
        self.timestep = timestep

        if perturbation_mode not in {'pulse', 'iid'}:
            raise ValueError("perturbation_mode must be 'pulse' or 'iid'")

        if bimanual_arms is None:
            bimanual_arms = BimanualArms(timestep=timestep)
        self.bimanual_arms = bimanual_arms

        self.effector_left = bimanual_arms.get_effector_left()
        self.effector_right = bimanual_arms.get_effector_right()

        # Convert task duration into simulator steps.
        self.hold_steps = int(hold_duration_ms / 1000.0 / timestep)

        # MotorNet evolves one arm per environment instance.
        self._env_left = _SingleArmEnvWrapper(
            self.effector_left,
            self._compute_center_joint_angles(),
            device=device
        )
        self._env_right = _SingleArmEnvWrapper(
            self.effector_right,
            self._compute_center_joint_angles(),
            device=device
        )

        # Keep targets and endpoint histories for hold-phase measurements.
        self.targets_left = None
        self.targets_right = None
        self.start_pos_left = None
        self.start_pos_right = None
        self.current_step = 0
        self.position_history_left = []
        self.position_history_right = []
        self.pulse_onset = None
        self.pulse_end = None
        self.pulse_force_left = None
        self.pulse_force_right = None
        self._last_obs_left = None
        self._last_obs_right = None
        self._last_info_left = None
        self._last_info_right = None

        # Task indicators are off in the primary experiment but remain an ablation.
        self.obs_dim_per_arm = 21 + int(include_task_indicator)
        self.obs_dim_coordination = 4
        self.obs_dim_total = self.obs_dim_per_arm * 2 + self.obs_dim_coordination

        # The action activates six muscles for each arm.
        self.action_dim = 12

    @staticmethod
    def _sample_int_range(bounds, batch_size, device):
        """Sample integer values from an inclusive configuration range."""
        low, high = int(bounds[0]), int(bounds[1])
        if low > high:
            raise ValueError(f'invalid integer range {bounds}')
        return torch.randint(low, high + 1, (batch_size,), device=device)

    def _sample_pulses(self, batch_size):
        """Sample one fixed Cartesian force pulse for each trial and arm."""
        latest_end = self.hold_steps - self.recovery_min_steps
        if int(self.pulse_onset_range[1]) >= latest_end:
            raise ValueError('pulse_onset_range leaves no configured recovery window')
        onset = self._sample_int_range(self.pulse_onset_range, batch_size, self.device)
        duration = self._sample_int_range(self.pulse_duration_range, batch_size, self.device)
        duration = torch.minimum(duration, latest_end - onset)
        if torch.any(duration < 1):
            raise ValueError('pulse ranges leave no time for the recovery window')

        low, high = self.pulse_magnitude_range
        magnitude = low + (high - low) * torch.rand(batch_size, device=self.device)
        angle_left = 2.0 * np.pi * torch.rand(batch_size, device=self.device)
        angle_right = 2.0 * np.pi * torch.rand(batch_size, device=self.device)
        self.pulse_onset = onset
        self.pulse_end = onset + duration
        self.pulse_force_left = magnitude.unsqueeze(-1) * torch.stack(
            [torch.cos(angle_left), torch.sin(angle_left)], dim=-1)
        self.pulse_force_right = magnitude.unsqueeze(-1) * torch.stack(
            [torch.cos(angle_right), torch.sin(angle_right)], dim=-1)

    def get_phase_masks(self, next_step=False):
        """Return per-trial pre-pulse, pulse and recovery masks."""
        if self.pulse_onset is None:
            raise RuntimeError('reset must be called before get_phase_masks')
        step = self.current_step + (1 if next_step else 0)
        pulse = (step >= self.pulse_onset) & (step < self.pulse_end)
        recovery = step >= self.pulse_end
        return {'pre': step < self.pulse_onset, 'pulse': pulse, 'recovery': recovery}

    @property
    def device(self):
        """Return the Torch device used for task tensors."""
        return self._device

    @property
    def dt(self):
        """Return the simulator time step in seconds."""
        return self.timestep

    def _compute_center_joint_angles(self):
        """Return joint angles that place an effector at its start position."""
        shoulder_angle = np.deg2rad(45)
        elbow_angle = np.deg2rad(90)
        return torch.tensor([[shoulder_angle, elbow_angle]], dtype=torch.float32)

    def set_current_targets(self, targets_left=None, targets_right=None):
        """
        Set holding targets used in observations and position error.

        Left and right targets can be specified independently to represent
        unilateral stabilisation or a shared bimanual holding objective.

        Args:
            targets_left: Left-hand target positions of shape [batch, 2].
            targets_right: Right-hand target positions of shape [batch, 2].
        """
        if targets_left is not None:
            self.current_targets_left = targets_left
        if targets_right is not None:
            self.current_targets_right = targets_right

    def get_observation(self):
        """Rebuild the observation after an inference probe changes targets."""
        if any(value is None for value in (
                self._last_obs_left, self._last_obs_right,
                self._last_info_left, self._last_info_right)):
            raise RuntimeError('reset must be called before get_observation')
        return self._get_bimanual_obs(
            self._last_obs_left,
            self._last_obs_right,
            self._last_info_left,
            self._last_info_right,
        )

    def reset(self, batch_size=1, options=None):
        """
        Reset both effectors with targets at their starting positions.

        Returns:
            Observation tensor and task state dictionary.
        """
        # Begin both arms at their central start posture.
        obs_left, info_left = self._env_left.reset(batch_size=batch_size)
        obs_right, info_right = self._env_right.reset(batch_size=batch_size)
        self._last_obs_left = obs_left.to(self.device)
        self._last_obs_right = obs_right.to(self.device)
        self._last_info_left = info_left
        self._last_info_right = info_right

        # Express endpoints in the common workspace.
        start_local_left = info_left['states']['cartesian'][:, :2]
        start_local_right = info_right['states']['cartesian'][:, :2]

        self.start_pos_left = self.bimanual_arms.transform_left_to_world(start_local_left).to(self.device)
        self.start_pos_right = self.bimanual_arms.transform_right_to_world(start_local_right).to(self.device)

        # The holding objective stabilises each hand at its start position.
        self.targets_left = self.start_pos_left.clone()
        self.targets_right = self.start_pos_right.clone()

        self.current_targets_left = self.targets_left.clone()
        self.current_targets_right = self.targets_right.clone()

        # Clear the trajectory used to measure position variance.
        self.current_step = 0
        self.position_history_left = []
        self.position_history_right = []
        if self.perturbation_mode == 'pulse':
            self._sample_pulses(batch_size)
        else:
            self.pulse_onset = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            self.pulse_end = torch.full(
                (batch_size,), self.hold_steps, dtype=torch.long, device=self.device)
            self.pulse_force_left = torch.zeros(batch_size, 2, device=self.device)
            self.pulse_force_right = torch.zeros(batch_size, 2, device=self.device)

        obs = self._get_bimanual_obs(obs_left.to(self.device), obs_right.to(self.device),
                                     info_left, info_right)

        info = {
            'states_left': info_left['states'],
            'states_right': info_right['states'],
            'targets_left': self.targets_left,
            'targets_right': self.targets_right,
            'current_targets_left': self.current_targets_left,
            'current_targets_right': self.current_targets_right,
            'start_pos_left': self.start_pos_left,
            'start_pos_right': self.start_pos_right,
            'pos_world_left': self.start_pos_left,
            'pos_world_right': self.start_pos_right,
            'pulse_onset': self.pulse_onset,
            'pulse_end': self.pulse_end,
            'pulse_force_left': self.pulse_force_left,
            'pulse_force_right': self.pulse_force_right,
        }

        return obs, info

    def step(self, action, perturbation_mask=None):
        """
        Advance both arms under optional selective endpoint perturbation.

        Args:
            action: Muscle activations of shape [batch, 12], ordered as six
                left-arm commands followed by six right-arm commands.
            perturbation_mask: Optional [batch, 2] selector for left and
                right endpoint loads.

        Returns:
            obs, reward, terminated, truncated, info
        """
        batch_size = action.shape[0]
        self.current_step += 1

        # Split the combined motor command across the two effectors.
        action_left = action[:, :6]
        action_right = action[:, 6:]

        # Pulse mode applies a fixed force vector between sampled onset/end.
        if self.perturbation_mode == 'pulse':
            pulse_mask = self.get_phase_masks()['pulse'].unsqueeze(-1)
            external_force_left = self.pulse_force_left * pulse_mask
            external_force_right = self.pulse_force_right * pulse_mask
        elif self.external_force_std > 0:
            external_force_left = torch.randn(batch_size, 2, device=self.device) * self.external_force_std
            external_force_right = torch.randn(batch_size, 2, device=self.device) * self.external_force_std
        else:
            external_force_left = torch.zeros(batch_size, 2, device=self.device)
            external_force_right = torch.zeros(batch_size, 2, device=self.device)

        if perturbation_mask is not None:
            mask_left = perturbation_mask[:, 0].unsqueeze(-1)
            mask_right = perturbation_mask[:, 1].unsqueeze(-1)
            external_force_left = external_force_left * mask_left
            external_force_right = external_force_right * mask_right

        # Send the selected Cartesian loads to the MotorNet physics engine.
        obs_left, reward_left, term_left, trunc_left, info_left = self._env_left.step(
            action_left, endpoint_load=external_force_left
        )
        obs_right, reward_right, term_right, trunc_right, info_right = self._env_right.step(
            action_right, endpoint_load=external_force_right
        )
        self._last_obs_left = obs_left.to(self.device)
        self._last_obs_right = obs_right.to(self.device)
        self._last_info_left = info_left
        self._last_info_right = info_right

        # Express endpoint positions in the shared workspace.
        pos_local_left = info_left['states']['cartesian'][:, :2]
        pos_local_right = info_right['states']['cartesian'][:, :2]
        pos_world_left = self.bimanual_arms.transform_left_to_world(pos_local_left).to(self.device)
        pos_world_right = self.bimanual_arms.transform_right_to_world(pos_local_right).to(self.device)

        # Store positions for the hold-variance diagnostic.
        self.position_history_left.append(pos_world_left.detach().clone())
        self.position_history_right.append(pos_world_right.detach().clone())

        # Holding error measures displacement from the current targets.
        error_left = torch.norm(pos_world_left - self.current_targets_left, dim=-1)
        error_right = torch.norm(pos_world_right - self.current_targets_right, dim=-1)

        variance_left = self._compute_position_variance(self.position_history_left)
        variance_right = self._compute_position_variance(self.position_history_right)

        obs = self._get_bimanual_obs(obs_left.to(self.device), obs_right.to(self.device),
                                     info_left, info_right)

        # MotorNet may omit rewards, so task rewards default to zero.
        if reward_left is None:
            reward_left = torch.zeros(batch_size, device=self.device)
        if reward_right is None:
            reward_right = torch.zeros(batch_size, device=self.device)
        reward = reward_left + reward_right

        # Holding duration defines the task termination horizon.
        if self.current_step >= self.hold_steps:
            terminated = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        else:
            terminated = term_left | term_right

        truncated = trunc_left | trunc_right

        info = {
            'states_left': info_left['states'],
            'states_right': info_right['states'],
            'targets_left': self.targets_left,
            'targets_right': self.targets_right,
            'current_targets_left': self.current_targets_left,
            'current_targets_right': self.current_targets_right,
            'error_left': error_left,
            'error_right': error_right,
            'error_total': (error_left + error_right) / 2,
            'pos_world_left': pos_world_left,
            'pos_world_right': pos_world_right,
            'external_force_left': external_force_left,
            'external_force_right': external_force_right,
            'perturbation_phase': self.get_phase_masks(),
            'pulse_onset': self.pulse_onset,
            'pulse_end': self.pulse_end,
            'pulse_force_left': self.pulse_force_left,
            'pulse_force_right': self.pulse_force_right,
            'position_variance_left': variance_left,
            'position_variance_right': variance_right,
            'position_variance_total': (variance_left + variance_right) / 2,
        }

        return obs, reward, terminated, truncated, info

    def _compute_position_variance(self, position_history):
        """Return mean endpoint-position variance over an observed hold."""
        if len(position_history) < 2:
            return torch.tensor(0.0, device=self.device)

        positions = torch.stack(position_history, dim=0)
        variance = positions.var(dim=0).sum(dim=-1).mean()
        return variance

    def _get_bimanual_obs(self, obs_left, obs_right, info_left, info_right):
        """
        Construct the shared bimanual holding observation.

        Each arm supplies 16 MotorNet sensor values, five target-relative
        values and an optional task indicator. Four shared features represent
        inter-hand distance, relative position and midpoint displacement.
        """
        pos_local_left = info_left['states']['cartesian'][:, :2]
        pos_local_right = info_right['states']['cartesian'][:, :2]
        pos_world_left = self.bimanual_arms.transform_left_to_world(pos_local_left).to(self.device)
        pos_world_right = self.bimanual_arms.transform_right_to_world(pos_local_right).to(self.device)

        # Target-relative features describe deviations from the held posture.
        if self.current_targets_left is not None:
            relative_pos_left = self.current_targets_left - pos_world_left
            distance_left = torch.norm(relative_pos_left, dim=-1, keepdim=True)
            direction_left = relative_pos_left / (distance_left + 1e-8)
            target_info_left = torch.cat([relative_pos_left, direction_left, distance_left], dim=-1)
        else:
            target_info_left = torch.zeros(obs_left.shape[0], 5, device=self.device)

        if self.current_targets_right is not None:
            relative_pos_right = self.current_targets_right - pos_world_right
            distance_right = torch.norm(relative_pos_right, dim=-1, keepdim=True)
            direction_right = relative_pos_right / (distance_right + 1e-8)
            target_info_right = torch.cat([relative_pos_right, direction_right, distance_right], dim=-1)
        else:
            target_info_right = torch.zeros(obs_right.shape[0], 5, device=self.device)

        obs_left_full = torch.cat([obs_left.to(self.device), target_info_left], dim=-1)
        obs_right_full = torch.cat([obs_right.to(self.device), target_info_right], dim=-1)
        if self.include_task_indicator:
            hold_indicator = torch.ones(obs_left.shape[0], 1, device=self.device)
            obs_left_full = torch.cat([obs_left_full, hold_indicator], dim=-1)
            obs_right_full = torch.cat([obs_right_full, hold_indicator], dim=-1)

        # Shared features provide the controller with bimanual geometry.
        hand_distance = torch.norm(pos_world_right - pos_world_left, dim=-1, keepdim=True)
        relative_hand_pos = pos_world_right - pos_world_left

        # Midpoint displacement records coordinated drift from the targets.
        if self.current_targets_left is not None and self.current_targets_right is not None:
            target_midpoint = (self.current_targets_left + self.current_targets_right) / 2
            hand_midpoint = (pos_world_left + pos_world_right) / 2
            symmetry_indicator = torch.norm(hand_midpoint - target_midpoint,
                                           dim=-1, keepdim=True)
        else:
            symmetry_indicator = torch.zeros(obs_left.shape[0], 1, device=self.device)

        coordination_info = torch.cat([hand_distance, relative_hand_pos, symmetry_indicator], dim=-1)

        obs = torch.cat([obs_left_full, obs_right_full, coordination_info], dim=-1)

        # Fail early if upstream MotorNet observation dimensions change.
        assert obs.shape[-1] == self.obs_dim_total, (
            f"Obs dim mismatch: got {obs.shape[-1]}, expected {self.obs_dim_total}"
        )

        return obs

# ===== Coupled bimanual tracking-and-holding task (Task4) =====
class BimanualCoupledTrackHold(BimanualPosturalHolding):
    """Track with one hand while the other rejects an independent force pulse.

    ``role_bias`` is the probability of the canonical assignment (left holds,
    right tracks) on each trial.  It changes training exposure, not controller
    routing, and it is not included as an observation feature.  Mechanical
    coupling remains configurable for later ablations.  The primary fixed-role
    experiment sets it to zero and perturbs only the holding hand.
    """

    def __init__(
        self,
        bimanual_arms=None,
        device='cpu',
        duration_ms=1000,
        role_bias=0.8,
        role_assignment_mode='bernoulli',
        trajectory_amplitude_range=(0.04, 0.06),
        trajectory_frequency_range=(0.75, 1.25),
        spring_stiffness=20.0,
        damping_coefficient=1.0,
        max_coupling_force=3.0,
        hold_perturbation_mode='none',
        hold_pulse_onset_range=(5, 10),
        hold_pulse_duration_range=(10, 15),
        hold_pulse_magnitude_range=(0.5, 1.5),
        hold_recovery_min_steps=10,
        include_task_indicator=False,
        timestep=0.01,
    ):
        if not 0.0 <= float(role_bias) <= 1.0:
            raise ValueError('role_bias must be in [0, 1]')
        if role_assignment_mode not in {'bernoulli', 'exact_balanced'}:
            raise ValueError(
                "role_assignment_mode must be 'bernoulli' or 'exact_balanced'")
        if (role_assignment_mode == 'exact_balanced'
                and not np.isclose(float(role_bias), 0.5)):
            raise ValueError('exact_balanced role assignment requires role_bias=0.5')
        if duration_ms <= 0:
            raise ValueError('duration_ms must be positive')
        amp_low, amp_high = map(float, trajectory_amplitude_range)
        freq_low, freq_high = map(float, trajectory_frequency_range)
        if not 0.0 < amp_low <= amp_high:
            raise ValueError('trajectory_amplitude_range must be positive and ordered')
        if not 0.0 < freq_low <= freq_high:
            raise ValueError('trajectory_frequency_range must be positive and ordered')
        if spring_stiffness < 0 or damping_coefficient < 0:
            raise ValueError('spring and damping coefficients must be non-negative')
        if max_coupling_force is not None and max_coupling_force <= 0:
            raise ValueError('max_coupling_force must be positive or None')
        if hold_perturbation_mode not in {'none', 'pulse'}:
            raise ValueError("hold_perturbation_mode must be 'none' or 'pulse'")

        # Reuse the established 46-D bimanual observation and paired MotorNet
        # effectors. The parent stores and samples the established hold-pulse
        # parameters. Task4 selects that pulse only for the assigned hold hand.
        super().__init__(
            bimanual_arms=bimanual_arms,
            device=device,
            hold_duration_ms=duration_ms,
            external_force_std=0.0,
            perturbation_mode=(
                'pulse' if hold_perturbation_mode == 'pulse' else 'iid'),
            pulse_onset_range=hold_pulse_onset_range,
            pulse_duration_range=hold_pulse_duration_range,
            pulse_magnitude_range=hold_pulse_magnitude_range,
            recovery_min_steps=hold_recovery_min_steps,
            include_task_indicator=include_task_indicator,
            timestep=timestep,
        )
        self.duration_ms = int(duration_ms)
        self.task4_steps = self.hold_steps
        self.role_bias = float(role_bias)
        self.role_assignment_mode = role_assignment_mode
        self.trajectory_amplitude_range = (amp_low, amp_high)
        self.trajectory_frequency_range = (freq_low, freq_high)
        self.spring_stiffness = float(spring_stiffness)
        self.damping_coefficient = float(damping_coefficient)
        self.max_coupling_force = (
            None if max_coupling_force is None else float(max_coupling_force))
        self.hold_perturbation_mode = hold_perturbation_mode

        self.track_right = None
        self.trajectory_amplitude = None
        self.trajectory_frequency = None
        self.trajectory_direction = None
        self.rest_relative_position = None
        self._state_world_left = None
        self._state_world_right = None

    def _sample_trajectory(self, batch_size):
        amp_low, amp_high = self.trajectory_amplitude_range
        freq_low, freq_high = self.trajectory_frequency_range
        self.trajectory_amplitude = (
            amp_low + (amp_high - amp_low) * torch.rand(batch_size, device=self.device))
        self.trajectory_frequency = (
            freq_low + (freq_high - freq_low) * torch.rand(batch_size, device=self.device))
        # A vertical trajectory is symmetric across hands and keeps the
        # tracking demand identical under canonical and reversed assignments.
        direction_sign = torch.where(
            torch.rand(batch_size, device=self.device) < 0.5,
            -torch.ones(batch_size, device=self.device),
            torch.ones(batch_size, device=self.device),
        )
        self.trajectory_direction = torch.stack(
            [torch.zeros_like(direction_sign), direction_sign], dim=-1)

    def _targets_for_step(self, step):
        time_s = float(step) * self.timestep
        phase = 2.0 * np.pi * self.trajectory_frequency * time_s
        displacement = (
            self.trajectory_amplitude * torch.sin(phase)).unsqueeze(-1)
        displacement = displacement * self.trajectory_direction
        target_left = self.start_pos_left + torch.where(
            self.track_right.unsqueeze(-1), torch.zeros_like(displacement), displacement)
        target_right = self.start_pos_right + torch.where(
            self.track_right.unsqueeze(-1), displacement, torch.zeros_like(displacement))
        return target_left, target_right

    def reset(self, batch_size=1, options=None):
        """Reset Task4 and sample roles and a smooth rhythmic trajectory."""
        options = {} if options is None else options
        role_bias = float(options.get('role_bias', self.role_bias))
        role_assignment_mode = options.get(
            'role_assignment_mode', self.role_assignment_mode)
        if not 0.0 <= role_bias <= 1.0:
            raise ValueError('reset role_bias must be in [0, 1]')
        if role_assignment_mode not in {'bernoulli', 'exact_balanced'}:
            raise ValueError(
                "reset role_assignment_mode must be 'bernoulli' or 'exact_balanced'")
        if role_assignment_mode == 'exact_balanced':
            if not np.isclose(role_bias, 0.5):
                raise ValueError('exact_balanced role assignment requires role_bias=0.5')
            if batch_size % 2 != 0:
                raise ValueError(
                    'exact_balanced role assignment requires an even batch size')

        obs_left, info_left = self._env_left.reset(batch_size=batch_size)
        obs_right, info_right = self._env_right.reset(batch_size=batch_size)
        state_local_left = info_left['states']['cartesian'][:, :4]
        state_local_right = info_right['states']['cartesian'][:, :4]
        self._state_world_left = self.bimanual_arms.transform_left_to_world(
            state_local_left).to(self.device)
        self._state_world_right = self.bimanual_arms.transform_right_to_world(
            state_local_right).to(self.device)
        self.start_pos_left = self._state_world_left[:, :2].clone()
        self.start_pos_right = self._state_world_right[:, :2].clone()
        self.rest_relative_position = self.start_pos_right - self.start_pos_left

        if role_assignment_mode == 'exact_balanced':
            canonical = torch.cat([
                torch.ones(batch_size // 2, dtype=torch.bool, device=self.device),
                torch.zeros(batch_size // 2, dtype=torch.bool, device=self.device),
            ])
            self.track_right = canonical[
                torch.randperm(batch_size, device=self.device)]
        else:
            self.track_right = torch.rand(batch_size, device=self.device) < role_bias
        self._sample_trajectory(batch_size)
        self.current_step = 0
        self.position_history_left = []
        self.position_history_right = []
        if self.hold_perturbation_mode == 'pulse':
            self._sample_pulses(batch_size)
        else:
            # Historical Task4 checkpoints had no independent hold pulse.
            # Treat their full episode as the stability evaluation window.
            self.pulse_onset = torch.zeros(
                batch_size, dtype=torch.long, device=self.device)
            self.pulse_end = torch.full(
                (batch_size,), self.task4_steps,
                dtype=torch.long, device=self.device)
            self.pulse_force_left = torch.zeros(batch_size, 2, device=self.device)
            self.pulse_force_right = torch.zeros(batch_size, 2, device=self.device)

        # The first observation exposes the desired target for the first
        # physical control interval; no separate task/role indicator is added.
        target_left, target_right = self._targets_for_step(1)
        self.targets_left = target_left
        self.targets_right = target_right
        self.current_targets_left = target_left
        self.current_targets_right = target_right
        obs = self._get_bimanual_obs(
            obs_left.to(self.device), obs_right.to(self.device), info_left, info_right)
        info = {
            'states_left': info_left['states'],
            'states_right': info_right['states'],
            'start_pos_left': self.start_pos_left,
            'start_pos_right': self.start_pos_right,
            'pos_world_left': self.start_pos_left,
            'pos_world_right': self.start_pos_right,
            'current_targets_left': self.current_targets_left,
            'current_targets_right': self.current_targets_right,
            'track_right': self.track_right,
            'hold_left': self.track_right,
            'role_bias_used': role_bias,
            'role_assignment_mode': role_assignment_mode,
            'canonical_role_count': int(self.track_right.sum().item()),
            'reversed_role_count': int((~self.track_right).sum().item()),
            'trajectory_amplitude': self.trajectory_amplitude,
            'trajectory_frequency': self.trajectory_frequency,
            'trajectory_direction': self.trajectory_direction,
            'pulse_onset': self.pulse_onset,
            'pulse_end': self.pulse_end,
            'pulse_force_left': self.pulse_force_left,
            'pulse_force_right': self.pulse_force_right,
        }
        return obs, info

    def _coupling_forces_world(self):
        pos_left, pos_right = self._state_world_left[:, :2], self._state_world_right[:, :2]
        vel_left, vel_right = self._state_world_left[:, 2:4], self._state_world_right[:, 2:4]
        displacement = (pos_right - pos_left) - self.rest_relative_position
        relative_velocity = vel_right - vel_left
        force_left = (
            self.spring_stiffness * displacement
            + self.damping_coefficient * relative_velocity)
        if self.max_coupling_force is not None:
            norm = torch.norm(force_left, dim=-1, keepdim=True)
            scale = torch.clamp(self.max_coupling_force / (norm + 1e-8), max=1.0)
            force_left = force_left * scale
        return force_left, -force_left

    def step(self, action):
        """Advance both hands under optional coupling and a hold-hand pulse."""
        batch_size = action.shape[0]
        controlled_target_left = self.current_targets_left
        controlled_target_right = self.current_targets_right
        coupling_left_world, coupling_right_world = self._coupling_forces_world()
        phase = self.get_phase_masks(next_step=True)
        pulse_mask = phase['pulse'].unsqueeze(-1)
        pulse_left_world = self.pulse_force_left * pulse_mask
        pulse_right_world = self.pulse_force_right * pulse_mask
        hold_left = self.track_right.unsqueeze(-1)
        holding_force_left_world = torch.where(
            hold_left, pulse_left_world, torch.zeros_like(pulse_left_world))
        holding_force_right_world = torch.where(
            hold_left, torch.zeros_like(pulse_right_world), pulse_right_world)
        force_left_world = coupling_left_world + holding_force_left_world
        force_right_world = coupling_right_world + holding_force_right_world
        force_left_local = self.bimanual_arms.transform_left_force_world_to_local(
            force_left_world)
        force_right_local = self.bimanual_arms.transform_right_force_world_to_local(
            force_right_world)

        obs_left, reward_left, term_left, trunc_left, info_left = self._env_left.step(
            action[:, :6], endpoint_load=force_left_local)
        obs_right, reward_right, term_right, trunc_right, info_right = self._env_right.step(
            action[:, 6:], endpoint_load=force_right_local)
        self._state_world_left = self.bimanual_arms.transform_left_to_world(
            info_left['states']['cartesian'][:, :4]).to(self.device)
        self._state_world_right = self.bimanual_arms.transform_right_to_world(
            info_right['states']['cartesian'][:, :4]).to(self.device)
        pos_left, pos_right = self._state_world_left[:, :2], self._state_world_right[:, :2]
        self.position_history_left.append(pos_left.detach().clone())
        self.position_history_right.append(pos_right.detach().clone())

        error_left = torch.norm(pos_left - controlled_target_left, dim=-1)
        error_right = torch.norm(pos_right - controlled_target_right, dim=-1)
        self.current_step += 1
        next_step = min(self.current_step + 1, self.task4_steps)
        self.current_targets_left, self.current_targets_right = self._targets_for_step(next_step)
        obs = self._get_bimanual_obs(
            obs_left.to(self.device), obs_right.to(self.device), info_left, info_right)

        if reward_left is None:
            reward_left = torch.zeros(batch_size, device=self.device)
        if reward_right is None:
            reward_right = torch.zeros(batch_size, device=self.device)
        terminated = (
            torch.ones(batch_size, dtype=torch.bool, device=self.device)
            if self.current_step >= self.task4_steps else term_left | term_right)
        info = {
            'states_left': info_left['states'],
            'states_right': info_right['states'],
            'pos_world_left': pos_left,
            'pos_world_right': pos_right,
            'controlled_target_left': controlled_target_left,
            'controlled_target_right': controlled_target_right,
            'current_targets_left': self.current_targets_left,
            'current_targets_right': self.current_targets_right,
            'error_left': error_left,
            'error_right': error_right,
            'track_right': self.track_right,
            'hold_left': self.track_right,
            'perturbation_phase': self.get_phase_masks(),
            'holding_force_left_world': holding_force_left_world,
            'holding_force_right_world': holding_force_right_world,
            'endpoint_force_left_world': force_left_world,
            'endpoint_force_right_world': force_right_world,
            'coupling_force_left_world': coupling_left_world,
            'coupling_force_right_world': coupling_right_world,
        }
        return obs, reward_left + reward_right, terminated, trunc_left | trunc_right, info
