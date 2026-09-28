import math

import torch
import torch.nn as nn


class MultiplicativeNoiseLayer(nn.Module):
    """
    Apply signal-dependent Gaussian noise to muscle activations.

    The noisy command is ``u * (1 + k * epsilon)`` with standard normal
    ``epsilon`` and is clipped to the valid activation range ``[0, 1]``.
    """
    def __init__(self, noise_gain=0.0):
        """Store the gain used for signal-dependent action noise."""
        super().__init__()
        self.noise_gain = noise_gain

    def forward(self, u):
        """Return noisy muscle activations with the configured gain."""
        if self.noise_gain <= 1e-9:
            return u
        
        epsilon = torch.randn_like(u)
        noisy_u = u * (1.0 + self.noise_gain * epsilon)
        return torch.clamp(noisy_u, 0.0, 1.0)


class DelayBuffer:
    """
    Store recurrent states for delayed interhemispheric communication.

    A circular buffer persists across calls within an episode. After a state
    is pushed, ``get_delayed`` returns the state requested by the configured
    communication delay.
    """
    def __init__(self, delay_steps, hidden_size, device, semantics='causal_exact'):
        """Create storage for a delayed hidden state trajectory."""
        self.delay_steps = delay_steps
        self.semantics = semantics
        self.hidden_size = hidden_size
        self.device = device
        self.batch_size = None  # Set when an episode begins.
        self.buffer = None
        self.write_ptr = 0
        
        if semantics not in {'causal_exact', 'legacy_extra_step'}:
            raise ValueError("delay semantics must be 'causal_exact' or 'legacy_extra_step'")
        # Include one empty/start-state slot while the requested history fills.
        self.buf_size = delay_steps + 1 if delay_steps > 0 else 0
    
    def reset(self, batch_size):
        """Initialise the circular buffer for one episode batch."""
        self.batch_size = batch_size
        self.write_ptr = 0
        if self.delay_steps > 0:
            self.buffer = torch.zeros(batch_size, self.buf_size, self.hidden_size).to(self.device)
    
    def push(self, state):
        """Store the current recurrent state in the circular buffer."""
        if self.delay_steps == 0:
            return

        self.buffer[:, self.write_ptr, :] = state
        
        self.write_ptr = (self.write_ptr + 1) % self.buf_size

    def get_delayed(self, current_state=None):
        """
        Return the causally available source state for the receiving hemisphere.

        ``current_state`` is the source state from the preceding recurrent
        update.  With a positive exact delay ``delta``, the receiver at update
        ``t`` reads ``h[t-delta]``.  A zero transport delay uses the latest
        causal state ``h[t-1]``.  ``legacy_extra_step`` preserves checkpoints
        trained with the earlier ``h[t-1-delta]`` indexing.
        """
        if self.delay_steps == 0:
            return current_state
        
        # Before the next push, write_ptr identifies the slot after the newest
        # stored source state.  Exact indexing therefore subtracts delta, not
        # delta+1.  Keep the historical branch only for schema-v2 checkpoints.
        offset = self.delay_steps if self.semantics == 'causal_exact' else 1 + self.delay_steps
        read_ptr = (self.write_ptr - offset) % self.buf_size
        return self.buffer[:, read_ptr, :]

class BilateralNetwork(nn.Module):
    """
    Two-module recurrent controller used by Bilateral w/ CC and Bilateral w/o CC.

    Each hemisphere is represented by a GRU cell. In the connected model,
    learned corpus-callosum projections carry delayed hidden-state signals
    between hemispheres. The two readouts contribute to a shared 12-muscle
    action, which permits the contribution-based contralaterality index.
    """
    def __init__(self, input_dim, output_dim, hidden_size,
                 conduction_delay_steps=0,
                 delay_semantics='causal_exact',
                 noise_gain=0.0,
                 contra_fraction=0.8,
                 routing_normalization='rms',
                 cc_mode='full',
                 cc_bottleneck_dim=8,
                 shared_bias_trainable=True,
                 ensemble=False,
                 device='cpu'):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.delay_steps = conduction_delay_steps
        self.delay_semantics = delay_semantics
        self.effective_cc_lag_steps = max(1, conduction_delay_steps)
        self.ensemble = ensemble
        self.device = device
        self.contra_fraction = float(contra_fraction)
        self.routing_normalization = routing_normalization
        self.cc_mode = 'none' if ensemble else cc_mode
        self.cc_bottleneck_dim = int(cc_bottleneck_dim)
        self.shared_bias_trainable = bool(shared_bias_trainable)
        self.model_family = 'bilateral'
        self.supports_controller_interventions = True

        if output_dim % 2 != 0:
            raise ValueError('output_dim must contain equal left/right motor halves')
        if not 0.0 <= self.contra_fraction <= 1.0:
            raise ValueError('contra_fraction must be in [0, 1]')
        if routing_normalization not in {'rms', 'none'}:
            raise ValueError("routing_normalization must be 'rms' or 'none'")
        if self.cc_mode not in {'full', 'bottleneck', 'none'}:
            raise ValueError("cc_mode must be 'full', 'bottleneck', or 'none'")
        if self.cc_mode == 'bottleneck' and not 1 <= self.cc_bottleneck_dim <= hidden_size:
            raise ValueError('cc_bottleneck_dim must be between 1 and hidden_size')

        # Each GRU receives sensory input and a hidden-sized cross input.
        gru_input_size = input_dim + hidden_size

        self.gru_left = nn.GRUCell(input_size=gru_input_size, hidden_size=hidden_size)
        self.gru_right = nn.GRUCell(input_size=gru_input_size, hidden_size=hidden_size)

        # Full and bottleneck CC share the same hidden-sized receiver input.
        if self.cc_mode == 'full':
            self.w_l2r = nn.Linear(hidden_size, hidden_size, bias=False)
            self.w_r2l = nn.Linear(hidden_size, hidden_size, bias=False)
            self.cc_down_l2r = self.cc_up_l2r = None
            self.cc_down_r2l = self.cc_up_r2l = None
        elif self.cc_mode == 'bottleneck':
            self.w_l2r = self.w_r2l = None
            self.cc_down_l2r = nn.Linear(hidden_size, self.cc_bottleneck_dim, bias=False)
            self.cc_up_l2r = nn.Linear(self.cc_bottleneck_dim, hidden_size, bias=False)
            self.cc_down_r2l = nn.Linear(hidden_size, self.cc_bottleneck_dim, bias=False)
            self.cc_up_r2l = nn.Linear(self.cc_bottleneck_dim, hidden_size, bias=False)
        else:
            self.w_l2r = self.w_r2l = None
            self.cc_down_l2r = self.cc_up_l2r = None
            self.cc_down_r2l = self.cc_up_r2l = None

        # Separate bias-free readouts expose each module's motor contribution.
        self.readout_l = nn.Linear(hidden_size, output_dim, bias=False)
        self.readout_r = nn.Linear(hidden_size, output_dim, bias=False)
        self.shared_bias = nn.Parameter(
            torch.zeros(output_dim),
            requires_grad=self.shared_bias_trainable,
        )

        # Fixed anatomical routing gains.  For the left arm the right module
        # is contralateral; for the right arm the left module is contralateral.
        p = self.contra_fraction
        scale = math.sqrt(2.0 / (p * p + (1.0 - p) ** 2)) \
            if routing_normalization == 'rms' else 1.0
        half = output_dim // 2
        left_gains = torch.cat([
            torch.full((half,), scale * (1.0 - p)),
            torch.full((half,), scale * p),
        ])
        right_gains = torch.cat([
            torch.full((half,), scale * p),
            torch.full((half,), scale * (1.0 - p)),
        ])
        self.register_buffer('routing_gain_left', left_gains)
        self.register_buffer('routing_gain_right', right_gains)

        # Convert combined motor logits into bounded, noisy muscle commands.
        self.sigmoid = nn.Sigmoid()
        self.noise_layer = MultiplicativeNoiseLayer(noise_gain=noise_gain)

        # Delay buffers preserve interhemispheric history within an episode.
        self.buffer_l = DelayBuffer(
            conduction_delay_steps, hidden_size, device, semantics=delay_semantics)
        self.buffer_r = DelayBuffer(
            conduction_delay_steps, hidden_size, device, semantics=delay_semantics)

        self.reset_parameters()
    
    def reset_buffers(self, batch_size):
        """
        Reset interhemispheric delay state at the start of an episode.

        Rollout code must call this method before the first forward pass.
        """
        self.buffer_l.reset(batch_size)
        self.buffer_r.reset(batch_size)

    def reset_parameters(self):
        """Initialise recurrent, callosal and readout weights."""
        # Xavier initialisation is used for recurrent and readout weights.
        for name, param in self.named_parameters():
            if 'gru' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
        
        # Callosal connections start near zero to permit learned organisation.
        if self.cc_mode == 'full':
            nn.init.normal_(self.w_l2r.weight, mean=0.0, std=0.01)
            nn.init.normal_(self.w_r2l.weight, mean=0.0, std=0.01)
        elif self.cc_mode == 'bottleneck':
            for layer in (
                self.cc_down_l2r, self.cc_up_l2r,
                self.cc_down_r2l, self.cc_up_r2l,
            ):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)

        nn.init.xavier_uniform_(self.readout_l.weight)
        nn.init.xavier_uniform_(self.readout_r.weight)
        nn.init.constant_(self.shared_bias, 0.1)

    def functional_parameter_groups(self):
        """
        Return receiver-owned parameter groups for functional loss routing.

        A controller owns its recurrent cell, motor readout and incoming CC
        projection. The common motor bias is reported separately so fixed-role
        training can prevent it from becoming an unassigned optimisation path.
        """
        left = list(self.gru_left.parameters()) + list(self.readout_l.parameters())
        right = list(self.gru_right.parameters()) + list(self.readout_r.parameters())

        if self.cc_mode == 'full':
            left += list(self.w_r2l.parameters())
            right += list(self.w_l2r.parameters())
        elif self.cc_mode == 'bottleneck':
            left += list(self.cc_down_r2l.parameters())
            left += list(self.cc_up_r2l.parameters())
            right += list(self.cc_down_l2r.parameters())
            right += list(self.cc_up_l2r.parameters())

        return {
            'left': tuple(left),
            'right': tuple(right),
            'shared': (self.shared_bias,),
        }

    @staticmethod
    def _expand_mask(mask, reference):
        """Broadcast scalar or per-trial intervention gains over features."""
        if torch.is_tensor(mask):
            mask = mask.to(device=reference.device, dtype=reference.dtype)
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
        return mask

    def _project_cc(self, state, direction):
        """Project one source state through the configured CC pathway."""
        if self.cc_mode == 'none':
            return torch.zeros_like(state)
        if self.cc_mode == 'full':
            return self.w_l2r(state) if direction == 'l2r' else self.w_r2l(state)
        if direction == 'l2r':
            return self.cc_up_l2r(self.cc_down_l2r(state))
        return self.cc_up_r2l(self.cc_down_r2l(state))

    def forward(self, inputs, hidden_state=None, lesion_mask=None, intervention=None):
        """
        Args:
            inputs: [batch, time, input_dim]
            hidden_state: (h_l, h_r) tuple. If None, zeros.
            lesion_mask: Backward-compatible whole-module gain dictionary.
            intervention: Optional granular gains with ``cc_left``, ``cc_right``,
                ``readout_left`` and ``readout_right``. Gains may be scalars or
                per-trial tensors, enabling graded and phase-specific inference.
        Returns:
            outputs: noisy/clipped plant commands [batch, time, output_dim]
            hidden_states: (h_l_stack, h_r_stack)
            routed_contributions: post-routing, post-intervention contributions
            raw_contributions: pre-routing, pre-intervention readout logits
            deterministic_actions: pre-noise bounded muscle commands used for
                energy and co-contraction measurements
        """
        batch_size, time_steps, _ = inputs.shape

        # A lesion acts on both outgoing communication and motor contribution.
        if lesion_mask is not None and intervention is not None:
            raise ValueError('pass lesion_mask or intervention, not both')
        masks = intervention if intervention is not None else lesion_mask
        if masks is None:
            cc_mask_l, cc_mask_r = 1.0, 1.0
            ro_mask_l, ro_mask_r = 1.0, 1.0
        elif any(key.startswith(('cc_', 'readout_')) for key in masks):
            cc_mask_l = masks.get('cc_left', 1.0)
            cc_mask_r = masks.get('cc_right', 1.0)
            ro_mask_l = masks.get('readout_left', 1.0)
            ro_mask_r = masks.get('readout_right', 1.0)
        else:
            cc_mask_l = ro_mask_l = masks.get('left', 1.0)
            cc_mask_r = ro_mask_r = masks.get('right', 1.0)
        
        # Hidden states initialise to rest for each independent rollout.
        if hidden_state is None:
            h_l = torch.zeros(batch_size, self.hidden_size).to(self.device)
            h_r = torch.zeros(batch_size, self.hidden_size).to(self.device)
        else:
            h_l, h_r = hidden_state

        # The rollout initialises these persistent buffers before forwarding.
        buffer_l = self.buffer_l
        buffer_r = self.buffer_r

        # Keep complete trajectories for metrics and recurrent propagation.
        outputs_list = []
        h_l_list = []
        h_r_list = []
        c_l_list = []  # Left-module action contribution.
        c_r_list = []  # Right-module action contribution.
        raw_c_l_list = []
        raw_c_r_list = []
        deterministic_actions_list = []

        # Advance the recurrent controller one simulator step at a time.
        for t in range(time_steps):
            x_t = inputs[:, t, :] 
            
            # Retrieve the source states available through the communication lag.
            delayed_h_l = buffer_l.get_delayed(current_state=h_l)
            delayed_h_r = buffer_r.get_delayed(current_state=h_r)
            
            if delayed_h_l is not None and self.cc_mode != 'none':
                mask_l = self._expand_mask(cc_mask_l, delayed_h_l)
                cross_input_to_r = self._project_cc(delayed_h_l * mask_l, 'l2r')
            else:
                cross_input_to_r = torch.zeros(batch_size, self.hidden_size, device=self.device)

            if delayed_h_r is not None and self.cc_mode != 'none':
                mask_r = self._expand_mask(cc_mask_r, delayed_h_r)
                cross_input_to_l = self._project_cc(delayed_h_r * mask_r, 'r2l')
            else:
                cross_input_to_l = torch.zeros(batch_size, self.hidden_size, device=self.device)
            
            # Each module updates from shared sensory input and incoming signal.
            in_l = torch.cat([x_t, cross_input_to_l], dim=-1)
            in_r = torch.cat([x_t, cross_input_to_r], dim=-1)

            h_l = self.gru_left(in_l, h_l)
            h_r = self.gru_right(in_r, h_r)
            
            # Save the newly computed source state for future delayed reads.
            if self.delay_steps > 0:
                buffer_l.push(h_l)
                buffer_r.push(h_r)

            # Sum hemisphere-specific motor contributions before activation.
            # c_l and c_r exclude bias and define the CI measurement.
            raw_c_l = self.readout_l(h_l)
            raw_c_r = self.readout_r(h_r)
            mask_l = self._expand_mask(ro_mask_l, raw_c_l)
            mask_r = self._expand_mask(ro_mask_r, raw_c_r)
            c_l = raw_c_l * mask_l * self.routing_gain_left
            c_r = raw_c_r * mask_r * self.routing_gain_right

            raw_out = c_l + c_r + self.shared_bias
            activation = self.sigmoid(raw_out)
            noisy_activation = self.noise_layer(activation)

            outputs_list.append(noisy_activation)
            deterministic_actions_list.append(activation)
            h_l_list.append(h_l)
            h_r_list.append(h_r)
            c_l_list.append(c_l)  # Contributions before noise define CI.
            c_r_list.append(c_r)
            raw_c_l_list.append(raw_c_l)
            raw_c_r_list.append(raw_c_r)

        # Restore the [batch, time, feature] convention used by callers.
        outputs = torch.stack(outputs_list, dim=1)
        h_l_stack = torch.stack(h_l_list, dim=1)
        h_r_stack = torch.stack(h_r_list, dim=1)
        c_l_stack = torch.stack(c_l_list, dim=1)
        c_r_stack = torch.stack(c_r_list, dim=1)
        raw_c_l_stack = torch.stack(raw_c_l_list, dim=1)
        raw_c_r_stack = torch.stack(raw_c_r_list, dim=1)
        deterministic_actions = torch.stack(deterministic_actions_list, dim=1)

        return (
            outputs,
            (h_l_stack, h_r_stack),
            (c_l_stack, c_r_stack),
            (raw_c_l_stack, raw_c_r_stack),
            deterministic_actions,
        )


class UnilateralNetwork(nn.Module):
    """
    Single-module recurrent controller used as the monolithic baseline.

    A single GRU receives sensory observations and drives all muscle outputs.
    Its return structure matches the bilateral controller so the same training
    and evaluation routines can evaluate either model family.
    """

    def __init__(self, input_dim, output_dim, hidden_size=128,
                 noise_gain=0.0, device='cpu',
                 # Accepted so model construction follows one shared interface.
                 conduction_delay_steps=0):
        """Construct the single-module baseline controller."""
        super().__init__()

        self.hidden_size = hidden_size
        self.output_dim  = output_dim
        self.device      = device
        self.model_family = 'monolithic'
        self.supports_controller_interventions = False

        self.gru = nn.GRUCell(input_size=input_dim, hidden_size=hidden_size)

        self.readout = nn.Linear(hidden_size, output_dim, bias=False)
        self.bias    = nn.Parameter(torch.zeros(output_dim))

        self.sigmoid     = nn.Sigmoid()
        self.noise_layer = MultiplicativeNoiseLayer(noise_gain=noise_gain)

        self.reset_parameters()

    def reset_buffers(self, batch_size):
        """Accept episode resets through the shared controller interface."""
        pass

    def reset_parameters(self):
        """Initialise recurrent and readout weights."""
        for name, param in self.named_parameters():
            if 'gru' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.readout.weight)
        nn.init.constant_(self.bias, 0.1)

    def forward(self, inputs, hidden_state=None, lesion_mask=None, intervention=None):
        """
        Args:
            inputs: [batch, time, input_dim]
            hidden_state: Optional shared-interface tuple ``(h, _)``.
            lesion_mask: Accepted through the shared controller interface.

        Returns:
            outputs: [batch, time, output_dim]
            hidden_states: Repeated state pair in bilateral-compatible form.
            contributions: Repeated contribution pair in compatible form.
        """
        batch_size, time_steps, _ = inputs.shape

        # Use the first state slot exposed by the shared model interface.
        if hidden_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=self.device)
        else:
            h = hidden_state[0]

        outputs_list = []
        h_list       = []
        c_list       = []
        deterministic_actions_list = []

        for t in range(time_steps):
            x_t = inputs[:, t, :]

            h = self.gru(x_t, h)

            c        = self.readout(h)
            raw_out  = c + self.bias
            act      = self.sigmoid(raw_out)
            noisy    = self.noise_layer(act)

            outputs_list.append(noisy)
            deterministic_actions_list.append(act)
            h_list.append(h)
            c_list.append(c)

        outputs = torch.stack(outputs_list, dim=1)
        h_stack = torch.stack(h_list,       dim=1)
        c_stack = torch.stack(c_list,       dim=1)
        deterministic_actions = torch.stack(deterministic_actions_list, dim=1)

        # Repeat tensors so evaluation can use one controller return contract.
        return (
            outputs,
            (h_stack, h_stack),
            (c_stack, c_stack),
            (c_stack, c_stack),
            deterministic_actions,
        )


# Use an architecture-facing name while retaining the historical class name
# for checkpoint compatibility with earlier project code.
MonolithicNetwork = UnilateralNetwork
