import numpy as np

from .base import RewardFunction
from adv_building_gym.components.registry import ComponentRegistry


class ActionSmoothnessReward(RewardFunction):
    """Penalise high-frequency oscillation in actions via spectral analysis.

    For each action key, the reward maintains its own circular ring buffer of
    the last ``n_steps`` actions (zero-padded until full). On every step the
    just-taken action is appended and a real FFT (``numpy.fft.rfft``) is
    computed across the time axis per action dimension. The squared magnitude
    of the spectrum, |X[k]|², is the "energy" at frequency bin ``k``; bin 0
    is DC and bin ``N/2`` is the Nyquist frequency (an action that flips every
    single step). Energy concentrated near Nyquist is the hallmark of
    actuator oscillation that this reward is meant to discourage.

    Two operating modes are supported (selectable via ``mode``):

    1. ``mode='weighted'`` (default) — frequency-weighted spectral energy.
       Each bin's energy is multiplied by ``w[k] = (k / (N/2)) ** freq_exponent``,
       so DC has weight 0, Nyquist has weight 1, and low-frequency drifts get
       small weight. The total weighted energy is divided by the analytical
       worst case (see below) to yield a per-dim penalty in ``[0, 1]``.
       Smooth gradient w.r.t. frequency — preferred for RL because the policy
       receives a continuous signal that grows as oscillations get faster.

    2. ``mode='highband'`` — hard cutoff. Only bins above
       ``cutoff_bin = round(cutoff_fraction * (N/2))`` contribute. Equivalent
       to a brick-wall high-pass filter on the spectrum. Trends and ramps
       below the cutoff produce zero penalty. Use when you want a sharp
       distinction between "smooth control" and "oscillation".

    Per-key penalty aggregation: mean across action dimensions within a key,
    then **sum** across action keys. With per-key penalty bounded in ``[0, 1]``
    this gives ``osc ∈ [0, n_keys]`` and final raw reward ``-osc`` in
    ``[-n_keys, 0]`` — a pure penalty. ``self.max_reward_in_step`` is set to
    ``n_keys`` on first call so the breakdown / reward_rate denominator is
    correctly scaled.

    Worst-case derivation: for an action sequence ``a[n] = (-1)^n`` (a
    ±1 square wave of length ``N``), ``rfft`` concentrates all energy at the
    Nyquist bin with ``|X[N/2]|² = N²``. Both modes (weighted with weight 1
    at Nyquist, and highband if the cutoff lies below Nyquist) include this
    bin, so the worst-case is ``N²`` per dimension in both modes.

    See ``docs/action_smoothness_spectral.md`` for the full derivation and
    a worked numerical example.

    References:
    - ``numpy.fft.rfft``: https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
    - Parseval's theorem: https://en.wikipedia.org/wiki/Parseval%27s_theorem
    """

    # TODO VP 2026.04.24. : Read reference: Higher-Order Action Regularisation for RL in Building
    # Energy Management (NeurIPS 2025 UrbanAI Workshop)
    # Link: https://arxiv.org/abs/2601.02061

    # Initial sentinel value; overwritten on first get_reward() call once the
    # number of action keys is known (see `__init__` docstring).
    max_reward_in_step: float = 1.0

    # Internal mutable state (buffers, write heads, derived n_keys) must not
    # be round-tripped through YAML — they are reconstructed at runtime.
    _exclude_params = {"_buffers", "_head", "_n_action_keys"}

    def __init__(
        self,
        weight: float,
        name: str = "action_smoothness",
        n_steps: int = 16,
        mode: str = "weighted",
        freq_exponent: float = 1.0,
        cutoff_fraction: float = 0.5,
        max_reward_override: float | None = None,
    ) -> None:
        """Construct the reward.

        Args:
            weight: Linear scale applied to both ``reward`` and ``max_step``.
            name: Reward identifier (used in the breakdown dict).
            n_steps: FFT window length in control steps. Must be an even
                integer >= 4. The buffer is pre-allocated and zero-padded
                until ``n_steps`` real actions have been written.
            mode: ``'weighted'`` or ``'highband'`` — see class docstring.
            freq_exponent: Exponent ``p`` in ``w[k] = (k / (N/2))^p``
                (``mode='weighted'`` only). ``p > 1`` weights high freqs more
                aggressively; ``p < 1`` softens the differentiation.
            cutoff_fraction: Fraction of Nyquist above which bins contribute
                to the penalty (``mode='highband'`` only). ``0.5`` means only
                the upper half of the spectrum is penalised.
            max_reward_override: If not None, fixes ``max_reward_in_step`` to
                this value instead of auto-scaling to ``n_keys`` on first
                call. Mostly useful for tests.
        """
        super().__init__(weight, name)
        if n_steps < 4 or n_steps % 2 != 0:
            raise ValueError("n_steps must be an even integer >= 4")
        if mode not in ("weighted", "highband"):
            raise ValueError("mode must be 'weighted' or 'highband'")
        if not 0.0 <= cutoff_fraction <= 1.0:
            raise ValueError("cutoff_fraction must be in [0, 1]")

        self.n_steps = n_steps
        self.mode = mode
        self.freq_exponent = freq_exponent
        self.cutoff_fraction = cutoff_fraction
        self.max_reward_override = max_reward_override

        # Per-key ring buffer: shape (n_steps, D), float32, zero-padded.
        # Allocated lazily on first sight of each action key (D known then).
        self._buffers: dict[str, np.ndarray] = {}
        # Per-key write head — index of the *next* slot to overwrite.
        self._head: dict[str, int] = {}
        # Resolved on first get_reward() call.
        self._n_action_keys: int = 0

    def on_reset(self, states, info: dict | None = None) -> None:
        """Zero every buffer and reset write heads at episode start.

        Allocations are retained across episodes (amortised cost). On a
        reward hot-swap via the reward_switch callback a *new* instance is
        constructed, so buffers always start fresh there anyway.
        """
        for buf in self._buffers.values():
            buf.fill(0.0)
        for key in list(self._head.keys()):
            self._head[key] = 0

    def get_reward(self, actions: dict, states: dict, info: dict | None = None) -> tuple[float, float]:
        # Lazy one-time setup: scale max_reward_in_step to the number of
        # action keys so the reward range is [-n_keys, 0] and reward_rate
        # denominators stay correctly proportioned across configs with
        # different action-space sizes.
        if self._n_action_keys == 0:
            self._n_action_keys = len(actions)
            self.max_reward_in_step = (
                float(self.max_reward_override)
                if self.max_reward_override is not None
                else float(self._n_action_keys)
            )
        max_step = self.weight * self.max_reward_in_step

        # Sum per-key penalties (each in [0, 1]) across action keys.
        osc_total = 0.0
        for key, current_action in actions.items():
            a = np.atleast_1d(current_action).astype(np.float32).reshape(-1)
            D = a.shape[0]

            # Lazily allocate the (n_steps, D) ring buffer for this key.
            buf = self._buffers.get(key)
            if buf is None or buf.shape[1] != D:
                buf = np.zeros((self.n_steps, D), dtype=np.float32)
                self._buffers[key] = buf
                self._head[key] = 0

            # Write the just-taken action into the circular buffer.
            head = self._head[key]
            buf[head] = a
            self._head[key] = (head + 1) % self.n_steps

            # Reorder into time-ordered view: row 0 = oldest, row -1 = newest.
            # np.roll with -head puts the next-write slot at row 0 (which is
            # the oldest entry) and the just-written sample at row -1.
            x = np.roll(buf, -self._head[key], axis=0)

            osc_total += self._spectral_penalty(x)

        if self._n_action_keys == 0:
            return 0.0, max_step

        # Pure-penalty form: raw_reward in [-max_reward_in_step, 0].
        raw_reward = -osc_total
        return float(self.weight * raw_reward), max_step

    def _spectral_penalty(self, x: np.ndarray) -> float:
        """Return per-key spectral penalty in [0, 1] for time-ordered buffer.

        ``x`` has shape ``(N, D)``. Steps:

        1. Mean-subtract per dim (drops DC explicitly — numerically cleaner
           than relying on the FFT's bin 0).
        2. ``rfft`` along axis 0 → ``X`` of shape ``(K+1, D)``, ``K = N//2``.
        3. Energy ``E = |X|²``.
        4. Apply mode-specific bin weighting / cutoff.
        5. Normalise by ``N²`` (worst case for ±1 amplitude over the window).
        6. Average across the D action dimensions.
        """
        N = self.n_steps
        K = N // 2  # Nyquist bin index

        x = x - x.mean(axis=0, keepdims=True)
        X = np.fft.rfft(x, axis=0)            # shape (K+1, D)
        E = (X.real ** 2 + X.imag ** 2)       # shape (K+1, D); |X|²

        if self.mode == "weighted":
            # Frequency-weighted: w[0] = 0 (DC), w[K] = 1 (Nyquist).
            # The (k/K) factor monotonically increases with bin index, so
            # high-frequency energy dominates the sum.
            k = np.arange(K + 1, dtype=np.float32)
            w = (k / K) ** self.freq_exponent      # shape (K+1,)
            weighted = E * w[:, None]              # broadcast over D
            energy_per_dim = weighted.sum(axis=0)  # shape (D,)
        else:
            # 'highband': hard cutoff — only bins strictly above
            # ``cutoff_bin`` contribute. cutoff_fraction == 0 means all bins
            # except DC; cutoff_fraction == 1 means none (penalty 0).
            cutoff_bin = int(round(self.cutoff_fraction * K))
            # Slice from cutoff_bin+1 .. K inclusive.
            if cutoff_bin >= K:
                return 0.0
            energy_per_dim = E[cutoff_bin + 1 :].sum(axis=0)  # shape (D,)

        # Worst-case (per dim) is the ±1 square wave with all energy at
        # Nyquist: |X[K]|² = N². Both modes include the Nyquist bin (weighted
        # with weight 1, highband above any cutoff < 1.0), so the analytical
        # worst case is N² per dimension.
        worst_per_dim = float(N * N)
        penalty_per_dim = energy_per_dim / worst_per_dim   # shape (D,)
        # Mean across dims so single-D and multi-D actions are comparable.
        # Clip into [0, 1] to absorb numerical noise.
        return float(np.clip(penalty_per_dim.mean(), 0.0, 1.0))


ComponentRegistry.register('reward', ActionSmoothnessReward)
