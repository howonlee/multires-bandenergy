"""Randomized multiresolution band-energy search (sublinear in time samples).

The core idea: binary-search over frequency bands, where each yes/no decision
is a band-energy comparison. No FFT, and never a full pass over the signal.

Band energy is read off a small **autocorrelation sketch**. We fetch a few
random contiguous blocks of ``L << N`` samples, estimate the autocorrelation

    R(tau) = mean_n x[n] x[n + tau]   for tau = 0 .. M,

and form the energy in any band [f_lo, f_hi] analytically:

    E[f_lo, f_hi] = int_{f_lo}^{f_hi} S(f) df
                  = R(0) (f_hi - f_lo) + 2 sum_{tau>=1} R(tau) g(tau),

    g(tau) = int_{f_lo}^{f_hi} cos(2 pi f tau / fs) df   (the band kernel).

This is what raises the contrast per decision. A contiguous block gives
*coherent processing gain*: a tone shows up in R(tau) as a clean cosine of
amplitude ~A^2/2 at every lag, while broadband noise contributes only at
tau = 0. Scattered point samples -- in frequency or in time-pairs -- have no
such gain, so for a narrow tone in a long signal their band estimates are pure
variance and the greedy search derails. (See the prototypes that motivated
this in git history, and why_n_independent.md for the scaling argument.)

The sketch is computed **once** and answers every band query for free, so the
whole search reads only

    samples_read = n_blocks * block_len        (independent of N, depth, repeats)

with frequency resolution ~ fs / M set by the max lag M. To resolve finer
bands, raise M (and block_len) -- that, not N, is what the sample count is
coupled to.

This module provides:

    - ArraySampler:                 random-access / block reader over a signal
    - band_energy_kernel:           analytic band-integral kernel g(tau)  (pure)
    - autocorrelation_sketch:       R(tau) from random blocks
    - band_energy_from_sketch:      energy in [f_lo, f_hi] from a sketch   (pure)
    - multiresolution_band_search:  global dominant-band search on a sketch (pure)
    - local_peak_search:            hill-climb to a local peak on a sketch  (pure)
    - find_dominant_frequency_sublinear: convenience wrapper (global)
    - find_local_peak_sublinear:         convenience wrapper (local)
"""

import numpy as np


class ArraySampler:
    """Random-access / block sampler for a NumPy array.

    For huge signals this could instead read from mmap, disk, network, or
    compressed storage without changing the search code.

        sampler(indices) -> x[indices]            (point reads)
        sampler.block(start, length) -> x[start:start+length]  (contiguous read)
    """

    def __init__(self, x):
        self.x = np.asarray(x)
        self.N = len(self.x)

    def __call__(self, indices):
        return self.x[indices]

    def block(self, start, length):
        return self.x[start:start + length]


def band_energy_kernel(tau, f_lo, f_hi, fs):
    """Analytic band-integral kernel ``g(tau)``.

    ``g(tau) = int_{f_lo}^{f_hi} cos(2 pi f tau / fs) df``, the real, even
    kernel obtained by integrating the power spectral density over the band::

        g(tau) = fs/(2 pi tau) [sin(2 pi f_hi tau / fs) - sin(2 pi f_lo tau / fs)]   (tau != 0)
        g(0)   = f_hi - f_lo

    Pure function; ``tau`` may be a scalar or array of sample lags.
    """

    tau = np.asarray(tau, dtype=float)
    out = np.empty(tau.shape, dtype=float)

    nz = tau != 0.0
    tnz = tau[nz]
    out[nz] = (fs / (2.0 * np.pi * tnz)) * (
        np.sin(2.0 * np.pi * f_hi * tnz / fs)
        - np.sin(2.0 * np.pi * f_lo * tnz / fs)
    )
    out[~nz] = f_hi - f_lo
    return out


def autocorrelation_sketch(
    sampler,
    N,
    max_lag,
    n_blocks=4,
    block_len=None,
    rng=None,
    remove_mean=True,
):
    """Estimate the autocorrelation R(tau), tau = 0..max_lag, from random blocks.

    Reads ``n_blocks * block_len`` samples total -- the only signal access the
    whole search needs. Block starts are drawn uniformly at random and clamped
    so each block fits inside ``[0, N)``. Per-block lag products are averaged
    over the block (a biased/Bartlett autocorrelation, which keeps the kernel
    sum well-behaved), then averaged across blocks.

    Parameters
    ----------
    max_lag : int
        Largest lag M. Frequency resolution is ~ fs / M.
    n_blocks : int
        Number of random blocks; more blocks reduce estimator variance.
    block_len : int or None
        Samples per block. Defaults to ``4 * max_lag`` so every lag has ample
        support. Clamped to N.

    Returns
    -------
    R : ndarray, shape (max_lag + 1,)
        Estimated autocorrelation at lags 0..max_lag.
    """

    if rng is None:
        rng = np.random.default_rng()

    if block_len is None:
        block_len = 4 * max_lag
    block_len = int(min(block_len, N))

    if block_len <= max_lag:
        raise ValueError("block_len must exceed max_lag to estimate all lags")

    max_start = max(1, N - block_len)

    R = np.zeros(max_lag + 1, dtype=float)
    for _ in range(n_blocks):
        start = int(rng.integers(0, max_start))
        blk = np.asarray(sampler.block(start, block_len), dtype=float)
        if remove_mean:
            blk = blk - blk.mean()
        for tau in range(max_lag + 1):
            R[tau] += np.mean(blk[: block_len - tau] * blk[tau:])

    R /= n_blocks
    return R


def band_energy_from_sketch(R, fs, f_lo, f_hi):
    """Energy in [f_lo, f_hi] from an autocorrelation sketch (pure function).

    ``E = R(0) (f_hi - f_lo) + 2 sum_{tau>=1} R(tau) g(tau)`` with ``g`` the
    :func:`band_energy_kernel`. Only relative comparisons between bands matter
    for the search, so no extra normalization is applied.
    """

    if not (0 <= f_lo < f_hi <= fs / 2):
        raise ValueError("Require 0 <= f_lo < f_hi <= fs/2")

    taus = np.arange(R.size, dtype=float)
    g = band_energy_kernel(taus, f_lo, f_hi, fs)
    return float(R[0] * g[0] + 2.0 * np.sum(R[1:] * g[1:]))


def multiresolution_band_search(
    R,
    fs,
    f_lo=0.0,
    f_hi=None,
    max_depth=24,
    min_width_hz=None,
    return_trace=False,
):
    """Global dominant-band search on an autocorrelation sketch (pure function).

    At each level: split the current band into halves, compare their energies
    from the sketch, and keep the higher-energy half. Reads no samples -- all
    cost was paid building ``R``.

    ``min_width_hz`` defaults to the sketch's frequency resolution ``fs / M``;
    searching below that is meaningless because R is truncated at lag M.
    Reliable only when there is a dominant band.
    """

    if f_hi is None:
        f_hi = fs / 2

    max_lag = R.size - 1
    resolution = fs / max_lag
    if min_width_hz is None:
        min_width_hz = resolution

    lo = float(f_lo)
    hi = float(f_hi)

    trace = []
    depth = 0

    for depth in range(max_depth):
        if hi - lo <= min_width_hz:
            break

        mid = 0.5 * (lo + hi)
        e_left = band_energy_from_sketch(R, fs, lo, mid)
        e_right = band_energy_from_sketch(R, fs, mid, hi)

        if e_left >= e_right:
            decision = "left"
            hi = mid
        else:
            decision = "right"
            lo = mid

        if return_trace:
            trace.append(
                {
                    "depth": depth,
                    "band": (lo, hi),
                    "split": mid,
                    "left_energy": e_left,
                    "right_energy": e_right,
                    "decision": decision,
                }
            )

    result = {
        "band": (lo, hi),
        "center_hz": 0.5 * (lo + hi),
        "width_hz": hi - lo,
        "depth": depth + 1,
        "resolution_hz": resolution,
    }

    if return_trace:
        result["trace"] = trace

    return result


def local_peak_search(
    R,
    fs,
    f0,
    init_step_hz=None,
    resolution_hz=None,
    max_iter=64,
    return_trace=False,
):
    """Hill-climb to a *local* spectral peak near a seed ``f0`` (pure function).

    Evaluates band energy just below, at, and just above the current center,
    moves toward the larger neighbor, and halves the step once centered. Reads
    no samples -- all cost was paid building ``R``. Compared with the global
    search this only explores ``log(search_radius / resolution)`` levels, so it
    is the cheaper choice when a rough location is already known.

    Parameters
    ----------
    f0 : float
        Seed frequency (Hz) to start the climb from.
    init_step_hz : float or None
        Initial neighbor offset / probe-band width. Defaults to 16x the
        resolution, giving a few halvings before convergence.
    resolution_hz : float or None
        Stop once the step shrinks to this value. Defaults to the sketch
        resolution ``fs / M`` (going finer is meaningless past lag M).
    """

    nyq = fs / 2
    if not (0.0 <= f0 <= nyq):
        raise ValueError("Require 0 <= f0 <= fs/2")

    max_lag = R.size - 1
    sketch_res = fs / max_lag
    if resolution_hz is None:
        resolution_hz = sketch_res
    if init_step_hz is None:
        init_step_hz = 16.0 * resolution_hz

    def energy_at(center, width):
        # Clamp the probe band to the valid spectrum [0, fs/2].
        half = 0.5 * width
        lo = max(0.0, center - half)
        hi = min(nyq, center + half)
        if not (lo < hi):
            return float("-inf")
        return band_energy_from_sketch(R, fs, lo, hi)

    center = float(f0)
    step = float(init_step_hz)

    trace = []

    for it in range(max_iter):
        if step <= resolution_hz:
            break

        e_center = energy_at(center, step)
        e_left = energy_at(center - step, step)
        e_right = energy_at(center + step, step)

        if e_left > e_center and e_left >= e_right:
            decision = "left"
            new_center = max(0.0, center - step)
        elif e_right > e_center and e_right > e_left:
            decision = "right"
            new_center = min(nyq, center + step)
        else:
            # Centered on the peak: refine by halving the step.
            decision = "center"
            new_center = center
            step *= 0.5

        if return_trace:
            trace.append(
                {
                    "iter": it,
                    "center": center,
                    "step": step,
                    "left_energy": e_left,
                    "center_energy": e_center,
                    "right_energy": e_right,
                    "decision": decision,
                }
            )

        center = new_center

    result = {
        "center_hz": center,
        "final_step_hz": step,
        "iters": len(trace) if return_trace else None,
        "resolution_hz": sketch_res,
    }

    if return_trace:
        result["trace"] = trace

    return result


def _resolution_to_max_lag(fs, resolution_hz):
    """Max lag needed to resolve ``resolution_hz`` (resolution ~ fs / M)."""
    return max(8, int(np.ceil(fs / resolution_hz)))


def find_dominant_frequency_sublinear(
    x,
    fs,
    resolution_hz=50.0,
    n_blocks=4,
    block_len=None,
    seed=None,
):
    """Convenience wrapper: global dominant-frequency search over [0, fs/2].

    Builds the autocorrelation sketch (the only samples read) then searches it.
    """

    rng = np.random.default_rng(seed)
    sampler = ArraySampler(x)
    N = len(x)

    max_lag = _resolution_to_max_lag(fs, resolution_hz)
    R = autocorrelation_sketch(
        sampler, N, max_lag=max_lag, n_blocks=n_blocks,
        block_len=block_len, rng=rng,
    )

    result = multiresolution_band_search(
        R, fs, f_lo=0.0, f_hi=fs / 2, min_width_hz=resolution_hz,
    )
    bl = int(min(block_len if block_len else 4 * max_lag, N))
    result["samples_read"] = n_blocks * bl
    result["sample_fraction"] = result["samples_read"] / N
    return result["center_hz"], result


def find_local_peak_sublinear(
    x,
    fs,
    f0,
    resolution_hz=25.0,
    init_step_hz=None,
    n_blocks=4,
    block_len=None,
    seed=None,
):
    """Convenience wrapper: local-peak hill-climb near a seed frequency f0.

    Builds the autocorrelation sketch (the only samples read) then climbs it.
    """

    rng = np.random.default_rng(seed)
    sampler = ArraySampler(x)
    N = len(x)

    max_lag = _resolution_to_max_lag(fs, resolution_hz)
    R = autocorrelation_sketch(
        sampler, N, max_lag=max_lag, n_blocks=n_blocks,
        block_len=block_len, rng=rng,
    )

    result = local_peak_search(
        R, fs, f0=f0, init_step_hz=init_step_hz, resolution_hz=resolution_hz,
    )
    bl = int(min(block_len if block_len else 4 * max_lag, N))
    result["samples_read"] = n_blocks * bl
    result["sample_fraction"] = result["samples_read"] / N
    return result["center_hz"], result


if __name__ == "__main__":
    fs = 48_000
    N = 1_000_000

    rng = np.random.default_rng(0)
    t = np.arange(N) / fs

    true_freq = 7130.0
    x = np.sin(2 * np.pi * true_freq * t) + 0.2 * rng.standard_normal(N)

    # Global dominant-band search.
    freq, info = find_dominant_frequency_sublinear(
        x, fs, resolution_hz=25, n_blocks=4, seed=123,
    )
    print("global  ->", freq)
    print("  band:", info["band"])
    print("  samples read:", info["samples_read"],
          "fraction:", info["sample_fraction"])

    # Local peak near a rough seed (e.g. 7000 Hz).
    peak, pinfo = find_local_peak_sublinear(
        x, fs, f0=7000.0, resolution_hz=10, n_blocks=4, seed=123,
    )
    print("local   ->", peak)
    print("  samples read:", pinfo["samples_read"],
          "fraction:", pinfo["sample_fraction"])
