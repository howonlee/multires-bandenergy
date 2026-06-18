# Why the sample count decouples from N — and what it's coupled to

This note explains why `multires_bandenergy.py` can find a spectral peak while
reading a number of time-domain samples that does **not** grow with the signal
length `N`, and identifies the quantities that *do* set the budget.

The empirical evidence (from the module's own test sweep): for a tone at
7130 Hz in noise, with `resolution_hz=25` and `n_blocks=4`,

| N          | samples read | fraction of signal | recovered freq |
|------------|--------------|--------------------|----------------|
| 100,000    | 30,720       | 30.7 %             | ~7137 Hz       |
| 1,000,000  | 30,720       | 3.1 %              | ~7137 Hz       |
| 10,000,000 | 30,720       | 0.3 %              | ~7137 Hz       |

Same samples read, same accuracy, 100× range of N.

---

## 1. The budget formula has no N in it

We never scan the signal. We fetch a small **autocorrelation sketch** — a few
random contiguous blocks of length `L` — and compute

    R(tau) = mean_n x[n] x[n+tau],   tau = 0 .. M.

Every band-energy question is then answered analytically from `R` (see
`band_energy_from_sketch`), reading zero further samples. So the *entire*
search — global binary search plus any number of local hill-climb steps — costs

    samples_read = n_blocks * block_len

and that expression simply does not contain `N`. Increase `N` from a million to
a trillion and, holding the sketch parameters fixed, you read exactly the same
number of samples. The search *logic* runs in `O(depth)` band evaluations, but
those touch the cached `R`, not the signal.

Contrast this with the two estimators that *failed* (preserved in git history):

- **Random frequency probes** — evaluate `|X(f)|^2` at random frequencies in a
  band. A pure tone is a spectral spike of width `fs / N`. In a wide band a
  finite set of random probes lands within `fs/N` of the spike with vanishing
  probability, so at the coarse levels the tone is invisible and the estimate is
  pure noise floor. Worse as N grows.
- **Random time-pairs** — estimate the quadratic form `sum x[n]x[n'] g(n-n')` by
  sampling random pairs `(n,n')`. Unbiased, but the variance of `x[n]x[n']g`
  over scattered pairs swamps the tone's contribution; the greedy binary search
  flips a coin and derails by depth 2.

Both are "sublinear" on paper yet useless here, because being sublinear in
*count* is not the same as having *contrast per decision*. Which brings us to
the real story.

---

## 2. What actually buys reliability: coherent gain, not sample count

A contiguous block of length `L` is not just `L` cheap samples — it is `L`
*phase-coherent* samples. For a tone `A·sin(2π f0 t)` plus white noise of
variance `σ²`, the block autocorrelation is

    R(tau) ≈ (A²/2) cos(2π f0 tau / fs)   +   σ² · [tau == 0].

The tone leaves a clean cosine **at every lag** `tau = 1 .. M`. The noise
contributes **only at `tau = 0`**. When we integrate `R(tau) g(tau)` over a band
that contains `f0`, the cosine and the band kernel line up constructively across
all `M` lags — an `O(M)` coherent build-up — while for a band that excludes
`f0` the cosine and kernel beat against each other and cancel. That is the
contrast that makes each binary decision correct. The module measured it at
~100× (tone half-band 12030 vs. empty halves ~120–280).

Scattered point samples never get this build-up: with random `(n, n')` lags the
cosine phase is random, so there is nothing to integrate coherently — only
variance to average down, and averaging variance down is expensive.

So the lever that matters is *coherence length*, supplied here by block length
`L` and lag depth `M`, neither of which needs to grow with `N`.

---

## 3. What the sample count *is* coupled to

`samples_read = n_blocks · block_len`, with `block_len ≈ 4·M`. Each factor maps
to a real requirement:

### Frequency resolution → `M` (the max lag)
Resolving frequencies to `Δf` needs lags out to roughly `M ≈ fs / Δf`
(`_resolution_to_max_lag`). A truncated autocorrelation of depth `M` simply
cannot distinguish features finer than `fs / M`; the band kernel oscillates
faster than `R` is known. So
**samples_read grows like `fs / Δf` (linearly in target resolution), not with N.**
This is the only knob with a hard floor: if you demand *exact DFT-bin*
resolution `Δf = fs / N`, then `M ≈ N` and the budget becomes `O(N)` again.
N re-enters precisely when you insist on bin-exact answers. For an approximate
*Hz*-resolution peak, it stays out.

### Noise / confidence → `n_blocks` (and `M`)
More blocks average down the variance of `R(tau)`; the variance of each band
decision falls like `1 / (n_blocks · effective_support)`. The number of blocks
needed scales with how hard the decision is — see contrast below — and with the
acceptable failure probability `δ` as `log(1/δ)` (you take the union bound over
the `~log2(fs/2 / Δf)` decisions in the search). Still no `N`.

### Spectral contrast → multiplies `n_blocks`
Let `gap` be the normalized energy separation between the winning and losing
half-band. Reliable decisions need roughly

    n_blocks · M  =  O( log(depth/δ) / gap² ).

For a **fixed-SNR** tone the gap is *N-independent*: as `N` grows, the tone's
total energy `∝ N` and the in-band noise energy `∝ N` grow together, so their
*ratio* — the gap — holds. That is the deep reason the budget can stay flat: at
fixed SNR, the difficulty of each decision does not change with N. If instead
amplitude is fixed while only noise duration grows (SNR falling with N), the gap
shrinks and you would need more blocks — but that is N entering through the
*signal model / SNR*, not through the algorithm.

### Search span → `depth` (logarithmic, and free of samples)
Global search runs `depth ≈ log2((fs/2) / Δf)` levels; a local hill-climb near a
seed runs `log2(search_radius / Δf)`. Either way it is logarithmic, and because
every level reuses the one cached sketch, **depth costs zero extra samples.**

---

## 4. One-line summary

    samples_read  ≈  n_blocks · (4 · fs / Δf)
                  ≈  O( log(1/δ) / gap²  ·  fs / Δf )

- **Coupled to:** desired frequency resolution `Δf`, confidence `δ`, and inverse
  spectral contrast `1/gap²` (which at fixed SNR is itself independent of N).
- **Not coupled to:** the signal length `N` — *unless* you demand bin-exact
  resolution `Δf = fs/N`, at which point `M ≈ N` and `N` legitimately returns.

For an approximate local peak above the noise floor with cheap random access,
the honest answer is: the work is set by *how finely and how confidently* you
want to locate the peak, not by *how long* the signal is.
