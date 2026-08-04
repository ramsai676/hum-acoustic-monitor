"""Synthetic machine audio, built from how machines actually fail.

Real fault recordings are the bottleneck in this field — the public sets are
tens of gigabytes and mostly cover industrial machines nobody has at home. So
validation here is done against synthesised signals whose structure follows the
established vibration-analysis failure modes:

  bearing wear   -> a periodic train of sharp, exponentially-decaying impulses
                    at a characteristic defect frequency, exciting resonance
  friction / dry -> a broadband rise, strongest in the mid-to-high bands
  imbalance      -> sidebands appearing either side of the running frequency

Just as important are the *nuisance* cases: a machine that is fine, recorded
badly. Mic moved, someone talking, a door slamming. A detector that alarms on
those is useless in a real room, so they are scored explicitly.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000


def _pink(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink-ish noise. Real rooms and machines are nowhere near white."""
    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    spectrum /= np.sqrt(freqs)
    out = np.fft.irfft(spectrum, n)
    return out / (np.std(out) + 1e-12)


HARMONICS = (1.0, 0.45, 0.22, 0.11, 0.05)


def healthy(seconds: float = 4.0, rng: np.random.Generator | None = None,
            running_hz: float = 100.0, level: float = 1.0,
            harmonics: tuple[float, ...] = HARMONICS) -> np.ndarray:
    """A well machine: a stable running tone plus harmonics, over a noise floor.

    Includes slow amplitude drift and slight frequency wander, because nothing
    mechanical is perfectly steady and a detector tuned on a perfectly steady
    signal falls apart on the first real recording.
    """
    rng = rng or np.random.default_rng(0)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE

    # Frequency wander, done as *phase* modulation.
    #
    # The obvious-looking `sin(2*pi*f*t*wander(t))` is wrong: differentiating the
    # argument gives an instantaneous frequency with a `t * d(wander)/dt` term
    # that grows without bound. Over a 4 s clip it is invisible; over a 90 s
    # baseline it swept the running frequency by +/-34%, so the detector's own
    # reference was less stable than the faults it was meant to catch.
    # Integrating the frequency deviation properly keeps it bounded at +/-0.2%.
    deviation, mod_hz = 0.002, 0.3
    phase_wander = -deviation * np.cos(2 * np.pi * mod_hz * t) / (2 * np.pi * mod_hz)

    signal = np.zeros(n)
    for harmonic, amp in enumerate(harmonics, start=1):
        signal += amp * np.sin(2 * np.pi * running_hz * harmonic * (t + phase_wander))

    drift = 1.0 + 0.05 * np.sin(2 * np.pi * 0.12 * t + rng.uniform(0, 6.28))
    return level * (signal * drift + 0.06 * _pink(n, rng))


def bearing_fault(seconds: float = 4.0, rng: np.random.Generator | None = None,
                  severity: float = 1.0, defect_hz: float = 37.0) -> np.ndarray:
    """Healthy signal plus an impulse train ringing a structural resonance."""
    rng = rng or np.random.default_rng(1)
    base = healthy(seconds, rng)
    n = base.size
    t = np.arange(n) / SAMPLE_RATE

    impulses = np.zeros(n)
    period = int(SAMPLE_RATE / defect_hz)
    ring_len = int(0.004 * SAMPLE_RATE)
    ring_t = np.arange(ring_len) / SAMPLE_RATE
    # A struck resonance: decaying sinusoid around 3.2 kHz.
    ring = np.exp(-ring_t * 900.0) * np.sin(2 * np.pi * 3200.0 * ring_t)

    for start in range(period, n - ring_len, period):
        jitter = rng.integers(-period // 20, period // 20 + 1)
        pos = np.clip(start + jitter, 0, n - ring_len - 1)
        impulses[pos:pos + ring_len] += ring * rng.uniform(0.8, 1.2)

    return base + severity * 0.35 * impulses


def friction_fault(seconds: float = 4.0, rng: np.random.Generator | None = None,
                   severity: float = 1.0) -> np.ndarray:
    """Dry bearing / cavitation: a broadband rise weighted to higher bands."""
    rng = rng or np.random.default_rng(2)
    base = healthy(seconds, rng)
    n = base.size

    noise = rng.standard_normal(n)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    # Emphasise 1.5-6 kHz, where dry friction actually shows up.
    shape = np.clip((freqs - 1200.0) / 4500.0, 0.0, 1.0)
    spectrum *= shape
    shaped = np.fft.irfft(spectrum, n)
    shaped /= np.std(shaped) + 1e-12

    return base + severity * 0.18 * shaped


def imbalance_fault(seconds: float = 4.0, rng: np.random.Generator | None = None,
                    severity: float = 1.0, running_hz: float = 100.0) -> np.ndarray:
    """Rotor imbalance: a large rise in the 1x running-speed component.

    Corrected from an earlier version that modelled this as +/-12 Hz sidebands,
    which was wrong twice over. Textbook imbalance is characterised by a dominant
    1x radial component; sidebands indicate *modulation* — looseness or
    misalignment — which is a different mode (see `looseness_fault`). And a 12 Hz
    offset is finer than any practical filterbank resolves, so that version was
    measuring the spectrum analyser rather than the detector.

    Note also that the 1x component is *scaled*, not given a separate
    randomly-phased copy. Adding at random phase can partially cancel the
    existing fundamental, which is why an earlier version produced scores
    anywhere between 4.0 and 11.2 at one fixed severity — half the time the
    "fault" was making the machine quieter at 1x.
    """
    rng = rng or np.random.default_rng(3)
    boosted = (HARMONICS[0] * (1.0 + 1.5 * severity),) + HARMONICS[1:]
    return healthy(seconds, rng, running_hz=running_hz, harmonics=boosted)


def looseness_fault(seconds: float = 4.0, rng: np.random.Generator | None = None,
                    severity: float = 1.0, running_hz: float = 100.0,
                    shaft_hz: float = 50.0) -> np.ndarray:
    """Mechanical looseness: sidebands at +/- shaft speed around the harmonic.

    Offset by a full shaft rotation rather than a few Hz, which is both what
    really happens and coarse enough for a filterbank to resolve.
    """
    rng = rng or np.random.default_rng(7)
    base = healthy(seconds, rng)
    t = np.arange(base.size) / SAMPLE_RATE
    sidebands = (
        np.sin(2 * np.pi * (running_hz * 2 - shaft_hz) * t)
        + np.sin(2 * np.pi * (running_hz * 2 + shaft_hz) * t)
    )
    return base + severity * 0.45 * sidebands


# ---- nuisances: machine is FINE, recording is not ----

def nuisance_gain(seconds: float = 4.0, rng: np.random.Generator | None = None,
                  gain_db: float = 6.0) -> np.ndarray:
    """Somebody moved the phone closer. Everything gets louder, uniformly."""
    rng = rng or np.random.default_rng(4)
    return healthy(seconds, rng) * (10 ** (gain_db / 20.0))


def nuisance_speech(seconds: float = 4.0, rng: np.random.Generator | None = None
                    ) -> np.ndarray:
    """Someone talks near the machine — formant-like energy, 300 Hz-3 kHz."""
    rng = rng or np.random.default_rng(5)
    base = healthy(seconds, rng)
    n = base.size
    t = np.arange(n) / SAMPLE_RATE

    speech = np.zeros(n)
    for f0, amp in [(320.0, 1.0), (900.0, 0.6), (2400.0, 0.3)]:
        warble = f0 * (1 + 0.08 * np.sin(2 * np.pi * 4.5 * t))
        speech += amp * np.sin(2 * np.pi * warble * t)

    # Syllable-rate envelope, only over the middle of the clip.
    envelope = np.clip(np.sin(2 * np.pi * 3.0 * t) ** 2, 0, 1)
    window = np.zeros(n)
    window[n // 3: 2 * n // 3] = 1.0
    return base + 0.30 * speech * envelope * window


def nuisance_slam(seconds: float = 4.0, rng: np.random.Generator | None = None
                  ) -> np.ndarray:
    """A single door slam. Loud, broadband, and over in 200 ms."""
    rng = rng or np.random.default_rng(6)
    base = healthy(seconds, rng)
    n = base.size
    pos = n // 2
    length = int(0.2 * SAMPLE_RATE)
    decay = np.exp(-np.arange(length) / (0.03 * SAMPLE_RATE))
    base[pos:pos + length] += 2.5 * decay * rng.standard_normal(length)
    return base


FAULTS = {
    "bearing": bearing_fault,
    "friction": friction_fault,
    "imbalance": imbalance_fault,
    "looseness": looseness_fault,
}

NUISANCES = {
    "mic moved (+6 dB)": nuisance_gain,
    "speech nearby": nuisance_speech,
    "door slam": nuisance_slam,
}
