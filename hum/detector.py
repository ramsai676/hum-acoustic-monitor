"""The anomaly detector.

Deliberately simple maths, chosen so the identical algorithm can run in the
browser with no ML runtime: framing -> mel-band log energies -> robust per-band
z-scores against a learned baseline.

Why not an autoencoder. Two reasons, both practical:

  1. A machine's "normal" is a few minutes of audio captured on the spot by
     whoever is holding the phone. That is nowhere near enough data to train a
     network without overfitting the room it was recorded in.
  2. Per-band z-scores are *interpretable*. The detector can say "energy rose
     in the 2-4 kHz bands", which is what lets a maintenance person believe it.
     An autoencoder reconstruction error says only "different", which is the
     kind of output people learn to ignore.

Robust statistics (median / MAD) rather than mean / std throughout: the baseline
is captured in the real world, so a door slam or a cough during those two minutes
must not widen the band and blind the detector for the rest of the session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# MAD -> standard-deviation equivalent for a normal distribution.
MAD_TO_SIGMA = 1.4826


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=float) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=float) / 2595.0) - 1.0)


def band_edges(n_bands: int, sample_rate: int, f_min: float = 30.0,
               f_split: float = 500.0, f_max: float | None = None,
               low_fraction: float = 0.45) -> np.ndarray:
    """Band edges: linear below f_split, logarithmic above.

    Not mel. Mel spacing is tuned to human speech perception, which puts most of
    its resolution between 500 Hz and 4 kHz and treats everything below 500 Hz as
    roughly one lump. Machines live in that lump — running frequencies and their
    first harmonics sit between 20 and 400 Hz, and imbalance shows up as
    sidebands only a few Hz either side of them.

    Measured directly: with mel spacing, rotor imbalance scored AUC 0.63 and was
    never detected, because both sidebands landed inside a single wide band. A
    linear low end resolves them.
    """
    f_max = f_max or sample_rate / 2.0
    n_edges = n_bands + 2
    n_low = max(2, int(round(n_edges * low_fraction)))

    low = np.linspace(f_min, f_split, n_low)
    high = np.logspace(np.log10(f_split), np.log10(f_max), n_edges - n_low + 1)[1:]
    return np.concatenate([low, high])


def filterbank(edges: np.ndarray, n_fft: int, sample_rate: int) -> np.ndarray:
    """Triangular filterbank from explicit edges, shape (n_bands, n_fft//2 + 1)."""
    n_bands = len(edges) - 2
    n_bins = n_fft // 2 + 1

    bin_points = np.floor((n_fft + 1) * edges / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_bins - 1)

    fb = np.zeros((n_bands, n_bins))
    for i in range(n_bands):
        left, centre, right = bin_points[i], bin_points[i + 1], bin_points[i + 2]
        if centre == left:
            centre = min(left + 1, n_bins - 1)
        if right == centre:
            right = min(centre + 1, n_bins - 1)
        for b in range(left, centre):
            fb[i, b] = (b - left) / max(centre - left, 1)
        for b in range(centre, right):
            fb[i, b] = (right - b) / max(right - centre, 1)

    # Area-normalise so wide high-frequency bands do not dominate purely by width.
    areas = fb.sum(axis=1, keepdims=True)
    return fb / np.maximum(areas, 1e-9)


@dataclass
class Features:
    # 4096 @ 16 kHz gives 3.9 Hz bins — enough to resolve imbalance sidebands
    # that sit ~12 Hz off a harmonic. 1024 could not.
    n_bands: int = 40
    n_fft: int = 4096
    hop: int = 1024
    sample_rate: int = 16000
    gain_invariant: bool = True
    _fb: np.ndarray | None = field(default=None, repr=False)
    _window: np.ndarray | None = field(default=None, repr=False)
    _edges: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._edges = band_edges(self.n_bands, self.sample_rate)
        self._fb = filterbank(self._edges, self.n_fft, self.sample_rate)
        self._window = np.hanning(self.n_fft)

    @property
    def edges(self) -> np.ndarray:
        return self._edges

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """Audio -> (n_frames, n_bands) log-band-energy matrix."""
        audio = np.asarray(audio, dtype=float).ravel()
        if audio.size < self.n_fft:
            return np.zeros((0, self.n_bands))

        n_frames = 1 + (audio.size - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(n_frames)[:, None]
        frames = audio[idx] * self._window

        spectrum = np.abs(np.fft.rfft(frames, axis=1)) ** 2
        bands = spectrum @ self._fb.T

        # dB. The floor stops silence turning into a huge negative outlier that
        # would otherwise read as a dramatic "anomaly" whenever the machine stops.
        log_bands = 10.0 * np.log10(np.maximum(bands, 1e-12))

        if self.gain_invariant:
            # Subtract each frame's own mean, keeping only spectral *shape*.
            #
            # Without this, moving the phone 20 cm closer raises every band by
            # the same few dB and the detector screams. Measured: a uniform
            # +6 dB gain produced a 100% false-alarm rate before this line.
            #
            # The cost is real and worth stating: a fault whose only symptom is
            # "the whole machine got louder, identically across the spectrum"
            # becomes invisible. Every mechanical failure mode modelled here
            # changes shape, and mic distance varies constantly in real use, so
            # the trade is strongly worth it.
            log_bands = log_bands - log_bands.mean(axis=1, keepdims=True)

        return log_bands


@dataclass
class Baseline:
    """What this machine sounds like when it is well."""

    median: np.ndarray
    scale: np.ndarray          # MAD-derived sigma per band
    threshold: float           # score above which we call it anomalous
    n_frames: int

    def to_dict(self) -> dict:
        return {
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "threshold": float(self.threshold),
            "n_frames": int(self.n_frames),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Baseline":
        return cls(np.array(d["median"]), np.array(d["scale"]),
                   float(d["threshold"]), int(d["n_frames"]))


class Detector:
    """Learn a machine's normal sound, then score how far from it we are."""

    # Chosen on the validation set, not plucked from the air. A continuous
    # baseline recording is *less* variable than independent clips of the same
    # machine taken later, so a margin near 1.0 alarms constantly on healthy
    # audio. The binding constraint is nearby speech, the loudest realistic
    # nuisance, which peaks around 1.8x the baseline reference; 2.5x clears it
    # with room to spare. See validate.py for the measured distributions.
    DEFAULT_MARGIN = 2.5

    def __init__(self, features: Features | None = None, top_k: int = 4,
                 threshold_margin: float = DEFAULT_MARGIN, clip_seconds: float = 4.0,
                 smooth: int = 5):
        self.features = features or Features()
        self.top_k = top_k
        self.threshold_margin = threshold_margin
        self.clip_seconds = clip_seconds
        self.smooth = smooth
        self.baseline: Baseline | None = None

    # ---- learning ----

    def fit(self, normal_audio: np.ndarray) -> Baseline:
        frames = self.features(normal_audio)
        if frames.shape[0] < 8:
            raise ValueError(
                f"need at least 8 frames of baseline audio, got {frames.shape[0]}"
            )

        median = np.median(frames, axis=0)
        mad = np.median(np.abs(frames - median), axis=0)
        scale = MAD_TO_SIGMA * mad

        # A perfectly steady band gives MAD 0, which would make its z-score
        # infinite on the tiniest change. Floor it relative to the loudest band's
        # spread so a dead-quiet band cannot dominate the score.
        floor = max(0.5, 0.05 * float(np.max(scale)) if scale.size else 0.5)
        scale = np.maximum(scale, floor)

        # Calibrate the threshold on the SAME statistic the decision uses.
        #
        # This was got wrong once, and the failure was quiet and instructive: the
        # decision used each clip's median frame score while the threshold was a
        # high quantile of individual baseline frames. Those live on completely
        # different scales, so the threshold sat roughly twice as high as any real
        # clip ever scored — separation was perfect (AUC 1.000) and detection was
        # 0%. A threshold must be calibrated against the exact statistic it gates.
        self.baseline = Baseline(median, scale, threshold=np.inf, n_frames=frames.shape[0])

        per_frame = self._smooth(self.score_frames(frames))
        per_clip = int(self.clip_seconds * self.features.sample_rate / self.features.hop)

        if per_clip >= 2 and per_frame.size >= per_clip * 2:
            # Overlapping windows (50% hop) rather than disjoint ones: a 90 s
            # baseline only yields ~22 disjoint 4 s clips, which is a thin sample
            # to take a maximum over.
            step = max(1, per_clip // 2)
            medians = [
                float(np.median(per_frame[i:i + per_clip]))
                for i in range(0, per_frame.size - per_clip + 1, step)
            ]
            # Max over baseline clips: the healthiest machine still has its worst
            # few seconds, and the detector must not alarm on that.
            reference = float(np.max(medians))
        else:
            reference = float(np.median(per_frame))

        self.baseline.threshold = reference * self.threshold_margin
        return self.baseline

    # ---- scoring ----

    def z_frames(self, frames: np.ndarray) -> np.ndarray:
        if self.baseline is None:
            raise RuntimeError("call fit() before scoring")
        return (frames - self.baseline.median) / self.baseline.scale

    def score_frames(self, frames: np.ndarray) -> np.ndarray:
        """One score per frame: RMS of the top-k absolute band deviations.

        Top-k rather than all-bands because a real fault usually moves a handful
        of bands hard, not every band a little. Averaging across all 32 bands
        dilutes exactly the signal we are looking for.
        """
        if frames.shape[0] == 0:
            return np.zeros(0)
        z = np.abs(self.z_frames(frames))
        k = min(self.top_k, z.shape[1])
        top = np.sort(z, axis=1)[:, -k:]
        return np.sqrt((top ** 2).mean(axis=1))

    def _smooth(self, scores: np.ndarray) -> np.ndarray:
        """Short median smoother.

        A single transient (a dropped spanner) is not a machine fault, and an
        unsmoothed detector alarms on every one of them.
        """
        k = self.smooth
        if k <= 1 or scores.size < k:
            return scores
        pad = k // 2
        padded = np.pad(scores, pad, mode="edge")
        return np.array([np.median(padded[i:i + k]) for i in range(scores.size)])

    def score(self, audio: np.ndarray) -> np.ndarray:
        return self._smooth(self.score_frames(self.features(audio)))

    def is_anomalous(self, audio: np.ndarray) -> tuple[bool, float]:
        """Decide on the *median* frame score, not a high percentile.

        A machine fault is persistent — it is present in essentially every frame
        for as long as the machine runs. Room noise is not: someone talking
        covers maybe a third of a clip, a dropped tool a twentieth.

        Scoring on the 90th percentile meant anything covering >10% of the clip
        tripped the alarm, which gave a 100% false-alarm rate on nearby speech.
        Requiring persistence costs sensitivity to genuinely intermittent faults
        (an occasional knock), and that is a real limitation — but a detector
        that cries wolf every time someone walks past gets switched off, and
        then it detects nothing at all.
        """
        scores = self.score(audio)
        if scores.size == 0:
            return False, 0.0
        persistent = float(np.median(scores))
        return persistent > self.baseline.threshold, persistent

    def explain(self, audio: np.ndarray) -> dict:
        """Which bands went wrong — the part that makes the alert believable."""
        frames = self.features(audio)
        if frames.shape[0] == 0:
            return {"bands": [], "verdict": "no audio"}

        z = self.z_frames(frames)
        mean_z = z.mean(axis=0)
        order = np.argsort(-np.abs(mean_z))[:5]

        edges = self.features.edges
        bands = [
            {
                "band": int(i),
                "hz_low": round(float(edges[i]), 1),
                "hz_high": round(float(edges[i + 2]), 1),
                "z": round(float(mean_z[i]), 2),
                "direction": "louder" if mean_z[i] > 0 else "quieter",
            }
            for i in order
        ]
        anomalous, peak = self.is_anomalous(audio)
        return {
            "anomalous": anomalous,
            "score": round(peak, 2),
            "threshold": round(float(self.baseline.threshold), 2),
            "bands": bands,
        }
