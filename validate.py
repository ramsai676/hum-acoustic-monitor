"""Does the detector actually work?

Reports, for each failure mode and severity, how separable faulty audio is from
healthy audio (ROC AUC over per-frame scores) and how often the clip-level
decision fires. Then — the part that decides whether this is usable in a real
room rather than a quiet lab — how often it false-alarms on a machine that is
perfectly fine but recorded badly.

    python validate.py

Exits non-zero if the detector regresses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hum.detector import Detector                    # noqa: E402
from hum.signals import FAULTS, NUISANCES, healthy   # noqa: E402

TRIALS = 20
CLIP_SECONDS = 4.0
BASELINE_SECONDS = 90.0

# Severities we require detection at. 0.25x is reported but not required — at
# that level several faults are genuinely inside the healthy distribution.
REQUIRED = (1.0, 0.5)
REPORTED = (1.0, 0.5, 0.25)

MAX_NUISANCE_ALARM = 0.10
MIN_DETECTION = 0.80
MIN_AUC = 0.90


def main() -> int:
    detector = Detector()
    detector.fit(healthy(BASELINE_SECONDS, np.random.default_rng(100)))
    threshold = detector.baseline.threshold
    features = detector.features

    print("\n  Hum — detector validation")
    print(f"  {TRIALS} trials/condition · {CLIP_SECONDS:.0f}s clips · "
          f"{BASELINE_SECONDS:.0f}s baseline")
    print(f"  frame {features.n_fft} ({1000 * features.n_fft / features.sample_rate:.0f} ms) · "
          f"{features.n_bands} bands · threshold {threshold:.2f}\n")

    healthy_frames = np.concatenate([
        detector.score(healthy(CLIP_SECONDS, np.random.default_rng(1000 + i)))
        for i in range(TRIALS)
    ])

    def clip_scores(make, **kw) -> np.ndarray:
        return np.array([
            float(np.median(detector.score(
                make(CLIP_SECONDS, np.random.default_rng(2000 + i), **kw)
            )))
            for i in range(TRIALS)
        ])

    ok = True

    print("  FAULT DETECTION")
    print(f"    {'failure mode':<20}{'AUC':>7}{'score range':>16}{'detected':>11}")
    print(f"    {'-' * 54}")

    for name, make in FAULTS.items():
        for severity in REPORTED:
            frames = np.concatenate([
                detector.score(make(CLIP_SECONDS, np.random.default_rng(2000 + i),
                                    severity=severity))
                for i in range(TRIALS)
            ])
            auc = roc_auc_score(
                np.r_[np.zeros(healthy_frames.size), np.ones(frames.size)],
                np.r_[healthy_frames, frames],
            )
            scores = clip_scores(make, severity=severity)
            rate = float(np.mean(scores > threshold))

            note = "" if severity in REQUIRED else "   (not required)"
            print(f"    {name + ' @' + format(severity, 'g') + 'x':<20}"
                  f"{auc:>7.3f}{f'{scores.min():.1f}-{scores.max():.1f}':>16}"
                  f"{rate:>10.0%}{note}")

            if severity in REQUIRED and (auc < MIN_AUC or rate < MIN_DETECTION):
                ok = False

    print("\n  FALSE ALARMS  (machine is healthy — must stay quiet)")
    print(f"    {'condition':<20}{'score range':>16}{'alarms':>11}")
    print(f"    {'-' * 47}")

    quiet = {"plain healthy": lambda s, r: healthy(s, r)}
    quiet.update(NUISANCES)

    for name, make in quiet.items():
        scores = np.array([
            float(np.median(detector.score(
                make(CLIP_SECONDS, np.random.default_rng(3000 + i))
            )))
            for i in range(TRIALS)
        ])
        rate = float(np.mean(scores > threshold))
        flag = "" if rate <= MAX_NUISANCE_ALARM else "   <-- too noisy"
        print(f"    {name:<20}{f'{scores.min():.1f}-{scores.max():.1f}':>16}"
              f"{rate:>10.0%}{flag}")
        if rate > MAX_NUISANCE_ALARM:
            ok = False

    print("\n  EXPLANATION  (bearing fault, 1x) — why it alarmed")
    report = detector.explain(
        FAULTS["bearing"](CLIP_SECONDS, np.random.default_rng(9), severity=1.0)
    )
    print(f"    score {report['score']} vs threshold {report['threshold']}")
    for band in report["bands"][:3]:
        print(f"      {band['hz_low']:>7.0f}-{band['hz_high']:<7.0f} Hz  "
              f"{band['direction']:<8} z={band['z']:+.2f}")

    print(f"\n  {'PASS' if ok else 'FAIL'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
