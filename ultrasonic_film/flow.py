"""Experimental local signal-pattern indicators, not verified two-phase regimes.

Two time bases:

* ``nominal_fps`` given (default in the CLI, ``--flow-timebase sequence``): the frame
  time is ``(sequence - sequence0) / fps``. A contiguous sequence only proves that no
  frame was lost, not that the device sampled uniformly or at the nominal rate, so the
  rate is checked on every computation from PC receive times of the current window:
  the window is split into quarters and, per quarter, the lower envelope (minimum) of
  ``receive time - sequence / nominal_fps`` is taken (the least-delayed frame). The
  overall (Q1->Q4), early (Q1->Q2) and late (Q3->Q4) rates derived from its slope must
  all be within ``RATE_TOLERANCE`` of ``nominal_fps``. Otherwise the result is
  ``TIMING_UNCERTAIN`` with no frequency, and results stay ``TIMING_UNCERTAIN`` until
  the whole window was received after the failure. The axis uses the validated overall
  rate. Transport delays are positive, so the lower envelope tolerates socket batching
  and the floor-quantised 15.625 ms Windows clock (equal or bunched receive times alone
  are not treated as a device fault).

  Limits: receive times include network and OS delay, not device acquisition time. A
  rate change shorter than about a quarter window, a slow drift inside the tolerance,
  or irregular device sampling that is hidden behind a steady transport delay is not
  detected. A real trigger timestamp from the device would be required for that.
* ``nominal_fps=None`` (``--flow-timebase receive``): PC receive times are used as is
  and poor timing is rejected (``TIMING_UNCERTAIN``) instead of inventing an axis.
  Equal timestamps (batching, coarse clocks) are allowed; only a decreasing timestamp
  or a sequence jump resets the window.

Statistics use unsmoothed detections (after any RF averaging).
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.signal import detrend


@dataclass(frozen=True)
class FlowFeatures:
    pattern: str = 'WARMUP'
    computed_at_s: float = float('nan')
    mean_h_mm: float = float('nan')
    sigma_h_mm: float = float('nan')
    amplitude_cv: float = float('nan')
    wave_h_pp_mm: float = float('nan')
    f_wave_hz: float = float('nan')
    internal_fraction: float = float('nan')
    valid_fraction: float = 0.0


RATE_TOLERANCE = 0.05   # overall/early/late window rates must be within +-5 % of nominal_fps


class FlowRegimeClassifier:
    """Heuristics for candidates only; thresholds require labeled lab validation.

    Compute at <=10 Hz over a two-second window by default. Four seconds are
    required for two cycles at 0.5 Hz. No annular/slug geometry is inferred.
    ``computed_at_s`` is always the PC receive timestamp of the computing frame.
    """
    def __init__(self, window_s=2.0, update_s=0.1, nominal_fps=None):
        if not (math.isfinite(window_s) and 1 <= window_s <= 10):
            raise ValueError('window_s must be between 1 and 10 seconds')
        if not (math.isfinite(update_s) and 0 < update_s <= window_s):
            raise ValueError('invalid update interval')
        if nominal_fps is not None and not (math.isfinite(nominal_fps) and nominal_fps > 0):
            raise ValueError('nominal_fps must be positive')
        self.window_s, self.update_s, self.nominal_fps = window_s, update_s, nominal_fps
        self.samples = deque(maxlen=4096)   # (receive time, h, amplitude, internal, valid, sequence)
        self.latest = FlowFeatures()
        self.last_sequence = None
        self._anchor = None                 # (receive time, sequence) at the last reset
        self.fps = nominal_fps              # rate used for the sequence time base (validated)
        self.rates = (float('nan'),) * 3    # last overall/early/late receive-time rate estimates
        self.timing_reason = ''             # why the last computation was TIMING_UNCERTAIN
        self._recover_after = None          # sequence of the last timing failure

    @property
    def timebase(self):
        return 'receive' if self.nominal_fps is None else 'sequence'

    def _reset(self, timestamp, sequence):
        self.samples.clear()
        self._anchor = (timestamp, sequence)
        self.fps = self.nominal_fps
        self._recover_after = None

    def _check_rate(self, x):
        """Sequence time base: validate the frame rate of this window from receive times.

        -> reason string ('' when valid). Sets ``self.rates`` and, when valid, ``self.fps``.
        """
        # Receive time = acquisition time + transport delay (>= 0). The lower envelope of
        # (receive time - sequence/nominal) per quarter follows the acquisition clock and is
        # insensitive to batching/queueing delays; a floor-quantised clock biases every
        # quarter alike. Its slope s between quarters gives rate = nominal / (1 + s).
        seq = x[:, 5] / self.nominal_fps
        offset = x[:, 0] - seq
        quarters = np.array_split(np.arange(len(x)), 4)
        points = [((seq[q[0]] + seq[q[-1]]) / 2, offset[q].min()) for q in quarters]
        def rate(a, b):
            slope = (points[b][1] - points[a][1]) / (points[b][0] - points[a][0])
            return self.nominal_fps / (1 + slope) if slope > -1 else float('inf')
        self.rates = (rate(0, 3), rate(0, 1), rate(2, 3))
        bad = [name for name, r in zip(('overall', 'early', 'late'), self.rates)
               if not abs(r / self.nominal_fps - 1) <= RATE_TOLERANCE]
        if bad:
            self._recover_after = x[-1, 5]
            return (f"frame rate {', '.join(f'{n} {r:.1f}' for n, r in zip(('overall', 'early', 'late'), self.rates))}"
                    f" fps vs nominal {self.nominal_fps:g} (+-{RATE_TOLERANCE:.0%}): {'/'.join(bad)} outside")
        if self._recover_after is not None and x[0, 5] <= self._recover_after:
            return 'recovering: window still contains frames received before the last timing failure'
        self._recover_after = None
        self.fps = self.rates[0]
        return ''

    def _times(self, rows):
        """Time axis (s) of sample rows in the active time base."""
        rows = np.asarray(rows, dtype=float).reshape(-1, 6)
        if self.nominal_fps is None:
            return rows[:, 0]
        return (rows[:, 5] - self._anchor[1]) / self.fps

    def _time1(self, row):
        """Scalar time of one sample row (hot path, no numpy)."""
        return row[0] if self.nominal_fps is None else (row[5] - self._anchor[1]) / self.fps

    def update(self, timestamp, h_mm, amplitude, internal=False, valid=True, sequence=None):
        if not math.isfinite(timestamp) or (self.nominal_fps is not None and sequence is None):
            self.samples.clear()
            self._anchor = None
            self.latest = FlowFeatures('TIMING_UNCERTAIN')
            return self.latest
        # Equal timestamps are normal (socket batching, 15.625 ms Windows clock);
        # only a clock going backwards or a lost frame breaks the window.
        first = self._anchor is None
        discontinuity = not first and ((self.samples and timestamp < self.samples[-1][0]) or (
            sequence is not None and self.last_sequence is not None and sequence != self.last_sequence + 1))
        self.last_sequence = sequence
        if first or discontinuity:
            self._reset(timestamp, sequence)
        if discontinuity:
            self.latest = FlowFeatures('TIMING_UNCERTAIN', timestamp)
            self.timing_reason = 'discontinuity: receive time decreased or a frame sequence was lost'
        valid = bool(valid and math.isfinite(h_mm) and h_mm > 0 and math.isfinite(amplitude) and amplitude > 0)
        self.samples.append((timestamp, h_mm, amplitude, float('nan') if internal is None else bool(internal),
                             valid, float('nan') if sequence is None else float(sequence)))
        now_axis = self._time1(self.samples[-1])
        while self.samples and now_axis - self._time1(self.samples[0]) > self.window_s:
            self.samples.popleft()
        if not valid:
            # Label only this row: keeping it in ``latest`` would hold INVALID_DATA for
            # ``update_s`` and defeat the >= 90 % valid-fraction rule of the window.
            return FlowFeatures('INVALID_DATA', timestamp, valid_fraction=self.latest.valid_fraction)
        if discontinuity:
            return self.latest
        if (math.isfinite(self.latest.computed_at_s) and
                timestamp - self.latest.computed_at_s < self.update_s):
            return self.latest
        self.latest = self._compute(timestamp)
        return self.latest

    def _compute(self, now):
        x = np.asarray(self.samples, dtype=float)
        axis = self._times(x)
        if len(x) < 30 or axis[-1] - axis[0] < 0.95 * self.window_s:
            return FlowFeatures('WARMUP', now)
        if self.nominal_fps is not None:
            # Timing is judged on all frames (valid or not): a validity gap is not a clock fault.
            reason = self._check_rate(x)
            if reason:
                self.timing_reason = reason
                return FlowFeatures('TIMING_UNCERTAIN', now, valid_fraction=float(x[:, 4].mean()))
            axis = self._times(x)      # re-evaluated with the validated rate
        good = x[:, 4].astype(bool)
        fraction = float(good.mean())
        if fraction < 0.9:
            return FlowFeatures('INVALID_DATA', now, valid_fraction=fraction)
        t, h, amp = axis[good], x[good, 1], x[good, 2]
        dt = np.diff(t)
        # A gap of > 0.1 s of invalid frames is a data gap in either time base. The spread
        # test only applies to PC receive times: the sequence axis is uniform by
        # construction and its rate was validated in _check_rate.
        if dt.max() > 0.1 or (self.nominal_fps is None and np.std(dt) / np.mean(dt) > 0.5):
            self.timing_reason = ('gap > 0.1 s without valid frames' if dt.max() > 0.1
                                  else 'irregular receive intervals (receive time base)')
            return FlowFeatures('TIMING_UNCERTAIN', now, valid_fraction=fraction)
        self.timing_reason = ''
        mean, sigma = float(np.mean(h)), float(np.std(h))
        cv = float(np.std(amp) / np.mean(amp))
        internal_fraction = float(np.mean(x[good, 3]))
        pp = float(np.ptp(h))
        wave = float('nan')
        span = t[-1] - t[0]
        uniform = np.interp(np.linspace(t[0], t[-1], len(t)), t, h)
        oscillation = detrend(uniform)
        power = np.abs(np.fft.rfft(oscillation * np.hanning(len(t)))) ** 2
        frequencies = np.fft.rfftfreq(len(t), span / (len(t) - 1))
        resolution = frequencies[1]
        band = (frequencies >= max(resolution, 0.5 - resolution)) & (frequencies <= 15)
        # At least two cycles, substantial oscillatory variance, and a concentrated
        # spectral peak: a linear drift or broadband noise is not labeled a wave.
        if band.any() and sigma > 0.2 and np.std(oscillation) > 0.75 * sigma:
            indices = np.flatnonzero(band)
            peak = indices[np.argmax(power[band])]
            total = float(power[1:].sum())
            concentration = float(power[max(1, peak-1):peak+2].sum()) / max(total, 1e-30)
            estimated = float(frequencies[peak])
            if 0 < peak < len(power)-1:
                left, mid, right = np.log(np.maximum(power[peak-1:peak+2], 1e-30))
                curvature = left - 2*mid + right
                if curvature < 0:
                    estimated += float(np.clip(0.5*(left-right)/curvature, -.5, .5)) * resolution
            if (concentration >= 0.65 and power[band].sum() / max(total, 1e-30) >= 0.6
                    and estimated * span >= 1.9 and 0.5 - resolution/2 <= estimated <= 15):
                wave = estimated
        centered = h - mean
        skew = float(np.mean(centered ** 3) / sigma ** 3) if sigma > 1e-6 else 0.0
        negative_spike = sigma > 0.2 and skew < -1 and np.min(centered) < -3 * sigma
        if cv > 0.35 and (internal_fraction >= 0.05 or negative_spike):
            pattern = 'INTERMITTENT_CANDIDATE'
        elif mean > 2 and sigma > 0.2 and math.isfinite(wave):
            pattern = 'WAVY_CANDIDATE'
        elif internal_fraction >= 0.05:
            pattern = 'INTERNAL_ECHO_CANDIDATE'
        elif sigma < 0.05 and cv < 0.1:
            pattern = 'THIN_STABLE_CANDIDATE' if mean < 2 else 'SMOOTH_CANDIDATE' if mean > 2 else 'UNKNOWN'
        else:
            pattern = 'UNKNOWN'
        return FlowFeatures(pattern, now, mean, sigma, cv, pp, wave, internal_fraction, fraction)
