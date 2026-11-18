"""Synthetic pattern tests do not establish physical flow-regime accuracy."""
import csv
from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ultrasonic_film.flow import FlowRegimeClassifier
from ultrasonic_film.processing import EchoConfig, EchoDSP
from ultrasonic_film import massflow as mf


class FlowTests(unittest.TestCase):
    def simulate(self, kind, window=2):
        classifier = FlowRegimeClassifier(window, update_s=0.05)
        t = np.arange(int(300 * (window + 1))) / 300
        for i, s in enumerate(t):
            h, amp, internal = 10 + 0.01 * np.sin(2*np.pi*3*s), 1., False
            if kind == 'thin':
                h -= 9.5
            elif kind == 'wavy':
                h = 10 + 0.6 * np.sin(2*np.pi*3*s)
            elif kind == 'slow':
                h = 10 + 0.6 * np.sin(2*np.pi*0.5*s)
            elif kind == 'intermittent':
                internal = i % 20 < 5
                amp = 0.1 if internal else 1.0
                h = 5.0 if internal else 10.0
            elif kind == 'drift':
                h = 10 + s
            f = classifier.update(s, h, amp, internal, sequence=i)
        return classifier, f

    def test_four_patterns_and_wave_metrics(self):
        for kind, pattern in [('smooth', 'SMOOTH_CANDIDATE'), ('thin', 'THIN_STABLE_CANDIDATE'),
                              ('wavy', 'WAVY_CANDIDATE'), ('intermittent', 'INTERMITTENT_CANDIDATE')]:
            with self.subTest(kind=kind):
                _, f = self.simulate(kind)
                self.assertEqual(f.pattern, pattern)
                if kind == 'wavy':
                    self.assertAlmostEqual(f.f_wave_hz, 3, delta=0.15)
                    self.assertAlmostEqual(f.wave_h_pp_mm, 1.2, delta=0.01)
                    self.assertAlmostEqual(f.sigma_h_mm, 0.6 / np.sqrt(2), delta=0.02)

    def test_slow_wave_requires_longer_window(self):
        _, short = self.simulate('slow')
        _, long = self.simulate('slow', window=5)
        self.assertTrue(math.isnan(short.f_wave_hz))
        self.assertAlmostEqual(long.f_wave_hz, 0.5, delta=0.11)

    def test_drift_is_not_a_wave(self):
        _, f = self.simulate('drift')
        self.assertEqual(f.pattern, 'UNKNOWN')
        self.assertTrue(math.isnan(f.f_wave_hz))

    def test_unavailable_internal_analysis_remains_unknown(self):
        c = FlowRegimeClassifier()
        for i in range(900):
            f = c.update(i/300, .5, 1., internal=None)
        self.assertEqual(f.pattern, 'THIN_STABLE_CANDIDATE')
        self.assertTrue(math.isnan(f.internal_fraction))

    def test_invalid_frame_and_sequence_gap_clear_label(self):
        c, _ = self.simulate('smooth')
        f = c.update(3., float('nan'), 1., valid=False, sequence=900)
        self.assertEqual(f.pattern, 'INVALID_DATA')
        f = c.update(3.01, 10., 1., sequence=903)
        self.assertEqual(f.pattern, 'TIMING_UNCERTAIN')
        self.assertTrue(math.isnan(f.f_wave_hz))

    def test_jitter_and_dropouts_are_not_classified(self):
        for kind in ['jitter', 'dropouts']:
            c = FlowRegimeClassifier()
            t = 0.
            for i in range(1000):
                t += (0.0001 if i % 2 else 0.0065) if kind == 'jitter' else 1/300
                f = c.update(t, 10., 1., valid=kind != 'dropouts' or i % 2 == 0)
            self.assertIn(f.pattern, ['TIMING_UNCERTAIN', 'INVALID_DATA'])

    def test_features_and_csv_available_from_processor(self):
        fs = 62.5e6
        p = mf.EchoProcessor(flow_features=True)
        p.dsp = mf.EchoDSP(fs, secondary_peaks=True)
        t = np.arange(16384) / fs * 1e6
        pulse = lambda center, amp: amp*np.exp(-0.5*((t-center)/0.25)**2)*np.sin(2*np.pi*3.5*(t-center))
        counts = np.rint((pulse(4.5, .25)+pulse(32.1, .1))*8192).astype(np.int16)
        with tempfile.TemporaryDirectory() as folder:
            w = mf.TofWriter(Path(folder)/'tof.csv', flow_features=True)
            try:
                for k in range(0, 640, 64):
                    p._process([('frame', i, i/300, counts, w, i, i/300) for i in range(k, k+64)])
            finally:
                w.close()
            with (Path(folder)/'tof.csv').open() as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 640)
        self.assertIn('flow', p.latest)
        self.assertEqual(rows[-1]['flow_pattern'], 'SMOOTH_CANDIDATE')
        self.assertGreater(float(rows[-1]['h_mm']), 10)


class SecondaryPeakTests(unittest.TestCase):
    def test_internal_peak_and_reverberation_rejection_preserve_main(self):
        config = replace(EchoConfig(), threshold_noise=None, noise_threshold=.1,
                         secondary_peaks=True, plate_period_us=3.744)
        dsp = EchoDSP(62.5e6, config)
        t = dsp.waktu_us
        bump = lambda center, amp: amp*np.exp(-0.5*((t-center)/.15)**2)
        row = bump(4.5, 1.)+bump(4.5+3.744, .14)+bump(14.5, .4)+bump(32.1, 1.)+bump(60, .3)
        e = dsp.detect(row)
        plain = EchoDSP(62.5e6, replace(config, secondary_peaks=False)).detect(row)
        self.assertEqual(e[:4], plain[:4])
        self.assertEqual(e.n_peaks, 2)
        self.assertAlmostEqual(e.tof_internal_us, 10, delta=.04)

    def test_surface_only_and_missing_echo_are_not_bubbles(self):
        dsp = EchoDSP(62.5e6, replace(EchoConfig(), secondary_peaks=True, threshold_noise=None))
        t = dsp.waktu_us
        row = np.exp(-((t-4.5)/.2)**2)+np.exp(-((t-32.1)/.2)**2)
        e = dsp.detect(row)
        self.assertEqual(e.n_peaks, 1)
        self.assertTrue(math.isnan(e.tof_internal_us))
        self.assertEqual(dsp.detect(row*0).n_peaks, 0)

    def test_plate_rejection_edges_are_not_internal_peaks(self):
        """Review M1: hard zeroing left reverberation edges that looked like internal peaks."""
        fs = 62.5e6
        t = np.arange(8000) / fs * 1e6
        pulse = lambda c, a: a * np.exp(-.5 * ((t - c) / .25) ** 2) * np.sin(2 * np.pi * 3.5 * (t - c))
        rng = np.random.default_rng(3)
        for water, bubble in ((0.03, 0.0), (0.05, 0.0), (0.10, 0.04)):
            with self.subTest(water=water, bubble=bubble):
                frames = np.array([pulse(4.5, .25) + sum(pulse(4.5 + k * 3.744, .25 * .15 ** k) for k in (1, 2, 3))
                                   + pulse(24.5, water) + pulse(17.5, bubble) + rng.normal(0, .004, t.size)
                                   for _ in range(300)])
                dsp = EchoDSP(fs, replace(EchoConfig(), secondary_peaks=True, plate_period_us=3.744))
                echoes = [dsp.detect(row) for row in dsp.envelope(frames)]
                internal = np.array([e.tof_internal_us for e in echoes])
                self.assertAlmostEqual(np.nanmedian([e.tof_air_us for e in echoes]), 20.0, delta=0.05)
                if bubble:   # a real internal echo away from the plate windows is still found
                    self.assertGreater(np.mean(np.isfinite(internal)), 0.95)
                    self.assertAlmostEqual(np.nanmedian(internal), 13.0, delta=0.05)
                else:        # before the fix: 37 % at +4.56 µs (0.03 V) and 12 % at +2.94 µs (0.05 V)
                    self.assertLess(np.mean(np.isfinite(internal)), 0.01)


def windows_clock(t):
    """Python time.monotonic() on Windows (GetTickCount64): 15.625 ms steps."""
    return np.floor(np.asarray(t) / 0.015625) * 0.015625


class TimebaseTests(unittest.TestCase):
    """Receive times on the lab PC came from a 15.625 ms clock (78.6 % of dt == 0)."""

    def feed(self, classifier, times, fps=300, n=1500, f_hz=3.0, invalid=None):
        labels = []
        for i in range(n):
            h = 3.0 + 0.3 * math.sin(2 * math.pi * f_hz * i / fps)
            ok = invalid is None or not invalid[i]
            labels.append(classifier.update(times[i], h, 1.0, internal=False, valid=ok, sequence=i))
        return labels

    def test_quantised_windows_timestamps_pattern(self):
        times = windows_clock(np.arange(1500) / 300)
        self.assertEqual(list(np.round(np.diff(times[:6]) * 1e3, 3)), [0.0, 0.0, 0.0, 0.0, 15.625])
        sequence = self.feed(FlowRegimeClassifier(nominal_fps=300), times)
        patterns = [f.pattern for f in sequence[900:]]
        self.assertEqual(set(patterns), {'WAVY_CANDIDATE'})
        self.assertAlmostEqual(sequence[-1].f_wave_hz, 3.0, delta=0.05)
        # Receive time base: equal timestamps no longer clear the window (old flow.py:48),
        # but the irregular clock is still reported instead of guessed.
        receive = FlowRegimeClassifier()
        labels = self.feed(receive, times)
        self.assertGreater(len(receive.samples), 500)
        self.assertEqual(labels[-1].pattern, 'TIMING_UNCERTAIN')

    def test_effective_fps_estimated_from_coarse_clock(self):
        true_fps = 299.5                       # live server logs 299.3-299.7 FPS
        times = windows_clock(np.arange(1500) / true_fps)
        c = FlowRegimeClassifier(nominal_fps=300)
        labels = self.feed(c, times, fps=true_fps)
        self.assertAlmostEqual(c.fps, true_fps, delta=true_fps * 0.005)
        self.assertAlmostEqual(labels[-1].f_wave_hz, 3.0, delta=0.03)

    def run_rate(self, receive, true_t, f_hz=3.0):
        """Wave on the true acquisition time; classifier sees receive times + sequence only."""
        c = FlowRegimeClassifier(nominal_fps=300)
        labels = [c.update(receive[i], 3.0 + 0.3 * math.sin(2 * math.pi * f_hz * true_t[i]), 1.0,
                           internal=False, sequence=i) for i in range(len(true_t))]
        return c, labels

    def assert_no_wrong_frequency(self, labels, f_hz=3.0, tolerance=0.1):
        for f in labels:
            if f.pattern == 'WAVY_CANDIDATE' or math.isfinite(f.f_wave_hz):
                self.assertEqual(f.pattern, 'WAVY_CANDIDATE')
                self.assertAlmostEqual(f.f_wave_hz, f_hz, delta=tolerance)

    def test_actual_rate_half_of_nominal_is_timing_uncertain(self):
        """Review: 150 FPS data with nominal 300 was reported as a valid ~6 Hz wave."""
        true_t = np.arange(2400) / 150
        for receive in (true_t, windows_clock(true_t)):
            c, labels = self.run_rate(receive, true_t)
            self.assertNotIn('WAVY_CANDIDATE', {f.pattern for f in labels})
            self.assertTrue(all(math.isnan(f.f_wave_hz) for f in labels))
            self.assertEqual(labels[-1].pattern, 'TIMING_UNCERTAIN')
            self.assertAlmostEqual(c.rates[0], 150, delta=3)
            self.assertIn('outside', c.timing_reason)

    def test_slowdown_after_normal_start_is_detected(self):
        onset, window = 1200, 600                       # 4 s at 300 FPS, then 200 FPS
        true_t = np.r_[np.arange(onset) / 300, (onset - 1) / 300 + np.arange(1, 1501) / 200]
        c, labels = self.run_rate(windows_clock(true_t), true_t)
        self.assertIn('WAVY_CANDIDATE', [f.pattern for f in labels[:onset]])
        self.assert_no_wrong_frequency(labels)
        late = [f.pattern for f in labels[onset + window // 2:]]
        self.assertNotIn('WAVY_CANDIDATE', late)     # the old result is not carried on as valid
        self.assertEqual(labels[-1].pattern, 'TIMING_UNCERTAIN')
        self.assertTrue(math.isnan(labels[-1].f_wave_hz))

    def test_bunched_arrivals_do_not_give_a_wrong_valid_frequency(self):
        rng = np.random.default_rng(11)
        true_t = np.arange(3000) / 300
        bunched = np.ceil(true_t * 30) / 30                          # groups of 10, equal timestamps
        delayed = np.maximum.accumulate(true_t + rng.exponential(0.008, true_t.size))
        for receive in (bunched, delayed, windows_clock(delayed)):
            c, labels = self.run_rate(receive, true_t)
            self.assert_no_wrong_frequency(labels, tolerance=0.1)
            # batching alone is not a device fault: the 3 Hz wave is still reported
            self.assertGreater([f.pattern for f in labels[900:]].count('WAVY_CANDIDATE') / 2100, 0.95)
        slow = np.arange(2400) / 150                                  # bunched AND wrong rate
        c, labels = self.run_rate(np.ceil(slow * 15) / 15, slow)
        self.assertTrue(all(math.isnan(f.f_wave_hz) for f in labels))

    def test_valid_again_only_after_a_full_window_of_good_timing(self):
        window = 600
        good1 = np.arange(1200) / 300
        slow = good1[-1] + np.arange(1, 451) / 150                   # 3 s at 150 FPS
        good2 = slow[-1] + np.arange(1, 1801) / 300
        true_t = np.r_[good1, slow, good2]
        recovered = len(good1) + len(slow)
        c, labels = self.run_rate(windows_clock(true_t), true_t)
        self.assert_no_wrong_frequency(labels)
        patterns = [f.pattern for f in labels]
        self.assertIn('TIMING_UNCERTAIN', patterns[len(good1):recovered + window])
        self.assertNotIn('WAVY_CANDIDATE', patterns[len(good1) + window // 2:recovered + window])
        self.assertEqual(patterns[-1], 'WAVY_CANDIDATE')
        self.assertAlmostEqual(labels[-1].f_wave_hz, 3.0, delta=0.05)

    def test_sequence_timebase_requires_sequence_and_resets_on_gap(self):
        c = FlowRegimeClassifier(nominal_fps=300)
        self.assertEqual(c.update(0.0, 3.0, 1.0).pattern, 'TIMING_UNCERTAIN')
        labels = self.feed(c, windows_clock(np.arange(900) / 300), n=900)
        self.assertEqual(labels[-1].pattern, 'WAVY_CANDIDATE')
        self.assertEqual(c.update(3.1, 3.0, 1.0, sequence=950).pattern, 'TIMING_UNCERTAIN')
        self.assertEqual(len(c.samples), 1)

    def test_scattered_invalid_frames_do_not_hold_invalid_label(self):
        """Review H2: one invalid frame used to hold INVALID_DATA for update_s (0.1 s)."""
        rng = np.random.default_rng(7)
        for rate in (0.01, 0.05, 0.09):
            with self.subTest(invalid_rate=rate):
                invalid = rng.random(3000) < rate
                c = FlowRegimeClassifier(nominal_fps=300)
                labels = self.feed(c, np.arange(3000) / 300, n=3000, invalid=invalid)
                valid_rows = [f for f, bad in zip(labels[900:], invalid[900:]) if not bad]
                self.assertTrue(all(f.pattern == 'INVALID_DATA'
                                    for f, bad in zip(labels[900:], invalid[900:]) if bad))
                # Valid rows: INVALID_DATA only from the window rule (< 90 % valid), never a hold-off.
                self.assertTrue(all(f.valid_fraction < 0.9 for f in valid_rows if f.pattern == 'INVALID_DATA'))
                if rate <= 0.05:   # flow.py at 7272fee: 74.1 % (1 %) and 20.9 % (5 %) of these rows were WAVY
                    wavy = sum(f.pattern == 'WAVY_CANDIDATE' for f in valid_rows) / len(valid_rows)
                    self.assertGreater(wavy, 0.99)
        c = FlowRegimeClassifier(nominal_fps=300)
        invalid = np.zeros(1500, dtype=bool)
        invalid[600:] = rng.random(900) < 0.2          # > 10 % invalid still rejects the window
        self.assertEqual(self.feed(c, np.arange(1500) / 300, invalid=invalid)[-1].pattern, 'INVALID_DATA')

    def test_reprocess_uses_sequence_timebase_for_quantised_recordings(self):
        fs = 62.5e6
        t = np.arange(16384) / fs * 1e6
        pulse = lambda c, a: a * np.exp(-.5 * ((t - c) / .25) ** 2) * np.sin(2 * np.pi * 3.5 * (t - c))
        from ultrasonic_film.recording import FrameRecorder
        with tempfile.TemporaryDirectory() as folder:
            receive = windows_clock(np.arange(900) / 300)
            with FrameRecorder(folder, sample_rate_hz=fs, trigger='AWG_PE', host='test') as rec:
                for i in range(900):
                    surface = 32.1 + 0.8 * math.sin(2 * math.pi * 3 * i / 300)   # h ~ 20 +- 0.6 mm
                    counts = np.rint((pulse(4.5, .25) + pulse(surface, .2)) * 8192).astype(np.int16)
                    rec.append(i, counts.tobytes(), receive[i])
            for timebase, expected in (('sequence', 'WAVY_CANDIDATE'), ('receive', 'TIMING_UNCERTAIN')):
                with self.subTest(timebase=timebase):
                    mf.reprocess(folder, show_plot=False, flow_features=True, flow_timebase=timebase)
                    with (Path(folder) / 'tof_offline.csv').open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(rows[-1]['flow_pattern'], expected)
                    if timebase == 'sequence':
                        self.assertAlmostEqual(float(rows[-1]['f_wave_hz']), 3.0, delta=0.05)

    def test_flow_without_plate_rejection_warns(self):
        saved = mf.PANTULAN_US
        try:
            mf.PANTULAN_US = None
            self.assertIn('--pantulan-us', mf.flow_plate_warning(True))
            self.assertIsNone(mf.flow_plate_warning(False))
            mf.PANTULAN_US = 3.744
            self.assertIsNone(mf.flow_plate_warning(True))
        finally:
            mf.PANTULAN_US = saved


if __name__ == '__main__':
    unittest.main()
