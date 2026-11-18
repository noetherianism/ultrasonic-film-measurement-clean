"""Synthetic thin-layer baseline removal and offline recording integration tests."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ultrasonic_film import thin_layer as lt
from ultrasonic_film import massflow as mf
from ultrasonic_film.recording import FrameRecorder, SAMPLES_PER_FRAME

FS = 62.5e6
C = 1466.0
T_AK = 4.5


def frames(n, seed, h_mm=None, rasio=0.8, air_tebal_us=27.6, f_mhz=3.2, amp_akrilik=0.3):
    """Volt mentah: TX, echo akrilik 3,2 MHz (+jitter 20 ns), pantulan plat, air tebal ATAU lapisan tipis."""
    rng = np.random.default_rng(seed)
    t = np.arange(SAMPLES_PER_FRAME) / FS * 1e6
    pulse = lambda t0, amp: amp * np.exp(-0.5 * ((t - t0) / 0.25) ** 2) * np.sin(2 * np.pi * f_mhz * (t - t0))
    out = []
    for _ in range(n):
        j = rng.normal(0, 0.02)
        x = pulse(1.0, 0.8) + pulse(T_AK + j, amp_akrilik) + pulse(T_AK + j + 3.744, amp_akrilik * 0.15)
        if h_mm is None:
            x += pulse(T_AK + j + air_tebal_us, 0.25)
        else:
            tau, amp, k = 2 * h_mm * 1e-3 / C * 1e6, 0.3 * rasio, 1
            while abs(amp) > 0.003:
                x += pulse(T_AK + j + k * tau, amp)
                amp *= -0.37
                k += 1
        out.append(x + rng.normal(0, 0.004, t.size))
    return np.array(out)


def rekam(directory, volts):
    counts = np.clip(np.round(volts / mf.VOLT_PER_COUNT), -8192, 8191).astype('<i2')
    with FrameRecorder(directory, sample_rate_hz=FS, trigger='AWG_PE', host='test') as rec:
        for i, row in enumerate(counts):
            rec.append(i, row.tobytes(), i / 300)
    return directory


class BaselineTipisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = lt.BaselineTipis.dari_rekaman(frames(200, 1)[:, :mf.IDX_MAX], FS, T_AK, 27.6)

    def test_half_millimetre_layer_measured(self):
        hasil = self.baseline.proses(frames(80, 2, h_mm=0.5)[:, :mf.IDX_MAX])
        ok = [r for r in hasil if r[0] == 'OK']
        self.assertGreaterEqual(len(ok) / len(hasil), 0.95)
        h = C * np.array([r[2] for r in ok]) / 2000
        self.assertAlmostEqual(np.median(h), 0.5, delta=0.02)
        self.assertLess(np.std(h), 0.02)

    def test_thicker_layer_and_strong_echo(self):
        for h_mm, rasio in ((1.5, 0.8), (0.7, 2.3)):
            with self.subTest(h=h_mm, rasio=rasio):
                hasil = self.baseline.proses(frames(40, 3, h_mm=h_mm, rasio=rasio)[:, :mf.IDX_MAX])
                h = C * np.array([r[2] for r in hasil if r[0] == 'OK']) / 2000
                self.assertGreaterEqual(h.size, 38)
                self.assertAlmostEqual(np.median(h), h_mm, delta=0.03)

    def test_no_layer_not_detected(self):
        hasil = self.baseline.proses(frames(100, 4)[:, :mf.IDX_MAX])   # air tebal: tidak ada echo di jendela cari
        status = [r[0] for r in hasil]
        self.assertGreaterEqual(status.count('AIR_TIPIS_TIDAK_TERDETEKSI') / len(status), 0.97)

    def test_threshold_from_baseline_residual(self):
        b = self.baseline
        self.assertEqual(b.threshold.shape, (b.n,))
        self.assertTrue(np.all(b.threshold[b.cari] > 0))
        self.assertAlmostEqual(b.info['t_akrilik_us'], T_AK, delta=0.05)
        for key in ('fit_us', 'cari_us', 'filter_hz', 'faktor_thr', 'sisa_p999_maks'):
            self.assertIn(key, b.info)

    def test_changed_coupling_flagged_not_reported_as_layer(self):
        moved = frames(80, 8, f_mhz=2.9, amp_akrilik=0.24)[:, :mf.IDX_MAX]   # transduser dipasang ulang
        b = lt.BaselineTipis.dari_rekaman(frames(200, 1)[:, :mf.IDX_MAX], FS, T_AK, 27.6)
        status = [r[0] for r in b.proses(moved)]
        self.assertGreater(b.cocok_rasio, lt.COCOK_FAKTOR)
        self.assertGreaterEqual(status.count('BASELINE_TIDAK_COCOK') / len(status), 0.9)
        self.assertLessEqual(status.count('OK'), 4)
        # Review L1: frames 1-3 used to skip the check (len >= 4); now checked from the first frame.
        first = lt.BaselineTipis.dari_rekaman(frames(200, 1)[:, :mf.IDX_MAX], FS, T_AK, 27.6)
        self.assertEqual([r[0] for r in first.proses(moved[:3])], ['BASELINE_TIDAK_COCOK'] * 3)
        b_off = lt.BaselineTipis.dari_rekaman(frames(200, 1)[:, :mf.IDX_MAX], FS, T_AK, 27.6, cocok_faktor=0)
        self.assertNotIn('BASELINE_TIDAK_COCOK', [r[0] for r in b_off.proses(moved)][4:])
        same = lt.BaselineTipis.dari_rekaman(frames(200, 1)[:, :mf.IDX_MAX], FS, T_AK, 27.6)
        same.proses(frames(80, 9)[:, :mf.IDX_MAX])
        self.assertLess(same.cocok_rasio, 2.0)

    def test_shallow_baseline_rejected(self):
        with self.assertRaisesRegex(ValueError, 'air tebal'):
            lt.BaselineTipis.dari_rekaman(frames(20, 5, air_tebal_us=6.0)[:, :mf.IDX_MAX], FS, T_AK, 6.0)


class MassfloBaselineTests(unittest.TestCase):
    def test_reprocess_with_baseline_writes_height(self):
        with tempfile.TemporaryDirectory() as directory:
            tebal = rekam(Path(directory) / 'tebal', frames(160, 6))
            tipis = rekam(Path(directory) / 'tipis', frames(60, 7, h_mm=0.5))
            try:
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    mf.main(['--proses-ulang', str(tipis), '--tanpa-plot', '--baseline', str(tebal)])
                self.assertIsNotNone(mf.BASELINE)
            finally:
                mf.BASELINE = None
            with (tipis / 'tof_offline.csv').open(encoding='utf-8') as recording:
                rows = list(csv.DictReader(recording))
        self.assertIn('Baseline lapisan tipis', out.getvalue())
        ok = [r for r in rows if r['status'] == 'OK']
        self.assertGreaterEqual(len(ok) / len(rows), 0.9)
        h = np.array([float(r['h_mm']) for r in ok])
        self.assertAlmostEqual(np.median(h), 0.5, delta=0.03)

    def test_cli_validation(self):
        for argv in (['--baseline', 'tidak_ada_folder'], ['--baseline-faktor', '2'], ['--baseline-cocok', '3']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mf.parse_args(argv)


if __name__ == '__main__':
    unittest.main()
