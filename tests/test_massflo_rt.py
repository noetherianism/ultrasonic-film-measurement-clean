"""Echo DSP regression tests and acquisition/recording checks using a native simulator."""
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, sosfiltfilt
from dataclasses import FrozenInstanceError, replace

from ultrasonic_film.processing import EchoConfig, EchoDSP

ROOT = Path(__file__).resolve().parents[1]
from ultrasonic_film import massflow as mf
from ultrasonic_film.recording import load_recording

FS = 62.5e6


def reference_scpi_algorithm(data):
    """Salinan langsung loop DSP massflo_single_rt.py versi SCPI (sebelum digabung)."""
    nyq = 0.5 * FS
    b, a = butter(8, [mf.FILTER_LOW / nyq, mf.FILTER_HIGH / nyq], btype='band')
    waktu_us = np.arange(mf.IDX_MAX) / FS * 1e6
    data_filtered = filtfilt(b, a, data) * mf.DIGITAL_GAIN
    env = np.convolve(np.abs(hilbert(data_filtered)), np.ones(mf.MA_WINDOW) / mf.MA_WINDOW, mode='same')
    env[env < mf.NOISE_THRESHOLD] = 0.0
    env[waktu_us < mf.BLANK_US] = 0.0
    mask = (waktu_us > 2.5) & (waktu_us < 6.0)
    waktu_akrilik = waktu_air = 0.0
    if np.max(env[mask]) > mf.NOISE_THRESHOLD:
        idx = np.where(mask)[0][np.argmax(env[mask])]
        waktu_akrilik = waktu_us[idx]
        rel = waktu_us - waktu_akrilik
        mask_air = (rel > 1.5) & (rel < 120.0)
        if np.max(env[mask_air]) > mf.NOISE_THRESHOLD:
            waktu_air = rel[np.where(mask_air)[0][np.argmax(env[mask_air])]]
    return waktu_akrilik, waktu_air


def synthetic(t_akrilik, t_air, amp_air, noise, seed):
    t = np.arange(mf.IDX_MAX) / FS * 1e6
    rng = np.random.default_rng(seed)
    pulse = lambda t0, amp: amp * np.exp(-0.5 * ((t - t0) / 0.4) ** 2) * np.sin(2 * np.pi * 4.0 * t)
    x = pulse(1.0, 0.8) + pulse(t_akrilik, 0.3) + rng.normal(0, noise, t.size)
    if t_air is not None:
        x += pulse(t_akrilik + t_air, amp_air)
    return x


class DSPTests(unittest.TestCase):
    def test_explicit_configuration_is_independent_of_cli_changes(self):
        config = EchoConfig(threshold_noise=None, noise_threshold=0.15, ma_window=128)
        dsp = EchoDSP(FS, config)
        signal = synthetic(4.0, 10.0, 0.08, 0.004, 1)
        expected = dsp.envelope(signal)
        with dsp_options('--ma', '16', '--threshold', '0.5'):
            np.testing.assert_array_equal(dsp.envelope(signal), expected)
            cli_dsp = mf.EchoDSP(FS)
        self.assertEqual(cli_dsp.config.ma_window, 16)
        self.assertEqual(cli_dsp.config.noise_threshold, 0.5)
        self.assertEqual(dsp.config, config)
        with self.assertRaises(FrozenInstanceError):
            config.ma_window = 64

    def test_explicit_count_scale_and_window(self):
        config = replace(EchoConfig(), sample_count=1000, threshold_noise=None, volt_per_count=0.25)
        dsp = EchoDSP(FS, config)
        counts = np.arange(16384, dtype=np.int16)
        volts = dsp.to_volts(counts)
        self.assertEqual(volts.shape, (1000,))
        np.testing.assert_array_equal(volts, counts[:1000] * 0.25)

    def test_moving_average_matches_numpy_convolve_same(self):
        dsp = mf.EchoDSP(FS)
        x = np.random.default_rng(1).random((3, mf.IDX_MAX))
        csum = np.zeros((3, mf.IDX_MAX + 1))
        np.cumsum(x, axis=-1, out=csum[:, 1:])
        fast = (csum[:, dsp._ma_hi] - csum[:, dsp._ma_lo]) / mf.MA_WINDOW
        for row, got in zip(x, fast):
            np.testing.assert_allclose(got, np.convolve(row, np.ones(mf.MA_WINDOW) / mf.MA_WINDOW, 'same'), atol=1e-12)

    def test_batch_tof_matches_original_scpi_algorithm(self):
        with dsp_options('--ma', '128', '--threshold', '0.15'):   # mode manual = algoritma lama
            self._compare_with_reference()

    def _compare_with_reference(self):
        dsp = mf.EchoDSP(FS)
        cases = [(4.0, 10.0, 0.08), (3.5, 2.5, 0.1), (5.0, 30.0, 0.05), (4.2, 80.0, 0.06),
                 (4.0, None, 0.0), (None, None, 0.0)]
        signals = []
        for i, (ak, air, amp) in enumerate(cases):
            x = synthetic(ak if ak else 4.0, air, amp, 0.004, i)
            if ak is None:
                x = np.random.default_rng(9).normal(0, 0.004, mf.IDX_MAX)
            signals.append(x)
        env = dsp.envelope(np.array(signals))
        for (ak, air, _), x, row in zip(cases, signals, env):
            ref_ak, ref_air = reference_scpi_algorithm(x)
            echo = dsp.detect(row)
            with self.subTest(akrilik=ak, air=air):
                if ref_ak == 0:
                    self.assertEqual(echo.status, 'AKRILIK_TIDAK_TERDETEKSI')
                    continue
                self.assertAlmostEqual(echo.waktu_akrilik_us, ref_ak, delta=2 / FS * 1e6)
                if ref_air == 0:
                    self.assertEqual(echo.status, 'AIR_TIDAK_TERDETEKSI')
                else:
                    self.assertEqual(echo.status, 'OK')
                    self.assertAlmostEqual(echo.tof_air_us, ref_air, delta=2 / FS * 1e6)
                    if air is not None:
                        self.assertAlmostEqual(echo.tof_air_us, air, delta=0.15)

    def test_display_curve_removes_multiple_reflection_only_for_display(self):
        dsp = mf.EchoDSP(FS)
        row = dsp.envelope(synthetic(4.0, 10.0, 0.08, 0.0, 0))
        echo = dsp.detect(row)
        rel, shown = dsp.display_curve(row, echo)
        self.assertTrue(np.all(shown[rel > echo.tof_air_us + mf.PANTULAN_GANDA_US] == 0))
        self.assertEqual(dsp.detect(row), echo)

    @unittest.skipUnless(os.environ.get('RUN_PERFORMANCE_TESTS') == '1', 'timing benchmark requires an idle host; opt in with RUN_PERFORMANCE_TESTS=1')
    def test_dsp_keeps_up_with_300_fps(self):
        dsp = mf.EchoDSP(FS)
        batch = np.array([synthetic(4.0, 10.0, 0.08, 0.004, i) for i in range(64)])
        start = time.perf_counter()
        for row in dsp.envelope(batch):
            dsp.detect(row)
        per_frame = (time.perf_counter() - start) / 64
        self.assertLess(per_frame, 1 / 300 * 0.8, f'{per_frame*1e3:.2f} ms/frame')


@unittest.skipUnless(shutil.which('g++') and sys.platform == 'linux', 'native simulator requires Linux g++')
class NativeStreamTests(unittest.TestCase):
    def test_300fps_stream_records_every_frame_with_tof_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory) / 'live')
            subprocess.run(['g++', '-std=c++20', '-O2', '-I' + str(ROOT / 'tests/fake_api'),
                            str(ROOT / 'ultrasonic_film/native/rp_acquisition.cpp'), '-pthread',
                            '-o', executable], check=True, capture_output=True)
            process = subprocess.Popen([executable, '0', 'direct', '300', 'ultrasound', '5000000', '1', '1',
                                        '2', 'single-ch1', 'live'], stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True)
            log, ready = [], threading.Event()
            def read_log():
                for line in process.stdout:
                    log.append(line)
                    if line.startswith('READY '):
                        ready.set()
            reader = threading.Thread(target=read_log)
            reader.start()
            args = mf.parse_args(['127.0.0.1', '--rekam', '--rekam-detik', '1.5',
                                  '--folder-rekaman', str(Path(directory) / 'rekaman')])
            processor = mf.EchoProcessor()
            processor.start()
            worker = mf.MassFlowAcquisition(args, None, None, processor)
            rp = None
            try:
                self.assertTrue(ready.wait(10), ''.join(log))
                rp = mf.live.NativeTransport('127.0.0.1', timeout=3, port=5001)
                failures = []
                def receive():
                    try:
                        worker.receive(rp, 62.5e6)
                    except Exception as error:
                        failures.append(error)
                receiver = threading.Thread(target=receive)
                receiver.start()
                time.sleep(2.5)
                worker.stop_event.set()
                receiver.join(timeout=10)
                self.assertFalse(failures, failures)
                processor.inbox.put(('stop',))
                processor.join(timeout=30)
                self.assertIsNone(processor.error)
                self.assertEqual(process.wait(timeout=10), 0)
                reader.join(timeout=5)
                self.assertIn(f'DONE frames={worker.frames} ', ''.join(log))
                self.assertIn('SIM_RELEASE output=0', ''.join(log))
                self.assertGreater(worker.frames / 2.5, 250)
                self.assertEqual(processor.processed, worker.frames)

                folder, = (Path(directory) / 'rekaman').iterdir()
                waveforms, metadata = load_recording(folder)
                self.assertTrue(metadata['complete'])
                self.assertAlmostEqual(len(waveforms), 450, delta=15)
                np.testing.assert_array_equal(waveforms[0], np.arange(16384) % 1024)
                rows = (folder / 'tof.csv').read_text().splitlines()
                self.assertEqual(rows[0], mf.TofWriter.HEADER.strip())
                self.assertEqual(len(rows) - 1, len(waveforms))
                self.assertEqual([int(r.split(',')[0]) for r in rows[1:]], list(range(len(waveforms))))
                self.assertTrue((folder / 'acquisition.json').is_file())
                with contextlib.redirect_stdout(io.StringIO()):
                    tof = mf.reprocess(folder, show_plot=False)
                self.assertEqual(len(tof), len(waveforms))
                offline = (folder / 'tof_offline.csv').read_text().splitlines()
                self.assertEqual([r.split(',')[3:] for r in offline[1:]], [r.split(',')[3:] for r in rows[1:]])
            finally:
                worker.stop_event.set()
                if rp is not None:
                    rp.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
                reader.join(timeout=5)
                process.stdout.close()


class AveragingTests(unittest.TestCase):
    def test_running_average_matches_mean_of_last_n_frames(self):
        processor = mf.EchoProcessor(averages=4)
        frames = np.random.default_rng(3).normal(size=(10, mf.IDX_MAX))
        out = np.vstack([processor._running_average(frames[:6]), processor._running_average(frames[6:])])
        for i in range(10):
            np.testing.assert_allclose(out[i], frames[max(0, i - 3):i + 1].mean(axis=0), atol=1e-12)

    def test_averaging_stabilises_tof_in_noisy_frames(self):
        rng = np.random.default_rng(7)
        frames = np.array([synthetic(4.0, 40.0, 0.03, 0.04, int(rng.integers(1 << 30))) for _ in range(96)])
        def spread(averages):
            dsp = mf.EchoDSP(FS)   # baru per konfigurasi: riwayat noise threshold otomatis tidak tercampur
            processor = mf.EchoProcessor(averages=averages)
            volts = processor._running_average(frames) if averages > 1 else frames
            tof = np.array([dsp.detect(row).tof_air_us for row in dsp.envelope(volts[32:])])
            hit = np.isfinite(tof) & (np.abs(tof - 40.0) < 0.5)
            return hit.mean()
        saved = mf.AIR_MULAI_US
        mf.AIR_MULAI_US = 5.0   # lewati ekor envelope akrilik (MA 128 sampel ≈ 2 µs)
        try:
            single, averaged = spread(1), spread(16)
        finally:
            mf.AIR_MULAI_US = saved
        self.assertGreater(averaged, 0.95)
        self.assertGreater(averaged, single + 0.2)

    def test_cli_dsp_overrides(self):
        with dsp_options('--rata', '16', '--median', '30', '--filter', '3.5', '6.5',
                         '--blank-us', '8', '--air-us', '30', '60', '--threshold', '0.1'):
            self.assertEqual((mf.FILTER_LOW, mf.FILTER_HIGH), (3.5e6, 6.5e6))
            self.assertEqual((mf.BLANK_US, mf.AIR_MULAI_US, mf.AIR_SELESAI_US, mf.NOISE_THRESHOLD), (8, 30, 60, 0.1))
            self.assertIsNone(mf.THRESHOLD_NOISE)   # --threshold = mode manual
        self.assertEqual(mf.THRESHOLD_NOISE, 1.2)


@contextlib.contextmanager
def dsp_options(*argv):
    """Terapkan opsi CLI DSP sementara, lalu kembalikan semua konstanta modul."""
    names = ('FILTER_LOW', 'FILTER_HIGH', 'BLANK_US', 'AKRILIK_US', 'AIR_MULAI_US', 'AIR_SELESAI_US',
             'NOISE_THRESHOLD', 'MA_WINDOW', 'THRESHOLD_NOISE', 'THRESHOLD_AKRILIK', 'NOISE_REF_US',
             'PANTULAN_US', 'PANTULAN_R', 'C_AIR', 'SUHU_AIR')
    saved = {n: getattr(mf, n) for n in names}
    try:
        mf.apply_dsp_options(mf.parse_args(list(argv)))
        yield
    finally:
        for n, v in saved.items():
            setattr(mf, n, v)


def plate_signal(amp_reverb, water_us, amp_water, seed, noise=0.004, t_akrilik=4.53, period=3.744):
    """Echo akrilik + pantulan ulang plat (amp_reverb^k) + echo air, pembawa 3,5 MHz."""
    t = np.arange(mf.IDX_MAX) / FS * 1e6
    pulse = lambda t0, amp: amp * np.exp(-0.5 * ((t - t0) / 0.3) ** 2) * np.sin(2 * np.pi * 3.5 * (t - t0))
    x = pulse(t_akrilik, 0.3) + np.random.default_rng(seed).normal(0, noise, t.size)
    for k in range(1, 6):
        x += pulse(t_akrilik + k * period, 0.3 * amp_reverb ** k)
    if water_us is not None:
        x += pulse(t_akrilik + water_us, 0.3 * amp_water)
    return x


class OptionTests(unittest.TestCase):
    def test_preset_fills_unset_options_and_explicit_options_win(self):
        args = mf.parse_args(['--preset', 'akrilik5mm'])
        self.assertEqual((args.ma, args.threshold_noise, args.pantulan_us), (32, 1.2, 3.744))
        args = mf.parse_args(['--preset', 'akrilik5mm', '--ma', '64', '--pantulan-r', '0.3'])
        self.assertEqual((args.ma, args.threshold_noise, args.pantulan_r), (64, 1.2, 0.3))
        with dsp_options('--preset', 'akrilik5mm'):
            self.assertEqual((mf.MA_WINDOW, mf.THRESHOLD_NOISE, mf.PANTULAN_US, mf.PANTULAN_R), (32, 1.2, 3.744, 0.15))
        self.assertEqual((mf.MA_WINDOW, mf.THRESHOLD_NOISE, mf.PANTULAN_US), (32, 1.2, None))   # default baru
        args = mf.parse_args(['--preset', 'akrilik5mm', '--threshold', '0.2'])   # manual menimpa preset
        self.assertEqual((args.threshold, args.threshold_noise, args.pantulan_us), (0.2, None, 3.744))

    def test_invalid_option_combinations_rejected(self):
        for argv in (['--threshold', '0.2', '--threshold-noise', '1.2'], ['--ma', '0'],
                     ['--pantulan-us', '1.0'], ['--pantulan-r', '1.5'], ['--preset', 'kaca'],
                     ['--threshold', '0.2', '--noise-us', '90', '125'], ['--threshold', '0.2', '--threshold-akrilik', '0.8'],
                     ['--threshold-akrilik', '0'], ['--noise-us', '90', '93'], ['--noise-us', '100', '140']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mf.parse_args(argv)

    def test_threshold_noise_is_factor_times_p999_of_noise_peaks(self):
        with dsp_options('--ma', '32', '--threshold-noise', '1.2'):
            dsp = mf.EchoDSP(FS)
            frames = np.array([plate_signal(0.15, 20.0, 0.3, seed) for seed in range(40)])
            env = dsp.envelope(frames)
            # envelope mentah (tanpa threshold/blank) untuk menghitung acuan
            raw = np.abs(hilbert(sosfiltfilt(dsp.sos, frames, axis=-1) * mf.DIGITAL_GAIN, axis=-1))
            raw = np.array([np.convolve(r, np.ones(32) / 32, 'same') for r in raw])
            expected = 1.2 * np.percentile(raw[:, dsp.noise_idx].max(axis=1), 99.9)
            self.assertAlmostEqual(dsp.threshold, expected, places=9)
            self.assertTrue(np.all((env == 0) | (env >= dsp.threshold_akrilik)))
            self.assertAlmostEqual(dsp.threshold_akrilik, mf.THRESHOLD_AKRILIK * dsp.threshold)
            self.assertEqual(len(dsp._noise_peaks), 40)
            for row in env:
                echo = dsp.detect(row)
                self.assertEqual(echo.status, 'OK')
                self.assertAlmostEqual(echo.tof_air_us, 20.0, delta=0.05)

    def test_pantulan_rejects_plate_reverberation_but_keeps_water(self):
        # pantulan pertama (0,25 x akrilik) lebih kuat dari echo air (0,12 x akrilik)
        frames = np.array([plate_signal(0.25, 20.0, 0.12, seed) for seed in range(8)])
        with dsp_options('--ma', '32', '--air-us', '1.0', '120'):
            dsp = mf.EchoDSP(FS)
            tof = [dsp.detect(row).tof_air_us for row in dsp.envelope(frames)]
            np.testing.assert_allclose(tof, 3.744, atol=0.1)      # tanpa opsi: pantulan terpilih
        with dsp_options('--ma', '32', '--air-us', '1.0', '120', '--pantulan-us', '3.744', '--pantulan-r', '0.25'):
            dsp = mf.EchoDSP(FS)
            tof = [dsp.detect(row).tof_air_us for row in dsp.envelope(frames)]
            np.testing.assert_allclose(tof, 20.0, atol=0.05)

    def test_pantulan_keeps_strong_water_echo_on_a_reverberation_multiple(self):
        frames = np.array([plate_signal(0.15, 2 * 3.744, 0.5, seed) for seed in range(4)])
        with dsp_options('--ma', '32', '--air-us', '1.0', '120', '--pantulan-us', '3.744'):
            dsp = mf.EchoDSP(FS)
            for row in dsp.envelope(frames):
                echo = dsp.detect(row)
                self.assertEqual(echo.status, 'OK')
                self.assertAlmostEqual(echo.tof_air_us, 2 * 3.744, delta=0.1)

    def test_manual_mode_is_original_algorithm(self):
        frames = np.array([plate_signal(0.15, 20.0, 0.3, seed) for seed in range(6)])
        with dsp_options('--ma', '128', '--threshold', '0.15'):
            dsp = mf.EchoDSP(FS)
            env = dsp.envelope(frames)
            self.assertEqual(dsp.threshold, mf.NOISE_THRESHOLD)
            self.assertEqual(len(dsp._noise_peaks), 0)
            for x, row in zip(frames, env):
                ref_ak, ref_air = reference_scpi_algorithm(x)
                echo = dsp.detect(row)
                self.assertAlmostEqual(echo.waktu_akrilik_us, ref_ak, delta=2 / FS * 1e6)
                self.assertAlmostEqual(echo.tof_air_us, ref_air, delta=2 / FS * 1e6)


def ringing_only(seed, noise=0.004):
    """Ekor ringing TX yang meluruh melewati tepi blanking, tanpa echo akrilik (transduser di udara)."""
    t = np.arange(mf.IDX_MAX) / FS * 1e6
    ring = 0.5 * np.exp(-t / 0.6) * np.sin(2 * np.pi * 3.5 * t)
    return ring + np.random.default_rng(seed).normal(0, noise, t.size)


class AutoThresholdTests(unittest.TestCase):
    def test_ringing_tail_is_not_taken_as_acrylic(self):
        frames = np.array([ringing_only(seed) for seed in range(300)])   # riwayat noise realistis (program: 600)
        with dsp_options():
            dsp = mf.EchoDSP(FS)
            env = dsp.envelope(frames)
            ia = dsp.idx_akrilik[dsp._akrilik_mulai]
            self.assertTrue(np.all(env[:, ia] == 0))   # ekor ringing di tepi blanking dibuang
            akrilik = np.array([dsp.detect(row).waktu_akrilik_us for row in env])
            palsu = akrilik[np.isfinite(akrilik)]
            # threshold akrilik 0,8 x 1,2 x p99,9 noise: sesekali puncak noise murni (~0,1%) boleh lolos,
            # tapi tidak pernah ekor ringing di tepi blanking
            self.assertLessEqual(palsu.size, 3)
            self.assertTrue(np.all(palsu > 3.0), palsu)
        with dsp_options('--ma', '128', '--threshold', '0.05'):   # algoritma lama (threshold tetap): tertipu ringing
            dsp = mf.EchoDSP(FS)
            for row in dsp.envelope(frames[:20]):
                self.assertAlmostEqual(dsp.detect(row).waktu_akrilik_us, 2.512, delta=0.02)

    def test_acrylic_echo_after_ringing_still_found(self):
        frames = np.array([ringing_only(seed) + plate_signal(0.15, 20.0, 0.3, seed + 50, noise=0) for seed in range(8)])
        with dsp_options():
            dsp = mf.EchoDSP(FS)
            for row in dsp.envelope(frames):
                echo = dsp.detect(row)
                self.assertAlmostEqual(echo.waktu_akrilik_us, 4.53, delta=0.05)
                self.assertAlmostEqual(echo.tof_air_us, 20.0, delta=0.05)

    def test_acrylic_factor_applies_below_air_threshold(self):
        with dsp_options():
            dsp = mf.EchoDSP(FS)
            dsp.envelope(np.array([plate_signal(0.15, None, 0, seed) for seed in range(20)]))
            thr = dsp.threshold
            t = dsp.waktu_us
            bump = lambda t0, amp: amp * np.clip(1 - np.abs(t - t0) / 0.3, 0, None)
            row = bump(4.5, 0.95 * thr) + bump(24.5, 2 * thr)   # akrilik di antara 0,9x dan 1x threshold
            echo = dsp.detect(row)
            self.assertEqual(echo.status, 'OK')
            self.assertAlmostEqual(echo.tof_air_us, 20.0, delta=0.05)
            mf.THRESHOLD_AKRILIK = 1.0
            dsp.threshold_akrilik = thr
            self.assertEqual(dsp.detect(row).status, 'AKRILIK_TIDAK_TERDETEKSI')

    def test_second_round_trip_excluded_from_noise_window(self):
        tof = 52.0                        # bolak-balik ke-2 di 4,53 + 104 = 108,5 µs (dalam 90-125 µs)
        t = np.arange(mf.IDX_MAX) / FS * 1e6
        frames = []
        for seed in range(30):
            x = plate_signal(0.15, tof, 0.3, seed)
            x += 0.03 * np.exp(-0.5 * ((t - 4.53 - 2 * tof) / 0.3) ** 2) * np.sin(2 * np.pi * 3.5 * t)
            frames.append(x)
        frames = np.array(frames)
        with dsp_options():
            plain = mf.EchoDSP(FS)
            plain.envelope(frames)
            excl = mf.EchoDSP(FS)
            excl._terakhir = (4.53, tof)
            excl.envelope(frames)
            self.assertEqual(plain.noise_dikecualikan, 0)
            self.assertGreater(excl.noise_dikecualikan, 0)
            self.assertLess(excl.threshold, plain.threshold / 3)
            # alur normal: deteksi OK mengisi ToF terakhir, batch berikutnya mengecualikan
            dsp = mf.EchoDSP(FS)
            for row in dsp.envelope(frames[:15]):
                dsp.detect(row)
            self.assertAlmostEqual(dsp._terakhir[1], tof, delta=0.05)
            dsp.envelope(frames[15:])
            self.assertGreater(dsp.noise_dikecualikan, 0)

    def test_short_tof_never_blanks_whole_noise_window(self):
        # ToF 3,71 µs (pantulan plat tanpa air): kelipatan tinggi akan menutup seluruh 90-125 µs
        frames = np.array([plate_signal(0.4, None, 0, seed) for seed in range(20)])
        with dsp_options():
            dsp = mf.EchoDSP(FS)
            dsp._terakhir = (4.53, 3.71)
            dsp.envelope(frames)
            self.assertEqual(dsp.noise_dikecualikan, 0)
            dsp._terakhir = (4.53, 50.0)        # 2x -> 104,5 µs: hanya -2..+9,5 µs dari 35 µs dikecualikan
            dsp.envelope(frames)
            self.assertAlmostEqual(dsp.noise_dikecualikan * 16e-3, 11.5, delta=0.1)

    def test_noise_window_option(self):
        with dsp_options('--noise-us', '60', '80'):
            dsp = mf.EchoDSP(FS)
            self.assertAlmostEqual(dsp.waktu_us[dsp.noise_idx[0]], 60.0, delta=0.02)
            self.assertAlmostEqual(dsp.waktu_us[dsp.noise_idx[-1]], 80.0, delta=0.02)


class WaterHeightTests(unittest.TestCase):
    def test_marczak_speed_of_sound(self):
        self.assertAlmostEqual(mf.c_air_marczak(15.0), 1465.96, delta=0.01)
        self.assertAlmostEqual(mf.c_air_marczak(20.0), 1482.38, delta=0.01)
        self.assertAlmostEqual(mf.c_air_marczak(25.0), 1496.73, delta=0.01)
        with self.assertRaises(ValueError):
            mf.c_air_marczak(-1.0)

    def test_c_options_and_height(self):
        self.assertEqual(mf.C_AIR, 1466.0)
        self.assertAlmostEqual(mf.tinggi_air_mm(55.216), 40.473, places=3)   # tes air ~40 mm
        with dsp_options('--c-air', '1480'):
            self.assertEqual((mf.C_AIR, mf.SUHU_AIR), (1480.0, None))
        with dsp_options('--c-air', '1400', '--suhu-air', '20'):            # suhu menimpa --c-air
            self.assertAlmostEqual(mf.C_AIR, 1482.38, delta=0.01)
            self.assertEqual(mf.SUHU_AIR, 20.0)
        self.assertEqual((mf.C_AIR, mf.SUHU_AIR), (1466.0, None))
        for argv in (['--c-air', '900'], ['--suhu-air', '120']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                mf.parse_args(argv)

    def test_tof_csv_has_height_column(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = mf.TofWriter(Path(directory) / 'tof.csv')
            writer.write(0, 0, 0.0, mf.Echo('OK', 4.48, 54.976, 1.2))
            writer.write(1, 1, 0.003, mf.Echo('AIR_TIDAK_TERDETEKSI', 4.48, float('nan'), float('nan')))
            writer.close()
            rows = (Path(directory) / 'tof.csv').read_text().splitlines()
        self.assertTrue(rows[0].endswith(',h_mm'))
        self.assertAlmostEqual(float(rows[1].split(',')[-1]), 1466.0 * 54.976 / 2000, places=5)
        self.assertEqual(rows[2].split(',')[-1], '')


class BacklogTests(unittest.TestCase):
    def test_dsp_backlog_warns_and_recording_stays_lossless(self):
        import struct
        n = 200
        payload = (np.arange(mf.SAMPLES_PER_FRAME) % 100).astype('<i2').tobytes()
        with tempfile.TemporaryDirectory() as folder:
            args = mf.parse_args(['--rekam', '--folder-rekaman', folder])
            processor = mf.EchoProcessor()            # not started: the DSP falls behind completely
            worker = mf.MassFlowAcquisition(args, None, None, processor)

            class Socket:
                def sendall(self, data):
                    pass

            class FakeServer:
                _socket = Socket()

                def __init__(self):
                    frames = [struct.pack('!4I', mf.MAGIC_TX_ON, i, mf.SAMPLES_PER_FRAME, mf.BYTES_PER_FRAME) + payload
                              for i in range(n)]
                    self.data = b''.join(frames) + struct.pack('!4I', mf.MAGIC_END, n, 0, 0)
                    self.pos = 0

                def read_exact(self, size):
                    out = self.data[self.pos:self.pos + size]
                    self.pos += size
                    if self.pos == len(self.data) - 16:   # last payload served: user closes the window
                        worker.stop_event.set()
                    return out

            with contextlib.redirect_stdout(io.StringIO()) as out:
                worker.receive(FakeServer(), 62.5e6)
            while not processor.inbox.empty():        # the idle DSP thread would close tof.csv
                item = processor.inbox.get_nowait()
                if item[0] == 'close':
                    item[1].close()
            self.assertEqual(worker.backlog_events, 1)
            self.assertGreaterEqual(worker.backlog_max, n)
            self.assertIn('DSP tertinggal', out.getvalue())
            recording, = [p for p in Path(folder).iterdir() if p.is_dir()]
            waveforms, metadata = load_recording(recording)
            self.assertTrue(metadata['complete'])
            self.assertEqual(len(waveforms), n)
            waveforms._mmap.close()


class GuiTests(unittest.TestCase):
    def test_gui_renders_latest_frame_trend_and_status(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        processor = mf.EchoProcessor(flow_features=True)
        processor.inbox.put(('config', FS))
        counts = [np.round(synthetic(4.0, 10.0, 0.08, 0.002, i) / mf.VOLT_PER_COUNT).astype(np.int16) for i in range(30)]
        for i, c in enumerate(counts):
            processor.inbox.put(('frame', i, 100 + i / 300, c, None, None, None))
        processor.inbox.put(('stop',))
        processor.run()
        worker = mf.MassFlowAcquisition(mf.parse_args([]), None, None, processor)
        worker.rx_fps, worker.frames = 300.0, 30
        class Timer:
            def add_callback(self, callback): self.callback = callback
            def start(self): self.callback()
            def stop(self): pass
        def show():
            fig = plt.gcf()
            fig.canvas.draw()
            self.assertRegex(fig.axes[1].texts[0].get_text(), r'ToF Air: (9\.9|10\.0)\d µs  \|  h = 7\.[23]\d mm')
            self.assertEqual(len(fig.axes[2].lines[0].get_xdata()), 30)
            self.assertIn('RX  300.0 FPS', fig.texts[0].get_text())
            self.assertIn('EXPERIMENTAL WARMUP', fig.texts[0].get_text())
        try:
            with patch.object(mf.shutil, 'which', return_value=sys.executable), \
                 patch.object(mf.getpass, 'getpass', return_value='x'), \
                 patch.object(mf, 'EchoProcessor', return_value=processor), \
                 patch.object(mf, 'MassFlowAcquisition', return_value=worker), \
                 patch.object(processor, 'start'), patch.object(processor, 'join'), \
                 patch.object(worker, 'start'), patch.object(worker, 'join'), \
                 patch.object(worker, 'is_alive', return_value=False), \
                 patch.object(FigureCanvasAgg, 'new_timer', return_value=Timer()), \
                 patch.object(plt, 'show', side_effect=show), \
                 contextlib.redirect_stdout(io.StringIO()):
                mf.main(['--host-key', 'SHA256:test-only-mocked', '--flow-features'])
            self.assertTrue(worker.stop_event.is_set())
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
