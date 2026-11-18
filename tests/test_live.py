"""Live DMA protocol and noise-analysis checks; no real hardware timing claim."""
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

from ultrasonic_film import live

ROOT = Path(__file__).resolve().parents[1]


class AnalysisTests(unittest.TestCase):
    def make_analysis(self):
        return live.NoiseAnalysis(62.5e6, (10, 200), 20, 3e6, 4.8e6)

    def test_spectrum_locates_tone_and_counts_clipping(self):
        raw = (1000*np.sin(2*np.pi*4e6*np.arange(16384)/62.5e6)).astype(np.int16)
        raw[:2] = [-8192, 8191]
        stats = self.make_analysis().update(raw, 1)
        peak = stats['frequency_mhz'][np.argmax(stats['psd_db'])]
        self.assertAlmostEqual(peak, 4, delta=0.02)
        self.assertEqual(stats['clipped'], 2)
        self.assertTrue(np.isfinite(stats['envelope']).all())

    def test_average_reduces_uncorrelated_noise_and_resets_on_mode_change(self):
        rng = np.random.default_rng(123)
        analysis = self.make_analysis()
        for _ in range(20):
            stats = analysis.update(rng.normal(0, 100, 16384).astype(np.int16), 1)
        self.assertEqual(stats['n'], 20)
        self.assertLess(stats['mean_rms'], stats['rms']*0.3)
        stats = analysis.update(np.zeros(16384, dtype=np.int16), 2)
        self.assertEqual(stats['n'], 1)
        self.assertEqual(stats['mean_rms'], 0)

    def test_cli_defaults_and_invalid_values(self):
        args = live.parse_args([])
        self.assertEqual(args.fps, 300)
        self.assertEqual(args.tx, 'off')
        for options in (['--fps', 'nan'], ['--fps', '0'], ['--amplitude', '2'],
                        ['--gate-us', '10', '1000'], ['--averages', '0']):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    live.parse_args(options)

    def test_connection_settings_are_portable_and_can_use_environment(self):
        with patch.dict(os.environ, {}, clear=True):
            args = live.parse_args([])
            self.assertEqual(args.host, 'redpitaya.local')
            self.assertIsNone(args.host_key)
        with patch.dict(os.environ, {'RP_HOST': 'lab-rp.local', 'RP_HOST_KEY': 'test-fingerprint'}):
            args = live.parse_args([])
            self.assertEqual(args.host, 'lab-rp.local')
            self.assertEqual(args.host_key, 'test-fingerprint')
            self.assertEqual(live.parse_args(['override.local']).host, 'override.local')

    def test_hardware_entrypoint_requires_host_key_before_password_prompt(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, 'argv', ['film-diagnostics']), \
             patch.object(live.getpass, 'getpass') as password, \
             patch.object(live.subprocess, 'run') as remote:
            with self.assertRaisesRegex(SystemExit, 'RP_HOST_KEY'):
                live.main()
            password.assert_not_called()
            remote.assert_not_called()

    def test_default_bandpass_preserves_five_mhz_carrier(self):
        args = live.parse_args([])
        self.assertLess(args.filter_low, 5e6)
        self.assertGreater(args.filter_high, 5e6)
        analysis = live.NoiseAnalysis(62.5e6, (10, 200), 20, args.filter_low, args.filter_high)
        raw = (1000*np.sin(2*np.pi*5e6*np.arange(16384)/62.5e6)).astype(np.int16)
        stats = analysis.update(raw, 1)
        mask = analysis.mask
        self.assertGreater(np.std(stats['filtered'][mask]), 0.9*np.std(raw[mask]))

    def test_averaging_improves_weak_echo_without_shifting_peak(self):
        fs = 62.5e6
        time_us = np.arange(16384)/fs*1e6
        echo = 100*np.exp(-0.5*((time_us-15)/0.5)**2)*np.cos(2*np.pi*5*time_us)
        truth = live.NoiseAnalysis(fs, (10, 200), 20, 3.5e6, 6.5e6).update(echo, 1)
        rng = np.random.default_rng(123)
        analysis = live.NoiseAnalysis(fs, (10, 200), 20, 3.5e6, 6.5e6)
        mask = (time_us > 5) & (time_us < 60)
        for index in range(20):
            result = analysis.update(echo+rng.normal(0, 90, 16384), 1)
            error = np.std(result['filtered'][mask]-truth['filtered'][mask])
            if index == 0:
                initial_error = error
        self.assertLess(error, 0.3*initial_error)
        echo_gate = (time_us > 12) & (time_us < 18)
        peak_time = time_us[echo_gate][np.argmax(result['envelope'][echo_gate])]
        self.assertAlmostEqual(peak_time, 15, delta=0.25)
        self.assertTrue(np.all(result['envelope'][time_us < 2.5] == 0))

    def test_gui_renders_all_panels_using_device_sample_rate(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        worker = live.LiveAcquisition(live.parse_args([]))
        worker.sample_rate = 31.25e6
        worker.rx_fps = 300
        worker.latest = (1, 1, False, np.arange(16384, dtype=np.int16) % 100, time.perf_counter())
        class Timer:
            def add_callback(self, callback): self.callback = callback
            def start(self): self.callback()
            def stop(self): pass
        def show():
            figure = plt.gcf()
            figure.canvas.draw()
            self.assertEqual(len(figure.axes), 6)
            self.assertAlmostEqual(figure.axes[0].lines[0].get_xdata()[-1], 16383/31.25e6*1e6)
            self.assertGreater(len(figure.axes[3].lines[0].get_ydata()), 100)
            self.assertIn('RX 300.0 FPS', figure.texts[0].get_text())
        try:
            with patch.object(sys, 'argv', ['film-diagnostics', '--host-key', 'test-fingerprint']), \
                 patch.object(live.shutil, 'which', return_value=sys.executable), \
                 patch.object(live.getpass, 'getpass', return_value='test-placeholder'), \
                 patch.object(live, 'LiveAcquisition', return_value=worker), \
                 patch.object(worker, 'start'), patch.object(worker, 'join'), \
                 patch.object(worker, 'is_alive', return_value=False), \
                 patch.object(FigureCanvasAgg, 'new_timer', return_value=Timer()), \
                 patch.object(plt, 'show', side_effect=show), \
                 contextlib.redirect_stdout(io.StringIO()):
                live.main()
            self.assertTrue(worker.stop_event.is_set())
        finally:
            plt.close('all')


@unittest.skipUnless(shutil.which('g++') and sys.platform == 'linux', 'native simulator requires Linux g++')
class NativeLiveTests(unittest.TestCase):
    def test_live_stream_switches_tx_and_quits_with_output_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory)/'live')
            subprocess.run(['g++', '-std=c++20', '-Wall', '-Wextra', '-Werror',
                            '-I'+str(ROOT/'tests/fake_api'),
                            str(ROOT/'ultrasonic_film/native/rp_acquisition.cpp'),
                            '-pthread', '-o', executable], check=True, capture_output=True)
            process = subprocess.Popen([executable, '0', 'direct', '300', 'ultrasound',
                                        '5000000', '1', '1', '2', 'unchanged', 'live'],
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            log, failures = [], []
            ready = threading.Event()
            def read_log():
                for line in process.stdout:
                    log.append(line)
                    if line.startswith('READY '): ready.set()
            reader = threading.Thread(target=read_log)
            reader.start()
            worker = live.LiveAcquisition(live.parse_args([]))
            receiver = None
            rp = None
            def until(predicate):
                deadline = time.monotonic()+5
                while time.monotonic() < deadline:
                    if failures:
                        raise failures[0]
                    if predicate(): return
                    threading.Event().wait(0.005)
                self.fail('Timed out waiting for live frame state: '+''.join(log))
            try:
                self.assertTrue(ready.wait(10), ''.join(log))
                rp = live.NativeTransport('127.0.0.1', timeout=3, port=5001)
                def receive():
                    try:
                        worker.receive(rp, 62500000)
                    except Exception as error:
                        failures.append(error)
                receiver = threading.Thread(target=receive)
                receiver.start()
                until(lambda: worker.frames >= 5 and worker.snapshot()[0] is not None)
                self.assertFalse(worker.snapshot()[0][2])
                np.testing.assert_array_equal(worker.snapshot()[0][3], np.zeros(16384))
                first_epoch = worker.snapshot()[0][1]
                worker.set_tx(True)
                until(lambda: worker.snapshot()[0] is not None and worker.snapshot()[0][2])
                packet = worker.snapshot()[0]
                self.assertGreater(packet[1], first_epoch)
                np.testing.assert_array_equal(packet[3], np.arange(16384) % 1024)
                worker.set_tx(False)
                until(lambda: worker.snapshot()[0] is not None and not worker.snapshot()[0][2])
                np.testing.assert_array_equal(worker.snapshot()[0][3], np.zeros(16384))
                worker.stop_event.set()
                receiver.join(timeout=10)
                self.assertFalse(receiver.is_alive())
                self.assertFalse(failures, failures)
                self.assertEqual(process.wait(timeout=10), 0)
                reader.join(timeout=5)
                self.assertIn('SIM_RELEASE output=0', ''.join(log))
                self.assertIn(f'DONE frames={worker.frames} ', ''.join(log))
            finally:
                worker.stop_event.set()
                if rp is not None: rp.close()
                if process.poll() is None:
                    process.kill()
                    process.wait()
                if receiver: receiver.join(timeout=5)
                reader.join(timeout=5)
                process.stdout.close()


if __name__ == '__main__':
    unittest.main()
