"""Offline integration tests. These do not establish hardware timing or FPS."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from ultrasonic_film import benchmark as launcher

ROOT = Path(__file__).resolve().parents[1]


class ClientTests(unittest.TestCase):
    def run_receiver(self, wire, *options):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(15)
        port = listener.getsockname()[1]
        failures = []

        def serve():
            try:
                with listener:
                    connection, _ = listener.accept()
                    with connection:
                        self.assertEqual(connection.recv(1), b'G')
                        for offset in range(0, len(wire), 113):
                            connection.sendall(wire[offset:offset + 113])
            except Exception as exc:
                failures.append(exc)

        worker = threading.Thread(target=serve)
        worker.start()
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'ultrasonic_film.native_client', '127.0.0.1',
                 '--port', str(port), *options], capture_output=True, text=True,
                timeout=30, env={**os.environ, 'MPLBACKEND': 'Agg'})
        finally:
            worker.join(timeout=16)
            listener.close()
        self.assertFalse(worker.is_alive())
        self.assertFalse(failures, failures)
        return result

    def test_ultrasound_frame_plot_and_saved_samples(self):
        sample_rate = 62.5e6
        samples = (1500 * np.sin(2 * np.pi * 4e6 * np.arange(16384) / sample_rate)).astype('<i2')
        wire = struct.pack('!4I', 0x52505831, 0, 16384, 32768) + samples.tobytes()
        wire += struct.pack('!4I', 0x52504531, 1, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            image = Path(directory) / 'waveform.png'
            waveform = Path(directory) / 'waveform.npz'
            result = self.run_receiver(
                wire, '--output', str(output), '--plot-frame', '0',
                '--plot-output', str(image), '--waveform-output', str(waveform),
                '--sample-rate', str(sample_rate), '--trigger', 'AWG_PE', '--ultrasound')
            self.assertEqual(result.returncode, 0, result.stderr)
            stats = json.loads(output.read_text())
            self.assertEqual(stats['frames'], 1)
            self.assertEqual(stats['trigger'], 'AWG_PE')
            self.assertEqual(stats['sample_rate_hz'], sample_rate)
            self.assertTrue(stats['complete'])
            self.assertEqual(image.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            with np.load(waveform) as saved:
                np.testing.assert_array_equal(saved['raw'], samples)
                self.assertAlmostEqual(saved['time_us'][1], 0.016)
                self.assertEqual(str(saved['trigger']), 'AWG_PE')

    def test_invalid_sequence_is_rejected(self):
        result = self.run_receiver(struct.pack('!4I', 0x52504531, 3, 0, 0))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Sequence mismatch', result.stderr)

    def test_original_benchmark_mode_still_receives_frames(self):
        wire = struct.pack('!4I', 0x52504631, 0, 16384, 32768) + bytes(32768)
        wire += struct.pack('!4I', 0x52504531, 1, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            result = self.run_receiver(wire, '--output', str(output))
            self.assertEqual(result.returncode, 0, result.stderr)
            stats = json.loads(output.read_text())
            self.assertEqual(stats['backend'], 'native-standard')
            self.assertEqual(stats['trigger'], 'NOW')
            self.assertEqual(stats['sample_min'], 0)

    def test_truncated_payload_is_rejected(self):
        wire = struct.pack('!4I', 0x52505831, 0, 16384, 32768) + b'\0' * 100
        result = self.run_receiver(wire)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ConnectionError', result.stderr)

    def test_zero_frame_run_is_rejected(self):
        result = self.run_receiver(struct.pack('!4I', 0x52504531, 0, 0, 0))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('No complete acquisition frames', result.stderr)


class LauncherTests(unittest.TestCase):
    def test_hardware_entrypoint_requires_host_key_before_password_prompt(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(sys, 'argv', ['film-benchmark']), \
             patch.object(launcher.getpass, 'getpass') as password, \
             patch.object(launcher.subprocess, 'run') as remote, \
             contextlib.redirect_stderr(io.StringIO()) as error:
            with self.assertRaises(SystemExit) as exc:
                launcher.main()
            self.assertEqual(exc.exception.code, 2)
            self.assertIn('RP_HOST_KEY', error.getvalue())
            password.assert_not_called()
            remote.assert_not_called()

    def test_invalid_ultrasound_options_fail_before_ssh(self):
        for options in (['--fps', '0'], ['--mode', 'standard'], ['--amplitude', 'nan'],
                        ['--frequency', '0'], ['--cycles', '0'], ['--blank-us', '-1'],
                        ['--filter-low', '5000000', '--filter-high', '3000000']):
            with self.subTest(options=options), patch.object(sys, 'argv', [
                    'film-benchmark', '--ultrasound', *options]):
                with patch.object(launcher.getpass, 'getpass') as password:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                        launcher.main()
                    self.assertEqual(exc.exception.code, 2)
                    password.assert_not_called()

    def test_ultrasound_defaults_and_device_rate_are_forwarded(self):
        class Process:
            stdout = io.StringIO('READY sample_rate=62500000 trigger=AWG_PE\nDONE\n')
            def wait(self, **kwargs): return 0
            def poll(self): return 0

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.json'
            commands = []
            def run(command, **kwargs):
                commands.append(command)
                if command[1:3] == ['-m', 'ultrasonic_film.native_client']:
                    output.write_text(json.dumps({'frames': 1500, 'trigger': 'AWG_PE'}))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(sys, 'argv', ['film-benchmark', '--host-key', 'test-fingerprint', '--ultrasound', '--output', str(output)]), \
                 patch.object(launcher.shutil, 'which', return_value=sys.executable), \
                 patch.object(launcher.getpass, 'getpass', return_value='test-placeholder'), \
                 patch.object(launcher.subprocess, 'run', side_effect=run), \
                 patch.object(launcher.subprocess, 'Popen', return_value=Process()) as remote, \
                 contextlib.redirect_stdout(io.StringIO()):
                launcher.main()
            command = remote.call_args.args[0][-1]
            self.assertIn('direct 300 ultrasound 5000000.0 1 1.0 2 unchanged', command)
            client = commands[-1]
            self.assertEqual(client[client.index('--sample-rate') + 1], '62500000.0')
            self.assertEqual(client[client.index('--trigger') + 1], 'AWG_PE')
            self.assertIn('--ultrasound', client)
            self.assertIn('--waveform-output', client)
            stats = json.loads(output.read_text())
            self.assertEqual(stats['target_fps'], 300)
            self.assertEqual(stats['decimation'], 2)
            self.assertEqual(stats['extra_warmup_shots'], 5)


@unittest.skipUnless(shutil.which('g++') and sys.platform == 'linux', 'native simulator requires Linux g++')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.executable = str(Path(cls.directory.name) / 'benchmark')
        subprocess.run(['g++', '-std=c++20', '-Wall', '-Wextra', '-Werror',
                        '-I' + str(ROOT / 'tests/fake_api'), str(ROOT / 'ultrasonic_film/native/rp_acquisition.cpp'),
                        '-pthread', '-o', cls.executable], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def command(self):
        return [self.executable, '0.1', 'direct', '100', 'ultrasound', '5000000', '1', '1', '2']

    def test_trigger_failure_disables_generator(self):
        result = subprocess.run(self.command(), capture_output=True, text=True, timeout=10,
                                env={**os.environ, 'RP_FAKE_FAIL_TRIGGER': '1'})
        self.assertEqual(result.returncode, 1)
        self.assertIn('SIM_RELEASE output=0', result.stderr)

    def test_capture_longer_than_frame_period_is_rejected_before_shots(self):
        command = self.command()
        command[3] = '1000000'
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn('capture window must fit frame period', result.stderr)
        self.assertIn('SIM_RELEASE output=0 shots=0', result.stderr)

    def test_termination_while_waiting_for_receiver_disables_generator(self):
        process = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = []
        ready = threading.Event()
        def reader():
            for line in process.stdout:
                log.append(line)
                if line.startswith('READY '): ready.set()
        worker = threading.Thread(target=reader)
        worker.start()
        try:
            self.assertTrue(ready.wait(10), ''.join(log))
            process.terminate()
            self.assertNotEqual(process.wait(timeout=10), 0)
            worker.join(timeout=10)
            self.assertIn('SIM_RELEASE output=0 shots=5', ''.join(log))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            worker.join(timeout=10)
            process.stdout.close()

    def test_synchronized_frames_and_normal_cleanup(self):
        # Real TCP and native executable, simulated ADC/AWG. Port matches production.
        process = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = []
        ready = threading.Event()
        def reader():
            for line in process.stdout:
                log.append(line)
                if line.startswith('READY '): ready.set()
        thread = threading.Thread(target=reader)
        thread.start()
        try:
            self.assertTrue(ready.wait(10), ''.join(log))
            with socket.create_connection(('127.0.0.1', 5001), timeout=5) as client:
                def exact(size):
                    data = bytearray()
                    while len(data) < size:
                        part = client.recv(size-len(data))
                        self.assertTrue(part)
                        data.extend(part)
                    return data
                client.sendall(b'G')
                count = 0
                while True:
                    magic, sequence, samples, length = struct.unpack('!4I', exact(16))
                    self.assertEqual(sequence, count)
                    if magic == 0x52504531:
                        self.assertEqual((samples, length), (0, 0))
                        break
                    self.assertEqual((magic, samples, length), (0x52505831, 16384, 32768))
                    data = np.frombuffer(exact(length), dtype='<i2')
                    np.testing.assert_array_equal(data, np.arange(16384) % 1024)
                    count += 1
                self.assertGreater(count, 0)
            self.assertEqual(process.wait(timeout=10), 0)
            thread.join(timeout=10)
            self.assertIn(f'SIM_RELEASE output=0 shots={count+5}', ''.join(log))
            self.assertIn('sample_rate=62500000 trigger=AWG_PE', ''.join(log))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            thread.join(timeout=10)
            process.stdout.close()


if __name__ == '__main__':
    unittest.main()
