"""Recording integrity tests using synthetic TCP frames, not hardware FPS tests."""
import json
import contextlib
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tests import test_ultrasound

ROOT = Path(__file__).resolve().parents[1]
from ultrasonic_film.recording import FrameRecorder, load_recording


class RecordingTests(unittest.TestCase):
    run_receiver = test_ultrasound.ClientTests.run_receiver

    def test_all_300_frames_round_trip_and_offline_plot(self):
        wire = bytearray()
        for sequence in range(300):
            data = np.arange(16384, dtype='<i2')
            data[0] = sequence
            wire.extend(struct.pack('!4I', 0x52505831, sequence, 16384, 32768))
            wire.extend(data.tobytes())
        wire.extend(struct.pack('!4I', 0x52504531, 300, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run'
            output = Path(directory) / 'result.json'
            result = self.run_receiver(wire, '--record-dir', str(run), '--output', str(output),
                                       '--sample-rate', '62500000', '--trigger', 'AWG_PE')
            self.assertEqual(result.returncode, 0, result.stderr)
            saved, metadata = load_recording(run)
            self.assertEqual(saved.shape, (300, 16384))
            np.testing.assert_array_equal(saved[:, 0], np.arange(300))
            for waveform in saved:
                np.testing.assert_array_equal(waveform[1:], np.arange(1, 16384))
            self.assertEqual(metadata['raw_bytes'], 300 * 32768)
            self.assertEqual(metadata['trigger'], 'AWG_PE')
            saved._mmap.close()  # Release the Windows file mapping before temporary-folder cleanup.
            times = np.loadtxt(run/'frame_times.csv', delimiter=',', skiprows=1)
            np.testing.assert_array_equal(times[:, 0], np.arange(300))
            self.assertTrue(np.all(np.diff(times[:, 1]) >= 0))
            stats = json.loads(output.read_text())
            self.assertEqual(stats['frames_saved'], stats['frames'])
            self.assertTrue(stats['recording_complete'])
            image = Path(directory)/'offline.png'
            plotted = subprocess.run([sys.executable, '-m', 'ultrasonic_film.plot_recording', str(run),
                                      '--frame', '299', '--output', str(image)],
                                     env={**os.environ, 'MPLBACKEND': 'Agg'},
                                     capture_output=True, text=True, timeout=20)
            self.assertEqual(plotted.returncode, 0, plotted.stderr)
            self.assertEqual(image.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_disconnect_keeps_previous_frame_but_marks_run_incomplete(self):
        wire = struct.pack('!4I', 0x52505831, 0, 16384, 32768) + bytes(32768)
        wire += struct.pack('!4I', 0x52505831, 1, 16384, 32768) + bytes(20)
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_receiver(wire, '--record-dir', directory)
            self.assertNotEqual(result.returncode, 0)
            metadata = json.loads((Path(directory)/'recording.json').read_text())
            self.assertFalse(metadata['complete'])
            self.assertEqual(metadata['frames_saved'], 1)
            self.assertEqual((Path(directory)/'frames.bin').stat().st_size, 32768)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                load_recording(directory)

    def test_existing_recording_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with FrameRecorder(directory, sample_rate_hz=62500000, trigger='AWG_PE', host='test') as recorder:
                recorder.append(0, bytes(32768), 0.1)
            original = (Path(directory)/'recording.json').read_bytes()
            with self.assertRaises(FileExistsError):
                FrameRecorder(directory, sample_rate_hz=62500000, trigger='AWG_PE', host='test')
            self.assertEqual((Path(directory)/'recording.json').read_bytes(), original)

    def test_disk_error_is_reported_as_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(OSError, 'disk full'):
                with FrameRecorder(directory, sample_rate_hz=62500000, trigger='AWG_PE', host='test') as recorder:
                    recorder.raw = Mock(wraps=recorder.raw)
                    recorder.raw.write.side_effect = OSError('disk full')
                    recorder.append(0, bytes(32768), 0.1)
            metadata = json.loads((Path(directory)/'recording.json').read_text())
            self.assertFalse(metadata['complete'])
            self.assertEqual(metadata['frames_saved'], 0)
            self.assertIn('disk full', metadata['error'])

    def test_loader_rejects_file_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            with FrameRecorder(directory, sample_rate_hz=62500000, trigger='AWG_PE', host='test') as recorder:
                recorder.append(0, bytes(32768), 0.1)
            with (Path(directory)/'frames.bin').open('r+b') as raw:
                raw.truncate(10)
            with self.assertRaisesRegex(ValueError, 'size'):
                load_recording(directory)

    def test_flush_error_cannot_mark_buffered_data_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(OSError, 'flush failed'):
                with FrameRecorder(directory, sample_rate_hz=62500000, trigger='AWG_PE', host='test') as recorder:
                    recorder.append(0, bytes(32768), 0.1)
                    close_files = recorder.files.close
                    def fail_after_close():
                        close_files()
                        raise OSError('flush failed')
                    recorder.files = Mock(wraps=recorder.files)
                    recorder.files.close.side_effect = fail_after_close
            metadata = json.loads((Path(directory)/'recording.json').read_text())
            self.assertFalse(metadata['complete'])
            self.assertIn('flush failed', metadata['error'])


class LauncherRecordingTests(unittest.TestCase):
    def test_each_run_gets_unique_folder_and_default_outputs_follow_it(self):
        launcher = test_ultrasound.launcher
        class Process:
            def __init__(self):
                self.stdout = io.StringIO('READY sample_rate=62500000 trigger=AWG_PE\nDONE\n')
            def wait(self, **kwargs): return 0
            def poll(self): return 0

        with tempfile.TemporaryDirectory() as directory:
            commands = []
            def run(command, **kwargs):
                if 'ultrasonic_film.native_client' in command:
                    commands.append(command)
                    Path(command[command.index('--output')+1]).write_text(json.dumps({'frames': 1}))
                return subprocess.CompletedProcess(command, 0)
            for _ in range(2):
                with patch.object(sys, 'argv', ['film-benchmark', '--host-key', 'SHA256:test-placeholder', '--ultrasound',
                                               '--save-all', '--recordings-dir', directory]), \
                     patch.object(launcher.shutil, 'which', return_value=sys.executable), \
                     patch.object(launcher.getpass, 'getpass', return_value='test-placeholder'), \
                     patch.object(launcher.subprocess, 'run', side_effect=run), \
                     patch.object(launcher.subprocess, 'Popen', side_effect=lambda *a, **kw: Process()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    launcher.main()
            folders = []
            for command in commands:
                folder = Path(command[command.index('--record-dir')+1])
                folders.append(folder)
                self.assertEqual(folder.parent, Path(directory).resolve())
                for flag in ('--output', '--plot-output', '--waveform-output'):
                    self.assertEqual(Path(command[command.index(flag)+1]).parent, folder)
                parameters = json.loads((folder/'acquisition.json').read_text())
                self.assertEqual(parameters['target_fps'], 300)
                self.assertEqual(parameters['decimation'], 2)
                self.assertEqual(json.loads((folder/'rp_benchmark_result.json').read_text())['frames'], 1)
            self.assertNotEqual(*folders)



if __name__ == '__main__':
    unittest.main()
