"""Driver Siglent diuji terhadap server SCPI tiruan (bukan osiloskop asli)."""
import socket
import struct
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from ultrasonic_film import scope as ss

TDIV, DELAY, INTERVAL, VDIV, OFFSET, CODE_PER_DIV, POINTS = 10e-6, 50e-6, 5e-9, 0.5, 0.1, 7680.0, 20000


def preamble(tdiv=TDIV, delay=DELAY):
    desc = bytearray(346)
    struct.pack_into('<f', desc, 0x9C, VDIV)
    struct.pack_into('<f', desc, 0xA0, OFFSET)
    struct.pack_into('<f', desc, 0xA4, CODE_PER_DIV)
    struct.pack_into('<h', desc, 0xAC, 12)
    struct.pack_into('<f', desc, 0xB0, INTERVAL)
    struct.pack_into('<d', desc, 0xB4, delay)
    struct.pack_into('<h', desc, 0x144, ss.TDIV_ENUM.index(tdiv))
    struct.pack_into('<f', desc, 0x148, 1.0)
    return bytes(desc)


def codes(channel):
    n = np.arange(POINTS)
    return (np.round(1000 * np.sin(n / 50)) + (0 if channel == 'C1' else 2000)).astype('<i2')


def block(data, terminator=b'\n'):
    return b'#9' + f'{len(data):09d}'.encode() + data + terminator


class FakeSiglent(threading.Thread):
    def __init__(self, maxpoint=8000, delay_sign=1):
        super().__init__(daemon=True)
        self.delay_sign = delay_sign   # firmware: preamble delay = sign * :TIMebase:DELay
        self.state = {'tdiv': TDIV, 'delay': DELAY * delay_sign, 'mode': 'AUTO', 'source': 'C2',
                      'slope': 'FALLing', 'level': 0.0, 'run': False}
        self.server = socket.socket()
        self.server.bind(('127.0.0.1', 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        self.maxpoint = maxpoint
        self.armed_empty = False   # True: belum ada akuisisi -> preamble berisi nol (perilaku alat asli)

    def run(self):
        conn, _ = self.server.accept()
        self.server.close()
        source, start, count, step, buffer = 'C1', 0, POINTS, 1, b''
        with conn:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buffer += chunk
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    cmd = line.decode().strip()
                    reply = None
                    if cmd == '*IDN?':
                        reply = b'Siglent Technologies,SDS804X HD,SIMULATED,4.8.12\n'
                    elif cmd.startswith(':CHANnel') and cmd.endswith('SWITch?'):
                        reply = b'ON\n' if cmd[8] in '12' else b'OFF\n'
                    elif cmd == ':TIMebase:SCALe?':
                        reply = f'{self.state["tdiv"]:.2E}\n'.encode()
                    elif cmd == ':TIMebase:DELay?':
                        reply = f'{self.state["delay"]:.2E}\n'.encode()
                    elif cmd.startswith(':TIMebase:SCALe '):
                        self.state['tdiv'] = float(cmd.split()[1])
                    elif cmd.startswith(':TIMebase:DELay '):
                        self.state['delay'] = float(cmd.split()[1])
                    elif cmd.startswith(':TRIGger:EDGE:SOURce '):
                        self.state['source'] = cmd.split()[1]
                    elif cmd.startswith(':TRIGger:EDGE:SLOPe '):
                        self.state['slope'] = cmd.split()[1]
                    elif cmd.startswith(':TRIGger:EDGE:LEVel '):
                        self.state['level'] = float(cmd.split()[1])
                    elif cmd.startswith(':TRIGger:MODE '):
                        self.state['mode'] = cmd.split()[1]
                    elif cmd == ':TRIGger:RUN':
                        self.state['run'] = True
                    elif cmd == ':TRIGger:EDGE:SOURce?':
                        reply = f'{self.state["source"]}\n'.encode()
                    elif cmd == ':TRIGger:EDGE:SLOPe?':
                        reply = f'{self.state["slope"]}\n'.encode()
                    elif cmd == ':TRIGger:EDGE:LEVel?':
                        reply = f'{self.state["level"]:.2E}\n'.encode()
                    elif cmd == ':TRIGger:MODE?':
                        reply = f'{self.state["mode"]}\n'.encode()
                    elif cmd == ':TRIGger:STATus?':
                        reply = b"Trig'd\n"
                    elif cmd == '*OPC?':
                        reply = b'1\n'
                    elif cmd == ':ACQuire:SRATe?':
                        reply = b'2.00E+08\n'
                    elif cmd == ':ACQuire:POINts?':
                        reply = f'{POINTS:.2E}\n'.encode()
                    elif cmd == ':WAVeform:MAXPoint?':
                        reply = f'{self.maxpoint}\n'.encode()
                    elif cmd.startswith(':WAVeform:SOURce '):
                        source = cmd.split()[1]
                    elif cmd.startswith(':WAVeform:STARt '):
                        start = int(cmd.split()[1])
                    elif cmd.startswith(':WAVeform:POINt '):
                        count = int(cmd.split()[1])
                    elif cmd.startswith(':WAVeform:INTerval '):
                        step = int(cmd.split()[1])
                    elif cmd == ':WAVeform:PREamble?':
                        if self.armed_empty:
                            empty = bytearray(346)
                            struct.pack_into('<f', empty, 0xB0, float('inf'))
                            reply = block(bytes(empty))
                        else:
                            reply = block(preamble(self.state['tdiv'], self.state['delay'] * self.delay_sign))
                    elif cmd == ':WAVeform:DATA?':
                        # SDS804X HD fw 4.8.12 mengakhiri blok DATA dengan '\n\n' (terukur di alat asli).
                        data = codes(source)[start:start + count * step:step]
                        reply = block(data.tobytes(), b'\n\n')
                    if reply:
                        # Kirim terpecah untuk menguji pembacaan parsial.
                        for i in range(0, len(reply), 1500):
                            conn.sendall(reply[i:i + 1500])


class SiglentTests(unittest.TestCase):
    def test_setup_aligns_window_for_either_delay_sign_convention(self):
        for sign in (1, -1):
            with self.subTest(delay_sign=sign):
                fake = FakeSiglent(delay_sign=sign)
                fake.start()
                scope = ss.SiglentScope('127.0.0.1', fake.port)
                try:
                    result = scope.setup_for_redpitaya('C1', 0.2, 10e-6)
                    self.assertAlmostEqual(result['window_us'][0], 0.0, delta=0.01)
                    self.assertAlmostEqual(result['window_us'][1], (POINTS - 1) * INTERVAL * 1e6, delta=0.01)
                    self.assertEqual(fake.state['source'], 'C1')
                    self.assertEqual(fake.state['slope'], 'RISing')
                    self.assertEqual(fake.state['mode'], 'NORMal')
                    self.assertAlmostEqual(fake.state['level'], 0.2)
                    self.assertTrue(fake.state['run'])
                finally:
                    scope.close()

    def test_waveform_conversion_and_chunked_read(self):
        fake = FakeSiglent()
        fake.start()
        scope = ss.SiglentScope('127.0.0.1', fake.port)
        try:
            self.assertIn('SDS804X HD', scope.idn())
            self.assertTrue(scope.channel_on('C2'))
            self.assertFalse(scope.channel_on('C3'))
            settings = scope.settings()
            self.assertEqual(settings['points'], POINTS)
            for ch in ('C1', 'C2'):
                t, v, pre = scope.waveform(ch)
                self.assertEqual(v.size, POINTS)
                expected = codes(ch) / CODE_PER_DIV * VDIV - OFFSET
                np.testing.assert_allclose(v, expected, atol=1e-7)
                self.assertAlmostEqual(t[0], -TDIV * 5 + DELAY)
                self.assertAlmostEqual(t[1] - t[0], INTERVAL, places=12)
                self.assertEqual(pre['adc_bit'], 12)
            self.assertEqual(scope.idn().split(',')[1], 'SDS804X HD')  # stream tetap sinkron
        finally:
            scope.close()

    def test_reader_thread_publishes_both_channels_in_microseconds(self):
        fake = FakeSiglent()
        fake.start()
        original = ss.SiglentScope.__init__
        def connect(self, host, port=5025, timeout=3.0):
            original(self, host, fake.port, timeout)
        ss.SiglentScope.__init__ = connect
        reader = ss.ScopeReader('127.0.0.1', ('C1', 'C2', 'C3'), rate_hz=20)
        try:
            reader.start()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not reader.snapshot()[0]:
                time.sleep(0.02)
            waves, info, error, _ = reader.snapshot()
            self.assertIsNone(error)
            self.assertEqual(info, "SDS804X HD [trigger Trig'd]")
            self.assertEqual(sorted(waves), ['C1', 'C2'])
            self.assertAlmostEqual(waves['C1'][0][0], (-TDIV * 5 + DELAY) * 1e6)
        finally:
            reader.stop_event.set()
            reader.join(timeout=5)
            ss.SiglentScope.__init__ = original

    def test_max_points_decimates_instead_of_truncating_window(self):
        fake = FakeSiglent()
        fake.start()
        scope = ss.SiglentScope('127.0.0.1', fake.port)
        try:
            t, v, _ = scope.waveform('C1', max_points=POINTS // 10)
            self.assertEqual(v.size, POINTS // 10)
            self.assertAlmostEqual(t[0], -TDIV * 5 + DELAY)
            self.assertAlmostEqual(t[1] - t[0], 10 * INTERVAL, places=12)
            self.assertGreater(t[-1] - t[0], 0.99 * (POINTS - 1) * INTERVAL)  # jendela penuh, bukan 10%
            np.testing.assert_allclose(v, (codes('C1') / CODE_PER_DIV * VDIV - OFFSET)[::10], atol=1e-7)
            t, v, _ = scope.waveform('C1')        # tanpa batas: interval kembali 1
            self.assertEqual(v.size, POINTS)
        finally:
            scope.close()

    def test_empty_preamble_before_trigger(self):
        fake = FakeSiglent()
        fake.armed_empty = True
        fake.start()
        scope = ss.SiglentScope('127.0.0.1', fake.port)
        try:
            start, end = scope.window_us('C1')     # dari :TIMebase + :ACQuire, bukan preamble kosong
            self.assertAlmostEqual(start, (-TDIV * 5 + DELAY) * 1e6)
            self.assertAlmostEqual(end, start + (POINTS - 1) / 2e8 * 1e6)
            with self.assertRaisesRegex(RuntimeError, 'menunggu trigger'):
                scope.waveform('C1')
            self.assertEqual(scope.query(':TRIGger:MODE?'), 'AUTO')
        finally:
            scope.close()

    def test_double_newline_after_data_block_keeps_queries_in_sync(self):
        fake = FakeSiglent()
        fake.start()
        scope = ss.SiglentScope('127.0.0.1', fake.port)
        try:
            for _ in range(3):
                scope.waveform('C1')
                self.assertEqual(scope.query(':TRIGger:STATus?'), "Trig'd")
                self.assertFalse(scope.channel_on('C3'))
                self.assertEqual(scope.query(':TRIGger:MODE?'), 'AUTO')
        finally:
            scope.close()

    def test_massflo_gui_shows_scope_panel_aligned_with_raw(self):
        import contextlib, io
        from unittest.mock import patch
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from ultrasonic_film import massflow as mf
        processor = mf.EchoProcessor()
        worker = mf.MassFlowAcquisition(mf.parse_args([]), None, None, processor)
        class FakeReader:
            def __init__(self, *a, **k): self.stop_event = threading.Event()
            def start(self): pass
            def snapshot(self):
                t = np.linspace(0, 100, 20000)
                return {'C1': (t, np.sin(t)), 'C2': (t, 0.1 * np.cos(t))}, 'SDS804X HD', None, 5.0
        class Timer:
            def add_callback(self, callback): self.callback = callback
            def start(self): self.callback()
            def stop(self): pass
        def show():
            fig = plt.gcf()
            fig.canvas.draw()
            ax_raw, ax_scope = fig.axes[0], fig.axes[1]
            self.assertEqual(ax_scope.get_xlim(), ax_raw.get_xlim())
            self.assertEqual(len(ax_scope.lines[0].get_xdata()), 20000)
            self.assertIn('Osiloskop SDS804X HD', fig.texts[0].get_text())
        try:
            with patch.object(mf.shutil, 'which', return_value=sys.executable), \
                 patch.object(mf.getpass, 'getpass', return_value='x'), \
                 patch.object(mf, 'EchoProcessor', return_value=processor), \
                 patch.object(mf, 'MassFlowAcquisition', return_value=worker), \
                 patch.object(ss, 'ScopeReader', FakeReader), \
                 patch.object(processor, 'start'), patch.object(processor, 'join'), \
                 patch.object(worker, 'start'), patch.object(worker, 'join'), \
                 patch.object(worker, 'is_alive', return_value=False), \
                 patch.object(FigureCanvasAgg, 'new_timer', return_value=Timer()), \
                 patch.object(plt, 'show', side_effect=show), \
                 contextlib.redirect_stdout(io.StringIO()):
                mf.main(['--scope', '--host-key', 'SHA256:test-only-mocked'])
        finally:
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
