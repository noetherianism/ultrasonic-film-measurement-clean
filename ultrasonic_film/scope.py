# -*- coding: utf-8 -*-
"""
Baca waveform Siglent SDS800X HD / SDS1000X HD / SDS2000X HD lewat LAN (SCPI raw socket, port 5025).

Dipakai oleh film-live (--scope) agar layar laptop bisa dibandingkan
langsung dengan layar osiloskop. Bisa juga dijalankan sendiri untuk cek koneksi:

    film-scope siglent.local              # info + plot C1, C2
    film-scope siglent.local --ch C2 --no-plot
    film-scope siglent.local --setup       # atur trigger C1 + 10 µs/div, jendela 0..100 µs

Konversi data mengikuti contoh resmi Siglent (juga dipakai pymeasure):
    volt  = code / code_per_div * vdiv - offset
    waktu = -(tdiv * 10 / 2) + index * interval + delay
"""

import argparse
import math
import os
import socket
import struct
import threading
import time

import numpy as np

HORI_DIV = 10
TDIV_ENUM = [200e-12, 500e-12, 1e-9, 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9,
             1e-6, 2e-6, 5e-6, 10e-6, 20e-6, 50e-6, 100e-6, 200e-6, 500e-6,
             1e-3, 2e-3, 5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 200e-3, 500e-3,
             1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]


def parse_block(raw):
    """IEEE 488.2 definite-length block '#<n><len><data>' -> data bytes."""
    start = raw.find(b'#')
    if start < 0:
        raise ValueError('Respons osiloskop bukan blok biner')
    digits = int(raw[start + 1:start + 2])
    length = int(raw[start + 2:start + 2 + digits])
    data = raw[start + 2 + digits:start + 2 + digits + length]
    if len(data) != length:
        raise ValueError(f'Blok biner terpotong: {len(data)} dari {length} byte')
    return bytes(data)


def parse_preamble(desc):
    """WAVEDESC biner (setelah header blok) -> dict. Offset sesuai programming guide Siglent HD."""
    if len(desc) < 0x14C:
        raise ValueError(f'Preamble terlalu pendek ({len(desc)} byte)')
    f = lambda off: struct.unpack_from('<f', desc, off)[0]
    probe = f(0x148)
    if not math.isfinite(probe) or probe == 0:
        probe = 1.0
    tdiv_index = struct.unpack_from('<h', desc, 0x144)[0]
    return dict(
        vdiv=f(0x9C) * probe,
        offset=f(0xA0) * probe,
        code_per_div=f(0xA4),
        adc_bit=struct.unpack_from('<h', desc, 0xAC)[0],
        interval=f(0xB0),
        delay=struct.unpack_from('<d', desc, 0xB4)[0],
        tdiv=TDIV_ENUM[tdiv_index] if 0 <= tdiv_index < len(TDIV_ENUM) else float('nan'),
        probe=probe,
    )


def preamble_valid(pre):
    """Preamble berisi nol (interval inf) bila belum ada akuisisi sejak timebase diubah."""
    return math.isfinite(pre['interval']) and pre['interval'] > 0 and pre['code_per_div'] > 0


class SiglentScope:
    def __init__(self, host, port=5025, timeout=3.0):
        self.host, self.port = host, port
        self.sock = socket.create_connection((host, port), timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self._buffer = bytearray()

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def write(self, command):
        self.sock.sendall(command.encode('ascii') + b'\n')

    def _fill(self):
        chunk = self.sock.recv(1 << 16)
        if not chunk:
            raise ConnectionError('Koneksi osiloskop terputus')
        self._buffer.extend(chunk)

    def query(self, command):
        self.write(command)
        while True:
            while b'\n' not in self._buffer:
                self._fill()
            line, _, rest = bytes(self._buffer).partition(b'\n')
            self._buffer = bytearray(rest)
            # Baris kosong = sisa terminator '\n\n' dari :WAVeform:DATA? (SDS800X HD fw 4.8), bukan jawaban.
            if line.strip():
                return line.decode('ascii', 'replace').strip()

    def query_block(self, command):
        """Kirim query yang menjawab dengan blok biner; baca tepat sepanjang header."""
        self.write(command)
        while True:
            start = self._buffer.find(b'#')
            if start >= 0 and len(self._buffer) >= start + 2:
                digits = int(self._buffer[start + 1:start + 2])
                if len(self._buffer) >= start + 2 + digits:
                    length = int(self._buffer[start + 2:start + 2 + digits])
                    end = start + 2 + digits + length
                    while len(self._buffer) < end + 1:  # + terminator '\n'
                        self._fill()
                    data = bytes(self._buffer[start + 2 + digits:end])
                    self._buffer = bytearray(self._buffer[end:].lstrip(b'\r\n'))
                    return data
            self._fill()

    def idn(self):
        return self.query('*IDN?')

    def channel_on(self, channel):
        return self.query(f':CHANnel{channel[1:]}:SWITch?').upper().startswith('ON')

    def settings(self):
        return dict(
            tdiv=float(self.query(':TIMebase:SCALe?')),
            tdelay=float(self.query(':TIMebase:DELay?')),
            srate=float(self.query(':ACQuire:SRATe?')),
            points=int(float(self.query(':ACQuire:POINts?'))),
        )

    def window_us(self, channel):
        """Rentang waktu (µs) dari preamble channel, tanpa memindahkan data waveform."""
        self.write(f':WAVeform:SOURce {channel}')
        pre = parse_preamble(self.query_block(':WAVeform:PREamble?'))
        points = int(float(self.query(':ACQuire:POINts?')))
        if not preamble_valid(pre):
            # Belum ter-trigger: hitung dari setelan. Terverifikasi di SDS804X HD fw 4.8.12:
            # DELay +50 µs pada 10 µs/div -> jendela 0..100 µs (DELay positif = jendela setelah trigger).
            s = self.settings()
            start = -(s['tdiv'] * HORI_DIV / 2) + s['tdelay']
            return start * 1e6, (start + (s['points'] - 1) / s['srate']) * 1e6
        tdiv = pre['tdiv'] if math.isfinite(pre['tdiv']) else float(self.query(':TIMebase:SCALe?'))
        start = -(tdiv * HORI_DIV / 2) + pre['delay']
        return start * 1e6, (start + (points - 1) * pre['interval']) * 1e6

    def setup_for_redpitaya(self, trigger_ch='C1', level=0.2, tdiv=10e-6, window_start_us=0.0):
        """Trigger edge naik normal di trigger_ch; jendela waktu mulai di window_start_us.

        Tanda :TIMebase:DELay berbeda antar firmware, jadi arah geser diverifikasi
        dari sumbu waktu preamble dan dibalik bila perlu.
        """
        n = trigger_ch[1:]
        self.write(f':CHANnel{n}:SWITch ON')
        self.write(':TRIGger:TYPE EDGE')
        self.write(f':TRIGger:EDGE:SOURce {trigger_ch}')
        self.write(':TRIGger:EDGE:SLOPe RISing')
        self.write(f':TRIGger:EDGE:LEVel {level:.4E}')
        self.write(':TRIGger:MODE NORMal')
        self.write(f':TIMebase:SCALe {tdiv:.4E}')
        shift = tdiv * HORI_DIV / 2 - window_start_us * 1e-6
        best = None
        for delay in (shift, -shift):
            self.write(f':TIMebase:DELay {delay:.4E}')
            self.query('*OPC?')
            time.sleep(0.3)
            start, end = self.window_us(trigger_ch)
            error = abs(start - window_start_us)
            if best is None or error < best[0]:
                best = (error, delay, start, end)
            if error < tdiv * 1e6 * 0.5:
                break
        if best[1] != delay:
            self.write(f':TIMebase:DELay {best[1]:.4E}')
            self.query('*OPC?')
            time.sleep(0.3)
            best = (best[0], best[1]) + self.window_us(trigger_ch)
        self.write(':TRIGger:RUN')
        return dict(
            trigger_source=self.query(':TRIGger:EDGE:SOURce?'),
            trigger_slope=self.query(':TRIGger:EDGE:SLOPe?'),
            trigger_level=self.query(':TRIGger:EDGE:LEVel?'),
            trigger_mode=self.query(':TRIGger:MODE?'),
            tdiv=self.query(':TIMebase:SCALe?'),
            delay=self.query(':TIMebase:DELay?'),
            window_us=(round(best[2], 3), round(best[3], 3)),
        )

    def waveform(self, channel, max_points=None):
        """-> (waktu_detik, volt, preamble). Seluruh jendela layar.

        max_points tidak memotong jendela: titik dijarangkan di osiloskop (:WAVeform:INTerval)
        sehingga rentang waktu tetap penuh (mis. 200k titik 0..100 µs -> 20k titik tiap 5 ns).
        """
        self.write(f':WAVeform:SOURce {channel}')
        self.write(':WAVeform:STARt 0')
        pre = parse_preamble(self.query_block(':WAVeform:PREamble?'))
        if not preamble_valid(pre):
            raise RuntimeError(f'belum ada waveform {channel}: osiloskop menunggu trigger '
                               '(cek sinyal TX di channel trigger dan level trigger)')
        word = pre['adc_bit'] > 8
        self.write(f':WAVeform:WIDTh {"WORD" if word else "BYTE"}')
        total = int(float(self.query(':ACQuire:POINts?')))
        step = math.ceil(total / max_points) if max_points and total > max_points else 1
        self.write(f':WAVeform:INTerval {step}')
        count = math.ceil(total / step)
        piece = int(float(self.query(':WAVeform:MAXPoint?'))) or count
        self.write(f':WAVeform:POINt {min(piece, count)}')
        raw = bytearray()
        for start in range(0, count, piece):
            self.write(f':WAVeform:STARt {start * step}')
            raw.extend(self.query_block(':WAVeform:DATA?'))
        dtype = '<i2' if word else 'i1'
        codes = np.frombuffer(bytes(raw[:len(raw) - len(raw) % (2 if word else 1)]), dtype=dtype)[:count]
        volts = codes.astype(np.float64) / pre['code_per_div'] * pre['vdiv'] - pre['offset']
        tdiv = pre['tdiv'] if math.isfinite(pre['tdiv']) else float(self.query(':TIMebase:SCALe?'))
        times = -(tdiv * HORI_DIV / 2) + np.arange(codes.size) * (pre['interval'] * step) + pre['delay']
        return times, volts, pre


class ScopeReader(threading.Thread):
    """Ambil waveform osiloskop berkala di thread terpisah; simpan hanya yang terbaru."""

    def __init__(self, host, channels=('C1', 'C2'), rate_hz=5.0, max_points=20000, setup=None):
        super().__init__(daemon=True)
        self.setup = setup         # dict argumen setup_for_redpitaya, atau None
        self.host, self.channels, self.period = host, tuple(channels), 1.0 / rate_hz
        self.max_points = max_points
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = {}           # channel -> (t_us, volts)
        self.info = ''
        self.error = None
        self.updates = 0
        self.fps = 0.0
        self.trigger_status = ''

    def snapshot(self):
        with self.lock:
            info = f'{self.info} [trigger {self.trigger_status}]' if self.trigger_status else self.info
            return dict(self.latest), info, self.error, self.fps

    def run(self):
        scope = None
        while not self.stop_event.is_set():
            try:
                if scope is None:
                    scope = SiglentScope(self.host)
                    idn = scope.idn()
                    active = [ch for ch in self.channels if scope.channel_on(ch)]
                    if self.setup is not None:
                        result = scope.setup_for_redpitaya(**self.setup)
                        print(f'Osiloskop diatur: {result}', flush=True)
                        self.setup = None
                    with self.lock:
                        self.info, self.error = idn.split(',')[1] if ',' in idn else idn, None
                    if not active:
                        raise RuntimeError(f'Channel {", ".join(self.channels)} mati di osiloskop')
                started = time.perf_counter()
                try:
                    trig = scope.query(':TRIGger:STATus?')
                except Exception:
                    trig = '?'
                result = {}
                for ch in active:
                    t, v, _ = scope.waveform(ch, self.max_points)
                    result[ch] = (t * 1e6, v)
                elapsed = time.perf_counter() - started
                with self.lock:
                    self.latest = result
                    self.trigger_status = trig
                    self.updates += 1
                    self.fps = 1.0 / max(elapsed, 1e-6) if elapsed > self.period else 1.0 / self.period
                self.stop_event.wait(max(0.0, self.period - elapsed))
            except Exception as exc:
                with self.lock:
                    self.error = f'Osiloskop {type(exc).__name__}: {exc}'
                if scope is not None:
                    scope.close()
                    scope = None
                self.stop_event.wait(2.0)  # coba sambung ulang
        if scope is not None:
            scope.close()


def main(argv=None):
    p = argparse.ArgumentParser(description='Cek koneksi & baca waveform Siglent HD')
    p.add_argument('host', nargs='?', default=os.environ.get('SIGLENT_HOST', 'siglent.local'))
    p.add_argument('--ch', nargs='+', default=['C1', 'C2'])
    p.add_argument('--no-plot', action='store_true')
    p.add_argument('--setup', action='store_true', help='atur trigger & timebase agar sejajar dengan Red Pitaya')
    p.add_argument('--trigger-ch', default='C1', help='channel yang tersambung ke OUT1 (burst TX)')
    p.add_argument('--trigger-level', type=float, default=0.2, help='level trigger (V)')
    p.add_argument('--tdiv-us', type=float, default=10.0, help='timebase µs/div')
    args = p.parse_args(argv)
    scope = SiglentScope(args.host)
    try:
        print('IDN      :', scope.idn())
        print('Trigger  :', scope.query(':TRIGger:STATus?'), '| sumber', scope.query(':TRIGger:EDGE:SOURce?'),
              '| level', scope.query(':TRIGger:EDGE:LEVel?'), '| mode', scope.query(':TRIGger:MODE?'))
        if args.setup:
            print('Setup    :', scope.setup_for_redpitaya(args.trigger_ch.upper(), args.trigger_level,
                                                          args.tdiv_us * 1e-6))
        s = scope.settings()
        print(f"Timebase : {s['tdiv']*1e6:g} µs/div, delay {s['tdelay']*1e6:g} µs, "
              f"{s['srate']/1e6:g} MSa/s, {s['points']} titik")
        print(f"Jendela  : {(-s['tdiv']*5 + s['tdelay'])*1e6:.2f} .. {(s['tdiv']*5 + s['tdelay'])*1e6:.2f} µs "
              '(0 = titik trigger; DELay positif menggeser jendela ke kanan / setelah trigger)')
        data = {}
        for ch in args.ch:
            if not scope.channel_on(ch):
                print(f'{ch}       : mati')
                continue
            try:
                t, v, pre = scope.waveform(ch)
            except RuntimeError as exc:
                print(f'{ch}       : {exc}')
                continue
            data[ch] = (t, v)
            print(f"{ch}       : Vpp {v.max()-v.min():.4f} V, {v.size} titik, {t[0]*1e6:.2f} .. {t[-1]*1e6:.2f} µs, "
                  f"V {v.min():.4f} .. {v.max():.4f} | vdiv {pre['vdiv']:g} V, offset {pre['offset']:g} V, "
                  f"adc {pre['adc_bit']} bit, code/div {pre['code_per_div']:g}, delay {pre['delay']*1e6:g} µs")
    finally:
        scope.close()
    if data and not args.no_plot:
        import matplotlib.pyplot as plt
        colors = {'C1': '#c9a400', 'C2': '#e8710a', 'C3': '#1f77b4', 'C4': '#2ca02c'}
        plt.figure(figsize=(11, 4))
        for ch, (t, v) in data.items():
            plt.plot(t * 1e6, v, lw=0.7, color=colors.get(ch), label=ch)
        plt.xlabel('Waktu dari trigger osiloskop (µs)')
        plt.ylabel('Tegangan (V)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
