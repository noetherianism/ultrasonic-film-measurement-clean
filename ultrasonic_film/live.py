"""Live CH1 noise inspection over native DMA (target 300 acquisition FPS). No waveform, image, or result files are saved."""
import argparse
from collections import deque
import math
import os
import threading
import time
import sys
import struct
import subprocess
import tempfile
import queue
import re
import shutil
import getpass
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, periodogram
from .transport import NativeTransport


class LiveAcquisition(threading.Thread):
    """Receive every DMA frame off the GUI thread; retain only the latest one."""
    def __init__(self, args, password=None, plink=None):
        super().__init__(daemon=True)
        self.args, self.password, self.plink = args, password, plink
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.desired_tx = args.tx == 'on'
        self.latest = None
        self.error = None
        self.status = 'Membangun server DMA...'
        self.frames = 0
        self.started = None
        self.sample_rate = None
        self.rx_fps = 0.0

    def set_tx(self, enabled):
        with self.lock:
            self.desired_tx = enabled
            self.latest = None

    def snapshot(self):
        with self.lock:
            return self.latest, self.error, self.status, self.desired_tx

    def check_stop(self):
        if self.stop_event.is_set():
            raise InterruptedError('Stopped')

    def receive(self, rp, sample_rate):
        if not (math.isfinite(sample_rate) and sample_rate > 2*self.args.filter_high and
                max(self.args.gate_us[1], self.args.echo_us[1]) <= 16383/sample_rate*1e6):
            raise ValueError('Device sample rate does not support filter/gate settings')
        self.sample_rate = sample_rate
        with self.lock:
            requested_tx = self.desired_tx
        self.started = time.perf_counter()
        rp._socket.sendall(b'1' if requested_tx else b'0')
        epoch, previous_tx = 0, None
        times = deque()
        quitting = False
        while True:
            with self.lock:
                desired = self.desired_tx
            if self.stop_event.is_set() and not quitting:
                rp._socket.sendall(b'Q')
                quitting = True
            elif desired != requested_tx and not quitting:
                rp._socket.sendall(b'1' if desired else b'0')
                requested_tx = desired
            magic, sequence, samples, size = struct.unpack('!4I', rp.read_exact(16))
            if sequence != self.frames:
                raise ValueError(f'Live sequence mismatch: {sequence} != {self.frames}')
            if magic == 0x52504531:
                if samples or size:
                    raise ValueError('Invalid live end marker')
                if not quitting:
                    raise RuntimeError('Live server stopped unexpectedly')
                break
            if magic not in (0x52554C30, 0x52554C31) or samples != 16384 or size != 32768:
                raise ValueError('Invalid live frame header')
            raw = np.frombuffer(rp.read_exact(size), dtype='<i2').copy()
            tx = magic == 0x52554C31
            if tx != previous_tx:
                epoch += 1
                previous_tx = tx
            self.frames += 1
            now = time.perf_counter()
            times.append(now)
            while len(times) > 2 and now-times[0] > 1:
                times.popleft()
            self.rx_fps = (len(times)-1)/(now-times[0]) if len(times) > 1 and now > times[0] else 0.0
            with self.lock:
                if tx == self.desired_tx:
                    self.latest = (self.frames, epoch, tx, raw, now)
                    self.status = f'TX {"ON / AWG_PE" if tx else "OFF / NOW"}'

    def run(self):
        process, rp, reader_thread = None, None, None
        try:
            root = Path(__file__).resolve().parent
            source = root/'native/rp_acquisition.cpp'
            if not self.args.host_key:
                raise ValueError('Set --host-key or RP_HOST_KEY to the verified SSH host fingerprint')
            with tempfile.TemporaryDirectory(prefix='rp-live-') as temporary:
                pwfile = Path(temporary)/'password'
                pwfile.write_text(self.password+'\n', encoding='utf-8')
                self.password = None
                command = [self.plink, '-ssh', '-batch', '-hostkey', self.args.host_key,
                           '-pwfile', str(pwfile), f'{self.args.user}@{self.args.host}']
                build = ('mkdir -p /tmp/ultrasonic-rp-live && cat > /tmp/ultrasonic-rp-live/live.cpp && '
                         'g++ -O3 -std=c++20 /tmp/ultrasonic-rp-live/live.cpp -I/opt/redpitaya/include '
                         '-L/opt/redpitaya/lib -Wl,-rpath,/opt/redpitaya/lib -lrp -pthread -o /tmp/ultrasonic-rp-live/live')
                subprocess.run(command+[build], input=source.read_bytes(), check=True, timeout=120)
                self.check_stop()
                remote = (f'/tmp/ultrasonic-rp-live/live 0 direct {self.args.fps} ultrasound '
                          f'{self.args.frequency} {self.args.cycles} {self.args.amplitude} '
                          f'{self.args.decimation} {self.args.mux_profile} live')
                process = subprocess.Popen(command+[remote], stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                lines = queue.Queue()
                def reader():
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        lines.put(line)
                    lines.put(None)
                reader_thread = threading.Thread(target=reader, daemon=True)
                reader_thread.start()
                deadline = time.perf_counter()+60
                while True:
                    self.check_stop()
                    if time.perf_counter() > deadline:
                        raise TimeoutError('Native live server did not become ready')
                    try:
                        line = lines.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if line is None:
                        raise RuntimeError('Native live server exited before READY')
                    if line.startswith('READY '):
                        break
                match = re.search(r'\bsample_rate=([0-9.]+)', line)
                if not match:
                    raise ValueError('Missing device sample rate')
                self.check_stop()
                rp = NativeTransport(self.args.host, timeout=self.args.timeout, port=5001)
                if sys.platform == 'win32':
                    rp.enable_quick_ack()
                self.receive(rp, float(match.group(1)))
                if process.wait(timeout=10):
                    raise RuntimeError('Native live server reported failure')
        except InterruptedError:
            pass
        except Exception as exc:
            with self.lock:
                self.error = f'{type(exc).__name__}: {exc}'
            print(self.error, flush=True)
        finally:
            self.password = None
            if rp is not None:
                rp.close()
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                if reader_thread is not None:
                    reader_thread.join(timeout=1)
                process.stdout.close()

class NoiseAnalysis:
    def __init__(self, sample_rate, gate_us, averages, filter_low, filter_high,
                 blank_us=2.5, envelope_us=0.2):
        self.sample_rate = sample_rate
        self.time_us = np.arange(16384) / sample_rate * 1e6
        self.mask = (self.time_us >= gate_us[0]) & (self.time_us <= gate_us[1])
        if np.count_nonzero(self.mask) < 32:
            raise ValueError('Gate must contain at least 32 samples')
        self.history = deque(maxlen=averages)
        self.total = np.zeros(16384)
        self.blank = self.time_us < blank_us
        self.smoothing = max(1, int(round(envelope_us*sample_rate/1e6)))
        self.epoch = None
        self.sos = butter(8, [filter_low, filter_high], fs=sample_rate, btype='bandpass', output='sos')

    def clear(self):
        self.history.clear()
        self.total.fill(0)

    def update(self, raw, epoch):
        if epoch != self.epoch:
            self.clear()
            self.epoch = epoch
        values = raw.astype(float)
        gated = values[self.mask]
        if len(self.history) == self.history.maxlen:
            self.total -= self.history.popleft()
        self.history.append(values)
        self.total += values
        averaged = self.total / len(self.history)
        mean = averaged[self.mask]
        # Average signed, trigger-aligned waveforms before nonlinear envelope detection.
        filtered = sosfiltfilt(self.sos, averaged)
        envelope = np.abs(hilbert(filtered))
        if self.smoothing > 1:
            envelope = np.convolve(envelope, np.ones(self.smoothing)/self.smoothing, mode='same')
        # Blanking affects the processed display only, after filtering/envelope extraction.
        filtered[self.blank] = 0
        envelope[self.blank] = 0
        frequency, psd = periodogram(gated, fs=self.sample_rate, window='hann', detrend='constant')
        return dict(gated=gated, mean=mean, filtered=filtered, envelope=envelope,
                    frequency_mhz=frequency/1e6, psd_db=10*np.log10(np.maximum(psd, 1e-20)),
                    rms=float(np.std(gated)), mean_rms=float(np.std(mean)),
                    clipped=int(np.count_nonzero((raw <= -8192) | (raw >= 8191))), n=len(self.history))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('host', nargs='?', default=os.environ.get('RP_HOST', 'redpitaya.local'),
                   help='device hostname (default: RP_HOST or redpitaya.local)')
    p.add_argument('--user', default='root')
    p.add_argument('--host-key', default=os.environ.get('RP_HOST_KEY'),
                   help='verified SSH host fingerprint (default: RP_HOST_KEY)')
    p.add_argument('--tx', choices=('off', 'on'), default='off', help='initial transmitter state (default: off)')
    p.add_argument('--fps', type=float, default=300, help='native DMA acquisition target (default: 300 FPS)')
    p.add_argument('--display-fps', type=float, default=15)
    p.add_argument('--frequency', type=float, default=5e6)
    p.add_argument('--amplitude', type=float, default=1.0)
    p.add_argument('--cycles', type=int, default=1)
    p.add_argument('--decimation', type=int, choices=(1, 2, 4, 8), default=2)
    p.add_argument('--adc-rate', type=float, default=125e6, help='initial base-rate estimate; device readback overrides axes')
    p.add_argument('--gate-us', nargs=2, type=float, default=(10, 200), metavar=('START', 'END'))
    p.add_argument('--averages', type=int, default=20)
    p.add_argument('--filter-low', type=float, help='bandpass lower edge; default 0.7 * burst frequency')
    p.add_argument('--filter-high', type=float, help='bandpass upper edge; default 1.3 * burst frequency')
    p.add_argument('--echo-us', nargs=2, type=float, default=(2.5, 60), metavar=('START', 'END'),
                   help='processed echo display window in microseconds')
    p.add_argument('--blank-us', type=float, default=2.5, help='hide transmit interval in processed curves only')
    p.add_argument('--envelope-us', type=float, default=0.2, help='envelope moving-average width in microseconds')
    p.add_argument('--timeout', type=float, default=3)
    p.add_argument('--mux-profile', choices=('unchanged', 'single-ch1'), default='unchanged')
    args = p.parse_args(argv)
    if args.filter_low is None:
        args.filter_low = 0.7*args.frequency
    if args.filter_high is None:
        args.filter_high = 1.3*args.frequency
    values = (args.fps, args.display_fps, args.frequency, args.amplitude, args.adc_rate,
              args.filter_low, args.filter_high, args.timeout, *args.gate_us,
              *args.echo_us, args.blank_us, args.envelope_us)
    if not all(map(math.isfinite, values)):
        p.error('All numeric parameters must be finite')
    sample_rate = args.adc_rate/args.decimation
    if not (1 <= args.fps <= 300 and 0 < args.display_fps <= 60 and args.timeout > 0 and
            1 <= args.frequency <= 50e6 and 0 < args.amplitude <= 1 and 1 <= args.cycles <= 1000 and
            1 <= args.averages <= 200 and 0 < args.filter_low < args.filter_high < sample_rate/2 and
            0 <= args.blank_us < args.echo_us[1] and 0 <= args.envelope_us <= 10 and
            0 <= args.echo_us[0] < args.echo_us[1] <= 16383/sample_rate*1e6 and
            0 <= args.gate_us[0] < args.gate_us[1] <= 16383/sample_rate*1e6):
        p.error('Invalid rate, burst, filter, averaging count, or gate outside capture window')
    if args.cycles/args.frequency >= min(16384/sample_rate, 1/args.fps):
        p.error('Burst must fit capture window and frame period')
    if np.count_nonzero((np.arange(16384)/sample_rate*1e6 >= args.gate_us[0]) &
                       (np.arange(16384)/sample_rate*1e6 <= args.gate_us[1])) < 32:
        p.error('Gate must contain at least 32 samples')
    if np.count_nonzero((np.arange(16384)/sample_rate*1e6 >= max(args.echo_us[0], args.blank_us)) &
                       (np.arange(16384)/sample_rate*1e6 <= args.echo_us[1])) < 32:
        p.error('Echo window after blanking must contain at least 32 samples')
    return args


def main():
    args = parse_args()
    if not args.host_key:
        raise SystemExit('Set --host-key or RP_HOST_KEY to the verified SSH host fingerprint')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Button
    analysis = NoiseAnalysis(args.adc_rate/args.decimation, args.gate_us, args.averages,
                             args.filter_low, args.filter_high, args.blank_us, args.envelope_us)
    plink = shutil.which('plink') or r'C:\Program Files\PuTTY\plink.exe'
    if not Path(plink).is_file():
        raise SystemExit('PuTTY plink.exe is required, as for film-benchmark')
    password = getpass.getpass(f'SSH password for {args.user}@{args.host}: ')
    worker = LiveAcquisition(args, password, plink)
    password = None
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.subplots_adjust(bottom=0.22, hspace=0.35, wspace=0.25)
    raw_ax, filtered_ax, gate_ax, fft_ax = axes.flat
    raw_line, = raw_ax.plot(analysis.time_us, np.zeros(16384), lw=0.6)
    raw_ax.set(title='IN1 raw — penuh, tanpa blanking', xlabel='Waktu dari trigger (us)', ylabel='ADC counts')
    raw_ax.set_ylim(-8500, 8500)
    raw_ax.axvspan(*args.gate_us, color='green', alpha=0.1)
    echo_mask = (analysis.time_us >= args.echo_us[0]) & (analysis.time_us <= args.echo_us[1])
    filtered_line, = filtered_ax.plot(analysis.time_us[echo_mask], np.zeros(np.count_nonzero(echo_mask)),
                                    lw=0.6, label='Bandpass dari rata-rata waveform')
    env_line, = filtered_ax.plot(analysis.time_us[echo_mask], np.zeros(np.count_nonzero(echo_mask)),
                               lw=1.2, label='Envelope halus')
    filtered_ax.set(title=f'Echo — average → bandpass {args.filter_low/1e6:g}–{args.filter_high/1e6:g} MHz',
                    xlabel='Waktu dari trigger (us)', ylabel='ADC counts (processed)')
    filtered_ax.set_xlim(*args.echo_us)
    filtered_ax.legend(loc='upper right')
    gate_time = analysis.time_us[analysis.mask]
    gate_line, = gate_ax.plot(gate_time, np.zeros(len(gate_time)), lw=0.5, alpha=0.6, label='Frame terakhir')
    mean_line, = gate_ax.plot(gate_time, np.zeros(len(gate_time)), lw=1, label='Rata-rata frame tampil')
    gate_ax.set(title='Zoom gate — lihat komponen yang berulang', xlabel='Waktu dari trigger (us)', ylabel='ADC counts')
    gate_ax.legend(loc='upper right')
    fft_line, = fft_ax.plot([], [], lw=0.8)
    fft_ax.set(title='PSD gate, DC dihilangkan', xlabel='Frekuensi (MHz)', ylabel='dB (counts²/Hz)')
    fft_ax.set_xlim(0, analysis.sample_rate/2e6)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    radio = RadioButtons(fig.add_axes([0.02, 0.025, 0.12, 0.12]), ('TX OFF', 'TX ON'), active=int(args.tx == 'on'))
    reset = Button(fig.add_axes([0.16, 0.06, 0.13, 0.05]), 'Reset average')
    text = fig.text(0.32, 0.06, 'Menghubungkan...', fontsize=10)
    radio.on_clicked(lambda label: worker.set_tx(label == 'TX ON'))
    reset.on_clicked(lambda event: analysis.clear())
    last_number = None
    shown = 0
    gui_start = time.perf_counter()

    def refresh():
        nonlocal last_number, shown, analysis, gui_start, echo_mask
        packet, error, status, desired_tx = worker.snapshot()
        if error:
            text.set_text('ERROR: ' + error + '\nTutup jendela; periksa terminal.')
        elif packet is None or packet[2] != desired_tx:
            text.set_text(status + '\nMenunggu frame TX ' + ('ON' if desired_tx else 'OFF') + '...')
        elif packet[0] != last_number:
            number, epoch, tx, raw, received = packet
            if worker.sample_rate != analysis.sample_rate:
                analysis = NoiseAnalysis(worker.sample_rate, args.gate_us, args.averages,
                                         args.filter_low, args.filter_high, args.blank_us, args.envelope_us)
                echo_mask = (analysis.time_us >= args.echo_us[0]) & (analysis.time_us <= args.echo_us[1])
                raw_line.set_xdata(analysis.time_us)
                for line in (filtered_line, env_line):
                    line.set_xdata(analysis.time_us[echo_mask])
                gate_line.set_xdata(analysis.time_us[analysis.mask])
                mean_line.set_xdata(analysis.time_us[analysis.mask])
                raw_ax.set_xlim(analysis.time_us[0], analysis.time_us[-1])
                filtered_ax.set_xlim(*args.echo_us)
                gate_ax.set_xlim(*args.gate_us)
                fft_ax.set_xlim(0, analysis.sample_rate/2e6)
            stats = analysis.update(raw, epoch)
            raw_line.set_ydata(raw)
            filtered_line.set_ydata(stats['filtered'][echo_mask])
            env_line.set_ydata(stats['envelope'][echo_mask])
            gate_line.set_ydata(stats['gated'])
            mean_line.set_ydata(stats['mean'])
            fft_line.set_data(stats['frequency_mhz'], stats['psd_db'])
            for ax in (filtered_ax, gate_ax, fft_ax):
                ax.relim()
                ax.autoscale_view(scalex=False)
            if not shown:
                gui_start = time.perf_counter()
            shown += 1
            last_number = number
            acq_rate = worker.rx_fps
            text.set_text(f'{status} | RX {acq_rate:.1f} FPS | Display {(shown-1)/max(time.perf_counter()-gui_start, 1e-6):.1f} FPS\n'
                          f'Gate RMS AC {stats["rms"]:.1f} | mean RMS AC {stats["mean_rms"]:.1f} counts | '
                          f'average {stats["n"]} | clipped {stats["clipped"]}/16384\n'
                          'Sumbu zoom auto-scale. Bandingkan angka RMS, bukan tinggi grafik saja.')
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=max(1, int(1000/args.display_fps)))
    timer.add_callback(refresh)
    fig.canvas.mpl_connect('close_event', lambda event: worker.stop_event.set())
    print(f'Live DMA tanpa simpan data. Target RX {args.fps:g} FPS; tampilan terpisah. TX OFF = NOW; TX ON = AWG_PE. Tutup jendela / Ctrl+C untuk berhenti.', flush=True)
    worker.start()
    timer.start()
    try:
        plt.show()
    finally:
        timer.stop()
        worker.stop_event.set()
        worker.join(timeout=10)
        if worker.is_alive():
            print('Worker belum berhenti; konfirmasi OUTPUT1 OFF pada perangkat.', flush=True)


if __name__ == '__main__':
    main()
