import argparse
from collections import deque
from datetime import datetime
from dataclasses import replace, asdict
import getpass
import json
import math
import os
from pathlib import Path
import queue
import shutil
import struct
import tempfile
import threading
import time

import numpy as np

from . import live
from . import thin_layer as lapisan_tipis
from .recording import FrameRecorder, SAMPLES_PER_FRAME, load_recording
from .processing import Echo, EchoConfig, EchoDSP as _EchoDSP
from .flow import FlowFeatures, FlowRegimeClassifier

IP_ADDRESS   = os.environ.get('RP_HOST', 'redpitaya.local')
SCOPE_IP     = os.environ.get('SIGLENT_HOST', 'siglent.local')
FREQ_HZ      = 5_000_000
N_CYCLES     = 1
AWG_VOLT     = 1.0
ACQ_DEC      = 2
TARGET_FPS   = 300
DSP_BACKLOG_WARN = 150
DISPLAY_FPS  = 15
MUX_PROFILE  = 'single-ch1'

FILTER_LOW      = 3_000_000
FILTER_HIGH     = 4_800_000
DIGITAL_GAIN    = 12.0
MA_WINDOW       = 32
NOISE_THRESHOLD = 0.15
BLANK_US        = 2.5

AKRILIK_US        = (2.5, 6.0)
AIR_MULAI_US      = 1.5
AIR_SELESAI_US    = 120.0
PANTULAN_GANDA_US = 2.5

C_AIR    = 1466.0
SUHU_AIR = None


def c_air_marczak(suhu_c):
    if not 0 <= suhu_c <= 95:
        raise ValueError('Rumus Marczak berlaku 0-95 °C')
    t = suhu_c
    return (1.402385e3 + 5.038813 * t - 5.799136e-2 * t ** 2 + 3.287156e-4 * t ** 3
            - 1.398845e-6 * t ** 4 + 2.787860e-9 * t ** 5)


def tinggi_air_mm(tof_us):
    return C_AIR * tof_us / 2000.0


BASELINE = None

THRESHOLD_NOISE   = 1.2
THRESHOLD_AKRILIK = 0.8
AKRILIK_NAIK      = 1.5
NOISE_REF_US      = (90.0, 125.0)
NOISE_HISTORY     = 600
NOISE_KECUALI_US  = (-2.0, 9.5)
NOISE_KECUALI_M   = (2, 3)
PANTULAN_US       = None
PANTULAN_R        = 0.15
PANTULAN_LEBAR_US = 0.8
PANTULAN_MARGIN   = 2.0
PRESETS = {'akrilik5mm': dict(ma=32, threshold_noise=1.2, pantulan_us=3.744)}

IDX_MAX = 8000
VOLT_PER_COUNT = 1.0 / 8192

FOLDER_REKAMAN = Path('recordings')
TREND_DETIK    = 10

MAGIC_TX_ON, MAGIC_TX_OFF, MAGIC_END = 0x52554C31, 0x52554C30, 0x52504531
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2


def dsp_config():
    return EchoConfig(
        filter_low_hz=FILTER_LOW,
        filter_high_hz=FILTER_HIGH,
        digital_gain=DIGITAL_GAIN,
        ma_window=MA_WINDOW,
        noise_threshold=NOISE_THRESHOLD,
        blank_us=BLANK_US,
        acrylic_window_us=AKRILIK_US,
        water_start_us=AIR_MULAI_US,
        water_end_us=AIR_SELESAI_US,
        display_tail_us=PANTULAN_GANDA_US,
        threshold_noise=THRESHOLD_NOISE,
        threshold_acrylic=THRESHOLD_AKRILIK,
        acrylic_rise=AKRILIK_NAIK,
        noise_window_us=NOISE_REF_US,
        noise_history=NOISE_HISTORY,
        noise_exclude_us=NOISE_KECUALI_US,
        noise_exclude_multiples=NOISE_KECUALI_M,
        plate_period_us=PANTULAN_US,
        plate_ratio=PANTULAN_R,
        plate_half_width_us=PANTULAN_LEBAR_US,
        plate_margin=PANTULAN_MARGIN,
        sample_count=IDX_MAX,
        volt_per_count=VOLT_PER_COUNT,
    )


class EchoDSP(_EchoDSP):

    def __init__(self, sample_rate, config=None, secondary_peaks=False):
        selected = config if config is not None else dsp_config()
        if secondary_peaks:
            selected = replace(selected, secondary_peaks=True)
        super().__init__(sample_rate, selected)


class TofWriter:
    HEADER = 'frame_index,stream_sequence,receive_elapsed_seconds,status,waktu_akrilik_us,tof_air_us,amp_air,h_mm\n'
    FLOW_HEADER = ('tof_internal_us,n_peaks,flow_pattern,flow_age_s,mean_h_mm,sigma_h_mm,'
                   'amplitude_cv,f_wave_hz,wave_h_pp_mm,internal_fraction,valid_fraction')

    def __init__(self, path, mode='x', flow_features=False):
        self.path = Path(path)
        self.file = self.path.open(mode, encoding='utf-8', newline='', buffering=1024 * 1024)
        self.flow_features = flow_features
        self.file.write(self.HEADER.rstrip() + ',' + self.FLOW_HEADER + '\n' if flow_features else self.HEADER)
        self.rows = 0

    def write(self, index, sequence, elapsed, echo, flow=None, flow_age=float('nan')):
        fmt = lambda x: '' if math.isnan(x) else f'{x:.6f}'
        row = (f'{index},{sequence},{elapsed:.9f},{echo.status},'
                        f'{fmt(echo.waktu_akrilik_us)},{fmt(echo.tof_air_us)},{fmt(echo.amp_air)},'
                        f'{fmt(tinggi_air_mm(echo.tof_air_us))}')
        if self.flow_features:
            flow = flow or FlowFeatures()
            values = (flow_age, flow.mean_h_mm, flow.sigma_h_mm, flow.amplitude_cv,
                      flow.f_wave_hz, flow.wave_h_pp_mm, flow.internal_fraction, flow.valid_fraction)
            row += f',{fmt(echo.tof_internal_us)},{echo.n_peaks},{flow.pattern},' + ','.join(map(fmt, values))
        self.file.write(row + '\n')
        self.rows += 1

    def close(self):
        self.file.close()


def deteksi_batch(dsp, volts):
    env = dsp.envelope(volts)
    echoes = [dsp.detect(row) for row in env]
    if BASELINE is not None:
        echoes = [Echo(*hasil) for hasil in BASELINE.proses(volts)]
    return env, echoes


def muat_baseline(folder, rata=1, faktor=None, cocok=None):
    folder = Path(folder)
    waveforms, metadata = load_recording(folder)
    fs = metadata['sample_rate_hz']
    dsp = EchoDSP(fs)
    volts = dsp.to_volts(waveforms)
    echoes = []
    for start in range(0, len(volts), 256):
        echoes += [dsp.detect(row) for row in dsp.envelope(volts[start:start + 256])]
    akrilik = np.array([e.waktu_akrilik_us for e in echoes])
    tof = np.array([e.tof_air_us for e in echoes])
    if np.mean(np.isfinite(tof)) < 0.9:
        raise ValueError(f'Baseline {folder.name}: echo air hanya terdeteksi di {np.mean(np.isfinite(tof))*100:.0f}% frame')
    baseline = lapisan_tipis.BaselineTipis.dari_rekaman(
        volts, fs, float(np.nanmedian(akrilik)), float(np.nanmedian(tof)), rata=rata,
        faktor_thr=lapisan_tipis.FAKTOR_THR if faktor is None else faktor,
        cocok_faktor=lapisan_tipis.COCOK_FAKTOR if cocok is None else cocok,
        info=dict(folder=str(folder.resolve()), tinggi_air_baseline_mm=tinggi_air_mm(float(np.nanmedian(tof)))))
    print(f"Baseline lapisan tipis: {folder.name}, {len(volts)} frame, akrilik {baseline.t_akrilik_us:.3f} µs, "
          f"air baseline {baseline.info['tinggi_air_baseline_mm']:.1f} mm, threshold sisa maks "
          f"{baseline.threshold[baseline.cari].max():.4f} (faktor {baseline.faktor_thr:g}), rata {rata}", flush=True)
    return baseline


class EchoProcessor(threading.Thread):
    MAX_BATCH = 64

    def __init__(self, trend_seconds=TREND_DETIK, averages=1, median_n=1, flow_features=False, flow_window=2.0,
                 flow_nominal_fps=None):
        super().__init__(daemon=True)
        self.averages = max(1, int(averages))
        self._ring = deque()
        self._sum = None
        self.median_n = max(1, int(median_n))
        self._recent_tof = deque(maxlen=self.median_n)
        self.inbox = queue.SimpleQueue()
        self.lock = threading.Lock()
        self.trend_seconds = trend_seconds
        self.dsp = None
        self.latest = None
        self.history = deque()
        self.processed = 0
        self.detected = 0
        self.dsp_fps = 0.0
        self._times = deque()
        self.error = None
        self.flow = FlowRegimeClassifier(flow_window, nominal_fps=flow_nominal_fps) if flow_features else None

    def snapshot(self):
        with self.lock:
            history = np.array(self.history, dtype=float).reshape(-1, 2)
            return self.latest, history, self.processed, self.dsp_fps, self.error

    def run(self):
        try:
            stop = False
            while not stop:
                item, batch = self.inbox.get(), []
                while True:
                    if item[0] == 'frame':
                        batch.append(item)
                        if len(batch) >= self.MAX_BATCH:
                            self._process(batch)
                            batch = []
                    else:
                        if batch:
                            self._process(batch)
                            batch = []
                        if item[0] == 'config':
                            if BASELINE is not None and abs(BASELINE.fs - item[1]) > 1:
                                raise ValueError(f'Sample rate {item[1]} Hz != baseline {BASELINE.fs} Hz')
                            self.dsp = EchoDSP(item[1], secondary_peaks=self.flow is not None)
                            if self.flow is not None:
                                self.flow = FlowRegimeClassifier(self.flow.window_s, nominal_fps=self.flow.nominal_fps)
                        elif item[0] == 'close':
                            item[1].close()
                        elif item[0] == 'stop':
                            stop = True
                            break
                    try:
                        item = self.inbox.get_nowait()
                    except queue.Empty:
                        break
                if batch:
                    self._process(batch)
        except Exception as exc:
            with self.lock:
                self.error = f'DSP {type(exc).__name__}: {exc}'
            print(self.error, flush=True)

    def _process(self, batch):
        volts = self.dsp.to_volts(np.stack([item[3] for item in batch]))
        if self.averages > 1:
            volts = self._running_average(volts)
        env, echoes = deteksi_batch(self.dsp, volts)
        feature = None
        for item, echo in zip(batch, echoes):
            if self.flow is not None:
                feature = self.flow.update(item[2], tinggi_air_mm(echo.tof_air_us), echo.amp_air,
                                           internal=None if BASELINE is not None else math.isfinite(echo.tof_internal_us),
                                           valid=echo.status == 'OK', sequence=item[1])
            if item[4] is not None:
                if feature is None:
                    item[4].write(item[5], item[1], item[6], echo)
                else:
                    item[4].write(item[5], item[1], item[6], echo, feature, item[2] - feature.computed_at_s)
        now = time.perf_counter()
        with self.lock:
            for item, echo in zip(batch, echoes):
                if echo.status == 'OK':
                    self.history.append((item[2], echo.tof_air_us))
                    self._recent_tof.append(echo.tof_air_us)
                    self.detected += 1
            while self.history and self.history[0][0] < batch[-1][2] - self.trend_seconds:
                self.history.popleft()
            self.processed += len(batch)
            self._times.append((now, self.processed))
            while len(self._times) > 2 and now - self._times[0][0] > 1:
                self._times.popleft()
            t0, n0 = self._times[0]
            self.dsp_fps = (self.processed - n0) / (now - t0) if now > t0 else 0.0
            self.latest = dict(sequence=batch[-1][1], received=batch[-1][2], dsp=self.dsp,
                               flow=feature,
                               volts=volts[-1], env=env[-1], echo=echoes[-1],
                               tof_median=float(np.median(self._recent_tof)) if self._recent_tof else float('nan'),
                               averages=min(len(self._ring), self.averages) if self.averages > 1 else 1)

    def _running_average(self, volts):
        out = np.empty_like(volts)
        if self._sum is None:
            self._sum = np.zeros(volts.shape[1])
        for i, row in enumerate(volts):
            if len(self._ring) == self.averages:
                self._sum -= self._ring.popleft()
            self._ring.append(row)
            self._sum += row
            out[i] = self._sum / len(self._ring)
        return out


class MassFlowAcquisition(live.LiveAcquisition):

    def __init__(self, args, password, plink, processor):
        super().__init__(args, password, plink)
        self.processor = processor
        self.record_requested = bool(args.rekam)
        self.record_dir = None
        self.record_frames = 0
        self.record_message = ''
        self.backlog = 0
        self.backlog_max = 0
        self.backlog_events = 0

    def set_recording(self, enabled):
        with self.lock:
            self.record_requested = enabled

    def recording_snapshot(self):
        with self.lock:
            return self.record_requested, self.record_dir, self.record_frames, self.record_message

    def _open_recording(self, sample_rate, now):
        parent = Path(self.args.folder_rekaman).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=parent))
        parameters = dict(
            program='ultrasonic_film.massflow', host=self.args.host, target_fps=self.args.fps,
            sample_rate_hz=sample_rate, decimation=ACQ_DEC, burst_frequency_hz=FREQ_HZ,
            burst_cycles=N_CYCLES, generator_amplitude_v=AWG_VOLT, mux_profile=self.args.mux_profile,
            trigger='AWG_PE', volt_per_count=VOLT_PER_COUNT,
            experimental_flow=dict(enabled=getattr(self.args, 'flow_features', False),
                                   window_s=getattr(self.args, 'flow_window', 2.0),
                                   version=2, timebase=getattr(self.args, 'flow_timebase', 'sequence'),
                                   nominal_fps=self.args.fps,
                                   timing=('frame sequence / frame rate validated per window from receive '
                                           'times (time.perf_counter); not device trigger timestamps'
                                           if getattr(self.args, 'flow_timebase', 'sequence') == 'sequence'
                                           else 'PC receive time (time.perf_counter)'),
                                   plate_rejection=PANTULAN_US is not None,
                                   physical_regime_validated=False),
            dsp=dict(filter_low_hz=FILTER_LOW, filter_high_hz=FILTER_HIGH, filter_order=8,
                     digital_gain=DIGITAL_GAIN, ma_window=MA_WINDOW, noise_threshold=NOISE_THRESHOLD,
                     blank_us=BLANK_US, akrilik_us=AKRILIK_US, air_mulai_us=AIR_MULAI_US,
                     air_selesai_us=AIR_SELESAI_US, idx_max=IDX_MAX,
                     threshold_mode='manual' if THRESHOLD_NOISE is None else 'otomatis',
                     threshold_noise=THRESHOLD_NOISE, threshold_akrilik=THRESHOLD_AKRILIK,
                     noise_ref_us=NOISE_REF_US, noise_history=NOISE_HISTORY, noise_kecuali_us=NOISE_KECUALI_US,
                     noise_kecuali_m=NOISE_KECUALI_M, akrilik_naik=AKRILIK_NAIK,
                     pantulan_us=PANTULAN_US, pantulan_r=PANTULAN_R, pantulan_lebar_us=PANTULAN_LEBAR_US,
                     pantulan_margin=PANTULAN_MARGIN, preset=getattr(self.args, 'preset', None),
                     baseline_lapisan_tipis=(None if BASELINE is None
                                             else dict(BASELINE.info, faktor_thr=BASELINE.faktor_thr)),
                     rata_frame=getattr(self.args, 'rata', 1),
                     c_air_m_s=C_AIR, suhu_air_c=SUHU_AIR,
                     c_air_sumber=('Marczak dari --suhu-air' if SUHU_AIR is not None
                                   else '--c-air' if getattr(self.args, 'c_air', None) is not None
                                   else 'default (terukur osiloskop)'),
                     catatan='tof.csv dihitung dari waveform rata-rata N frame bila rata_frame > 1; frames.bin tetap mentah'),
            created_local=datetime.now().isoformat())
        (directory / 'acquisition.json').write_text(json.dumps(parameters, indent=2), encoding='utf-8')
        recorder = FrameRecorder(directory, sample_rate_hz=sample_rate, trigger='AWG_PE', host=self.args.host)
        recorder.metadata.update(backend='dma-direct-live', units='DMA raw ADC counts')
        recorder.__enter__()
        try:
            writer = TofWriter(directory / 'tof.csv', flow_features=getattr(self.args, 'flow_features', False))
        except BaseException as exc:
            recorder.__exit__(type(exc), exc, None)
            raise
        with self.lock:
            self.record_dir, self.record_frames = directory, 0
            self.record_message = f'Merekam ke {directory}'
        print(f'Mulai rekam: {directory}', flush=True)
        return dict(recorder=recorder, writer=writer, start=now, dir=directory)

    def _close_recording(self, rec, exc=None):
        recorder = rec['recorder']
        try:
            recorder.__exit__(type(exc) if exc else None, exc, None)
            message = f'Rekaman selesai: {recorder.frames} frame -> {rec["dir"]}'
        except Exception as error:
            message = f'Rekaman bermasalah ({type(error).__name__}: {error}) -> {rec["dir"]}'
        finally:
            self.processor.inbox.put(('close', rec['writer']))
        if exc is not None:
            message = f'Rekaman terhenti karena error ({recorder.frames} frame) -> {rec["dir"]}'
        with self.lock:
            self.record_dir, self.record_message = None, message
        print(message, flush=True)

    def receive(self, rp, sample_rate):
        if not (math.isfinite(sample_rate) and 2 * FILTER_HIGH < sample_rate and
                IDX_MAX / sample_rate * 1e6 > AKRILIK_US[1]):
            raise ValueError('Sample rate perangkat tidak cocok dengan konfigurasi filter/jendela')
        self.sample_rate = sample_rate
        self.processor.inbox.put(('config', sample_rate))
        self.started = time.perf_counter()
        rp._socket.sendall(b'1')
        times, quitting, rec = deque(), False, None
        try:
            while True:
                if self.stop_event.is_set() and not quitting:
                    rp._socket.sendall(b'Q')
                    quitting = True
                magic, sequence, samples, size = struct.unpack('!4I', rp.read_exact(16))
                if sequence != self.frames:
                    raise ValueError(f'Live sequence mismatch: {sequence} != {self.frames}')
                if magic == MAGIC_END:
                    if samples or size:
                        raise ValueError('Invalid live end marker')
                    if not quitting:
                        raise RuntimeError('Live server stopped unexpectedly')
                    break
                if magic != MAGIC_TX_ON or samples != SAMPLES_PER_FRAME or size != BYTES_PER_FRAME:
                    raise ValueError('Invalid live frame header')
                payload = rp.read_exact(size)
                now = time.perf_counter()
                self.frames += 1
                times.append(now)
                while len(times) > 2 and now - times[0] > 1:
                    times.popleft()
                self.rx_fps = (len(times) - 1) / (now - times[0]) if len(times) > 1 and now > times[0] else 0.0

                with self.lock:
                    want = self.record_requested
                if rec and self.args.rekam_detik and now - rec['start'] >= self.args.rekam_detik:
                    want = False
                    with self.lock:
                        self.record_requested = False
                if want and rec is None and not quitting:
                    rec = self._open_recording(sample_rate, now)
                elif not want and rec is not None:
                    self._close_recording(rec)
                    rec = None

                writer = index = elapsed = None
                if rec is not None:
                    index, elapsed, writer = rec['recorder'].frames, now - rec['start'], rec['writer']
                    rec['recorder'].append(index, payload, elapsed)
                    with self.lock:
                        self.record_frames = index + 1
                if self.processor.error is None:
                    counts = np.frombuffer(payload, dtype='<i2', count=IDX_MAX).copy()
                    self.processor.inbox.put(('frame', sequence, now, counts, writer, index, elapsed))
                    backlog = self.processor.inbox.qsize()
                    if backlog > DSP_BACKLOG_WARN >= self.backlog:
                        self.backlog_events += 1
                        print(f'PERINGATAN: DSP tertinggal, antrean {backlog} frame '
                              f'(> {DSP_BACKLOG_WARN}); perekaman tetap lossless.', flush=True)
                    self.backlog = backlog
                    self.backlog_max = max(self.backlog_max, backlog)
                with self.lock:
                    self.status = 'TX ON / AWG_PE'
        except BaseException as exc:
            if rec is not None:
                self._close_recording(rec, exc)
                rec = None
            raise
        finally:
            if rec is not None:
                self._close_recording(rec)


def reprocess(folder, show_plot=True, averages=1, flow_features=False, flow_window=2.0, flow_timebase='sequence'):
    folder = Path(folder)
    averager = EchoProcessor(averages=averages) if averages > 1 else None
    waveforms, metadata = load_recording(folder)
    dsp = EchoDSP(metadata['sample_rate_hz'], secondary_peaks=flow_features)
    nominal_fps = TARGET_FPS
    try:
        nominal_fps = float(json.loads((folder / 'acquisition.json').read_text(encoding='utf-8'))['target_fps'])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    classifier = (FlowRegimeClassifier(flow_window, nominal_fps=nominal_fps if flow_timebase == 'sequence' else None)
                  if flow_features else None)
    if BASELINE is not None and abs(BASELINE.fs - metadata['sample_rate_hz']) > 1:
        raise ValueError('Sample rate rekaman != baseline')
    timing = np.loadtxt(folder / 'frame_times.csv', delimiter=',', skiprows=1, ndmin=2)
    timing_valid = len(timing) == len(waveforms)
    elapsed = timing[:, 1] if timing_valid else np.arange(len(waveforms)) / nominal_fps
    flow_times = elapsed if timing_valid else np.full(len(waveforms), np.nan)
    writer = TofWriter(folder / 'tof_offline.csv', mode='w', flow_features=flow_features)
    (folder / 'analysis_offline.json').write_text(json.dumps(dict(
        dsp=asdict(dsp.config), averages=averages, c_air_m_s=C_AIR,
        baseline=None if BASELINE is None else BASELINE.info,
        flow_features=flow_features, flow_window_s=flow_window, flow_version=2,
        flow_timebase=flow_timebase, flow_nominal_fps=nominal_fps, plate_rejection=PANTULAN_US is not None,
        timing=('no valid frame_times.csv: flow unavailable' if not timing_valid
                else 'frame index / frame rate validated from frame_times.csv' if flow_timebase == 'sequence'
                else 'PC receive time')),
        indent=2), encoding='utf-8')
    tof = np.full(len(waveforms), np.nan)
    try:
        for start in range(0, len(waveforms), 256):
            volts = dsp.to_volts(waveforms[start:start + 256])
            if averager is not None:
                volts = averager._running_average(volts)
            env, echoes = deteksi_batch(dsp, volts)
            for offset, echo in enumerate(echoes):
                index = start + offset
                feature = None
                age = float('nan')
                if classifier is not None:
                    feature = classifier.update(flow_times[index],
                                                tinggi_air_mm(echo.tof_air_us), echo.amp_air,
                                                internal=None if BASELINE is not None else math.isfinite(echo.tof_internal_us),
                                                valid=echo.status == 'OK', sequence=index)
                    age = elapsed[index] - feature.computed_at_s
                writer.write(index, index, elapsed[index], echo, feature, age)
                tof[start + offset] = echo.tof_air_us
    finally:
        writer.close()
    valid = np.isfinite(tof)
    print(f'{len(tof)} frame diproses, air terdeteksi {valid.sum()} ({100 * valid.mean():.1f}%).')
    if valid.any():
        print(f'ToF air: median {np.median(tof[valid]):.3f} µs, '
              f'min {tof[valid].min():.3f}, max {tof[valid].max():.3f}, std {tof[valid].std():.3f}')
        print(f'Tinggi air: median {tinggi_air_mm(np.median(tof[valid])):.3f} mm, '
              f'std {tinggi_air_mm(tof[valid].std()):.4f} mm (c = {C_AIR:.1f} m/s)')
    if THRESHOLD_NOISE is None:
        print(f'Threshold manual {NOISE_THRESHOLD:g}, MA {MA_WINDOW}.')
    else:
        print(f'Threshold otomatis terakhir {dsp.threshold:.3f} (akrilik {dsp.threshold_akrilik:.3f}), MA {MA_WINDOW}, '
              f'noise {NOISE_REF_US[0]:g}-{NOISE_REF_US[1]:g} µs ({dsp.noise_dikecualikan} sampel dikecualikan).')
    if BASELINE is not None:
        print(f'Kecocokan baseline: median residu fit {BASELINE.cocok_rasio:.2f}x baseline '
              f'(> {BASELINE.cocok_faktor:g}x -> BASELINE_TIDAK_COCOK; 0 = mati)')
    if classifier is not None:
        from collections import Counter
        patterns = Counter(r.split(',')[10] for r in (folder / 'tof_offline.csv').read_text(encoding='utf-8').splitlines()[1:])
        print(f"Flow (eksperimental, basis waktu {classifier.timebase}"
              + (f", fps {classifier.fps:.2f}" if classifier.timebase == 'sequence' else '') + '): '
              + ', '.join(f'{k} {v}' for k, v in patterns.most_common()))
        if classifier.timing_reason:
            print(f'Alasan TIMING_UNCERTAIN terakhir: {classifier.timing_reason}')
    print(f'Hasil: {writer.path}')
    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(11, 4))
        plt.plot(elapsed, tof, '.', markersize=2)
        plt.xlabel('Waktu terima PC sejak mulai rekam (s)')
        plt.ylabel('ToF air (µs)')
        plt.title(f'ToF air per frame — {folder.name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(folder / 'tof_offline.png', dpi=150)
        plt.show()
    return tof


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Mass flow ultrasonik real-time 300 FPS + rekam')
    p.add_argument('host', nargs='?', default=IP_ADDRESS)
    p.add_argument('--user', default='root')
    p.add_argument('--host-key', default=os.environ.get('RP_HOST_KEY'), help='verified SSH fingerprint (or RP_HOST_KEY)')
    p.add_argument('--fps', type=float, default=TARGET_FPS, help='target akuisisi (1-300)')
    p.add_argument('--display-fps', type=float, default=DISPLAY_FPS)
    p.add_argument('--timeout', type=float, default=3)
    p.add_argument('--mux-profile', choices=('unchanged', 'single-ch1'), default=MUX_PROFILE)
    p.add_argument('--rekam', action='store_true', help='langsung mulai merekam saat frame pertama masuk')
    p.add_argument('--rekam-detik', type=float, help='hentikan rekaman otomatis setelah N detik')
    p.add_argument('--folder-rekaman', default=str(FOLDER_REKAMAN))
    p.add_argument('--proses-ulang', metavar='FOLDER', help='hitung ulang ToF dari folder rekaman, tanpa perangkat')
    p.add_argument('--tanpa-plot', action='store_true', help='untuk --proses-ulang: jangan tampilkan grafik')
    p.add_argument('--flow-features', action='store_true', help='experimental signal-pattern indicators; not validated flow regimes')
    p.add_argument('--flow-window', type=float, default=2.0, help='flow feature window in seconds (1-10; use 4+ for 0.5 Hz)')
    p.add_argument('--flow-timebase', choices=('sequence', 'receive'), default='sequence',
                   help='sequence (default): time = frame sequence / frame rate validated per window from PC '
                        'receive times (+-5%% of --fps; tolerates socket batching and the 15.625 ms Windows clock, '
                        'otherwise TIMING_UNCERTAIN); receive: PC receive times (rejected when irregular)')
    p.add_argument('--rata', type=int, default=1, metavar='N',
                   help='rata-rata N waveform terakhir sebelum filter (mis. 16); 1 = tanpa rata-rata')
    p.add_argument('--median', type=int, default=1, metavar='N',
                   help='angka ToF tampil = median N deteksi terakhir (mis. 30)')
    p.add_argument('--filter', nargs=2, type=float, metavar=('LOW_MHZ', 'HIGH_MHZ'),
                   help=f'bandpass MHz (default {FILTER_LOW/1e6:g} {FILTER_HIGH/1e6:g})')
    p.add_argument('--blank-us', type=float, help=f'mute awal µs (default {BLANK_US:g})')
    p.add_argument('--akrilik-us', nargs=2, type=float, metavar=('START', 'END'),
                   help=f'jendela cari akrilik µs dari trigger (default {AKRILIK_US[0]:g} {AKRILIK_US[1]:g})')
    p.add_argument('--air-us', nargs=2, type=float, metavar=('START', 'END'),
                   help=f'jendela cari air µs setelah akrilik (default {AIR_MULAI_US:g} {AIR_SELESAI_US:g})')
    p.add_argument('--threshold', type=float,
                   help='threshold TETAP (mode manual = algoritma SCPI lama); tanpa ini threshold otomatis. '
                        'Perilaku lama persis: --ma 128 --threshold 0.15')
    p.add_argument('--ma', type=int, metavar='N', help=f'MA_WINDOW sampel (default {MA_WINDOW}; lama 128)')
    p.add_argument('--threshold-noise', type=float, metavar='K',
                   help=f'threshold otomatis = K x p99,9 puncak noise di --noise-us (default {THRESHOLD_NOISE})')
    p.add_argument('--threshold-akrilik', type=float, metavar='F',
                   help=f'threshold akrilik = F x threshold otomatis (default {THRESHOLD_AKRILIK:g})')
    p.add_argument('--noise-us', nargs=2, type=float, metavar=('START', 'END'),
                   help=f'jendela noise untuk threshold otomatis, µs dari trigger (default '
                        f'{NOISE_REF_US[0]:g} {NOISE_REF_US[1]:g}); bolak-balik ke-2 dan ke-3 echo air dikecualikan otomatis')
    p.add_argument('--pantulan-us', type=float, metavar='US',
                   help='periode pantulan ulang plat akrilik (µs), mis. 3.744 untuk 5 mm; puncak di '
                        'akrilik + k*US diabaikan kecuali > 2x prediksi pantulan')
    p.add_argument('--pantulan-r', type=float, metavar='R',
                   help=f'rasio amplitudo per pantulan untuk --pantulan-us (default {PANTULAN_R:g}, terukur dengan air)')
    p.add_argument('--baseline', metavar='FOLDER',
                   help='lapisan air tipis (< ~2 mm): baseline removal memakai rekaman air TEBAL (>= 10 mm) '
                        'yang direkam tepat sebelumnya tanpa menggeser transduser')
    p.add_argument('--baseline-faktor', type=float, metavar='K',
                   help=f'threshold sisa = K x p99,9 sisa baseline (default {lapisan_tipis.FAKTOR_THR:g})')
    p.add_argument('--baseline-cocok', type=float, metavar='K',
                   help=f'tandai BASELINE_TIDAK_COCOK bila median residu fit > K x baseline '
                        f'(default {lapisan_tipis.COCOK_FAKTOR:g}; 0 = mati, mis. untuk lapisan <= 0,3 mm)')
    p.add_argument('--c-air', type=float, metavar='M_S',
                   help=f'kecepatan suara air untuk h = c*ToF/2 (default {C_AIR:g} m/s, terukur osiloskop)')
    p.add_argument('--suhu-air', type=float, metavar='T',
                   help='suhu air °C -> c dari rumus Marczak (menimpa --c-air), mis. 15 -> 1466 m/s')
    p.add_argument('--preset', choices=sorted(PRESETS),
                   help='kumpulan opsi: akrilik5mm = --ma 32 --threshold-noise 1.2 --pantulan-us 3.744 '
                        '(opsi yang ditulis eksplisit tetap menang)')
    p.add_argument('--scope', nargs='?', const=SCOPE_IP, metavar='IP',
                   help=f'tampilkan juga waveform osiloskop Siglent (default IP {SCOPE_IP})')
    p.add_argument('--scope-ch', nargs='+', default=['C1', 'C2'], help='channel osiloskop yang ditampilkan')
    p.add_argument('--scope-hz', type=float, default=5.0, help='laju ambil waveform osiloskop (default 5/detik)')
    p.add_argument('--scope-setup', action='store_true',
                   help='atur osiloskop otomatis: trigger edge naik di --scope-trigger, 10 µs/div, jendela 0..100 µs')
    p.add_argument('--scope-trigger', default='C1', help='channel osiloskop yang tersambung ke OUT1 (default C1)')
    p.add_argument('--scope-level', type=float, default=0.2, help='level trigger osiloskop (V, default 0.2)')
    args = p.parse_args(argv)
    if not (math.isfinite(args.fps) and 1 <= args.fps <= 300):
        p.error('--fps harus 1..300')
    if not (math.isfinite(args.display_fps) and 0 < args.display_fps <= 60):
        p.error('--display-fps harus 0..60')
    if args.rekam_detik is not None and not (math.isfinite(args.rekam_detik) and args.rekam_detik > 0):
        p.error('--rekam-detik harus positif')
    if not 1 <= args.rata <= 256 or not 1 <= args.median <= 3000:
        p.error('--rata 1..256, --median 1..3000')
    if args.filter and not 0 < args.filter[0] < args.filter[1] < 31:
        p.error('--filter LOW HIGH dalam MHz, 0 < LOW < HIGH < 31')
    for name in ('akrilik_us', 'air_us'):
        window = getattr(args, name)
        if window and not 0 <= window[0] < window[1]:
            p.error(f'--{name.replace("_", "-")} START < END')
    for key, value in PRESETS.get(args.preset, {}).items():
        if getattr(args, key) is None and not (key == 'threshold_noise' and args.threshold is not None):
            setattr(args, key, value)
    if args.ma is not None and not 1 <= args.ma <= 1024:
        p.error('--ma 1..1024')
    if args.threshold_noise is not None and not (math.isfinite(args.threshold_noise) and args.threshold_noise > 0):
        p.error('--threshold-noise harus positif')
    if args.threshold_akrilik is not None and not (math.isfinite(args.threshold_akrilik) and 0 < args.threshold_akrilik <= 2):
        p.error('--threshold-akrilik harus 0..2')
    if args.threshold is not None:
        given = [o for o, v in (('--threshold-noise', args.threshold_noise), ('--threshold-akrilik', args.threshold_akrilik),
                                ('--noise-us', args.noise_us)) if v is not None]
        if given:
            p.error(f'--threshold (manual) tidak bisa digabung dengan {", ".join(given)} (opsi threshold otomatis)')
    if args.noise_us and not (0 <= args.noise_us[0] < args.noise_us[1] - 5
                              and args.noise_us[1] <= IDX_MAX / (125e6 / ACQ_DEC) * 1e6):
        p.error(f'--noise-us START END: lebar >= 5 µs, END <= {IDX_MAX / (125e6 / ACQ_DEC) * 1e6:g} µs')
    if args.pantulan_us is not None and not (math.isfinite(args.pantulan_us) and args.pantulan_us > 2 * PANTULAN_LEBAR_US):
        p.error(f'--pantulan-us harus > {2 * PANTULAN_LEBAR_US:g}')
    if args.pantulan_r is not None and not 0 < args.pantulan_r < 1:
        p.error('--pantulan-r harus 0..1')
    if args.baseline is not None and not (Path(args.baseline) / 'recording.json').is_file():
        p.error(f'--baseline: {args.baseline} bukan folder rekaman (recording.json tidak ada)')
    if args.baseline_faktor is not None and not (math.isfinite(args.baseline_faktor) and args.baseline_faktor > 0):
        p.error('--baseline-faktor harus positif')
    if (args.baseline_faktor is not None or args.baseline_cocok is not None) and args.baseline is None:
        p.error('--baseline-faktor / --baseline-cocok hanya bersama --baseline')
    if args.baseline_cocok is not None and not (math.isfinite(args.baseline_cocok) and args.baseline_cocok >= 0):
        p.error('--baseline-cocok harus >= 0')
    if args.c_air is not None and not (math.isfinite(args.c_air) and 1000 <= args.c_air <= 2000):
        p.error('--c-air harus 1000..2000 m/s')
    if args.suhu_air is not None and not (math.isfinite(args.suhu_air) and 0 <= args.suhu_air <= 95):
        p.error('--suhu-air harus 0..95 °C (rentang rumus Marczak)')
    if not (math.isfinite(args.scope_hz) and 0 < args.scope_hz <= 30):
        p.error('--scope-hz harus 0..30')
    args.scope_ch = [ch.upper() for ch in args.scope_ch]
    if any(ch not in ('C1', 'C2', 'C3', 'C4') for ch in args.scope_ch):
        p.error('--scope-ch hanya C1..C4')
    if args.scope_setup and not args.scope:
        args.scope = SCOPE_IP
    args.frequency, args.cycles, args.amplitude, args.decimation, args.tx = FREQ_HZ, N_CYCLES, AWG_VOLT, ACQ_DEC, 'on'
    return args


def apply_dsp_options(args):
    global FILTER_LOW, FILTER_HIGH, BLANK_US, AKRILIK_US, AIR_MULAI_US, AIR_SELESAI_US, NOISE_THRESHOLD
    global MA_WINDOW, THRESHOLD_NOISE, THRESHOLD_AKRILIK, NOISE_REF_US, PANTULAN_US, PANTULAN_R
    global C_AIR, SUHU_AIR
    if args.c_air is not None:
        C_AIR = args.c_air
    if args.suhu_air is not None:
        SUHU_AIR = args.suhu_air
        C_AIR = c_air_marczak(args.suhu_air)
    if args.ma is not None:
        MA_WINDOW = args.ma
    if args.threshold_noise is not None:
        THRESHOLD_NOISE = args.threshold_noise
    if args.threshold_akrilik is not None:
        THRESHOLD_AKRILIK = args.threshold_akrilik
    if args.noise_us:
        NOISE_REF_US = tuple(args.noise_us)
    if args.pantulan_us is not None:
        PANTULAN_US = args.pantulan_us
    if args.pantulan_r is not None:
        PANTULAN_R = args.pantulan_r
    if args.filter:
        FILTER_LOW, FILTER_HIGH = args.filter[0] * 1e6, args.filter[1] * 1e6
    if args.blank_us is not None:
        BLANK_US = args.blank_us
    if args.akrilik_us:
        AKRILIK_US = tuple(args.akrilik_us)
    if args.air_us:
        AIR_MULAI_US, AIR_SELESAI_US = args.air_us
    if args.threshold is not None:
        NOISE_THRESHOLD = args.threshold
        THRESHOLD_NOISE = None


def flow_plate_warning(flow_features):
    if not flow_features or PANTULAN_US is not None or BASELINE is not None:
        return None
    return ('PERINGATAN: --flow-features tanpa --pantulan-us / --preset akrilik5mm. Pantulan ulang plat '
            'tidak ditolak, sehingga bila echo air lemah (kurang dari ~40% echo akrilik) pantulan plat bisa '
            'terhitung sebagai puncak internal (internal_fraction / INTERNAL_ECHO_CANDIDATE palsu).')


def main(argv=None):
    global BASELINE
    args = parse_args(argv)
    if not math.isfinite(args.flow_window) or not 1 <= args.flow_window <= 10:
        raise SystemExit('--flow-window must be between 1 and 10 seconds')
    apply_dsp_options(args)
    BASELINE = (muat_baseline(args.baseline, rata=args.rata, faktor=args.baseline_faktor, cocok=args.baseline_cocok)
                if args.baseline else None)
    peringatan = flow_plate_warning(args.flow_features)
    if peringatan:
        print(peringatan, flush=True)
    if args.proses_ulang:
        reprocess(args.proses_ulang, show_plot=not args.tanpa_plot, averages=args.rata,
                  flow_features=args.flow_features, flow_window=args.flow_window, flow_timebase=args.flow_timebase)
        return
    if not args.host_key:
        raise SystemExit('Live acquisition requires --host-key or RP_HOST_KEY with the verified SSH fingerprint.')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    plink = shutil.which('plink') or r'C:\Program Files\PuTTY\plink.exe'
    if not Path(plink).is_file():
        raise SystemExit('PuTTY plink.exe diperlukan (sama seperti ultrasonic_film.live)')
    password = os.environ.get('RP_PASSWORD') or getpass.getpass(f'SSH password for {args.user}@{args.host}: ')
    processor = EchoProcessor(averages=args.rata, median_n=args.median,
                              flow_features=args.flow_features, flow_window=args.flow_window,
                              flow_nominal_fps=args.fps if args.flow_timebase == 'sequence' else None)
    worker = MassFlowAcquisition(args, password, plink, processor)
    password = None

    scope = None
    if args.scope:
        from .scope import ScopeReader
        setup = (dict(trigger_ch=args.scope_trigger.upper(), level=args.scope_level)
                 if args.scope_setup else None)
        scope = ScopeReader(args.scope, args.scope_ch, args.scope_hz, setup=setup)

    nominal = EchoDSP(125e6 / ACQ_DEC)
    if scope is None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.subplots_adjust(top=0.95, bottom=0.18 if args.flow_features else 0.14, hspace=0.5)
    else:
        fig = plt.figure(figsize=(12, 11.5))
        grid = fig.add_gridspec(4, 1, top=0.96, bottom=0.18 if args.flow_features else 0.12, hspace=0.6)
        ax1 = fig.add_subplot(grid[0])
        ax_scope = fig.add_subplot(grid[1], sharex=ax1)
        ax2, ax3 = fig.add_subplot(grid[2]), fig.add_subplot(grid[3])

    line_raw, = ax1.plot(nominal.waktu_us, np.zeros(IDX_MAX), color='blue', linewidth=0.5)
    ax1.set_title(f"RAW DATA (frame terbaru, target {args.fps:g} FPS"
                  + (f", rata-rata {args.rata} frame)" if args.rata > 1 else ")"))
    ax1.set_ylabel("Tegangan (V, ≈ count/8192)")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-1.0, 1.0)
    ax1.grid(True, alpha=0.3)

    scope_lines = {}
    if scope is not None:
        warna = {'C1': '#c9a400', 'C2': '#e8710a', 'C3': '#1f77b4', 'C4': '#2ca02c'}
        for ch in args.scope_ch:
            scope_lines[ch], = ax_scope.plot([], [], color=warna[ch], linewidth=0.6, label=ch)
        ax_scope.set_title(f"OSILOSKOP {args.scope} — sumbu waktu sama dengan RAW DATA (0 = trigger)")
        ax_scope.set_xlabel("Waktu dari trigger (µs)")
        ax_scope.set_ylabel("Tegangan (V)")
        ax_scope.grid(True, alpha=0.3)
        ax_scope.legend(loc='upper right')
        ax1.set_xlabel("Waktu dari trigger AWG (µs)")

    line_env, = ax2.plot(nominal.waktu_us, np.zeros(IDX_MAX), color='red', linewidth=1.5)
    ax2.set_title("Real-Time Echo | Menunggu Sinyal..." + (" [lapisan tipis: baseline]" if BASELINE else ""))
    ax2.set_xlabel("Waktu Tempuh Relatif di dalam Air (µs)")
    ax2.set_ylabel("Amplitudo")
    ax2.set_xlim(-5, 40)
    ax2.set_ylim(-0.05, 1.5)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1, label="Batas Akrilik (0 µs)")
    marker_air = ax2.scatter([], [], color='blue', s=60, zorder=5, label='Permukaan Air')
    teks_tof = ax2.text(0.02, 0.80, 'ToF Air: - µs', transform=ax2.transAxes, fontsize=14,
                        fontweight='bold', color='blue',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue'))
    ax2.legend(loc='upper right')

    line_trend, = ax3.plot([], [], '.', color='blue', markersize=2)
    ax3.set_title(f"ToF Air setiap frame — {TREND_DETIK} detik terakhir (h dengan c = {C_AIR:.1f} m/s)")
    ax3.secondary_yaxis('right', functions=(tinggi_air_mm, lambda h: h * 2000.0 / C_AIR)).set_ylabel('h air (mm)')
    ax3.set_xlabel("Waktu (s, relatif terhadap frame terbaru)")
    ax3.set_ylabel("ToF Air (µs)")
    ax3.set_xlim(-TREND_DETIK, 0)
    ax3.grid(True, alpha=0.3)

    tombol = Button(fig.add_axes([0.02, 0.025, 0.14, 0.05]), 'Mulai Rekam')
    status = fig.text(0.18, 0.02, 'Membangun server DMA...', fontsize=9, family='monospace')

    def toggle_record(event):
        requested = worker.recording_snapshot()[0]
        worker.set_recording(not requested)
    tombol.on_clicked(toggle_record)

    state = dict(last=None, shown=0, start=None)

    def refresh():
        latest, history, processed, dsp_fps, dsp_error = processor.snapshot()
        error = worker.error or dsp_error
        requested, rec_dir, rec_frames, rec_message = worker.recording_snapshot()
        tombol.label.set_text('■ Stop Rekam' if requested else '● Mulai Rekam')

        if latest is not None and latest['sequence'] != state['last']:
            state['last'] = latest['sequence']
            state['shown'] += 1
            if state['start'] is None:
                state['start'] = time.perf_counter()
            dsp, echo = latest['dsp'], latest['echo']
            line_raw.set_data(dsp.waktu_us, latest['volts'])
            vmax = float(np.max(np.abs(latest['volts'])))
            ax1.set_ylim(-max(vmax * 1.2, 0.005), max(vmax * 1.2, 0.005))
            vpp = float(np.ptp(latest['volts']))
            state['raw_vpp'] = vpp
            state['raw_clip'] = vmax >= 0.999
            relatif, env = dsp.display_curve(latest['env'], echo)
            line_env.set_data(relatif, env)
            if math.isnan(echo.waktu_akrilik_us):
                ax2.set_title("Real-Time Echo | Akrilik Tidak Terdeteksi (Cek Gel/Kabel)")
            else:
                ax2.set_title(f"Real-Time Echo | Terkalibrasi Akrilik: -{echo.waktu_akrilik_us:.2f} µs")
            if BASELINE is not None:
                ax2.set_title(ax2.get_title() + f" [lapisan tipis, kecocokan baseline {BASELINE.cocok_rasio:.1f}x]")
            if echo.status == 'OK':
                y_air = float(np.interp(echo.tof_air_us, relatif, env)) if BASELINE is not None else echo.amp_air
                marker_air.set_offsets(np.c_[echo.tof_air_us, y_air])
                teks = f"ToF Air: {echo.tof_air_us:.2f} µs  |  h = {tinggi_air_mm(echo.tof_air_us):.2f} mm"
                if args.median > 1 and math.isfinite(latest['tof_median']):
                    teks += (f"\nmedian {args.median}: {latest['tof_median']:.2f} µs  |  "
                             f"h = {tinggi_air_mm(latest['tof_median']):.2f} mm")
                teks_tof.set_text(teks)
            else:
                marker_air.set_offsets(np.empty((0, 2)))
                teks_tof.set_text("Baseline tidak cocok: rekam ulang baseline (transduser bergeser?)"
                                  if echo.status == 'BASELINE_TIDAK_COCOK' else "ToF Air: Kosong/Terlalu Dangkal")
            max_env = float(np.max(env))
            if max_env > ax2.get_ylim()[1]:
                ax2.set_ylim(-0.05, max_env * 1.2)
            if len(history):
                line_trend.set_data(history[:, 0] - latest['received'], history[:, 1])
                lo, hi = history[:, 1].min(), history[:, 1].max()
                pad = max(0.05 * (hi - lo), 0.1)
                ax3.set_ylim(lo - pad, hi + pad)
            else:
                line_trend.set_data([], [])

        scope_line = None
        if scope is not None:
            waves, scope_info, scope_error, scope_fps = scope.snapshot()
            if waves and waves is not state.get('scope'):
                state['scope'] = waves
                lo, hi = math.inf, -math.inf
                for ch, line in scope_lines.items():
                    if ch in waves:
                        t_us, volts = waves[ch]
                        line.set_data(t_us, volts)
                        lo, hi = min(lo, float(volts.min())), max(hi, float(volts.max()))
                    else:
                        line.set_data([], [])
                if math.isfinite(lo):
                    pad = max(0.05 * (hi - lo), 1e-3)
                    ax_scope.set_ylim(lo - pad, hi + pad)
                t_first = min(w[0][0] for w in waves.values())
                t_last = max(w[0][-1] for w in waves.values())
                state['scope_span'] = (t_first, t_last)
            span = state.get('scope_span')
            if scope_error:
                scope_line = scope_error
            elif span:
                scope_line = (f"Osiloskop {scope_info}: {scope_fps:.1f} waveform/s, jendela "
                              f"{span[0]:.1f}..{span[1]:.1f} µs")
                if span[0] > 1 or span[1] < 60:
                    scope_line += '  (atur timebase osiloskop ≈10 µs/div, trigger di kiri, agar sejajar)'
            else:
                scope_line = f'Menghubungkan ke osiloskop {args.scope}...'

        display_fps = (state['shown'] - 1) / max(time.perf_counter() - state['start'], 1e-6) if state['start'] else 0.0
        backlog = processor.inbox.qsize()
        backlog_text = (f" DSP TERTINGGAL (maks {worker.backlog_max}, rekaman tetap lossless)"
                        if backlog > DSP_BACKLOG_WARN else '')
        lines = [f"RX {worker.rx_fps:6.1f} FPS | DSP {dsp_fps:6.1f} FPS (antrean {backlog}{backlog_text}) | "
                 f"Tampilan {display_fps:4.1f} FPS | total {worker.frames} frame, ToF {processed}"
                 + (f" | RAW Vpp {state['raw_vpp']*1e3:.1f} mV" if 'raw_vpp' in state else '')
                 + (' CLIPPING' if state.get('raw_clip') else '')]
        if error:
            lines.append('ERROR: ' + error + ' — periksa terminal.')
        elif rec_dir is not None:
            lines.append(f"● REKAM {rec_frames} frame ({rec_frames * BYTES_PER_FRAME / 1e6:.0f} MB) -> {rec_dir}")
        elif rec_message:
            lines.append(rec_message)
        elif worker.frames == 0:
            lines.append(worker.status)
        if scope_line:
            lines.append(scope_line)
        if latest and latest.get('flow') is not None:
            f = latest['flow']
            internal_text = f'{f.internal_fraction:.0%}' if math.isfinite(f.internal_fraction) else 'unavailable'
            lines.append(f'EXPERIMENTAL {f.pattern} | sigma h {f.sigma_h_mm:.3f} mm | '
                         f'wave {f.f_wave_hz:.2f} Hz | internal {internal_text} (not confirmed bubbles)')
        status.set_text('\n'.join(lines))
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=max(1, int(1000 / args.display_fps)))
    timer.add_callback(refresh)
    fig.canvas.mpl_connect('close_event', lambda event: worker.stop_event.set())

    print("-------------------------------------------------")
    print(f"REAL-TIME ACQUISITION — DMA target {args.fps:g} FPS, ToF dihitung setiap frame")
    print("Klik 'Mulai Rekam' untuk menyimpan. Tutup jendela / Ctrl+C untuk berhenti.")
    print("-------------------------------------------------", flush=True)
    processor.start()
    worker.start()
    if scope is not None:
        scope.start()
    timer.start()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nMenghentikan Sistem Real-Time...")
    finally:
        timer.stop()
        worker.stop_event.set()
        if scope is not None:
            scope.stop_event.set()
        worker.join(timeout=15)
        if worker.is_alive():
            print('Worker belum berhenti; konfirmasi OUTPUT1 OFF pada perangkat.', flush=True)
        processor.inbox.put(('stop',))
        processor.join(timeout=60)
        print(f"Selesai: {worker.frames} frame diterima, {processor.processed} frame diproses DSP, "
              f"air terdeteksi pada {processor.detected} frame.")
        print(f"Antrean DSP maks {worker.backlog_max} frame"
              + (f", {worker.backlog_events}x > {DSP_BACKLOG_WARN} (DSP tertinggal)" if worker.backlog_events else '')
              + '.')
        print("Sistem dihentikan dengan aman.")


if __name__ == '__main__':
    main()
