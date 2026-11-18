"""Offline echo processing independent of acquisition, plotting, and CLI state.

Defaults preserve the original laboratory implementation. Each processor keeps
an immutable configuration snapshot and its own adaptive-noise history.
"""
from collections import deque
from dataclasses import dataclass
import math
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, find_peaks

from .recording import SAMPLES_PER_FRAME


@dataclass(frozen=True)
class EchoConfig:
    """Echo filter/detector settings; time windows are in microseconds."""
    filter_low_hz: float = 3_000_000
    filter_high_hz: float = 4_800_000
    digital_gain: float = 12.0
    ma_window: int = 32
    noise_threshold: float = 0.15
    blank_us: float = 2.5
    acrylic_window_us: tuple[float, float] = (2.5, 6.0)
    water_start_us: float = 1.5
    water_end_us: float = 120.0
    display_tail_us: float = 2.5
    threshold_noise: float | None = 1.2
    threshold_acrylic: float = 0.8
    acrylic_rise: float = 1.5
    noise_window_us: tuple[float, float] = (90.0, 125.0)
    noise_history: int = 600
    noise_exclude_us: tuple[float, float] = (-2.0, 9.5)
    noise_exclude_multiples: tuple[int, ...] = (2, 3)
    plate_period_us: float | None = None
    plate_ratio: float = 0.15
    plate_half_width_us: float = 0.8
    plate_margin: float = 2.0
    sample_count: int = 8000
    volt_per_count: float = 1.0 / 8192
    secondary_peaks: bool = False
    secondary_relative_height: float = 0.15
    secondary_separation_us: float = 0.8
    # Secondary candidates within this distance of samples zeroed by plate rejection are
    # the reverberation envelope's edge left by the hard zeroing, not echoes.
    secondary_plate_edge_us: float = 0.15


class Echo(NamedTuple):
    status: str               # OK / AIR_TIDAK_TERDETEKSI / AKRILIK_TIDAK_TERDETEKSI
    waktu_akrilik_us: float   # NaN jika tidak terdeteksi
    tof_air_us: float         # relatif terhadap puncak akrilik
    amp_air: float
    tof_internal_us: float = float('nan')  # candidate only; not proof of a bubble
    n_peaks: int = 0                      # 0 also means analysis unavailable/off


class EchoDSP:
    def __init__(self, sample_rate, config=None):
        self.config = config if config is not None else EchoConfig()
        if not (math.isfinite(sample_rate) and 2 * self.config.filter_high_hz < sample_rate):
            raise ValueError(f'Sample rate {sample_rate} Hz tidak cocok dengan filter')
        if not (0 < self.config.sample_count <= SAMPLES_PER_FRAME and self.config.ma_window >= 1):
            raise ValueError('IDX_MAX/MA_WINDOW tidak valid')
        self.sample_rate = sample_rate
        # Server DMA menaruh trigger di sampel 0 (setara ACQ_DLY 8192 di SCPI).
        self.waktu_us = np.arange(self.config.sample_count) / sample_rate * 1e6
        # Filter Butterworth orde 8 yang sama; bentuk SOS agar stabil numerik.
        self.sos = butter(8, [self.config.filter_low_hz, self.config.filter_high_hz], btype='band', fs=sample_rate, output='sos')
        self.blank_mask = self.waktu_us < self.config.blank_us
        self.idx_akrilik = np.flatnonzero((self.waktu_us > self.config.acrylic_window_us[0]) & (self.waktu_us < self.config.acrylic_window_us[1]))
        # np.convolve(x, ones(M)/M, 'same') via cumsum: hasil sama, O(N), bisa 2-D.
        self.ma = self.config.ma_window
        k = np.arange(self.config.sample_count) + (self.config.ma_window - 1) // 2
        self._ma_hi = np.minimum(k + 1, self.config.sample_count)
        self._ma_lo = np.maximum(k - self.config.ma_window + 1, 0)
        self.noise_idx = np.flatnonzero((self.waktu_us >= self.config.noise_window_us[0]) & (self.waktu_us < self.config.noise_window_us[1]))
        if self.config.threshold_noise is not None and self.noise_idx.size == 0:
            raise ValueError('NOISE_REF_US di luar jendela IDX_MAX')
        self._noise_peaks = deque(maxlen=self.config.noise_history)
        self.threshold = self.config.noise_threshold   # threshold terakhir yang dipakai (otomatis bila THRESHOLD_NOISE)
        self.threshold_akrilik = self.config.noise_threshold
        self.noise_dikecualikan = 0        # jumlah sampel jendela noise yang dikecualikan pada batch terakhir
        self._akrilik_mulai = int(np.searchsorted(self.waktu_us[self.idx_akrilik], self.config.blank_us))
        self._terakhir = None              # (waktu_akrilik, tof_air) deteksi OK terakhir

    def to_volts(self, counts):
        return np.asarray(counts)[..., :self.config.sample_count].astype(np.float64) * self.config.volt_per_count

    def envelope(self, data):
        """data: volt, bentuk (IDX_MAX,) atau (batch, IDX_MAX)."""
        filtered = sosfiltfilt(self.sos, data, axis=-1) * self.config.digital_gain
        env_raw = np.abs(hilbert(filtered, axis=-1))
        csum = np.zeros(env_raw.shape[:-1] + (self.config.sample_count + 1,))
        np.cumsum(env_raw, axis=-1, out=csum[..., 1:])
        env = (csum[..., self._ma_hi] - csum[..., self._ma_lo]) / self.ma
        self.threshold = self._update_threshold(env)
        self.threshold_akrilik = (self.threshold if self.config.threshold_noise is None
                                  else self.config.threshold_acrylic * self.threshold)
        if self.config.threshold_noise is not None:
            self._buang_ekor_ringing(env)   # sebelum threshold: riak di sekitar threshold tidak dianggap naik
        env[env < min(self.threshold, self.threshold_akrilik)] = 0.0
        env[..., self.blank_mask] = 0.0
        return env

    def _noise_idx_aktif(self):
        """Jendela noise tanpa bolak-balik ke-2 dan ke-3 echo air terakhir (absolut: akrilik + m*ToF).

        Bila lebih dari separuh jendela tertutup (mis. ToF kecil), jendela penuh dipakai agar
        threshold tidak membeku.
        """
        if self._terakhir is None:
            return self.noise_idx
        waktu_akrilik, tof = self._terakhir
        t = self.waktu_us[self.noise_idx]
        keep = np.ones(t.size, dtype=bool)
        for m in self.config.noise_exclude_multiples:
            pusat = waktu_akrilik + m * tof
            keep &= ~((t >= pusat + self.config.noise_exclude_us[0]) & (t <= pusat + self.config.noise_exclude_us[1]))
        return self.noise_idx[keep] if keep.sum() >= 0.5 * t.size else self.noise_idx

    def _update_threshold(self, env):
        """THRESHOLD_NOISE: faktor x p99,9 puncak noise (NOISE_REF_US) dari NOISE_HISTORY frame terakhir.

        p99,9 puncak di 90-125 µs = p99,9 puncak di 30-120 µs (diukur), jadi satu threshold berlaku
        untuk seluruh jendela cari air. Riwayat dipakai karena puncak noise per frame sangat bervariasi.
        Bolak-balik ke-2 echo air (mis. ~115 µs pada air 40 mm) dikecualikan agar threshold tidak naik.
        """
        if self.config.threshold_noise is None:
            return self.config.noise_threshold
        idx = self._noise_idx_aktif()
        self.noise_dikecualikan = self.noise_idx.size - idx.size
        self._noise_peaks.extend(np.atleast_1d(env[..., idx].max(axis=-1)))
        return self.config.threshold_noise * float(np.percentile(self._noise_peaks, 99.9))

    def _buang_ekor_ringing(self, env):
        """Di jendela akrilik (in-place): nolkan ekor ringing TX dari tepi blanking sampai ada kenaikan nyata.

        Kenaikan nyata = envelope > AKRILIK_NAIK x minimum sebelumnya (riak noise pada ekor ringing
        tidak dihitung). Terukur: tanpa akrilik 100% puncak jendela jatuh di tepi blanking (2,51 µs);
        dengan akrilik echo naik 4-8x dari lembah setelah ringing.
        """
        idx = self.idx_akrilik[self._akrilik_mulai:]
        if idx.size == 0:
            return
        a = env[..., idx]
        naik = a > self.config.acrylic_rise * np.minimum.accumulate(a, axis=-1)
        pertama = np.where(naik.any(axis=-1), naik.argmax(axis=-1), idx.size)
        a[np.arange(idx.size) < pertama[..., None]] = 0.0
        env[..., idx] = a

    def _tolak_pantulan(self, area_air, relatif_air, amp_akrilik):
        """Nolkan puncak di akrilik + k*PANTULAN_US (±lebar) yang tidak > margin x amp_akrilik x r^k."""
        k = np.rint(relatif_air / self.config.plate_period_us)
        dekat = (k >= 1) & (np.abs(relatif_air - k * self.config.plate_period_us) < self.config.plate_half_width_us)
        prediksi = self.config.plate_margin * amp_akrilik * self.config.plate_ratio ** np.maximum(k, 1)
        return np.where(dekat & (area_air <= prediksi), 0.0, area_air)

    def detect(self, env):
        """Satu baris envelope -> Echo. Mode manual (--threshold, tanpa opsi lain): identik dengan loop SCPI lama."""
        nan = float('nan')
        otomatis = self.config.threshold_noise is not None
        threshold = self.threshold if otomatis else self.config.noise_threshold
        threshold_akrilik = self.threshold_akrilik if otomatis else self.config.noise_threshold
        area = env[self.idx_akrilik]   # mode otomatis: ekor ringing sudah dibuang di envelope()
        if area.size == 0 or area.max() <= threshold_akrilik:
            return Echo('AKRILIK_TIDAK_TERDETEKSI', nan, nan, nan)
        waktu_akrilik = float(self.waktu_us[self.idx_akrilik[np.argmax(area)]])
        relatif = self.waktu_us - waktu_akrilik
        idx_air = np.flatnonzero((relatif > self.config.water_start_us) & (relatif < self.config.water_end_us))
        if idx_air.size:
            area_air = env[idx_air]
            if self.config.plate_period_us is not None:
                area_air = self._tolak_pantulan(area_air, relatif[idx_air], area.max())
            if area_air.max() > threshold:
                j = idx_air[np.argmax(area_air)]
                self._terakhir = (waktu_akrilik, float(relatif[j]))
                if self.config.secondary_peaks:
                    main = int(np.argmax(area_air))
                    floor = max(threshold, float(area_air[main]) * self.config.secondary_relative_height)
                    separation = max(1, int(round(self.config.secondary_separation_us * self.sample_rate / 1e6)))
                    peaks, _ = find_peaks(area_air, height=floor, prominence=floor * 0.5, distance=separation)
                    if self.config.plate_period_us is not None and len(peaks):
                        # Hard zeroing in _tolak_pantulan leaves the reverberation's envelope
                        # just outside the rejected run; find_peaks sees that edge as a peak
                        # with full prominence. Drop candidates touching a rejected run.
                        rejected = (area_air == 0) & (env[idx_air] > 0)
                        if rejected.any():
                            edge = max(1, int(round(self.config.secondary_plate_edge_us * self.sample_rate / 1e6)))
                            near = np.convolve(rejected, np.ones(2 * edge + 1), mode='same') > 0
                            peaks = peaks[~near[peaks]]
                    # Count only resolved peaks before the selected main echo. The
                    # main echo's identity is inherited; a stronger bubble can still
                    # be mistaken for the surface. Do not report later multiples.
                    internal = peaks[peaks < main - separation]
                    first = float(relatif[idx_air[internal[0]]]) if len(internal) else nan
                    return Echo('OK', waktu_akrilik, float(relatif[j]), float(env[j]), first, len(internal) + 1)
                return Echo('OK', waktu_akrilik, float(relatif[j]), float(env[j]))
        return Echo('AIR_TIDAK_TERDETEKSI', waktu_akrilik, nan, nan)

    def display_curve(self, env, echo):
        """Sumbu waktu relatif akrilik + penghapus pantulan ganda (tampilan saja)."""
        if math.isnan(echo.waktu_akrilik_us):
            return self.waktu_us, env
        relatif = self.waktu_us - echo.waktu_akrilik_us
        if not math.isnan(echo.tof_air_us):
            env = env.copy()
            env[relatif > echo.tof_air_us + self.config.display_tail_us] = 0.0
        return relatif, env
