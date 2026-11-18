# -*- coding: utf-8 -*-
"""
Pengukuran lapisan air tipis dengan baseline removal (Al-Aufi dkk. 2019, Flow Meas. Instrum. 66:67-78, §2.2).

Untuk lapisan < ~0,7 mm echo permukaan air bertumpuk dengan echo akrilik, jadi peak-picking ToF biasa
tidak bisa. Caranya:
  1. Baseline = waveform rata-rata (sejajar echo akrilik) dari rekaman air TEBAL (>= 10 mm), hanya di
     jendela sekitar echo akrilik + ekornya; echo air baseline harus di luar jendela.
  2. Per frame: sejajarkan baseline ke tepi naik echo akrilik frame itu (cross-correlation ternormalisasi,
     sub-sampel), skala amplitudo dengan least-squares di jendela yang sama, lalu kurangkan.
  3. Sisa = echo permukaan air (+ pantulan berulangnya tiap 2*delta/c). ToF = puncak pertama envelope
     sisa yang melewati threshold - puncak echo akrilik; h = c*ToF/2 (dihitung pemanggil).
  4. Threshold sisa per waktu relatif = faktor x p99,9 envelope sisa pada frame baseline yang tidak dipakai
     membuat template, jadi ketidakcocokan baseline (jitter, amplitudo) ikut diperhitungkan.

Jalur ini memakai filter sendiri (FILTER_TIPIS, orde rendah): bandpass program (orde 8, 3-4,8 MHz,
zero-phase) punya respons impuls +-1,5 µs sehingga echo air "bocor" ke tepi naik echo akrilik dan
membuat skala least-squares bias.

Dipakai oleh python -m ultrasonic_film --baseline FOLDER.
"""

import math
from collections import deque

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

REL_US       = (-1.5, 4.0)    # jendela baseline relatif puncak akrilik (µs)
FIT_US       = (-0.6, -0.2)   # tepi naik akrilik; overlap/pre-ringing tetap bergantung bentuk pulsa (lihat audit)
CARI_US      = (0.3, 3.0)     # cari echo air relatif puncak akrilik (< pantulan plat 3,744 µs)
GESER_US     = 0.25           # rentang cross-correlation (µs)
MA_SISA      = 8              # moving average envelope sisa (sampel; 0,13 µs pada 62,5 MHz)
FAKTOR_THR   = 1.5            # threshold sisa = faktor x p99,9 envelope sisa baseline
RELATIF_MAKS = 0.5            # puncak pertama harus >= ini x puncak sisa terbesar (echo air pertama =
                              # terbesar di deretnya; lobus-depan filter / sisa kecil tidak dihitung)
FILTER_TIPIS = (1.5e6, 5.0e6)
ORDE_TIPIS   = 4
PAD_US       = 3.0            # sinyal difilter di jendela +- pad ini (tepi filter jauh dari jendela)
COCOK_FAKTOR = 3.0            # median residu fit (COCOK_N frame) > ini x median baseline -> BASELINE_TIDAK_COCOK
COCOK_N      = 64             # (terukur, rata 16: transduser dipasang ulang 4,8x; sesi sama 1,2-1,6x).
                              # Lapisan <= 0,3 mm dengan echo kuat juga menimpa jendela fit -> ikut ditandai.
MIN_AIR_BASELINE_MM = 10.0    # baseline harus air tebal
C_CEK        = 1466.0         # hanya untuk cek ketebalan baseline


def _ma(x, n):
    if n <= 1:
        return x
    k = np.ones(n) / n
    return np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), -1, x)


def _parabola(y, k):
    """Puncak sub-sampel di sekitar indeks k."""
    if k <= 0 or k >= len(y) - 1:
        return float(k)
    a, b, c = y[k - 1], y[k], y[k + 1]
    d = a - 2 * b + c
    return k + (0.5 * (a - c) / d if d else 0.0)


def _geser(sig, s):
    """Tunda sinyal s sampel (pecahan, boleh negatif) dengan interpolasi linear; tepi diisi 0."""
    k = np.arange(sig.size)
    return np.interp(k - s, k, sig, left=0.0, right=0.0)


def rata_geser(x, n):
    """Rata-rata N frame terakhir (sama dengan EchoProcessor); frame awal memakai yang tersedia."""
    if n <= 1:
        return x
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - n, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


class BaselineTipis:
    """Template echo akrilik + threshold sisa; proses(volt) memberi ToF air tipis per frame."""

    def __init__(self, template, fs, t_akrilik_us, i0, threshold, faktor_thr=FAKTOR_THR,
                 filter_hz=FILTER_TIPIS, fit_sisa_med=None, cocok_faktor=COCOK_FAKTOR, info=None):
        self.template = np.asarray(template, dtype=float)
        self.fs = fs
        self.dt_us = 1e6 / fs
        self.filter_hz = tuple(filter_hz)
        self.sos = butter(ORDE_TIPIS, self.filter_hz, btype='band', fs=fs, output='sos')
        self.pad = int(round(PAD_US / self.dt_us))
        self.t_akrilik_us = t_akrilik_us                    # puncak envelope template (absolut, sub-sampel)
        self.i0 = int(i0)                                    # indeks sampel absolut awal jendela
        self.n = self.template.size
        self.rel = (self.i0 + np.arange(self.n)) * self.dt_us - t_akrilik_us
        self.fit = np.flatnonzero((self.rel >= FIT_US[0]) & (self.rel <= FIT_US[1]))
        self.cari = np.flatnonzero((self.rel >= CARI_US[0]) & (self.rel <= CARI_US[1]))
        self.lag = int(round(GESER_US / self.dt_us))
        self.threshold_dasar = np.asarray(threshold, dtype=float)   # p99,9 envelope sisa per sampel jendela
        self.faktor_thr = faktor_thr
        self.fit_sisa_med = fit_sisa_med          # median residu fit ternormalisasi pada frame baseline
        self.cocok_faktor = cocok_faktor          # 0 = pemeriksaan kecocokan mati
        self._fit_sisa = deque(maxlen=COCOK_N)
        self.cocok_rasio = float('nan')           # median residu fit terbaru / fit_sisa_med
        self.info = dict(info or {})

    @property
    def threshold(self):
        return self.faktor_thr * self.threshold_dasar

    def saring(self, volts):
        """volt mentah (frame, sampel) -> sinyal FILTER_TIPIS di jendela baseline (frame, n)."""
        x = np.atleast_2d(volts)
        lo, hi = max(0, self.i0 - self.pad), min(x.shape[1], self.i0 + self.n + self.pad)
        y = sosfiltfilt(self.sos, x[:, lo:hi], axis=-1)
        return y[:, self.i0 - lo:self.i0 - lo + self.n]

    # ---------- pembuatan dari rekaman air tebal
    @classmethod
    def dari_rekaman(cls, volts, fs, t_akrilik_kasar_us, tof_air_us, rata=1, faktor_thr=FAKTOR_THR,
                     filter_hz=FILTER_TIPIS, cocok_faktor=COCOK_FAKTOR, info=None):
        """volts: (frame, sampel) volt mentah rekaman air tebal (tanpa lapisan tipis).

        Frame genap -> template, frame ganjil -> threshold sisa (tidak ikut membuat template).
        """
        dt_us = 1e6 / fs
        if not (math.isfinite(tof_air_us) and C_CEK * tof_air_us / 2000 >= 0.9 * MIN_AIR_BASELINE_MM):
            raise ValueError(f'Baseline harus air tebal (>= {MIN_AIR_BASELINE_MM:g} mm); '
                             f'ToF air baseline {tof_air_us:.2f} µs')
        if tof_air_us <= REL_US[1] + 1.0:
            raise ValueError('Echo air baseline masuk jendela baseline')
        i_ak = int(round(t_akrilik_kasar_us / dt_us))
        i0 = i_ak + int(math.floor(REL_US[0] / dt_us))
        n = int(round((REL_US[1] - REL_US[0]) / dt_us))
        tmp = cls(np.zeros(n), fs, t_akrilik_kasar_us, i0, np.zeros(n), filter_hz=filter_hz)
        x = rata_geser(np.asarray(volts, dtype=float), rata)
        seg = tmp.saring(x)
        genap, ganjil = seg[0::2], seg[1::2]
        tpl = genap.mean(axis=0)
        for _ in range(3):   # sejajarkan tiap frame ke template, rata-ratakan lagi
            tmp = cls(tpl, fs, cls._puncak(tpl, i0, dt_us), i0, np.zeros(n), filter_hz=filter_hz)
            s = tmp._sejajar(genap)[0]
            tpl = np.mean([_geser(row, -si) for row, si in zip(genap, s)], axis=0)
        t_ak = cls._puncak(tpl, i0, dt_us)
        tmp = cls(tpl, fs, t_ak, i0, np.zeros(n), filter_hz=filter_hz)
        env, _, _, rf = tmp._envelope_sisa(ganjil)
        p = np.percentile(env, 99.9, axis=0)
        w = max(1, int(round(0.1 / dt_us)))
        p = np.array([p[max(0, i - w):i + w + 1].max() for i in range(n)])   # dihaluskan: maks +-0,1 µs
        info = dict(info or {}, frame=len(x), rata=rata, tof_air_baseline_us=float(tof_air_us),
                    t_akrilik_us=float(t_ak), rel_us=REL_US, fit_us=FIT_US, cari_us=CARI_US,
                    geser_us=GESER_US, ma_sisa=MA_SISA, filter_hz=list(filter_hz), orde=ORDE_TIPIS,
                    faktor_thr=faktor_thr, relatif_maks=RELATIF_MAKS, amp_akrilik=float(np.abs(hilbert(tpl)).max()),
                    sisa_p999_maks=float(p[tmp.cari].max()), sisa_p999_med=float(np.median(p[tmp.cari])),
                    fit_sisa_med=float(np.median(rf)), cocok_faktor=cocok_faktor, cocok_n=COCOK_N)
        return cls(tpl, fs, t_ak, i0, p, faktor_thr=faktor_thr, filter_hz=filter_hz,
                   fit_sisa_med=float(np.median(rf)), cocok_faktor=cocok_faktor, info=info)

    @staticmethod
    def _puncak(tpl, i0, dt_us):
        e = _ma(np.abs(hilbert(tpl)), MA_SISA)
        return (i0 + _parabola(e, int(np.argmax(e)))) * dt_us

    # ---------- inti
    def _sejajar(self, seg):
        """seg: (frame, n) terfilter. Geser sub-sampel tiap frame (sampel) + skala LS di jendela FIT."""
        seg = np.atleast_2d(seg)
        lags = np.arange(-self.lag, self.lag + 1)
        f = self.fit
        if f.size == 0 or f[0] - self.lag < 0 or f[-1] + self.lag >= self.n:
            raise ValueError('Jendela fit terlalu dekat tepi jendela baseline')
        # ternormalisasi: tanpa ini lag condong ke bagian echo yang lebih besar (energi), bukan bentuk
        cc = np.stack([(seg[:, f + L] @ self.template[f]) / (np.linalg.norm(seg[:, f + L], axis=1) + 1e-12)
                       for L in lags], axis=1)
        k = np.clip(np.argmax(cc, axis=1), 1, len(lags) - 2)
        r = np.arange(len(seg))
        a, b, c = cc[r, k - 1], cc[r, k], cc[r, k + 1]
        d = a - 2 * b + c
        s = lags[k] + 0.5 * (a - c) / np.where(d == 0, 1, d)
        tpl = np.array([_geser(self.template, si) for si in s])
        skala = np.einsum('ij,ij->i', seg[:, f], tpl[:, f]) / np.einsum('ij,ij->i', tpl[:, f], tpl[:, f])
        return s, skala, tpl

    def _envelope_sisa(self, seg):
        """-> envelope sisa, geser, skala, residu fit ternormalisasi (kecocokan baseline) per frame."""
        s, skala, tpl = self._sejajar(seg)
        sisa = np.atleast_2d(seg) - skala[:, None] * tpl
        f = self.fit
        rf = np.linalg.norm(sisa[:, f], axis=1) / (np.abs(skala) * np.linalg.norm(tpl[:, f], axis=1) + 1e-12)
        return _ma(np.abs(hilbert(sisa, axis=-1)), MA_SISA), s, skala, rf

    def proses(self, volts):
        """volts: (frame, sampel) atau (sampel,) volt mentah, sumbu waktu sama dengan baseline.

        -> list (status, waktu_akrilik_us, tof_air_us, amp_sisa).
        """
        env, s, skala, rf = self._envelope_sisa(self.saring(volts))
        thr = self.threshold
        c = self.cari
        hasil = []
        nan = float('nan')
        for e, si, sk, r in zip(env, s, skala, rf):
            t_ak = self.t_akrilik_us + si * self.dt_us
            self._fit_sisa.append(r)
            if self.fit_sisa_med:
                self.cocok_rasio = float(np.median(self._fit_sisa)) / self.fit_sisa_med
            # transduser bergeser / coupling berubah (atau lapisan < ~0,26 mm menimpa jendela fit)
            # Berlaku sejak frame pertama (median sampel yang ada): frame awal tidak boleh lolos tanpa
            # cek; bila ragu lebih aman ditandai tidak cocok daripada dilaporkan sebagai lapisan.
            tidak_cocok = bool(self.cocok_faktor) and self.cocok_rasio > self.cocok_faktor
            if tidak_cocok or not (abs(si) < self.lag - 1 and 0.5 < sk < 2.0):
                hasil.append(('BASELINE_TIDAK_COCOK', t_ak, nan, nan))
                continue
            atas = np.flatnonzero((e[c] > thr[c]) & (e[c] >= RELATIF_MAKS * e[c].max()))
            if atas.size == 0 or not (e[c] > thr[c]).any():
                hasil.append(('AIR_TIPIS_TIDAK_TERDETEKSI', t_ak, nan, nan))
                continue
            j = c[atas[0]]
            while j + 1 < self.n and e[j + 1] >= e[j]:   # naik ke puncak lokal pertama
                j += 1
            t_pk = (self.i0 + _parabola(e, j)) * self.dt_us
            hasil.append(('OK', t_ak, t_pk - t_ak, float(e[j])))
        return hasil
