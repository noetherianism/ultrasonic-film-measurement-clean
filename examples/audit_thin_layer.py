"""Reproducible synthetic filter/fit audit; never opens hardware or real recordings.

The controlled fit uses the known acrylic position to isolate amplitude bias.
It is not an end-to-end comparison of alternative thin-film detectors. Filter
settings remain experiments here; production BaselineTipis defaults are unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, iirnotch, sosfilt, sosfiltfilt, tf2sos

from ultrasonic_film.thin_layer import BaselineTipis

FS = 62.5e6
C = 1466.


def audit():
    t = np.arange(8000) / FS * 1e6
    center = 25.
    band = butter(4, (1.5e6, 5e6), btype='band', fs=FS, output='sos')
    notch = tf2sos(*iirnotch(5.96e6, 30., fs=FS))
    hp = butter(1, 0.5e6, btype='high', fs=FS, output='sos')
    filters = {'raw': lambda x: x, 'causal_highpass': lambda x: sosfilt(hp, x),
               'causal_bandpass': lambda x: sosfilt(band, x),
               'zero_phase_bandpass': lambda x: sosfiltfilt(band, x),
               'zero_phase_bandpass_notch': lambda x: sosfiltfilt(np.vstack((band, notch)), x)}
    rows = []
    rng = np.random.default_rng(20261019)
    # Different pulse widths/frequencies matter more than a single ideal example.
    for width in (0.15, 0.25, 0.4):
        for frequency in (3.2, 5.0):
            pulse = lambda offset: .3*np.exp(-.5*((t-offset)/width)**2)*np.sin(2*np.pi*frequency*(t-offset))
            reference = pulse(center)
            interference = .02*np.sin(2*np.pi*5.96*t) + rng.normal(0, .001, len(t))
            for name, filt in filters.items():
                ref = filt(reference)
                anchor = t[np.argmax(np.abs(hilbert(ref)))]
                fit = (t-anchor >= -.6) & (t-anchor <= -.2)
                mask = (t > anchor-4) & (t < anchor+4)
                env = np.abs(hilbert(ref))
                half = np.flatnonzero(mask & (env >= env.max()/2))
                fwhm = float(t[half[-1]]-t[half[0]])
                filtered_noise = filt(interference)
                quiet = (t > 60) & (t < 110)
                noise_rms = float(np.sqrt(np.mean(filtered_noise[quiet]**2)))
                for h in (.3, .5):
                    leaked = filt(.8*pulse(center+2000*h/C))
                    ratio = float(np.linalg.norm(leaked[fit])/np.linalg.norm(ref[fit]))
                    bias = float(np.dot(leaked[fit], ref[fit])/np.dot(ref[fit], ref[fit]))
                    rows.append(dict(filter=name, width_us=width, carrier_mhz=frequency,
                                     h_mm=h, fit_leakage_ratio=ratio, ls_scale_bias_percent=100*bias,
                                     pulse_fwhm_us=fwhm, anchor_delay_us=float(anchor-center),
                                     noise_rms_v=noise_rms,
                                     peak_to_noise_db=float(20*np.log10(env.max()/noise_rms)),
                                     tail_energy_fraction=float(np.sum(ref[np.abs(t-anchor)>1]**2)/np.sum(ref**2))))
    # Strictly causal compact-support probe: distinguish filter pre-ringing from
    # the physical leading tail already present in a Gaussian acoustic pulse.
    compact = np.zeros_like(t)
    active = (t >= center) & (t < center+.5)
    compact[active] = np.sin(2*np.pi*3.2*(t[active]-center))*np.hanning(active.sum())
    pre = {name: float(np.linalg.norm(filt(compact)[t<center])/np.linalg.norm(filt(compact)))
           for name, filt in filters.items()}
    # Run the actual unchanged detector with independent synthetic baseline/test
    # realizations, including shape stretch and no-layer false detections.
    def frames(n, seed, h=None, stretch=1.):
        rand = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            shifted = (t - 4.5 - rand.normal(0,.02))/stretch
            rf = .3*np.exp(-.5*(shifted/.25)**2)*np.sin(2*np.pi*3.2*shifted)
            rf += .045*np.exp(-.5*((shifted-3.744)/.25)**2)*np.sin(2*np.pi*3.2*(shifted-3.744))
            if h is not None:
                delta = shifted-2000*h/C
                rf += .24*np.exp(-.5*(delta/.25)**2)*np.sin(2*np.pi*3.2*delta)
            else:
                delta = t-32.1
                rf += .25*np.exp(-.5*(delta/.25)**2)*np.sin(2*np.pi*3.2*delta)
            out.append(rf+rand.normal(0,.004,len(t)))
        return np.array(out)
    baseline = frames(128, 1)
    detector = []
    for stretch in (.998, 1., 1.002):
        for h in (None, .3, .5):
            b = BaselineTipis.dari_rekaman(baseline, FS, 4.5, 27.6)
            result = b.proses(frames(64, 2, h, stretch))
            values = [C*r[2]/2000 for r in result if r[0]=='OK']
            detector.append(dict(stretch=stretch, h_mm=h, ok_fraction=len(values)/len(result),
                                 median_h_mm=float(np.median(values)) if values else None,
                                 median_error_mm=float(np.median(values)-h) if values and h is not None else None,
                                 baseline_mismatch_fraction=sum(r[0]=='BASELINE_TIDAK_COCOK' for r in result)/len(result)))
    return dict(synthetic_only=True, sample_rate_hz=FS, seed=20261019,
                caveat='Fit audit uses known position; raw/causal are not validated replacement detectors. Stretch is a shape perturbation, not a calibrated temperature.',
                fit_and_filter=rows, causal_probe_pre_energy_norm_ratio=pre,
                unchanged_detector=detector)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/thin-layer-audit.json'))
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Synthetic audit saved to {args.output}')
    for row in result['unchanged_detector']:
        print(row)


if __name__ == '__main__':
    main()
