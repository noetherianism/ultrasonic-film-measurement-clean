"""Generate deterministic synthetic echoes for an offline recording/replay demo.

This is not measured data or evidence of physical thickness accuracy.
Run after installing the project: python examples/create_demo_recording.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from ultrasonic_film.recording import FrameRecorder, SAMPLES_PER_FRAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('recordings/demo'))
    parser.add_argument('--frames', type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.frames <= 10000:
        parser.error('--frames must be between 1 and 10000')
    if args.output.exists():
        parser.error('Output already exists; choose a new directory')
    fs = 62_500_000
    time_us = np.arange(SAMPLES_PER_FRAME) / fs * 1e6
    rng = np.random.default_rng(20261019)

    def echo(center, amplitude):
        delta = time_us - center
        return amplitude * np.exp(-0.5 * (delta / 0.35) ** 2) * np.cos(2 * np.pi * 3.5 * delta)

    with FrameRecorder(args.output, sample_rate_hz=fs, trigger='SYNTHETIC', host='simulation') as recorder:
        for index in range(args.frames):
            volts = echo(4.5, 0.25) + echo(32.1, 0.12)
            volts += rng.normal(0, 0.001, SAMPLES_PER_FRAME)
            counts = np.clip(np.rint(volts * 8192), -8192, 8191).astype('<i2')
            recorder.append(index, counts.tobytes(), index / 300)
    (args.output / 'acquisition.json').write_text(json.dumps({
        'synthetic': True, 'description': 'Two Gaussian-windowed echoes plus seeded noise',
        'sample_rate_hz': fs, 'frames': args.frames,
        'note': 'No hardware timing, measurement accuracy or thin-film resolution is validated.'
    }, indent=2), encoding='utf-8')
    print(f'Created {args.frames} synthetic frames in {args.output}')
    print(f'Next: film-live --proses-ulang "{args.output}" --tanpa-plot')


if __name__ == '__main__':
    main()
