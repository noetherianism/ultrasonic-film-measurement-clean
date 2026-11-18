"""Offline DSP/feature/CSV throughput; excludes acquisition and GUI overhead."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from ultrasonic_film import massflow as mf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/processing-benchmark.json'))
    args = parser.parse_args()
    fs = 62.5e6
    t = np.arange(16384)/fs*1e6
    pulse = lambda c, a: a*np.exp(-.5*((t-c)/.25)**2)*np.sin(2*np.pi*3.5*(t-c))
    counts = np.rint((pulse(4.5,.25)+pulse(32.1,.1)+pulse(15,.03))*8192).astype(np.int16)
    report = []
    for enabled in (False, True):
        runs = []
        for repeat in range(3):
            p = mf.EchoProcessor(flow_features=enabled)
            p.dsp = mf.EchoDSP(fs, secondary_peaks=enabled)
            with tempfile.TemporaryDirectory() as tmp:
                w = mf.TofWriter(Path(tmp)/'tof.csv', flow_features=enabled)
                try:
                    start = time.perf_counter()
                    for k in range(0, 1216, 64):
                        p._process([('frame', i, i/300, counts, w, i, i/300) for i in range(k,k+64)])
                    w.file.flush()
                    runs.append(time.perf_counter()-start)
                finally:
                    w.close()
        report.append(dict(flow_features=enabled, frames_per_run=1216, elapsed_seconds=runs,
                           median_ms_per_frame=float(np.median(runs)/1216*1000),
                           median_frames_per_second=float(1216/np.median(runs))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
