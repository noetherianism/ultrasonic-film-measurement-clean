"""Plot any saved waveform from a --save-all run, without connecting to hardware."""
import argparse
import numpy as np
from .recording import load_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', help='run folder containing recording.json and frames.bin')
    parser.add_argument('--frame', type=int, default=0, help='zero-based frame index')
    parser.add_argument('--output', help='optional PNG output')
    args = parser.parse_args()
    waveforms, metadata = load_recording(args.directory)
    if not 0 <= args.frame < len(waveforms):
        parser.error(f'--frame must be between 0 and {len(waveforms)-1}')
    print(f"{len(waveforms)} frames saved, {metadata['samples_per_frame']} samples/frame, "
          f"sample rate {metadata['sample_rate_hz']:g} Hz")
    import matplotlib.pyplot as plt
    time_us = np.arange(waveforms.shape[1]) / metadata['sample_rate_hz'] * 1e6
    plt.figure(figsize=(11, 4))
    plt.plot(time_us, waveforms[args.frame], linewidth=0.6)
    plt.xlabel('Waktu dari trigger (us), belum dikalibrasi')
    plt.ylabel('Raw ADC counts')
    plt.title(f"Frame {args.frame} / {len(waveforms)-1} — {metadata['trigger']}")
    plt.grid(True)
    plt.tight_layout()
    if args.output:
        plt.savefig(args.output, dpi=150)
    plt.show()


if __name__ == '__main__':
    main()
