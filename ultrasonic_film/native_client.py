"""Receive fresh full acquisition frames from the native server on port 5001."""
import argparse
import json
import math
import socket
import struct
import time
import sys
from contextlib import ExitStack
from .recording import FrameRecorder
import numpy as np
from .transport import NativeTransport

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host')
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--record-dir', help='stream every frame to this recording directory')
    parser.add_argument('--output', default='rp_native_result.json')
    parser.add_argument(
        '--plot-frame', type=int,
        help='plot this zero-based frame number after receiving the run')
    parser.add_argument(
        '--sample-rate', type=float, default=125e6,
        help='sample rate in Hz for the plot (default: 125 MHz, decimation 1)')
    parser.add_argument(
        '--plot-output',
        help='also save the plot to this image path, for example waveform.png')
    parser.add_argument('--waveform-output', help='save selected raw samples and timing metadata as NPZ')
    parser.add_argument('--trigger', choices=('NOW', 'AWG_PE'), default='NOW')
    parser.add_argument('--ultrasound', action='store_true', help='add 3–4.8 MHz filter and envelope after acquisition')
    parser.add_argument('--filter-low', type=float, default=3e6)
    parser.add_argument('--filter-high', type=float, default=4.8e6)
    parser.add_argument('--blank-us', type=float, default=2.5)
    args = parser.parse_args()
    if args.plot_frame is not None and args.plot_frame < 0:
        parser.error('--plot-frame must be zero or greater')
    if not math.isfinite(args.sample_rate) or args.sample_rate <= 0:
        parser.error('--sample-rate must be a finite positive number')
    if args.plot_output and args.plot_frame is None:
        parser.error('--plot-output requires --plot-frame')
    if args.waveform_output and args.plot_frame is None:
        parser.error('--waveform-output requires --plot-frame')
    if args.ultrasound:
        if args.trigger != 'AWG_PE' or not all(map(math.isfinite, (args.filter_low, args.filter_high, args.blank_us))) or not (
                0 < args.filter_low < args.filter_high < args.sample_rate / 2 and args.blank_us >= 0):
            parser.error('--ultrasound requires AWG_PE, 0 < filter-low < filter-high < Nyquist, and blank-us >= 0')

    arrivals = []
    frames = 0
    mode = None
    waveform = None
    sample_min, sample_max = 32767, -32768
    with ExitStack() as session:
        recorder = session.enter_context(FrameRecorder(
            args.record_dir, sample_rate_hz=args.sample_rate, trigger=args.trigger,
            host=args.host)) if args.record_dir else None
        rp = NativeTransport(args.host, timeout=10, port=args.port)
        session.callback(rp.close)
        if sys.platform == 'win32':
            rp.enable_quick_ack()
        start = time.perf_counter()
        rp._socket.sendall(b'G')
        while True:
            magic, sequence, samples, size = struct.unpack('!4I', rp.read_exact(16))
            if sequence != frames:
                raise ValueError(f'Sequence mismatch {sequence} != {frames}')
            if magic == 0x52504531:
                if samples or size:
                    raise ValueError('Invalid end marker')
                break
            if magic not in (0x52504631, 0x52504431, 0x52505831) or samples != 16384 or size != 32768:
                raise ValueError('Invalid frame header')
            if mode is not None and mode != magic:
                raise ValueError('Data mode changed mid-run')
            mode = magic
            payload = rp.read_exact(size)
            received = time.perf_counter()-start
            data = np.frombuffer(payload, dtype='<i2')
            if data.size != 16384:
                raise ValueError('Incomplete frame')
            if sequence == args.plot_frame:
                waveform = data.copy()
            if recorder:
                recorder.metadata['backend'] = {0x52504631:'native-standard', 0x52504431:'dma', 0x52505831:'dma-direct'}[mode]
                recorder.metadata['units'] = 'calibrated raw ADC counts' if mode == 0x52504631 else 'DMA raw ADC counts'
                recorder.append(sequence, payload, received)
            frames += 1
            sample_min = min(sample_min,int(data.min()))
            sample_max = max(sample_max,int(data.max()))
            arrivals.append(received)
        if not frames:
            raise RuntimeError('No complete acquisition frames received')
    # With recording enabled this includes flushing files and finalizing the manifest.
    elapsed = time.perf_counter()-start
    result = dict(frames=frames, elapsed_seconds=elapsed, fps=frames/elapsed,
                  payload_MBps=frames*32768/elapsed/1e6, samples_per_frame=16384,
                  channels=1, units=('calibrated RAW' if mode == 0x52504631 else 'DMA RAW') + ' int16 little endian', trigger=args.trigger,
                  backend={0x52504631:'native-standard',0x52504431:'dma',0x52505831:'dma-direct'}.get(mode),
                  sample_min=sample_min, sample_max=sample_max,
                  sequence_valid=True, complete=True)
    result.update(sample_rate_hz=args.sample_rate, selected_frame=args.plot_frame)
    if recorder:
        result.update(recording_directory=str(recorder.directory), frames_saved=recorder.frames,
                      recording_complete=recorder.metadata['complete'])
        print(f'Semua {recorder.frames} waveform tersimpan: {recorder.directory}')
    if args.ultrasound:
        result.update(filter_low_hz=args.filter_low, filter_high_hz=args.filter_high,
                      envelope_blank_us=args.blank_us)
    # Whole one-second arrival windows only. Not a per-frame latency measurement.
    full_seconds = int(arrivals[-1]) if arrivals else 0
    counts = np.histogram(arrivals, bins=np.arange(full_seconds+1))[0] if full_seconds else []
    result['complete_second_frame_counts'] = [int(x) for x in counts]
    with open(args.output, 'w', encoding='utf-8') as out:
        json.dump(result,out,indent=2)
    print(json.dumps(result,indent=2))

    if args.plot_frame is not None:
        if waveform is None:
            raise RuntimeError(
                f'Requested frame {args.plot_frame} was not received; '
                f'the run contained {frames} frames')
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                'Plotting requires matplotlib. Install it with: '
                'python -m pip install matplotlib') from exc

        time_us = np.arange(waveform.size) / args.sample_rate * 1e6
        if args.waveform_output:
            np.savez_compressed(args.waveform_output, raw=waveform, time_us=time_us,
                                sample_rate_hz=args.sample_rate, frame_index=args.plot_frame,
                                trigger=args.trigger, units='raw ADC counts')
            print(f'Waveform tersimpan: {args.waveform_output}')
        if args.ultrasound:
            from scipy.signal import butter, sosfiltfilt, hilbert
            sos = butter(8, [args.filter_low, args.filter_high], btype='band', fs=args.sample_rate, output='sos')
            filtered = sosfiltfilt(sos, waveform.astype(float))
            envelope = np.abs(hilbert(filtered))
            # Match the existing application's transmit blanking interval.
            envelope[time_us < args.blank_us] = 0
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
            axes[0].plot(time_us, waveform, linewidth=0.6)
            axes[0].set_title(f'Ultrasound CH1 — frame {args.plot_frame}, AWG_PE')
            axes[0].set_ylabel('Raw ADC counts')
            axes[1].plot(time_us, filtered, linewidth=0.6,
                         label=f'Bandpass {args.filter_low/1e6:g}–{args.filter_high/1e6:g} MHz')
            axes[1].plot(time_us, envelope, label=f'Envelope (blank < {args.blank_us:g} us)')
            axes[1].set_ylabel('ADC counts (filtered)')
            axes[1].set_xlabel('Waktu dari trigger AWG (us), belum dikalibrasi')
            axes[1].legend()
            for ax in axes:
                ax.grid(True)
        else:
            plt.plot(time_us, waveform)
            plt.xlabel('Waktu (us)')
            plt.ylabel('Amplitudo (raw ADC counts)')
            plt.title(f'Waveform Red Pitaya - frame {args.plot_frame}')
            plt.grid(True)
        plt.tight_layout()
        if args.plot_output:
            plt.savefig(args.plot_output, dpi=150)
            print(f'Plot tersimpan: {args.plot_output}')
        plt.show()

if __name__ == '__main__':
    main()
