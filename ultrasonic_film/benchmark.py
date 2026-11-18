"""Build and run temporary native Red Pitaya benchmark over SSH, receive on PC."""
import argparse
import getpass
import json
import math
import os
import re
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
from datetime import datetime

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('host', nargs='?', default=os.environ.get('RP_HOST', 'redpitaya.local'),
                   help='device hostname (default: RP_HOST or redpitaya.local)')
    p.add_argument('--seconds', type=float, default=30)
    p.add_argument('--fps', type=float, help='default: maximum for benchmark, 300 for ultrasound')
    p.add_argument('--mode', choices=['standard','dma','direct'], default='direct')
    p.add_argument('--user', default='root')
    p.add_argument('--host-key', default=os.environ.get('RP_HOST_KEY'),
                   help='verified SSH host fingerprint (default: RP_HOST_KEY)')
    p.add_argument('--output', help='result JSON (default: rp_benchmark_result.json, or inside the recording folder)')
    p.add_argument('--plot-frame', type=int, help='plot this zero-based frame number')
    p.add_argument('--sample-rate', type=float,
                   help='optional expected sample rate; must match device readback')
    p.add_argument('--plot-output', help='also save the waveform plot to this image path')
    p.add_argument('--ultrasound', action='store_true', help='emit CH1 sine bursts and trigger ADC CH1 with AWG_PE')
    p.add_argument('--frequency', type=float, default=5e6, help='burst frequency in Hz (default: 5 MHz)')
    p.add_argument('--cycles', type=int, default=1, help='sine cycles per burst (default: 1)')
    p.add_argument('--amplitude', type=float, default=1.0, help='generator amplitude in V, not Vpp (default: 1)')
    p.add_argument('--decimation', type=int, choices=(1, 2, 4, 8), help='default: 2 for ultrasound, 1 otherwise')
    p.add_argument('--mux-profile', choices=('unchanged', 'single-ch1'), default='unchanged',
                   help='single-ch1 applies massflo DIO0..4_P = 0,0,0,0,1')
    p.add_argument('--waveform-output', help='save selected raw waveform and timing metadata as NPZ')
    p.add_argument('--filter-low', type=float, default=3e6, help='ultrasound plot bandpass lower edge in Hz')
    p.add_argument('--filter-high', type=float, default=4.8e6, help='ultrasound plot bandpass upper edge in Hz')
    p.add_argument('--blank-us', type=float, default=2.5, help='blank envelope before this time in microseconds')
    p.add_argument('--save-all', action='store_true', help='save every received waveform in a new run folder')
    p.add_argument('--recordings-dir', default='recordings', help='parent folder for --save-all runs (default: recordings)')
    args=p.parse_args()
    if args.fps is None:
        args.fps = 300 if args.ultrasound else 0
    if args.decimation is None:
        args.decimation = 2 if args.ultrasound else 1
    if args.ultrasound:
        if args.plot_frame is None:
            args.plot_frame = 0
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 120:
        p.error('--seconds must be in (0,120]')
    if not math.isfinite(args.fps) or args.fps < 0:
        p.error('--fps must be finite and nonnegative')
    if args.plot_frame is not None and args.plot_frame < 0:
        p.error('--plot-frame must be zero or greater')
    if args.sample_rate is not None and (not math.isfinite(args.sample_rate) or args.sample_rate <= 0):
        p.error('--sample-rate must be finite and positive')
    if args.plot_output and args.plot_frame is None:
        p.error('--plot-output requires --plot-frame')
    if args.waveform_output and args.plot_frame is None:
        p.error('--waveform-output requires --plot-frame')
    if args.mux_profile != 'unchanged' and not args.ultrasound:
        p.error('--mux-profile requires --ultrasound')
    if args.ultrasound:
        if args.mode == 'standard' or args.fps <= 0:
            p.error('--ultrasound requires DMA/direct mode and positive --fps')
        if not math.isfinite(args.frequency) or not 1 <= args.frequency <= 50e6:
            p.error('--frequency must be in [1, 50000000] Hz')
        if not 1 <= args.cycles <= 1000:
            p.error('--cycles must be in [1, 1000]')
        if not math.isfinite(args.amplitude) or not 0 < args.amplitude <= 1:
            p.error('--amplitude must be in (0, 1] V')
        if not all(map(math.isfinite, (args.filter_low, args.filter_high, args.blank_us))) or not (
                0 < args.filter_low < args.filter_high and args.blank_us >= 0):
            p.error('require 0 < --filter-low < --filter-high and --blank-us >= 0')
    if not args.host_key:
        p.error('Set --host-key or RP_HOST_KEY to the verified SSH host fingerprint')
    plink=shutil.which('plink') or r'C:\Program Files\PuTTY\plink.exe'
    if not Path(plink).is_file():
        p.error('PuTTY plink.exe is required')
    recording_dir = None
    if args.save_all:
        parent = Path(args.recordings_dir).resolve()
        parent.mkdir(parents=True, exist_ok=True)
        recording_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=parent))
        parameters = dict(host=args.host, target_fps=args.fps, requested_seconds=args.seconds,
                          ultrasound=args.ultrasound, mode=args.mode, decimation=args.decimation,
                          burst_frequency_hz=args.frequency if args.ultrasound else None,
                          burst_cycles=args.cycles if args.ultrasound else None,
                          generator_amplitude_v=args.amplitude if args.ultrasound else None,
                          mux_profile=args.mux_profile, extra_warmup_shots=5)
        (recording_dir/'acquisition.json').write_text(json.dumps(parameters, indent=2), encoding='utf-8')
        print(f'Folder rekaman: {recording_dir}', flush=True)
    output_folder = recording_dir or Path('.')
    args.output = args.output or str(output_folder/'rp_benchmark_result.json')
    if args.ultrasound:
        args.plot_output = args.plot_output or str(output_folder/'ultrasound_waveform.png')
        args.waveform_output = args.waveform_output or str(output_folder/'ultrasound_waveform.npz')
    root=Path(__file__).resolve().parent
    password=getpass.getpass(f'SSH password for {args.user}@{args.host}: ')
    # PuTTY reads a temporary password file, so the password is not in argv.
    process=None
    log=[]
    with tempfile.TemporaryDirectory(prefix='rp-benchmark-') as temporary:
        pwfile=Path(temporary)/'password'
        pwfile.write_text(password+'\n',encoding='utf-8')
        password=None
        command=[plink,'-ssh','-batch','-hostkey',args.host_key,'-pwfile',str(pwfile),f'{args.user}@{args.host}']
        build=('mkdir -p /tmp/ultrasonic-rp-fps && cat > /tmp/ultrasonic-rp-fps/benchmark.cpp && '
               'g++ -O3 -std=c++20 /tmp/ultrasonic-rp-fps/benchmark.cpp -I/opt/redpitaya/include '
               '-L/opt/redpitaya/lib -Wl,-rpath,/opt/redpitaya/lib -lrp -pthread -o /tmp/ultrasonic-rp-fps/benchmark')
        print('Building temporary benchmark on Red Pitaya...',flush=True)
        subprocess.run(command+[build],input=(root/'native/rp_acquisition.cpp').read_bytes(),check=True,timeout=120)
        acquisition = 'ultrasound' if args.ultrasound else 'benchmark'
        remote=(f'/tmp/ultrasonic-rp-fps/benchmark {args.seconds} {args.mode} {args.fps} '
                f'{acquisition} {args.frequency} {args.cycles} {args.amplitude} '
                f'{args.decimation} {args.mux_profile}')
        process=subprocess.Popen(command+[remote],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        lines=queue.Queue()
        def reader():
            for line in process.stdout:
                log.append(line.rstrip())
                print(line,end='',flush=True)
                lines.put(line)
            lines.put(None)
        thread=threading.Thread(target=reader,daemon=True)
        thread.start()
        try:
            while True:
                line=lines.get(timeout=60)
                if line is None:
                    raise RuntimeError('Server exited before READY')
                if line.startswith('READY '): break
            rate_match = re.search(r'\bsample_rate=([0-9.]+)', line)
            if rate_match is None:
                raise RuntimeError('Server did not report sample rate')
            sample_rate = float(rate_match.group(1))
            if args.sample_rate is not None and not math.isclose(args.sample_rate, sample_rate, rel_tol=1e-6):
                raise RuntimeError(f'Expected sample rate {args.sample_rate}, device reports {sample_rate}')
            if args.ultrasound and args.filter_high >= sample_rate / 2:
                raise RuntimeError('Bandpass upper edge must be below device Nyquist frequency')
            client_command = [
                sys.executable, '-m', 'ultrasonic_film.native_client', args.host,
                '--output', args.output, '--sample-rate', str(sample_rate),
                '--trigger', 'AWG_PE' if args.ultrasound else 'NOW',
            ]
            if args.plot_frame is not None:
                client_command.extend([
                    '--plot-frame', str(args.plot_frame),
                ])
            if args.plot_output:
                client_command.extend(['--plot-output', args.plot_output])
            if args.waveform_output:
                client_command.extend(['--waveform-output', args.waveform_output])
            if recording_dir:
                client_command.extend(['--record-dir', str(recording_dir)])
            if args.ultrasound:
                client_command.extend(['--ultrasound', '--filter-low', str(args.filter_low),
                                       '--filter-high', str(args.filter_high), '--blank-us', str(args.blank_us)])
            client_timeout = None if args.plot_frame is not None else args.seconds+30
            subprocess.run(client_command, check=True, timeout=client_timeout)
            if process.wait(timeout=15):
                raise RuntimeError('Native server reported an error')
            thread.join(timeout=2)
            result=json.loads(Path(args.output).read_text(encoding='utf-8'))
            result.update(target_fps=args.fps, requested_seconds=args.seconds, server_log=log)
            result.update(decimation=args.decimation, sample_rate_hz=sample_rate,
                          acquisition=acquisition)
            if args.ultrasound:
                result.update(burst_frequency_hz=args.frequency, burst_cycles=args.cycles,
                              generator_amplitude_v=args.amplitude, mux_profile=args.mux_profile,
                              extra_warmup_shots=5)
            Path(args.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            # Server has bounded duration and socket timeouts even on disconnect.

if __name__=='__main__':
    main()
