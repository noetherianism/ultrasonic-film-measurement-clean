"""Lossless frame storage: contiguous little-endian int16 plus receive times."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

SAMPLES_PER_FRAME = 16384
BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2


class FrameRecorder:
    """Buffered streaming to disk; storage failure aborts instead of dropping frames."""

    def __init__(self, directory, *, sample_rate_hz, trigger, host):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.metadata = dict(
            format_version=1, complete=False, frames_saved=0,
            samples_per_frame=SAMPLES_PER_FRAME, dtype='<i2', channels=1,
            raw_file='frames.bin', timing_file='frame_times.csv',
            sample_rate_hz=sample_rate_hz, trigger=trigger, host=host,
            units='raw ADC counts',
            timing='PC receive completion time relative to start; not hardware trigger time',
            created_utc=datetime.now(timezone.utc).isoformat())
        self.files = ExitStack()
        try:
            # Exclusive creation protects previous recordings, even for direct client use.
            self.manifest = self.directory / 'recording.json'
            with self.manifest.open('x', encoding='utf-8') as out:
                json.dump(self.metadata, out, indent=2)
            self.raw = self.files.enter_context((self.directory / 'frames.bin').open('xb', buffering=1024*1024))
            self.times = self.files.enter_context((self.directory / 'frame_times.csv').open('x', encoding='utf-8', newline=''))
            self.times.write('frame_index,receive_elapsed_seconds\n')
        except BaseException:
            self.files.close()
            raise

    def append(self, sequence, payload, elapsed):
        if sequence != self.frames or len(payload) != BYTES_PER_FRAME:
            raise ValueError('Recording frame sequence or payload size mismatch')
        if self.raw.write(payload) != BYTES_PER_FRAME:
            raise OSError('Incomplete waveform disk write')
        self.times.write(f'{sequence},{elapsed:.9f}\n')
        self.frames += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        close_error = None
        try:
            self.files.close()
        except BaseException as error:
            close_error = error
        size = (self.directory / 'frames.bin').stat().st_size
        complete = exc is None and close_error is None and self.frames > 0 and size == self.frames * BYTES_PER_FRAME
        self.metadata.update(frames_saved=min(self.frames, size // BYTES_PER_FRAME),
                             raw_bytes=size, complete=complete)
        if exc is not None or close_error is not None:
            error = close_error if close_error is not None else exc
            self.metadata['error'] = f'{type(error).__name__}: {error}'
        # Atomic replacement leaves the original incomplete manifest if finalization fails.
        temporary = self.directory / 'recording.json.tmp'
        with temporary.open('w', encoding='utf-8') as out:
            json.dump(self.metadata, out, indent=2)
        temporary.replace(self.manifest)
        if close_error is not None:
            raise close_error
        if exc is None and not complete:
            raise OSError('Recording did not finish with complete frames')
        return False


def load_recording(directory):
    """Memory-map a completed run without loading all waveforms into RAM."""
    directory = Path(directory)
    metadata = json.loads((directory / 'recording.json').read_text(encoding='utf-8'))
    if not metadata.get('complete'):
        raise ValueError('Recording is incomplete; inspect recording.json before recovering partial data')
    if (metadata.get('format_version') != 1 or metadata.get('dtype') != '<i2' or
            metadata.get('samples_per_frame') != SAMPLES_PER_FRAME):
        raise ValueError('Unsupported recording format')
    frames = metadata['frames_saved']
    if not isinstance(frames, int) or frames <= 0:
        raise ValueError('Invalid recorded frame count')
    raw = directory / 'frames.bin'
    if raw.stat().st_size != frames * BYTES_PER_FRAME:
        raise ValueError('Recorded file size does not match metadata')
    waveforms = np.memmap(raw, dtype='<i2', mode='r', shape=(frames, SAMPLES_PER_FRAME))
    return waveforms, metadata
