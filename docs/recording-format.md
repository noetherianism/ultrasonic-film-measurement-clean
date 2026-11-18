# Recording format and replay

New live recordings are placed in unique session directories under `./recordings`, relative to the process working directory, unless `--folder-rekaman` specifies another parent. Benchmark recording uses `--save-all` and `--recordings-dir`.

| File | Contents |
| --- | --- |
| `recording.json` | Format version, sample rate, trigger, host, dtype, frame count, completion state, and timing interpretation |
| `frames.bin` | Contiguous raw ADC counts: one channel, 16,384 samples/frame, little-endian signed int16 (`<i2`), 32,768 bytes/frame |
| `frame_times.csv` | Recording-local `frame_index` and PC `receive_elapsed_seconds` |
| `acquisition.json` | Requested acquisition settings; live thickness recordings also include DSP and sound-speed settings |
| `tof.csv` | Live detection status, echo timing, amplitude, thickness, and stream/recording indices |
| `tof_offline.csv` | Results from the latest offline replay; replaced by subsequent replay |
| `analysis_offline.json` | DSP, baseline, sound-speed and experimental-feature settings used by that replay; replaced on replay |
| `tof_offline.png` | Replay trend image when plotting is enabled |

The three core files are `recording.json`, `frames.bin`, and `frame_times.csv`. Other files depend on the command used. Benchmark result JSON and optional selected-frame NPZ/PNG exports are additional products, not substitutes for all-frame recording.

The recorder saves accepted raw frames before DSP. Waveform averaging and display smoothing do not alter `frames.bin`. A write or frame-sequence error is intended to abort rather than silently discard frames. Normal finalization marks the recording complete only when a nonzero frame count and the binary file size agree. The loader rejects incomplete or size-inconsistent recordings; recovery of a partial file is a separate, explicit task.

**Completion of raw storage does not by itself validate the DSP output.** Check that `tof.csv` contains the expected number of rows and inspect detection statuses and worker errors as well.

## Timing and units

`receive_elapsed_seconds` is the PC time at completion of frame reception, relative to recording start, from `time.perf_counter()`. Recordings made before 2026-10-24 used `time.monotonic()`, which on Windows has 15.625 ms resolution (many identical values); use frame indices for timing in those files. It is not a hardware trigger timestamp. Network buffering affects receive intervals, so those intervals cannot directly establish pulse timing jitter or synchronization with a scope waveform.

`tof.csv` columns are:

```text
frame_index,stream_sequence,receive_elapsed_seconds,status,waktu_akrilik_us,tof_air_us,amp_air,h_mm
```

`tof_air_us` is the detected round-trip delay relative to the acrylic echo; `h_mm` is derived from the selected sound speed. Invalid measurements use a status and non-finite values rather than a confirmed zero thickness. `amp_air` is detector-dependent, particularly in baseline mode; do not treat it as a calibrated acoustic amplitude. Raw ADC counts and their nominal voltage conversion are not a traceable voltage calibration.

With `--flow-features`, these extra columns are appended:

```text
tof_internal_us,n_peaks,flow_pattern,flow_age_s,mean_h_mm,sigma_h_mm,amplitude_cv,f_wave_hz,wave_h_pp_mm,internal_fraction,valid_fraction
```

These are experimental signal features, not confirmed bubble or flow-regime measurements. Internal timing is relative to the acrylic echo. `n_peaks=0` means analysis is unavailable/off or no primary echo was detected; baseline mode does not resolve internal echoes. An unavailable `internal_fraction` is blank. Features update at most 10 Hz and `flow_age_s` records their age; warmup, quality failures and unknown patterns are retained. Wave amplitude is the window's observed maximum-minus-minimum thickness, not an independently calibrated wave height. See [technical assessment](technical-assessment.md) for thresholds and timing limitations.

## Reprocess without hardware

```powershell
film-live --proses-ulang recordings/SESSION --tanpa-plot
film-live --proses-ulang recordings/SESSION --tanpa-plot --rata 8 --suhu-air 20
```

The second command illustrates parameter selection, not a recommendation to assume 20 °C. Use measured or otherwise justified conditions. Replay uses the actual sample rate in `recording.json`, but the current program defaults and CLI flags govern the DSP. Inspect `acquisition.json` and explicitly select settings when reproducing a previous analysis.

Replay overwrites `tof_offline.csv`; preserve each result and command before running the next analysis. The original `frames.bin` and live `tof.csv` remain the input/reference. Offline CSV stream indices are replay-local indices, not necessarily the original live stream sequence numbers. If timing-row count does not match the raw frame count, the inherited replay code falls back to nominal frame spacing; inspect the timing file before interpreting a trend quantitatively.

For an experimental thin-layer analysis:

```powershell
film-live --proses-ulang recordings/THIN_SESSION --baseline recordings/BASELINE_SESSION --tanpa-plot
```

The baseline must satisfy the detector's compatibility checks and come from the appropriate physical installation. Keep the baseline session and its metadata alongside the measurement. Do not silently reuse a baseline after reinstalling a transducer.

## What is not recorded

`--scope` provides a live Siglent preview only: no Siglent raw waveform archive or pulse-level association with Red Pitaya frames is implemented. Physical geometry, independent thickness reference, uncertainty, coupling changes, and laboratory observations must be recorded separately, for example in a private `session-notes.md` beside each run.

Recording metadata can contain hostnames and paths. Review these before sharing any dataset. Real experimental data and local configuration are intentionally excluded from this public source repository.
