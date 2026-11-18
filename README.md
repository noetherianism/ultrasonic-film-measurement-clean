# Ultrasonic film measurement

Pulse-echo acquisition, water-layer thickness estimation, and raw recording/replay for Red Pitaya, with an optional Siglent oscilloscope preview. The project packages the latest laboratory implementation into an installable Python application while retaining its signal-processing behavior and Indonesian command-line options.

The measured quantity is the delay between the acrylic reference echo and the water-surface echo. The application converts this round-trip delay to local water thickness using `h = c × Δt / 2`. The `massflow` module name is inherited from the original project: the application does **not** yet provide a validated mass-flow measurement, spatial reconstruction, or tomography.

This is research software. The refactor has been checked offline; a new hardware session is required before treating it as validated on a physical installation. In particular, synthetic thin-layer results do not establish a real-world 0.3 mm detection limit. See [validation status](docs/validation.md).

## Install

Use Python 3.10 or newer. From this repository, create and activate a virtual environment, then install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Offline replay needs Python and the installed dependencies. Live hardware acquisition additionally needs PuTTY `plink.exe` on Windows, SSH access to Red Pitaya, a C++20 compiler and the Red Pitaya SDK on the device, and network access to the native acquisition server on TCP port 5001. Plot windows need a working Matplotlib desktop backend.

## Start offline

Generate a synthetic example after installation:

```powershell
python examples/create_demo_recording.py --frames 1200
film-live --proses-ulang recordings/demo --tanpa-plot --flow-features
```

This generates illustrative echoes, not experimental evidence. The optional `--flow-features` adds internal-echo candidates and rolling signal-pattern metrics to GUI/CSV. Labels are unvalidated candidates, not confirmed annular, stratified or bubbly/slug flow. See the [technical assessment](docs/technical-assessment.md).

Use a completed recording folder from this application or the compatible original recorder:

```powershell
film-live --proses-ulang recordings/SESSION --tanpa-plot
film-replay-plot recordings/SESSION --frame 0 --output waveform.png
```

`SESSION` is a placeholder for your recording folder. No experimental recordings are bundled. Replay opens no hardware connection; it writes or replaces `tof_offline.csv` inside that folder. It uses the current DSP defaults and supplied flags, **not** an automatic restoration of every setting in `acquisition.json`. Save each analysis command and copy previous results before comparing parameter sets. Details: [recording format and replay](docs/recording-format.md).

## Commands

| Command | Purpose |
| --- | --- |
| `film-live` | Thickness display, raw recording, and offline reprocessing |
| `python -m ultrasonic_film` | Equivalent entry point for `film-live` |
| `film-diagnostics` | Live raw signal, noise, envelope, and spectrum diagnostics; starts with TX OFF and does not record |
| `film-scope` | Read Siglent settings and waveforms; optionally configure trigger/timebase |
| `film-benchmark` | Measure native acquisition/transfer performance; optional waveform recording |
| `film-replay-plot` | Plot a selected raw frame from a completed recording |

Run any command with `--help` for its complete options. Common thickness options retain their original names:

| Option | Meaning |
| --- | --- |
| `--rekam --rekam-detik 10` | Start recording automatically; stop recording after 10 seconds |
| `--folder-rekaman PATH` | Parent directory for new recording sessions; default `./recordings` |
| `--proses-ulang FOLDER --tanpa-plot` | Recompute a recording without hardware or a plot window |
| `--rata N` | Average the most recent N raw waveforms before DSP |
| `--median N` | Smooth the displayed ToF using recent valid detections |
| `--suhu-air T` | Use water temperature in °C to calculate sound speed |
| `--c-air C` | Supply water sound speed in m/s; `--suhu-air` takes precedence |
| `--baseline FOLDER` | Experimental thin-layer processing using a compatible thick-water reference |
| `--scope` | Add a Siglent waveform preview |

## Hardware session

The Red Pitaya host defaults to `RP_HOST`, then `redpitaya.local`. The optional scope defaults to `SIGLENT_HOST`, then `siglent.local`. Supply your device's verified SSH fingerprint with `--host-key` or `RP_HOST_KEY`; no laboratory fingerprint is bundled. Passwords are requested interactively. Keep local credentials and recordings outside version control.

First inspect the configured apparatus with diagnostics. Example, after setting `RP_HOST_KEY` to your verified fingerprint:

```powershell
film-diagnostics redpitaya.local
film-scope siglent.local --no-plot
```

Once the wiring, coupling, and measurement reference are ready:

```powershell
film-live redpitaya.local --rekam --rekam-detik 10
```

**Unlike diagnostics, `film-live` starts transmitting.** Its inherited settings are a 5 MHz, one-cycle burst, generator amplitude 1 V (not Vpp), decimation 2, and target acquisition rate 300 frames/s. Its default `single-ch1` mux profile sets `DIO0_P..DIO4_P` to `0,0,0,0,1`; use `--mux-profile unchanged` when the installation does not use that profile. The profile pins are not automatically restored on exit. These defaults describe the original apparatus and must be checked against yours.

The recording timer stops recording, not the live transmitter. Close the plot or press Ctrl+C to end the live session; the native server requests acquisition/output cleanup. Check the device state after interruption or loss of connectivity. Only one program should control the apparatus during a session.

The scope preview is polled independently. A matching time-axis display does not mean the scope and Red Pitaya captured the same pulse. `--scope` does not save Siglent waveforms into the Red Pitaya recording. `--scope-setup` also changes scope trigger/timebase settings.

Follow the [experiment protocol](docs/experiment-protocol.md) when reinstalling the apparatus or collecting a new baseline.

## Development and provenance

```powershell
python -m unittest discover -s tests -t . -v
```

Some native integration tests require Linux and a C++ compiler; skipped tests are not hardware validation. Read the [architecture](docs/architecture.md), [validation status](docs/validation.md), and [contribution guide](CONTRIBUTING.md) before changing acquisition or DSP behavior.

This repository is derived from [`noetherianism/ultrasonic_redpitaya`](https://github.com/noetherianism/ultrasonic_redpitaya) at commit `4d3c3b3efcbde108399fcdc6e0b9d55d1952144a`. The source repository is private. Attribution and file mapping are recorded in [provenance](docs/provenance.md). No project-wide license grant has been chosen for this repository yet.
