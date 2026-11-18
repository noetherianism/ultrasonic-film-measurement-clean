# Architecture

The refactor separates the application into an importable `ultrasonic_film` package. The first goal is to preserve the selected source version's numerical behavior and recording compatibility while removing machine-specific paths and making commands installable.

| Module | Responsibility |
| --- | --- |
| `massflow.py` | Thickness CLI and GUI; sound-speed conversion; DSP worker; recording and replay orchestration |
| `processing.py` | Immutable `EchoConfig`, ordinary echo detection, optional secondary-peak candidates |
| `flow.py` | Experimental rolling signal-pattern features with data/timing quality checks |
| `live.py` | SSH build/launch and native streaming receiver; independent signal diagnostics GUI |
| `thin_layer.py` | Experimental thick-water baseline construction, waveform alignment/scaling, residual detection, and mismatch checks |
| `scope.py` | Siglent SCPI connection, waveform decoding, and independently polled preview |
| `recording.py` | Versioned raw frame storage, completion checks, and memory-mapped loading |
| `transport.py` | Minimal socket transport, exact binary reads, and platform-specific receive handling |
| `native_client.py` | Benchmark stream receiver, statistics, and optional waveform export |
| `native/rp_acquisition.cpp` | Red Pitaya acquisition, generator control, pacing, DMA buffer access, and TCP framing |
| `benchmark.py` | Build and run the native benchmark and collect its results |
| `plot_recording.py` | Plot an individual recorded raw frame |

## Live data path

```text
Red Pitaya burst + ADC
        |
native acquisition server (DMA, sequence-numbered frames)
        |
TCP receiver on the PC
        +--> raw recorder: every accepted frame before DSP
        |
        +--> DSP queue --> echo detection --> ToF / thickness CSV
                                      |
                                      +--> latest frame + trend --> GUI

Siglent --> independent polling thread --> optional GUI preview
```

Acquisition, DSP, and display rates describe different operations. The GUI renders the latest available result rather than drawing every received frame. Check received-frame and processed-frame counts as well as recording completeness; a smooth plot alone does not demonstrate lossless acquisition or bounded DSP backlog.

Each raw frame contains 16,384 signed samples for one channel. The thickness DSP currently processes the first 8,000 samples. The actual sampling rate returned by the device determines the time axis; nominal ADC rate and decimation are not substitutes for that readback.

The thickness detector retains the original bandpass, Hilbert envelope, smoothing, thresholding, and echo selection. Water ToF is relative to the acrylic echo. For ToF in microseconds and sound speed in m/s, thickness in mm is `c × ToF / 2000`. The inherited 1466 m/s default belongs to the original experimental assumptions; a fresh session must record the speed or temperature used.

Ordinary echo detection and thin-layer baseline removal have separate filters. Baseline processing aligns a thick-water reference to the acrylic leading edge, fits amplitude, subtracts the reference, and thresholds the residual. It can report a mismatch instead of a thickness. A mismatch check is an algorithmic diagnostic, not a guarantee that coupling and temperature are unchanged.

## Boundaries of this refactor

`processing.EchoDSP` receives an immutable configuration and owns its detector history. The CLI compatibility adapter snapshots module settings when each processor is constructed. The thickness CLI and baseline selection still use module state, so use separate CLI processes for independent analyses. Experimental flow features are opt-in and do not change the primary echo-selection algorithm.

Neither the Siglent polling thread nor PC receive timestamps provide pulse-level synchronization between instruments. A synchronized, archived scope comparison requires an additional acquisition design. Multi-channel reconstruction and velocity or mass-flow estimation are outside the present measurement path.
