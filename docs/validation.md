# Validation status

Evidence from the original apparatus, synthetic tests, and tests of this refactor answer different questions. None alone establishes an end-to-end uncertainty or detection limit for a newly assembled measurement system.

## Historical acquisition benchmark

The source repository's `coba_redpitaya_300fps/REDPITAYA_BENCHMARK.md` records a paced DMA run with 18,000 complete, sequential frames over approximately 60 seconds: 299.934 frames/s measured at the PC and 299.999 frames/s at the server.

That test used **one channel, 16,384 samples/frame, decimation 1, software `NOW` triggering, and no plotting or waveform storage**. It supports the feasibility of the acquisition/transfer path in that installation. It does not validate 300 frames/s with ultrasound triggering, DSP, raw recording, and plotting active together. PC receive timing is not a hardware trigger-jitter measurement.

## Later source implementation

The selected source commit includes recording/replay, adaptive echo thresholds, sound-speed conversion, Siglent preview fixes, and experimental thin-layer baseline removal. Source comments describe evaluation on an acrylic plate and approximately 20/40 mm water layers, including improved echo-selection consistency with the newer threshold/smoothing settings.

Those are historical observations recorded by the source author, not new measurements of this refactor. The underlying experimental datasets and private lab history are not published here, so the quantitative source claims cannot be independently reproduced from this public checkout alone.

The thin-layer tests use generated RF waveforms with defined pulse shapes, added noise, jitter, and echoes. Passing a simulated 0.3 or 0.5 mm case establishes behavior for that test model. It does not establish the same accuracy, false-positive rate, or detection limit under real coupling, temperature, surface geometry, or material variation.

## Refactor verification

The software test suite covers recording integrity, binary-stream behavior, ordinary and baseline echo detection, scope waveform parsing, and other offline behavior. Run it with:

```powershell
python -m unittest discover -s tests -t . -v
```

Native tests that require Linux and a C++ compiler may be skipped on Windows. Report those skips separately from passes. The current refactor's test results should include the code revision, operating system, command, passed/failed/skipped counts, and numerical comparison against the selected source where applicable.

Initial verification on 2026-10-19 (Windows, Python 3.12; this repository's initial release tree):

- Full offline suite with `RUN_PERFORMANCE_TESTS=1`: 78 tests discovered, **72 passed, 6 Linux-only native simulator tests skipped**, no failures. The optional timing test passed in this run. Default CI leaves timing assertions off and runs native simulator tests on Linux. Ruff passed; wheel and source distribution built successfully.
- Six private source recordings, 8,895 frames total, were processed in default, manual MA128/threshold0.15 and acrylic-preset modes. Voltage conversion, envelopes, primary echo status, timing and amplitude matched source commit `4d3c3b3efcbde108399fcdc6e0b9d55d1952144a` exactly in all 26,685 frame/configuration comparisons. Data were read locally and are not published. This checks numerical preservation, not physical measurement accuracy. Internal-peak fields are new and are not included in the legacy comparison.
- The generated example completed a 1,200-frame recording, replay with feature CSV/analysis sidecar, and frame plot without hardware. All primary echoes were detected in this illustrative example.
- The repeatable synthetic thin-layer audit checks five filter choices, pulse widths/carriers, 0.3/0.5 mm delays, an injected 5.96 MHz interferer and small template-shape stretches. Under its limited baseline-detector model, all test echoes were detected and thick-water controls produced no thin-layer detections. This is not a real 0.3 mm acceptance test and does not establish a universal temperature tolerance. Run the script and inspect the complete assumptions in [technical assessment](technical-assessment.md).
- A three-run, 1,216-frame/run DSP+buffered-CSV benchmark measured median 0.943 ms/frame without experimental features and 1.081 ms/frame with them (about 1,061 and 925 frames/s). Network, GUI, hardware, and worst-case disk stalls are excluded. A concurrent CPU-heavy replay caused an earlier timing assertion to miss its deadline; timing assertions are therefore opt-in on an idle host (`RUN_PERFORMANCE_TESTS=1`), while functional tests remain unconditional. These medians do not guarantee 300 FPS end to end.

The optional feature suite covers four synthetic pattern families, internal-echo/reverberation separation, wave-window duration, monotonic drift, data gaps, bad timestamps, missing internal-echo analysis, and GUI/CSV integration. Candidate labels require independent physical validation. CI also checks installation, CLI entry points and packaged native source on Windows/Linux.

Revision of 2026-10-24 (independent review fixes, offline on the Windows lab laptop, no apparatus connected): receive clock `time.perf_counter()`; flow sequence time base with per-window frame-rate validation (lower envelope of receive times, overall/early/late rates within 5%, recovery only after a full clean window); row-only `INVALID_DATA`; plate-edge secondary candidates dropped; thin-layer mismatch check from the first frame; `--flow-features` without plate rejection warns; DSP queue backlog above 150 frames warns (recording unaffected).

- Offline suite: 90 tests, 83 passed, 7 skipped (Linux-only native simulator and opt-in timing tests); ruff passed.
- Synthetic regressions: nominal 300 with actual 150 frames/s now gives `TIMING_UNCERTAIN` instead of a valid 6 Hz wave; a stable ~300 frames/s stream (299.5, 15.625 ms clock) gives 3 Hz within 0.03 Hz; a slowdown after a normal start is rejected and earlier results are not carried on; bunched arrivals give no wrong valid frequency; validity returns only after a full window of good timing.
- The legacy comparison over the six private recordings (8,895 frames, default/manual/acrylic-preset) still matched exactly. This shows numerical preservation of the primary echo path, not physical accuracy.
- Replaying the 20 mm and 40 mm lab recordings with `--flow-features` gives `SMOOTH_CANDIDATE` after the two-second warmup instead of 100% `TIMING_UNCERTAIN`. These recordings contain still water; they do not test wave-frequency, intermittent or internal-echo classification.

Not yet tested: live acquisition with the GUI and the new clock on the apparatus, real wavy/intermittent flows with an independent reference, device-side sampling irregularity, and physical thickness accuracy against independent references.

No apparatus was connected during the initial refactor. Hardware validation of the packaged application remains pending.

## Next hardware acceptance session

Use the [experiment protocol](experiment-protocol.md) and record the actual software revision and instrument settings. Before the session, choose acceptable measurement error and false-detection rates from the intended experiment rather than tuning acceptance limits after seeing the data.

Check the following independently:

- **Acquisition integrity:** requested duration/rate, received and processed frame counts, sequence errors, file sizes, and raw/DSP completeness with recording and plotting active.
- **Reference measurement:** bias and repeatability at independently known water thicknesses, with reference uncertainty and temperature recorded.
- **Thin-layer behavior:** detection and false-detection rates, invalid/mismatch statuses, and errors across selected thicknesses; include conditions without a thin-layer echo in the detection window.
- **Within-session drift:** compare equivalent reference/baseline conditions before and after the test sequence.
- **Reinstallation repeatability:** repeat with a fresh assembly and a fresh baseline; keep this separate from stability across many frames in one mounting.

Mass-flow rate, spatial imaging, and tomography require additional physical measurements or geometry and are not validated outputs of the current application.
