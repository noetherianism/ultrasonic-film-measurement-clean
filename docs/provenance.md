# Source provenance and attribution

This repository begins from the latest selected laboratory implementation in:

- Source: [`noetherianism/ultrasonic_redpitaya`](https://github.com/noetherianism/ultrasonic_redpitaya) (private at the time of migration).
- Exact source commit: `4d3c3b3efcbde108399fcdc6e0b9d55d1952144a`.
- Source commit subject: `Thin water film option: baseline removal (--baseline FOLDER), after Al-Aufi et al. 2019`.

The source Git history contains the author names **noetherianism**, **Raihan Fauzy**, and **samsmith**. This attribution preserves the names in that history without asserting that each person authored every migrated file. Retained copyright or attribution notices in individual files continue to apply.

## Migration map

| Original source | Packaged destination |
| --- | --- |
| `massflo_single_rt.py` | `ultrasonic_film/massflow.py` and independent DSP in `processing.py` |
| `live_ultrasound.py` | `ultrasonic_film/live.py` |
| `lapisan_tipis.py` | `ultrasonic_film/thin_layer.py` |
| `siglent_scope.py` | `ultrasonic_film/scope.py` |
| `coba_redpitaya_300fps/recording_io.py` | `ultrasonic_film/recording.py` |
| Socket transport needed by the native client | `ultrasonic_film/transport.py` |
| `coba_redpitaya_300fps/rp_native_client.py` | `ultrasonic_film/native_client.py` |
| `coba_redpitaya_300fps/run_rp_benchmark.py` | `ultrasonic_film/benchmark.py` |
| `coba_redpitaya_300fps/rp_native_benchmark.cpp` | `ultrasonic_film/native/rp_acquisition.cpp` |
| `coba_redpitaya_300fps/plot_recording.py` | `ultrasonic_film/plot_recording.py` |
| Selected source test suite | `tests/`, updated for package imports |

The migration adds packaging, generic host configuration, public documentation, and verification around the retained measurement implementation. It omits historical experiment scripts, duplicate snapshots, actual recordings, machine-specific configuration, private conversations, and raw operational logs.

The old repository remains the private laboratory history. The new repository is intended to be the main codebase; experiments should identify the version of this code they use instead of maintaining a second divergent copy of the same core implementation.

## License status

The selected source tree has no project-wide license file. A project-wide license grant has not yet been chosen for this repository, and this migration does not introduce a new grant on behalf of prior contributors. No `LICENSE` file is added by this refactor. License selection and any upstream attribution requirements remain an explicit maintainer decision.

## Scientific references inherited from source

The thin-layer implementation attributes its baseline-removal approach to Al-Aufi et al. (2019), *Flow Measurement and Instrumentation*, volume 66, pages 67–78, section 2.2. The sound-speed function attributes its temperature polynomial to Marczak (1997). These attributions describe the implementation's provenance; this migration does not independently establish that the experimental apparatus meets a publication's conditions or reproduces its accuracy.
