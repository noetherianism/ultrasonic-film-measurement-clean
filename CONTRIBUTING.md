# Contributing

Keep changes focused and distinguish refactoring from a change to the measurement method. A packaging change should preserve representative numerical outputs. A detector change should state the hypothesis, failure case, data used, and effect on invalid or false detections as well as successful measurements.

Install an editable checkout and run the offline suite:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -t . -v
```

Report the OS, code revision, failures, and skipped tests. Linux/native simulation tests and physical instrument tests are different forms of evidence. Do not label offline tests as hardware verification.

Timing assertions depend on host load and are opt-in: set `RUN_PERFORMANCE_TESTS=1` on an otherwise idle host. `python examples/benchmark_processing.py` separately reports repeated baseline/feature/CSV timings. Do not run numerical replay or other CPU-heavy jobs alongside performance measurements; no CI timing result certifies physical acquisition throughput.

For acquisition or recording changes, check sequence handling, exact binary payload sizes, interruption/finalization behavior, and that an error is visible rather than silently dropping frames. For DSP changes, compare the same raw input and record the exact parameters. Avoid checking only a plot or the mean of successful detections.

Before submitting:

- Keep laboratory addresses, credentials, private chat content, and actual experimental recordings out of commits. Use generic hostnames and synthetic fixtures.
- Preserve source attribution and update [provenance](docs/provenance.md) when importing code. No new project-wide license grant has been selected yet.
- Update command examples and recording documentation when behavior changes.
- State whether hardware was exercised and which configuration was used. Coordinate one hardware controller per session.
- Document incompatible storage changes with a format version and a migration strategy; do not silently reinterpret existing raw files.

Use the [experiment protocol](docs/experiment-protocol.md) for physical validation and the [validation status](docs/validation.md) to distinguish evidence already available from work still pending.
