# Native media support and hard resource limits

MojiLex only decodes untrusted media when its isolated worker can enforce the
configured hard memory limit, CPU limit and wall timeout. Windows uses a Job
Object; Unix applies resource limits in a fresh interpreter before importing
media decoders. `mojilex doctor` probes the actual Unix limit setup, not merely
the presence of Python resource constants.

Some macOS versions, including the macOS 26.6.2 GitHub runner tested with Python
3.11 and 3.13, reserve hundreds of GiB of virtual address space before decoding.
Their kernel rejects the worker's exact 512 MiB address-space limit. On those
hosts the native media backend is unavailable: `doctor` reports unavailable and
actual decoding raises a dependency error before creating output. The program
does not increase the memory limit or fall back to an unbounded decoder.

Dataset commands and installation remain available. Native media processing
requires a host where `mojilex doctor` confirms the backend, such as the tested
Windows or Linux environments. Synthetic decoder unit tests still run on macOS;
only real isolated-decoder integration tests are capability-gated. CI separately
asserts the fail-closed behavior and the installed wheel's diagnostic output.
