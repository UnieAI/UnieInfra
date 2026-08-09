## 1. Kernel layer — close the two gaps

- [ ] 1.1 Add a selection report API to `tokenspeed_kernel.selection` that, for one
      `(family, mode, format_signature, platform, traits)`, returns every registered
      candidate with its outcome: selected, eligible-but-ranked-lower,
      excluded-by-capability, or excluded-by-signature/trait.
- [ ] 1.2 Make an unsatisfiable `override=` / `solution=` request raise, naming the request
      and the rule it failed, instead of falling through to ranking. Cover the YAML, env
      var, and context-manager override paths.
- [ ] 1.3 Tests: a kernel scoped `(10,0)`–`(10,3)` is excluded on a 12.x platform, is
      reported as excluded-by-capability, and raises when requested by name.
- [ ] 1.4 Document the solution-naming convention (implementation lineage, not source
      project) and the upper-bound review rule in `tokenspeed-kernel`'s contributing notes.

## 2. Autotune persistence

- [ ] 2.1 Define the pin-file format and its environment fingerprint (device name, arch,
      candidate name set, shape signature).
- [ ] 2.2 Write results after a measured selection; load and apply them at startup.
- [ ] 2.3 On fingerprint mismatch, report what differed and do not apply the file.
- [ ] 2.4 Tests: round-trip; mismatch is rejected loudly; loading a pin performs no
      measurement.

## 3. Scheduler policy seam — preemption first

- [ ] 3.1 Define the preemption policy interface in `tokenspeed-scheduler/csrc/scheduler/`
      (choose a victim from the retractable set), plus a name → factory registry resolved
      once at scheduler construction.
- [ ] 3.2 Register the current behaviour ("largest `TokenSize()`") as a named policy and
      make it the default. Move `csrc/scheduler/operations/forward.cpp:1277` behind the
      interface.
- [ ] 3.3 Carry the policy name on `SchedulerConfig` and expose it as a server arg.
- [ ] 3.4 Tests: default policy reproduces today's victim choice on a fixed candidate set;
      every rank resolves the same policy; an unknown policy name fails at construction.
- [ ] 3.5 Confirm no measurable per-step scheduler cost from the indirection, at 1 / 8 /
      32 / 128 in-flight requests.

## 4. Runtime backend registries

- [ ] 4.1 Give `runtime/layers/attention/registry.py` and `runtime/sampling/registry.py`
      the same exclusion-reporting surface as the kernel layer.
- [ ] 4.2 Make an unsatisfiable backend request fail loudly rather than falling back.

## 5. Operator surface

- [ ] 5.1 One command or endpoint that prints, for this machine, the selected
      implementation and the exclusion reasons for every registered alternative, across
      kernel / runtime backend / scheduler policy.
- [ ] 5.2 Document the override precedence order (config file → context manager → env var
      → explicit argument) in one place.

## 6. Close-out

- [ ] 6.1 Verify no default changed: run the existing sweep before and after and compare
      the selected implementation for every operation exercised.
- [ ] 6.2 Record in `unieinfra/xref/70-PORTS.md` that the ports now land on this contract,
      and which registry each port's "註冊成" column refers to.
