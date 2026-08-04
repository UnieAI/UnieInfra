"""Admission gates for measurement results.

Each gate here corresponds to a specific way a real conclusion went wrong. The
comments name the failure, because a gate whose reason has been forgotten is the
first one someone removes when it is inconvenient.

A gate returns a ``GateResult``. ``ok=False`` means the measurement may not be
used for a promotion decision -- not that the candidate is bad. The distinction
matters: a rejected measurement should be re-run, a rejected candidate should
not.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field


@dataclass
class GateResult:
    name: str
    ok: bool
    detail: str
    values: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def period_blocked_mean(series: list[float], period: int = 2) -> tuple[float, float, int]:
    """Mean and standard error over whole limit-cycle periods.

    Serving throughput under KV pressure settles into a period-2 cycle: the
    scheduler can only admit a request as one retires, so the batch falls into
    lockstep and alternates between a prefill burst and a decode burst. Measured
    swing reached 50% of the mean at 90% KV utilisation.

    Averaging raw windows inherits that swing. Averaging whole periods does not,
    because the variance being removed is structure rather than noise. The same
    Qwen3-32B run reads 547.3 +/- 203 by window mean and 547.3 +/- 0.4 by period
    mean -- the same estimate of the level, ~500x tighter.

    Returns (mean, standard_error, n_periods). SE is computed over periods, so
    it is honest about how many independent observations there really are.
    """
    if len(series) < period * 2:
        return (st.mean(series) if series else float("nan"), float("nan"), 0)
    blocks = [
        sum(series[i:i + period]) / period
        for i in range(0, len(series) - period + 1, period)
    ]
    se = st.pstdev(blocks) / (len(blocks) ** 0.5) if len(blocks) > 1 else float("nan")
    return st.mean(blocks), se, len(blocks)


def gate_stationary(raw_windows: list[float], drift_tol: float = 0.05,
                    envelope_tol: float = 0.08,
                    swing_floor: float = 0.10) -> GateResult:
    """Has the run reached steady state?

    Takes RAW measurement windows. Passing period-blocked means instead will
    silently return PASS: blocking has already removed the alternation, so swing
    collapses below ``swing_floor``, the envelope test is skipped, and drift
    across the smoothed series is small. Verified -- feeding this the period
    means of the very arm it was written to catch (539, 531, 533, 525, 527, 525)
    returns "settled", while its 12 raw windows correctly return
    "envelope -26.7% on 35.3% swing".

    That is a nasty ordering dependency, so: gate first, block second.

    Origin: a 13-sigma "-5.19% regression" that re-ran at -0.03%. The arm it was
    measured against had not converged -- its period means fell 539 -> 525
    throughout. A non-converged midpoint distorts the segments on BOTH sides and
    in opposite directions, which is why the ladder appeared to drop and then
    partially recover. High sigma proves two numbers differ, not that they are
    comparable.

    Three signals, because each misses a different failure:
      drift     second-half vs first-half mean. Catches monotone trends, blind
                to oscillation (equal halves by construction).
      swing     peak-to-peak of the alternation as a fraction of the mean.
      envelope  whether the alternation's highs and lows are themselves
                converging or diverging. This is the one that caught the case
                above: both arms had large swing and small drift, but their
                envelopes moved in opposite directions.

    The envelope test is only applied above ``swing_floor``. Below a few percent
    swing the "lows" and "highs" are the same distribution split arbitrarily, so
    their trends are noise and their difference is noise doubled -- one settled
    run scored -80.6% envelope on a 3.9% swing before this guard existed.
    """
    series = raw_windows
    n = len(series)
    if n < 6:
        return GateResult("stationary", False, f"only {n} windows; need >= 6")

    half = n // 2
    first, second = st.mean(series[:half]), st.mean(series[half:])
    drift = (second - first) / first if first else 0.0

    even, odd = series[0::2], series[1::2]
    lo, hi = (even, odd) if st.mean(even) < st.mean(odd) else (odd, even)
    mean = st.mean(series)
    swing = (st.mean(hi) - st.mean(lo)) / mean if mean else 0.0

    def trend(seq: list[float]) -> float:
        k = max(1, len(seq) // 3)
        a, b = st.mean(seq[:k]), st.mean(seq[-k:])
        return (b - a) / a if a else 0.0

    envelope = trend(hi) - trend(lo)

    problems = []
    if abs(drift) > drift_tol:
        problems.append(f"drift {drift:+.1%}")
    elif swing > swing_floor and abs(envelope) > envelope_tol:
        problems.append(f"envelope {envelope:+.1%} on {swing:.1%} swing")

    return GateResult(
        "stationary", not problems,
        "settled" if not problems else "; ".join(problems),
        dict(drift=drift, swing=swing, envelope=envelope, n=n),
    )


def gate_counter_agreement(client_tokens: float, server_tokens: float,
                           tol: float = 0.02) -> GateResult:
    """Do the client's and the server's token counts agree?

    Origin: the largest wrong result of the session. The benchmark client
    counted one token per SSE chunk. Under speculative decoding the engine emits
    every token accepted in a step as a single chunk, so a chunk carries k+1
    tokens and the client undercounted by the acceptance factor. A +45% gain was
    reported as a -42% loss, consistently, across six configurations.

    The check already existed and reported 0.34-0.69 on every speculative arm
    against ~1.00 on every baseline. It was written into every result file and
    never read. That is why this is a GATE and not a logged statistic: being
    computed is not the same as being seen.

    Note the failure mode is asymmetric and therefore especially dangerous: it
    only distorts comparisons where tokens-emitted-per-step DIFFERS between the
    arms. Every baseline looked perfect, which made the results look internally
    consistent.
    """
    if not server_tokens:
        return GateResult("counter_agreement", False,
                          "no server-side counter; client numbers unverifiable")
    frac = client_tokens / server_tokens
    ok = abs(frac - 1.0) <= tol
    return GateResult(
        "counter_agreement", ok,
        f"client counted {frac:.1%} of server tokens"
        + ("" if ok else " -- use the server counter"),
        dict(frac=frac),
    )


def gate_reference_match(candidate: str, reference: str,
                         exact: bool = True) -> GateResult:
    """Does the output match a known-good path, rather than expectation?

    Origin: hours spent 'fixing' attention backends for gemma-4, whose output
    ``" France is France is France is ..."`` looked obviously broken. Plain HF
    transformers with sdpa -- no paging, no CUDA graphs, no custom attention --
    produces exactly the same text. It is an instruction-tuned checkpoint given
    a bare completion prompt; degenerating is its normal behaviour.

    So the comparison is against a reference path, and **reproducing the
    reference's bad output is a PASS**. Matching what you expected to see is not
    the test; matching what the model actually does is.

    ``exact=False`` allows a looser check for paths whose numerics legitimately
    differ (a reduction reordered by a kernel change will diverge on near-ties,
    and greedy decoding compounds that into different text). In that mode the
    caller is responsible for supplying a comparison that is still meaningful --
    typically a distributional check rather than a string compare.
    """
    if exact:
        ok = candidate == reference
        detail = "identical to reference" if ok else (
            f"differs: ref={reference[:48]!r} got={candidate[:48]!r}")
    else:
        ok = bool(candidate.strip())
        detail = "non-empty (loose mode; caller must verify distribution)"
    return GateResult("reference_match", ok, detail)


def gate_switch_engaged(log_text: str, expect: list[str],
                        forbid: list[str] | None = None) -> GateResult:
    """Did the configuration actually take effect?

    Origin: three flags silently ignored in a single session.
      - ``VLLM_ATTENTION_BACKEND`` had been removed from that vLLM entirely;
        setting it did nothing and produced no error.
      - ``--compilation-config pass_config`` was overwritten by the
        optimization-level table, so three arms measured the same thing.
      - ``UNIEINFRA_GEMM=0`` disabled one of two entry points; ``UNIEINFRA_QUANT``
        stayed on.

    Each cost a full set of runs, and each was invisible in the results: the
    arms simply agreed, which reads as "this layer contributes nothing" rather
    than "this layer never ran". Assert on the run's own log instead.
    """
    missing = [s for s in expect if s not in log_text]
    present = [s for s in (forbid or []) if s in log_text]
    ok = not missing and not present
    bits = []
    if missing:
        bits.append(f"expected but absent: {missing}")
    if present:
        bits.append(f"forbidden but present: {present}")
    return GateResult("switch_engaged", ok,
                      "engaged" if ok else "; ".join(bits))


def gate_attribution(model: str | None, concurrency: int | None,
                     shape: str | None) -> GateResult:
    """Is the result quoted with the conditions it holds under?

    Origin: nothing measured here generalises across all three axes.
      - UnieInfra's kernel layer: +4.2% at 8B batch-1, 0.00% at 32B C=32.
      - TokenSpeed: +3.84% at C=128, -7.62% at C=32, same model.
      - The attention split: +2.60% at C=64 and +60.45% at C=256, same shape.

    A number without its conditions is not a weaker claim, it is a different and
    false one. This gate is bookkeeping rather than statistics, and it is here
    because the failure is one of reporting, which no amount of measurement
    rigour prevents.
    """
    missing = [n for n, v in
               (("model", model), ("concurrency", concurrency), ("shape", shape))
               if v in (None, "")]
    return GateResult("attribution", not missing,
                      "fully attributed" if not missing
                      else f"missing: {missing}")


def gate_environment_ready(tool: str, probe_output: str,
                           must_contain: list[str]) -> GateResult:
    """Did the external tool actually emit data, on a cheap probe, before we
    spent an expensive workload on it?

    Origin: four consecutive failed attempts to get ncu to produce a profile,
    none of them related to the kernel under study -- a container missing
    CAP_SYS_ADMIN, a second container resolving the wrong vLLM, an unregistered
    quantization method, and a flashinfer version mismatch. Each cost a ~7
    minute model load to discover.

    The first is the one this gate exists for. ncu connected, the workload ran to
    completion, the profiler disconnected cleanly, the exit code was zero, and it
    had profiled **nothing**: no error, no report, no ``==PROF== Profiling...``
    lines. A human notices the missing file. An unattended loop reads "no
    profiling data" as "no problem found" and moves on, and every downstream
    decision inherits that silence.

    So: prove the tool emits the artifact you need on a 3-second workload, then
    load the 70 GB model. Ten seconds of ``tiny.py`` located the permission
    problem after ten minutes had already been spent not finding it.

    ``must_contain`` should name evidence of *output*, not of *success*. Exit
    codes and "Disconnected" lines were present in the failing case; the
    ``==PROF== Profiling`` line and a non-empty report were not.
    """
    missing = [s for s in must_contain if s not in probe_output]
    ok = not missing
    return GateResult(
        "environment_ready", ok,
        f"{tool} produces output"
        if ok else f"{tool} ran but produced no evidence of: {missing}",
    )


def evaluate_gates(gates: list[GateResult]) -> tuple[bool, str]:
    """All gates must pass. Returns (ok, human-readable summary)."""
    failed = [g for g in gates if not g.ok]
    lines = [str(g) for g in gates]
    if failed:
        lines.append(f"=> MEASUREMENT REJECTED ({len(failed)} gate(s) failed); "
                     "re-run, do not judge the candidate on this")
    else:
        lines.append("=> measurement admissible")
    return (not failed), "\n".join(lines)
