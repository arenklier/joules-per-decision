# Manifest

Everything the Data Availability statement promises is present, with one
stated exception recorded at the end. `scripts/validation/verify_deposit.py`
recomputes the paper's headline statistics from these files and compares them
against the printed values; run it from the repository root.

## Validation scripts, and what each needs

Run from the repository root.

| Script | Needs | Checks |
|---|---|---|
| `scripts/validation/verify_deposit.py` | nothing | recomputes the paper's headline statistics from the result files |
| `scripts/validation/oracle_validate.py` | nothing | re-derives all 120 ground truths by a second implementation; prints 120/120 |
| `scripts/validation/agent_loop_selfcheck.py` | nothing | the agentic loop's control flow against a stubbed model |
| `scripts/experiments/validators.py` | nothing | what the loop's ground-truth-free checks can and cannot see |
| `scripts/validation/netarena_selfcheck.py` | **a NetArena checkout** | grades NetArena's reference solution with NetArena's own harness |

The last one is the exception. It executes the benchmark authors' grading
code, which is theirs and is not redistributed here, so it exits with an
explanation rather than a result unless `NETARENA` points at a checkout:

    NETARENA=/path/to/NetArena/app-malt python scripts/validation/netarena_selfcheck.py

## Result files

### Accuracy and prompting condition

| File | Backs |
|---|---|
| `robustness_result.json` | Tables 3 and 4: per-task direct vs chain-of-thought correctness at 1.5B, 3B, 7B, 14B over 120 tasks |
| `robustness_rerun2.json` | second session, used for the reproducibility spread |
| `mechanism_result.json` | Table 5: the format-constrained condition on a 60-task draw |
| `marker_audit_result.json` | grading-rule audit at 14B and 1.5B: strict marker rule vs last-line fallback |
| `reasoning_tax_result.json`, `reasoning_tax2_result.json` | the reasoning-tax measurements |
| `reasoning_tax2_cap512_result.json`, `reasoning_tax2_cap1024_result.json` | pitfall 4: the same models under a 512- and a 1024-token output cap, which reach opposite conclusions |

### Energy

| File | Backs |
|---|---|
| `ladder_result.json`, `ladder2_result.json`, `ladder3_result.json` | Table 2: energy per token across the parameter ladder and the cross-family controls |
| `l40s_probe_result.json` | idle vs model-resident power on the L40S |
| `utilization_result.json`, `utilization_1p5b_result.json`, `utilization_3b_result.json`, `utilization_14b_result.json` | batching sweeps and the idle-share economics |
| `edge_result.json`, `edge_result2.json` | the RTX A2000 edge arm |
| `placement_result.json` | the edge-versus-datacenter placement model |

### Agentic iteration

| File | Backs |
|---|---|
| `agent_loop_result.json` | Table 6 and Figure 3: the single-shot, repair, self-review and crossed arms at four model sizes, merged across the two sessions below |
| `agent_loop_result_run1.json` | the first session, all four sizes, retained with its contaminated 7B cells intact |
| `agent_loop_7b_result.json` | the clean 7B re-measurement that replaced them |
| `agent_pilot_result.json` | the 48-task pilot that sized the experiment and caught two design faults before the full run |

A foreign job entered the GPU while 7B was being measured in the first
session. The contention guard of Section 3.2 flagged it: the self-review cell
closed 247 W above its opening baseline and read 174.50 J per task against
92.59 J on clean re-measurement, and the crossed cell returned a negative net
energy, because its own opening baseline had been taken while the intruder was
resident. Those cells were discarded, not corrected.
`scripts/analysis/merge_agent_runs.py` performs the substitution and writes
every discarded cell and its reason into the merged file's `provenance` field.
The two sessions agree where both are clean, which is what licenses splicing
them: 7B single shot read 45.02 against 44.52 J and 7B repair 62.63 against
62.76 J, a 1.1% and 0.2% difference across independent sessions.

`scripts/experiments/validators.py` holds the ground-truth-free checks the
repair loop runs on, and its `selftest()` establishes what they can and cannot
see: across the 120 tasks no correct answer is rejected, malformed output is
caught in every family, and an answer that is well formed and merely wrong is
caught in none. `scripts/validation/agent_loop_selfcheck.py` exercises the
loop's control flow against a stubbed model, so the termination bound and the
check's blindness to a wrong-but-well-formed answer are asserted rather than
described.

### Quantization on the edge card

| File | Backs |
|---|---|
| `edge_quant_result.json` | four configurations on the RTX A2000 in one session: 1.5B at bfloat16 and at NF4, then 3B and 7B at NF4, with per-task grading, per-cell energy and resident idle draw |

All four share one cold-idle baseline and one thermal history, and none of
them carries a drift flag. They run at batch 8, which is what the 7B model
leaves room for in 6 GB; the comparison between configurations is therefore
internally consistent, and its absolute energies are not comparable with the
batch-32 figures elsewhere in the paper.

Two things about accuracy in this file need stating rather than discovering.
The 1.5B bfloat16 cell reads 38.3% where the paper reports 39.2%, one task of
120. The cause is batch composition under left padding, which is testable and
was tested: the same configuration reads 40.8% at batch 40 and 38.3% at batch
8, so this task set is sensitive to grouping at roughly a point and a half
either way. And this arm runs transformers 5.2 on torch 2.5.1, since
bitsandbytes needs a newer stack than the published edge measurements used.

`scripts/analysis/edge_quant_analysis.py` recomputes the placement cost model
of Table 12 from these numbers. The resident idle floors here sit below the
published 12.24 W because the card idles cooler in this session, 5.64 W cold,
so the rows measured here should be read against each other rather than
against the published row.

### Serving framework

vLLM cannot be installed beside the stack the rest of this paper was measured
on, so the comparison runs in three arms and the middle one is what makes it
readable.

| File | Backs |
|---|---|
| `vllm_sweep_hf_result.json` | the published batching sweep's code path re-run on vLLM's own stack: transformers 5.17, torch 2.13, CUDA 13 |
| `vllm_sweep_vllm_result.json` | vLLM 0.31 on that same stack, concurrency 1 to 256, from a cold card |
| `vllm_sweep_vllm_rerun.json` | the same sweep from a hot card, which is what settles the drift flags |

Published Table 8 against `vllm_sweep_hf_result.json` isolates the software
stack; `vllm_sweep_hf_result.json` against `vllm_sweep_vllm_result.json`
isolates continuous batching. Reading vLLM against the published table alone
would credit the framework with whatever the newer kernels did.
`scripts/analysis/vllm_analysis.py` performs both subtractions and recomputes
the idle-share economics from them.

Two configuration decisions are recorded in the result files rather than left
implicit. vLLM runs with its FlashInfer sampler disabled, because that
component JIT-compiles kernels against the host toolkit and this host's nvcc
is CUDA 12.0 while the wheel requires 12.4 or newer; decoding throughout is
greedy, which makes the sampler an argmax either way. And energy at small
batch sizes comes from a replay window rather than per-repeat windows: a
single 128-token generation runs about 1.2 s under vLLM, which is twelve
samples at 10 Hz and under the integration floor of Section 3.2, whose own
remedy is to replay inside one longer window and divide. Each row names which
window its energy came from.

Every cell in both sweeps carries a drift flag, because a sweep heats the card
and the 5 W guard is the naive one Section 3.2's second pitfall warns about.
The two vLLM sweeps, run from a cold and a hot card, agree to between 0.3 and
8.7 per cent with a median of 2.5, which is the same order as the 0.3 to 4.2
per cent the published 14B sweep reproduced to.

### External benchmark

| File | Backs |
|---|---|
| `netarena_sample.json`, `netarena_sample2.json`, `netarena_prompts.json` | the NetArena task draws and prompts as used |
| `netarena_energy_result.json` | NetArena energy measurements |
| `netarena_grade_result.json`, `netarena_grade_result2.json` | NetArena grading output |

## Logs

`data/logs/` holds the run log for each experiment above, including the
queue-wait and contention-guard output that shows whether a cell ran on an
otherwise idle card.

## Stated exception: the raw grid series is not redistributed

`data/results/carbon_intensity_hourly_means.csv` holds the 24 hour-of-day
carbon-intensity means that Section 4.11 and Figure 6 are built from. The
underlying 32-day hourly generation-mix series is not archived here: it comes
from EPIAS under that operator's terms of use.
`scripts/experiments/epias_fetch.py` regenerates it from an EPIAS account.

The consequence is that the hour-of-day figures (348.9 and 411.9 gCO2eq/kWh,
the 1.18x ratio, the 7.0% and 15.3% savings) can be checked from this deposit,
while the raw 32-day range (269-464 gCO2eq/kWh, 1.72x, and the 42% figure)
cannot be without such an account.

## Stated exception: no per-sample power traces

The sampler accumulates its 10 Hz readings in memory and writes out the
integrated energy per measurement cell, not the individual samples. Per-sample
power traces therefore do not exist for the runs reported in the paper, and
none is deposited here. What is deposited is the integrated per-cell energy,
the idle baselines it is net of, and the run logs. Reproducing a raw trace
means re-running `scripts/measurement/power_sampler.py`.

This is recorded here rather than left to be discovered, because the phrase
"raw power-sampler logs" would otherwise suggest per-sample data is available.
