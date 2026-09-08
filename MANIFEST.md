# Manifest

Everything the Data Availability statement promises is present, with one
stated exception recorded at the end. `scripts/validation/verify_deposit.py`
recomputes the paper's headline statistics from these files and compares them
against the printed values; run it from the repository root.

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

## Stated exception: no per-sample power traces

The sampler accumulates its 10 Hz readings in memory and writes out the
integrated energy per measurement cell, not the individual samples. Per-sample
power traces therefore do not exist for the runs reported in the paper, and
none is deposited here. What is deposited is the integrated per-cell energy,
the idle baselines it is net of, and the run logs. Reproducing a raw trace
means re-running `scripts/measurement/power_sampler.py`.

This is recorded here rather than left to be discovered, because the phrase
"raw power-sampler logs" would otherwise suggest per-sample data is available.
