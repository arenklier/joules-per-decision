# Changelog

This deposit is versioned on Zenodo. The concept DOI
[10.5281/zenodo.22666116](https://doi.org/10.5281/zenodo.22666116) always
resolves to the newest version; each version also has its own DOI, which
does not move.

## v1.2.0 — 2026-10-28

Three experiments the first deposit predates, and the files that back them.

### Agentic iteration

`scripts/experiments/agent_loop.py` crosses self-review against repair on the
same 120 tasks, at four model sizes, against the single-shot baseline.

The load-bearing decision is what the repair check is allowed to see.
`scripts/experiments/validators.py` derives every check from the prompt alone,
in the manner of a configuration linter: a proposed address must lie inside
the network being subnetted and be aligned to the requested prefix length, a
chosen route must be one of the routes offered, an ACL verdict must be one of
the two verdicts. The subnet index the task states is deliberately unused,
because combining it with the alignment rule would determine the answer and
the check would be the grader in disguise. Running that file executes its
`selftest()`, which establishes across all 120 tasks that no correct answer is
rejected, malformed output is caught in every family, and a well-formed wrong
answer is caught in none, bar one subnetting task in forty.

- `data/results/agent_loop_result.json` — the merged table
- `data/results/agent_loop_result_run1.json` — the first session, kept with
  its contaminated 7B cells intact
- `data/results/agent_loop_7b_result.json` — the clean re-measurement that
  replaced them
- `data/results/agent_pilot_result.json` — the 48-task pilot that sized the
  experiment and caught two design faults before the full run
- `scripts/analysis/merge_agent_runs.py` — performs the substitution and
  writes every discarded cell and its reason into the merged file
- `scripts/validation/agent_loop_selfcheck.py` — the loop's control flow
  against a stubbed model

### Serving framework

`scripts/experiments/vllm_sweep.py`, in three arms rather than two, because
vLLM cannot be installed beside the stack the rest of the paper runs on. The
middle arm re-runs the published sweep's own code path on vLLM's stack, so
the software and the framework can be subtracted from each other. A two-arm
comparison would credit continuous batching with whatever the newer kernels
did.

- `data/results/vllm_sweep_hf_result.json` — the middle arm
- `data/results/vllm_sweep_vllm_result.json` — vLLM 0.31, from a cold card
- `data/results/vllm_sweep_vllm_rerun.json` — the same sweep from a hot card,
  which is what settles the thermal-drift flags
- `scripts/analysis/vllm_analysis.py`

### Quantization on the edge card

`scripts/experiments/edge_quant.py` measures four configurations on the RTX
A2000 in one session against one cold baseline: 1.5B at bfloat16 and at NF4,
then 3B and 7B at NF4.

- `data/results/edge_quant_result.json`
- `scripts/analysis/edge_quant_analysis.py`

### Corrections to the deposit itself

- **The MANIFEST pointed at the wrong tables.** Adding three sections to the
  paper moved six of its cross-references: the placement cost model became
  Table 12, not 10; the batching sweep Table 8, not 7; the pitfalls Section
  3.2, not 3.4; the carbon analysis Section 4.11 and Figure 6, not 4.8 and 5.
  All corrected, and `tools/audit/manifest_refs.py` now pins each reference to
  the LaTeX label it means and fails if the numbering moves again.
- **The README described one software stack and no serving framework.** That
  stopped being true with the two arms above. It now names all three stacks
  and says energies are comparable only within the arm that measured them.
- **Five figure PDFs were deposited twice.** `make_figures.py` wrote into the
  current directory, so running it from `scripts/analysis/` left a second,
  byte-identical copy there. Both figure scripts now resolve `figures/` from
  their own location; the duplicates are removed. The figures themselves are
  unchanged — regenerated output matches the deposited files content for
  content, differing only in the timestamp matplotlib embeds.
- **Two validation scripts could not run where they were deposited.**
  `oracle_validate.py` raised `ModuleNotFoundError`, and it is the script
  backing the paper's "120 of 120 agree" claim; `netarena_selfcheck.py` needed
  a NetArena checkout and gave a traceback instead of saying so.
- `CITATION.cff` now carries the concept DOI rather than v1.0.0's version
  DOI, and both authors' ORCIDs.

### Note on v1.1.0

A v1.1.0 archive was built on 2026-10-12 and never deposited. Its contents —
the hourly carbon-intensity means, the colour-corrected figures, and Figure 2
redrawn against the adjusted p-values — are included here. The public record
therefore goes v1.0.0 to v1.2.0.

## v1.0.0 — 2026-09-08

First deposit: power sampler, task generator and grading function, the
independent ground-truth re-derivation, the grading-rule audit, per-task
result files, run logs, statistical analyses, and figure generation, for the
scaling ladder, the reasoning tax, the output-format mechanism, the solver
baseline, token-role decomposition, batching and idle-power economics, the
placement sweep, carbon-aware scheduling, and the NetArena arm.
