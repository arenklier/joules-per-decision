# Joules per Decision

Measurement code, raw results, and analysis scripts for *Joules per Decision:
Measuring the Energy Cost of LLM Agents in Network Operations*.

The paper measures GPU board power while language models perform network
operations tasks — subnet arithmetic, ACL rule evaluation, and BGP best-path
selection — across four Qwen2.5 sizes on a datacenter accelerator and an
edge-class card, and on NetArena, an externally maintained benchmark.

## What this repository is for

The paper offers a measurement protocol and a set of measurement pitfalls as
contributions in their own right. Both are worth more if a reader can re-run
them than if they have to be taken on trust, which is why the sampler, the
task generator, the grader, and the per-task result files are all here rather
than only the plots.

`MANIFEST.md` lists what is present and what is still outstanding. Read it
before assuming a table can be reproduced from this deposit.

## Layout

```
scripts/measurement/   power sampler, idle-baseline probes, energy scaling ladder
scripts/experiments/   task generator, prompting conditions, batching and edge sweeps,
                       NetArena runners, grid carbon-intensity fetch
scripts/validation/    independent ground-truth re-derivation, grading-rule audit,
                       NetArena reference-solution self-check
scripts/analysis/      statistics, placement model, figure generation
data/results/          per-task and per-cell result files
data/logs/             run logs
figures/               the figures as they appear in the paper
```

## Measurement method in one paragraph

Energy comes from a single streaming `nvidia-smi` process sampling
instantaneous board power at 10 Hz, integrated over wall-clock time by the
trapezoidal rule. Every reported figure is net of an idle baseline measured
immediately before, and where practical after, each measurement cell. The
instrument sees GPU board power and nothing else: host CPU and DRAM draw,
power-supply losses, chassis cooling, and facility overhead are all outside
it. Every joule, watt, kilowatt-hour, and kilogram of CO2eq in the paper is
therefore a board-level figure, not a whole-system or whole-site one.

## Five pitfalls this code guards against

Each of these produced a confidently wrong number before it was caught, and
two of them inverted a conclusion:

1. **Contention.** Another process on the card inflates energy by up to 2.9x.
   Guard: check the compute-process list before and after every cell, and take
   a fresh idle baseline per cell, since a global baseline cannot detect
   contention that starts partway through a run.
2. **Thermal drift.** The idle floor climbs several watts under sustained load
   and settles back. It is not contention and must not be guarded as such, or
   good cells get discarded.
3. **Sub-second windows.** A direct answer can finish in 0.1–0.3 s, which at
   10 Hz is one to three samples — below the sampler's noise floor, and capable
   of returning negative energy. Guard: replay a whole prompt set inside one
   multi-second window, and refuse to report any window under ~30 samples.
4. **Truncation scored as error.** An output cap set too low truncates long
   generations, and each truncated generation scores as incorrect. At a
   512-token cap this yielded "chain-of-thought harms accuracy at every
   scale"; at 1024 tokens the same model reached the opposite conclusion.
5. **Clock ramp.** Energy measured before clocks settle is not comparable to
   energy measured after.

## Reproducing

The scripts expect an NVIDIA GPU with `nvidia-smi`, PyTorch 2.4.1 (CUDA 12.1),
and HuggingFace `transformers` 4.44.2. Models run in bfloat16 with sampling
disabled. No serving framework is used, so there is no continuous batching and
no cross-request prefix cache; batching is a padded batch passed to one
`generate()` call.

Hardware in the paper: NVIDIA L40S (46 GB) as the datacenter proxy, NVIDIA RTX
A2000 (6 GB) as the gateway-class edge proxy, driver 580.173.02.

Absolute joule figures will not reproduce on different hardware. Ratios and
multipliers, which are most of what the paper reports, should.

## Citation

See `CITATION.cff`.
