"""Quantization on the edge card: does a bigger, cheaper model change the answer?

The placement result rests on one accuracy gap. The edge arm runs Qwen2.5-1.5B
in bfloat16 because that is what fits in the A2000's 6 GB, and it scores 39.2%
against the datacenter card's 50.0%. The paper notes that a task where the
larger model's advantage is much wider could flip the ranking, and separately
that the scaling law was never shown to hold at the quantized precisions where
edge deployment actually lives.

Both gaps close the same way. Four bits per weight puts a 7B model inside 6 GB,
so the question is no longer which model fits but which quantized model buys
the most correct decisions per joule. The answer decides whether the edge arm
in the placement analysis was underselling itself.

Four configurations, all on the same card and the same 120 tasks:

    1.5B bf16   the published edge arm, re-measured here as the baseline
    1.5B nf4    what quantization costs at fixed model size
    3B   nf4    what the freed memory buys
    7B   nf4    the largest that fits at four bits

Reporting all four separates two effects that a single quantized cell would
confound: quantization loses accuracy at fixed size, and the size it affords
gains it back. Only the net matters for placement, but only the split explains
it.

The protocol is the one used throughout: a streaming nvidia-smi at 10 Hz,
trapezoidal integration, a per-cell idle baseline subtracted, and a refusal to
report a window under the sample floor. Two things specific to this machine.
It has two cards, which is where Section 3.4's fourth pitfall came from, so the
sampler is pointed at the A2000's nvidia-smi index rather than at whatever
index CUDA chose. And it is a workstation someone uses, so a burst of interface
work during a cell shows up as baseline drift and the cell says so.
"""
import gc
import json
import os
import statistics as st
import subprocess
import time
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from power_sampler import PowerSampler
from tasks_net import build, grade

# nvidia-smi's index for the A2000. CUDA's own ordering is not the same thing
# on this machine, and conflating them is the fourth pitfall.
GPU_IDX = int(os.environ.get("GPU_IDX", 1))
GEN_CAP = int(os.environ.get("GEN_CAP", 48))      # the direct condition's cap
BATCH = int(os.environ.get("BATCH", 8))
REPEATS = int(os.environ.get("REPEATS", 3))
MIN_SAMPLES = 30
DRIFT_W = 2.0        # the A2000 idles near 8 W, so the guard scales with it
SETTLE_S = int(os.environ.get("SETTLE_S", 20))
OUT = os.environ.get("OUT", "edge_quant_result.json")

CONFIGS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5, "bf16"),
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5, "nf4"),
    ("Qwen/Qwen2.5-3B-Instruct", 3.0, "nf4"),
    ("Qwen/Qwen2.5-7B-Instruct", 7.0, "nf4"),
]

# Verbatim from robustness_run.py, which produced the published accuracies.
# Using only the first of the three gives 35.0% at 1.5B against the 39.2% the
# paper reports, because the published figure is the mean over all three; the
# templates are part of the measurement, not decoration.
DIRECT_TEMPLATES = [
    "\n\nDo not explain. Reply with exactly one line in the form ANSWER: <value>",
    "\n\nGive only the final answer, with no working. Format: ANSWER: <value>",
    "\n\nRespond with a single line containing just the result: ANSWER: <value>",
]

# Settable so a pilot runs the experiment itself at reduced size, and so one
# configuration can be re-measured later without re-running the rest.
_only = [w.strip() for w in os.environ.get("ONLY_CONFIGS", "").split(",")
         if w.strip()]
if _only:
    CONFIGS = [c for c in CONFIGS
               if "%g%s" % (c[1], c[2]) in _only or c[2] in _only]


def baseline(s, seconds, label):
    t0 = time.time()
    time.sleep(seconds)
    t1 = time.time()
    w = s.mean_w(t0, t1)[GPU_IDX]
    series = [smp[1][GPU_IDX] for smp in s.window(t0, t1)]
    sd = st.pstdev(series) if len(series) > 1 else 0.0
    print("    [%s] %6.2f W (sd %.2f, n=%d)" % (label, w, sd, len(series)),
          flush=True)
    return w, sd


def card_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            idx, name = [p.strip() for p in line.split(",")]
            if int(idx) == GPU_IDX:
                return name
    except Exception:
        pass
    return "unknown"


def load_model(repo, precision):
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kw = dict(device_map={"": 0}, trust_remote_code=True)
    if precision == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    else:
        kw["dtype"] = torch.bfloat16

    try:
        model = AutoModelForCausalLM.from_pretrained(repo, **kw)
    except TypeError:                      # dtype was torch_dtype before 4.56
        if "dtype" in kw:
            kw["torch_dtype"] = kw.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(repo, **kw)

    model.eval()
    cfg = model.generation_config
    cfg.do_sample = False
    for k in ("temperature", "top_p", "top_k"):
        setattr(cfg, k, None)
    if cfg.pad_token_id is None:
        cfg.pad_token_id = tok.eos_token_id
    return tok, model


def run_batches(tok, model, tasks, batch):
    """Grade every task, in padded batches. Accuracy only; energy is timed
    separately so that grading overhead stays outside the window."""
    correct, rows = 0, []
    for i in range(0, len(tasks), batch):
        chunk = tasks[i:i + batch]
        texts = [tok.apply_chat_template(
            [{"role": "user",
              "content": t["prompt"] + DIRECT_TEMPLATES[t["tpl"]]}],
            tokenize=False, add_generation_prompt=True) for t in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=GEN_CAP)
        n_in = enc["input_ids"].shape[-1]
        for t, o in zip(chunk, out):
            text = tok.decode(o[n_in:], skip_special_tokens=True)
            ok, got = grade(text, t["gt"])
            correct += bool(ok)
            rows.append({"task": t["id"], "family": t["family"],
                         "correct": bool(ok), "got": got})
    return correct / len(tasks), rows


def energy_cell(s, tok, model, tasks, batch, pre_w):
    """One replay window over a fixed batch, divided by the replays.

    A single batch finishes in a couple of seconds on this card, which is
    under the integration floor, so the replays go inside one window the way
    Section 3.4 prescribes rather than each getting their own.
    """
    texts = [tok.apply_chat_template(
        [{"role": "user",
          "content": t["prompt"] + DIRECT_TEMPLATES[t["tpl"]]}],
        tokenize=False, add_generation_prompt=True) for t in tasks[:batch]]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
    with torch.inference_mode():           # warm this shape
        model.generate(**enc, max_new_tokens=4, min_new_tokens=4)
    torch.cuda.synchronize()
    time.sleep(1.5)

    a = time.time()
    for _ in range(REPEATS):
        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=GEN_CAP,
                           min_new_tokens=GEN_CAP)
        torch.cuda.synchronize()
    b = time.time()

    n = len(s.window(a, b))
    ej = s.energy_j(a, b)
    dur = b - a
    net = None if (ej is None or n < MIN_SAMPLES) else ej[GPU_IDX] - pre_w * dur
    return {"dur_s": dur, "n_samples": n, "batch": batch, "repeats": REPEATS,
            "net_j": net,
            "j_per_req": None if net is None else net / (REPEATS * batch),
            "req_per_s": (REPEATS * batch) / dur}


def main():
    tasks = build(n_per_family=40)
    for i, t in enumerate(tasks):          # the published round-robin
        t["tpl"] = i % 3
    name = card_name()
    print("edge card: %s (nvidia-smi index %d), %d tasks, batch %d, cap %d"
          % (name, GPU_IDX, len(tasks), BATCH, GEN_CAP), flush=True)
    print("torch sees: %s" % torch.cuda.get_device_name(0), flush=True)

    s = PowerSampler(interval_ms=100).start()
    cold_idle, cold_sd = baseline(s, 20, "cold idle")

    results = []
    for repo, pb, precision in CONFIGS:
        label = "%gB %s" % (pb, precision)
        print("\n=== %s ===" % label, flush=True)
        entry = {"repo": repo, "params_b": pb, "precision": precision,
                 "card": name, "cold_idle_w": cold_idle}
        tok = model = None
        try:
            t0 = time.time()
            tok, model = load_model(repo, precision)
            entry["load_s"] = time.time() - t0
            entry["vram_gib"] = torch.cuda.memory_allocated() / 2 ** 30
            print("    loaded in %.1f s, %.2f GiB resident"
                  % (entry["load_s"], entry["vram_gib"]), flush=True)

            resident, _ = baseline(s, 20, "resident idle")
            entry["resident_idle_w"] = resident
            entry["resident_minus_cold_w"] = resident - cold_idle

            pre_w, _ = baseline(s, 12, "cell pre")
            cell = energy_cell(s, tok, model, tasks, BATCH, pre_w)
            time.sleep(SETTLE_S)
            post_w, _ = baseline(s, 12, "cell post")
            cell.update(pre_w=pre_w, post_w=post_w,
                        drift_w=abs(post_w - pre_w),
                        dirty=abs(post_w - pre_w) > DRIFT_W)
            entry["energy"] = cell
            print("    %6.2f J/req  %5.2f req/s  drift %.2f W%s"
                  % (cell["j_per_req"] if cell["j_per_req"] else float("nan"),
                     cell["req_per_s"], cell["drift_w"],
                     "  DIRTY" if cell["dirty"] else ""), flush=True)

            acc, rows = run_batches(tok, model, tasks, BATCH)
            entry["acc"] = acc
            entry["rows"] = rows
            jc = (cell["j_per_req"] / acc) if (acc and cell["j_per_req"]) \
                else None
            entry["j_per_correct"] = jc
            print("    accuracy %5.1f%%   %s J per correct decision"
                  % (acc * 100, "%.1f" % jc if jc else "-"), flush=True)
        except Exception as exc:
            entry["error"] = "%s: %s" % (type(exc).__name__, exc)
            print("    FAILED: %s" % entry["error"], flush=True)
            traceback.print_exc()
        finally:
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
        results.append(entry)
        json.dump({"card": name, "gpu_idx": GPU_IDX, "cold_idle": cold_idle,
                   "batch": BATCH, "gen_cap": GEN_CAP, "results": results},
                  open(OUT, "w"), indent=1)

    print("\n=== EDGE QUANTIZATION ===")
    print("%-12s %8s %9s %10s %12s %9s"
          % ("config", "VRAM", "accuracy", "J/req", "J/correct", "idle"))
    for e in results:
        if "acc" not in e:
            continue
        print("%-12s %7.2fG %8.1f%% %10.2f %12s %7.2fW"
              % ("%gB %s" % (e["params_b"], e["precision"]), e["vram_gib"],
                 e["acc"] * 100, e["energy"]["j_per_req"],
                 "%.1f" % e["j_per_correct"] if e["j_per_correct"] else "-",
                 e["resident_idle_w"]))
    s.stop()


if __name__ == "__main__":
    main()
