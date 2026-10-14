"""What does a real serving stack do to the numbers in this paper?

Every crossover volume reported here is decided by where dynamic energy meets
the idle power floor, and the measurements behind them call
`transformers.generate()` directly: no continuous batching, no paged KV cache,
no cross-request scheduling. A production deployment runs vLLM, which moves
both sides of that meeting point.

Measuring that honestly needs three arms, not two. vLLM brings its own
software stack -- torch 2.13 and CUDA 13 against the paper's torch 2.4.1 and
CUDA 12.1 -- so a bare vLLM-versus-published comparison confounds the serving
framework with the kernels underneath it, which is exactly the kind of
confident wrong number Section 3.4 is a list of. So:

    a  published Table 7      transformers 4.44.2, torch 2.4.1, cu121
    b  ARM=hf   (here)        transformers 5.17,   torch 2.13,  cu130
    c  ARM=vllm (here)        vLLM 0.31 on the same stack as b

b minus a is the stack. c minus b is the serving framework. Arm b runs the
published sweep's code path unchanged: a padded batch handed to one
generate() call, 128 forced output tokens, same prompts, same batch sizes.

Four quantities come out, in the order they matter:

1. The resident idle floor. vLLM pre-allocates its KV cache at startup,
   holding most of the card before a request arrives. Allocated memory that
   is not accessed should cost little, but the idle-dominance headline rests
   on that floor, so it is measured rather than assumed. The comparison point
   is Section 4.5's +0.23 W for 3B weights resident.
2. Dynamic energy per request at matched work, swept over concurrency.
3. Throughput, which converts those into per-day volumes.
4. Where the two meet, which decides whether the 93-99.9% idle band and the
   placement crossover survive a serving framework.

The two arms run as separate processes because they cannot share the card:
vLLM holds most of it by design. For the same reason this refuses to start
while another job is resident rather than evicting it.

    ARM=hf   OUT=vllm_sweep_hf.json   python vllm_sweep.py
    ARM=vllm OUT=vllm_sweep_vllm.json python vllm_sweep.py
"""
import gc
import json
import os
import statistics as st
import subprocess
import time
import traceback

os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
# FlashInfer JIT-compiles its sampling kernels against the host toolkit,
# and this host's nvcc is CUDA 12.0 while the wheel expects 12.4 or newer,
# so the build fails on an option nvcc 12.0 does not know. Decoding here is
# greedy, which makes the sampler an argmax either way, so the kernels are
# switched off rather than worked around. Recorded in the result file.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# vLLM shells out to ninja; the venv's bin has to be on PATH for that.
os.environ["PATH"] = os.pathsep.join(
    [os.path.dirname(os.sys.executable), os.environ.get("PATH", "")])

from power_sampler import PowerSampler
from tasks_net import build

ARM = os.environ.get("ARM", "vllm")
MODELS = [
    ("Qwen/Qwen2.5-3B-Instruct", 3.0),
    ("Qwen/Qwen2.5-14B-Instruct", 14.0),
]
# The published sweep stops at 64 because 14B could not run it. vLLM can, and
# that it can is part of the finding, so the vLLM arm carries two more rungs.
BATCHES_HF = [1, 2, 4, 8, 16, 32, 64]
BATCHES_VLLM = [1, 2, 4, 8, 16, 32, 64, 128, 256]
GEN = 128                 # forced output tokens, matched to Table 7
REPEATS = int(os.environ.get("REPEATS", 3))
GPU_UTIL = float(os.environ.get("GPU_UTIL", 0.90))
MAX_LEN = int(os.environ.get("MAX_MODEL_LEN", 2048))
MIN_SAMPLES = 30          # Section 3.4's integration floor
DRIFT_W = 5.0
MY_PID = os.getpid()
DESKTOP = "gnome-remote-desktop"
OUT = os.environ.get("OUT", "vllm_sweep_%s_result.json" % ARM)


def foreign_pids():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=15).stdout
    except Exception:
        return []
    found = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid == MY_PID or DESKTOP in parts[1]:
            continue
        found.append((pid, parts[1]))
    return found


def wait_for_exclusive(max_h=12):
    """vLLM takes most of the card at startup. Starting beside another job
    would corrupt this measurement and starve that one, so hold instead."""
    deadline = time.time() + max_h * 3600
    clean = 0
    while time.time() < deadline:
        fp = foreign_pids()
        if not fp:
            clean += 1
            if clean >= 3:
                return True
            time.sleep(15)
        else:
            clean = 0
            print("  [queue] busy: %s" % (fp,), flush=True)
            time.sleep(60)
    return False


def baseline(s, seconds, label):
    t0 = time.time()
    time.sleep(seconds)
    t1 = time.time()
    w = s.mean_w(t0, t1)
    series = [smp[1][0] for smp in s.window(t0, t1)]
    sd = st.pstdev(series) if len(series) > 1 else 0.0
    print("    [%s] %7.2f W (sd %.2f, n=%d)" % (label, w[0], sd, len(series)),
          flush=True)
    return w[0], sd


def timed(s, fn, pre_w):
    """One measured window. Returns the window's record."""
    a = time.time()
    fn()
    b = time.time()
    n = len(s.window(a, b))
    ej = s.energy_j(a, b)
    return {"dur_s": b - a, "n_samples": n,
            "net_j": None if (ej is None or n < MIN_SAMPLES)
                     else ej[0] - pre_w * (b - a)}


def repeat_window(s, fn, pre_w, repeats, gap=1.5):
    """Replay inside one window and divide, plus the per-replay windows.

    A single 128-token generation at batch 1 runs about 2.7 s on this stack,
    which is 27 samples at 10 Hz: under the floor Section 3.4 sets, so the
    published sweep's per-repeat windows would simply refuse the cell. That
    section's own remedy is to replay inside one longer window and divide by
    the number of replays, which is what the outer window here does. The
    inner windows are retained so the per-repeat spread is available wherever
    it is measurable.
    """
    outer_a = time.time()
    per = []
    for i in range(repeats):
        per.append(timed(s, fn, pre_w))
        if i != repeats - 1:
            time.sleep(gap)
    outer_b = time.time()
    n = len(s.window(outer_a, outer_b))
    ej = s.energy_j(outer_a, outer_b)
    dur = outer_b - outer_a
    outer = {"dur_s": dur, "n_samples": n,
             "net_j": None if (ej is None or n < MIN_SAMPLES)
                      else ej[0] - pre_w * dur}
    return per, outer


def summarise(batch, per, outer, pre_w, post_w):
    """The replay window is the reported figure; the per-repeat windows give
    the spread where they are long enough to be integrated at all."""
    good = [p for p in per if p["net_j"] is not None]
    dur = st.mean([p["dur_s"] for p in per])
    jb = (outer["net_j"] / len(per)) if outer["net_j"] is not None else None
    src = "replay_window"
    if jb is None and good:
        jb = st.mean([p["net_j"] for p in good])
        src = "per_repeat_mean"
    drift = abs(post_w - pre_w)
    return {"batch": batch, "dur_s": dur, "n_measurable": len(good),
            "energy_from": src if jb is not None else None,
            "replay_samples": outer["n_samples"],
            "j_batch": jb, "j_req": (jb / batch) if jb is not None else None,
            "j_token": (jb / (batch * GEN)) if jb is not None else None,
            "j_batch_sd": (st.pstdev([p["net_j"] for p in good])
                           if len(good) > 1 else None),
            "req_per_s": batch / dur, "tok_per_s": batch * GEN / dur,
            "pre_w": pre_w, "post_w": post_w, "drift_w": drift,
            "dirty": drift > DRIFT_W}


def show(row):
    """Say which window the energy came from, since at small batch sizes the
    per-repeat windows fall under the integration floor and only the replay
    window survives. Printing a number next to "below floor" would read as a
    contradiction."""
    src = {"replay_window": "replay", "per_repeat_mean": "per-rep",
           None: "none"}[row["energy_from"]]
    print("    batch %3d: %6.2f s  %9.2f J/req  %7.2f req/s  "
          "drift %4.1f W  energy from %s (%d/%d windows over floor)%s"
          % (row["batch"], row["dur_s"],
             row["j_req"] if row["j_req"] is not None else float("nan"),
             row["req_per_s"], row["drift_w"], src,
             row["n_measurable"], REPEATS,
             "  DIRTY" if row["dirty"] else ""),
          flush=True)


# ------------------------------------------------------------------ arm: hf

def arm_hf(s, repo, prompts, cold_idle):
    """The published sweep's code path, on the new stack."""
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM

    entry = {"arm": "hf", "transformers": transformers.__version__,
             "torch": torch.__version__, "cuda": torch.version.cuda}

    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    tok.padding_side = "left"          # decoder-only batching needs it
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:                               # torch_dtype was renamed in 4.56
        model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.bfloat16, device_map="cuda:0",
            trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map="cuda:0",
            trust_remote_code=True)
    model.eval()
    cfg = model.generation_config
    cfg.do_sample = False
    for k in ("temperature", "top_p", "top_k"):
        setattr(cfg, k, None)
    if cfg.pad_token_id is None:
        cfg.pad_token_id = tok.eos_token_id

    resident, _ = baseline(s, 20, "resident idle")
    entry.update(cold_idle_w=cold_idle, resident_idle_w=resident,
                 resident_minus_cold_w=resident - cold_idle)
    print("    weights resident cost %+.2f W over cold"
          % (resident - cold_idle), flush=True)

    rows, oom_at = [], None
    for B in BATCHES_HF:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False,
                                         add_generation_prompt=True)
                 for p in prompts[:B]]
        try:
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
            with torch.inference_mode():
                model.generate(**enc, max_new_tokens=4, min_new_tokens=4)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            oom_at = B
            print("    batch %3d: OOM -- concurrency ceiling" % B, flush=True)
            gc.collect(); torch.cuda.empty_cache()
            break
        time.sleep(1.5)

        pre_w, _ = baseline(s, 10, "b=%d pre" % B)

        def run():
            with torch.inference_mode():
                model.generate(**enc, max_new_tokens=GEN, min_new_tokens=GEN)
            torch.cuda.synchronize()

        per, outer = repeat_window(s, run, pre_w, REPEATS)
        post_w, _ = baseline(s, 10, "b=%d post" % B)
        row = summarise(B, per, outer, pre_w, post_w)
        row["vram_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        rows.append(row)
        show(row)

    entry["rows"] = rows
    entry["oom_at"] = oom_at
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return entry


# ---------------------------------------------------------------- arm: vllm

def arm_vllm(s, repo, prompts, cold_idle):
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    entry = {"arm": "vllm", "vllm": vllm.__version__,
             "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
             "torch": torch.__version__, "cuda": torch.version.cuda,
             "gpu_util": GPU_UTIL, "max_model_len": MAX_LEN}

    sp = SamplingParams(temperature=0.0, max_tokens=GEN, min_tokens=GEN,
                        ignore_eos=True)
    t0 = time.time()
    llm = LLM(model=repo, dtype="bfloat16", gpu_memory_utilization=GPU_UTIL,
              max_model_len=MAX_LEN, enforce_eager=False)
    print("    engine up in %.1f s" % (time.time() - t0), flush=True)

    # Graph capture and the first allocations happen on the first request, so
    # warm before taking the resident floor.
    llm.generate(prompts[:2], sp)
    time.sleep(3)
    resident, _ = baseline(s, 20, "resident idle")
    entry.update(cold_idle_w=cold_idle, resident_idle_w=resident,
                 resident_minus_cold_w=resident - cold_idle)
    print("    engine resident costs %+.2f W over cold"
          % (resident - cold_idle), flush=True)

    rows = []
    for B in BATCHES_VLLM:
        batch = [prompts[i % len(prompts)] for i in range(B)]
        pre_w, _ = baseline(s, 10, "b=%d pre" % B)
        per, outer = repeat_window(s, lambda: llm.generate(batch, sp), pre_w,
                                   REPEATS)
        post_w, _ = baseline(s, 10, "b=%d post" % B)
        row = summarise(B, per, outer, pre_w, post_w)
        rows.append(row)
        show(row)

    entry["rows"] = rows
    del llm
    gc.collect()
    return entry


# ------------------------------------------------------------------- driver

def main():
    if ARM not in ("hf", "vllm"):
        raise SystemExit("ARM must be hf or vllm, got %r" % ARM)
    tasks = build(n_per_family=40)
    prompts = [t["prompt"] for t in tasks]
    print("arm=%s, %d prompts, %d forced tokens, %d repeats"
          % (ARM, len(prompts), GEN, REPEATS), flush=True)

    print("waiting for an exclusive GPU ...", flush=True)
    if not wait_for_exclusive():
        print("gave up waiting")
        return

    s = PowerSampler(interval_ms=100).start()
    cold_idle, _ = baseline(s, 20, "cold idle")

    results = []
    for repo, pb in MODELS:
        print("\n=== %s (%gB) [%s] ===" % (repo, pb, ARM), flush=True)
        entry = {"repo": repo, "params_b": pb}
        try:
            if foreign_pids():
                print("    foreign job on GPU -- waiting", flush=True)
                wait_for_exclusive()
            fn = arm_hf if ARM == "hf" else arm_vllm
            entry.update(fn(s, repo, prompts, cold_idle))
        except Exception as exc:
            entry["error"] = "%s: %s" % (type(exc).__name__, exc)
            print("    FAILED: %s" % entry["error"], flush=True)
            traceback.print_exc()
        finally:
            gc.collect()
            time.sleep(10)
        results.append(entry)
        json.dump({"arm": ARM, "cold_idle": cold_idle, "gen_tokens": GEN,
                   "repeats": REPEATS, "results": results},
                  open(OUT, "w"), indent=1)

    print("\n=== %s SWEEP ===" % ARM.upper())
    print("%-8s %6s %10s %9s %11s" % ("model", "batch", "J/req", "req/s",
                                      "J/token"))
    for e in results:
        for r in e.get("rows", []):
            if r.get("j_req") is None:
                continue
            print("%-8s %6d %10.2f %9.2f %11.4f"
                  % ("%gB" % e["params_b"], r["batch"], r["j_req"],
                     r["req_per_s"], r["j_token"]))
    s.stop()


if __name__ == "__main__":
    main()
