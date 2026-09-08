"""Edge arm, third attempt -- with a protocol built for a workstation card.

Two runs on the A2000 both came back flagged, and the diagnosis was not what I
first assumed. It is not the operator's applications: the baseline taken BEFORE
the model loads is clean in both runs (sd 0.09), while the one taken after
loading is elevated and noisy (sd 6.7). That is the card's own power management
with a live CUDA context, not outside contention.

The second symptom pins it down further. Batch 1 measured 9.4 s in one run and
30.6 s in the other for identical work, while batch 32 agreed to 8.7% -- the
first and shortest cell is the unstable one. A four-token warm-up does not lift
this card off its idle clocks, so the first real measurement pays for the ramp.

So: warm hard before anything is measured, take the resident baseline only once
the card is in its steady active state, keep it warm between cells, shuffle the
batch order so any residual ramp cannot masquerade as a batch-size effect, and
record SM clocks per cell so the diagnosis is evidenced rather than asserted.
"""
import gc, json, os, random, statistics as st, subprocess, sys, time, traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from power_sampler import PowerSampler
from tasks_net import build

REPO = os.environ.get("REPO", "Qwen/Qwen2.5-1.5B-Instruct")
GPU_IDX = int(os.environ.get("GPU_IDX", 1))
BATCHES = [1, 2, 4, 8, 16, 32]
GEN = int(os.environ.get("GEN", 128))
REPEATS = 5
WARM_S = 30.0
MIN_SAMPLES = 30
DRIFT_W = 2.0
OUT = os.environ.get("OUT", "edge_result2.json")
SUFFIX = "\n\nAnswer in the form ANSWER: <value>"


def clocks():
    """SM clock and performance state -- evidence for the ramp diagnosis."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,pstate,temperature.gpu",
             "--format=csv,noheader,nounits", "-i", str(GPU_IDX)],
            capture_output=True, text=True, timeout=10).stdout.strip()
        sm, ps, temp = [x.strip() for x in out.split(",")]
        return int(sm), ps, int(temp)
    except Exception:
        return None, None, None


def baseline(s, sec, label):
    t0 = time.time(); time.sleep(sec); t1 = time.time()
    ser = [x[1][GPU_IDX] for x in s.window(t0, t1)]
    if not ser:
        return None, None
    w, sd = st.mean(ser), (st.pstdev(ser) if len(ser) > 1 else 0.0)
    sm, ps, temp = clocks()
    print(f"  [{label}] {w:6.2f} W (sd {sd:.2f}, n={len(ser)})  "
          f"sm {sm} MHz, {ps}, {temp}C", flush=True)
    return w, sd


def energy(s, a, b, idle_w):
    win = s.window(a, b)
    if len(win) < MIN_SAMPLES:
        return None, len(win)
    tot = 0.0
    for (ta, pa), (tb, pb) in zip(win, win[1:]):
        tot += 0.5 * (pa[GPU_IDX] + pb[GPU_IDX]) * (tb - ta)
    return tot - idle_w * (b - a), len(win)


def encode(tok, prompts):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True)
             for p in prompts]
    return tok(texts, return_tensors="pt", padding=True).to("cuda:0")


def main():
    tasks = build(n_per_family=40)
    prompts = [t["prompt"] + SUFFIX for t in tasks]
    print(f"edge arm v2: {REPO} on {torch.cuda.get_device_name(0)}", flush=True)

    s = PowerSampler(interval_ms=100).start()
    cold_idle, cold_sd = baseline(s, 20, "cold idle (no CUDA context)")

    tok = AutoTokenizer.from_pretrained(REPO)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16)
    model = model.to("cuda:0").eval()
    cfg = model.generation_config
    cfg.do_sample = False
    for k in ("temperature", "top_p", "top_k"):
        setattr(cfg, k, None)
    cfg.pad_token_id = tok.pad_token_id
    print(f"  loaded, {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    # ---- hard warm-up: run real work until the clocks stop climbing ---------
    print(f"  warming for {WARM_S:.0f} s of continuous generation ...", flush=True)
    warm_enc = encode(tok, prompts[:4])
    t_warm = time.time()
    ramp = []
    while time.time() - t_warm < WARM_S:
        with torch.inference_mode():
            model.generate(**warm_enc, max_new_tokens=GEN, min_new_tokens=GEN)
        torch.cuda.synchronize()
        ramp.append(clocks()[0])
    print(f"  sm clock during warm-up: {ramp}", flush=True)

    pre_w, pre_sd = baseline(s, 15, "resident idle (warm)")

    order = list(BATCHES)
    random.Random(20260906).shuffle(order)
    print(f"  measurement order (shuffled): {order}", flush=True)

    rows, oom_at = [], None
    for B in order:
        try:
            enc = encode(tok, prompts[:B])
            with torch.inference_mode():          # keep clocks up, shape warm
                model.generate(**enc, max_new_tokens=GEN, min_new_tokens=GEN)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            oom_at = B
            print(f"  batch {B:3d}: OOM -- concurrency ceiling", flush=True)
            gc.collect(); torch.cuda.empty_cache()
            continue

        per, sms = [], []
        for _ in range(REPEATS):
            torch.cuda.synchronize(); a = time.time()
            with torch.inference_mode():
                model.generate(**enc, max_new_tokens=GEN, min_new_tokens=GEN)
            torch.cuda.synchronize(); b = time.time()
            j, ns = energy(s, a, b, pre_w)
            per.append({"dur_s": b - a, "n_samples": ns, "net_j": j})
            sms.append(clocks()[0])
            time.sleep(0.5)                       # short: long gaps drop clocks

        good = [p["net_j"] for p in per if p["net_j"] is not None]
        durs = [p["dur_s"] for p in per]
        jr = [g / B for g in good]
        row = {"batch": B, "dur_s": st.mean(durs), "dur_sd": st.pstdev(durs),
               "n_measurable": len(good), "sm_clocks": sms,
               "j_req": st.mean(jr) if jr else None,
               "j_req_sd": st.pstdev(jr) if len(jr) > 1 else 0.0,
               "j_req_min": min(jr) if jr else None,
               "j_req_max": max(jr) if jr else None,
               "req_per_s": B / st.mean(durs),
               "tok_per_s": B * GEN / st.mean(durs),
               "vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        cv = (row["j_req_sd"] / row["j_req"] * 100) if row["j_req"] else float("nan")
        rows.append(row)
        print(f"  batch {B:3d}: {row['dur_s']:6.2f}s (sd {row['dur_sd']:.2f})  "
              f"{row['j_req'] if row['j_req'] else float('nan'):8.2f} J/req "
              f"(sd {row['j_req_sd']:.2f}, cv {cv:4.1f}%, "
              f"{row['j_req_min']:.1f}-{row['j_req_max']:.1f})  "
              f"{row['tok_per_s']:6.1f} tok/s  sm {min(sms)}-{max(sms)}", flush=True)
        del enc
        torch.cuda.reset_peak_memory_stats()
        gc.collect(); torch.cuda.empty_cache()

    post_w, post_sd = baseline(s, 15, "resident idle post")
    drift = abs(post_w - pre_w) if (post_w and pre_w) else float("nan")
    dirty = drift > DRIFT_W
    print(f"  drift {drift:+.2f} W -> {'DIRTY' if dirty else 'clean'}", flush=True)

    rows.sort(key=lambda r: r["batch"])
    best = min((r for r in rows if r["j_req"]), key=lambda r: r["j_req"], default=None)
    out = {"device": torch.cuda.get_device_name(0), "repo": REPO, "gen": GEN,
           "cold_idle_w": cold_idle, "cold_idle_sd": cold_sd,
           "idle_w": pre_w, "idle_sd": pre_sd, "post_w": post_w,
           "drift_w": drift, "dirty": dirty, "warm_s": WARM_S,
           "warm_ramp_sm": ramp, "order": order, "oom_at_batch": oom_at,
           "rows": rows}
    if best:
        out["best_batch"] = best["batch"]
        out["j_req_best"] = best["j_req"]
        b1 = next((r for r in rows if r["batch"] == 1), None)
        out["batch_saving"] = (b1["j_req"] / best["j_req"]) if b1 and b1["j_req"] else None
        daily = pre_w * 86400
        out["daily_idle_kwh"] = daily / 3.6e6
        out["rate_curve"] = [
            {"req_per_day": r, "j_per_req_total": daily / r + best["j_req"],
             "idle_share": (daily / r) / (daily / r + best["j_req"])}
            for r in (100, 1000, 10000, 100000, 1000000)]
        print(f"\n  best batch {best['batch']} -> {best['j_req']:.2f} J/req", flush=True)
        print(f"  idle floor {pre_w:.2f} W = {out['daily_idle_kwh']:.3f} kWh/day",
              flush=True)
        print(f"  cold idle was {cold_idle:.2f} W -- the CUDA context itself costs "
              f"{pre_w - cold_idle:+.2f} W", flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT}", flush=True)
    s.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
