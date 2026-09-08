"""The edge arm: the same serving protocol on a small accelerator.

The placement question needs both ends measured on the same instrument, not one
measured and the other assumed. This runs the batching sweep from the L40S on an
RTX A2000 -- 6 GB, 70 W cap, the class of card that would sit in a gateway
rather than a rack.

Two numbers decide the trade-off and they pull in opposite directions: the edge
card idles at roughly a tenth of the L40S, but its memory ceiling caps
concurrency far sooner, so it amortises much less. Which wins depends on the
request rate, which is exactly what the placement analysis has to compute.

Runs on the operator's own workstation, so the guards matter more than usual:
the idle floor was measured at 7.81 W with sd 0.12 even with office
applications open, but a burst of UI work during a cell would still show up as
baseline drift, and any cell that drifts is discarded rather than reported.
"""
import gc, json, os, statistics as st, subprocess, sys, time, traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from power_sampler import PowerSampler
from tasks_net import build

REPO = os.environ.get("REPO", "Qwen/Qwen2.5-1.5B-Instruct")
GPU_IDX = int(os.environ.get("GPU_IDX", 1))     # nvidia-smi index of the A2000
BATCHES = [1, 2, 4, 8, 16, 32]
GEN = int(os.environ.get("GEN", 128))
REPEATS = 3
MIN_SAMPLES = 30
DRIFT_W = 1.5                                   # tighter: this card's sd is 0.12
OUT = os.environ.get("OUT", "edge_result.json")
SUFFIX = "\n\nAnswer in the form ANSWER: <value>"


def baseline(s, sec, label):
    t0 = time.time(); time.sleep(sec); t1 = time.time()
    ser = [x[1][GPU_IDX] for x in s.window(t0, t1)]
    if not ser:
        return None, None
    w, sd = st.mean(ser), (st.pstdev(ser) if len(ser) > 1 else 0.0)
    print(f"  [{label}] {w:6.2f} W (sd {sd:.2f}, n={len(ser)})", flush=True)
    return w, sd


def energy(s, a, b, idle_w):
    win = s.window(a, b)
    if len(win) < MIN_SAMPLES:
        return None, len(win)
    tot = 0.0
    for (ta, pa), (tb, pb) in zip(win, win[1:]):
        tot += 0.5 * (pa[GPU_IDX] + pb[GPU_IDX]) * (tb - ta)
    return tot - idle_w * (b - a), len(win)


def load_model():
    tok = AutoTokenizer.from_pretrained(REPO)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:                                        # transformers 5.x renamed this
        model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.bfloat16)
    model = model.to("cuda:0").eval()
    cfg = model.generation_config
    cfg.do_sample = False
    for k in ("temperature", "top_p", "top_k"):
        setattr(cfg, k, None)
    cfg.pad_token_id = tok.pad_token_id
    return tok, model


def main():
    tasks = build(n_per_family=40)
    prompts = [t["prompt"] + SUFFIX for t in tasks]
    name = torch.cuda.get_device_name(0)
    print(f"edge arm: {REPO} on {name}", flush=True)
    print(f"{len(prompts)} prompts, batches {BATCHES}, {GEN} forced tokens", flush=True)

    s = PowerSampler(interval_ms=100).start()
    open_idle, open_sd = baseline(s, 20, "opening idle")

    print("loading (first run downloads ~3.1 GB) ...", flush=True)
    t0 = time.time()
    tok, model = load_model()
    print(f"  loaded in {time.time()-t0:.1f} s, "
          f"{torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    pre_w, _ = baseline(s, 12, "resident idle pre")
    rows, oom_at = [], None

    for B in BATCHES:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts[:B]]
        try:
            enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
            with torch.inference_mode():
                model.generate(**enc, max_new_tokens=4, min_new_tokens=4)
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            oom_at = B
            print(f"  batch {B:3d}: OOM -- concurrency ceiling", flush=True)
            gc.collect(); torch.cuda.empty_cache()
            break
        time.sleep(1.5)

        per = []
        for _ in range(REPEATS):
            torch.cuda.synchronize(); a = time.time()
            with torch.inference_mode():
                model.generate(**enc, max_new_tokens=GEN, min_new_tokens=GEN)
            torch.cuda.synchronize(); b = time.time()
            j, ns = energy(s, a, b, pre_w)
            per.append({"dur_s": b - a, "n_samples": ns, "net_j": j})
            time.sleep(1.5)

        good = [p for p in per if p["net_j"] is not None]
        dur = st.mean([p["dur_s"] for p in per])
        row = {"batch": B, "dur_s": dur, "n_measurable": len(good),
               "j_batch": st.mean([p["net_j"] for p in good]) if good else None,
               "j_req": (st.mean([p["net_j"] for p in good]) / B) if good else None,
               "req_per_s": B / dur, "tok_per_s": B * GEN / dur,
               "vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
        rows.append(row)
        print(f"  batch {B:3d}: {dur:6.2f} s  "
              f"{row['j_req'] if row['j_req'] else float('nan'):8.2f} J/req  "
              f"{row['req_per_s']:6.2f} req/s  {row['tok_per_s']:7.1f} tok/s  "
              f"{row['vram_gib']:5.2f} GiB  ({len(good)}/{REPEATS})", flush=True)
        del enc
        torch.cuda.reset_peak_memory_stats()
        gc.collect(); torch.cuda.empty_cache()

    post_w, post_sd = baseline(s, 12, "resident idle post")
    drift = abs(post_w - pre_w) if (post_w and pre_w) else float("nan")
    dirty = drift > DRIFT_W
    print(f"  drift {drift:+.2f} W  {'DIRTY' if dirty else 'clean'}", flush=True)

    best = min((r for r in rows if r["j_req"]), key=lambda r: r["j_req"], default=None)
    out = {"device": name, "repo": REPO, "gen": GEN, "idle_w": pre_w,
           "idle_sd_open": open_sd, "open_idle": open_idle, "post_w": post_w,
           "drift_w": drift, "dirty": dirty, "oom_at_batch": oom_at, "rows": rows}
    if best:
        out["best_batch"] = best["batch"]
        out["j_req_best"] = best["j_req"]
        out["batch_saving"] = rows[0]["j_req"] / best["j_req"] if rows[0]["j_req"] else None
        daily_idle_j = pre_w * 86400
        out["daily_idle_kwh"] = daily_idle_j / 3.6e6
        out["rate_curve"] = [
            {"req_per_day": r, "j_per_req_total": daily_idle_j / r + best["j_req"],
             "idle_share": (daily_idle_j / r) / (daily_idle_j / r + best["j_req"])}
            for r in (100, 1000, 10000, 100000, 1000000)]
        print(f"\n  best batch {best['batch']} -> {best['j_req']:.2f} J/req "
              f"({out['batch_saving']:.1f}x better than batch 1)", flush=True)
        print(f"  idle floor {pre_w:.2f} W = {out['daily_idle_kwh']:.3f} kWh/day", flush=True)
        for c in out["rate_curve"]:
            print(f"    {c['req_per_day']:>9,} req/day: {c['j_per_req_total']:9.1f} J/req, "
                  f"idle {c['idle_share']*100:5.1f}%", flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    print(f"\nwrote {OUT}", flush=True)
    s.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
