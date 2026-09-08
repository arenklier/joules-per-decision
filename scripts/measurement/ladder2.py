"""Contention-aware re-run of the energy ladder.

The first run was invalidated halfway through when a second job took the GPU,
so this version refuses to trust a cell it cannot prove was measured alone:

  * it waits for an exclusive GPU before starting,
  * it takes a fresh resident-idle baseline per model and nets against THAT
    rather than against a global opening baseline,
  * it re-checks the baseline and the compute-process list after the
    generations, and discards and retries any cell that drifted.

All seven models are re-measured, not just the four that were spoiled: the
three clean ones then double as a reproducibility check against run 1.
"""
import os, gc, json, time, subprocess, statistics as st, traceback
os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from power_sampler import PowerSampler

MAX_NEW = 128
RUNS_PER_PROMPT = 3
DRIFT_W = 5.0            # a cell whose baseline moves more than this is dirty
CONTAMINATION_W = 30.0   # loose ceiling: real contention cost ~100 W, thermal
                         # drift over a long run costs under 10
MAX_RETRY = 3
ONLY = [m for m in os.environ.get("ONLY", "").split(",") if m]
OUT = os.environ.get("OUT", "ladder2_result.json")
FREE_W = 110.0           # idle floor measured at 84 W; allow headroom
WAIT_POLL_S = 60
MAX_WAIT_H = 12
MY_PID = os.getpid()
DESKTOP = "gnome-remote-desktop"

LADDER = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5", 1.5),
    ("Qwen/Qwen2.5-3B-Instruct",   "qwen2.5", 3.0),
    ("Qwen/Qwen2.5-7B-Instruct",   "qwen2.5", 7.0),
    ("Qwen/Qwen2.5-14B-Instruct",  "qwen2.5", 14.0),
    ("NousResearch/Meta-Llama-3.1-8B-Instruct", "llama3.1", 8.0),
    ("mistralai/Mistral-7B-Instruct-v0.2",      "mistral",  7.0),
    ("microsoft/Phi-3.5-mini-instruct",         "phi3.5",   3.8),
]

PROMPTS = [
    ("errdisable",
     "A Cisco access switch port is stuck in err-disabled state after a user "
     "plugged in an unmanaged switch. Name the most likely cause and the "
     "command that clears it."),
    ("ospf_adj",
     "Two directly connected routers never form an OSPF adjacency; both "
     "interfaces are up/up and in area 0. List the checks you would run, in "
     "order, and what each one rules out."),
    ("bgp_leak",
     "A downstream customer suddenly advertises 40000 prefixes to your eBGP "
     "session. State the immediate mitigation and the permanent control that "
     "prevents a recurrence."),
]


def foreign_pids():
    """Compute processes on the GPU that are neither ours nor the desktop."""
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


def gpu_watts():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=15).stdout.strip().splitlines()[0]
        return float(out)
    except Exception:
        return float("nan")


def wait_for_exclusive():
    """Block until the GPU has been free of other work for three checks."""
    deadline = time.time() + MAX_WAIT_H * 3600
    clean = 0
    while time.time() < deadline:
        fp, w = foreign_pids(), gpu_watts()
        if not fp and w < FREE_W:
            clean += 1
            print(f"  [queue] free ({w:.1f} W) {clean}/3", flush=True)
            if clean >= 3:
                return True
            time.sleep(20)
        else:
            if clean:
                print("  [queue] busy again, counter reset", flush=True)
            clean = 0
            who = ", ".join(f"{p}:{n}" for p, n in fp) or "-"
            print(f"  [queue] busy: {w:.1f} W, foreign={who}", flush=True)
            time.sleep(WAIT_POLL_S)
    return False


def baseline(s, seconds, label):
    t0 = time.time()
    time.sleep(seconds)
    t1 = time.time()
    w = s.mean_w(t0, t1)
    series = [smp[1][0] for smp in s.window(t0, t1)]
    sd = st.pstdev(series) if len(series) > 1 else 0.0
    print(f"    [{label}] {w[0]:7.2f} W (sd {sd:.2f}, n={len(series)})", flush=True)
    return w[0], sd


def measure(s, repo, family, params_b, open_idle):
    """One cell, retried until it can be shown to have run alone."""
    for attempt in range(1, MAX_RETRY + 1):
        if foreign_pids():
            print(f"    foreign job present, waiting before attempt {attempt}",
                  flush=True)
            wait_for_exclusive()
        entry = {"repo": repo, "family": family, "params_b": params_b,
                 "attempt": attempt}
        t_load = time.time()
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, torch_dtype=torch.bfloat16, device_map="cuda:0",
            trust_remote_code=True)
        model.eval()
        cfg = model.generation_config
        cfg.do_sample = False
        for k in ("temperature", "top_p", "top_k"):
            setattr(cfg, k, None)
        entry["load_s"] = time.time() - t_load
        entry["vram_gib"] = torch.cuda.memory_allocated() / 2**30
        print(f"    loaded {entry['load_s']:.1f} s, {entry['vram_gib']:.2f} GiB",
              flush=True)

        pre_w, pre_sd = baseline(s, 12, "resident idle pre")
        enc0 = tok([tok.apply_chat_template(
            [{"role": "user", "content": PROMPTS[0][1]}],
            tokenize=False, add_generation_prompt=True)],
            return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            model.generate(**enc0, max_new_tokens=8)
        torch.cuda.synchronize()
        time.sleep(2)

        rows = []
        for pname, ptext in PROMPTS:
            enc = tok([tok.apply_chat_template(
                [{"role": "user", "content": ptext}],
                tokenize=False, add_generation_prompt=True)],
                return_tensors="pt").to("cuda:0")
            for r in range(RUNS_PER_PROMPT):
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=MAX_NEW)
                torch.cuda.synchronize()
                t1 = time.time()
                ntok = int(out.shape[-1] - enc["input_ids"].shape[-1])
                e = s.energy_j(t0, t1)
                if e is None:
                    continue
                dur = t1 - t0
                rows.append({"prompt": pname, "rep": r, "dur_s": dur,
                             "new_tok": ntok, "total_j": e[0],
                             "net_j": e[0] - pre_w * dur,
                             "tok_per_s": ntok / dur})
                time.sleep(2)

        post_w, post_sd = baseline(s, 12, "resident idle post")
        intruders = foreign_pids()
        drift = abs(post_w - pre_w)
        # Net energy is already taken against this cell's own pre-baseline, so a
        # floor that has crept up as the card warms is subtracted correctly and
        # is not grounds to discard -- the first version of this guard threw
        # away three good cells over ~8 W of thermal drift. What matters is
        # that the floor held STILL across the cell, plus a loose ceiling that
        # still catches a genuine second job.
        dirty = (drift > DRIFT_W or bool(intruders)
                 or (pre_w - open_idle) > CONTAMINATION_W)

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)

        if dirty:
            if drift > DRIFT_W:
                why = f"drift {drift:.1f} W"
            elif intruders:
                why = f"foreign {intruders}"
            else:
                why = f"pre-baseline {pre_w:.1f} W vs open {open_idle:.1f} W"
            print(f"    DIRTY ({why}) -- discarding attempt {attempt}", flush=True)
            if attempt == MAX_RETRY:
                entry.update(dirty=True, reason=why, pre_w=pre_w, post_w=post_w)
                return entry
            continue

        net = [r["net_j"] for r in rows]
        jt = [r["net_j"] / r["new_tok"] for r in rows if r["new_tok"]]
        tps = [r["tok_per_s"] for r in rows]
        wl = [r["total_j"] / r["dur_s"] for r in rows]
        entry.update(dirty=False, pre_w=pre_w, pre_sd=pre_sd, post_w=post_w,
                     post_sd=post_sd, drift_w=drift, n=len(rows), rows=rows,
                     net_j_mean=st.mean(net), net_j_sd=st.pstdev(net),
                     j_per_tok=st.mean(jt), j_per_tok_sd=st.pstdev(jt),
                     tok_per_s=st.mean(tps), loaded_w=st.mean(wl))
        print(f"    -> {st.mean(net):8.1f} J/req (sd {st.pstdev(net):.1f}, "
              f"n={len(rows)})  {st.mean(jt):7.3f} J/tok  {st.mean(tps):5.1f} tok/s"
              f"  {st.mean(wl):6.1f} W  drift {drift:+.2f} W", flush=True)
        return entry
    return {"repo": repo, "dirty": True, "reason": "retries exhausted"}


def main():
    print("=== queued: waiting for an exclusive GPU ===", flush=True)
    if not wait_for_exclusive():
        print("gave up waiting")
        return
    print("=== GPU free, starting ===", flush=True)

    s = PowerSampler(interval_ms=100).start()
    open_idle, open_sd = baseline(s, 20, "opening idle")

    ladder = [c for c in LADDER if not ONLY or c[0] in ONLY]
    print(f"cells to measure: {len(ladder)}", flush=True)

    results = []
    for repo, family, pb in ladder:
        print(f"\n=== {repo} ({family}, {pb}B) ===", flush=True)
        try:
            results.append(measure(s, repo, family, pb, open_idle))
        except Exception as exc:
            print(f"    FAILED {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            results.append({"repo": repo, "family": family, "params_b": pb,
                            "error": f"{type(exc).__name__}: {exc}"})
        json.dump({"open_idle": open_idle, "open_sd": open_sd,
                   "max_new": MAX_NEW, "results": results},
                  open(OUT, "w"), indent=1)

    close_idle, _ = baseline(s, 15, "closing idle")
    print(f"\nidle drift over whole run: {close_idle - open_idle:+.2f} W", flush=True)

    print("\n=== SUMMARY (clean cells only) ===")
    header = (f"{'model':44s} {'B':>5} {'J/req':>9} {'J/tok':>8} "
              f"{'tok/s':>7} {'W':>7} {'drift':>7}")
    print(header)
    for e in results:
        if e.get("dirty") is False:
            print(f"{e['repo']:44s} {e['params_b']:5.1f} {e['net_j_mean']:9.1f} "
                  f"{e['j_per_tok']:8.3f} {e['tok_per_s']:7.1f} "
                  f"{e['loaded_w']:7.1f} {e['drift_w']:+7.2f}")
        else:
            reason = e.get("reason", e.get("error", "?"))
            print(f"{e['repo']:44s} {e.get('params_b', 0):5.1f}   "
                  f"DISCARDED: {reason[:44]}")

    json.dump({"open_idle": open_idle, "open_sd": open_sd,
               "close_idle": close_idle, "max_new": MAX_NEW,
               "results": results}, open(OUT, "w"), indent=1)
    s.stop()


if __name__ == "__main__":
    main()
