"""What does chain-of-thought cost, and what does it buy?

Same verifiable tasks under two prompting conditions, on the same clean-room
protocol as the ladder: per-model resident-idle baseline, foreign-process
check, pre/post drift guard. The two conditions are interleaved task by task
so the card's thermal drift lands on both equally rather than on whichever ran
second.

The metric that decides the question is not joules per attempt but joules per
CORRECT answer -- reasoning is only expensive if the accuracy it buys does not
pay for the tokens it burns.
"""
import os, gc, json, time, subprocess, statistics as st, traceback
os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from power_sampler import PowerSampler
from tasks_net import build, grade

MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ("Qwen/Qwen2.5-3B-Instruct",   3.0),
    ("Qwen/Qwen2.5-7B-Instruct",   7.0),
    ("Qwen/Qwen2.5-14B-Instruct",  14.0),
]

DIRECT_MAX = 48
COT_MAX = 512
DRIFT_W = 5.0
CONTAMINATION_W = 30.0
MY_PID = os.getpid()
DESKTOP = "gnome-remote-desktop"
OUT = os.environ.get("OUT", "reasoning_tax_result.json")

DIRECT_SUFFIX = ("\n\nDo not explain. Reply with exactly one line in the form "
                 "ANSWER: <value>")
COT_SUFFIX = ("\n\nThink step by step, showing your working. Then finish with "
              "exactly one final line in the form ANSWER: <value>")


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
    """Hold until nothing else is on the GPU. A transient job from another
    session should delay this run, not abort it."""
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
            print(f"  [queue] busy: {fp}", flush=True)
            time.sleep(60)
    return False


def baseline(s, seconds, label):
    t0 = time.time(); time.sleep(seconds); t1 = time.time()
    w = s.mean_w(t0, t1)
    series = [smp[1][0] for smp in s.window(t0, t1)]
    sd = st.pstdev(series) if len(series) > 1 else 0.0
    print(f"    [{label}] {w[0]:7.2f} W (sd {sd:.2f}, n={len(series)})", flush=True)
    return w[0], sd


def run_one(model, tok, s, prompt, max_new, pre_w):
    enc = tok([tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)],
              return_tensors="pt").to("cuda:0")
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**enc, max_new_tokens=max_new)
    torch.cuda.synchronize(); t1 = time.time()
    n_in = enc["input_ids"].shape[-1]
    gen_ids = out[0][n_in:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    e = s.energy_j(t0, t1)
    dur = t1 - t0
    return {"dur_s": dur, "new_tok": int(gen_ids.shape[-1]),
            "prompt_tok": int(n_in), "text": text,
            "total_j": None if e is None else e[0],
            "net_j": None if e is None else e[0] - pre_w * dur}


def main():
    tasks = build()
    print(f"{len(tasks)} verifiable tasks "
          f"({', '.join(sorted(set(t['family'] for t in tasks)))})", flush=True)

    print("waiting for an exclusive GPU ...", flush=True)
    if not wait_for_exclusive():
        print("gave up waiting")
        return

    s = PowerSampler(interval_ms=100).start()
    open_idle, _ = baseline(s, 20, "opening idle")

    results = []
    for repo, pb in MODELS:
        print(f"\n=== {repo} ({pb}B) ===", flush=True)
        entry = {"repo": repo, "params_b": pb}
        try:
            if foreign_pids():
                print("    foreign job on GPU -- waiting", flush=True)
                wait_for_exclusive()

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
            if cfg.pad_token_id is None:
                cfg.pad_token_id = tok.eos_token_id
            print(f"    loaded {time.time()-t_load:.1f} s", flush=True)

            pre_w, _ = baseline(s, 12, "resident idle pre")
            run_one(model, tok, s, tasks[0]["prompt"] + DIRECT_SUFFIX, 8, pre_w)
            torch.cuda.synchronize(); time.sleep(2)

            rows = []
            for i, t in enumerate(tasks):
                for cond, suffix, cap in (("direct", DIRECT_SUFFIX, DIRECT_MAX),
                                          ("cot", COT_SUFFIX, COT_MAX)):
                    r = run_one(model, tok, s, t["prompt"] + suffix, cap, pre_w)
                    ok, got = grade(r["text"], t["gt"])
                    r.update(task=t["id"], family=t["family"], cond=cond,
                             gt=t["gt"], got=got, correct=bool(ok))
                    r.pop("text")
                    rows.append(r)
                    time.sleep(1)
                if (i + 1) % 8 == 0:
                    print(f"    ... {i+1}/{len(tasks)} tasks", flush=True)

            post_w, _ = baseline(s, 12, "resident idle post")
            drift = abs(post_w - pre_w)
            intruders = foreign_pids()
            dirty = (drift > DRIFT_W or bool(intruders)
                     or (pre_w - open_idle) > CONTAMINATION_W)
            entry.update(pre_w=pre_w, post_w=post_w, drift_w=drift,
                         dirty=dirty, rows=rows)

            for cond in ("direct", "cot"):
                sub = [r for r in rows if r["cond"] == cond and r["net_j"] is not None]
                if not sub:
                    continue
                acc = sum(r["correct"] for r in sub) / len(sub)
                jt = st.mean([r["net_j"] for r in sub])
                tk = st.mean([r["new_tok"] for r in sub])
                j_correct = (sum(r["net_j"] for r in sub) /
                             sum(r["correct"] for r in sub)) if acc > 0 else float("inf")
                entry[cond] = {"n": len(sub), "acc": acc, "j_per_task": jt,
                               "tok_per_task": tk, "j_per_correct": j_correct}
                print(f"    {cond:6s}: acc {acc*100:5.1f}%  {jt:8.1f} J/task  "
                      f"{tk:6.1f} tok  {j_correct:10.1f} J/correct", flush=True)
            if dirty:
                print(f"    WARNING: cell dirty (drift {drift:.1f} W, "
                      f"foreign {intruders})", flush=True)
            del model, tok
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {entry['error']}", flush=True)
            traceback.print_exc()
        finally:
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        results.append(entry)
        json.dump({"open_idle": open_idle, "results": results},
                  open(OUT, "w"), indent=1)

    print("\n=== REASONING TAX ===")
    print(f"{'model':30s} {'acc dir':>8} {'acc cot':>8} {'J dir':>9} {'J cot':>9} "
          f"{'x energy':>9} {'J/corr dir':>11} {'J/corr cot':>11}")
    for e in results:
        if "direct" in e and "cot" in e:
            d, c = e["direct"], e["cot"]
            mult = c["j_per_task"] / d["j_per_task"] if d["j_per_task"] else float("nan")
            print(f"{e['repo'].split('/')[-1]:30s} {d['acc']*100:7.1f}% "
                  f"{c['acc']*100:7.1f}% {d['j_per_task']:9.1f} {c['j_per_task']:9.1f} "
                  f"{mult:8.2f}x {d['j_per_correct']:11.1f} {c['j_per_correct']:11.1f}")
    json.dump({"open_idle": open_idle, "results": results}, open(OUT, "w"), indent=1)
    s.stop()


if __name__ == "__main__":
    main()
