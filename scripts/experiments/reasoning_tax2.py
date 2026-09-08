"""Reasoning tax, second attempt -- with a measurement floor this time.

Run 1 produced negative joules for the direct condition. The cause was not the
model but the instrument: a direct answer takes ~0.1 s, which at 10 Hz is one
or two power samples, and nvidia-smi's power.draw is itself an internal
average over roughly a second. Sub-second work simply cannot be integrated.

The fix is amortisation. Greedy decoding is deterministic, so replaying the
same prompt changes nothing about the output -- only about how long the card
is busy. The direct condition therefore runs as whole-set sweeps timed end to
end, long enough to carry a real integral, and grading is taken from the first
sweep. Anything measured over fewer than MIN_SAMPLES points is reported as
unmeasurable rather than as a number.

Accuracy is also reported per family and against a majority-class baseline:
answering DENY to every ACL scores 56% on its own, and an aggregate that hides
that is not an accuracy at all.
"""
import os, gc, json, time, subprocess, statistics as st, traceback
from collections import Counter, defaultdict
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
DIRECT_SWEEPS = 3          # repeat the whole set, so each window is seconds long
MIN_SAMPLES = 30           # below this a window is not integrable
DRIFT_W = 5.0
CONTAMINATION_W = 30.0
MY_PID = os.getpid()
DESKTOP = "gnome-remote-desktop"
OUT = os.environ.get("OUT", "reasoning_tax2_result.json")

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


def encode(tok, prompt):
    return tok([tok.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)],
               return_tensors="pt").to("cuda:0")


def majority_baseline(tasks):
    """Score of the best constant answer within each family."""
    per_fam, total = {}, 0
    for fam in sorted(set(t["family"] for t in tasks)):
        gts = [t["gt"] for t in tasks if t["family"] == fam]
        best = Counter(gts).most_common(1)[0][1]
        per_fam[fam] = best / len(gts)
        total += best
    return per_fam, total / len(tasks)


def main():
    tasks = build()
    fam_base, overall_base = majority_baseline(tasks)
    print(f"{len(tasks)} tasks; majority-class baseline overall "
          f"{overall_base*100:.1f}% ({ {k: round(v*100,1) for k,v in fam_base.items()} })",
          flush=True)

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
                print("    foreign job -- waiting", flush=True)
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
            with torch.inference_mode():
                model.generate(**encode(tok, tasks[0]["prompt"] + DIRECT_SUFFIX),
                               max_new_tokens=8)
            torch.cuda.synchronize(); time.sleep(2)

            # ---- direct: whole-set sweeps, timed as one window each ----------
            direct_enc = [encode(tok, t["prompt"] + DIRECT_SUFFIX) for t in tasks]
            sweep_j, sweep_dur, sweep_n = [], [], []
            direct_correct, direct_tok = [], []
            for sw in range(DIRECT_SWEEPS):
                torch.cuda.synchronize(); t0 = time.time()
                outs = []
                for enc in direct_enc:
                    with torch.inference_mode():
                        outs.append(model.generate(**enc, max_new_tokens=DIRECT_MAX))
                torch.cuda.synchronize(); t1 = time.time()
                nsmp = len(s.window(t0, t1))
                e = s.energy_j(t0, t1)
                sweep_dur.append(t1 - t0); sweep_n.append(nsmp)
                sweep_j.append(None if e is None else e[0] - pre_w * (t1 - t0))
                if sw == 0:                      # greedy: every sweep is identical
                    for t, enc, out in zip(tasks, direct_enc, outs):
                        gen = out[0][enc["input_ids"].shape[-1]:]
                        ok, got = grade(tok.decode(gen, skip_special_tokens=True),
                                        t["gt"])
                        direct_correct.append((t["family"], bool(ok)))
                        direct_tok.append(int(gen.shape[-1]))
                print(f"    direct sweep {sw+1}: {t1-t0:6.2f} s, {nsmp} samples, "
                      f"{sweep_j[-1]:.1f} J total", flush=True)
                time.sleep(2)

            usable = [j for j, n in zip(sweep_j, sweep_n)
                      if j is not None and n >= MIN_SAMPLES]
            direct_stats = {
                "measurable": len(usable) > 0,
                "sweeps_n_samples": sweep_n,
                "sweep_dur_s": sweep_dur,
                "j_per_task": (st.mean(usable) / len(tasks)) if usable else None,
                "j_per_task_sd": (st.pstdev([u / len(tasks) for u in usable])
                                  if len(usable) > 1 else None),
                "tok_per_task": st.mean(direct_tok),
            }

            # ---- cot: each generation is long enough to time on its own -------
            cot_rows = []
            for i, t in enumerate(tasks):
                enc = encode(tok, t["prompt"] + COT_SUFFIX)
                torch.cuda.synchronize(); t0 = time.time()
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=COT_MAX)
                torch.cuda.synchronize(); t1 = time.time()
                gen = out[0][enc["input_ids"].shape[-1]:]
                ok, got = grade(tok.decode(gen, skip_special_tokens=True), t["gt"])
                nsmp = len(s.window(t0, t1))
                e = s.energy_j(t0, t1)
                cot_rows.append({
                    "task": t["id"], "family": t["family"], "correct": bool(ok),
                    "new_tok": int(gen.shape[-1]), "dur_s": t1 - t0,
                    "n_samples": nsmp,
                    "net_j": None if (e is None or nsmp < MIN_SAMPLES)
                             else e[0] - pre_w * (t1 - t0)})
                time.sleep(1)
                if (i + 1) % 16 == 0:
                    print(f"    ... cot {i+1}/{len(tasks)}", flush=True)

            post_w, _ = baseline(s, 12, "resident idle post")
            drift = abs(post_w - pre_w)
            entry["dirty"] = (drift > DRIFT_W or bool(foreign_pids())
                              or (pre_w - open_idle) > CONTAMINATION_W)
            entry.update(pre_w=pre_w, post_w=post_w, drift_w=drift,
                         direct=direct_stats, cot_rows=cot_rows)

            # ---- aggregate ----------------------------------------------------
            def fam_acc(pairs):
                d = defaultdict(list)
                for fam, ok in pairs:
                    d[fam].append(ok)
                return {f: sum(v) / len(v) for f, v in sorted(d.items())}

            cot_pairs = [(r["family"], r["correct"]) for r in cot_rows]
            d_acc = sum(ok for _, ok in direct_correct) / len(direct_correct)
            c_acc = sum(ok for _, ok in cot_pairs) / len(cot_pairs)
            cj = [r["net_j"] for r in cot_rows if r["net_j"] is not None]
            entry["direct"].update(acc=d_acc, fam_acc=fam_acc(direct_correct))
            entry["cot"] = {
                "acc": c_acc, "fam_acc": fam_acc(cot_pairs),
                "n_measurable": len(cj),
                "j_per_task": st.mean(cj) if cj else None,
                "j_per_task_sd": st.pstdev(cj) if len(cj) > 1 else None,
                "tok_per_task": st.mean([r["new_tok"] for r in cot_rows]),
            }
            dj, cjm = direct_stats["j_per_task"], entry["cot"]["j_per_task"]
            entry["energy_multiplier"] = (cjm / dj) if (dj and cjm) else None
            entry["j_per_correct_direct"] = (dj / d_acc) if (dj and d_acc) else None
            entry["j_per_correct_cot"] = (cjm / c_acc) if (cjm and c_acc) else None

            print(f"    direct: acc {d_acc*100:5.1f}%  {dj if dj else float('nan'):8.2f} J/task  "
                  f"{direct_stats['tok_per_task']:6.1f} tok  {entry['direct']['fam_acc']}",
                  flush=True)
            print(f"    cot   : acc {c_acc*100:5.1f}%  {cjm if cjm else float('nan'):8.2f} J/task  "
                  f"{entry['cot']['tok_per_task']:6.1f} tok  {entry['cot']['fam_acc']}",
                  flush=True)
            if entry["energy_multiplier"]:
                print(f"    energy multiplier: {entry['energy_multiplier']:.1f}x",
                      flush=True)
            del model, tok, direct_enc
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {entry['error']}", flush=True)
            traceback.print_exc()
        finally:
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        results.append(entry)
        json.dump({"open_idle": open_idle, "n_tasks": len(tasks),
                   "majority_baseline": overall_base, "fam_baseline": fam_base,
                   "results": results}, open(OUT, "w"), indent=1)

    print("\n=== REASONING TAX ===")
    print(f"majority-class baseline: {overall_base*100:.1f}%")
    print(f"{'model':26s} {'acc dir':>8} {'acc cot':>8} {'J dir':>8} {'J cot':>9} "
          f"{'xE':>7} {'J/corr dir':>11} {'J/corr cot':>11}")
    for e in results:
        if "cot" not in e:
            continue
        d, c = e["direct"], e["cot"]
        f = lambda v, w=8, p=1: (f"{v:{w}.{p}f}" if v is not None else " " * (w - 2) + "--")
        print(f"{e['repo'].split('/')[-1]:26s} {d['acc']*100:7.1f}% {c['acc']*100:7.1f}% "
              f"{f(d['j_per_task'])} {f(c['j_per_task'],9)} {f(e['energy_multiplier'],7)} "
              f"{f(e['j_per_correct_direct'],11)} {f(e['j_per_correct_cot'],11)}")
    json.dump({"open_idle": open_idle, "n_tasks": len(tasks),
               "majority_baseline": overall_base, "fam_baseline": fam_base,
               "results": results}, open(OUT, "w"), indent=1)
    s.stop()


if __name__ == "__main__":
    main()
