"""External-validity pass: the same energy protocol on a published benchmark.

Everything measured so far used tasks I generated, which is the one objection
that no amount of internal rigour answers. This run replaces them with a
stratified sample of NetArena (ICLR 2026, Froot-NetSys) capacity-planning
tasks, and -- importantly -- uses NetArena's OWN prompts, extracted verbatim
from its text_utils.py rather than reworded. Its BASE_PROMPT and COT_PROMPT
already define exactly the two conditions this study contrasts, so the
direct-vs-CoT split is the benchmark's design, not mine.

Scope, stated plainly: this measures ENERGY and TOKEN profile only. NetArena
grades by executing the generated Python against a 1.9 MB graph and comparing
the result to a reference implementation; wiring that up is a separate job, so
no accuracy is claimed here. What this run does establish is whether the
energy findings -- the ~45x CoT multiplier above all -- survive a real task
distribution with long system prompts and code-length outputs, instead of the
short question-answer workload they were measured on.
"""
import os, gc, json, time, subprocess, statistics as st, traceback
from collections import defaultdict
os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from power_sampler import PowerSampler

MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ("Qwen/Qwen2.5-3B-Instruct",   3.0),
    ("Qwen/Qwen2.5-7B-Instruct",   7.0),
    ("Qwen/Qwen2.5-14B-Instruct",  14.0),
]

BASE_MAX = int(os.environ.get("BASE_MAX", 768))
COT_MAX = int(os.environ.get("COT_MAX", 2048))
MIN_SAMPLES = 30
DRIFT_W = 5.0
CONTAMINATION_W = 30.0
MY_PID = os.getpid()
DESKTOP = "gnome-remote-desktop"
OUT = os.environ.get("OUT", "netarena_energy_result.json")

P = json.load(open("netarena_prompts.json", encoding="utf-8"))
TASKS = json.load(open("netarena_sample.json", encoding="utf-8"))


def build_prompt(question, cot):
    """NetArena's own composition: base prompt, optionally the CoT line, suffix."""
    parts = [P["BASE_PROMPT"]]
    if cot:
        parts.append(P["COT_PROMPT"])
    parts.append(P["PROMPT_SUFFIX"].replace("{input}", question))
    return "\n".join(parts)


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


def wait_for_exclusive(max_h=float(os.environ.get("MAX_WAIT_H", 96))):
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


def main():
    print(f"{len(TASKS)} NetArena tasks, base prompt {len(P['BASE_PROMPT'])} chars, "
          f"caps base={BASE_MAX} cot={COT_MAX}", flush=True)
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

            t0 = time.time()
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
            print(f"    loaded {time.time()-t0:.1f} s", flush=True)

            pre_w, _ = baseline(s, 12, "resident idle pre")
            warm = tok([tok.apply_chat_template(
                [{"role": "user", "content": build_prompt(TASKS[0]["question"], False)}],
                tokenize=False, add_generation_prompt=True)],
                return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                model.generate(**warm, max_new_tokens=8)
            torch.cuda.synchronize(); time.sleep(2)

            rows = []
            for i, t in enumerate(TASKS):
                for cond, cot, cap in (("base", False, BASE_MAX),
                                       ("cot", True, COT_MAX)):
                    prompt = build_prompt(t["question"], cot)
                    enc = tok([tok.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True)],
                        return_tensors="pt").to("cuda:0")
                    torch.cuda.synchronize(); a = time.time()
                    with torch.inference_mode():
                        o = model.generate(**enc, max_new_tokens=cap)
                    torch.cuda.synchronize(); b = time.time()
                    n_in = int(enc["input_ids"].shape[-1])
                    n_new = int(o.shape[-1] - n_in)
                    ns = len(s.window(a, b))
                    ej = s.energy_j(a, b)
                    rows.append({
                        "task": t["id"], "label": t["label"], "cond": cond,
                        "prompt_tok": n_in, "new_tok": n_new, "dur_s": b - a,
                        "n_samples": ns, "capped": n_new >= cap,
                        "net_j": None if (ej is None or ns < MIN_SAMPLES)
                                 else ej[0] - pre_w * (b - a)})
                    time.sleep(1)
                if (i + 1) % 15 == 0:
                    print(f"    ... {i+1}/{len(TASKS)}", flush=True)

            post_w, _ = baseline(s, 12, "resident idle post")
            drift = abs(post_w - pre_w)
            entry["dirty"] = (drift > DRIFT_W or bool(foreign_pids())
                              or (pre_w - open_idle) > CONTAMINATION_W)
            entry.update(pre_w=pre_w, post_w=post_w, drift_w=drift, rows=rows)

            for cond in ("base", "cot"):
                sub = [r for r in rows if r["cond"] == cond]
                j = [r["net_j"] for r in sub if r["net_j"] is not None]
                entry[cond] = {
                    "n": len(sub), "n_measurable": len(j),
                    "j_per_task": st.mean(j) if j else None,
                    "j_per_task_sd": st.pstdev(j) if len(j) > 1 else None,
                    "prompt_tok": st.mean([r["prompt_tok"] for r in sub]),
                    "new_tok": st.mean([r["new_tok"] for r in sub]),
                    "capped": sum(r["capped"] for r in sub) / len(sub),
                    "tok_per_s": st.mean([r["new_tok"] / r["dur_s"] for r in sub]),
                }
                c = entry[cond]
                print(f"    {cond:5s}: {c['j_per_task']:8.1f} J/task  "
                      f"{c['prompt_tok']:6.0f} prompt tok  {c['new_tok']:6.1f} new tok  "
                      f"{c['capped']*100:4.0f}% capped  {c['tok_per_s']:5.1f} tok/s",
                      flush=True)
            if entry["base"]["j_per_task"]:
                entry["multiplier"] = entry["cot"]["j_per_task"] / entry["base"]["j_per_task"]
                print(f"    CoT energy multiplier: {entry['multiplier']:.1f}x", flush=True)
            del model, tok
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {entry['error']}", flush=True)
            traceback.print_exc()
        finally:
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        results.append(entry)
        json.dump({"open_idle": open_idle, "n_tasks": len(TASKS),
                   "base_max": BASE_MAX, "cot_max": COT_MAX,
                   "results": results}, open(OUT, "w"), indent=1)

    print("\n=== NETARENA ENERGY (no accuracy claimed) ===")
    print(f"{'model':10s} {'J base':>9} {'J cot':>9} {'xE':>6} {'tok base':>9} "
          f"{'tok cot':>8} {'cap b/c':>10} {'drift':>7}")
    for e in results:
        if "multiplier" not in e:
            continue
        b_, c_ = e["base"], e["cot"]
        print(f"{e['repo'].split('/')[-1].replace('Qwen2.5-','').replace('-Instruct',''):10s} "
              f"{b_['j_per_task']:9.1f} {c_['j_per_task']:9.1f} {e['multiplier']:5.1f}x "
              f"{b_['new_tok']:9.1f} {c_['new_tok']:8.1f} "
              f"{b_['capped']*100:4.0f}%/{c_['capped']*100:3.0f}% {e['drift_w']:6.2f}W"
              f"{'  DIRTY' if e['dirty'] else ''}")
    json.dump({"open_idle": open_idle, "n_tasks": len(TASKS), "base_max": BASE_MAX,
               "cot_max": COT_MAX, "results": results}, open(OUT, "w"), indent=1)
    s.stop()


if __name__ == "__main__":
    main()
