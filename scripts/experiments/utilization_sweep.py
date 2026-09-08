"""Serving economics: what a request actually costs once the card is shared.

Everything measured so far was one request at a time on an otherwise idle GPU,
which is the least realistic way anyone would run this. Two things change under
real serving and both move the number a lot:

  * batching amortises the per-step cost across concurrent requests, so the
    dynamic energy per request falls as the batch grows;
  * the idle floor is paid whether or not anyone asks anything, so at low
    request rates it dominates the bill entirely.

This measures the first directly and supplies the parameters to compute the
second. Every sequence is forced to generate exactly GEN tokens (min_new ==
max_new) so that batch sizes are compared on identical work rather than on
whatever length each batch happened to stop at.
"""
import os, gc, json, time, subprocess, statistics as st, traceback
os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from power_sampler import PowerSampler
from tasks_net import build

MODELS = [("Qwen/Qwen2.5-3B-Instruct", 3.0), ("Qwen/Qwen2.5-14B-Instruct", 14.0)]
_only = os.environ.get("ONLY", "")
if _only:
    MODELS = [m for m in MODELS if _only in m[0]]
BATCHES = [1, 2, 4, 8, 16, 32, 64]
GEN = int(os.environ.get("GEN", 128))
REPEATS = 3
MIN_SAMPLES = 30
DRIFT_W = 5.0
CONTAMINATION_W = 30.0
MY_PID = os.getpid()
DESKTOP = "gnome-remote-desktop"
OUT = os.environ.get("OUT", "utilization_result.json")

SUFFIX = "\n\nAnswer in the form ANSWER: <value>"


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
        p = [x.strip() for x in line.split(",")]
        if len(p) < 2:
            continue
        try:
            pid = int(p[0])
        except ValueError:
            continue
        if pid != MY_PID and DESKTOP not in p[1]:
            found.append((pid, p[1]))
    return found


def wait_for_exclusive(max_h=96):
    deadline = time.time() + max_h * 3600
    clean = 0
    while time.time() < deadline:
        if not foreign_pids():
            clean += 1
            if clean >= 3:
                return True
            time.sleep(15)
        else:
            clean = 0
            print(f"  [queue] busy: {foreign_pids()}", flush=True)
            time.sleep(60)
    return False


def baseline(s, sec, label):
    t0 = time.time(); time.sleep(sec); t1 = time.time()
    w = s.mean_w(t0, t1)
    ser = [x[1][0] for x in s.window(t0, t1)]
    sd = st.pstdev(ser) if len(ser) > 1 else 0.0
    print(f"    [{label}] {w[0]:7.2f} W (sd {sd:.2f})", flush=True)
    return w[0]


def main():
    tasks = build(n_per_family=40)
    prompts = [t["prompt"] + SUFFIX for t in tasks]
    print(f"{len(prompts)} prompts, batches {BATCHES}, {GEN} forced tokens each",
          flush=True)
    print("waiting for an exclusive GPU ...", flush=True)
    if not wait_for_exclusive():
        return

    s = PowerSampler(interval_ms=100).start()
    open_idle = baseline(s, 20, "opening idle")

    results = []
    for repo, pb in MODELS:
        print(f"\n=== {repo} ({pb}B) ===", flush=True)
        entry = {"repo": repo, "params_b": pb, "gen_tokens": GEN}
        try:
            if foreign_pids():
                wait_for_exclusive()
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            # decoder-only batching needs left padding or the generated text
            # starts after the pad run and the comparison is meaningless
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, device_map="cuda:0",
                trust_remote_code=True)
            model.eval()
            cfg = model.generation_config
            cfg.do_sample = False
            for k in ("temperature", "top_p", "top_k"):
                setattr(cfg, k, None)
            cfg.pad_token_id = tok.pad_token_id

            pre_w = baseline(s, 12, "resident idle pre")
            entry["idle_w"] = pre_w

            rows = []
            oom_at = None
            for B in BATCHES:
                texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                                 tokenize=False,
                                                 add_generation_prompt=True)
                         for p in prompts[:B]]
                # Running out of VRAM at a given batch is a deployment finding,
                # not a crash: record where the ceiling is and keep the rows
                # already measured.
                try:
                    enc = tok(texts, return_tensors="pt", padding=True).to("cuda:0")
                    with torch.inference_mode():                  # warm this shape
                        model.generate(**enc, max_new_tokens=4, min_new_tokens=4)
                    torch.cuda.synchronize()
                except torch.cuda.OutOfMemoryError as oom:
                    oom_at = B
                    print(f"    batch {B:3d}: OOM -- concurrency ceiling reached "
                          f"({str(oom)[:60]}...)", flush=True)
                    del texts
                    gc.collect(); torch.cuda.empty_cache()
                    break
                time.sleep(1.5)

                per = []
                for _ in range(REPEATS):
                    torch.cuda.synchronize(); a = time.time()
                    with torch.inference_mode():
                        model.generate(**enc, max_new_tokens=GEN, min_new_tokens=GEN)
                    torch.cuda.synchronize(); b = time.time()
                    ns = len(s.window(a, b))
                    ej = s.energy_j(a, b)
                    per.append({"dur_s": b - a, "n_samples": ns,
                                "net_j": None if (ej is None or ns < MIN_SAMPLES)
                                         else ej[0] - pre_w * (b - a)})
                    time.sleep(1.5)
                good = [p for p in per if p["net_j"] is not None]
                dur = st.mean([p["dur_s"] for p in per])
                row = {
                    "batch": B, "dur_s": dur, "n_measurable": len(good),
                    "j_batch": st.mean([p["net_j"] for p in good]) if good else None,
                    "j_req": (st.mean([p["net_j"] for p in good]) / B) if good else None,
                    "req_per_s": B / dur, "tok_per_s": B * GEN / dur,
                    "vram_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                }
                rows.append(row)
                print(f"    batch {B:3d}: {dur:6.2f} s  "
                      f"{row['j_batch'] if row['j_batch'] else float('nan'):8.1f} J/batch  "
                      f"{row['j_req'] if row['j_req'] else float('nan'):7.2f} J/req  "
                      f"{row['req_per_s']:6.2f} req/s  {row['tok_per_s']:7.1f} tok/s  "
                      f"({len(good)}/{REPEATS} olculebilir)", flush=True)
                del enc
                torch.cuda.reset_peak_memory_stats()
                gc.collect(); torch.cuda.empty_cache()

            post_w = baseline(s, 12, "resident idle post")
            entry["drift_w"] = abs(post_w - pre_w)
            entry["dirty"] = (entry["drift_w"] > DRIFT_W or bool(foreign_pids())
                              or (pre_w - open_idle) > CONTAMINATION_W)
            entry["rows"] = rows
            entry["oom_at_batch"] = oom_at

            # ---- what an operator actually pays -----------------------------
            # A card that is powered on costs P_idle all day whether or not it
            # serves. Total daily energy = idle + dynamic, so the per-request
            # bill depends on the request rate, not only on the model.
            best = min((r for r in rows if r["j_req"]), key=lambda r: r["j_req"], default=None)
            if best:
                entry["best_batch"] = best["batch"]
                entry["j_req_best"] = best["j_req"]
                entry["batch_saving"] = rows[0]["j_req"] / best["j_req"] if rows[0]["j_req"] else None
                daily_idle_j = pre_w * 86400
                entry["daily_idle_kwh"] = daily_idle_j / 3.6e6
                curve = []
                for rate in (100, 1000, 10000, 100000, 1000000):
                    tot = daily_idle_j / rate + best["j_req"]
                    curve.append({"req_per_day": rate, "j_per_req_total": tot,
                                  "idle_share": (daily_idle_j / rate) / tot})
                entry["rate_curve"] = curve
                print(f"    best batch {best['batch']} -> {best['j_req']:.2f} J/req "
                      f"({entry['batch_saving']:.1f}x better than batch 1)", flush=True)
                for c in curve:
                    print(f"      {c['req_per_day']:>8,} req/day: "
                          f"{c['j_per_req_total']:10.1f} J/req total, "
                          f"idle is {c['idle_share']*100:5.1f}%", flush=True)
            del model, tok
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {entry['error']}", flush=True)
            traceback.print_exc()
        finally:
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        results.append(entry)
        json.dump({"open_idle": open_idle, "gen": GEN, "results": results},
                  open(OUT, "w"), indent=1)

    print("\n=== SERVING ECONOMICS ===")
    for e in results:
        if "rows" not in e:
            continue
        n = e["repo"].split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")
        print(f"\n{n}  (idle {e['idle_w']:.1f} W, drift {e['drift_w']:.2f} W"
              f"{', DIRTY' if e['dirty'] else ''})")
        print(f"  {'batch':>6} {'J/req':>9} {'req/s':>8} {'tok/s':>9} {'VRAM GiB':>9}")
        for r in e["rows"]:
            print(f"  {r['batch']:6d} {r['j_req'] if r['j_req'] else float('nan'):9.2f} "
                  f"{r['req_per_s']:8.2f} {r['tok_per_s']:9.1f} {r['vram_gib']:9.2f}")
    json.dump({"open_idle": open_idle, "gen": GEN, "results": results},
              open(OUT, "w"), indent=1)
    s.stop()


if __name__ == "__main__":
    main()
