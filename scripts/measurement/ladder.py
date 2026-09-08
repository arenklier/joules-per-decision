"""Energy-per-decision across a model-size ladder on one L40S.

Same family (Qwen2.5 1.5B -> 14B) isolates scale; three ~7-8B models from
other families sit at one rung to separate a scale effect from a family
effect. Every cell is a fixed 128 greedy tokens, so joules-per-token is
comparable across rows without a length confound.
"""
import os, gc, json, time, traceback, statistics as st
os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from power_sampler import PowerSampler

MAX_NEW = 128
RUNS_PER_PROMPT = 3

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


def idle_watts(s, seconds, label):
    t0 = time.time(); time.sleep(seconds); t1 = time.time()
    w = s.mean_w(t0, t1)
    series = [smp[1][0] for smp in s.window(t0, t1)]
    sd = st.pstdev(series) if len(series) > 1 else 0.0
    print(f"  [{label}] {w[0]:7.2f} W  (sd {sd:.2f}, n={len(series)})", flush=True)
    return w[0], sd


def main():
    s = PowerSampler(interval_ms=100).start()
    print("=== global idle baseline (empty GPU) ===", flush=True)
    idle0, idle0_sd = idle_watts(s, 20, "idle")

    results = []
    for repo, family, params_b in LADDER:
        print(f"\n=== {repo}  ({family}, {params_b}B) ===", flush=True)
        entry = {"repo": repo, "family": family, "params_b": params_b}
        try:
            t_load = time.time()
            tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16, device_map="cuda:0",
                trust_remote_code=True)
            model.eval()
            # greedy only -- drop sampling knobs so the config warnings and any
            # stochasticity both go away
            gc_cfg = model.generation_config
            gc_cfg.do_sample = False
            for k in ("temperature", "top_p", "top_k"):
                setattr(gc_cfg, k, None)
            load_s = time.time() - t_load
            vram = torch.cuda.memory_allocated() / 2**30
            print(f"  loaded in {load_s:.1f} s, {vram:.2f} GiB", flush=True)
            entry.update(load_s=load_s, vram_gib=vram)

            res_w, res_sd = idle_watts(s, 12, "resident idle")
            entry.update(resident_w=res_w, resident_sd=res_sd)

            # warm-up, unmeasured
            enc0 = tok([tok.apply_chat_template(
                [{"role": "user", "content": PROMPTS[0][1]}],
                tokenize=False, add_generation_prompt=True)],
                return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                model.generate(**enc0, max_new_tokens=8)
            torch.cuda.synchronize(); time.sleep(2)

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
                                 "net_j": e[0] - idle0 * dur,
                                 "tok_per_s": ntok / dur})
                    time.sleep(2)

            if rows:
                net = [r["net_j"] for r in rows]
                jt = [r["net_j"] / r["new_tok"] for r in rows if r["new_tok"]]
                tps = [r["tok_per_s"] for r in rows]
                w_load = [r["total_j"] / r["dur_s"] for r in rows]
                entry.update(
                    n=len(rows),
                    net_j_mean=st.mean(net), net_j_sd=st.pstdev(net),
                    j_per_tok=st.mean(jt), j_per_tok_sd=st.pstdev(jt),
                    tok_per_s=st.mean(tps), loaded_w=st.mean(w_load))
                print(f"  -> {st.mean(net):7.1f} J/req (sd {st.pstdev(net):.1f}, n={len(rows)})"
                      f"   {st.mean(jt):.3f} J/tok   {st.mean(tps):5.1f} tok/s"
                      f"   {st.mean(w_load):6.1f} W", flush=True)
            del model, tok
        except Exception as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {entry['error']}", flush=True)
            traceback.print_exc()
        finally:
            for v in ("model", "tok"):
                if v in dir():
                    pass
            gc.collect(); torch.cuda.empty_cache(); time.sleep(3)
        results.append(entry)
        json.dump({"idle_w": idle0, "idle_sd": idle0_sd, "max_new": MAX_NEW,
                   "results": results}, open("ladder_result.json", "w"), indent=1)

    print("\n=== closing idle baseline (drift check) ===", flush=True)
    idle1, _ = idle_watts(s, 15, "idle")
    print(f"  drift over run: {idle1 - idle0:+.2f} W", flush=True)

    print("\n=== SUMMARY ===")
    print(f"{'model':44s} {'B':>5} {'J/req':>9} {'J/tok':>8} {'tok/s':>7} {'W':>7}")
    for e in results:
        if "net_j_mean" in e:
            print(f"{e['repo']:44s} {e['params_b']:5.1f} {e['net_j_mean']:9.1f} "
                  f"{e['j_per_tok']:8.3f} {e['tok_per_s']:7.1f} {e['loaded_w']:7.1f}")
        else:
            print(f"{e['repo']:44s} {e['params_b']:5.1f}   {e.get('error','?')[:40]}")

    json.dump({"idle_w": idle0, "idle_sd": idle0_sd, "idle_close": idle1,
               "max_new": MAX_NEW, "results": results},
              open("ladder_result.json", "w"), indent=1)
    s.stop()


if __name__ == "__main__":
    main()
