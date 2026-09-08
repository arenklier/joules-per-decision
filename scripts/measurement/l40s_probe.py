"""Idle-vs-load GPU energy probe on the L40S, mirroring the local A2000 run.

Same question as before and the only one that matters here: does the energy a
single generation costs separate from the idle floor? The L40S idles at ~84 W
where the A2000 idled at 8 W, so the separation has to be re-measured rather
than assumed.
"""
import os, time, json, statistics as st, sys
os.environ.setdefault("HF_HOME", "/mnt/data/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from power_sampler import PowerSampler

MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW = 128
N_RUNS = 5

PROMPT = ("A Cisco access switch port is stuck in err-disabled state after a "
          "user plugged in an unmanaged switch. Name the most likely cause and "
          "the command that clears it.")


def main():
    s = PowerSampler(interval_ms=100).start()

    print("Idle baseline, 20 s ...", flush=True)
    ti0 = time.time(); time.sleep(20); ti1 = time.time()
    idle_w = s.mean_w(ti0, ti1)
    idle_series = [smp[1][0] for smp in s.window(ti0, ti1)]
    print(f"  samples={len(idle_series)}  idle = {idle_w[0]:.2f} W "
          f"(sd {st.pstdev(idle_series):.2f}, min {min(idle_series):.2f}, "
          f"max {max(idle_series):.2f})", flush=True)

    print(f"\nLoading {MODEL} (bf16) ...", flush=True)
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"  loaded in {time.time()-t_load:.1f} s  "
          f"({torch.cuda.memory_allocated()/2**30:.2f} GiB allocated)", flush=True)

    # Idle-with-weights-resident: the floor a served model pays for existing.
    print("\nResident-idle baseline (weights loaded, no work), 15 s ...", flush=True)
    tr0 = time.time(); time.sleep(15); tr1 = time.time()
    res_w = s.mean_w(tr0, tr1)
    res_series = [smp[1][0] for smp in s.window(tr0, tr1)]
    print(f"  resident idle = {res_w[0]:.2f} W (sd {st.pstdev(res_series):.2f})",
          flush=True)

    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([text], return_tensors="pt").to("cuda:0")

    print("\nWarm-up (not measured) ...", flush=True)
    with torch.inference_mode():
        model.generate(**enc, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize(); time.sleep(3)

    print("\nMeasured generations:", flush=True)
    rows = []
    for i in range(N_RUNS):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()
        t1 = time.time()
        new_tok = out.shape[-1] - enc["input_ids"].shape[-1]
        e = s.energy_j(t0, t1)
        if e is None:
            print(f"  run {i}: window too short ({t1-t0:.2f} s)")
            continue
        dur = t1 - t0
        net_cold = e[0] - idle_w[0] * dur      # vs an empty GPU
        net_res = e[0] - res_w[0] * dur        # vs a GPU already holding weights
        rows.append((dur, new_tok, e[0], net_cold, net_res))
        print(f"  run {i}: {dur:6.2f} s  {new_tok:4d} tok  total {e[0]:7.1f} J  "
              f"net(vs idle) {net_cold:7.1f} J  net(vs resident) {net_res:7.1f} J",
              flush=True)
        time.sleep(4)

    if not rows:
        print("\nNo measurable runs."); s.stop(); return

    print("\n--- verdict ---")
    loaded_w = [r[2] / r[0] for r in rows]
    print(f"loaded mean      = {st.mean(loaded_w):7.2f} W")
    print(f"idle (empty GPU) = {idle_w[0]:7.2f} W   -> delta {st.mean(loaded_w)-idle_w[0]:+7.2f} W")
    print(f"idle (resident)  = {res_w[0]:7.2f} W   -> delta {st.mean(loaded_w)-res_w[0]:+7.2f} W")
    nc = [r[3] for r in rows]; nr = [r[4] for r in rows]
    print(f"net J / request (vs idle)     = {st.mean(nc):8.2f}  (sd {st.pstdev(nc):.2f}, n={len(nc)})")
    print(f"net J / request (vs resident) = {st.mean(nr):8.2f}  (sd {st.pstdev(nr):.2f})")
    jt = [r[3] / r[1] for r in rows if r[1]]
    print(f"J / output token (vs idle)    = {st.mean(jt):.4f}")
    sd_idle = st.pstdev(idle_series) or 1e-9
    print(f"separation: delta / idle-sd   = {(st.mean(loaded_w)-idle_w[0])/sd_idle:.1f} sigma")

    json.dump({"model": MODEL, "idle_w": idle_w[0], "resident_w": res_w[0],
               "rows": rows, "idle_sd": st.pstdev(idle_series)},
              open("l40s_probe_result.json", "w"), indent=1)
    s.stop()


if __name__ == "__main__":
    main()
