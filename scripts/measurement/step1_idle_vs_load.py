"""Decisive feasibility test: does per-request GPU energy rise above idle noise?

If the idle band and the loaded band overlap, joules-per-decision cannot be
measured on this hardware and the whole study is dead. Everything else is
downstream of this one number.
"""
import json, time, statistics as st, urllib.request, sys
from power_sampler import PowerSampler

OLLAMA = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b"

PROMPT = ("A Cisco access switch port is stuck in err-disabled state after a "
          "user plugged in an unmanaged switch. Name the most likely cause and "
          "the command that clears it.")


def generate(prompt, num_predict=128):
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"num_predict": num_predict,
                                   "temperature": 0.0}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        payload = json.load(r)
    t1 = time.time()
    return t0, t1, payload


def main():
    s = PowerSampler(interval_ms=100).start()

    print("Sampling idle for 20 s ...", flush=True)
    ti0 = time.time()
    time.sleep(20)
    ti1 = time.time()

    idle_w = s.mean_w(ti0, ti1)
    n_gpu = len(idle_w)
    idle_series = [[smp[1][g] for smp in s.window(ti0, ti1)] for g in range(n_gpu)]
    print(f"idle samples: {len(idle_series[0])}")
    for g in range(n_gpu):
        sd = st.pstdev(idle_series[g]) if len(idle_series[g]) > 1 else 0.0
        print(f"  GPU{g} idle = {idle_w[g]:6.2f} W  (sd {sd:.2f}, "
              f"min {min(idle_series[g]):.2f}, max {max(idle_series[g]):.2f})")

    print("\nWarm-up request (loads the model into VRAM, not measured) ...",
          flush=True)
    generate("hello", num_predict=8)
    time.sleep(3)

    print("\nMeasured requests:", flush=True)
    rows = []
    for i in range(5):
        t0, t1, p = generate(PROMPT)
        e = s.energy_j(t0, t1)
        if e is None:
            print(f"  run {i}: window too short for {t1-t0:.2f}s -- no samples")
            continue
        dur = t1 - t0
        tok = p.get("eval_count", 0)
        # energy above the idle floor: what the request actually cost
        net = [e[g] - idle_w[g] * dur for g in range(n_gpu)]
        rows.append((dur, tok, e, net))
        print(f"  run {i}: {dur:6.2f} s  {tok:4d} tok  "
              f"total J = {[f'{x:.1f}' for x in e]}  "
              f"net J = {[f'{x:.1f}' for x in net]}")
        time.sleep(4)

    if not rows:
        print("\nNo measurable runs.")
        return

    print("\n--- verdict ---")
    for g in range(n_gpu):
        loaded_w = [r[2][g] / r[0] for r in rows]
        net_j = [r[3][g] for r in rows]
        print(f"GPU{g}: loaded mean = {st.mean(loaded_w):6.2f} W vs "
              f"idle {idle_w[g]:6.2f} W   ->  delta {st.mean(loaded_w)-idle_w[g]:+6.2f} W")
        print(f"      net energy per request = {st.mean(net_j):8.2f} J "
              f"(sd {st.pstdev(net_j):.2f}, n={len(net_j)})")
        per_tok = [r[3][g] / r[1] for r in rows if r[1]]
        if per_tok:
            print(f"      joules per output token = {st.mean(per_tok):.4f} J")
    s.stop()


if __name__ == "__main__":
    main()
