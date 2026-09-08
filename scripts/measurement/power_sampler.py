"""Continuous GPU power sampler built on a single streaming nvidia-smi process.

One process, polled at a fixed interval, so the sampler itself costs almost
nothing -- spawning nvidia-smi per sample would cost more than the signal we
are trying to measure.
"""
import subprocess, threading, time, bisect


class PowerSampler:
    def __init__(self, interval_ms=100):
        self.interval_ms = interval_ms
        self.samples = []          # (wall_time, [watts per gpu])
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def _run(self):
        cmd = ["nvidia-smi", "--query-gpu=index,power.draw",
               "--format=csv,noheader,nounits", f"--loop-ms={self.interval_ms}"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True,
                                      bufsize=1)
        pending = {}
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                idx, watt = int(parts[0]), float(parts[1])
            except ValueError:
                continue
            if idx in pending:                  # new sweep started -> flush
                self.samples.append((time.time(),
                                     [pending[k] for k in sorted(pending)]))
                pending = {}
            pending[idx] = watt

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(1.5)                          # let the stream warm up
        return self

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()

    def window(self, t0, t1):
        """Samples inside [t0, t1]."""
        times = [s[0] for s in self.samples]
        i = bisect.bisect_left(times, t0)
        j = bisect.bisect_right(times, t1)
        return self.samples[i:j]

    def energy_j(self, t0, t1, gpu=None):
        """Trapezoidal integral of power over the window, in joules per GPU."""
        w = self.window(t0, t1)
        if len(w) < 2:
            return None
        n = len(w[0][1])
        out = [0.0] * n
        for (ta, pa), (tb, pb) in zip(w, w[1:]):
            dt = tb - ta
            for g in range(n):
                out[g] += 0.5 * (pa[g] + pb[g]) * dt
        return out if gpu is None else out[gpu]

    def mean_w(self, t0, t1):
        w = self.window(t0, t1)
        if not w:
            return None
        n = len(w[0][1])
        return [sum(s[1][g] for s in w) / len(w) for g in range(n)]
