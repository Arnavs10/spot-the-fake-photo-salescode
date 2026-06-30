"""
bench.py -- measures average latency per image (the number the brief asks for).

  python bench.py            # benchmarks over ./data
  python bench.py some.jpg   # benchmarks repeatedly on one image
"""
import sys, glob, os, time, platform
import numpy as np
from predict import predict, _model

_model()  # warm up: load model once before timing

if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    paths = [sys.argv[1]] * 50
else:
    paths = glob.glob("data/real/*") + glob.glob("data/screen/*")

# warm-up run (first call pays import/JIT-ish costs)
predict(paths[0])

t = []
for p in paths:
    s = time.perf_counter()
    predict(p)
    t.append((time.perf_counter() - s) * 1000)

t = np.array(t)
print(f"device : {platform.processor() or platform.machine()}  "
      f"({platform.system()})")
print(f"images : {len(t)}")
print(f"latency: mean {t.mean():.1f} ms | median {np.median(t):.1f} ms | "
      f"p95 {np.percentile(t,95):.1f} ms")
print(f"throughput: ~{1000/t.mean():.0f} images/sec on one core")
