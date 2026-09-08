from src.optimizer_loop import run_loop
from src.workload import load_workload

import yaml
import traceback
import logging
logging.getLogger("src.verifier").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING, format="%(message)s")

config = yaml.safe_load(open("config.yaml"))
workload = load_workload(config["workload"]["path"])

print("Starting run_loop()... (this calls the live Gemini API and runs real DB timing, expect several minutes)")
try:
    result = run_loop(workload, config)
except Exception:
    print("run_loop() raised an exception:")
    traceback.print_exc()
    raise SystemExit(1)

print(f"\nrounds_run: {result.rounds_run}")
print(f"baseline_weighted_seconds: {result.baseline_weighted_seconds}")
print(f"accepted: {len(result.accepted)}")
print(f"rejected: {len(result.rejected)}")

for label, group in [("ACCEPTED", result.accepted), ("REJECTED", result.rejected)]:
    print(f"\n--- {label} ---")
    for sc in group:
        r = sc.result
        c = r.candidate
        desc = f"{r.candidate_kind} on {getattr(c, 'table', getattr(c, 'name', '?'))}"
        print(f"  {desc}: speedup={r.speedup:.3f}x reward={sc.reward:.4f}")
        print(f"    reason: {sc.reason}")
        if r.write_baseline_seconds and r.write_candidate_seconds:
            regression = r.write_candidate_seconds / r.write_baseline_seconds - 1
            print(
                f"    write_regression: {regression:.3f} (trials: {r.write_regression_trials})")
