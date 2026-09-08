from src.backends.postgres_backend import PostgresBackend
from src.candidates import IndexCandidate
from src.verifier import verify_index_candidate_with_trials
from src.workload import load_workload
import yaml
import logging

logging.getLogger("src.verifier").setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING, format="%(message)s")

config = yaml.safe_load(open("config.yaml"))
pg = config["sandbox"]["postgres"]


def backend_factory():
    b = PostgresBackend(pg["dsn"], pg["template_db"])
    b.clone_from_source()
    return b


workload = load_workload(config["workload"]["path"])

candidate = IndexCandidate(
    table="lineitem",
    columns=["l_shipdate", "l_returnflag"],
    rationale="re-check: run_loop showed this composite candidate with negative write regression, confirming or ruling out before trusting the number",
)

result = verify_index_candidate_with_trials(
    backend_factory,
    candidate,
    workload=workload,
    baseline_seconds=1.0,
    repeat=5,
    warmup=1,
    max_storage_bytes=500 * 1024 * 1024,
    write_bench_rows=config["write_bench"]["rows"],
    write_trials=config["write_bench"]["trials"],
)

print("write_baseline_seconds:", result.write_baseline_seconds)
print("write_candidate_seconds:", result.write_candidate_seconds)
print("regression:", result.write_candidate_seconds /
      result.write_baseline_seconds - 1)
print("all trial ratios:", result.write_regression_trials)
print("error:", result.error)
