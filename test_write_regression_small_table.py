from src.backends.postgres_backend import PostgresBackend
from src.candidates import IndexCandidate
from src.verifier import verify_index_candidate
from src.workload import load_workload
import yaml

config = yaml.safe_load(open("config.yaml"))
pg = config["sandbox"]["postgres"]


def backend_factory():
    b = PostgresBackend(pg["dsn"], pg["template_db"])
    b.clone_from_source()
    return b


workload = load_workload(config["workload"]["path"])

candidate = IndexCandidate(
    table="nation",
    columns=["n_regionkey"],
    rationale="small-table index test candidate for write-regression sanity check",
)

result = verify_index_candidate(
    backend_factory,
    candidate,
    workload=workload,
    baseline_seconds=1.0,
    repeat=5,
    warmup=1,
    write_bench_rows=5000,
    max_storage_bytes=500 * 1024 * 1024,
)

print("write_baseline_seconds:", result.write_baseline_seconds)
print("write_candidate_seconds:", result.write_candidate_seconds)
print("error:", result.error)
