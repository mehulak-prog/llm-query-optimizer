from src.backends.postgres_backend import PostgresBackend
from src.candidates import IndexCandidate
from src.verifier import verify_index_candidate
from src.workload import load_workload
import yaml
import time

config = yaml.safe_load(open("config.yaml"))
pg = config["sandbox"]["postgres"]


def backend_factory():
    b = PostgresBackend(pg["dsn"], pg["template_db"])
    b.clone_from_source()
    return b


workload = load_workload(config["workload"]["path"])

candidate = IndexCandidate(
    table="part",
    columns=["p_brand"],
    rationale="confirming predicted write-timing cost on an FK-parent table referenced by lineitem (398K rows) with no supporting index on lineitem.l_partkey",
)

t0 = time.perf_counter()
result = verify_index_candidate(
    backend_factory,
    candidate,
    workload=workload,
    baseline_seconds=1.0,
    repeat=5,
    warmup=1,
    max_storage_bytes=500 * 1024 * 1024,
    write_bench_rows=config["write_bench"]["rows"],
)
print(f"total time: {time.perf_counter() - t0:.1f}s")
print("write_baseline_seconds:", result.write_baseline_seconds)
print("write_candidate_seconds:", result.write_candidate_seconds)
print("error:", result.error)
