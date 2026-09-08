from src.backends.postgres_backend import PostgresBackend
from src.candidates import IndexCandidate
from src.verifier import apply_index
from src.workload import load_workload
import yaml
import time

config = yaml.safe_load(open("config.yaml"))
pg = config["sandbox"]["postgres"]


def backend_factory():
    b = PostgresBackend(pg["dsn"], pg["template_db"])
    b.clone_from_source()
    return b


candidate = IndexCandidate(
    table="lineitem", columns=["l_shipdate", "l_returnflag"],
    rationale="decomposed timing check",
)


def timed_ops(backend, offset, n_rows=5000):
    columns = [c["name"] for c in backend.get_columns("lineitem")]
    select_exprs = [f"{c} + {offset}" if c ==
                    "l_linenumber" else c for c in columns]
    cols_csv = ", ".join(columns)
    select_csv = ", ".join(select_exprs)

    t0 = time.perf_counter()
    backend.execute_dml(
        f"INSERT INTO lineitem ({cols_csv}) SELECT {select_csv} FROM lineitem ORDER BY l_orderkey LIMIT {n_rows}")
    insert_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    backend.execute_dml(
        f"UPDATE lineitem SET l_shipdate = l_shipdate WHERE l_linenumber > {offset}")
    update_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    backend.execute_dml(f"DELETE FROM lineitem WHERE l_linenumber > {offset}")
    delete_t = time.perf_counter() - t0

    return insert_t, update_t, delete_t


# baseline (no candidate index)
b1 = backend_factory()
base = timed_ops(b1, 10_000_000)
b1.close()

# candidate (index applied)
b2 = backend_factory()
apply_index(b2, candidate)
cand = timed_ops(b2, 10_000_000)
b2.close()

print(f"{'op':10} {'baseline':>10} {'candidate':>10} {'delta':>8}")
for name, base_t, cand_t in zip(["INSERT", "UPDATE", "DELETE"], base, cand):
    print(f"{name:10} {base_t:10.4f} {cand_t:10.4f} {(cand_t/base_t - 1)*100:+7.1f}%")
