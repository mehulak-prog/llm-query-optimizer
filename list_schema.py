from src.backends.postgres_backend import PostgresBackend
import yaml

config = yaml.safe_load(open("config.yaml"))
pg = config["sandbox"]["postgres"]

backend = PostgresBackend(pg["dsn"], pg["template_db"])
backend.clone_from_source()

for table in backend.list_tables():
    cols = backend.get_columns(table)
    row_count = backend.get_row_count(table)
    print(f"\n{table}  ({row_count} rows)")
    for c in cols:
        pk = " [PK]" if c["is_primary_key"] else ""
        print(f"  {c['name']} ({c['data_type']}){pk}")

backend.close()
