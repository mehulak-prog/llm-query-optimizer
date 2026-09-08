# LLM-as-Planner-Assistant: A Verification-Loop Architecture for Query Optimization

A research scaffold for the system described in `docs/research_framing.md`:
an LLM reads a SQL workload, a schema, and sample table statistics, and
proposes indexes, materialized views, and query rewrites. Every proposal is
replayed on a sandbox database and scored by *actual measured execution*,
not by the LLM's self-reported confidence. That measured runtime is the
reward signal that closes the loop.

```
 workload.sql ─┐
 schema.sql    ├──► LLMPlanner ──► candidates (index/view/rewrite JSON)
 stats.json   ─┘         ▲                │
                          │                ▼
                   refine w/ results   SandboxDB (replay + EXPLAIN + timing)
                          │                │
                          └──── Verifier ◄─┘
                                   │
                                   ▼
                             Reward + ranked
                             recommendation set
                                   │
                                   ▼
                        Benchmark vs. baselines
                 (Postgres EXPLAIN advisor, rule-based
                  AutoAdmin-style, EverSQL/pganalyze-style)
```

The novelty this scaffold is built to support is **not** "LLM writes SQL
advice." It's the closed loop: candidate generation is cheap and
unreliable; verification against a real execution engine is the part that
makes the recommendations trustworthy, and the part nobody in the pure-ML
literature bothers to build because it requires DB internals (cost
estimation, index selection theory, plan stability) most ML researchers
don't have, and DB researchers haven't yet wired up to an LLM candidate
generator with a refinement loop.

## What's actually implemented vs. what you need to plug in

This scaffold **runs end-to-end today** on a synthetic TPC-H-style schema
using SQLite as the sandbox backend (zero external dependencies, so you can
verify the whole pipeline works before spending API credits or standing up
Postgres). You said you'll do "all the replacing parts" — here's exactly
what those are:

| Component | Current state | What to replace |
|---|---|---|
| `src/llm_client.py` | Mock heuristic generator + a real Anthropic-API code path already wired up | Set `ANTHROPIC_API_KEY`, flip `mock_mode: false` in `config.yaml`. Prompts are in `src/prompts/` — tune them. |
| `src/backends/sqlite_backend.py` | Fully working sandbox | Fine to keep for dev/unit tests |
| `src/backends/postgres_backend.py` | Stub with correct interface, `psycopg2` calls written but untested against a live server | Point at a real disposable Postgres instance (e.g. a Docker container or RDS snapshot clone). This is where "actual execution as reward" becomes real instead of a SQLite proxy. |
| `data/schema.sql`, `data/generate_data.py` | Synthetic 6-table TPC-H-style schema + data generator | Swap for your real (anonymized/sampled) schema + a `pg_dump`/sample of real stats |
| `data/workload.json` | 12 synthetic queries with weights | Swap for real workload capture (`pg_stat_statements`, slow query log, etc.) |
| `src/baselines/` | Two working heuristic baselines that approximate AutoAdmin-style and EverSQL/pganalyze-style rule logic | Optionally replace with the actual tools' output if you have access, for a stronger baseline comparison in the paper |
| `experiments/run_demo.py` | Runs the full loop on SQLite today | Add a `run_postgres_experiment.py` once the Postgres backend is live |

## Quickstart

```bash
cd llm-query-optimizer
pip install -r requirements.txt
python data/generate_data.py                 # builds sandbox.db (SQLite)
python experiments/run_demo.py               # runs the full loop + benchmark
```

This will:
1. Build a synthetic TPC-H-style SQLite database (`data/sandbox.db`).
2. Extract schema + statistics.
3. Run the LLM planner (mock mode by default — no API key needed) to get
   candidate indexes/views/rewrites.
4. Replay each candidate in an isolated sandbox copy, measure runtime
   before/after, and verify correctness (result-set hash comparison for
   rewrites).
5. Score candidates with the reward function in `src/reward.py`.
6. Run the two baseline advisors on the same workload.
7. Print a comparison table and write `experiments/results.csv`.

To use the real Claude API instead of the mock generator:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# edit config.yaml: llm.mock_mode: false
python experiments/run_demo.py
```

## Why this is a defensible paper, not a toy

See `docs/research_framing.md` for:
- Positioning against AutoAdmin (Chaudhuri & Narasayya), Postgres's
  `pg_hypopg`/`HypoPG`-based EXPLAIN advisors, and commercial rule-based
  tools (EverSQL, pganalyze).
- A related-work skeleton you can fill in with citations.
- Suggested benchmark suites (TPC-H, TPC-DS, the Join Order Benchmark /
  IMDB) and evaluation metrics (runtime reduction, storage overhead,
  recommendation precision/recall against a brute-force optimum on small
  instances, plan stability across LLM samples).
- Target venues and what each expects from an evaluation section.
- The specific gap: DB venues rarely have LLM expertise on PC; ML venues
  don't score query-plan quality. AIDB/DBML workshops are named as the
  place to get fast feedback before a full VLDB/SIGMOD submission.

## Project layout

```
llm-query-optimizer/
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── src/
│   ├── schema.py           # schema extraction
│   ├── stats.py             # table/column statistics collection
│   ├── workload.py          # workload loading + parsing
│   ├── llm_client.py        # LLM candidate generator (mock + real API)
│   ├── prompts/             # prompt templates sent to the LLM
│   ├── candidates.py         # typed candidate objects + JSON (de)serialization
│   ├── backends/             # pluggable sandbox DB backends
│   │   ├── base.py
│   │   ├── sqlite_backend.py
│   │   └── postgres_backend.py
│   ├── sandbox.py            # sandbox lifecycle: clone, apply, measure, rollback
│   ├── verifier.py           # before/after timing + correctness verification
│   ├── reward.py             # reward / scoring function
│   ├── optimizer_loop.py     # the closed loop orchestrator
│   ├── baselines/
│   │   ├── explain_advisor.py    # Postgres-EXPLAIN-style heuristic baseline
│   │   └── rule_based.py         # EverSQL/pganalyze-style rule baseline
│   ├── benchmark.py          # runs system + baselines, produces comparison table
│   └── metrics.py
├── data/
│   ├── schema.sql
│   ├── generate_data.py
│   └── workload.json
├── experiments/
│   ├── run_demo.py
│   └── results.csv           # produced after running
├── tests/
│   ├── test_backends.py
│   ├── test_candidates.py
│   └── test_reward.py
└── docs/
    ├── architecture.md
    └── research_framing.md
```
