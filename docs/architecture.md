# Architecture

## Data flow

```
                    ┌─────────────────────────────────────────┐
                    │              source database              │
                    │   (real Postgres in prod; SQLite demo)    │
                    └───────────────────┬───────────────────────┘
                                         │ clone (per trial, isolated)
                                         ▼
   config.yaml ──►  ┌──────────────────────────────────────────┐
                    │            SandboxBackend                  │
                    │  (src/backends/{sqlite,postgres}_backend)  │
                    └───────┬───────────────────────┬────────────┘
                            │                        │
                 schema.py  │             stats.py   │
                 extract_schema()        collect_stats()
                            │                        │
                            ▼                        ▼
                    DatabaseSchema              TableStats dict
                            │                        │
                            └──────────┬─────────────┘
                                       ▼
                        workload.py: Workload (weighted queries)
                                       │
                                       ▼
                        src/llm_client.py: LLMPlanner.propose()
                     (mock heuristic OR real Anthropic API call)
                                       │
                                       ▼
                        src/candidates.py: CandidateSet
                       (IndexCandidate | ViewCandidate | RewriteCandidate)
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
             verify_index_*     verify_view_*      verify_rewrite_*
           (src/verifier.py -- each candidate gets its OWN fresh sandbox clone)
                    │                  │                  │
                    └──────────────────┼──────────────────┘
                                       ▼
                          src/reward.py: score()
                     (measured speedup -> accept/reject + reward)
                                       │
                       accepted/rejected fed back as
                     prior_results_text for refinement round
                       (bounded by llm.max_refinement_rounds)
                                       │
                                       ▼
                        src/optimizer_loop.py: LoopResult
                                       │
                                       ▼
                   src/benchmark.py: compare against baselines
              (src/baselines/explain_advisor.py, rule_based.py)
                                       │
                                       ▼
                    experiments/run_demo.py: table + results.csv
```

## Why each verification trial gets its own sandbox clone

Two candidates must never be measured in a sandbox that still carries the
side effects of a previous candidate -- otherwise "candidate B was fast"
might really mean "candidate A's index, still present, did the work."
`_make_backend_factory` in `optimizer_loop.py` returns a factory, not a
live connection, specifically so every `verify_*` call can clone fresh
state. This costs more wall-clock time than reusing one sandbox, but it's
the only way the measured numbers are attributable to a single candidate
-- which is the entire point of using execution as ground truth instead
of the LLM's self-reported confidence.

## Why indexes, views, and rewrites are typed objects, not free text

`src/candidates.py` refuses to accept anything that isn't a structured,
schema-validated candidate. An LLM response like "you should probably add
some indexes on the orders table" cannot be mechanically applied to a
sandbox or mechanically rolled back, so it cannot be verified, so it's
useless to this system by construction. The strict parser
(`parse_llm_response`) is a design decision, not an implementation detail
-- it's what keeps "verification loop" from meaning "we eyeballed the
LLM's suggestions."

## Where the SQLite demo diverges from the real system

Documented here so nobody mistakes demo numbers for paper numbers:

1. **Optimizer sophistication.** SQLite's query planner is much simpler
   than Postgres/MySQL's cost-based optimizer. An index that gives a 5x
   speedup in SQLite might give 50x or 1.1x in Postgres depending on plan
   selection details this system doesn't control. Use SQLite only to
   validate plumbing correctness.
2. **No true materialized views.** SQLite has no `CREATE MATERIALIZED
   VIEW`; `sqlite_backend.py` silently downgrades to a regular view. This
   means the SQLite demo cannot actually show the storage/staleness
   trade-off a materialized view makes in Postgres. The Postgres backend
   does not have this limitation.
3. **No native query rewriting via views.** Postgres and most real
   engines transparently rewrite queries to use a matching materialized
   view under specific conditions; SQLite doesn't, so
   `verify_view_candidate`'s "did the workload get faster" signal is
   weaker on SQLite than it will be on Postgres.
4. **Mock LLM mode is a heuristic fixture**, not a language model. It's
   there so `run_demo.py` works with zero setup and so you can unit-test
   the verifier/reward/loop without spending API calls. Flip
   `llm.mock_mode: false` before generating any numbers you intend to put
   in a paper.
