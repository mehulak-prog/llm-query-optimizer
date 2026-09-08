# Research framing notes

These are working notes, not a finished related-work section -- fill in
citations from actual reading before submission. Treat every unlinked
claim below as "verify this before you write it down as fact."

## The gap this system targets

- DB venues (VLDB, SIGMOD, ICDE) have decades of index-selection and
  view-selection theory (AutoAdmin's what-if optimizer, INUM, workload
  compression) but are only starting to publish LLM-adjacent work, and
  most of what exists is "LLM writes SQL from natural language"
  (text-to-SQL), not "LLM assists the physical design / query
  optimization layer."
- ML venues (NeurIPS, ICLR) that touch databases mostly focus on learned
  cost estimators, learned indexes (the Kraska et al. "Case for Learned
  Index Structures" line), or text-to-SQL benchmarks (Spider, BIRD).
  Almost none evaluate against real measured execution on a workload the
  way a systems paper would.
- The specific claim to defend: an LLM used as a *candidate generator*
  inside a *verification loop that uses real execution as ground truth*
  is a different system-design point than either (a) trusting the LLM's
  own judgment of what will be fast, or (b) a classical cost-model-only
  advisor with no LLM. Neither pure-ML nor pure-DB work currently
  occupies this point cleanly -- that's the lane.

## Related work to actually read and cite (not yet verified/summarized here)

- Chaudhuri, S. & Narasayya, V. -- the AutoAdmin project papers on
  index selection and what-if analysis (Microsoft Research, late
  1990s-2000s). This is the closest classical baseline; read the
  original what-if-optimizer papers, not just secondary summaries.
- HypoPG (Postgres hypothetical index extension) -- practical tool this
  system's Postgres backend can lean on for storage estimation; cite the
  project directly.
- Any recent (2023-2026) VLDB/SIGMOD/ICDE papers combining LLMs with
  physical database design, index selection, or query optimization --
  search "LLM index selection", "LLM query optimizer", "LLM database
  tuning" on top of the usual venue proceedings, not just arXiv, since
  arXiv preprints in this space move fast and quality varies.
- Text-to-SQL literature (Spider, BIRD, and the models built for them)
  is adjacent but should be explicitly distinguished in the paper: this
  system does not translate natural language to SQL; it optimizes an
  existing SQL workload's physical design.
- Learned query optimizers (Neo, Bao from MSR; Balsa) are a different
  point in the design space (learned cost models / plan selection
  inside the optimizer) worth a paragraph distinguishing from
  "LLM-as-external-advisor," which is what this system is.
- EverSQL and pganalyze -- cite their public documentation/blog posts on
  advisor logic (not their closed-source implementation) for the
  rule-based baseline comparison; be precise that `src/baselines/` is a
  faithful-effort reimplementation of documented behavior, not their
  actual code.

## Suggested benchmark suites, in increasing realism/cost

1. **This repo's synthetic TPC-H-shaped workload** (SQLite) -- plumbing
   validation only, not paper-worthy on its own.
2. **Full TPC-H at scale factor 1-10 on Postgres** -- standard, well
   understood, reviewers will recognize it immediately. Good primary
   benchmark.
3. **TPC-DS** -- more complex query shapes (larger joins, more
   subqueries), useful for showing the rewrite-candidate path earns its
   keep, not just the index path.
4. **Join Order Benchmark (JOB) / IMDB dataset** -- real-world skewed
   data distributions (not TPC's synthetic uniform-ish generator), which
   is exactly where statistics-blind rule-based baselines
   (`rule_based.py`) tend to fail and where the LLM+verification
   approach should differentiate itself, if it's actually working.
5. **A real, anonymized production workload** if you can get access to
   one -- this is what reviewers will most want to see for the
   "genuinely novel as a system" claim, and what separates a workshop
   paper from a VLDB/SIGMOD submission.

## Evaluation metrics to report (draft list)

- **Weighted workload runtime reduction** (this repo already computes
  this via `weighted_total_seconds` in `sandbox.py` / `verifier.py`).
- **Storage overhead** of accepted index/view candidates
  (`estimate_index_storage_bytes`, or actual on-disk size after
  `ANALYZE` on Postgres -- more accurate, do this for the paper).
- **Write-path regression.** `reward.py` currently applies a flat
  penalty as a placeholder (see the TODO comment there). For the paper,
  actually measure INSERT/UPDATE/DELETE throughput before/after each
  accepted index on a representative write mix -- a flat penalty will
  not survive review.
- **Recommendation precision/recall against a brute-force optimum** on a
  small enough schema/workload that exhaustive search over index
  combinations is tractable (this is the standard way index-selection
  papers validate they're not just accepting random improvements).
- **Plan stability across LLM samples.** Run the planner N times at
  temperature > 0 on the identical input and report variance in accepted
  candidates -- reviewers will ask about this given LLM non-determinism,
  better to have the number ready.
- **Cost.** Report $ and latency of the LLM calls themselves (tokens in
  schema+stats+workload can get large -- `to_llm_text()` methods in
  `schema.py`/`stats.py` are the place to add truncation/summarization
  if a real workload's schema is too big to fit the context window
  cheaply).
- **Verification loop overhead.** Wall-clock cost of cloning + replaying
  the sandbox per candidate; this is the practical cost of trusting
  execution over self-reported LLM confidence, and a systems paper
  should be upfront about it rather than only showing the runtime wins.

## Target venues and what each wants

- **VLDB / SIGMOD / ICDE (full research track):** wants the full
  pipeline validated on standard benchmarks (TPC-H/TPC-DS/JOB) at
  realistic scale, a real baseline comparison against AutoAdmin-style
  what-if analysis (not just a reimplementation -- ideally the actual
  tool or a very faithful one), and an honest cost/overhead discussion.
  This is a 12+ page systems paper; expect at least one full review
  cycle of "why not just use a cost-based optimizer" pushback that the
  evaluation needs to preempt.
- **AIDB / DBML workshops (co-located with VLDB/SIGMOD):** the right
  place for early results -- a working system on TPC-H plus a couple of
  the harder evaluation metrics above (precision/recall against brute
  force, plan stability) is a completely reasonable workshop paper and
  will get you specialist feedback before a full submission.
- **ICSE / FSE industry track:** reframe around the software-engineering
  angle -- "verification-loop architecture for AI-assisted systems
  changes," of which query optimization is one instance -- and lean less
  on DB-internals depth, more on the SE contribution of closing an LLM
  suggestion loop with an executable ground truth. Different related
  work section (cite AI-assisted code review / AI-assisted refactoring
  verification work instead of AutoAdmin).

## Honest risks / reviewer objections to pre-empt

- "Why not just use a cost-based what-if optimizer, skip the LLM
  entirely?" -- Answer needs actual evidence the LLM finds candidates a
  pure cost-model search misses (e.g. because it can reason about
  semantically-equivalent rewrites, or synthesize a materialized view
  shape that isn't just "index this column") or that it searches the
  candidate space more efficiently. If the evaluation only shows the LLM
  reinventing what AutoAdmin already finds, that's a negative result
  worth reporting honestly, not hiding.
- "Verification-loop cost." Cloning + replaying a full workload per
  candidate is expensive; a real deployment story (how often would this
  run, on what schedule, against what workload sample) needs to be in
  the paper, not just the microbenchmark numbers.
- "LLM non-determinism / prompt sensitivity." Report variance across
  seeds/temperature, and consider an ablation on prompt design
  (`src/prompts/`) since reviewers familiar with LLM evals will ask.
