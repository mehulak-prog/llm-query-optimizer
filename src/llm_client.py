"""
The LLM planner. Two modes, switched by config.yaml's llm.mock_mode:

- mock_mode: true  -> a deterministic heuristic generator that mimics the
  *shape* of LLM output (JSON candidates) using simple statistics-driven
  rules. This exists so the rest of the pipeline (sandbox, verifier,
  reward, benchmark) can be developed and tested without API calls, and
  so `experiments/run_demo.py` works out of the box with zero setup.
  It is NOT a substitute for the real experiment -- it's a plumbing
  fixture. Do not report mock-mode numbers as your paper's results.

- mock_mode: false -> calls the real Anthropic API with the schema,
  stats, and workload rendered into the prompt, and parses the response
  with src/candidates.py's strict parser.

This is exactly the piece you said you'd handle the "replacing" for --
the mock generator is a placeholder, the real API path is wired up and
ready for your ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional

from .schema import DatabaseSchema
from .stats import TableStats
from .workload import Workload
from .candidates import CandidateSet, IndexCandidate, ViewCandidate, parse_llm_response

_PROMPT_DIR = Path(__file__).parent / "prompts"


def _build_user_prompt(
    schema: DatabaseSchema,
    stats: dict,
    workload: Workload,
    prior_results_text: Optional[str] = None,
    storage_budget_mb: Optional[float] = None,
) -> str:
    from .stats import stats_to_llm_text

    parts = [
        "SCHEMA:",
        schema.to_llm_text(),
        "STATISTICS:",
        stats_to_llm_text(stats),
        "WORKLOAD (representative queries with relative frequency weights):",
        workload.to_llm_text(),
    ]
    if storage_budget_mb is not None:
        parts.append(
            f"STORAGE BUDGET: You have a TOTAL budget of {storage_budget_mb} MB "
            f"across ALL indexes you recommend combined -- this is a shared cap "
            f"enforced identically for every system being compared, not a "
            f"per-index limit. Propose candidates that make good strategic use "
            f"of this budget: prefer fewer, well-chosen composite indexes that "
            f"cover multiple queries in the workload over many small "
            f"single-column indexes, if that gets more workload coverage per "
            f"MB spent. Leaving most of the budget unused is not inherently "
            f"safer -- it likely means leaving real workload speedup on the "
            f"table."
        )
    if prior_results_text:
        parts.append("PRIOR ROUND RESULTS:")
        parts.append(prior_results_text)
    return "\n".join(parts)


class LLMPlanner:
    def __init__(self, config: dict):
        self.config = config["llm"]
        self.storage_budget_mb = config.get(
            "candidates", {}).get("shared_storage_budget_mb")
        self.mock_mode = self.config.get("mock_mode", True)
        self.provider = self.config.get("provider", "anthropic")

        if self.mock_mode:
            return

        if self.provider == "anthropic":
            import anthropic  # deferred import; not needed in mock mode

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "llm.mock_mode is false and provider is 'anthropic' but "
                    "ANTHROPIC_API_KEY is not set. export ANTHROPIC_API_KEY=... "
                    "or set mock_mode: true in config.yaml"
                )
            self._client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "groq":
            # Groq's API is OpenAI-compatible, so we reuse the `openai`
            # package pointed at Groq's base_url instead of pulling in a
            # separate Groq SDK. Free-tier friendly -- used here to
            # iterate on prompts cheaply before spending on the real
            # Anthropic API.
            import openai

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "llm.mock_mode is false and provider is 'groq' but "
                    "GROQ_API_KEY is not set. export GROQ_API_KEY=... "
                    "or set mock_mode: true in config.yaml"
                )
            self._client = openai.OpenAI(
                api_key=api_key, base_url="https://api.groq.com/openai/v1"
            )

        elif self.provider == "gemini":
            # Same OpenAI-compatible pattern as Groq above, pointed at
            # Google's endpoint instead. Used purely to pipeline-stress-test
            # (crash-free-ness, parsing robustness) with a much more
            # generous free-tier rate limit than Groq's gpt-oss-120b (far
            # higher TPM/RPD) -- output quality is not the point here, only
            # whether the pipeline itself runs clean.
            import openai

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "llm.mock_mode is false and provider is 'gemini' but "
                    "GEMINI_API_KEY is not set. export GEMINI_API_KEY=... "
                    "or set mock_mode: true in config.yaml"
                )
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )

        else:
            raise RuntimeError(
                f"Unknown llm.provider: {self.provider!r}. "
                "Expected 'anthropic' or 'groq'."
            )

    def propose(
        self,
        schema: DatabaseSchema,
        stats: dict,
        workload: Workload,
        prior_results_text: Optional[str] = None,
    ) -> CandidateSet:
        if self.mock_mode:
            return self._mock_propose(schema, stats, workload)
        if self.provider == "groq":
            return self._groq_propose(schema, stats, workload, prior_results_text)
        if self.provider == "gemini":
            return self._gemini_propose(schema, stats, workload, prior_results_text)
        return self._real_propose(schema, stats, workload, prior_results_text)

    # ---- real API path (Anthropic) --------------------------------------
    def _real_propose(self, schema, stats, workload, prior_results_text) -> CandidateSet:
        system_file = (
            "refinement_system.txt" if prior_results_text else "planner_system.txt"
        )
        system_prompt = (_PROMPT_DIR / system_file).read_text()
        user_prompt = _build_user_prompt(
            schema, stats, workload, prior_results_text, self.storage_budget_mb)

        response = self._client.messages.create(
            model=self.config["model"],
            max_tokens=self.config.get("max_tokens", 4000),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            extra_body={"temperature": self.config.get("temperature", 0.4)},
        )
        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        return parse_llm_response(raw_text, known_tables=list(schema.tables.keys()))

    # ---- free-tier prototyping path (Groq, OpenAI-compatible) -----------
    def _groq_propose(self, schema, stats, workload, prior_results_text) -> CandidateSet:
        """Same prompts as the real Anthropic path -- system/user prompt
        text is provider-agnostic on purpose, so switching providers is
        just a config change, not a prompt rewrite. Response parsing
        differs (OpenAI-style choices[0].message.content vs Anthropic's
        content blocks), which is the only real difference here.
        """
        system_file = (
            "refinement_system.txt" if prior_results_text else "planner_system.txt"
        )
        system_prompt = (_PROMPT_DIR / system_file).read_text()
        user_prompt = _build_user_prompt(
            schema, stats, workload, prior_results_text, self.storage_budget_mb)

        response = self._client.chat.completions.create(
            model=self.config.get("groq_model", "openai/gpt-oss-120b"),
            # gpt-oss-120b spends tokens on internal chain-of-thought before
            # writing its actual answer; if that reasoning eats the whole
            # max_tokens budget, the visible response comes back empty
            # (a known, documented behavior of this model on Groq -- not a
            # bug in our code). reasoning_effort="low" reduces how much it
            # reasons before answering, and a higher max_tokens leaves more
            # headroom regardless -- both together make empty responses
            # much less likely.
            max_tokens=self.config.get("max_tokens", 6000),
            reasoning_effort=self.config.get("groq_reasoning_effort", "low"),
            temperature=self.config.get("temperature", 0.4),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = response.choices[0].message.content
        return parse_llm_response(raw_text, known_tables=list(schema.tables.keys()))

    # ---- pipeline stress-testing path (Gemini, OpenAI-compatible) -------
    def _gemini_propose(self, schema, stats, workload, prior_results_text) -> CandidateSet:
        """Same prompts/pattern as _groq_propose -- used to check the
        pipeline itself is crash-free (parsing, storage budget, refinement
        rounds) against a much more generous free-tier rate limit than
        Groq's gpt-oss-120b. Output quality is explicitly not the point of
        this path; that's what the real Anthropic run is for.
        """
        system_file = (
            "refinement_system.txt" if prior_results_text else "planner_system.txt"
        )
        system_prompt = (_PROMPT_DIR / system_file).read_text()
        user_prompt = _build_user_prompt(
            schema, stats, workload, prior_results_text, self.storage_budget_mb)

        response = self._client.chat.completions.create(
            model=self.config.get("gemini_model", "gemini-2.5-flash"),
            max_tokens=self.config.get(
                "gemini_max_tokens", self.config.get("max_tokens", 16000)),
            temperature=self.config.get("temperature", 0.4),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = response.choices[0].message.content
        return parse_llm_response(raw_text, known_tables=list(schema.tables.keys()))

    # ---- mock path (heuristic placeholder) ------------------------------
    def _mock_propose(self, schema: DatabaseSchema, stats: dict, workload: Workload) -> CandidateSet:
        """A simple, explainable heuristic: for each query, look at columns
        used in WHERE/JOIN/ORDER BY (approximated via naive keyword search
        since this is a plumbing fixture, not the real system), and
        propose an index if the column has high estimated selectivity and
        no existing index already covers it.

        This deliberately mirrors classic AutoAdmin-style "what-if" column
        selection, MINUS the cost-based what-if analysis -- that gap is
        exactly what the real LLM path + verification loop is meant to
        close, and why mock-mode numbers should never appear in the paper.
        """
        import re

        # (table, col) -> mention count
        candidate_columns: dict[tuple[str, str], int] = {}

        for q in workload.queries:
            sql_lower = q.sql.lower()
            for table_name, table in schema.tables.items():
                if table_name.lower() not in sql_lower:
                    continue
                for col in table.columns:
                    pattern = rf"\b{re.escape(col.name.lower())}\b\s*(=|>|<|>=|<=|in|between)"
                    join_pattern = rf"\bon\b[^;]*\b{re.escape(col.name.lower())}\b"
                    order_pattern = rf"order by[^;]*\b{re.escape(col.name.lower())}\b"
                    if (
                        re.search(pattern, sql_lower)
                        or re.search(join_pattern, sql_lower)
                        or re.search(order_pattern, sql_lower)
                    ):
                        key = (table_name, col.name)
                        candidate_columns[key] = candidate_columns.get(
                            key, 0) + int(q.weight)

        existing_indexed_cols = set()
        for table_name, table in schema.tables.items():
            for idx in table.existing_indexes:
                for c in idx.columns:
                    existing_indexed_cols.add((table_name, c))
            for col in table.columns:
                if col.is_primary_key:
                    existing_indexed_cols.add((table_name, col.name))

        scored = []
        for (table_name, col_name), mentions in candidate_columns.items():
            if (table_name, col_name) in existing_indexed_cols:
                continue
            t_stats = stats.get(table_name)
            if not t_stats or col_name not in t_stats.columns:
                continue
            c_stats = t_stats.columns[col_name]
            row_count = max(t_stats.row_count, 1)
            selectivity = c_stats.distinct_count_estimate / row_count
            if selectivity < 0.01:
                continue  # low-cardinality columns rarely benefit from a B-tree index
            score = mentions * selectivity
            scored.append((score, table_name, col_name))

        scored.sort(reverse=True)
        max_indexes = self._max_indexes()
        candidates = CandidateSet()
        for score, table_name, col_name in scored[:max_indexes]:
            candidates.indexes.append(
                IndexCandidate(
                    table=table_name,
                    columns=[col_name],
                    rationale=(
                        f"[mock] column appears in filter/join/order-by across the "
                        f"workload with weighted mention count and estimated "
                        f"selectivity score={score:.3f}; no existing index covers it."
                    ),
                )
            )
        return candidates

    def _max_indexes(self) -> int:
        return 6  # kept simple for the mock fixture; real path is bounded via config.candidates
