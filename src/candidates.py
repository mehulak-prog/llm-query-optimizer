"""
Typed representations of what the LLM proposes: indexes, materialized
views, and query rewrites. Kept deliberately narrow (DDL-generating,
not free text) so every candidate can be mechanically applied to the
sandbox and mechanically rolled back -- an LLM that returns prose
"you should probably index the orders table" is not verifiable and is
explicitly rejected by the parser below.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Literal
from enum import Enum


class CandidateType(str, Enum):
    INDEX = "index"
    MATERIALIZED_VIEW = "materialized_view"
    REWRITE = "rewrite"


@dataclass
class IndexCandidate:
    table: str
    columns: List[str]
    rationale: str
    unique: bool = False
    kind: Literal["index"] = "index"

    @property
    def name(self) -> str:
        # Strip any " ASC"/" DESC" direction suffix before building the
        # identifier name -- to_ddl() below still uses self.columns as-is
        # (direction is valid inline SQL there), this is purely to keep
        # the generated index NAME a valid identifier (no spaces).
        base_cols = [c.split()[0] for c in self.columns]
        cols = "_".join(base_cols)
        # respect identifier length limits
        return f"idx_llm_{self.table}_{cols}"[:63]

    def to_ddl(self) -> str:
        uniq = "UNIQUE " if self.unique else ""
        cols = ", ".join(self.columns)
        return f"CREATE {uniq}INDEX {self.name} ON {self.table} ({cols});"

    def to_drop_ddl(self) -> str:
        return f"DROP INDEX IF EXISTS {self.name};"


@dataclass
class ViewCandidate:
    name: str
    definition_sql: str
    rationale: str
    replaces_query_names: List[str] = field(default_factory=list)
    kind: Literal["materialized_view"] = "materialized_view"

    def to_ddl(self, materialized: bool) -> str:
        mat = "MATERIALIZED " if materialized else ""
        return f"CREATE {mat}VIEW {self.name} AS {self.definition_sql};"

    def to_drop_ddl(self, materialized: bool) -> str:
        mat = "MATERIALIZED " if materialized else ""
        return f"DROP {mat}VIEW IF EXISTS {self.name};"


@dataclass
class RewriteCandidate:
    target_query_name: str
    original_sql: str
    rewritten_sql: str
    rationale: str
    kind: Literal["rewrite"] = "rewrite"


@dataclass
class CandidateSet:
    indexes: List[IndexCandidate] = field(default_factory=list)
    views: List[ViewCandidate] = field(default_factory=list)
    rewrites: List[RewriteCandidate] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.indexes or self.views or self.rewrites)


class CandidateParseError(Exception):
    pass


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str, kind: str) -> None:
    if not _IDENTIFIER_RE.match(name):
        raise CandidateParseError(
            f"Unsafe/invalid {kind} identifier: {name!r}")


def _normalize_flat_candidate_list(items: list, raw_text: str) -> dict:
    """Converts a flat array of candidate objects (some models' preferred
    shape despite the prompt's schema) into the nested object shape the
    rest of parse_llm_response expects. Item "type" is used to route each
    entry; "reason" is accepted as an alias for "rationale" since that's
    the substitution seen in practice.
    """
    result = {"indexes": [], "materialized_views": [], "rewrites": []}

    for item in items:
        if not isinstance(item, dict):
            raise CandidateParseError(
                f"Flat candidate array contained a non-object entry: {item!r}\n"
                f"Raw response: {raw_text[:1000]}"
            )

        item_type = item.get("type", "index" if "columns" in item else None)
        rationale = item.get("rationale", item.get("reason", ""))

        if item_type == "index":
            result["indexes"].append(
                {
                    "table": item.get("table"),
                    "columns": item.get("columns", []),
                    "unique": item.get("unique", False),
                    "rationale": rationale,
                }
            )
        elif item_type in ("materialized_view", "view"):
            result["materialized_views"].append(
                {
                    "name": item.get("name"),
                    "definition_sql": item.get("definition_sql"),
                    "replaces_query_names": item.get("replaces_query_names", []),
                    "rationale": rationale,
                }
            )
        elif item_type == "rewrite":
            result["rewrites"].append(
                {
                    "target_query_name": item.get("target_query_name"),
                    "original_sql": item.get("original_sql"),
                    "rewritten_sql": item.get("rewritten_sql"),
                    "rationale": rationale,
                }
            )
        else:
            raise CandidateParseError(
                f"Flat candidate array entry has unrecognized/missing 'type': "
                f"{item!r}\nRaw response: {raw_text[:1000]}"
            )

    return result


def parse_llm_response(raw_text: str, known_tables: List[str]) -> CandidateSet:
    """Parses (and validates) the JSON candidate set returned by the LLM.

    Strict on purpose: unknown tables, non-identifier column/table names,
    or malformed JSON all raise CandidateParseError rather than silently
    dropping candidates, because a silently-dropped bad candidate is a
    silently-wrong experiment result.
    """
    text = raw_text.strip()
    # Strip markdown code fences if the model wrapped its JSON.
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*|\s*```$", "",
                      text.strip(), flags=re.MULTILINE)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        # Some models (seen with Gemini 3.6 Flash) surround the actual JSON
        # payload with prose -- narrating their approach, giving inline
        # `code` examples of individual fields before the real answer, etc.
        # Two fallbacks, tried in order of how strong a signal they are:
        #
        # 1. An explicit ```json ... ``` fence ANYWHERE in the text (not
        #    just at the very start). This is a strong signal -- a fenced
        #    block is much more likely to be the intended full payload than
        #    incidental brackets in prose. Use the LAST such fence, since
        #    earlier ones are more likely to be small inline examples.
        #
        # 2. Falling back to json.JSONDecoder().raw_decode() from the
        #    first '{' or '[' in the text. This is weaker: prose can
        #    contain its OWN valid-looking JSON fragments before the real
        #    payload (e.g. "3. `lineitem` on `["l_returnflag", ...]`" as an
        #    inline example in a numbered list) which raw_decode will
        #    happily parse as a complete, syntactically valid but WRONG
        #    JSON value. This is why the fence search above is tried
        #    first and preferred whenever one exists.
        #
        # Neither fallback relaxes validation -- CandidateParseError is
        # still raised loudly if no valid JSON payload can be found at all.
        fence_matches = re.findall(
            r"```(?:json)?\s*(.*?)\s*```", raw_text, flags=re.DOTALL)
        data = None
        if fence_matches:
            try:
                data = json.loads(fence_matches[-1])
            except json.JSONDecodeError:
                data = None

        if data is None:
            start_candidates = [i for i in (
                text.find("{"), text.find("[")) if i != -1]
            if not start_candidates:
                raise CandidateParseError(
                    f"LLM response was not valid JSON: {e}\nRaw: {raw_text[:500]}"
                )
            start = min(start_candidates)
            try:
                data, _ = json.JSONDecoder().raw_decode(text, start)
            except json.JSONDecodeError as e2:
                raise CandidateParseError(
                    f"LLM response was not valid JSON, even after skipping "
                    f"{start} chars of leading prose to find the first '{{' or "
                    f"'[': {e2}\nRaw: {raw_text[:500]}"
                )

    if isinstance(data, list):
        # Some models (seen consistently with Groq's openai/gpt-oss-120b)
        # ignore the nested-object schema in the system prompt and return a
        # flat array of candidates instead, using different key names
        # ("type" instead of an implicit array membership, "reason"
        # instead of "rationale"). Rather than fight this per-model via
        # prompt tweaks alone, normalize known flat-array shapes into the
        # expected {"indexes": [...], "materialized_views": [...],
        # "rewrites": [...]} object here, then fall through to the same
        # strict validation as the well-formed case below.
        data = _normalize_flat_candidate_list(data, raw_text)

    if not isinstance(data, dict):
        raise CandidateParseError(
            f"LLM response was valid JSON but not the expected object shape "
            f"(got {type(data).__name__} instead of an object with "
            f"'indexes'/'materialized_views'/'rewrites' keys). This usually "
            f"means the model ignored the schema in the system prompt -- "
            f"some models (especially smaller/open ones) do this more than "
            f"others. Raw response: {raw_text[:1000]}"
        )

    result = CandidateSet()
    known_tables_set = set(known_tables)

    for idx in data.get("indexes", []):
        if not isinstance(idx, dict):
            raise CandidateParseError(
                f"Index candidate entry is not an object: {idx!r}\n"
                f"Raw response: {raw_text[:1000]}"
            )
        table = idx.get("table")
        if table is None:
            raise CandidateParseError(
                f"Index candidate missing required 'table' field: {idx!r}\n"
                f"Raw response: {raw_text[:1000]}"
            )
        if table not in known_tables_set:
            raise CandidateParseError(
                f"Index proposed on unknown table: {table!r}\n"
                f"Raw response: {raw_text[:1000]}"
            )
        _validate_identifier(table, "table")
        columns = idx.get("columns")
        if not columns:
            raise CandidateParseError(
                f"Index candidate missing required 'columns' field (or it "
                f"was empty): {idx!r}\nRaw response: {raw_text[:1000]}"
            )
        normalized_columns = []
        for c in columns:
            # Postgres allows "col DESC"/"col ASC" directly in an index's
            # column list (it affects scan order for ORDER BY queries).
            # Some models (seen with Groq's gpt-oss-120b) include this
            # when they've correctly reasoned about a query's ORDER BY --
            # validate the identifier part, keep the direction if valid.
            parts = c.split()
            base = parts[0]
            direction = parts[1].upper() if len(parts) > 1 else None
            if len(parts) > 2 or (direction and direction not in ("ASC", "DESC")):
                raise CandidateParseError(
                    f"Invalid column entry: {c!r} (expected a plain column "
                    f"name, optionally followed by ASC or DESC)"
                )
            _validate_identifier(base, "column")
            normalized_columns.append(
                f"{base} {direction}" if direction else base)

        result.indexes.append(
            IndexCandidate(
                table=table,
                columns=normalized_columns,
                rationale=idx.get("rationale", ""),
                unique=idx.get("unique", False),
            )
        )

    for v in data.get("materialized_views", []):
        if not isinstance(v, dict):
            raise CandidateParseError(
                f"View candidate entry is not an object: {v!r}\n"
                f"Raw response: {raw_text[:1000]}"
            )
        name = v.get("name")
        definition_sql = v.get("definition_sql")
        if name is None:
            raise CandidateParseError(
                f"View candidate missing required 'name' field: {v!r}\n"
                f"Raw response: {raw_text[:1000]}"
            )
        if not definition_sql:
            raise CandidateParseError(
                f"View candidate missing required 'definition_sql' field: {v!r}\n"
                f"Raw response: {raw_text[:1000]}"
            )
        _validate_identifier(name, "view name")
        result.views.append(
            ViewCandidate(
                name=name,
                definition_sql=definition_sql,
                rationale=v.get("rationale", ""),
                replaces_query_names=v.get("replaces_query_names", []),
            )
        )

    for r in data.get("rewrites", []):
        if not isinstance(r, dict):
            raise CandidateParseError(
                f"Rewrite candidate entry is not an object: {r!r}\n"
                f"Raw response: {raw_text[:1000]}"
            )
        target_query_name = r.get("target_query_name")
        original_sql = r.get("original_sql")
        rewritten_sql = r.get("rewritten_sql")
        missing = [
            field_name for field_name, val in (
                ("target_query_name", target_query_name),
                ("original_sql", original_sql),
                ("rewritten_sql", rewritten_sql),
            ) if not val
        ]
        if missing:
            raise CandidateParseError(
                f"Rewrite candidate missing required field(s) {missing}: {r!r}\n"
                f"Raw response: {raw_text[:1000]}"
            )
        result.rewrites.append(
            RewriteCandidate(
                target_query_name=target_query_name,
                original_sql=original_sql,
                rewritten_sql=rewritten_sql,
                rationale=r.get("rationale", ""),
            )
        )

    return result
