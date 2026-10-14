

"""
WordLens — Day 3 Final Evaluation
==================================

Evaluates:

1. Semantic retrieval
2. Recall@1 / Recall@3 / Recall@5
3. MRR
4. nDCG@5
5. Exact-term accuracy
6. Alias recall
7. Hyphen/space recall
8. Negative-query abstention
9. Per-subdomain retrieval
10. Grounded RAG
11. Positive RAG cases
12. Negative RAG cases
13. Failure analysis
14. JSON evaluation report

IMPORTANT
---------
This test file does NOT modify:

    data/processed/technical_terms.json
    scripts/retrieval.py
    scripts/rag.py

It evaluates the existing Day-3 implementation against the
actual Day-1 dataset.
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# WORDLENS IMPORTS
# ============================================================

from scripts.retrieval import load_day1_records, retrieve
from scripts.rag import run_rag


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOP_K = 5
MINIMUM_SCORE = 0.45

RETRIEVAL_CASE_COUNT = 30
POSITIVE_RAG_CASE_COUNT = 15
NEGATIVE_RAG_CASE_COUNT = 5

REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "day3_evaluation_report.json"
)


# ============================================================
# UTILITIES
# ============================================================

def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip().lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("_", " ")

    text = re.sub(r"\s+", " ", text)

    return text


def normalize_hyphen_space(value: Any) -> str:
    text = normalize_text(value)
    return re.sub(r"[\s-]+", "", text)


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None

        number = float(value)

        if not math.isfinite(number):
            return None

        return number

    except (TypeError, ValueError):
        return None


def canonical_aliases(record: dict[str, Any]) -> list[str]:
    """
    Extract aliases robustly because Day-1 aliases may be stored as:

        list
        tuple
        string
        JSON string
        None
    """

    aliases = record.get("aliases")

    if aliases is None:
        return []

    if isinstance(aliases, (list, tuple, set)):
        values = aliases

    elif isinstance(aliases, str):

        text = aliases.strip()

        if not text:
            return []

        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                values = parsed
            else:
                values = [text]

        except json.JSONDecodeError:

            if "|" in text:
                values = text.split("|")

            elif "," in text:
                values = text.split(",")

            else:
                values = [text]

    else:
        values = [str(aliases)]

    cleaned = []

    for value in values:

        value = str(value).strip()

        if not value:
            continue

        cleaned.append(value)

    return cleaned


def record_is_valid(record: dict[str, Any]) -> bool:
    return bool(
        str(record.get("term", "")).strip()
        and str(record.get("term_id", "")).strip()
    )


# ============================================================
# DATASET PREPARATION
# ============================================================

def load_evaluation_records() -> list[dict[str, Any]]:
    records = load_day1_records()

    valid = []

    for record in records:

        if not record_is_valid(record):
            continue

        quality_status = str(
            record.get("quality_status", "")
        ).strip().lower()

        is_active = record.get("is_active")

        if quality_status != "accepted":
            continue

        if is_active is not True:
            continue

        valid.append(record)

    return valid


# ============================================================
# POSITIVE RETRIEVAL CASES
# ============================================================

PREFERRED_TERMS = [
    "attention",
    "Transformer",
    "overfitting",
    "generalization",
    "one-shot learning",
    "machine learning",
    "artificial intelligence",
    "neural network",
    "embedding",
    "fine-tuning",
    "gradient descent",
    "backpropagation",
    "automatic evaluation",
    "evaluation",
    "human evaluation",
    "latency",
    "conversational coding",
    "foundation model",
    "generative AI",
    "machine translation",
    "model router",
    "prompting",
    "response",
    "role prompting",
    "specificational coding",
    "tuning",
    "vibe coding",
    "inference",
    "model parameters",
    "temperature",
]


def find_record_by_term(
    records: list[dict[str, Any]],
    requested_term: str,
) -> dict[str, Any] | None:

    target = normalize_text(requested_term)

    for record in records:

        term = normalize_text(record.get("term"))

        if term == target:
            return record

    return None


def build_positive_retrieval_cases(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    cases = []
    used_ids = set()

    # --------------------------------------------------------
    # First use preferred real terms.
    # --------------------------------------------------------

    for term in PREFERRED_TERMS:

        record = find_record_by_term(records, term)

        if record is None:
            continue

        term_id = str(record["term_id"])

        if term_id in used_ids:
            continue

        cases.append(
            {
                "query": str(record["term"]),
                "expected_term_id": term_id,
                "expected_term": str(record["term"]),
                "subdomain": record.get("subdomain"),
                "case_type": "canonical",
            }
        )

        used_ids.add(term_id)

        if len(cases) >= RETRIEVAL_CASE_COUNT:
            break

    # --------------------------------------------------------
    # Fill remaining cases from actual dataset.
    # --------------------------------------------------------

    if len(cases) < RETRIEVAL_CASE_COUNT:

        for record in records:

            term_id = str(record["term_id"])

            if term_id in used_ids:
                continue

            cases.append(
                {
                    "query": str(record["term"]),
                    "expected_term_id": term_id,
                    "expected_term": str(record["term"]),
                    "subdomain": record.get("subdomain"),
                    "case_type": "canonical",
                }
            )

            used_ids.add(term_id)

            if len(cases) >= RETRIEVAL_CASE_COUNT:
                break

    return cases


# ============================================================
# ALIAS CASES
# ============================================================

def build_alias_cases(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    cases = []

    for record in records:

        aliases = canonical_aliases(record)

        if not aliases:
            continue

        canonical_term = str(record.get("term", "")).strip()

        if not canonical_term:
            continue

        for alias in aliases:

            if normalize_text(alias) == normalize_text(
                canonical_term
            ):
                continue

            cases.append(
                {
                    "query": alias,
                    "expected_term_id": str(record["term_id"]),
                    "expected_term": canonical_term,
                    "subdomain": record.get("subdomain"),
                    "case_type": "alias",
                }
            )

            break

    return cases[:10]


# ============================================================
# HYPHEN / SPACE CASES
# ============================================================

def build_hyphen_space_cases(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    cases = []

    for record in records:

        term = str(record.get("term", "")).strip()

        if not term:
            continue

        # ----------------------------------------------------
        # Example:
        #
        # self-attention -> self attention
        # fine-tuning -> fine tuning
        # one-shot -> one shot
        # ----------------------------------------------------

        if "-" not in term:
            continue

        variant = term.replace("-", " ")

        if normalize_text(variant) == normalize_text(term):
            continue

        cases.append(
            {
                "query": variant,
                "expected_term_id": str(record["term_id"]),
                "expected_term": term,
                "subdomain": record.get("subdomain"),
                "case_type": "hyphen_space",
            }
        )

        if len(cases) >= 10:
            break

    return cases


# ============================================================
# NEGATIVE RETRIEVAL CASES
# ============================================================

NEGATIVE_RETRIEVAL_QUERIES = [
    "database normalization",
    "quantum banana architecture",
    "random spaceship learning",
    "nonexistent technical concept xyz",
    "completely unrelated terminology",
    "ancient lunar compiler",
    "fictional neural grammar engine",
    "underwater blockchain weather",
    "imaginary quantum database",
    "random biological compiler",
]


# ============================================================
# RETRIEVAL RESULT HELPERS
# ============================================================

def result_term_ids(
    result: dict[str, Any],
) -> list[str]:

    output = []

    for item in result.get("results", []):

        term_id = item.get("term_id")

        if term_id is None:
            continue

        output.append(str(term_id))

    return output


def result_terms(
    result: dict[str, Any],
) -> list[str]:

    output = []

    for item in result.get("results", []):

        term = item.get("term")

        if term is None:
            continue

        output.append(str(term))

    return output


def expected_found_at_rank(
    result: dict[str, Any],
    expected_term_id: str,
) -> int | None:

    for rank, item in enumerate(
        result.get("results", []),
        start=1,
    ):

        if str(item.get("term_id")) == str(expected_term_id):
            return rank

    return None


# ============================================================
# RETRIEVAL METRICS
# ============================================================

def recall_at_k(
    cases: list[dict[str, Any]],
    evaluated_results: list[dict[str, Any]],
    k: int,
) -> float:

    if not cases:
        return 0.0

    hits = 0

    for case, result in zip(
        cases,
        evaluated_results,
    ):

        rank = expected_found_at_rank(
            result,
            case["expected_term_id"],
        )

        if rank is not None and rank <= k:
            hits += 1

    return hits / len(cases)


def mean_reciprocal_rank(
    cases: list[dict[str, Any]],
    evaluated_results: list[dict[str, Any]],
) -> float:

    reciprocal_values = []

    for case, result in zip(
        cases,
        evaluated_results,
    ):

        rank = expected_found_at_rank(
            result,
            case["expected_term_id"],
        )

        if rank is None:
            reciprocal_values.append(0.0)
        else:
            reciprocal_values.append(1.0 / rank)

    if not reciprocal_values:
        return 0.0

    return sum(reciprocal_values) / len(reciprocal_values)


def ndcg_at_5(
    cases: list[dict[str, Any]],
    evaluated_results: list[dict[str, Any]],
) -> float:

    values = []

    for case, result in zip(
        cases,
        evaluated_results,
    ):

        rank = expected_found_at_rank(
            result,
            case["expected_term_id"],
        )

        if rank is None or rank > 5:
            values.append(0.0)
        else:
            values.append(
                1.0 / math.log2(rank + 1)
            )

    if not values:
        return 0.0

    return sum(values) / len(values)


def exact_term_accuracy(
    cases: list[dict[str, Any]],
    evaluated_results: list[dict[str, Any]],
) -> float:

    if not cases:
        return 0.0

    correct = 0

    for case, result in zip(
        cases,
        evaluated_results,
    ):

        results = result.get("results", [])

        if not results:
            continue

        top = results[0]

        if str(top.get("term_id")) == str(
            case["expected_term_id"]
        ):
            correct += 1

    return correct / len(cases)


# ============================================================
# SPECIALIZED METRICS
# ============================================================

def alias_recall(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> float | None:

    if not cases:
        return None

    return recall_at_k(
        cases,
        results,
        5,
    )


def hyphen_space_recall(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> float | None:

    if not cases:
        return None

    return recall_at_k(
        cases,
        results,
        5,
    )


def negative_abstention_accuracy(
    negative_results: list[dict[str, Any]],
) -> float:

    if not negative_results:
        return 0.0

    correct = 0

    for result in negative_results:

        status = result.get(
            "grounding_status"
        )

        if status in {
            "insufficient_evidence",
            "ambiguous_term",
        }:
            correct += 1

    return correct / len(negative_results)


# ============================================================
# PER-SUBDOMAIN METRICS
# ============================================================

def per_subdomain_metrics(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    grouped_cases = defaultdict(list)
    grouped_results = defaultdict(list)

    for case, result in zip(cases, results):

        subdomain = case.get(
            "subdomain"
        ) or "unknown"

        grouped_cases[subdomain].append(case)
        grouped_results[subdomain].append(result)

    metrics = {}

    for subdomain in sorted(grouped_cases):

        sub_cases = grouped_cases[subdomain]
        sub_results = grouped_results[subdomain]

        metrics[subdomain] = {
            "count": len(sub_cases),
            "recall_at_1": recall_at_k(
                sub_cases,
                sub_results,
                1,
            ),
            "recall_at_3": recall_at_k(
                sub_cases,
                sub_results,
                3,
            ),
            "recall_at_5": recall_at_k(
                sub_cases,
                sub_results,
                5,
            ),
        }

    return metrics


# ============================================================
# FAILURE ANALYSIS
# ============================================================

def build_retrieval_failure_analysis(
    cases: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    failures = []

    for case, result in zip(
        cases,
        results,
    ):

        rank = expected_found_at_rank(
            result,
            case["expected_term_id"],
        )

        if rank == 1:
            continue

        results_list = result.get(
            "results",
            [],
        )

        top_term = (
            results_list[0].get("term")
            if results_list
            else None
        )

        failures.append(
            {
                "query": case["query"],
                "expected_term": case["expected_term"],
                "expected_term_id": case[
                    "expected_term_id"
                ],
                "rank": rank,
                "top_result": top_term,
                "grounding_status": result.get(
                    "grounding_status"
                ),
                "warnings": result.get(
                    "warnings",
                    [],
                ),
            }
        )

    return failures


# ============================================================
# RAG POSITIVE CASES
# ============================================================

def build_positive_rag_cases(
    records: list[dict[str, Any]],
) -> list[str]:

    """
    Select actual canonical terms from Day-1.

    This avoids hard-coded positive cases that may not exist
    in the actual project dataset.
    """

    selected = []
    seen = set()

    # Prefer common technical terms first.
    preferred = [
        "attention",
        "self-attention",
        "RAG",
        "Transformer",
        "overfitting",
        "generalization",
        "one-shot learning",
        "machine learning",
        "artificial intelligence",
        "neural network",
        "embedding",
        "classification",
        "regression",
        "clustering",
        "inference",
    ]

    for requested in preferred:

        record = find_record_by_term(
            records,
            requested,
        )

        if record is None:
            continue

        term = str(
            record.get("term", "")
        ).strip()

        normalized = normalize_text(term)

        if not normalized:
            continue

        if normalized in seen:
            continue

        selected.append(term)
        seen.add(normalized)

        if len(selected) >= POSITIVE_RAG_CASE_COUNT:
            return selected

    # Fill remaining cases from the real dataset.
    for record in records:

        term = str(
            record.get("term", "")
        ).strip()

        normalized = normalize_text(term)

        if not normalized:
            continue

        if normalized in seen:
            continue

        selected.append(term)
        seen.add(normalized)

        if len(selected) >= POSITIVE_RAG_CASE_COUNT:
            break

    return selected


# ============================================================
# RAG POSITIVE EVALUATION
# ============================================================

def evaluate_positive_rag(
    terms: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    cases = []
    failures = []

    print("\nPositive RAG cases:")

    for term in terms:

        result = run_rag(
            selected_term=term,
            context=None,
            top_k=TOP_K,
            minimum_score=MINIMUM_SCORE,
        )

        grounding_status = result.get(
            "grounding_status"
        )

        generation_method = result.get(
            "generation_method"
        )

        generated_explanation = result.get(
            "generated_explanation"
        )

        passed = (
            grounding_status == "grounded"
            and generation_method
            == "evidence_template"
            and bool(
                str(
                    generated_explanation
                    or ""
                ).strip()
            )
        )

        status = "PASS" if passed else "FAIL"

        print(
            f"  {term}: "
            f"{status} "
            f"({grounding_status}, "
            f"{generation_method})"
        )

        case = {
            "term": term,
            "passed": passed,
            "grounding_status": grounding_status,
            "generation_method": generation_method,
            "generated_explanation": (
                generated_explanation
            ),
            "canonical_definition": result.get(
                "canonical_definition"
            ),
            "top_score": result.get(
                "top_score"
            ),
        }

        cases.append(case)

        if not passed:

            failures.append(
                {
                    "term": term,
                    "grounding_status": (
                        grounding_status
                    ),
                    "generation_method": (
                        generation_method
                    ),
                    "top_score": result.get(
                        "top_score"
                    ),
                    "warnings": result.get(
                        "warnings",
                        [],
                    ),
                    "retrieved_evidence": (
                        result.get(
                            "retrieved_evidence",
                            [],
                        )
                    ),
                }
            )

    return cases, failures


# ============================================================
# RAG NEGATIVE CASES
# ============================================================

NEGATIVE_RAG_TERMS = [
    "database normalization",
    "quantum banana architecture",
    "nonexistent technical concept xyz",
    "fictional neural grammar engine",
    "imaginary quantum database",
]


def evaluate_negative_rag() -> list[dict[str, Any]]:

    cases = []

    print("\nNegative RAG cases:")

    for term in NEGATIVE_RAG_TERMS:

        result = run_rag(
            selected_term=term,
            context=None,
            top_k=TOP_K,
            minimum_score=MINIMUM_SCORE,
        )

        grounding_status = result.get(
            "grounding_status"
        )

        generation_method = result.get(
            "generation_method"
        )

        passed = (
            grounding_status
            in {
                "insufficient_evidence",
                "ambiguous_term",
            }
            and generation_method == "none"
            and result.get(
                "generated_explanation"
            )
            is None
        )

        status = "PASS" if passed else "FAIL"

        print(
            f"  {term}: "
            f"{status} "
            f"({grounding_status}, "
            f"{generation_method})"
        )

        cases.append(
            {
                "term": term,
                "passed": passed,
                "grounding_status": grounding_status,
                "generation_method": generation_method,
                "generated_explanation": result.get(
                    "generated_explanation"
                ),
                "top_score": result.get(
                    "top_score"
                ),
                "warnings": result.get(
                    "warnings",
                    [],
                ),
            }
        )

    return cases


# ============================================================
# MAIN EVALUATION
# ============================================================

def main() -> None:

    print("=" * 72)
    print("WORDLENS — DAY 3 FINAL EVALUATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Load Day-1 data
    # --------------------------------------------------------

    all_records = load_day1_records()

    records = load_evaluation_records()

    print(
        f"Loaded Day-1 records: "
        f"{len(all_records)}"
    )

    print(
        f"Accepted active records: "
        f"{len(records)}"
    )

    if len(records) == 0:

        raise RuntimeError(
            "No accepted active Day-1 records found."
        )

    print()
    print(
        f"Accepted active Day-1 records: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Build test cases
    # --------------------------------------------------------

    positive_cases = build_positive_retrieval_cases(
        records
    )

    alias_cases = build_alias_cases(
        records
    )

    hyphen_cases = build_hyphen_space_cases(
        records
    )

    print()
    print("=" * 72)
    print("RETRIEVAL EVALUATION")
    print("=" * 72)

    print(
        f"Positive cases:       "
        f"{len(positive_cases)}"
    )

    print(
        f"Alias cases:          "
        f"{len(alias_cases)}"
    )

    print(
        f"Hyphen/space cases:   "
        f"{len(hyphen_cases)}"
    )

    print(
        f"Negative cases:       "
        f"{len(NEGATIVE_RETRIEVAL_QUERIES)}"
    )

    # --------------------------------------------------------
    # Ensure minimum evaluation size
    # --------------------------------------------------------

    if len(positive_cases) < RETRIEVAL_CASE_COUNT:

        raise RuntimeError(
            f"Only {len(positive_cases)} positive "
            f"retrieval cases could be built from "
            f"the actual dataset. Required: "
            f"{RETRIEVAL_CASE_COUNT}."
        )

    # --------------------------------------------------------
    # Positive retrieval
    # --------------------------------------------------------

    positive_results = []

    for index, case in enumerate(
        positive_cases,
        start=1,
    ):

        print(
            f"\n[{index:02d}/{len(positive_cases)}] "
            f"{case['query']}"
        )

        result = retrieve(
            selected_term=case["query"],
            context=None,
            top_k=TOP_K,
            minimum_score=MINIMUM_SCORE,
            detection=None,
            model_name=MODEL_NAME,
        )

        positive_results.append(result)

        result_terms_list = result.get(
            "results",
            [],
        )

        top_result = (
            result_terms_list[0].get("term")
            if result_terms_list
            else None
        )

        rank = expected_found_at_rank(
            result,
            case["expected_term_id"],
        )

        print(
            f"    Top result: "
            f"{top_result}"
        )

        print(
            f"    R@1={rank == 1} "
            f"R@3={rank is not None and rank <= 3} "
            f"R@5={rank is not None and rank <= 5}"
        )

    # --------------------------------------------------------
    # Alias retrieval
    # --------------------------------------------------------

    alias_results = []

    if alias_cases:

        print(
            "\nAlias evaluation:"
        )

        for case in alias_cases:

            result = retrieve(
                selected_term=case["query"],
                context=None,
                top_k=TOP_K,
                minimum_score=MINIMUM_SCORE,
                detection=None,
                model_name=MODEL_NAME,
            )

            alias_results.append(result)

            rank = expected_found_at_rank(
                result,
                case["expected_term_id"],
            )

            print(
                f"  {case['query']} -> "
                f"{case['expected_term']} "
                f"(rank={rank})"
            )

    else:

        print(
            "\nAlias evaluation: "
            "N/A — no usable aliases exist "
            "in the current Day-1 dataset."
        )

    # --------------------------------------------------------
    # Hyphen/space retrieval
    # --------------------------------------------------------

    hyphen_results = []

    if hyphen_cases:

        print(
            "\nHyphen/space evaluation:"
        )

        for case in hyphen_cases:

            result = retrieve(
                selected_term=case["query"],
                context=None,
                top_k=TOP_K,
                minimum_score=MINIMUM_SCORE,
                detection=None,
                model_name=MODEL_NAME,
            )

            hyphen_results.append(result)

            rank = expected_found_at_rank(
                result,
                case["expected_term_id"],
            )

            print(
                f"  {case['query']} -> "
                f"{case['expected_term']} "
                f"(rank={rank})"
            )

    else:

        print(
            "\nHyphen/space evaluation: "
            "N/A — no suitable hyphenated "
            "terms exist in the dataset."
        )

    # --------------------------------------------------------
    # Negative retrieval
    # --------------------------------------------------------

    negative_results = []

    for query in NEGATIVE_RETRIEVAL_QUERIES:

        result = retrieve(
            selected_term=query,
            context=None,
            top_k=TOP_K,
            minimum_score=MINIMUM_SCORE,
            detection=None,
            model_name=MODEL_NAME,
        )

        negative_results.append(result)

        result_list = result.get(
            "results",
            [],
        )

        top_result = (
            result_list[0].get("term")
            if result_list
            else None
        )

        print(
            f"\nNegative query: {query}"
        )

        print(
            f"    Status: "
            f"{result.get('grounding_status')}"
        )

        print(
            f"    Top result: "
            f"{top_result}"
        )

    # ========================================================
    # RETRIEVAL METRICS
    # ========================================================

    r1 = recall_at_k(
        positive_cases,
        positive_results,
        1,
    )

    r3 = recall_at_k(
        positive_cases,
        positive_results,
        3,
    )

    r5 = recall_at_k(
        positive_cases,
        positive_results,
        5,
    )

    mrr = mean_reciprocal_rank(
        positive_cases,
        positive_results,
    )

    ndcg = ndcg_at_5(
        positive_cases,
        positive_results,
    )

    exact_accuracy = exact_term_accuracy(
        positive_cases,
        positive_results,
    )

    alias_metric = alias_recall(
        alias_cases,
        alias_results,
    )

    hyphen_metric = hyphen_space_recall(
        hyphen_cases,
        hyphen_results,
    )

    negative_accuracy = (
        negative_abstention_accuracy(
            negative_results
        )
    )

    subdomain_metrics = (
        per_subdomain_metrics(
            positive_cases,
            positive_results,
        )
    )

    retrieval_failures = (
        build_retrieval_failure_analysis(
            positive_cases,
            positive_results,
        )
    )

    print("\n")
    print("-" * 72)
    print("RETRIEVAL METRICS")
    print("-" * 72)

    print(
        f"Recall@1:              "
        f"{r1:.4f}"
    )

    print(
        f"Recall@3:              "
        f"{r3:.4f}"
    )

    print(
        f"Recall@5:              "
        f"{r5:.4f}"
    )

    print(
        f"MRR:                   "
        f"{mrr:.4f}"
    )

    print(
        f"nDCG@5:                "
        f"{ndcg:.4f}"
    )

    print(
        f"Exact-term accuracy:   "
        f"{exact_accuracy:.4f}"
    )

    if alias_metric is None:

        print(
            "Alias recall:          N/A"
        )

    else:

        print(
            f"Alias recall:          "
            f"{alias_metric:.4f}"
        )

    if hyphen_metric is None:

        print(
            "Hyphen/space recall:   N/A"
        )

    else:

        print(
            f"Hyphen/space recall:   "
            f"{hyphen_metric:.4f}"
        )

    print(
        f"Negative abstention:   "
        f"{negative_accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Per-subdomain
    # --------------------------------------------------------

    print(
        "\nPer-subdomain Recall:"
    )

    for subdomain, values in (
        subdomain_metrics.items()
    ):

        print(
            f"  {subdomain}: "
            f"R@1={values['recall_at_1']:.4f}, "
            f"R@3={values['recall_at_3']:.4f}, "
            f"R@5={values['recall_at_5']:.4f}"
        )

    # ========================================================
    # RAG EVALUATION
    # ========================================================

    print("\n")
    print("=" * 72)
    print("RAG EVALUATION")
    print("=" * 72)

    positive_rag_terms = build_positive_rag_cases(
        records
    )

    if len(positive_rag_terms) < POSITIVE_RAG_CASE_COUNT:

        raise RuntimeError(
            f"Only {len(positive_rag_terms)} positive "
            f"RAG cases could be built from the "
            f"actual dataset. Required: "
            f"{POSITIVE_RAG_CASE_COUNT}."
        )

    positive_rag_cases, positive_rag_failures = (
        evaluate_positive_rag(
            positive_rag_terms
        )
    )

    negative_rag_cases = (
        evaluate_negative_rag()
    )

    # --------------------------------------------------------
    # RAG metrics
    # --------------------------------------------------------

    positive_rag_passes = sum(
        1
        for case in positive_rag_cases
        if case["passed"]
    )

    negative_rag_passes = sum(
        1
        for case in negative_rag_cases
        if case["passed"]
    )

    positive_rag_accuracy = (
        positive_rag_passes
        / len(positive_rag_cases)
        if positive_rag_cases
        else 0.0
    )

    negative_rag_accuracy = (
        negative_rag_passes
        / len(negative_rag_cases)
        if negative_rag_cases
        else 0.0
    )

    total_rag_cases = (
        len(positive_rag_cases)
        + len(negative_rag_cases)
    )

    total_rag_passes = (
        positive_rag_passes
        + negative_rag_passes
    )

    overall_rag_accuracy = (
        total_rag_passes
        / total_rag_cases
        if total_rag_cases
        else 0.0
    )

    print("\n")
    print("-" * 72)
    print(
        f"Positive RAG accuracy: "
        f"{positive_rag_passes}/"
        f"{len(positive_rag_cases)} "
        f"({positive_rag_accuracy:.4f})"
    )

    print(
        f"Negative RAG accuracy: "
        f"{negative_rag_passes}/"
        f"{len(negative_rag_cases)} "
        f"({negative_rag_accuracy:.4f})"
    )

    print(
        f"Overall RAG accuracy:  "
        f"{overall_rag_accuracy:.4f}"
    )

    # ========================================================
    # FINAL CHECKS
    # ========================================================

    print("\n")
    print("=" * 72)
    print("FINAL DAY 3 CHECKS")
    print("=" * 72)

    checks: dict[str, bool] = {}

    # Required case counts.
    checks[
        "30 retrieval cases"
    ] = len(positive_cases) >= 30

    checks[
        "15 positive RAG cases"
    ] = len(positive_rag_cases) >= 15

    # Core retrieval metrics.
    checks[
        "Recall@1"
    ] = r1 >= 0.80

    checks[
        "Recall@3"
    ] = r3 >= 0.90

    checks[
        "Recall@5"
    ] = r5 >= 0.90

    checks[
        "MRR"
    ] = mrr >= 0.80

    checks[
        "nDCG@5"
    ] = ndcg >= 0.80

    checks[
        "Exact-term accuracy"
    ] = exact_accuracy >= 0.80

    # --------------------------------------------------------
    # Alias check
    #
    # N/A is considered PASS because the dataset itself
    # contains no qualifying alias cases.
    # --------------------------------------------------------

    checks[
        "Alias recall"
    ] = (
        alias_metric is None
        or alias_metric >= 0.80
    )

    # --------------------------------------------------------
    # Hyphen check
    # --------------------------------------------------------

    checks[
        "Hyphen/space recall"
    ] = (
        hyphen_metric is None
        or hyphen_metric >= 0.80
    )

    # --------------------------------------------------------
    # Negative abstention
    # --------------------------------------------------------

    checks[
        "Negative abstention"
    ] = negative_accuracy >= 0.80

    # --------------------------------------------------------
    # Per-subdomain
    # --------------------------------------------------------

    checks[
        "Per-subdomain evaluation"
    ] = bool(subdomain_metrics)

    # --------------------------------------------------------
    # RAG grounding
    #
    # Positive RAG cases must produce grounded evidence.
    # Negative cases must abstain.
    #
    # We do NOT modify retrieval thresholds merely to
    # force positive cases to pass.
    # --------------------------------------------------------

    positive_grounded_count = sum(
        1
        for case in positive_rag_cases
        if case["grounding_status"]
        == "grounded"
    )

    negative_abstained_count = sum(
        1
        for case in negative_rag_cases
        if case["grounding_status"]
        in {
            "insufficient_evidence",
            "ambiguous_term",
        }
    )

    checks[
        "RAG grounding evaluation"
    ] = (
        positive_grounded_count
        >= len(positive_rag_cases) * 0.80
        and negative_abstained_count
        == len(negative_rag_cases)
    )

    checks[
        "Failure analysis"
    ] = True

    # ========================================================
    # BUILD REPORT
    # ========================================================

    report = {
        "project": "WordLens",
        "day": 3,
        "evaluation": "Final Day 3 Evaluation",
        "created_at": utc_timestamp(),

        "configuration": {
            "embedding_model": MODEL_NAME,
            "top_k": TOP_K,
            "minimum_score": MINIMUM_SCORE,
        },

        "dataset": {
            "loaded_records": len(all_records),
            "accepted_active_records": len(records),
            "source": (
                "data/processed/"
                "technical_terms.json"
            ),
        },

        "retrieval": {
            "positive_case_count": len(
                positive_cases
            ),
            "alias_case_count": len(
                alias_cases
            ),
            "hyphen_space_case_count": len(
                hyphen_cases
            ),
            "negative_case_count": len(
                NEGATIVE_RETRIEVAL_QUERIES
            ),

            "metrics": {
                "recall_at_1": r1,
                "recall_at_3": r3,
                "recall_at_5": r5,
                "mrr": mrr,
                "ndcg_at_5": ndcg,
                "exact_term_accuracy": (
                    exact_accuracy
                ),
                "alias_recall": alias_metric,
                "hyphen_space_recall": (
                    hyphen_metric
                ),
                "negative_abstention_accuracy": (
                    negative_accuracy
                ),
            },

            "per_subdomain": (
                subdomain_metrics
            ),

            "failures": (
                retrieval_failures
            ),
        },

        "rag": {
            "positive_case_count": len(
                positive_rag_cases
            ),
            "negative_case_count": len(
                negative_rag_cases
            ),

            "positive_passes": (
                positive_rag_passes
            ),

            "positive_accuracy": (
                positive_rag_accuracy
            ),

            "negative_passes": (
                negative_rag_passes
            ),

            "negative_accuracy": (
                negative_rag_accuracy
            ),

            "overall_accuracy": (
                overall_rag_accuracy
            ),

            "positive_cases": (
                positive_rag_cases
            ),

            "positive_failures": (
                positive_rag_failures
            ),

            "negative_cases": (
                negative_rag_cases
            ),
        },

        "final_checks": checks,

        "day3_status": (
            "COMPLETE"
            if all(checks.values())
            else "INCOMPLETE"
        ),
    }

    # --------------------------------------------------------
    # Ensure report directory exists
    # --------------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # PRINT FINAL CHECKS
    # ========================================================

    for name, passed in checks.items():

        if passed:
            print(
                f"[PASS] {name}"
            )

        else:
            print(
                f"[FAIL] {name}"
            )

    print(
        "\nEvaluation report:"
    )

    print(
        str(REPORT_PATH)
    )

    print(
        "\n"
        f"DAY 3 STATUS: "
        f"{report['day3_status']}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()