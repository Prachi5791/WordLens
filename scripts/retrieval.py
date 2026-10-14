

"""
Semantic Retrieval + FAISS Vectorstore

Responsibilities
----------------
1. Load Day-1 technical terminology dataset.
2. Validate accepted/active records.
3. Construct canonical retrieval documents.
4. Generate normalized embeddings using BGE.
5. Build and persist FAISS IndexIDMap2(IndexFlatIP).
6. Persist metadata, records, embeddings, and build report.
7. Validate persisted vectorstore.
8. Retrieve technical terminology for a selected term/context.
9. Apply lexical matching and semantic acceptance rules.
10. Abstain when evidence is insufficient.

CLI
---
Build:
    python scripts/retrieval.py build

Validate:
    python scripts/retrieval.py validate

Retrieve:
    python scripts/retrieval.py retrieve --term "attention"

Retrieve with context:
    python scripts/retrieval.py retrieve \
        --term "attention" \
        --context "The transformer uses attention to focus on relevant tokens."

Options:
    --batch-size
    --top-k
    --minimum-score
    --no-semantic
"""

# ---------------------------------------------------------------------
# Native-library safety for Apple Silicon / CPU execution
# ---------------------------------------------------------------------

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from functools import lru_cache

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "technical_terms.json"

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"
RECORDS_PATH = VECTORSTORE_DIR / "records.json"
EMBEDDINGS_PATH = VECTORSTORE_DIR / "embeddings.npy"
BUILD_REPORT_PATH = VECTORSTORE_DIR / "build_report.json"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_TOP_K = 5

# This is the basic FAISS similarity threshold.
DEFAULT_MINIMUM_SCORE = 0.45

# Strong semantic evidence.
SEMANTIC_STRONG_THRESHOLD = 0.78

# Moderate semantic evidence can be accepted when the top result is
# clearly separated from the next result.
SEMANTIC_MODERATE_THRESHOLD = 0.68

# Minimum separation between the first and second semantic candidates.
SEMANTIC_MARGIN_THRESHOLD = 0.05

# Lexical evidence is useful when the selected query contains the
# canonical term or normalized variant.
LEXICAL_EXACT_BOOST = 0.08
LEXICAL_VARIANT_BOOST = 0.05
LEXICAL_PARTIAL_BOOST = 0.04

# Do not allow lexical evidence alone to override very weak semantic
# evidence.
LEXICAL_MIN_SEMANTIC_SCORE = 0.45


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Return SHA-256 hash of a file."""
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def knowledge_signature(records: Sequence[Dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{r.get('term_id','')}|{r.get('normalized_term','')}|{r.get('source','')}"
        for r in sorted(records, key=lambda x: str(x.get('term_id','')))
    )
    return __import__('hashlib').sha256(payload.encode('utf-8')).hexdigest()


def normalize_text(value: Any) -> str:
    """
    Normalize text for matching.

    Examples:
        "Self-Attention" -> "self attention"
        "retrieval_augmented generation" -> "retrieval augmented generation"
    """
    if value is None:
        return ""

    text = str(value).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.replace("\\", " ")

    # Remove punctuation while preserving alphanumeric characters.
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_aliases(value: Any) -> List[str]:
    """Normalize aliases from a list/string/JSON-like value."""
    if value is None:
        return []

    if isinstance(value, list):
        values = value

    elif isinstance(value, tuple):
        values = list(value)

    elif isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        # Handle JSON list strings.
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)

                if isinstance(parsed, list):
                    values = parsed
                else:
                    values = [text]

            except Exception:
                values = [text]

        else:
            values = [text]

    else:
        values = [value]

    result = []

    for item in values:
        normalized = normalize_text(item)

        if normalized and normalized not in result:
            result.append(normalized)

    return result


def json_safe(value: Any) -> Any:
    """Convert numpy/path objects into JSON-safe Python objects."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    return value


def write_json(path: Path, data: Any) -> None:
    """Write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            json_safe(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_day1_records() -> List[Dict[str, Any]]:
    """Load accepted seed records and merge verified dynamic MySQL knowledge.

    Day-1 JSON remains the reproducible seed/source of truth. Dynamic records
    are additional verified knowledge and are included only when their
    normalized term is not already represented by the seed dataset.
    """
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Day-1 dataset not found:\n{DATASET_PATH}")

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    records = data.get("records", data.get("data", [])) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError("Unsupported Day-1 dataset format.")

    accepted: List[Dict[str, Any]] = []
    seen_terms: set[str] = set()

    for record in records:
        if str(record.get("quality_status", "")).strip().lower() != "accepted":
            continue
        if record.get("is_active", True) is not True:
            continue
        normalized = normalize_text(record.get("normalized_term", record.get("term", "")))
        if not normalized or normalized in seen_terms:
            continue
        accepted.append(record)
        seen_terms.add(normalized)

    dynamic_count = 0
    try:
        from backend.app.db.database import SessionLocal
        from backend.app.models.terminology import TechnicalTerm
        with SessionLocal() as session:
            dynamic_records = (
                session.query(TechnicalTerm)
                .filter(TechnicalTerm.is_active.is_(True), TechnicalTerm.quality_status == "accepted")
                .order_by(TechnicalTerm.id.asc())
                .all()
            )
            for row in dynamic_records:
                normalized = normalize_text(row.normalized_term or row.term)
                if not normalized or normalized in seen_terms:
                    continue
                accepted.append({
                    "term_id": row.term_id,
                    "term": row.term,
                    "normalized_term": row.normalized_term,
                    "aliases": row.aliases or [],
                    "domain": row.domain,
                    "subdomain": row.subdomain,
                    "definition": row.definition,
                    "simplified_definition": row.simplified_definition,
                    "source": row.source,
                    "source_url": row.source_url,
                    "confidence": row.confidence,
                    "quality_status": row.quality_status,
                    "is_active": row.is_active,
                })
                seen_terms.add(normalized)
                dynamic_count += 1
    except Exception as exc:
        print(f"Dynamic MySQL knowledge unavailable; using seed dataset only: {exc}")

    print(f"Loaded Day-1 records: {len(records)}")
    print(f"Accepted seed records: {len(accepted) - dynamic_count}")
    print(f"Accepted dynamic records: {dynamic_count}")
    print(f"Total retrieval records: {len(accepted)}")
    return accepted


# ---------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------

def validate_records(records: Sequence[Dict[str, Any]]) -> None:
    """Validate records before vectorstore construction."""

    if not records:
        raise ValueError("No accepted active terminology records found.")

    required_fields = [
        "term_id",
        "term",
        "normalized_term",
        "definition",
        "source",
        "source_url",
    ]

    term_ids = set()
    normalized_terms = set()

    for index, record in enumerate(records):
        for field in required_fields:
            if field not in record:
                raise ValueError(
                    f"Record {index} is missing required field: {field}"
                )

        term_id = str(record["term_id"]).strip()
        normalized_term = normalize_text(record["normalized_term"])
        term = str(record["term"]).strip()
        definition = str(record["definition"]).strip()

        if not term_id:
            raise ValueError(
                f"Record {index} has an empty term_id."
            )

        if not term:
            raise ValueError(
                f"Record {index} has an empty term."
            )

        if not normalized_term:
            raise ValueError(
                f"Record {index} has an empty normalized_term."
            )

        if not definition:
            raise ValueError(
                f"Record {index} has an empty definition."
            )

        if term_id in term_ids:
            raise ValueError(
                f"Duplicate term_id detected: {term_id}"
            )

        if normalized_term in normalized_terms:
            raise ValueError(
                f"Duplicate normalized_term detected: {normalized_term}"
            )

        term_ids.add(term_id)
        normalized_terms.add(normalized_term)


# ---------------------------------------------------------------------
# Retrieval document construction
# ---------------------------------------------------------------------

def build_retrieval_document(record: Dict[str, Any]) -> str:
    """
    Construct the canonical document used for embedding.

    Source URLs are intentionally excluded from the embedding text.
    """

    term = str(record.get("term", "")).strip()

    aliases = normalize_aliases(record.get("aliases"))

    domain = str(
        record.get("domain", "")
    ).strip()

    subdomain = str(
        record.get("subdomain", "")
    ).strip()

    definition = str(
        record.get("definition", "")
    ).strip()

    aliases_text = ", ".join(aliases) if aliases else "None"

    return (
        f"Term: {term}\n"
        f"Aliases: {aliases_text}\n"
        f"Domain: {domain}\n"
        f"Subdomain: {subdomain}\n"
        f"Definition: {definition}"
    )


def build_retrieval_documents(
    records: Sequence[Dict[str, Any]]
) -> List[str]:

    documents = [
        build_retrieval_document(record)
        for record in records
    ]

    print(
        f"Retrieval documents created: {len(documents)}"
    )

    return documents


# ---------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------

@lru_cache(maxsize=2)
def load_embedding_model(
    model_name: str = DEFAULT_MODEL,
) -> SentenceTransformer:
    """Load BGE once per backend process and reuse it for all requests."""
    print(f"Loading embedding model: {model_name}")
    print("Device: cpu")
    return SentenceTransformer(model_name, device="cpu")


def encode_documents(
    model: SentenceTransformer,
    documents: Sequence[str],
    batch_size: int = 8,
) -> np.ndarray:
    """Encode documents into normalized float32 vectors."""

    embeddings = model.encode(
        list(documents),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
        device="cpu",
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    return embeddings


def encode_query(
    model: SentenceTransformer,
    query: str,
) -> np.ndarray:
    """Encode one query into a normalized float32 vector."""

    # BGE models support query/document-specific instructions through
    # SentenceTransformer's encode_query when available.
    if hasattr(model, "encode_query"):
        embedding = model.encode_query(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device="cpu",
        )
    else:
        embedding = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device="cpu",
        )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    return embedding


# ---------------------------------------------------------------------
# Embedding validation
# ---------------------------------------------------------------------

def validate_embeddings(
    embeddings: np.ndarray,
) -> Dict[str, Any]:

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D. Got shape={embeddings.shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3,
    ):
        raise ValueError(
            "Embeddings are not normalized."
        )

    return {
        "count": int(embeddings.shape[0]),
        "dimension": int(embeddings.shape[1]),
        "dtype": str(embeddings.dtype),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
    }


# ---------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------

def build_faiss_index(
    embeddings: np.ndarray,
) -> faiss.IndexIDMap2:

    dimension = int(embeddings.shape[1])

    base_index = faiss.IndexFlatIP(dimension)

    index = faiss.IndexIDMap2(base_index)

    ids = np.arange(
        embeddings.shape[0],
        dtype=np.int64,
    )

    index.add_with_ids(
        embeddings,
        ids,
    )

    return index


# ---------------------------------------------------------------------
# Persisted metadata
# ---------------------------------------------------------------------

def build_records_payload(
    records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    payload = []

    for record in records:

        aliases = record.get("aliases")

        payload.append(
            {
                "term_id": record.get("term_id"),
                "term": record.get("term"),
                "aliases": aliases if aliases is not None else [],
                "domain": record.get("domain"),
                "subdomain": record.get("subdomain"),
                "definition": record.get("definition"),
                "source": record.get("source"),
                "source_url": record.get("source_url"),
                "confidence": record.get("confidence"),
            }
        )

    return payload


def build_metadata(
    records: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    model_name: str,
    dataset_hash: str,
) -> Dict[str, Any]:

    id_mapping = {}

    for faiss_id, record in enumerate(records):
        id_mapping[str(faiss_id)] = record["term_id"]

    return {
        "vectorstore_version": "day3_v1",
        "created_at": utc_now(),
        "embedding_model": model_name,
        "embedding_dimension": int(embeddings.shape[1]),
        "metric": "inner_product",
        "normalization": "L2",
        "index_type": "IndexIDMap2(IndexFlatIP)",
        "dataset_path": str(DATASET_PATH),
        "dataset_sha256": dataset_hash,
        "record_count": len(records),
        "knowledge_signature": knowledge_signature(records),
        "faiss_id_to_term_id": id_mapping,
    }


# ---------------------------------------------------------------------
# Build vectorstore
# ---------------------------------------------------------------------

def build_vectorstore(
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 8,
) -> None:

    print()
    print("=== WordLens Day 3: Vectorstore Build ===")
    print()

    records = load_day1_records()

    validate_records(records)

    documents = build_retrieval_documents(records)

    model = load_embedding_model(model_name)

    embeddings = encode_documents(
        model=model,
        documents=documents,
        batch_size=batch_size,
    )

    embedding_info = validate_embeddings(
        embeddings
    )

    print(
        f"Embedding dimension: "
        f"{embedding_info['dimension']}"
    )

    index = build_faiss_index(
        embeddings
    )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_hash = sha256_file(
        DATASET_PATH
    )

    metadata = build_metadata(
        records=records,
        embeddings=embeddings,
        model_name=model_name,
        dataset_hash=dataset_hash,
    )

    records_payload = build_records_payload(
        records
    )

    faiss.write_index(
        index,
        str(INDEX_PATH),
    )

    write_json(
        METADATA_PATH,
        metadata,
    )

    write_json(
        RECORDS_PATH,
        records_payload,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    build_report = {
        "status": "success",
        "created_at": utc_now(),
        "dataset": {
            "path": str(DATASET_PATH),
            "sha256": dataset_hash,
            "records_loaded": len(records),
            "knowledge_signature": knowledge_signature(records),
            "includes_verified_dynamic_mysql_records": True,
        },
        "embedding": embedding_info,
        "faiss": {
            "index_type": "IndexIDMap2(IndexFlatIP)",
            "metric": "inner_product",
            "ntotal": int(index.ntotal),
        },
        "model": model_name,
        "build_config": {
            "batch_size": batch_size,
            "device": "cpu",
            "normalized_embeddings": True,
        },
        "artifacts": {
            "index": str(INDEX_PATH),
            "metadata": str(METADATA_PATH),
            "records": str(RECORDS_PATH),
            "embeddings": str(EMBEDDINGS_PATH),
            "report": str(BUILD_REPORT_PATH),
        },
    }

    write_json(
        BUILD_REPORT_PATH,
        build_report,
    )

    print()
    print("Vectorstore built successfully.")
    print(f"Index:       {INDEX_PATH}")
    print(f"Metadata:    {METADATA_PATH}")
    print(f"Records:     {RECORDS_PATH}")
    print(f"Embeddings:  {EMBEDDINGS_PATH}")
    print(f"Report:      {BUILD_REPORT_PATH}")


# ---------------------------------------------------------------------
# Load persisted vectorstore
# ---------------------------------------------------------------------

def load_vectorstore():
    """Load FAISS index and persisted metadata/records."""

    required = [
        INDEX_PATH,
        METADATA_PATH,
        RECORDS_PATH,
        EMBEDDINGS_PATH,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Vectorstore artifacts missing:\n"
            + "\n".join(missing)
        )

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    with RECORDS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    return (
        index,
        metadata,
        records,
        embeddings,
    )


# ---------------------------------------------------------------------
# Vectorstore validation
# ---------------------------------------------------------------------

def validate_vectorstore() -> None:

    print()
    print("=== Vectorstore Validation ===")

    index, metadata, records, embeddings = (
        load_vectorstore()
    )

    if index.ntotal != len(records):
        raise ValueError(
            "FAISS count does not match records count."
        )

    if embeddings.shape[0] != len(records):
        raise ValueError(
            "Embedding count does not match records count."
        )

    metadata_dimension = int(
        metadata["embedding_dimension"]
    )

    if embeddings.shape[1] != metadata_dimension:
        raise ValueError(
            "Embedding dimension mismatch."
        )

    if index.d != metadata_dimension:
        raise ValueError(
            "FAISS dimension mismatch."
        )

    embedding_info = validate_embeddings(
        embeddings
    )

    expected_hash = sha256_file(
        DATASET_PATH
    )

    stored_hash = metadata.get(
        "dataset_sha256"
    )

    if stored_hash != expected_hash:
        raise ValueError(
            "Dataset SHA-256 does not match "
            "the persisted vectorstore."
        )

    current_records = load_day1_records()
    stored_signature = metadata.get("knowledge_signature")
    current_signature = knowledge_signature(current_records)
    if stored_signature and stored_signature != current_signature:
        raise ValueError(
            "Knowledge signature does not match the persisted vectorstore. "
            "Rebuild the FAISS index after terminology changes."
        )

    mapping = metadata.get(
        "faiss_id_to_term_id",
        {},
    )

    if len(mapping) != len(records):
        raise ValueError(
            "FAISS ID mapping count does not match records."
        )

    print("Status: PASSED")
    print(f"Vectors: {index.ntotal}")
    print(f"Dimension: {index.d}")
    print(f"Records: {len(records)}")
    print(f"Dataset SHA-256: {expected_hash}")


# ---------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------

def build_query(
    selected_term: Optional[str],
    context: Optional[str],
) -> str:

    selected_term = (
        selected_term.strip()
        if selected_term
        else ""
    )

    context = (
        context.strip()
        if context
        else ""
    )

    if not selected_term and not context:
        raise ValueError(
            "At least selected_term or context is required."
        )

    return (
        f"Selected term: "
        f"{selected_term or 'Not provided'}\n"
        f"Research context: "
        f"{context or 'Not provided'}"
    )


# ---------------------------------------------------------------------
# Lexical matching
# ---------------------------------------------------------------------

def lexical_boost(
    selected_term: Optional[str],
    record: Dict[str, Any],
) -> Tuple[float, bool, str]:
    """
    Determine lexical evidence between query and record.

    Returns:
        boost
        matched_selected_term
        match_type

    Match types:
        exact_canonical
        exact_alias
        normalized_variant
        canonical_phrase
        alias_phrase
        none
    """

    if not selected_term:
        return 0.0, False, "none"

    query = normalize_text(
        selected_term
    )

    if not query:
        return 0.0, False, "none"

    canonical = normalize_text(
        record.get("term", "")
    )

    aliases = normalize_aliases(
        record.get("aliases")
    )

    # -------------------------------------------------------------
    # Exact canonical match
    # -------------------------------------------------------------

    if query == canonical:
        return (
            LEXICAL_EXACT_BOOST,
            True,
            "exact_canonical",
        )

    # -------------------------------------------------------------
    # Exact alias match
    # -------------------------------------------------------------

    if query in aliases:
        return (
            LEXICAL_EXACT_BOOST,
            True,
            "exact_alias",
        )

    # -------------------------------------------------------------
    # Hyphen / space / punctuation normalized variant
    # -------------------------------------------------------------

    raw_term = str(
        record.get("term", "")
    ).strip().lower()

    raw_query = str(
        selected_term
    ).strip().lower()

    raw_term_variant = normalize_text(
        raw_term
    )

    raw_query_variant = normalize_text(
        raw_query
    )

    if (
        raw_query_variant == raw_term_variant
        and raw_query != raw_term
    ):
        return (
            LEXICAL_VARIANT_BOOST,
            True,
            "normalized_variant",
        )

    # -------------------------------------------------------------
    # Canonical phrase containment
    #
    # Example:
    #   query = "attention mechanism"
    #   canonical = "attention"
    #
    # The selected phrase clearly contains the canonical term.
    # -------------------------------------------------------------

    query_tokens = query.split()

    if (
        len(query_tokens) >= 1
        and canonical
        and len(canonical.split()) >= 1
    ):
        if canonical in query:
            return (
                LEXICAL_PARTIAL_BOOST,
                True,
                "canonical_phrase",
            )

    # -------------------------------------------------------------
    # Alias phrase containment
    # -------------------------------------------------------------

    for alias in aliases:
        if alias and alias in query:
            return (
                LEXICAL_PARTIAL_BOOST,
                True,
                "alias_phrase",
            )

    return 0.0, False, "none"


# ---------------------------------------------------------------------
# Retrieval acceptance / abstention
# ---------------------------------------------------------------------

def determine_grounding_status(
    top_result: Optional[Dict[str, Any]],
    second_result: Optional[Dict[str, Any]],
    minimum_score: float,
) -> Tuple[str, str]:
    """
    Decide whether retrieval evidence is sufficient.

    Important:
    A high semantic score alone is NOT sufficient.

    The decision combines:
        - semantic score
        - lexical evidence
        - semantic margin

    This prevents unrelated queries from being treated as grounded.
    """

    if top_result is None:
        return (
            "insufficient_evidence",
            "No retrieval result was returned.",
        )

    semantic_score = float(
        top_result.get("semantic_score", 0.0)
    )

    lexical = float(
        top_result.get("lexical_boost", 0.0)
    )

    matched = bool(
        top_result.get("matched_selected_term", False)
    )

    second_score = (
        float(
            second_result.get(
                "semantic_score",
                0.0,
            )
        )
        if second_result
        else 0.0
    )

    margin = semantic_score - second_score

    # -------------------------------------------------------------
    # Basic score rejection
    # -------------------------------------------------------------

    if semantic_score < minimum_score:
        return (
            "insufficient_evidence",
            (
                f"Top semantic score "
                f"{semantic_score:.4f} is below "
                f"minimum score {minimum_score:.4f}."
            ),
        )

    # -------------------------------------------------------------
    # Exact/variant lexical match
    #
    # This is strong evidence, provided the semantic score is not
    # extremely weak.
    # -------------------------------------------------------------

    if matched and lexical > 0:
        if semantic_score >= LEXICAL_MIN_SEMANTIC_SCORE:
            return (
                "grounded",
                (
                    "Canonical, alias, or normalized lexical "
                    "evidence supports the top result."
                ),
            )

    # -------------------------------------------------------------
    # Strong semantic match
    # -------------------------------------------------------------

    if semantic_score >= SEMANTIC_STRONG_THRESHOLD:
        return (
            "grounded",
            (
                "Strong semantic similarity supports "
                "the top retrieval result."
            ),
        )

    # -------------------------------------------------------------
    # Moderate semantic match
    #
    # Requires a meaningful gap from the next result.
    # -------------------------------------------------------------

    if (
        semantic_score >= SEMANTIC_MODERATE_THRESHOLD
        and margin >= SEMANTIC_MARGIN_THRESHOLD
    ):
        return (
            "grounded",
            (
                "Moderate semantic similarity with sufficient "
                "separation from competing results."
            ),
        )

    # -------------------------------------------------------------
    # Otherwise abstain
    # -------------------------------------------------------------

    return (
        "insufficient_evidence",
        (
            f"Semantic evidence is insufficient: "
            f"score={semantic_score:.4f}, "
            f"margin={margin:.4f}, "
            f"lexical_boost={lexical:.4f}."
        ),
    )


# ---------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------

def retrieve(
    selected_term: Optional[str],
    context: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    minimum_score: Optional[float] = None,
    detection: Optional[Dict[str, Any]] = None,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:

    if minimum_score is None:
        minimum_score = DEFAULT_MINIMUM_SCORE

    if top_k < 1:
        raise ValueError(
            "top_k must be >= 1."
        )

    query = build_query(
        selected_term,
        context,
    )

    index, metadata, records, _ = (
        load_vectorstore()
    )

    model = load_embedding_model(
        model_name
    )

    query_embedding = encode_query(
        model,
        query,
    )

    if query_embedding.shape[1] != index.d:
        raise ValueError(
            "Query embedding dimension does not "
            "match FAISS index dimension."
        )

    search_k = min(
        max(top_k, 10),
        index.ntotal,
    )

    distances, ids = index.search(
        query_embedding,
        search_k,
    )

    candidates = []

    for semantic_score, faiss_id in zip(
        distances[0],
        ids[0],
    ):

        if faiss_id < 0:
            continue

        faiss_id = int(faiss_id)

        if faiss_id >= len(records):
            continue

        record = records[faiss_id]

        boost, matched, match_type = (
            lexical_boost(
                selected_term,
                record,
            )
        )

        final_score = (
            float(semantic_score)
            + float(boost)
        )

        candidates.append(
            {
                "faiss_id": faiss_id,
                "term_id": record.get(
                    "term_id"
                ),
                "term": record.get(
                    "term"
                ),
                "aliases": record.get(
                    "aliases",
                    [],
                ),
                "domain": record.get(
                    "domain"
                ),
                "subdomain": record.get(
                    "subdomain"
                ),
                "definition": record.get(
                    "definition"
                ),
                "source": record.get(
                    "source"
                ),
                "source_url": record.get(
                    "source_url"
                ),
                "semantic_score": round(
                    float(semantic_score),
                    6,
                ),
                "lexical_boost": round(
                    float(boost),
                    6,
                ),
                "final_score": round(
                    final_score,
                    6,
                ),
                "matched_selected_term": bool(
                    matched
                ),
                "lexical_match_type": match_type,
            }
        )

    # -------------------------------------------------------------
    # Rank using final score.
    # -------------------------------------------------------------

    candidates.sort(
        key=lambda item: (
            item["final_score"],
            item["semantic_score"],
        ),
        reverse=True,
    )

    candidates = candidates[:top_k]

    for rank, result in enumerate(
        candidates,
        start=1,
    ):
        result["rank"] = rank

    # -------------------------------------------------------------
    # Grounding decision.
    #
    # Use semantic scores for margin rather than final scores so that
    # lexical boosting cannot artificially create a large margin.
    # -------------------------------------------------------------

    top_result = (
        candidates[0]
        if candidates
        else None
    )

    second_result = (
        candidates[1]
        if len(candidates) > 1
        else None
    )

    grounding_status, grounding_reason = (
        determine_grounding_status(
            top_result=top_result,
            second_result=second_result,
            minimum_score=minimum_score,
        )
    )

    # -------------------------------------------------------------
    # Keep results transparent.
    #
    # RAG must respect grounding_status and must not generate from
    # an insufficient-evidence result.
    # -------------------------------------------------------------

    result = {
        "selected_term": selected_term,
        "context": context,
        "retrieval_query": query,
        "retrieval_status": "success",
        "grounding_status": grounding_status,
        "grounding_reason": grounding_reason,
        "top_k": top_k,
        "minimum_score": minimum_score,
        "detection": detection,
        "results": candidates,
    }

    return result


# ---------------------------------------------------------------------
# Deterministic persisted-query check
# ---------------------------------------------------------------------

def deterministic_query_check(
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Run the same query twice after loading the persisted vectorstore.

    This verifies that the persisted index can be reloaded and queried
    deterministically.
    """

    first = retrieve(
        selected_term="attention",
        context=None,
        top_k=5,
        model_name=model_name,
    )

    second = retrieve(
        selected_term="attention",
        context=None,
        top_k=5,
        model_name=model_name,
    )

    first_ids = [
        item["term_id"]
        for item in first["results"]
    ]

    second_ids = [
        item["term_id"]
        for item in second["results"]
    ]

    identical = (
        first_ids == second_ids
    )

    return {
        "identical_top_k": identical,
        "first_term_ids": first_ids,
        "second_term_ids": second_ids,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def create_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "WordLens Day 3 semantic retrieval "
            "and FAISS vectorstore."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -------------------------------------------------------------
    # BUILD
    # -------------------------------------------------------------

    build_parser = subparsers.add_parser(
        "build",
        help="Build the persisted FAISS vectorstore.",
    )

    build_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer embedding model.",
    )

    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Embedding batch size.",
    )

    # -------------------------------------------------------------
    # VALIDATE
    # -------------------------------------------------------------

    subparsers.add_parser(
        "validate",
        help="Validate persisted vectorstore.",
    )

    # -------------------------------------------------------------
    # RETRIEVE
    # -------------------------------------------------------------

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Retrieve terminology for a query.",
    )

    retrieve_parser.add_argument(
        "--term",
        required=False,
        default=None,
        help="Selected technical term.",
    )

    retrieve_parser.add_argument(
        "--context",
        required=False,
        default=None,
        help="Optional research-paper context.",
    )

    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of results to return.",
    )

    retrieve_parser.add_argument(
        "--minimum-score",
        type=float,
        default=DEFAULT_MINIMUM_SCORE,
        help="Minimum FAISS semantic score.",
    )

    retrieve_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer embedding model.",
    )

    # Compatibility flag retained intentionally.
    retrieve_parser.add_argument(
        "--no-semantic",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser


def main() -> None:

    parser = create_parser()

    args = parser.parse_args()

    try:

        if args.command == "build":

            if args.batch_size < 1:
                raise ValueError(
                    "--batch-size must be >= 1."
                )

            build_vectorstore(
                model_name=args.model,
                batch_size=args.batch_size,
            )

        elif args.command == "validate":

            validate_vectorstore()

        elif args.command == "retrieve":

            if not args.term and not args.context:
                raise ValueError(
                    "Provide --term and/or --context."
                )

            output = retrieve(
                selected_term=args.term,
                context=args.context,
                top_k=args.top_k,
                minimum_score=args.minimum_score,
                model_name=args.model,
            )

            print(
                json.dumps(
                    json_safe(output),
                    indent=2,
                    ensure_ascii=False,
                )
            )

    except KeyboardInterrupt:

        print(
            "\nOperation cancelled."
        )

        sys.exit(130)

    except Exception as exc:

        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()