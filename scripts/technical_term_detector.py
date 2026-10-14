


from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from scripts.scientific_term_discovery import (
    classify_rejection,
    generate_candidates as discover_candidates,
    normalize_lookup,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT / "data" / "processed" / "technical_terms.json"
DEFAULT_MODEL_DIR = ROOT / "models" / "technical_term_detector"
DEFAULT_EMBEDDINGS_PATH = DEFAULT_MODEL_DIR / "vocabulary_embeddings.npz"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "model_metadata.json"
DEFAULT_EXTERNAL_CACHE = DEFAULT_MODEL_DIR / "external_lookup_cache.json"
DEFAULT_MODEL_NAME = "allenai/scibert_scivocab_uncased"

DEFAULT_CONFIG: dict[str, Any] = {
    "minimum_candidate_length": 2,
    "maximum_candidate_tokens": 5,
    "semantic_candidate_limit": 12,
    "semantic_score_threshold": 0.58,
    "exact_match_confidence": 0.98,
    "alias_match_confidence": 0.95,
    "normalized_variant_confidence": 0.92,
    "external_lookup_enabled": False,
}

REQUIRED_DAY1_FIELDS = {
    "term_id", "term", "normalized_term", "aliases", "domain", "subdomain",
    "definition", "simplified_definition", "source", "source_url",
    "confidence", "quality_status", "is_active",
}

METHOD_PRIORITY = {
    "exact_canonical": 5,
    "exact_alias": 4,
    "normalized_variant": 3,
    "external": 2,
    "semantic_candidate": 1,
}


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_pdf_text(value: str) -> str:
    if not value:
        return ""
    value = str(value).replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"(?<=\w)\n(?=\w)", " ", value)
    value = re.sub(r"\s*\n\s*", " ", value)
    return re.sub(r"[ \t]+", " ", value).strip()


def dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_day1_vocabulary(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Day 1 dataset not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("Day 1 dataset must be a non-empty JSON list.")
    missing = REQUIRED_DAY1_FIELDS - set(records[0].keys())
    if missing:
        raise ValueError(f"Day 1 dataset is missing required fields: {sorted(missing)}")
    active = [r for r in records if r.get("quality_status") == "accepted" and bool(r.get("is_active"))]
    if not active:
        raise ValueError("Day 1 dataset contains no accepted active vocabulary records.")
    return active


def _clean_alias(value: Any) -> str:
    return re.sub(r"\s*\([^)]*\)", "", str(value or "")).strip()


def _lookup_key(value: str) -> str:
    return normalize_lookup(value)


def _pattern(phrase: str) -> re.Pattern[str]:
    phrase = _clean_alias(phrase)
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[\s\-]+")
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class VocabularyIndex:
    records: tuple[dict[str, Any], ...]
    canonical: dict[str, int]
    aliases: dict[str, list[int]]
    normalized: dict[str, list[int]]

    @classmethod
    def build(cls, records: Sequence[dict[str, Any]]) -> "VocabularyIndex":
        canonical: dict[str, int] = {}
        aliases: dict[str, list[int]] = {}
        normalized: dict[str, list[int]] = {}
        for i, record in enumerate(records):
            canonical.setdefault(normalize_text(record.get("normalized_term") or record.get("term")), i)
            normalized.setdefault(_lookup_key(str(record.get("term", ""))), []).append(i)
            for alias in record.get("aliases") or []:
                key = _clean_alias(alias)
                if key:
                    aliases.setdefault(_lookup_key(key), []).append(i)
        return cls(tuple(records), canonical, aliases, normalized)

    def get(self, index: int) -> dict[str, Any]:
        return self.records[index]


@dataclass
class VocabularyEmbeddingStore:
    embeddings: Any
    records: Sequence[dict[str, Any]]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        import numpy as np
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.records = tuple(self.records)
        self.metadata = dict(self.metadata or {})
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != len(self.records):
            raise ValueError("Vocabulary embeddings must be 2D and match record count.")

    @property
    def embedding_dimension(self) -> int:
        return int(self.embeddings.shape[1])


@dataclass(frozen=True)
class TextUnit:
    text: str
    sentence_index: int
    page_number: int | None = None
    page_char_start: int = 0


@dataclass(frozen=True)
class Candidate:
    surface: str
    normalized: str
    sentence: str
    sentence_index: int
    char_start: int
    char_end: int
    page_number: int | None
    extraction_method: str
    score: float = 0.0


@dataclass
class Detection:
    detection_id: str
    term_id: str | None
    term: str
    matched_text: str
    normalized_matched_text: str
    aliases_matched: list[str]
    domain: str | None
    subdomain: str | None
    definition: str | None
    sentence: str
    sentence_index: int
    page_number: int | None
    char_start: int
    char_end: int
    extraction_method: str
    confidence: float
    confidence_reason: str
    semantic_similarity: float | None = None
    semantic_margin: float | None = None
    vocabulary_source: str | None = None
    vocabulary_source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    text = normalize_pdf_text(text)
    if not text:
        return []
    try:
        import spacy
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        return [(s.text.strip(), s.start_char, s.end_char) for s in doc.sents if s.text.strip()]
    except Exception:
        result = []
        start = 0
        for match in re.finditer(r"(?<=[.!?])\s+|\n+", text):
            segment = text[start:match.start()].strip()
            if segment:
                left = start + len(text[start:match.start()]) - len(text[start:match.start()].lstrip())
                result.append((segment, left, left + len(segment)))
            start = match.end()
        tail = text[start:].strip()
        if tail:
            result.append((tail, len(text) - len(tail), len(text)))
        return result


def build_text_units(text: str, page_number: int | None = None) -> list[TextUnit]:
    return [
        TextUnit(sentence, i, page_number, start)
        for i, (sentence, start, _) in enumerate(split_sentences(text))
    ]


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    try:
        import pymupdf
        document = pymupdf.open(path)
    except ImportError:
        import fitz
        document = fitz.open(path)
    try:
        if len(document) == 0:
            raise ValueError("PDF contains no pages.")
        pages = []
        for i in range(len(document)):
            text = normalize_pdf_text(document.load_page(i).get_text("text"))
            pages.append({"page_number": i + 1, "text": text})
        if not any(page["text"] for page in pages):
            raise ValueError("No extractable text was found in the PDF.")
        return pages
    finally:
        document.close()


def extract_pdf_units(pdf_path: str | Path) -> list[TextUnit]:
    units: list[TextUnit] = []
    global_index = 0
    for page in extract_pdf_pages(pdf_path):
        for sentence, start, _ in split_sentences(page["text"]):
            units.append(TextUnit(sentence, global_index, page["page_number"], start))
            global_index += 1
    return units


def _make_detection(
    record: dict[str, Any],
    matched_text: str,
    unit: TextUnit,
    start: int,
    end: int,
    method: str,
    confidence: float,
    aliases: list[str] | None = None,
    semantic_similarity: float | None = None,
) -> Detection:
    return Detection(
        detection_id=uuid.uuid4().hex[:16],
        term_id=str(record.get("term_id")) if record.get("term_id") is not None else None,
        term=str(record.get("term") or matched_text),
        matched_text=matched_text,
        normalized_matched_text=normalize_lookup(matched_text),
        aliases_matched=aliases or [],
        domain=record.get("domain"),
        subdomain=record.get("subdomain"),
        definition=record.get("definition"),
        sentence=unit.text,
        sentence_index=unit.sentence_index,
        page_number=unit.page_number,
        char_start=unit.page_char_start + start,
        char_end=unit.page_char_start + end,
        extraction_method=method,
        confidence=float(confidence),
        confidence_reason={
            "exact_canonical": "Exact canonical terminology match.",
            "exact_alias": "Exact alias terminology match.",
            "normalized_variant": "Normalized spelling/hyphenation variant match.",
        }.get(method, "Open-world linguistic candidate scored by SciBERT representation."),
        semantic_similarity=semantic_similarity,
        semantic_margin=None,
        vocabulary_source=record.get("source"),
        vocabulary_source_url=record.get("source_url"),
    )


def find_exact_matches(unit: TextUnit, vocabulary: VocabularyIndex) -> list[Detection]:
    matches: list[Detection] = []
    patterns: list[tuple[str, dict[str, Any], str, float, list[str]]] = []
    for record in vocabulary.records:
        term = str(record.get("term", "")).strip()
        if term:
            patterns.append((term, record, "exact_canonical", DEFAULT_CONFIG["exact_match_confidence"], []))
        for alias in record.get("aliases") or []:
            alias_text = _clean_alias(alias)
            if alias_text:
                patterns.append((alias_text, record, "exact_alias", DEFAULT_CONFIG["alias_match_confidence"], [alias_text]))

    occupied: list[tuple[int, int]] = []
    for phrase, record, method, confidence, aliases in sorted(patterns, key=lambda x: len(x[0]), reverse=True):
        for match in _pattern(phrase).finditer(unit.text):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            matches.append(_make_detection(record, match.group(0), unit, match.start(), match.end(), method, confidence, aliases))
            occupied.append((match.start(), match.end()))

    # Normalized variants such as self attention vs self-attention.
    for record in vocabulary.records:
        canonical = normalize_lookup(str(record.get("term", "")))
        if not canonical or " " not in canonical:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(canonical).replace(r"\ ", r"[\s\-]+") + r"(?!\w)", re.I)
        for match in pattern.finditer(unit.text):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            matches.append(_make_detection(record, match.group(0), unit, match.start(), match.end(), "normalized_variant", DEFAULT_CONFIG["normalized_variant_confidence"]))
            occupied.append((match.start(), match.end()))
    return matches


def generate_ngram_candidates(unit: TextUnit, limit: int | None = None) -> list[Candidate]:
    """Open-world candidate generation. No Day-1 vocabulary or technical suffix list is used."""
    try:
        candidates = discover_candidates(unit.text, top_n=limit or int(DEFAULT_CONFIG["semantic_candidate_limit"]))
    except Exception:
        return []
    output: list[Candidate] = []
    for candidate in candidates:
        surface = normalize_text(candidate.text)
        if not surface:
            continue
        validation = classify_rejection(surface, context=unit.text)
        if validation.get("decision") != "candidate":
            continue
        start = int(candidate.start_char)
        end = int(candidate.end_char)
        if start < 0 or end <= start or end > len(unit.text):
            match = re.search(re.escape(surface).replace(r"\ ", r"\s+"), unit.text, re.I)
            if not match:
                continue
            start, end = match.start(), match.end()
        output.append(Candidate(surface, normalize_lookup(surface), unit.text, unit.sentence_index, start, end, unit.page_number, candidate.source, float(candidate.score)))
    return output


def generate_candidates(unit: TextUnit, vocabulary: VocabularyIndex) -> list[Candidate]:
    exact = [
        Candidate(d.matched_text, d.normalized_matched_text, d.sentence, d.sentence_index, d.char_start - unit.page_char_start, d.char_end - unit.page_char_start, d.page_number, d.extraction_method, d.confidence)
        for d in find_exact_matches(unit, vocabulary)
    ]
    existing = {(c.char_start, c.char_end) for c in exact}
    semantic = [c for c in generate_ngram_candidates(unit) if (c.char_start, c.char_end) not in existing]
    return exact + semantic


def _sort_key(detection: Detection):
    return (METHOD_PRIORITY.get(detection.extraction_method, 0), detection.char_end - detection.char_start, detection.confidence)


def overlaps(first: Detection, second: Detection) -> bool:
    return not (first.char_end <= second.char_start or second.char_end <= first.char_start)


def resolve_overlaps(detections: Sequence[Detection]) -> list[Detection]:
    accepted: list[Detection] = []
    for candidate in sorted(detections, key=lambda d: (d.sentence_index, d.char_start, -(d.char_end-d.char_start))):
        conflicting = [d for d in accepted if d.sentence_index == candidate.sentence_index and overlaps(d, candidate)]
        if not conflicting:
            accepted.append(candidate)
            continue
        if all(_sort_key(candidate) > _sort_key(existing) for existing in conflicting):
            accepted = [d for d in accepted if d not in conflicting]
            accepted.append(candidate)
    return sorted(accepted, key=lambda d: (d.sentence_index, d.char_start, d.char_end))


class TechnicalTermDetector:
    """Fast open-world detector.

    Day-1 vocabulary is used for deterministic local matching only. It is not
    a closed detection gate. Unknown candidates are represented as semantic
    candidates so the UI can resolve them online when the user hovers/selects.
    """

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        use_semantic: bool = True,
        use_external: bool = False,
        encoder: Any | None = None,
        embedding_store: VocabularyEmbeddingStore | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.vocabulary = VocabularyIndex.build(load_day1_vocabulary(self.dataset_path))
        self.use_semantic = bool(use_semantic)
        self.use_external = bool(use_external)
        self.config = dict(DEFAULT_CONFIG)
        if config:
            self.config.update(config)
        self.encoder = encoder
        self.embedding_store = embedding_store
        self.vocabulary_embeddings = embedding_store.embeddings if embedding_store else None

    def lookup_term(self, term: str) -> dict[str, Any]:
        normalized = normalize_lookup(term)
        for record in self.vocabulary.records:
            if normalize_lookup(str(record.get("term", ""))) == normalized:
                return {**record, "lookup_method": "local"}
            for alias in record.get("aliases") or []:
                if normalize_lookup(str(alias)) == normalized:
                    return {**record, "lookup_method": "alias"}

        # External resolution belongs to explicit knowledge lookup, not PDF upload.
        try:
            from scripts.dynamic_knowledge import resolve_term
            result = resolve_term(term, persist=True)
            if result.get("decision") == "resolved":
                return result
        except Exception:
            pass
        return {
            "term": term, "definition": None, "source": None,
            "source_url": None, "term_id": None, "aliases": [],
            "confidence": 0.0, "lookup_method": "not_found",
        }

    def detect_units(self, units: Sequence[TextUnit]) -> list[Detection]:
        detections: list[Detection] = []
        for unit in units:
            detections.extend(find_exact_matches(unit, self.vocabulary))

        exact_spans = {(d.sentence_index, d.char_start, d.char_end) for d in detections}
        if self.use_semantic:
            candidates: list[Candidate] = []
            for unit in units:
                # The scientific discovery model is run once per sentence, with a
                # small cap. This replaces the previous 96+ candidate loop.
                for candidate in generate_ngram_candidates(unit, int(self.config["semantic_candidate_limit"])):
                    absolute = (candidate.sentence_index, unit.page_char_start + candidate.char_start, unit.page_char_start + candidate.char_end)
                    if absolute in exact_spans:
                        continue
                    candidates.append(candidate)

            for candidate in candidates:
                if candidate.score < float(self.config["semantic_score_threshold"]):
                    continue
                validation = classify_rejection(candidate.surface, context=candidate.sentence)
                if validation.get("decision") != "candidate":
                    continue
                detection = Detection(
                    detection_id=uuid.uuid4().hex[:16],
                    term_id=None,
                    term=candidate.surface,
                    matched_text=candidate.surface,
                    normalized_matched_text=candidate.normalized,
                    aliases_matched=[],
                    domain=None,
                    subdomain=None,
                    definition=None,
                    sentence=candidate.sentence,
                    sentence_index=candidate.sentence_index,
                    page_number=candidate.page_number,
                    char_start=candidate.char_start + next((u.page_char_start for u in units if u.sentence_index == candidate.sentence_index), 0),
                    char_end=candidate.char_end + next((u.page_char_start for u in units if u.sentence_index == candidate.sentence_index), 0),
                    extraction_method="semantic_candidate",
                    confidence=min(0.89, max(0.58, candidate.score)),
                    confidence_reason="Open-world candidate supported by linguistic structure and SciBERT contextual representation; not yet externally validated.",
                    semantic_similarity=candidate.score,
                    semantic_margin=None,
                    vocabulary_source=None,
                    vocabulary_source_url=None,
                )
                detections.append(detection)

        return resolve_overlaps(detections)

    def detect_text(self, text: str, use_semantic: bool | None = None) -> list[Detection]:
        original = self.use_semantic
        if use_semantic is not None:
            self.use_semantic = bool(use_semantic)
        try:
            return self.detect_units(build_text_units(text))
        finally:
            self.use_semantic = original

    def detect_pdf(self, pdf_path: str | Path) -> list[Detection]:
        return self.detect_units(extract_pdf_units(pdf_path))


def normalize_term_for_eval(term: str) -> str:
    return normalize_lookup(term)


def evaluate_term_sets(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, float]:
    expected_set = {normalize_term_for_eval(x) for x in expected}
    predicted_set = {normalize_term_for_eval(x) for x in predicted}
    tp = len(expected_set & predicted_set)
    fp = len(predicted_set - expected_set)
    fn = len(expected_set - predicted_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "true_positive": tp, "false_positive": fp, "false_negative": fn}


def evaluate_detections(detector: TechnicalTermDetector, fixture_path: str | Path, run_semantic: bool = True) -> dict[str, Any]:
    fixtures = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    expected: list[str] = []
    predicted: list[str] = []
    for fixture in fixtures:
        if not fixture.get("evaluation_enabled", True):
            continue
        expected.extend(item["canonical_term"] for item in fixture.get("expected", []) if item.get("canonical_term"))
        predicted.extend(d.term for d in detector.detect_text(str(fixture.get("sentence", "")), use_semantic=run_semantic))
    return {"exact_normalized_term": evaluate_term_sets(expected, predicted)}


def print_detections(detections: Sequence[Detection]) -> None:
    for detection in detections:
        print(json.dumps(detection.to_dict(), ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WordLens open-world technical term detector")
    sub = parser.add_subparsers(dest="command", required=True)
    p_text = sub.add_parser("detect-text")
    p_text.add_argument("text")
    p_pdf = sub.add_parser("detect-pdf")
    p_pdf.add_argument("path")
    p_lookup = sub.add_parser("lookup")
    p_lookup.add_argument("term")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    detector = TechnicalTermDetector(use_semantic=True, use_external=False)
    if args.command == "detect-text":
        print_detections(detector.detect_text(args.text))
    elif args.command == "detect-pdf":
        print_detections(detector.detect_pdf(args.path))
    else:
        print(json.dumps(detector.lookup_term(args.term), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
