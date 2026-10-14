

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, List, Optional, Sequence

import numpy as np
import spacy
import torch
from transformers import AutoModel, AutoTokenizer


# =============================================================================
# Configuration
# =============================================================================

SCIBERT_MODEL = "allenai/scibert_scivocab_uncased"
SPACY_MODEL = "en_core_web_sm"

MIN_CANDIDATE_LENGTH = 2
MAX_CANDIDATE_TOKENS = 5
DEFAULT_TOP_N = 12
SCIBERT_BATCH_SIZE = 16
SCIBERT_MAX_LENGTH = 128

# These are general linguistic function words only. They are NOT a technical
# vocabulary and are never used to decide whether a concept belongs to a domain.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "for", "from", "had", "has", "have", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "me", "my", "myself", "no",
    "nor", "not", "of", "on", "or", "our", "ours", "ourselves", "she",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "us", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "you", "your", "yours", "yourself", "yourselves",
}

# General English action/function words. These prevent obvious standalone
# grammatical actions from being treated as knowledge concepts. This is not a
# domain-specific technical list.
GRAMMATICAL_WORDS = {
    "use", "uses", "used", "using",
    "make", "makes", "made", "making",
    "show", "shows", "shown", "showing",
    "provide", "provides", "provided", "providing",
    "generate", "generates", "generated", "generating",
    "produce", "produces", "produced", "producing",
    "include", "includes", "included", "including",
    "contain", "contains", "contained", "containing",
    "perform", "performs", "performed", "performing",
    "allow", "allows", "allowed", "allowing",
    "enable", "enables", "enabled", "enabling",
    "get", "gets", "got", "getting",
    "give", "gives", "given", "giving",
    "take", "takes", "taken", "taking",
    "apply", "applies", "applied", "applying",
    "consider", "considers", "considered", "considering",
    "represent", "represents", "represented", "representing",
    "describe", "describes", "described", "describing",
    "calculate", "calculates", "calculated", "calculating",
    "predict", "predicts", "predicted", "predicting",
    "classify", "classifies", "classified", "classifying",
}

# Generic leading modifiers that commonly produce ordinary phrases such as
# "other components", "each word", or "more generally". These are linguistic
# filters, not technical terminology.
GENERIC_LEADING_WORDS = {
    "all", "another", "any", "both", "each", "either", "every", "few",
    "less", "least", "many", "more", "most", "much", "neither", "other",
    "several", "some", "such", "various",
}

QUANTITY_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "hundred", "hundreds", "thousand", "thousands", "million",
    "millions", "billion", "billions", "trillion", "trillions", "dozen", "dozens",
    "half", "quarter", "tens",
}

MODIFIER_DEPS = {
    "amod", "compound", "nmod", "poss", "appos", "acl", "flat", "fixed"
}

EDGE_PUNCTUATION = " \t\n\r.,;:!?()[]{}<>“”‘’\"'`"


# =============================================================================
# Data structure
# =============================================================================

@dataclass
class Candidate:
    text: str
    score: float
    source: str
    context: str = ""
    start_char: int = -1
    end_char: int = -1
    sentence_index: int = -1
    reason_code: str = "CANDIDATE"
    reason: str = "Candidate accepted for knowledge validation."


# =============================================================================
# spaCy
# =============================================================================

@lru_cache(maxsize=1)
def _load_spacy():
    try:
        return spacy.load(
            SPACY_MODEL,
            disable=["ner", "textcat", "lemmatizer"],
        )
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exc


# Compatibility aliases used by earlier project code.
def _get_nlp():
    return _load_spacy()


_NLP = _load_spacy()


# =============================================================================
# SciBERT
# =============================================================================

_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModel] = None


def _load_scibert():
    global _TOKENIZER, _MODEL

    if _TOKENIZER is None or _MODEL is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(SCIBERT_MODEL)
        _MODEL = AutoModel.from_pretrained(SCIBERT_MODEL)
        _MODEL.eval()

    return _TOKENIZER, _MODEL


# =============================================================================
# Normalization / compatibility API
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize visible text while preserving lexical meaning."""
    if not text:
        return ""

    text = str(text)
    text = text.replace("\u00a0", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalize_surface(text: str) -> str:
    """Normalize a candidate surface and remove quotation artifacts."""
    text = normalize_text(text)
    text = text.replace('"', " ").replace("'", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(EDGE_PUNCTUATION)
    text = re.sub(r"\s+", " ", text).strip()

    # Never keep an isolated hyphen at a phrase boundary.
    text = re.sub(r"(?:^|\s)-(?=\w)", " ", text)
    text = re.sub(r"(?<=\w)-(?=\s|$)", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_lookup(text: str) -> str:
    """Public normalization used by dynamic_knowledge.py and repositories."""
    text = _normalize_surface(text).lower()
    text = re.sub(r"\s*[-/]\s*", " ", text)
    text = re.sub(r"[^\w\s-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _normalized_key(text: str) -> str:
    return normalize_lookup(text)


# =============================================================================
# Basic linguistic checks
# =============================================================================

def _contains_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _contains_url(text: str) -> bool:
    lowered = text.lower()
    return "http://" in lowered or "https://" in lowered or "www." in lowered


def _is_punctuation_only(text: str) -> bool:
    return not any(char.isalnum() for char in text)


def _is_stopword_candidate(text: str) -> bool:
    key = normalize_lookup(text)
    return bool(key) and " " not in key and key in STOPWORDS


def _has_real_content_word(doc) -> bool:
    return any(token.is_alpha or token.like_num for token in doc)


def _contains_finite_verb(doc) -> bool:
    """Return True for clause-like verbs, but not lexical participles in compounds."""
    for token in doc:
        if token.pos_ == "AUX":
            return True
        if token.pos_ != "VERB" and not token.tag_.startswith("VB"):
            continue

        # Hyphenated technical compounds often contain a participle that
        # spaCy tags as VERB, e.g. ``retrieval-augmented``.  In a compact
        # compound this is not evidence that the candidate is a clause.
        if token.tag_ in {"VBN", "VBG"} and token.dep_ in {
            "amod", "compound", "nmod", "pcomp", "attr", "ROOT"
        }:
            text = str(getattr(doc, "text", ""))
            if "-" in text and len([part for part in text.split("-") if part]) >= 2:
                continue

        return True
    return False


def _has_determiner(doc) -> bool:
    return any(token.dep_ == "det" for token in doc)


def _has_meaningful_modifier(doc) -> bool:
    if doc is None:
        return False
    if hasattr(doc, "dep_") and hasattr(doc, "pos_"):
        return doc.dep_ in MODIFIER_DEPS
    try:
        return any(token.dep_ in MODIFIER_DEPS for token in doc)
    except TypeError:
        return False


def _contains_standalone_hyphen(doc) -> bool:
    if doc is None:
        return False
    if hasattr(doc, "text") and hasattr(doc, "pos_"):
        return doc.text == "-"
    try:
        tokens = [token for token in doc if not token.is_space]
    except TypeError:
        return False
    return len(tokens) == 1 and tokens[0].text == "-"


def _is_quantity_phrase(tokens) -> bool:
    if not tokens:
        return False

    quantity_count = sum(
        1 for token in tokens
        if token.like_num or token.lower_ in QUANTITY_WORDS
    )

    # "tens to millions", "two hundred", etc.
    return len(tokens) > 1 and quantity_count >= 2


def _is_generic_leading_phrase(tokens) -> bool:
    if len(tokens) <= 1:
        return False
    return tokens[0].lower_ in GENERIC_LEADING_WORDS


def _has_concept_structure(doc) -> bool:
    """General linguistic structure check; no technical vocabulary."""
    tokens = [t for t in doc if not t.is_space and not t.is_punct]

    if not tokens or len(tokens) > MAX_CANDIDATE_TOKENS:
        return False

    if _contains_standalone_hyphen(doc):
        return False

    if _is_quantity_phrase(tokens):
        return False

    if _is_generic_leading_phrase(tokens):
        return False

    if tokens[0].lower_ in STOPWORDS or tokens[-1].lower_ in STOPWORDS:
        return False

    if _contains_finite_verb(doc):
        return False

    # A multi-word concept should contain at least one noun/proper noun, or a
    # lexical adjective that modifies a noun-like expression.
    if not any(token.pos_ in {"NOUN", "PROPN"} for token in tokens):
        return False

    return True


# =============================================================================
# Public candidate validation
# =============================================================================

def classify_rejection(
    text: str,
    context: Optional[str] = None,
) -> dict[str, Any]:
    """
    Decide whether text is structurally suitable for downstream knowledge
    validation.

    IMPORTANT: is_technical=True means "candidate suitable for validation".
    It does NOT mean the term has already been proven technical.
    """
    normalized = _normalize_surface(text)

    def reject(code: str, reason: str) -> dict[str, Any]:
        return {
            "is_technical": False,
            "decision": "rejected",
            "reason_code": code,
            "reason": reason,
        }

    def accept(reason: str = "Candidate has usable linguistic structure and can proceed to knowledge validation.") -> dict[str, Any]:
        return {
            "is_technical": True,
            "decision": "candidate",
            "reason_code": "CANDIDATE",
            "reason": reason,
        }

    if not normalized:
        return reject("EMPTY", "Candidate is empty after normalization.")

    if _is_punctuation_only(normalized):
        return reject("PUNCTUATION", "Candidate contains no meaningful lexical content.")

    if _contains_url(normalized):
        return reject("URL", "Candidate is a URL rather than a technical concept.")

    if _contains_number(normalized):
        return reject("QUANTITY", "Candidate contains numeric content and is treated as a quantity expression.")

    parsed = _NLP(normalized)
    meaningful = [t for t in parsed if not t.is_space and not t.is_punct]

    if not meaningful or not _has_real_content_word(parsed):
        return reject("NO_CONTENT", "Candidate contains no meaningful lexical content.")

    if _is_stopword_candidate(normalized):
        return reject("STOPWORD", "Candidate is a standalone function word.")

    if len(meaningful) == 1:
        token = meaningful[0]
        if token.lower_ in STOPWORDS:
            return reject("STOPWORD", "Candidate is a standalone function word.")
        if token.lower_ in GRAMMATICAL_WORDS:
            return reject("GRAMMATICAL_WORD", "Candidate is a general action/function word rather than a useful concept candidate.")
        if token.pos_ not in {"NOUN", "PROPN", "ADJ", "VERB", "ADV", "NUM"}:
            return reject("GENERAL_LANGUAGE", "Candidate is not a useful standalone lexical unit.")
        return accept()

    if len(normalize_lookup(normalized)) < MIN_CANDIDATE_LENGTH:
        return reject("TOO_SHORT", "Candidate is too short.")

    if not _has_concept_structure(parsed):
        # Hyphenated lexical compounds are valid candidate shapes even when
        # one component is morphologically tagged as a participial verb.
        # This is a linguistic rule, not a technical-term whitelist.
        if "-" in normalized and all(
            part.isalpha() for part in normalized.split("-") if part
        ) and len([part for part in normalized.split("-") if part]) >= 2:
            lexical_parts = [
                token for token in parsed
                if not token.is_space and not token.is_punct
            ]
            if lexical_parts and all(
                token.pos_ in {"NOUN", "PROPN", "ADJ", "VERB"}
                for token in lexical_parts
            ):
                return accept("Hyphenated lexical compound can proceed to knowledge validation.")

        if _contains_finite_verb(parsed):
            return reject("GRAMMATICAL_PHRASE", "Candidate contains a verb and behaves like a clause rather than a compact concept.")
        if _is_quantity_phrase(meaningful):
            return reject("QUANTITY", "Candidate is primarily a quantitative expression rather than a concept.")
        if _is_generic_leading_phrase(meaningful):
            return reject("GENERIC_NOUN_PHRASE", "Candidate begins with a generic quantifier/determiner phrase rather than a compact concept.")
        return reject("GENERAL_LANGUAGE", "Candidate does not have sufficiently compact concept-like linguistic structure.")

    return accept("Multi-word candidate has compact linguistic concept structure and can proceed to knowledge validation.")


# =============================================================================
# Candidate scoring / construction
# =============================================================================

def _candidate_score(text: str, source: str, doc_span=None) -> float:
    score = 0.70
    words = normalize_lookup(text).split()

    if source == "noun_chunk":
        score += 0.08
    elif source == "hyphenated":
        score += 0.05

    if len(words) >= 2:
        score += 0.05

    if doc_span is not None and _has_meaningful_modifier(doc_span):
        score += 0.05

    return min(score, 0.95)


def _make_candidate(
    text: str,
    source: str,
    context: str,
    start_char: int = -1,
    end_char: int = -1,
    sentence_index: int = -1,
    doc_span=None,
) -> Optional[Candidate]:
    surface = _normalize_surface(text)
    if not surface:
        return None

    validation = classify_rejection(surface, context=context)
    if validation["decision"] != "candidate":
        return None

    return Candidate(
        text=surface,
        score=_candidate_score(surface, source, doc_span),
        source=source,
        context=context,
        start_char=start_char,
        end_char=end_char,
        sentence_index=sentence_index,
        reason_code=str(validation.get("reason_code", "CANDIDATE")),
        reason=str(validation.get("reason", "Candidate accepted for knowledge validation.")),
    )


# =============================================================================
# Candidate extraction
# =============================================================================

def _atomic_candidates(
    sentence_doc,
    sentence_index: int,
    sentence_text: str,
) -> Iterable[Candidate]:
    """
    Emit atomic lexical candidates independently of multi-word phrases.

    Therefore both of these can exist:
        token embeddings
        token
        embeddings
    """
    for token in sentence_doc:
        if token.is_space or token.is_punct or token.like_url:
            continue

        if not (token.is_alpha or token.text.replace("-", "").isalnum()):
            continue

        surface = _normalize_surface(token.text)
        if not surface:
            continue

        candidate = _make_candidate(
            text=surface,
            source="token",
            context=sentence_text,
            start_char=token.idx,
            end_char=token.idx + len(token.text),
            sentence_index=sentence_index,
            doc_span=token,
        )
        if candidate is not None:
            yield candidate


def _noun_chunk_candidates(
    sentence_doc,
    sentence_index: int,
    sentence_text: str,
) -> Iterable[Candidate]:
    """Extract only compact noun phrases from spaCy's original parse."""
    try:
        chunks = list(sentence_doc.noun_chunks)
    except Exception:
        chunks = []

    for chunk in chunks:
        tokens = [t for t in chunk if not t.is_space and not t.is_punct]
        if len(tokens) <= 1:
            continue

        # Prevent parser artifacts such as:
        #   attention encodes vectors
        #   -width sequence
        # from becoming phrases.
        if any(t.pos_ in {"VERB", "AUX"} or t.tag_.startswith("VB") for t in tokens):
            continue

        if any(t.text == "-" for t in chunk):
            continue

        if tokens[0].lower_ in STOPWORDS or tokens[-1].lower_ in STOPWORDS:
            continue

        if _is_quantity_phrase(tokens) or _is_generic_leading_phrase(tokens):
            continue

        surface = _normalize_surface(chunk.text)
        candidate = _make_candidate(
            text=surface,
            source="noun_chunk",
            context=sentence_text,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            sentence_index=sentence_index,
            doc_span=chunk,
        )
        if candidate is not None:
            yield candidate

        # Also expose linguistically valid suffix noun phrases. This is not an
        # arbitrary sliding n-gram mechanism; the suffix must remain a noun
        # phrase in the dependency parse.
        for start in range(chunk.start + 1, chunk.end - 1):
            subspan = sentence_doc.doc[start:chunk.end]
            subtokens = [t for t in subspan if not t.is_space and not t.is_punct]

            if len(subtokens) < 2:
                continue
            if any(t.pos_ in {"VERB", "AUX"} or t.tag_.startswith("VB") for t in subtokens):
                continue
            if any(t.text == "-" for t in subspan):
                continue
            if subtokens[0].lower_ in STOPWORDS or subtokens[-1].lower_ in STOPWORDS:
                continue
            if _is_quantity_phrase(subtokens) or _is_generic_leading_phrase(subtokens):
                continue
            if not any(t.pos_ in {"NOUN", "PROPN"} for t in subtokens):
                continue

            candidate = _make_candidate(
                text=subspan.text,
                source="noun_chunk",
                context=sentence_text,
                start_char=subspan.start_char,
                end_char=subspan.end_char,
                sentence_index=sentence_index,
                doc_span=subspan,
            )
            if candidate is not None:
                yield candidate


def _hyphenated_candidates(
    sentence_doc,
    sentence_index: int,
    sentence_text: str,
) -> Iterable[Candidate]:
    """Preserve hyphenated lexical units from the original sentence text."""
    pattern = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+\b")

    for match in pattern.finditer(sentence_text):
        surface = _normalize_surface(match.group(0))
        candidate = _make_candidate(
            text=surface,
            source="hyphenated",
            context=sentence_text,
            start_char=match.start(),
            end_char=match.end(),
            sentence_index=sentence_index,
        )
        if candidate is not None:
            yield candidate


# =============================================================================
# Deduplication / contextual scoring
# =============================================================================

def _deduplicate_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    best: dict[str, Candidate] = {}
    source_priority = {"noun_chunk": 3, "hyphenated": 2, "token": 1}

    for candidate in candidates:
        key = _normalized_key(candidate.text)
        if not key:
            continue

        current = best.get(key)
        if current is None:
            best[key] = candidate
            continue

        current_priority = source_priority.get(current.source, 0)
        candidate_priority = source_priority.get(candidate.source, 0)

        if candidate.score > current.score or (
            candidate.score == current.score and candidate_priority > current_priority
        ):
            best[key] = candidate

    return list(best.values())


def _apply_contextual_scores(
    candidates: List[Candidate],
    batch_size: int = SCIBERT_BATCH_SIZE,
) -> List[Candidate]:
    """
    Blend deterministic linguistic quality with SciBERT context similarity.

    SciBERT is a representation model here, not a trained NER classifier.
    """
    if not candidates:
        return []

    try:
        contexts = [candidate.context or candidate.text for candidate in candidates]
        terms = [candidate.text for candidate in candidates]

        context_vectors = encode(contexts, batch_size=batch_size)
        term_vectors = encode(terms, batch_size=batch_size)

        similarities = np.sum(context_vectors * term_vectors, axis=1)

        for candidate, similarity in zip(candidates, similarities):
            semantic = float(np.clip((float(similarity) + 1.0) / 2.0, 0.0, 1.0))
            candidate.score = float(
                np.clip(
                    0.70 * candidate.score + 0.30 * semantic,
                    0.0,
                    1.0,
                )
            )
    except Exception:
        # SciBERT is an enhancement. Discovery must remain usable if model
        # loading fails or the environment is offline.
        pass

    return candidates


def generate_candidates(text: str, top_n: int = DEFAULT_TOP_N) -> List[Candidate]:
    """
    Open-world scientific concept discovery.

    Pipeline:
        text
          -> spaCy linguistic analysis
          -> noun phrases + hyphenated units + atomic lexical units
          -> structural filtering
          -> SciBERT contextual scoring
          -> deduplication/ranking

    The Day-1 knowledge base is NOT a detection gate.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if top_n <= 0:
        return []

    text = normalize_text(text)
    if not text:
        return []

    doc = _NLP(text)
    candidates: List[Candidate] = []

    for sentence_index, sentence in enumerate(doc.sents):
        sentence_text = normalize_text(sentence.text)
        if not sentence_text:
            continue

        # IMPORTANT: atomic extraction is independent from phrase extraction.
        candidates.extend(_atomic_candidates(sentence, sentence_index, sentence_text))
        candidates.extend(_noun_chunk_candidates(sentence, sentence_index, sentence_text))
        candidates.extend(_hyphenated_candidates(sentence, sentence_index, sentence_text))

    candidates = _deduplicate_candidates(candidates)
    if not candidates:
        return []

    candidates = _apply_contextual_scores(candidates)

    candidates.sort(
        key=lambda c: (
            -float(c.score),
            -len(normalize_lookup(c.text).split()),
            c.start_char,
            c.text.lower(),
        )
    )

    return candidates[: int(top_n)]


# =============================================================================
# SciBERT encoding API
# =============================================================================

def _mean_pool(hidden_states, attention_mask) -> np.ndarray:
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    masked = hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).detach().cpu().numpy()


def encode(
    texts: Sequence[str],
    batch_size: int = SCIBERT_BATCH_SIZE,
    max_length: int = SCIBERT_MAX_LENGTH,
) -> np.ndarray:
    """Return L2-normalized mean-pooled SciBERT vectors."""
    if isinstance(texts, str):
        texts = [texts]

    texts = list(texts)
    if not texts:
        return np.empty((0, 768), dtype=np.float32)

    tokenizer, model = _load_scibert()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    vectors: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(texts), max(1, batch_size)):
            batch = [str(x) for x in texts[start:start + max(1, batch_size)]]

            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            output = model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.maximum(norms, 1e-12)
            vectors.append(pooled.astype(np.float32))

    return np.concatenate(vectors, axis=0).astype(np.float32)


def encode_candidates(
    candidates: Sequence[Candidate],
    batch_size: int = SCIBERT_BATCH_SIZE,
    max_length: int = SCIBERT_MAX_LENGTH,
) -> np.ndarray:
    """Encode candidate + surrounding context for diagnostics/downstream use."""
    if not candidates:
        return np.empty((0, 768), dtype=np.float32)

    texts = [
        f"Term: {candidate.text}. Context: {candidate.context}"
        if candidate.context
        else candidate.text
        for candidate in candidates
    ]

    return encode(texts, batch_size=batch_size, max_length=max_length)


# =============================================================================
# Diagnostics / CLI
# =============================================================================

def inspect_candidates(text: str, top_n: int = 20, encode_vectors: bool = False):
    candidates = generate_candidates(text, top_n=top_n)
    if not encode_vectors:
        return candidates
    return candidates, encode_candidates(candidates)


if __name__ == "__main__":
    sample = (
        "In machine learning, attention is a method that determines the importance "
        "of each component in a sequence relative to the other components in that "
        "sequence. In natural language processing, importance is represented by "
        '"soft" weights assigned to each word in a sentence. More generally, '
        "attention encodes vectors called token embeddings across a fixed-width "
        "sequence that can range from tens to millions of tokens in size."
    )

    for candidate in generate_candidates(sample, top_n=30):
        print(
            f"- {candidate.text} | "
            f"score={candidate.score:.4f} | "
            f"source={candidate.source} | "
            f"reason={candidate.reason_code}"
        )
