
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = RAW / ".cache"
PROCESSED = DATA / "processed"
EVALUATION = DATA / "evaluation"
for p in (RAW, CACHE, PROCESSED, EVALUATION):
    p.mkdir(parents=True, exist_ok=True)

JSON_PATH = PROCESSED / "technical_terms.json"
CSV_PATH = PROCESSED / "technical_terms.csv"
REPORT_PATH = PROCESSED / "validation_report.json"
SUMMARY_PATH = PROCESSED / "dataset_summary.md"
ACCEPTED_PATH = EVALUATION / "accepted_terms.csv"
REVIEW_PATH = EVALUATION / "needs_review_terms.csv"
REJECTED_PATH = EVALUATION / "rejected_terms.csv"

REQUIRED_FIELDS = [
    "term_id", "term", "normalized_term", "aliases", "domain", "subdomain",
    "definition", "simplified_definition", "simplified_definition_status",
    "source", "source_url", "source_definition", "source_retrieved_at",
    "confidence", "confidence_reason", "quality_status", "quality_notes", "is_active",
]

@dataclass(frozen=True)
class Source:
    source_id: str
    name: str
    url: str
    priority: int

SOURCES = [
    Source("google_genai", "Google Machine Learning Glossary — Generative AI", "https://developers.google.com/machine-learning/glossary/generative", 1),
    Source("google_ml", "Google Machine Learning Glossary", "https://developers.google.com/machine-learning/glossary", 2),
    Source("google_cloud_genai", "Google Cloud Generative AI Glossary", "https://docs.cloud.google.com/docs/generative-ai/glossary", 3),
    Source("nist_csrc", "NIST CSRC Glossary", "https://csrc.nist.gov/glossary?ipp-sm=all", 4),
]

DOMAIN_HINTS = {
    "Artificial Intelligence": ["artificial intelligence", "intelligent agent", "reasoning", "planning", "autonomous", "knowledge representation"],
    "Machine Learning": ["machine learning", "supervised learning", "unsupervised learning", "reinforcement learning", "classification", "regression", "clustering", "overfitting", "generalization"],
    "Deep Learning": ["deep learning", "neural network", "convolutional", "recurrent", "backpropagation", "gradient", "activation function"],
    "NLP": ["natural language", "language model", "tokenization", "linguistic", "machine translation", "named entity", "text classification", "semantic"],
    "Computer Vision": ["computer vision", "image", "video", "object detection", "image segmentation", "optical character", "visual"],
    "Data Science": ["data science", "feature engineering", "dimensionality reduction", "statistical", "sampling", "data preprocessing"],
    "Software Engineering": ["software engineering", "software architecture", "software testing", "deployment", "version control", "debugging", "api"],
    "Databases": ["database", "relational", "sql", "transaction", "schema", "query optimization", "nosql"],
    "Computer Networks": ["network protocol", "transport layer", "tcp", "http", "routing", "packet", "dns", "distributed system"],
    "Cybersecurity": ["cybersecurity", "security", "privacy", "adversarial", "attack", "threat", "vulnerability", "encryption", "injection"],
}

SUBDOMAIN_HINTS = {
    "foundations": ["artificial intelligence", "machine learning", "deep learning", "foundation model", "neural network"],
    "transformer_llm": ["transformer", "attention", "encoder", "decoder", "language model", "large language model", "positional encoding"],
    "representation_context": ["token", "tokenization", "embedding", "representation", "context window", "latent space"],
    "training_alignment": ["pretraining", "pre-trained", "fine-tuning", "instruction tuning", "reinforcement learning", "alignment", "distillation", "regularization", "gradient descent"],
    "prompting_inference": ["prompt", "prompt engineering", "zero-shot", "few-shot", "inference", "decoding", "sampling", "temperature", "beam search"],
    "retrieval_agents": ["retrieval augmented", "retriever", "vector search", "semantic search", "knowledge base", "agent", "tool calling", "orchestration"],
    "generative_media": ["multimodal", "image generation", "text-to-image", "diffusion", "generative adversarial", "variational autoencoder"],
    "evaluation": ["evaluation", "benchmark", "ground truth", "factuality", "accuracy", "precision", "recall", "latency"],
    "safety_security": ["hallucination", "prompt injection", "jailbreak", "adversarial", "privacy", "fairness", "bias", "security"],
}

TECHNICAL_CUES = {
    "algorithm", "architecture", "attention", "benchmark", "classifier", "classification", "convolution",
    "embedding", "encoder", "decoder", "gradient", "inference", "neural", "parameter", "prompt", "retrieval",
    "semantic", "token", "transformer", "vector", "training", "learning", "model", "network", "protocol",
    "database", "encryption", "authentication", "vulnerability", "cybersecurity", "optimization", "representation",
    "generative", "autonomous", "agent", "quantum", "robot", "genomics", "bioinformatics", "medical", "finite element",
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "WordLens-DatasetBuilder/2.0 educational research project"})


def norm(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9+#./ -]", " ", text)
    text = re.sub(r"[-/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def term_id(term: str) -> str:
    return "term_" + hashlib.sha256(norm(term).encode()).hexdigest()[:16]


def cache_path(source: Source) -> Path:
    return CACHE / source.source_id / "source.html"


def fetch(source: Source, refresh: bool, offline: bool) -> tuple[str | None, str]:
    path = cache_path(source)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (offline or not refresh):
        return path.read_text(encoding="utf-8", errors="ignore"), "cached"
    if offline:
        return None, "offline_no_cache"
    parsed = RobotFileParser()
    try:
        parsed.set_url(urljoin(source.url, "/robots.txt"))
        parsed.read()
        if not parsed.can_fetch(SESSION.headers["User-Agent"], source.url):
            return None, "robots_blocked"
    except Exception:
        pass
    try:
        response = SESSION.get(source.url, timeout=30)
        response.raise_for_status()
        path.write_text(response.text, encoding="utf-8")
        time.sleep(0.5)
        return response.text, "downloaded"
    except Exception as exc:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="ignore"), "cached_after_error"
        return None, f"error:{type(exc).__name__}"


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def plausible(term: str, definition: str) -> bool:
    t = norm(term)
    d = norm(definition)
    if not t or len(t) < 2 or len(t) > 160 or len(d) < 30 or len(d) > 6000:
        return False
    if t in {"see also", "definition", "term", "glossary", "home"}:
        return False
    words = set((t + " " + d).split())
    cue_score = len(words & TECHNICAL_CUES)
    acronym = bool(re.fullmatch(r"[A-Za-z0-9+.-]{2,12}", term.strip()))
    multiword = len(t.split()) >= 2
    return cue_score >= 1 or acronym or multiword


def classify(term: str, definition: str) -> tuple[str | None, str | None]:
    text = norm(term + " " + definition)
    domain_scores = {k: sum(1 for x in v if norm(x) in text) for k, v in DOMAIN_HINTS.items()}
    sub_scores = {k: sum(1 for x in v if norm(x) in text) for k, v in SUBDOMAIN_HINTS.items()}
    domain = max(domain_scores, key=domain_scores.get) if domain_scores and max(domain_scores.values()) else None
    sub = max(sub_scores, key=sub_scores.get) if sub_scores and max(sub_scores.values()) else None
    return domain, sub


def parse_google(html: str, source: Source) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records = []
    headings = soup.find_all("h2")
    for i, heading in enumerate(headings):
        term = clean(heading.get_text(" ", strip=True))
        if not term:
            continue
        parts = []
        for sibling in heading.next_siblings:
            if isinstance(sibling, Tag) and sibling.name == "h2":
                break
            if not isinstance(sibling, Tag) or sibling.name not in {"p", "li", "dd", "blockquote"}:
                continue
            value = clean(sibling.get_text(" ", strip=True))
            if value and not value.startswith(("Skip to", "Sign in", "Privacy Policy", "Terms of Service")):
                parts.append(value)
        definition = clean(" ".join(dict.fromkeys(parts)))
        if plausible(term, definition):
            records.append({"term": term, "aliases": [], "definition": definition, "source": source.name, "source_url": source.url, "source_priority": source.priority})
    return records


def parse_nist(html: str, source: Source) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for link in soup.find_all("a", href=True):
        term = clean(link.get_text(" ", strip=True))
        href = link.get("href", "")
        if not term or "/glossary/term/" not in href:
            continue
        container = link.parent
        definition = ""
        if container:
            for node in container.find_all_next(["p", "dd"], limit=3):
                candidate = clean(node.get_text(" ", strip=True))
                if len(candidate) >= 30:
                    definition = candidate
                    break
        if plausible(term, definition):
            records.append({"term": term, "aliases": [], "definition": definition, "source": source.name, "source_url": urljoin(source.url, href), "source_priority": source.priority})
    return records


def make_record(item: dict, retrieved_at: str) -> dict:
    domain, subdomain = classify(item["term"], item["definition"])
    confidence = 0.90 if domain else 0.80
    return {
        "term_id": term_id(item["term"]),
        "term": item["term"],
        "normalized_term": norm(item["term"]),
        "aliases": item.get("aliases", []),
        "domain": domain,
        "subdomain": subdomain,
        "definition": item["definition"],
        "simplified_definition": None,
        "simplified_definition_status": "not_generated",
        "source": item["source"],
        "source_url": item["source_url"],
        "source_definition": item["definition"],
        "source_retrieved_at": retrieved_at,
        "confidence": confidence,
        "confidence_reason": "source-derived glossary entry; taxonomy labels are optional hints",
        "quality_status": "accepted",
        "quality_notes": "Validated source-derived record.",
        "is_active": True,
    }


def load_existing() -> list[dict]:
    if not JSON_PATH.exists():
        return []
    try:
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def deduplicate(records: list[dict]) -> list[dict]:
    by_key = {}
    for r in records:
        key = r["normalized_term"]
        old = by_key.get(key)
        if old is None or (r.get("confidence", 0), len(r.get("definition", ""))) > (old.get("confidence", 0), len(old.get("definition", ""))):
            by_key[key] = r
    return sorted(by_key.values(), key=lambda r: r["normalized_term"])


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_FIELDS)
        writer.writeheader()
        for r in records:
            row = dict(r)
            row["aliases"] = json.dumps(row.get("aliases") or [], ensure_ascii=False)
            writer.writerow(row)


def validate(records: list[dict]) -> list[str]:
    errors = []
    seen = set()
    for r in records:
        missing = [k for k in REQUIRED_FIELDS if k not in r]
        if missing:
            errors.append(f"{r.get('term_id')}: missing {missing}")
        if r.get("term_id") in seen:
            errors.append(f"duplicate term_id: {r.get('term_id')}")
        seen.add(r.get("term_id"))
        if not r.get("term") or not r.get("normalized_term") or not r.get("definition"):
            errors.append(f"invalid empty field: {r.get('term_id')}")
    return errors


def build(refresh: bool, offline: bool) -> dict:
    existing = load_existing()
    existing_by_id = {r.get("term_id"): r for r in existing if r.get("term_id")}
    source_status = {}
    discovered = []
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for source in SOURCES:
        content, status = fetch(source, refresh, offline)
        source_status[source.source_id] = status
        if not content:
            continue
        parsed = parse_nist(content, source) if source.source_id == "nist_csrc" else parse_google(content, source)
        discovered.extend(make_record(x, retrieved_at) for x in parsed)
    merged = list(existing_by_id.values())
    for record in discovered:
        existing_record = existing_by_id.get(record["term_id"])
        if existing_record is None:
            merged.append(record)
    merged = deduplicate(merged)
    errors = validate(merged)
    accepted = [r for r in merged if r.get("quality_status") == "accepted" and r.get("is_active") is True]
    review = [r for r in merged if r.get("quality_status") == "needs_review"]
    JSON_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(CSV_PATH, merged)
    write_csv(ACCEPTED_PATH, accepted)
    write_csv(REVIEW_PATH, review)
    REJECTED_PATH.write_text("term,reason\n", encoding="utf-8")
    domains = {}
    for r in merged:
        d = r.get("domain") or "Unknown"
        domains[d] = domains.get(d, 0) + 1
    report = {
        "generated_at": retrieved_at,
        "raw_seed_records": len(existing),
        "new_discovered_records": max(0, len(merged) - len(existing)),
        "final_records": len(merged),
        "accepted": len(accepted),
        "needs_review": len(review),
        "validation_status": "passed" if not errors else "failed",
        "validation_errors": errors,
        "sources": source_status,
        "manual_term_list_used": False,
        "domain_rules_are_detection_gates": False,
        "taxonomy_is_optional": True,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(
        "# WordLens Technical Terminology Dataset\n\n"
        f"- Seed records: {len(existing)}\n"
        f"- New discovered records: {report['new_discovered_records']}\n"
        f"- Final records: {len(merged)}\n"
        f"- Accepted active: {len(accepted)}\n"
        "- Domain/subdomain rules are classification hints only.\n"
        "- New domains are allowed and may initially be labelled Unknown.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    build(args.refresh, args.offline)

if __name__ == "__main__":
    main()
