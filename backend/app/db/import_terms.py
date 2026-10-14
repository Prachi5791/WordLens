

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.database import Base, _engine
from backend.app.models.terminology import TechnicalTerm


def utc_now():
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must contain a list of records.")
    return data


def import_terms():
    settings = get_settings()
    path = Path(settings.dataset_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records = load_dataset(path)
    accepted = [r for r in records if r.get("quality_status") == "accepted" and r.get("is_active") is True]
    if not accepted:
        raise ValueError("Dataset contains no accepted active records.")
    required = {"term_id", "term", "normalized_term", "definition", "quality_status", "is_active"}
    for r in accepted:
        missing = required - set(r)
        if missing:
            raise ValueError(f"Record {r.get('term_id')} missing fields: {sorted(missing)}")
    dataset_hash = sha256_file(path)
    now = utc_now()
    Base.metadata.create_all(bind=_engine)
    inserted = updated = 0
    with Session(_engine) as session:
        for r in accepted:
            existing = session.execute(select(TechnicalTerm).where(TechnicalTerm.term_id == r["term_id"])).scalar_one_or_none()
            values = {
                "term_id": r["term_id"], "term": r["term"], "normalized_term": r["normalized_term"],
                "aliases": r.get("aliases") or [], "domain": r.get("domain"), "subdomain": r.get("subdomain"),
                "definition": r["definition"], "simplified_definition": r.get("simplified_definition"),
                "source": r.get("source"), "source_url": r.get("source_url"), "confidence": r.get("confidence"),
                "quality_status": "accepted", "is_active": True, "dataset_hash": dataset_hash,
                "imported_at": now, "updated_at": now,
            }
            if existing is None:
                values["created_at"] = now
                session.add(TechnicalTerm(**values)); inserted += 1
            else:
                for k, v in values.items(): setattr(existing, k, v)
                updated += 1
        session.commit()
    report = {"status":"success","total_dataset_records":len(records),"accepted_active_records":len(accepted),"skipped":len(records)-len(accepted),"inserted":inserted,"updated":updated,"dataset_hash":dataset_hash,"imported_at":now.isoformat()}
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    import_terms()
