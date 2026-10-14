from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.config import get_settings
from backend.app.db.database import _engine, initialize_database
from backend.app.models.terminology import TechnicalTerm

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Export accepted runtime WordLens terminology without overwriting the seed dataset.")
    parser.add_argument("--output", default="data/processed/runtime_technical_terms.json")
    args = parser.parse_args()
    settings = get_settings()
    initialize_database(settings.database_url)
    with Session(_engine) as session:
        rows = session.execute(select(TechnicalTerm).where(TechnicalTerm.is_active.is_(True), TechnicalTerm.quality_status == "accepted")).scalars().all()
        records = []
        for r in rows:
            records.append({
                "term_id": r.term_id, "term": r.term, "normalized_term": r.normalized_term,
                "aliases": r.aliases or [], "domain": r.domain, "subdomain": r.subdomain,
                "definition": r.definition, "simplified_definition": r.simplified_definition,
                "source": r.source, "source_url": r.source_url, "confidence": r.confidence,
                "quality_status": r.quality_status, "is_active": r.is_active,
                "dataset_hash": r.dataset_hash, "imported_at": r.imported_at.isoformat() if r.imported_at else None,
            })
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader(); writer.writerows(records)
    print(json.dumps({"status":"success","records":len(records),"json":str(path),"csv":str(csv_path),"generated_at":datetime.now(timezone.utc).isoformat()}, indent=2))

if __name__ == "__main__":
    main()
