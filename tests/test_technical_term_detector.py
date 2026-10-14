import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.technical_term_detector import (
    Candidate,
    Detection,
    TechnicalTermDetector,
    VocabularyEmbeddingStore,
    VocabularyIndex,
    _f1,
    build_text_units,
    evaluate_detections,
    extract_pdf_pages,
    generate_candidates,
    load_day1_vocabulary,
    normalize_lookup,
    split_sentences,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "technical_terms.json"


class TestDay1Vocabulary(unittest.TestCase):
    def test_accepted_only_and_count(self):
        records = load_day1_vocabulary(DATASET)
        self.assertEqual(len(records), 160)
        self.assertTrue(all(r["quality_status"] == "accepted" for r in records))
        self.assertTrue(all(r["is_active"] for r in records))

    def test_canonical_lookup(self):
        records = load_day1_vocabulary(DATASET)
        vocab = VocabularyIndex.build(records)
        idx = vocab.canonical[normalize_lookup("generative AI")]
        self.assertEqual(vocab.get(idx)["term"], "generative AI")

    def test_alias_lookup_with_test_record(self):
        record = dict(load_day1_vocabulary(DATASET)[0])
        record["term"] = "test canonical"
        record["normalized_term"] = "test canonical"
        record["term_id"] = "test_alias"
        record["aliases"] = ["TC"]
        vocab = VocabularyIndex.build([record])
        self.assertIn("tc", vocab.aliases)
        self.assertEqual(vocab.get(vocab.aliases["tc"][0])["term"], "test canonical")


    def test_alias_candidate_matching(self):
        record = dict(load_day1_vocabulary(DATASET)[0])
        record["term"] = "retrieval test"
        record["normalized_term"] = "retrieval test"
        record["term_id"] = "alias_test"
        record["aliases"] = ["RT"]
        vocab = VocabularyIndex.build([record])
        units = build_text_units("The RT method was evaluated.")
        candidates = generate_candidates(units[0], vocab)
        alias_candidates = [c for c in candidates if c.extraction_method == "exact_alias"]
        self.assertEqual(len(alias_candidates), 1)
        self.assertEqual(alias_candidates[0].surface, "RT")

    def test_hyphen_space_normalization(self):
        self.assertEqual(
            normalize_lookup("retrieval-augmented generation"),
            "retrieval augmented generation",
        )


class TestSegmentationAndCandidates(unittest.TestCase):
    def test_sentence_segmentation(self):
        units = build_text_units("Generative AI is useful. Fine-tuning adapts models.")
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0].sentence_index, 0)
        self.assertEqual(units[1].sentence_index, 1)

    def test_exact_and_variant_candidates(self):
        records = load_day1_vocabulary(DATASET)
        vocab = VocabularyIndex.build(records)
        units = build_text_units(
            "The system uses retrieval augmented generation for grounding."
        )
        candidates = generate_candidates(units[0], vocab)
        methods = {c.extraction_method for c in candidates}
        self.assertIn("normalized_variant", methods)

    def test_candidate_surface_preserved(self):
        records = load_day1_vocabulary(DATASET)
        vocab = VocabularyIndex.build(records)
        units = build_text_units("The system uses Generative AI.")
        candidates = generate_candidates(units[0], vocab)
        self.assertTrue(any(c.surface == "Generative AI" for c in candidates))


class TestOverlapAndOutput(unittest.TestCase):
    def test_overlap_resolution_prefers_canonical_then_longer(self):
        base = dict(
            detection_id="x",
            term_id="1",
            term="large language model",
            matched_text="large language model",
            normalized_matched_text="large language model",
            aliases_matched=[],
            domain="Artificial Intelligence",
            subdomain="transformer_llm",
            definition="definition",
            sentence="A large language model is used.",
            sentence_index=0,
            page_number=None,
            char_start=2,
            char_end=23,
            extraction_method="exact_canonical",
            confidence=0.98,
            confidence_reason="exact canonical vocabulary match",
            semantic_similarity=None,
            semantic_margin=None,
            vocabulary_source="test",
            vocabulary_source_url="test",
        )
        shorter = Detection(
            **{**base, "term_id": "2", "term": "language model",
               "matched_text": "language model", "normalized_matched_text": "language model",
               "char_start": 9, "char_end": 23}
        )
        detector = TechnicalTermDetector(DATASET)
        # Use the private resolver through the module namespace without loading SciBERT.
        from scripts.technical_term_detector import _resolve_overlaps
        result = _resolve_overlaps([base_detection := Detection(**base), shorter])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].term, "large language model")

    def test_confidence_is_deterministic(self):
        detector = TechnicalTermDetector(DATASET)
        result1 = detector.detect_text("Generative AI is used.", use_semantic=False)
        result2 = detector.detect_text("Generative AI is used.", use_semantic=False)
        self.assertEqual(result1, result2)
        self.assertEqual(result1[0]["confidence"], 0.98)
        self.assertEqual(result1[0]["confidence_reason"], "exact canonical vocabulary match")

    def test_output_schema(self):
        detector = TechnicalTermDetector(DATASET)
        result = detector.detect_text("Generative AI is used.", use_semantic=False)
        required = {
            "detection_id", "term_id", "term", "matched_text",
            "normalized_matched_text", "aliases_matched", "domain",
            "subdomain", "definition", "sentence", "sentence_index",
            "page_number", "char_start", "char_end", "extraction_method",
            "confidence", "confidence_reason", "semantic_similarity",
            "semantic_margin", "vocabulary_source", "vocabulary_source_url",
        }
        self.assertTrue(required.issubset(result[0].keys()))
        self.assertIsNone(result[0]["page_number"])


class MockEncoder:
    def __init__(self):
        self.model_name = "mock"
        self.device = "cpu"
        self.embedding_dimension = 2

    def encode(self, texts, batch_size=8):
        # Candidate gets [1,0]; first vocabulary vector gets [1,0].
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(texts), 1))


class TestSemanticScoring(unittest.TestCase):
    def test_deterministic_mock_semantic_scorer(self):
        records = load_day1_vocabulary(DATASET)
        vocab = VocabularyIndex.build(records)
        embeddings = np.zeros((len(records), 2), dtype=np.float32)
        embeddings[0] = [1.0, 0.0]
        embeddings[1] = [0.0, 1.0]
        store = VocabularyEmbeddingStore(embeddings, records, {"vocabulary_record_count": len(records)})
        encoder = MockEncoder()

        detector = TechnicalTermDetector(
            DATASET,
            encoder=encoder,
            embedding_store=store,
            config={"semantic_similarity_threshold": 0.5, "semantic_margin_threshold": 0.1},
        )
        result = detector.detect_text(
            "A specialized evaluation technique was applied to the model.",
            use_semantic=True,
        )
        self.assertIsInstance(result, list)

    def test_f1(self):
        self.assertAlmostEqual(_f1(1.0, 1.0), 1.0)
        self.assertAlmostEqual(_f1(0.5, 0.5), 0.5)


class TestEvaluation(unittest.TestCase):
    def test_evaluation_metric_correctness(self):
        detector = TechnicalTermDetector(DATASET)
        fixture = {
            "sentence_id": "metric",
            "sentence": "Generative AI is used.",
            "expected": [{
                "canonical_term": "generative AI",
                "expected_category": "exact",
                "char_start": 0,
                "char_end": 13,
            }],
            "evaluation_enabled": True,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump([fixture], handle)
            path = handle.name
        try:
            report = evaluate_detections(detector, path, run_semantic=False)
            self.assertEqual(report["exact_normalized_term"]["precision"], 1.0)
            self.assertEqual(report["exact_normalized_term"]["recall"], 1.0)
            self.assertEqual(report["exact_normalized_term"]["f1"], 1.0)
        finally:
            Path(path).unlink(missing_ok=True)


class TestPDFExtraction(unittest.TestCase):
    def test_invalid_path(self):
        with self.assertRaises(FileNotFoundError):
            extract_pdf_pages("/definitely/not/a/real/file.pdf")

    def test_empty_pdf(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
            path = Path(handle.name)
        try:
            document = fitz.open()
            document.new_page()
            document.save(path)
            document.close()
            with self.assertRaises(ValueError):
                extract_pdf_pages(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
