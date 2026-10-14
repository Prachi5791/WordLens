import unittest

from scripts.build_technical_dataset import (
    ALLOWED_SUBDOMAINS,
    normalize_term,
    stable_id,
    validate_record,
)


class TestNormalization(unittest.TestCase):

    def test_unicode_and_whitespace(self):
        self.assertEqual(
            normalize_term(
                "  Large  Language–Model  "
            ),
            "large language-model",
        )

    def test_case_normalization(self):
        self.assertEqual(
            normalize_term("RAG"),
            "rag",
        )


class TestStableId(unittest.TestCase):

    def test_deterministic(self):
        first = stable_id("large language model")
        second = stable_id("large language model")

        self.assertEqual(
            first,
            second,
        )


class TestValidation(unittest.TestCase):

    def test_valid_record(self):

        record = {
            "term_id": "abc123",
            "term": "RAG",
            "normalized_term": "rag",
            "aliases": [],
            "domain": "generative_ai",
            "subdomain": "retrieval_agents",
            "definition": (
                "Retrieval-augmented generation "
                "combines retrieval of relevant "
                "information with generation."
            ),
            "simplified_definition": "",
            "simplified_definition_status":
                "not_generated",
            "source":
                "Google Cloud Generative AI Glossary",
            "source_url":
                "https://cloud.google.com/docs/generative-ai/glossary",
            "source_definition":
                "Retrieval-augmented generation combines retrieval of relevant information with generation.",
            "source_retrieved_at":
                "2026-09-23T00:00:00+00:00",
            "confidence": 0.9,
            "confidence_reason":
                "authoritative source; valid subdomain",
            "quality_status":
                "accepted",
            "quality_notes":
                "validated",
            "is_active":
                True,
        }

        errors = validate_record(
            record
        )

        self.assertEqual(
            errors,
            [],
        )

    def test_invalid_subdomain(self):

        record = {
            "term_id": "abc123",
            "term": "RAG",
            "normalized_term": "rag",
            "aliases": [],
            "domain": "generative_ai",
            "subdomain": "invalid",
            "definition": (
                "A sufficiently long definition "
                "for testing validation behavior."
            ),
            "simplified_definition": "",
            "simplified_definition_status":
                "not_generated",
            "source":
                "Google Cloud Generative AI Glossary",
            "source_url":
                "https://cloud.google.com/docs/generative-ai/glossary",
            "source_definition":
                "A sufficiently long source definition.",
            "source_retrieved_at":
                "2026-09-23T00:00:00+00:00",
            "confidence": 0.9,
            "confidence_reason":
                "test",
            "quality_status":
                "accepted",
            "quality_notes":
                "test",
            "is_active":
                True,
        }

        errors = validate_record(
            record
        )

        self.assertIn(
            "invalid subdomain",
            errors,
        )


if __name__ == "__main__":
    unittest.main()