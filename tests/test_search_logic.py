from scripts.dynamic_knowledge import (
    _extract_abbreviation_expansions,
    _lookup_variants,
    _is_acronym,
    validate_contextual_usage,
)


def test_acronym_detection():
    assert _is_acronym("HDFS")
    assert _is_acronym("BERT")
    assert not _is_acronym("Kafka")


def test_context_expansion():
    context = "The Hadoop Distributed File System (HDFS) stores files across nodes."
    assert _extract_abbreviation_expansions("HDFS", context) == ["Hadoop Distributed File System"]


def test_variants():
    assert "vector" in _lookup_variants("vectors")
    assert "self attention" in _lookup_variants("self-attention")


def test_general_language_context_rejected():
    result = validate_contextual_usage(
        "ordinary",
        "Ordinary words such as model and system are not necessarily technical terms.",
        "Ordinary is a general word.",
        "Wikipedia",
        0.78,
    )
    assert result is not None
    assert result["reason_code"] == "GENERAL_LANGUAGE_CONTEXT"
