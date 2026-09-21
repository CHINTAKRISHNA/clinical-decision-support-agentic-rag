from src.citation.citation_engine import has_unverifiable_claims, parse_citations, verify_citations


def test_parse_citations_extracts_claim_and_chunk_id():
    answer = "Patient has hypertension [note-01::chunk-0::a1b2c3d4]. No further findings."
    parsed = parse_citations(answer)
    assert parsed == [("Patient has hypertension", "note-01::chunk-0::a1b2c3d4")]


def test_verify_citations_flags_missing_chunk_id():
    answer = "Patient has hypertension [note-01::chunk-9::deadbeef]."
    records = verify_citations(answer, context_by_chunk_id={"note-01::chunk-0::a1b2c3d4": "hypertension history"})
    assert len(records) == 1
    assert records[0].exists_in_context is False
    assert has_unverifiable_claims(records) is True


def test_verify_citations_accepts_supported_claim():
    chunk_id = "note-01::chunk-0::a1b2c3d4"
    answer = f"Patient has a history of hypertension [{chunk_id}]."
    records = verify_citations(
        answer, context_by_chunk_id={chunk_id: "The patient has a history of essential hypertension."}
    )
    assert records[0].exists_in_context is True
    assert records[0].lexically_supported is True
    assert has_unverifiable_claims(records) is False
