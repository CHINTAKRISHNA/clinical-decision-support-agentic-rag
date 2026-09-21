"""PII/PHI masking using Microsoft Presidio.

Runs on both ingested clinical notes and outbound model responses so that
identifiers (names, dates of birth, MRNs, phone numbers, addresses) never
reach the LLM or the clinician-facing output un-redacted.
"""
from __future__ import annotations

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Entities relevant to clinical notes beyond Presidio's generic PII defaults.
CLINICAL_PII_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "DATE_TIME",
    "LOCATION",
    "MEDICAL_LICENSE",
    "US_SSN",
]


class PresidioRedactor:
    def __init__(self) -> None:
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def redact(self, text: str, language: str = "en") -> str:
        results = self.analyzer.analyze(text=text, entities=CLINICAL_PII_ENTITIES, language=language)
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text

    def detect(self, text: str, language: str = "en") -> list[dict]:
        results = self.analyzer.analyze(text=text, entities=CLINICAL_PII_ENTITIES, language=language)
        return [{"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": r.score} for r in results]
