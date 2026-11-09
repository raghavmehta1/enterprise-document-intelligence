from pathlib import Path

from document_intelligence import DocumentIntelligenceModel


DOCS = Path(__file__).parents[1] / "data" / "documents"


def test_returns_grounded_answer_and_citation():
    result = DocumentIntelligenceModel().fit_directory(DOCS).ask(
        "How quickly must a security incident be reported?"
    )
    assert not result.insufficient_evidence
    assert "within one hour" in result.answer
    assert result.citations[0].source == "security_policy.txt"
    assert result.citations[0].quote in result.answer


def test_abstains_when_documents_do_not_support_answer():
    result = DocumentIntelligenceModel(threshold=0.18).fit_directory(DOCS).ask(
        "What is the cafeteria lunch menu?"
    )
    assert result.insufficient_evidence
    assert result.citations == ()

