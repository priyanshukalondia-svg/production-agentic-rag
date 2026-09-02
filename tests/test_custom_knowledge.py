from pathlib import Path

from production_agentic_rag import RAGPipeline, Settings
from production_agentic_rag.ingestion.loaders import load_directory

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"


def _default_pipe():
    pipe = RAGPipeline(Settings())
    pipe.add_documents(load_directory(CORPUS))
    return pipe


def test_default_mode_keeps_existing_handbook_behavior():
    pipe = _default_pipe()
    res = pipe.query("what is the return window?")
    assert "handbook:0" in res.citations
    assert "30 days" in res.answer.lower()
    assert pipe.custom_enabled is False


def test_custom_handbook_can_be_uploaded_and_indexed():
    pipe = RAGPipeline(Settings())
    added = pipe.add_custom_document("# Refund Policy\nCustomers can request a refund within 14 days.", "custom-policy")
    assert added > 0
    assert pipe.custom_knowledge["handbook"]
    pipe.enable_custom_knowledge(True)
    res = pipe.query("what is our refund policy?")
    assert any("custom" in citation for citation in res.citations)


def test_custom_qa_can_be_added_and_retrieved():
    pipe = RAGPipeline(Settings())
    pipe.add_custom_qa("What is our refund policy?", "Customers can request a refund within 14 days.")
    pipe.enable_custom_knowledge(True)
    res = pipe.query("What is our refund policy?")
    assert "14 days" in res.answer.lower()
    assert any("custom" in citation for citation in res.citations)


def test_custom_knowledge_takes_priority_over_default_when_conflicting():
    pipe = _default_pipe()
    pipe.add_custom_document("# Company Policy\nRefunds are available for 14 days only.", "conflict-policy")
    pipe.add_custom_qa("What is the return window?", "The return window is 14 days.")
    pipe.enable_custom_knowledge(True)
    res = pipe.query("What is the return window?")
    assert "14 days" in res.answer.lower()
    assert res.answer.lower().count("14") >= 1
    assert any("custom" in citation for citation in res.citations)


def test_default_handbook_is_used_as_fallback_when_custom_missing():
    pipe = _default_pipe()
    pipe.add_custom_document("# Warranty\nWe cover manufacturing defects for 2 years.", "custom-warranty")
    pipe.enable_custom_knowledge(True)
    res = pipe.query("what is the return window?")
    assert "30 days" in res.answer.lower()
    assert "handbook:0" in res.citations


def test_custom_knowledge_can_be_disabled_and_reenabled():
    pipe = _default_pipe()
    pipe.add_custom_document("# Refund Policy\nRefunds happen within 14 days.", "disable-policy")
    pipe.enable_custom_knowledge(True)
    assert pipe.custom_enabled is True
    pipe.enable_custom_knowledge(False)
    assert pipe.custom_enabled is False
    res = pipe.query("what is our refund policy?")
    assert "30 days" in res.answer.lower() or "refund" in res.answer.lower()


def test_custom_knowledge_can_be_deleted():
    pipe = RAGPipeline(Settings())
    pipe.add_custom_document("# Refund Policy\nRefunds happen within 14 days.", "delete-policy")
    assert pipe.custom_knowledge["handbook"]
    pipe.remove_custom_knowledge()
    assert pipe.custom_knowledge["handbook"] == []
    assert pipe.custom_knowledge["qa"] == []
    assert pipe.custom_enabled is False
