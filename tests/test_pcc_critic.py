import pytest
from pcc_critic import audit_response, inject_pcc, PCC_PRESETS

def test_audit_response_no_op_short():
    result = audit_response("Too short")
    assert result["verdict"] == "NO_OP"
    assert result["sycophancy"] == 0.0
    assert result["evidence_count"] == 0
    assert result["words"] == 0

def test_audit_response_no_op_few_words():
    # Long enough text (>20 chars), but few words (<50) and no markers
    text = "This is a slightly longer text but not enough words to be anything else."
    result = audit_response(text)
    assert result["verdict"] == "NO_OP"
    assert result["words"] < 50

def test_audit_response_sycophantic():
    # Need >0.5 sycophancy, which is > 1.5 syc markers. So 2 markers.
    text = "Great question! That's a great idea! " + "pad " * 50
    result = audit_response(text)
    assert result["verdict"] == "SYCOPHANTIC"
    assert result["sycophancy"] > 0.5

def test_audit_response_pass():
    # Need >= 3 ev_markers
    text = "There is a risk, a weakness, and a problem. " + "pad " * 50
    result = audit_response(text)
    assert result["verdict"] == "PASS"
    assert result["evidence_count"] >= 3

def test_audit_response_review():
    # Need >= 1 and < 3 ev_markers
    text = "There is a risk. " + "pad " * 50
    result = audit_response(text)
    assert result["verdict"] == "REVIEW"
    assert result["evidence_count"] == 1

def test_audit_response_needs_evidence():
    # Need >= 50 words, 0 ev_markers, syc <= 0.5
    text = "word " * 60
    result = audit_response(text)
    assert result["verdict"] == "NEEDS_EVIDENCE"
    assert result["words"] >= 50
    assert result["evidence_count"] == 0

def test_inject_pcc_valid_preset():
    prompt = "Review this code"
    preset = "極"
    result = inject_pcc(prompt, preset)
    assert "PCC Protocol: #極 (Maximum Precision)" in result
    assert "Be extremely concise" in result
    assert prompt in result

def test_inject_pcc_unknown_preset():
    prompt = "Review this code"
    preset = "unknown_preset"
    result = inject_pcc(prompt, preset)
    # Should fallback to 探
    assert "PCC Protocol: #探 (Critical Explorer)" in result
    assert prompt in result
