import pytest
from ms_server import extract_pcc_coord

def test_extract_pcc_coord_exact_match():
    text = "I1F2C3B4R5M6E7N8S9"
    assert extract_pcc_coord(text) == "I1F2C3B4R5M6E7N8S9"

def test_extract_pcc_coord_case_insensitive():
    text = "i1f2c3b4r5m6e7n8s9"
    assert extract_pcc_coord(text) == "I1F2C3B4R5M6E7N8S9"

def test_extract_pcc_coord_embedded():
    text = "The coordinates are I1F2C3B4R5M6E7N8S9 in the text."
    assert extract_pcc_coord(text) == "I1F2C3B4R5M6E7N8S9"

def test_extract_pcc_coord_no_match_returns_default():
    text = "No coordinates here."
    assert extract_pcc_coord(text) == "I5F5C5B5R5M5E5N5S5"

def test_extract_pcc_coord_empty_string():
    assert extract_pcc_coord("") == "I5F5C5B5R5M5E5N5S5"

def test_extract_pcc_coord_malformed():
    text = "I1F2C3B4R5M6E7N8" # Missing S9
    assert extract_pcc_coord(text) == "I5F5C5B5R5M5E5N5S5"

def test_extract_pcc_coord_multiple_matches_first_one():
    text = "First I1F2C3B4R5M6E7N8S9 then I9F8C7B6R5M4E3N2S1"
    assert extract_pcc_coord(text) == "I1F2C3B4R5M6E7N8S9"
