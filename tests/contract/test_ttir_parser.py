import os

import pytest

from triton_toyisa.ttir.parser import parse_raw

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")

@pytest.mark.parametrize("fixture_name", [
    "t0_vecadd.ttir",
    "t1_matmul.ttir",
    "t2_matmul_relu.ttir",
    "t3_modulo.ttir",
    "fuzz.ttir"
])
def test_parser_happy_path(fixture_name):
    filepath = os.path.join(FIXTURES_DIR, fixture_name)
    if not os.path.exists(filepath):
        pytest.skip(f"Fixture {fixture_name} not found. Run extraction first.")
        
    with open(filepath) as f:
        text = f.read()
        
    # The parser must never crash on any input
    raw_module = parse_raw(text, source_path=filepath)
    
    # If the parser encountered any syntactic issues, diagnostics will be populated
    assert len(raw_module.diagnostics) == 0, f"Parser encountered errors: {raw_module.diagnostics}"
    
    # It should have successfully parsed some operations
    assert len(raw_module.ops) > 0, "No operations parsed"
    
    # Check that we parsed the loc_table
    assert len(raw_module.loc_table) > 0, "No loc table parsed"
