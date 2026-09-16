from triton_toyisa.ttir.parser import parse_raw


def test_empty_input():
    # EC-001
    module = parse_raw("")
    assert not module.diagnostics

def test_comments_only():
    # EC-002
    module = parse_raw("// just a comment\n// another")
    assert not module.diagnostics

def test_unterminated_attribute_dict():
    # EC-011 vs EC-025
    text = "module { \n %0 = arith.constant 64 {value = 64 \n }"
    module = parse_raw(text)
    assert len(module.diagnostics) > 0
    assert module.diagnostics[0].layer == "syntax"

def test_truncated_module():
    # EC-025
    text = "module { "
    module = parse_raw(text)
    assert len(module.diagnostics) > 0
    assert "EOF" in str(module.diagnostics[0].found)

def test_deep_nesting():
    # EC-027
    text = "module { \n"
    for _i in range(250):
        text += "  scf.if %cond {\n"
    text += "    tt.return\n"
    for _i in range(250):
        text += "  }\n"
    text += "}\n"
    
    module = parse_raw(text)
    assert len(module.diagnostics) > 0
    assert module.diagnostics[0].layer == "syntax"

def test_crlf_bom_handling():
    # EC-019, EC-020
    text = "\xef\xbb\xbfmodule {\r\n  tt.return\r\n}\r\n"
    module = parse_raw(text)
    # The lexer should handle or strip BOM and CRLF gracefully
    assert not module.diagnostics

def test_minified_single_line():
    # EC-021
    text = 'module{%0=arith.constant 64:i32 tt.return}'
    module = parse_raw(text)
    # Shouldn't crash, might have diagnostics depending on strictness
    assert module is not None

def test_unknown_token():
    text = "module { \n %0 = @#$%^ \n }"
    module = parse_raw(text)
    assert len(module.diagnostics) > 0
    assert module.diagnostics[0].layer == "syntax"
