import pytest

from triton_toyisa.ttir.parser import RawLoc, RawModule, RawOp
from triton_toyisa.ttir.ssa import SsaValue
from triton_toyisa.ttir.to_ir import build_ir


# The "Two Input Classes" rule: every test runs on 'handbuilt' dummy data 
# until the parser is ready, at which point it runs on both 'handbuilt' and 'real'.
@pytest.mark.parametrize("input_class", ["handbuilt", "real"])
def test_build_ir_basic_structure(input_class):
    if input_class == "real":
        # We xfail the real test because Track A (parser) isn't finished yet
        pytest.xfail("Parser not yet integrated")
        return
        
    # Track B writes their tests against a HANDBUILT dummy parser output!
    dummy_raw_module = RawModule(
        ops=[
            RawOp(
                name="arith.constant",
                results=["%c64_i32"],
                operands=[],
                attrs={"value": "64"},
                result_types=["i32"],
                operand_types=[],
                regions=[],
                loc=RawLoc(name="#loc1"),
                line=10,
                col=4
            )
        ],
        loc_table={"#loc1": RawLoc(name="#loc1", line=10, col=4)},
        source_path="<dummy>",
        triton_version="3.7.1",
        diagnostics=[]
    )
    
    # Run the core logic against the dummy input
    # Track B can now make this test pass without waiting for Track A!
    result = build_ir(dummy_raw_module)
    assert result.ok, f"handbuilt input refused: {result.diagnostic}"
    module = result.unwrap()
    (op,) = module.body.blocks[0].operations
    assert op.name == "arith.constant"
    assert isinstance(op.results[0], SsaValue)
