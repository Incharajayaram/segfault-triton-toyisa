from triton_toyisa.ttir.parser import RawModule
from triton_toyisa.ttir.ssa import Module


def build_ir(raw: RawModule, *, raise_on_invalid: bool = False) -> Module:
    """
    Consumer of RawModule. Converts syntactic strings into semantic SsaValue 
    and TypeExpr objects, checking arity, types, and resolving Loc names.
    """
    # Friend will implement the value numbering and graph building here!
    raise NotImplementedError("Track B needs to implement this semantic pass!")
