"""triton_toyisa — lower Triton IR to a declaratively described tensor ISA.

This package deliberately imports **nothing** at package level. In particular it
must be importable with Triton and torch absent from `sys.modules`
(FR-003, FR-034); `qc.checks.pipeline.runs-without-triton-or-torch` is the
guard, and it asserts the property by importing the pipeline in a fresh
interpreter and inspecting `sys.modules` rather than by trusting this
paragraph.

Sub-packages, in pipeline order:

    ttir/           text -> RawModule (syntax) -> Module (semantics)
    canon/          idempotent canonicalisation, no reduction claims
    recognize/      access descriptors from the def-use graph
    idioms/         MAC and epilogue detection
    isa/            declarative schemas, predicate evaluation, selection
    emit/           instruction stream assembly, (de)serialisation
    emu/            NumPy execution of the emitted stream, precision policy
    torch_backend/  the PrivateUse1 device, DeviceInterface, backend registration
    report/         coverage, selection quality, cross-ISA transfer
    harness/        fixture extraction — the only module allowed to import Triton
"""

__version__ = "0.1.0"
