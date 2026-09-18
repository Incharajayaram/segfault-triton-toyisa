# Contract: PyTorch seam

**Module**: `src/triton_tritonflow/torch_backend/` (`compiler.py`, `device_interface.py`, `device.py`)
**Consumers**: `torch.compile`, PyTorch's device machinery. **Requirements**: FR-023 … FR-026, SC-001.
**Reference pattern** (structure, not content): `test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/compiler.py`.

## Interface

```python
# compiler.py
from torch._dynamo.backends.registry import register_backend

@register_backend
def tritonflow_backend(gm, example_inputs) -> Callable: ...

# device_interface.py
from torch._dynamo.device_interface import DeviceInterface, register_interface_for_device

@register_interface_for_device("tritonflow")
class ToyIsaInterface(DeviceInterface):
    # device slots: current_device, set_device, device_count, is_available, stream,
    #               current_stream, set_stream, synchronize, get_device_properties, ...
    class Event: ...
    class Stream: ...
    class Worker: ...

# device.py
def allocate(nbytes: int) -> DevicePtr
def copy_host_to_device(src, dst) -> None
def copy_device_to_host(src, dst) -> None
def synchronize() -> None
```

`DeviceInterface` spans **30 method slots across 4 nested classes** (`device`, `Event`, `Stream`, `Worker`);
most have usable defaults. The ones not meaningfully implementable are explicitly delegated or raise
`NotImplementedError` **with the reason recorded in the limitations document** — silent stubs that pretend to
work are a contract violation.

## Preconditions

- PyTorch is importable. The seam is the only part of the project that requires it (the pipeline itself runs
  from frozen fixtures without PyTorch or Triton installed).

## Postconditions

1. **Day-1 smoke test (FR-026).** With a hard-coded lowering for one kernel, a real `torch` operation compiles
   through `tritonflow_backend` and produces a correct result. This exists before the general pipeline is wired to
   the seam, so the integration risk is retired on day 1 rather than day 6.
2. **The registered device is visible (SC-001).** `device_count() >= 1`, `is_available()` is `True`,
   `current_device()`/`set_device()` round-trip, and `torch.tensor(...).to("tritonflow")` succeeds.
3. **Real lowering path.** Once wired, the backend consumes the `ttir` that Dynamo/Inductor produced, runs the
   pipeline, and executes the emitted program on the emulator. The emulator is the *device emulator* behind the
   seam — the role `libopenreg.so` plays — not a substitute for the seam.
4. **Fallback is explicit (FR-025).** An operation the pipeline cannot lower is not silently mis-executed: it
   falls back to eager PyTorch, and the fallback is recorded in `FallbackRecord` and surfaced in the coverage
   report. The eager floor is acknowledged in writing.
5. **No harness left behind.** Anything that bypasses a PyTorch interface to produce a result (a bespoke
   runner that never registers) is rejected in review (Constitution Principle IV).

## Failure modes

| Case | Required behaviour |
|---|---|
| `torch.compile` produces IR the parser cannot consume | `PARSE_UNSUPPORTED` recorded; the op runs via eager fallback; the compilation does not raise |
| Graph break in the compiled function | PyTorch handles it; the backend records the break in the coverage report rather than pretending full coverage |
| Unsupported dtype on the toy device | clear error naming the dtype; no silent upcast |
| Emulator and eager disagree beyond tolerance | **test failure**, not a warning. The tolerance is derived, not chosen (see `emulator.md`) |

## Tests

- **Integration** (`tests/integration/test_torch_seam.py`): 64×64 matmul through `torch.compile` matches eager
  within tolerance (SC-001). Vector add, matmul+relu, and a kernel with an unlowerable op (asserting fallback).
- **Contract**: device-visible assertions; `DeviceInterface` slot inventory test that fails when a new slot is
  added upstream without an explicit decision (the list is asserted, so an upstream addition is noticed).
- **Negative**: an op outside the corpus must produce a fallback record and a correct result, never a wrong one.
