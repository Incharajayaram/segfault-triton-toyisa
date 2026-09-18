"""Contract tests for torch_backend using standard library unittest.

Validates the device memory model, alignment promises, and PyTorch seam.
"""

from __future__ import annotations

import unittest

import numpy as np

from tritonflow.torch_backend.device import (
    DeviceError,
    OutOfStorage,
    ToyDevice,
)

try:
    import torch

    from tritonflow.torch_backend.device_interface import (
        NotSupportedError,
        TritonFlowInterface,
        install,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestTorchSeamContract(unittest.TestCase):
    """Real contract tests for the device plane and PyTorch integration."""

    def test_toy_device_alignment_invariant(self) -> None:
        """Every allocation base must satisfy base % alignment_words == 0."""
        dev = ToyDevice(alignment_words=4, capacity_words=1024)

        # Allocate odd-sized buffer (3 elements)
        p1 = dev.allocate(3)
        self.assertEqual(p1.base % 4, 0)

        # Second allocation must be bumped to next multiple of 4
        p2 = dev.allocate(5)
        self.assertEqual(p2.base % 4, 0)
        self.assertGreaterEqual(p2.base, p1.base + p1.length)

        dev.assert_alignment()

    def test_toy_device_data_roundtrip(self) -> None:
        """Host to device and back copy must reproduce the input array exactly."""
        dev = ToyDevice(alignment_words=4)
        shape = (8, 8)
        p = dev.allocate(shape)

        data = np.arange(64, dtype=np.float32).reshape(shape)
        dev.copy_host_to_device(data, p)
        out = dev.copy_device_to_host(p)

        np.testing.assert_array_equal(data, out)
        dev.free(p)

    def test_toy_device_copy_shape_mismatch(self) -> None:
        """Copying mismatched size must raise DeviceError."""
        dev = ToyDevice()
        p = dev.allocate((4, 4))
        mismatched = np.zeros((3, 3), dtype=np.float32)
        with self.assertRaises(DeviceError):
            dev.copy_host_to_device(mismatched, p)

    def test_toy_device_out_of_storage(self) -> None:
        """Exhausting capacity must raise OutOfStorage."""
        dev = ToyDevice(capacity_words=16)
        with self.assertRaises(OutOfStorage):
            dev.allocate(32)

    def test_toy_device_reset(self) -> None:
        """Reset clears allocations and restores zero base."""
        dev = ToyDevice(capacity_words=64)
        dev.allocate(16)
        self.assertGreater(dev.allocated_words, 0)
        dev.reset()
        self.assertEqual(dev.allocated_words, 0)
        self.assertEqual(dev.live_regions, 0)

    def test_pytorch_device_interface(self) -> None:
        """When PyTorch is installed, verify DeviceInterface registration and properties."""
        if not HAS_TORCH:
            self.skipTest("torch is not installed in the environment (seam extra required)")

        self.assertTrue(TritonFlowInterface.is_available())
        self.assertEqual(TritonFlowInterface.device_count(), 1)
        self.assertEqual(TritonFlowInterface.current_device(), 0)

        props = TritonFlowInterface.get_device_properties(0)
        self.assertIn("tritonflow", props.name)

        # Synchronous device raises on asynchronous stream operations
        with self.assertRaises(NotSupportedError):
            TritonFlowInterface.current_stream()


if __name__ == "__main__":
    unittest.main()
