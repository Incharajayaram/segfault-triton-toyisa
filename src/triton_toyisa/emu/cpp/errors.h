// errors.h — Exception hierarchy for the C++ emulator.
//
// Mirrors the Python exception classes in exec.py exactly:
//   ProgramNotExecutable, UnsupportedInstruction, StorageError,
//   MissingInput, ShapeMismatch.
//
// Contract: contracts/emulator.md — failure modes table.

#pragma once

#include <stdexcept>
#include <string>

namespace toyisa::emu {

/// A program carrying an UNSUPPORTED marker (postcondition 2).
/// Halts locally: the seam routes this kernel to the eager fallback.
class ProgramNotExecutable : public std::runtime_error {
public:
    std::string op_name;
    std::string reason;
    std::string loc_name;

    ProgramNotExecutable(const std::string& op_name,
                         const std::string& reason,
                         const std::string& loc_name = "")
        : std::runtime_error(
              "program is not executable: UNSUPPORTED at " +
              (loc_name.empty() ? std::string("?") : loc_name) +
              " (" + op_name + "): " + reason),
          op_name(op_name),
          reason(reason),
          loc_name(loc_name) {}
};

/// An instruction name this machine does not implement.
class UnsupportedInstruction : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// A memory access outside the emulated storage — named, never zero-filled.
class StorageError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// An SSA value the program reads that no instruction produced and no input
/// supplied.
class MissingInput : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/// An input's shape disagrees with the descriptor that addresses it.
class ShapeMismatch : public std::invalid_argument {
public:
    std::string expected;
    std::string got;

    ShapeMismatch(const std::string& message,
                  const std::string& expected,
                  const std::string& got)
        : std::invalid_argument(message),
          expected(expected),
          got(got) {}
};

}  // namespace toyisa::emu
