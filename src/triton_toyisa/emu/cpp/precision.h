// precision.h — Precision policy and derived tolerance.
//
// Mirrors precision.py exactly. Contract: contracts/emulator.md FR-022.
//
// Three facts drive this module:
// 1. tf32 is fp32 with a shorter mantissa (10 explicit bits vs 23).
// 2. The error is bounded, so a tolerance can be derived.
// 3. The accumulation order is declared, not chosen here.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace toyisa::emu {

// ---- Constants ---------------------------------------------------------- //

constexpr int TF32_EXPLICIT_MANTISSA_BITS = 10;
constexpr int FP32_EXPLICIT_MANTISSA_BITS = 23;

constexpr double TF32_HALF_ULP = 1.0 / (1 << (TF32_EXPLICIT_MANTISSA_BITS + 1));  // 2^-11
constexpr double FP32_HALF_ULP = 1.0 / (1 << (FP32_EXPLICIT_MANTISSA_BITS + 1));  // 2^-24

// ---- InputPrecision ----------------------------------------------------- //

enum class InputPrecision {
    TF32,
    IEEE,
};

inline InputPrecision parse_input_precision(const std::string& s) {
    if (s == "tf32") return InputPrecision::TF32;
    if (s == "ieee") return InputPrecision::IEEE;
    throw std::invalid_argument(
        "input_precision must be 'tf32' or 'ieee', got '" + s + "'");
}

inline const char* input_precision_str(InputPrecision p) {
    return p == InputPrecision::TF32 ? "tf32" : "ieee";
}

// ---- Tolerance ---------------------------------------------------------- //

/// A derived tolerance, with the derivation that produced it.
/// `value` is a relative bound unless `absolute` is set.
struct Tolerance {
    double value = 0.0;
    std::string derivation;
    bool absolute = false;

    /// The absolute error this tolerance admits for `reference`.
    double bound(const float* reference, size_t n) const {
        if (absolute) return value;
        double peak = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double v = std::abs(static_cast<double>(reference[i]));
            if (v > peak) peak = v;
        }
        return value * std::max(1.0, peak);
    }
};

// ---- Comparison result -------------------------------------------------- //

/// What a differential comparison observed, derivation included.
struct Comparison {
    bool ok = false;
    double max_error = 0.0;
    double bound_value = 0.0;
    Tolerance tolerance;
    std::string derivation;
};

// ---- tf32_truncate ------------------------------------------------------ //

/// Round values to tf32 precision, round-half-to-even.
/// Implemented on the raw bits because that is what the hardware does.
/// Non-finite values pass through untouched.
inline void tf32_truncate(float* out, const float* in, size_t n) {
    constexpr int dropped = FP32_EXPLICIT_MANTISSA_BITS - TF32_EXPLICIT_MANTISSA_BITS;  // 13
    constexpr uint32_t mask = (1u << dropped) - 1u;

    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &in[i], sizeof(uint32_t));

        // Check for non-finite (exponent all 1s).
        if (((bits >> 23) & 0xFF) == 0xFF) {
            out[i] = in[i];
            continue;
        }

        // Round-half-to-even: add (half - 1) + the kept LSB, then mask.
        uint32_t lsb = (bits >> dropped) & 1u;
        uint32_t rounding = (1u << (dropped - 1)) - 1u + lsb;
        uint32_t rounded = (bits + rounding) & ~mask;

        std::memcpy(&out[i], &rounded, sizeof(float));
    }
}

/// In-place tf32 truncation.
inline void tf32_truncate_inplace(float* data, size_t n) {
    tf32_truncate(data, data, n);
}

// ---- derive_tolerance --------------------------------------------------- //

/// The tolerance for one reduction, and the formula behind it.
inline Tolerance derive_tolerance(
    InputPrecision precision,
    int reduction_length,
    const std::string& dtype = "f32"
) {
    if (reduction_length < 0) {
        throw std::invalid_argument(
            "reduction_length must be non-negative, got " +
            std::to_string(reduction_length));
    }

    // Integer paths are exact.
    if (dtype.find("int") != std::string::npos ||
        dtype == "i8" || dtype == "i32" || dtype == "i64" ||
        dtype == "u8" || dtype == "u32") {
        return Tolerance{
            0.0,
            "integer path (" + dtype + "): exact, tolerance is 0 by contract postcondition 5",
            true
        };
    }

    if (precision == InputPrecision::IEEE) {
        double val = reduction_length * FP32_HALF_ULP;
        std::ostringstream oss;
        oss << "ieee fp32: " << reduction_length
            << " sequential accumulation steps, each bounded by the fp32 half-ulp 2^-"
            << (FP32_EXPLICIT_MANTISSA_BITS + 1)
            << " = " << FP32_HALF_ULP
            << "; bound = " << reduction_length << " * " << FP32_HALF_ULP
            << " = " << val;
        return Tolerance{val, oss.str(), false};
    }

    // TF32 path.
    double truncation = 2.0 * TF32_HALF_ULP;
    double accumulation = FP32_HALF_ULP;
    double val = reduction_length * (truncation + accumulation);
    std::ostringstream oss;
    oss << "tf32 inputs truncated to " << TF32_EXPLICIT_MANTISSA_BITS
        << " explicit mantissa bits (" << FP32_EXPLICIT_MANTISSA_BITS
        << " dropped): each input off by at most the tf32 half-ulp 2^-"
        << (TF32_EXPLICIT_MANTISSA_BITS + 1) << " = " << TF32_HALF_ULP
        << ", so each product by at most 2*" << TF32_HALF_ULP << " = " << truncation
        << "; accumulated over " << reduction_length
        << " products with fp32 step error " << accumulation
        << ": bound = " << reduction_length << " * " << (truncation + accumulation)
        << " = " << val;
    return Tolerance{val, oss.str(), false};
}

// ---- accumulate --------------------------------------------------------- //

/// Sum terms along the reduction axis in declared order, in fp32.
/// k_major_sequential: walk in increasing index order, adding one fp32 term
/// at a time. Deliberately a loop and not a vectorised sum (postcondition 4).
inline void accumulate(
    float* out,
    const float* terms,
    int axis_len,
    int outer_size,
    const std::string& order = "k_major_sequential"
) {
    if (order != "k_major_sequential") {
        throw std::invalid_argument(
            "accumulation order '" + order + "' is not implemented; "
            "the emulator refuses to reinterpret a declared order it does not "
            "perform (known: k_major_sequential)");
    }

    // terms is [axis_len, outer_size], output is [outer_size].
    for (int j = 0; j < outer_size; ++j) {
        out[j] = 0.0f;
    }
    if (axis_len == 0) return;

    // First slice.
    for (int j = 0; j < outer_size; ++j) {
        out[j] = terms[j];  // terms[0 * outer_size + j]
    }

    // Subsequent slices, sequential accumulation in fp32.
    for (int k = 1; k < axis_len; ++k) {
        for (int j = 0; j < outer_size; ++j) {
            out[j] = static_cast<float>(
                static_cast<float>(out[j]) +
                static_cast<float>(terms[k * outer_size + j])
            );
        }
    }
}

// ---- PrecisionPolicy ---------------------------------------------------- //

/// The precision contract one program is executed under.
class PrecisionPolicy {
public:
    InputPrecision input_precision = InputPrecision::IEEE;
    std::string accumulation_order = "k_major_sequential";
    int reduction_length = 0;

    PrecisionPolicy() = default;

    PrecisionPolicy(InputPrecision prec, const std::string& order, int red_len)
        : input_precision(prec),
          accumulation_order(order),
          reduction_length(red_len) {}

    /// Apply the declared input precision to a multiply operand (in-place).
    void truncate(float* data, size_t n) const {
        if (input_precision == InputPrecision::TF32) {
            tf32_truncate_inplace(data, n);
        }
        // IEEE: no truncation needed, values are already fp32.
    }

    /// The tolerance for this policy's declared reduction length.
    Tolerance tolerance_for(const std::string& dtype = "f32") const {
        return derive_tolerance(input_precision, reduction_length, dtype);
    }

    /// acc += truncate(a) @ truncate(b) under the declared order.
    /// Inputs are truncated before multiplying (postcondition 3), and
    /// the reduction is walked in order (postcondition 4).
    ///
    /// a: [m, k], b: [k, n], acc: [m, n] — all row-major.
    void multiply_accumulate(
        float* acc,
        const float* a, int m, int k_dim,
        const float* b, int n
    ) const {
        // Create truncated copies.
        std::vector<float> left(a, a + m * k_dim);
        std::vector<float> right(b, b + k_dim * n);
        truncate(left.data(), left.size());
        truncate(right.data(), right.size());

        // Sequential k-major accumulation (postcondition 4).
        for (int ki = 0; ki < k_dim; ++ki) {
            for (int mi = 0; mi < m; ++mi) {
                for (int ni = 0; ni < n; ++ni) {
                    acc[mi * n + ni] = static_cast<float>(
                        static_cast<float>(acc[mi * n + ni]) +
                        static_cast<float>(left[mi * k_dim + ki] * right[ki * n + ni])
                    );
                }
            }
        }
    }

    /// The differential check contracts/emulator.md requires.
    Comparison compare(
        const float* actual,
        const float* reference,
        size_t n,
        const std::string& dtype = "f32"
    ) const {
        Tolerance tol = tolerance_for(dtype);

        double max_err = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double diff = std::abs(
                static_cast<double>(actual[i]) - static_cast<double>(reference[i])
            );
            if (diff > max_err) max_err = diff;
        }

        double bnd = tol.bound(reference, n);
        return Comparison{
            max_err <= bnd,
            max_err,
            bnd,
            tol,
            tol.derivation,
        };
    }
};

}  // namespace toyisa::emu
