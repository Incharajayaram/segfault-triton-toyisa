// machine.cpp — MachineState and instruction execution.
//
// Mirrors exec.py. The emulator is a small machine that consumes the
// emitted instruction stream and nothing else.

#include "machine.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace tritonflow::emu {

// ---- Helpers ------------------------------------------------------------ //

static const char* PROGRAM_ID_AXES[] = {"x", "y", "z"};

static bool is_memory_instruction(const std::string& name) {
    return name == "DMA1D" || name == "DMA2D" || name == "LDG" || name == "LDS2D";
}

static bool is_mac_instruction(const std::string& name) {
    return name == "MAC8" || name == "MAC16" || name == "OPU8" || name == "OPU32";
}

static bool is_elementwise_instruction(const std::string& name) {
    return name == "EPI" || name == "VPU" || name == "CLAMP";
}

/// Parse the k=v;... descriptor key into its fields.
static std::map<std::string, std::string> descriptor_fields(
    const std::optional<std::string>& key
) {
    std::map<std::string, std::string> fields;
    if (!key || key->empty()) return fields;

    std::istringstream ss(*key);
    std::string part;
    while (std::getline(ss, part, ';')) {
        auto eq = part.find('=');
        if (eq != std::string::npos) {
            std::string name = part.substr(0, eq);
            std::string value = part.substr(eq + 1);
            // Trim whitespace.
            auto ltrim = [](std::string& s) {
                s.erase(0, s.find_first_not_of(" \t"));
            };
            auto rtrim = [](std::string& s) {
                auto pos = s.find_last_not_of(" \t");
                if (pos != std::string::npos) s.erase(pos + 1);
            };
            ltrim(name); rtrim(name);
            ltrim(value); rtrim(value);
            fields[name] = value;
        }
    }
    return fields;
}

/// Parse sizes from a descriptor key like "sizes=[64, 64]".
static std::vector<int> parse_sizes(const std::optional<std::string>& key) {
    auto fields = descriptor_fields(key);
    auto it = fields.find("sizes");
    if (it == fields.end()) return {};

    std::string raw = it->second;
    // Strip brackets.
    if (!raw.empty() && raw.front() == '[') raw.erase(0, 1);
    if (!raw.empty() && raw.back() == ']') raw.pop_back();

    std::vector<int> result;
    std::istringstream ss(raw);
    std::string item;
    while (std::getline(ss, item, ',')) {
        auto ltrim = [](std::string& s) {
            s.erase(0, s.find_first_not_of(" \t"));
        };
        ltrim(item);
        if (!item.empty()) {
            result.push_back(std::stoi(item));
        }
    }
    return result;
}

/// Get the declared shape from an instruction's constrained_on field.
static std::vector<int> declared_shape(const Instr& instr) {
    return parse_sizes(instr.constrained_on);
}

/// Find which inputs are pointer (buffer) inputs by scanning MemRef operands.
static std::set<std::string> pointer_inputs(const Program& program) {
    std::set<std::string> names;
    for (const auto* instr : program.instructions()) {
        for (const auto& role : instr->roles) {
            auto it = instr->operands.find(role);
            if (it == instr->operands.end()) continue;
            if (auto* memref = std::get_if<MemRef>(&it->second)) {
                auto fields = descriptor_fields(memref->access_key);
                auto base_it = fields.find("base");
                names.insert(base_it != fields.end() ? base_it->second : memref->base);
            }
        }
    }
    return names;
}

/// Require an operand by role, throw if missing.
static const Operand& require_operand(const Instr& instr, const std::string& role) {
    const Operand* op = instr.operand(role);
    if (!op) {
        throw UnsupportedInstruction(
            instr.name + " is missing required role '" + role + "'");
    }
    return *op;
}

/// Get the mask operand if present.
static const Value* resolve_mask(const Instr& instr, const MachineState& state) {
    const Operand* mask_op = instr.operand("mask");
    if (!mask_op) return nullptr;
    // We store it in a thread_local to avoid dangling pointer issues.
    thread_local Value mask_val;
    mask_val = state.resolve(*mask_op);
    return &mask_val;
}

/// Get the "in*" operands in role order.
static std::vector<Value> input_operands(const Instr& instr, const MachineState& state) {
    std::vector<Value> result;
    for (const auto& role : instr.roles) {
        if (role.substr(0, 2) == "in") {
            result.push_back(state.resolve(instr.operands.at(role)));
        }
    }
    return result;
}

/// Convert int64_t indices from a Value.
static std::vector<int64_t> to_int64_vec(const Value& v) {
    std::vector<int64_t> result(v.data.size());
    for (size_t i = 0; i < v.data.size(); ++i) {
        result[i] = static_cast<int64_t>(v.data[i]);
    }
    return result;
}

// ---- MachineState ------------------------------------------------------- //

MachineState MachineState::from_program(
    const Program& program,
    const std::map<std::string, Value>& inputs,
    const PrecisionPolicy& pol,
    int gx, int gy, int gz
) {
    MachineState state;
    state.program = &program;
    state.policy = pol;
    state.grid[0] = gx;
    state.grid[1] = gy;
    state.grid[2] = gz;

    auto ptrs = pointer_inputs(program);

    // Build storage: lay out buffers contiguously in flat memory.
    int offset = 0;
    std::vector<const Value*> buffer_order;
    for (const auto& name : program.inputs) {
        if (ptrs.find(name) == ptrs.end()) continue;
        auto it = inputs.find(name);
        if (it == inputs.end()) {
            throw MissingInput(
                "pointer input " + name + " was not supplied; the kernel addresses "
                "it and the emulator will not fabricate storage for it");
        }
        Storage s;
        s.name = name;
        s.base = offset;
        s.length = static_cast<int>(it->second.data.size());
        s.shape = it->second.shape;
        state.storages[name] = s;
        buffer_order.push_back(&it->second);
        offset += s.length;
    }

    // Concatenate all buffers into flat memory.
    state.memory.resize(offset);
    int pos = 0;
    for (const auto* buf : buffer_order) {
        std::memcpy(state.memory.data() + pos, buf->data.data(),
                     buf->data.size() * sizeof(float));
        pos += static_cast<int>(buf->data.size());
    }

    // Bind inputs.
    for (const auto& name : program.inputs) {
        auto st_it = state.storages.find(name);
        if (st_it != state.storages.end()) {
            state.values[name] = Value::from_int(st_it->second.base);
        } else {
            auto in_it = inputs.find(name);
            if (in_it != inputs.end()) {
                state.values[name] = in_it->second;
            } else {
                // Check program ID axes.
                bool found = false;
                for (int ax = 0; ax < 3; ++ax) {
                    if (name == PROGRAM_ID_AXES[ax]) {
                        state.values[name] = Value::from_int(state.grid[ax]);
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    throw MissingInput(
                        "input " + name + " was not supplied and is not a "
                        "program-id axis; the emulator refuses to default a "
                        "value the kernel reads");
                }
            }
        }
    }

    return state;
}

Value MachineState::resolve(const Operand& operand) const {
    if (auto* imm = std::get_if<Imm>(&operand)) {
        if (imm->is_int()) {
            return Value::from_int(imm->as_int());
        }
        return Value::from_float(imm->as_double());
    }
    if (auto* ssa = std::get_if<SsaRef>(&operand)) {
        auto it = values.find(ssa->name);
        if (it != values.end()) {
            return it->second;
        }
        throw MissingInput(
            ssa->name + " is read but nothing produced it; a re-threaded "
            "loop value with no producer is an unexecutable program, not a zero");
    }
    if (auto* mem = std::get_if<MemRef>(&operand)) {
        auto it = values.find(mem->base);
        if (it != values.end()) {
            return it->second;
        }
        throw MissingInput(
            mem->base + " is read but nothing produced it");
    }
    throw std::runtime_error("cannot resolve operand");
}

void MachineState::bind(const std::string& name, Value value) {
    values[name] = std::move(value);
}

Value MachineState::gather(const Value& indices, const Value* mask) const {
    auto flat = to_int64_vec(indices);
    check_bounds(flat);

    std::vector<float> out(flat.size());
    for (size_t i = 0; i < flat.size(); ++i) {
        out[i] = memory[flat[i]];
    }

    if (mask) {
        for (size_t i = 0; i < flat.size() && i < mask->data.size(); ++i) {
            if (mask->data[i] == 0.0f) {
                out[i] = 0.0f;
            }
        }
    }
    return Value::from_array(std::move(out), indices.shape);
}

void MachineState::scatter(const Value& indices, const Value& vals, const Value* mask) {
    auto flat = to_int64_vec(indices);
    check_bounds(flat);

    for (size_t i = 0; i < flat.size(); ++i) {
        if (mask && i < mask->data.size() && mask->data[i] == 0.0f) {
            continue;
        }
        float v = (i < vals.data.size()) ? vals.data[i] : vals.data[0];
        memory[flat[i]] = v;
    }
    mark_written(flat);
}

void MachineState::check_bounds(const std::vector<int64_t>& flat) const {
    if (flat.empty()) return;
    int64_t lo = *std::min_element(flat.begin(), flat.end());
    int64_t hi = *std::max_element(flat.begin(), flat.end());
    if (lo < 0 || hi >= static_cast<int64_t>(memory.size())) {
        std::ostringstream oss;
        oss << "access out of emulated storage: index range ["
            << lo << ", " << hi << "] but memory has "
            << memory.size() << " elements; refusing to zero-fill";
        throw StorageError(oss.str());
    }
}

void MachineState::mark_written(const std::vector<int64_t>& flat) {
    if (flat.empty()) return;
    int64_t lo = *std::min_element(flat.begin(), flat.end());
    int64_t hi = *std::max_element(flat.begin(), flat.end());
    for (const auto& [name, storage] : storages) {
        if (lo < storage.end() && hi >= storage.base) {
            written.insert(name);
        }
    }
}

std::map<std::string, Value> MachineState::outputs() const {
    std::map<std::string, Value> out;
    for (const auto& name : written) {
        auto it = storages.find(name);
        if (it == storages.end()) continue;
        const Storage& s = it->second;
        std::vector<float> buf(memory.begin() + s.base,
                               memory.begin() + s.end());
        out[name] = Value::from_array(std::move(buf), s.shape);
    }
    return out;
}

// ---- Instruction dispatch ----------------------------------------------- //

// Forward declarations.
static void apply_memory(const Instr& instr, MachineState& state);
static void apply_mac(const Instr& instr, MachineState& state, const PrecisionPolicy& policy);
static void apply_elementwise(const Instr& instr, MachineState& state);

void apply(const Instr& instr, MachineState& state, const PrecisionPolicy& policy) {
    if (is_memory_instruction(instr.name)) {
        apply_memory(instr, state);
        return;
    }
    if (is_mac_instruction(instr.name)) {
        apply_mac(instr, state, policy);
        return;
    }
    if (is_elementwise_instruction(instr.name)) {
        apply_elementwise(instr, state);
        return;
    }
    throw UnsupportedInstruction(
        "instruction '" + instr.name + "' is not implemented by this machine; "
        "an unimplemented instruction is a refusal, not a no-op");
}

// ---- Memory instructions ------------------------------------------------ //

static void apply_memory(const Instr& instr, MachineState& state) {
    const Value* mask = resolve_mask(instr, state);

    // Store: dst is a MemRef.
    const Operand* dst_op = instr.operand("dst");
    if (dst_op && std::holds_alternative<MemRef>(*dst_op)) {
        Value indices = state.resolve(*dst_op);
        const Operand& val_op = require_operand(instr, "value");
        Value payload = state.resolve(val_op);
        state.scatter(indices, payload, mask);
        return;
    }

    // Load: src is a MemRef.
    const Operand& src_op = require_operand(instr, "src");
    if (!std::holds_alternative<MemRef>(src_op)) {
        throw UnsupportedInstruction(
            instr.name + " has neither a dst nor a src memory operand");
    }
    if (instr.defs.empty()) {
        throw UnsupportedInstruction(instr.name + " load defines no value to bind");
    }
    Value src_val = state.resolve(src_op);
    Value loaded = state.gather(src_val, mask);

    // Reshape to declared shape if available.
    auto shape = declared_shape(instr);
    if (!shape.empty()) {
        loaded.shape = shape;
    }
    state.bind(instr.defs[0], std::move(loaded));
}

// ---- MAC instructions --------------------------------------------------- //

static void apply_mac(const Instr& instr, MachineState& state, const PrecisionPolicy& policy) {
    Value a = state.resolve(require_operand(instr, "a"));
    Value b = state.resolve(require_operand(instr, "b"));
    Value acc_val = state.resolve(require_operand(instr, "acc"));

    // Determine dimensions from shapes.
    int m = a.shape.size() >= 2 ? a.shape[0] : static_cast<int>(a.data.size());
    int k = a.shape.size() >= 2 ? a.shape[1] : 1;
    int n = b.shape.size() >= 2 ? b.shape[1] : static_cast<int>(b.data.size());

    // Prepare accumulator.
    std::vector<float> acc(m * n, 0.0f);
    if (!acc_val.is_scalar()) {
        for (size_t i = 0; i < std::min(acc.size(), acc_val.data.size()); ++i) {
            acc[i] = acc_val.data[i];
        }
    }

    // multiply_accumulate.
    policy.multiply_accumulate(acc.data(), a.data.data(), m, k, b.data.data(), n);

    if (instr.defs.empty()) {
        throw UnsupportedInstruction(instr.name + " defines no accumulator value");
    }
    state.bind(instr.defs[0], Value::from_array(std::move(acc), {m, n}));
}

// ---- Elementwise instructions ------------------------------------------- //

// Elementwise op handlers.
using ElemHandler = std::function<Value(const Instr&, MachineState&, const std::vector<int>&)>;

static Value elem_const(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    Value v = state.resolve(require_operand(instr, "value"));
    if (shape.empty()) return v;
    size_t n = 1;
    for (int d : shape) n *= d;
    std::vector<float> data(n, v.scalar_f());
    return Value::from_array(std::move(data), shape, v.is_int);
}

static Value elem_program_id(const Instr& instr, MachineState& state, const std::vector<int>&) {
    Value v = state.resolve(require_operand(instr, "value"));
    int axis = static_cast<int>(v.scalar_i());
    if (axis >= 3) {
        throw UnsupportedInstruction("program id axis out of range");
    }
    return Value::from_int(state.grid[axis]);
}

static Value elem_make_range(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    Value v = state.resolve(require_operand(instr, "value"));
    int length = static_cast<int>(v.scalar_i());
    std::vector<float> data(length);
    for (int i = 0; i < length; ++i) data[i] = static_cast<float>(i);
    return Value::from_array(std::move(data), {length}, true);
}

static Value elem_splat(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    if (ops.empty()) throw UnsupportedInstruction("splat has no input");
    float val = ops[0].scalar_f();
    auto sh = shape.empty() ? std::vector<int>{1} : shape;
    size_t n = 1;
    for (int d : sh) n *= d;
    std::vector<float> data(n, val);
    return Value::from_array(std::move(data), sh, ops[0].is_int);
}

static Value elem_broadcast(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    if (ops.empty()) throw UnsupportedInstruction("broadcast has no input");
    const Value& src = ops[0];
    size_t n = 1;
    for (int d : shape) n *= d;

    // Simple broadcast: repeat the source data to fill the target shape.
    std::vector<float> data(n);
    if (src.data.empty()) {
        std::fill(data.begin(), data.end(), 0.0f);
    } else {
        for (size_t i = 0; i < n; ++i) {
            data[i] = src.data[i % src.data.size()];
        }
    }
    return Value::from_array(std::move(data), shape, src.is_int);
}

static Value elem_expand_dims(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    if (ops.empty()) throw UnsupportedInstruction("expand_dims has no input");
    if (shape.empty()) throw UnsupportedInstruction("expand_dims has no recorded result shape");
    size_t n = 1;
    for (int d : shape) n *= d;
    if (n != ops[0].data.size()) {
        throw UnsupportedInstruction(
            "expand_dims result shape cannot be that reshape");
    }
    return Value::from_array(
        std::vector<float>(ops[0].data), shape, ops[0].is_int);
}

// Binary ops.
static Value elem_addi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(a + b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_addf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float a = ops[0].data[i % ops[0].data.size()];
        float b = ops[1].data[i % ops[1].data.size()];
        out[i] = static_cast<float>(static_cast<float>(a) + static_cast<float>(b));
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, false);
}

static Value elem_muli(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(a * b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_divsi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(b != 0 ? a / b : 0);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_remsi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(b != 0 ? a % b : 0);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_maxnumf(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        float a = ops[0].data[i % ops[0].data.size()];
        float b = ops[1].data[i % ops[1].data.size()];
        out[i] = std::max(a, b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, false);
}

static Value elem_cmpi(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    // arith.cmpi, executed as signed less-than the corpus uses (F8).
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = (a < b) ? 1.0f : 0.0f;
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

static Value elem_addptr(const Instr& instr, MachineState& state, const std::vector<int>& shape) {
    auto ops = input_operands(instr, state);
    size_t n = std::max(ops[0].data.size(), ops[1].data.size());
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        int64_t a = static_cast<int64_t>(ops[0].data[i % ops[0].data.size()]);
        int64_t b = static_cast<int64_t>(ops[1].data[i % ops[1].data.size()]);
        out[i] = static_cast<float>(a + b);
    }
    auto sh = shape.empty() ? (ops[0].shape.empty() ? ops[1].shape : ops[0].shape) : shape;
    return Value::from_array(std::move(out), sh, true);
}

// Dispatch table.
static const std::map<std::string, ElemHandler> ELEMENTWISE_OPS = {
    {"arith.constant", elem_const},
    {"tt.get_program_id", elem_program_id},
    {"tt.make_range", elem_make_range},
    {"tt.splat", elem_splat},
    {"tt.broadcast", elem_broadcast},
    {"tt.expand_dims", elem_expand_dims},
    {"arith.addi", elem_addi},
    {"arith.addf", elem_addf},
    {"arith.muli", elem_muli},
    {"arith.divsi", elem_divsi},
    {"arith.remsi", elem_remsi},
    {"arith.maxnumf", elem_maxnumf},
    {"arith.cmpi", elem_cmpi},
    {"tt.addptr", elem_addptr},
};

static void apply_elementwise(const Instr& instr, MachineState& state) {
    std::string op;
    if (instr.source) {
        op = instr.source->op_name;
    }
    if (op.empty()) {
        throw UnsupportedInstruction(
            instr.name + " carries no source operation; the elementwise "
            "unit cannot know which arithmetic to perform");
    }

    auto shape = declared_shape(instr);

    auto it = ELEMENTWISE_OPS.find(op);
    if (it == ELEMENTWISE_OPS.end()) {
        throw UnsupportedInstruction(
            "elementwise operation '" + op + "' is not implemented by this machine; "
            "refusing rather than substituting");
    }

    if (instr.defs.empty()) {
        throw UnsupportedInstruction(op + " defines no value to bind");
    }
    Value result = it->second(instr, state, shape);
    state.bind(instr.defs[0], std::move(result));
}

// ---- Loop execution ----------------------------------------------------- //

void run_loop(const Loop& loop, MachineState& state, const PrecisionPolicy& policy) {
    if (!loop.yields.empty() &&
        loop.yields.size() != loop.iter_args.size()) {
        throw StorageError(
            "loop yields " + std::to_string(loop.yields.size()) +
            " value(s) for " + std::to_string(loop.iter_args.size()) +
            " iter_args; cannot re-thread");
    }
    if (loop.inits.size() != loop.iter_args.size()) {
        throw StorageError(
            "loop has " + std::to_string(loop.inits.size()) +
            " initialiser(s) for " + std::to_string(loop.iter_args.size()) +
            " iter_args; cannot enter the loop");
    }

    // Bind iter_args from initialisers.
    for (size_t i = 0; i < loop.iter_args.size(); ++i) {
        Value init_val = state.resolve(SsaRef{loop.inits[i]});
        state.bind(loop.iter_args[i], std::move(init_val));
    }

    // Resolve loop bounds.
    int lower = 0, upper = 0, step = 1;
    if (loop.lower) {
        lower = static_cast<int>(state.resolve(SsaRef{*loop.lower}).scalar_i());
    }
    if (loop.upper) {
        upper = static_cast<int>(state.resolve(SsaRef{*loop.upper}).scalar_i());
    }
    if (loop.step) {
        step = static_cast<int>(state.resolve(SsaRef{*loop.step}).scalar_i());
    }
    if (step == 0) {
        throw StorageError("loop has step 0; it would never terminate");
    }

    int index = lower;
    int guard = (upper - lower) / step + 2;

    while (index < upper) {
        --guard;
        if (guard < 0) {
            throw StorageError("loop did not terminate; refusing to spin");
        }

        if (loop.induction_var) {
            state.bind(*loop.induction_var, Value::from_int(index));
        }

        // Execute body.
        for (const auto& item : loop.body) {
            if (auto* instr = std::get_if<Instr>(&item)) {
                apply(*instr, state, policy);
            }
            // UnsupportedMarker in body: handled by the markers() check
            // before we enter execution.
        }

        // Re-thread iter_args from yields.
        if (!loop.yields.empty()) {
            std::vector<Value> advanced;
            for (const auto& name : loop.yields) {
                advanced.push_back(state.resolve(SsaRef{name}));
            }
            for (size_t i = 0; i < loop.iter_args.size(); ++i) {
                state.bind(loop.iter_args[i], std::move(advanced[i]));
            }
        }

        index += step;
    }

    // Bind results from final iter_args.
    for (size_t i = 0; i < loop.results.size() && i < loop.iter_args.size(); ++i) {
        auto it = state.values.find(loop.iter_args[i]);
        if (it != state.values.end()) {
            state.bind(loop.results[i], it->second);
        }
    }
}

// ---- Top-level emulate -------------------------------------------------- //

std::map<std::string, Value> emulate(
    const Program& program,
    const std::map<std::string, Value>& inputs,
    const PrecisionPolicy& policy,
    int gx, int gy, int gz
) {
    // Postcondition 2: UNSUPPORTED halts locally.
    auto markers = program.markers();
    if (!markers.empty()) {
        throw ProgramNotExecutable(
            markers[0]->op_name,
            markers[0]->reason,
            markers[0]->loc_name.value_or(""));
    }

    MachineState state = MachineState::from_program(program, inputs, policy, gx, gy, gz);

    for (const auto& item : program.items) {
        if (auto* loop = std::get_if<Loop>(&item)) {
            run_loop(*loop, state, policy);
        } else if (auto* instr = std::get_if<Instr>(&item)) {
            apply(*instr, state, policy);
        }
        // UnsupportedMarker already caught above.
    }

    return state.outputs();
}

}  // namespace tritonflow::emu
