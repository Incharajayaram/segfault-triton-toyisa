# SEGFAULT Demo Guide

**IICT CompilerTech Hackathon 2025**

## Quick Start

```bash
cd /home/bb/project/segfault

# 1. Compiler Lowering & ISA Diff Showcase (Interactive 5-Stage Demo)
python3 demo_diff.py

# 2. Automated Non-Stop Diff Showcase
python3 demo_diff.py --auto

# 3. Interactive Web Compiler Diff Explorer (Open in browser)
# open demo_diff.html in your browser or serve via:
# python3 -m http.server 8080 -> visit http://localhost:8080/demo_diff.html

# 4. Standard Multi-Stage Demo (20 mins)
python3 demo.py

# 5. Quick Automated Demo (2 mins)
bash demo_quick.sh
```

---

## Talking Points

### Opening (30 sec)
"We built a compiler that automatically generates backend code for custom AI accelerators. Instead of 6 months per chip, you write a YAML schema and get a working backend in 2 weeks."

### Demo 1: Multi-ISA (During execution)
"Same Triton IR input. Three different outputs. Zero manual intervention. Schema-driven means the YAML describes the ISA, compiler does the rest."

### Demo 2: Execution (During verification)
"Not just 'it compiles' - we execute and verify numerical correctness. Tolerance bounds are derived from precision models, not arbitrary."

### Demo 3: Cross-ISA (After comparison)
"Bit-exact outputs across all ISAs. That's true portability. Write once, run anywhere - for real."

### Demo 4: PyTorch (After compilation)
"Standard ML framework integration. torch.compile is PyTorch's official API. We're not reinventing - we're extending."

### Demo 5: Coverage (Show matrix)
"Notice t3_modulo is refused on all ISAs? That's our negative control - correctly detecting unsupported operations."

### Demo 6: Schema (Show YAML)
"This YAML file IS the ISA. Want to add a new accelerator? Write this file. That's it. The compiler reads from here."

### Closing (1 min)
"We've shown compilation, execution, verification, portability, and framework integration. All with 100% automated testing. Key innovation: schema-driven design makes compiler construction accessible to hardware teams."

---

## Q&A Responses

**Q: Does this work for any PyTorch model?**
A: "Proof-of-concept with recorded kernels. Architecture supports arbitrary models - needs more kernels or FlagGems integration (180+ ops). 2-3 weeks work."

**Q: Performance vs hand-written?**
A: "We optimize development speed. Cost model drives selection, so competitive. C++ emulator is 10x faster than NumPy baseline."

**Q: vs LLVM?**
A: "LLVM is general-purpose. We're specialized for AI accelerators with tiled memory. Our schemas capture domain knowledge. LLVM = 6 months per ISA, ours = 2 weeks."

**Q: Real hardware?**
A: "Emulator is for development. For deployment, replace emulator with hardware drivers. Compiled ISA code stays the same."

**Q: Proprietary ISAs?**
A: "That's the point! You control the schema. Describe proprietary ISAs without exposing details. Schema is your interface."

---

## Emergency Fallbacks

If live demo fails:
1. Show test results: `python3 -m pytest tests/ -v | tail -5`
2. Show reports: `head -30 COVERAGE_REPORT.md`
3. Explain architecture with diagrams
4. Show code walkthrough

Test suite passing = safety net.

---

## File Checklist

Before presenting:

```bash
cd /home/bb/project/segfault

# 1. Check imports
python3 -c "import sys; sys.path.insert(0, 'src'); from triton_tritonflow.lower import lower_fixture; print('✅')"

# 2. Check tests
python3 -m pytest tests/ -q | tail -3

# 3. Check verification
python3 verify/run_all.py | tail -1

# 4. Check PyTorch
python3 -c "import sys; sys.path.insert(0, 'src'); from triton_tritonflow.torch_backend.compiler import tritonflow_backend; print('✅')"

# 5. Make executable
chmod +x demo.py demo_quick.sh
```

All checks must pass ✅

---

## Timing (Total: 20 mins)

- Opening: 0:30
- Demo 1: 3:00
- Demo 2: 3:00
- Demo 3: 2:00
- Demo 4: 4:00
- Demo 5: 2:00
- Demo 6: 3:00
- Closing + Q&A: 2:30

Good luck! 🚀
