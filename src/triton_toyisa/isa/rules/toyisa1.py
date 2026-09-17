"""ISA-1 lowering rules: rule name → candidate instruction set.

This is the only ISA-1-specific lowering logic in the pipeline (`plan.md` §6).
The three rule names are the recogniser's binding kinds (`recognize/op_shapes.py`
exports the same three constants), so the emitter needs no gluing layer: an
annotation's `kind` is the key into this table.

The rules are deliberately thin. Selection itself is `isa.select.select` over
the declarative schema; a rule exists to state what *rule name* an operation
class lowers under and which tile shape a MAC asks for — the facts that cannot
live in YAML because they come from the IR's op names, not from the ISA author.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ISA_RULES", "Rule", "rule_for"]


@dataclass(frozen=True)
class Rule:
    """One lowering rule: which binding kind maps to which schema rule name."""

    pattern: str  # the recogniser's binding kind ("memory" | "mac" | "elementwise")
    candidates: tuple[str, ...]  # instruction names, for documentation and reports

    def applies_to(self, kind: str) -> bool:
        return kind == self.pattern


#: ISA-1's rule table. The candidate lists mirror `isa/schemas/toyisa1.yaml` and
#: are checked against it by `qc/checks/isa.py` — a rule naming an instruction
#: the schema does not declare is a broken pipeline, not a runtime surprise.
ISA_RULES: tuple[Rule, ...] = (
    Rule(pattern="memory", candidates=("DMA1D", "DMA2D")),
    Rule(pattern="mac", candidates=("MAC8", "MAC16")),
    Rule(pattern="elementwise", candidates=("EPI",)),
)


def rule_for(kind: str) -> Rule | None:
    """The rule an annotation's `kind` lowers under, or `None` when unlowerable."""
    for rule in ISA_RULES:
        if rule.applies_to(kind):
            return rule
    return None
