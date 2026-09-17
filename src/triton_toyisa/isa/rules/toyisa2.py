"""ISA-2 lowering rules: rule name → candidate instruction set.

Authored per T063's protocol: written against the *schema* and the recogniser's
three binding kinds (`memory` / `mac` / `elementwise` from
`recognize/op_shapes.py`) WITHOUT reading `isa/rules/toyisa1.py` — the fact is
recorded here and in the commit message, because the transfer report's
`edits_outside_schema_and_rules` claim is only meaningful if this file was not
copied from ISA-1's.

The rule NAMES are the recogniser's constants, so this table is structurally
identical to any ISA's: that is the pipeline's contract, not evidence of
copying. The candidate lists are toyisa2's own instruction names and are
checked against the schema by the ISA checks — a rule naming an instruction the
schema does not declare is a broken pipeline.
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


#: ISA-2's rule table. Candidates mirror `isa/schemas/toyisa2.yaml`.
ISA_RULES: tuple[Rule, ...] = (
    Rule(pattern="memory", candidates=("LDG", "LDS2D")),
    Rule(pattern="mac", candidates=("OPU32", "OPU8")),
    Rule(pattern="elementwise", candidates=("VPU", "CLAMP")),
)


def rule_for(kind: str) -> Rule | None:
    """The rule an annotation's `kind` lowers under, or `None` when unlowerable."""
    for rule in ISA_RULES:
        if rule.applies_to(kind):
            return rule
    return None
