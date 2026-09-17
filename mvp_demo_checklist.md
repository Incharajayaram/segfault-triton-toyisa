# Minimum Viable Prototype Checklist

**Purpose**: A streamlined version of the full requirements checklist, focusing strictly on the critical paths needed to prove the core technical bets and deliver a functional prototype demo.

## Core Functionality & Contracts
- [ ] **CHK002** Every functional requirement names an observable behaviour, not an internal mechanism.
- [ ] **CHK004** The eager fallback floor is acknowledged as a requirement, not left implicit.
- [ ] **CHK005** The ISA description format is specified as a contract (fields, constraint language, mandatory accumulator semantics) rather than as an example.
- [ ] **CHK022** The data model and the contracts agree on field names (`strides` vs `stride`, `offsets` present, `shape` = wraparound boundary).
- [ ] **CHK035** Principle IV: the seam is the artifact of record and is proven before the pipeline is wired to it.

## Clear, Measurable Success
- [ ] **CHK010** Every success criterion is measurable from an artifact: a number, a boolean, or a file.
- [ ] **CHK016** The word "generate" is operationally defined (admissible enumeration + minimum cost), not asserted.
- [ ] **CHK017** "Fully lowered" has a definition that is checkable from the emitted program.

## Testability for Demo
- [ ] **CHK001** Every user story has an "Independent Test" that a single test file can satisfy.
- [ ] **CHK006** The failure route for unparseable input is explicitly handled with a named diagnostic type.
- [ ] **CHK007** True-negative cases (Tier 0, Tier 3) are handled and demonstrated.

## Generality (The "Not Just a Toy" Check)
- [ ] **CHK008** The transfer experiment specifies what must transfer *unchanged* and where edits are permitted.
- [ ] **CHK034** Principle III: the second ISA is a scheduled block and its measurement is required to prove transferability.
