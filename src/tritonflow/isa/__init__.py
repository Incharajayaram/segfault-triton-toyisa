"""isa: declarative instruction-set schemas, predicates and selection.

A schema is data (`isa/schemas/*.yaml`); a rules module is the small amount of
Python a schema cannot express (`isa/rules/*.py`). Selection is by least cost
among admissible candidates — never a default (FR-013..FR-018).
"""
