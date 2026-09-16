"""T-L2 contract test: the fixture canon itself.

This file is the reason two people can assert the same numbers. It validates
the frozen corpus and `fixtures/GOLDEN.json` against each other, in both
directions, and asserts the properties the pipeline depends on.

It passes today, before any pipeline code exists, and fails the moment the
canon drifts. Nothing here is hand-typed from memory: every expected value
comes from `GOLDEN.json` (observations) or from the reviewed `intent` block.
"""

from __future__ import annotations

import re

import pytest

from fixtures_lib import FIXTURES, load_golden, normalize, read_fixture, scan
from snapshot_fixtures import INTENT, check

ABS_PATH = re.compile(r'loc\("(/[^"]*)"')
TIERS = sorted(INTENT)


def test_canon_agrees_with_fixtures() -> None:
    """The same check CI runs, called in-process so failures are diffable."""
    assert check() == 0, "fixture canon drift: see make golden-check output above"


@pytest.mark.parametrize("name", TIERS)
def test_recorded_hash_matches_file(name: str) -> None:
    golden = load_golden()
    assert golden["observations"][name]["sha256"] == scan(read_fixture(name))["sha256"]


@pytest.mark.parametrize("name", TIERS)
def test_every_recorded_observation_matches(name: str) -> None:
    golden = load_golden()
    observed = scan(read_fixture(name))
    for key, want in golden["observations"][name].items():
        assert observed[key] == want, f"{name}.{key}: recorded {want!r}, observed {observed[key]!r}"


@pytest.mark.parametrize("name", TIERS)
def test_fixture_has_no_absolute_path(name: str) -> None:
    """R3: a machine-dependent fixture hash breaks every other person's run."""
    text = read_fixture(name)
    assert not ABS_PATH.search(text), f"{name} embeds an absolute source path"
    assert 'loc("LOCFILE"' in text, f"{name} was never normalized"


@pytest.mark.parametrize("name", TIERS)
def test_normalization_is_idempotent_and_total(name: str) -> None:
    raw = (FIXTURES / "raw" / f"{name}.ttir").read_text()
    once = normalize(raw)
    assert once == once.rstrip() + "\n"
    assert normalize(once) == once, "normalize() must be a fixed point"
    assert normalize(raw) == read_fixture(name), "fixtures/ is not normalize(fixtures/raw/)"


@pytest.mark.parametrize("name", TIERS)
def test_fixture_is_well_formed(name: str) -> None:
    obs = scan(read_fixture(name))
    assert obs["has_module_wrapper"], "ttir is expected to be wrapped in `module { ... }`"
    assert len(obs["funcs"]) == 1, f"exactly one kernel per fixture, got {obs['funcs']}"
    assert obs["ops"]["tt.return"] == 1, "exactly one terminator per kernel"
    assert obs["ssa_unique"] == obs["ssa_defs"], "SSA names must be unique definitions"
    text = read_fixture(name)
    assert text.endswith("\n") and not text.endswith("\n\n"), "exactly one trailing newline"
    assert "\r" not in text, "CRLF makes every hash platform-dependent"
    assert all(line == line.rstrip() for line in text.splitlines()), "trailing whitespace"
    assert obs["loc_uses"] > 0 and obs["loc_defs"] > 0, "the loc table is the debugging story"


# --- the four tiers' designed properties ------------------------------------


def test_t0_is_the_true_negative_control() -> None:
    """T0 must contain nothing for the MAC or epilogue idioms to match."""
    obs = scan(read_fixture("t0_vecadd"))
    assert obs["ops"]["tt.dot"] == 0
    assert obs["ops"]["scf.for"] == 0
    assert obs["ops"]["tt.load"] >= 1 and obs["ops"]["tt.store"] >= 1
    assert INTENT["t0_vecadd"]["expect_mac_idiom"] == 0


def test_t1_is_the_primary_kernel() -> None:
    """The one occurrence each of the loop, its recurrence and the MAC."""
    obs = scan(read_fixture("t1_matmul"))
    assert obs["ops"]["tt.dot"] == 1
    assert obs["ops"]["scf.for"] == 1
    assert obs["ops"]["scf.yield"] == 1
    assert obs["multi_result_ops"] == 1, "the loop carries 3 iter_args -> 3 results"
    assert obs["ops"]["tt.expand_dims"] >= 1
    assert obs["ops"]["tt.broadcast"] >= 1


def test_t1_loop_recurrence_shape() -> None:
    """The corrected failure mode: pointers are iter_args advanced by a constant."""
    text = read_fixture("t1_matmul")
    loop = next(line for line in text.splitlines() if "scf.for" in line)
    assert "iter_args(" in loop
    assert loop.count("-> (") == 1
    assert "tensor<64x32x!tt.ptr<f32>>" in loop, "the memory-tile pointer type"


def test_t1_declares_tf32_so_tolerance_is_derived_not_chosen() -> None:
    assert scan(read_fixture("t1_matmul"))["has_tf32"], "inputPrecision = tf32 must be present"


def test_t2_adds_an_epilogue_after_the_accumulator() -> None:
    t1, t2 = scan(read_fixture("t1_matmul")), scan(read_fixture("t2_matmul_relu"))
    assert t2["ops"]["tt.dot"] == t1["ops"]["tt.dot"] == 1
    assert t2["ops"]["tt.load"] == t1["ops"]["tt.load"] + 1
    assert t2["ops_total"] > t1["ops_total"]
    assert INTENT["t2_matmul_relu"]["expect_epilogue_idiom"] == 1


def test_t3_is_unstructured_and_must_not_resolve() -> None:
    obs = scan(read_fixture("t3_modulo"))
    assert obs["ops"]["arith.remsi"] >= 1, "modulo addressing comes from ttir, not the parser"
    assert obs["ops"]["tt.dot"] == 0
    assert INTENT["t3_modulo"]["expect_access"] == "unstructured"
    assert INTENT["t3_modulo"]["expect_fully_lowered"] is False


def test_region_nesting_is_exercised_by_t1_and_t2() -> None:
    depths = {n: scan(read_fixture(n))["max_region_depth"] for n in TIERS}
    assert depths["t0_vecadd"] == depths["t3_modulo"] == 2
    assert depths["t1_matmul"] == depths["t2_matmul_relu"] == 3, "loop body is a region"


def test_loc_table_carries_python_variable_names() -> None:
    """The debugging story: recognition failures can name the source variable."""
    names = set(scan(read_fixture("t1_matmul"))["loc_names"])
    assert {"a_ptrs", "b_ptrs", "c_ptrs", "acc", "a_ptr", "b_ptr", "c_ptr"} <= names
    assert {"M", "N", "K"} <= names


def test_corpus_is_a_graduated_ladder_not_four_copies() -> None:
    obs = {n: scan(read_fixture(n)) for n in TIERS}
    assert len({o["sha256"] for o in obs.values()}) == 4
    assert len({o["op_vocab"] for o in obs.values()}) >= 3
    assert max(o["ops_total"] for o in obs.values()) < 4 * min(o["ops_total"] for o in obs.values())
