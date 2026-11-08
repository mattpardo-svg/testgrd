"""Sensitivity checks: the guard that declares a sensitivity set invalid (part 1) and what
the loop does with an invalid set (part 2).

PART 1 - GUARD. A sensitivity analysis linearises around a reference current, which must agree
with the load flow of the same state (lf_initial for N-0, lf_nx for N-X). A failed analysis
returns a zero reference and all-zero sensitivities without an error. A contingency / branch
is invalid when any of these holds:
  1. no reference current for ANY monitored branch (not one branch: that branch is left out for its pair, see
     drop_unreferenced_nx; a branch the contingency disconnects has current 0 and no reference: valid);
  2. reference exactly zero while the load-flow current is above ZERO_CURRENT_AMPERE;
  3. all sensitivities of the contingency exactly zero while one of its load-flow currents is
     above ZERO_CURRENT_AMPERE;
  4. |reference - lf| / max(|lf|, RELATIVE_FLOOR_AMPERE) above the tolerance.

PART 2 - REUSE. An invalid set never reaches the MILP: the last VALID set of the same contingency
(N-X) or branch (N-0) replaces it, the constraint stays. A reused set never becomes the "last
valid" one. A stored set is usable only if every (branch, element) key is still monitored and
available and (N-X) the contingency still has a load flow; otherwise the functions return a
failure message and the caller stops the loop.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from pre_milp import log_summary
from pre_milp.sensitivity_table import copy_of

log = logging.getLogger(__name__)

# N-X deviation check (rule 4) of the AC modes
NX_RELATIVE_TOLERANCE = 1e-2
# calculation_mode dc, N-X deviation check (rule 4 above) only (DECISIONS 55, confirmed). The DC
# security analysis and the DC sensitivity analysis of OpenLoadFlow redistribute the distributed
# slack differently after a contingency that disconnects generation or load: the reference differs
# from the security analysis flow by a few MW (measured: at most 1.14 % / 2.8 A on the mock
# fixture, 0.72 % on the generic one; with distributed_slack off both agree to 1e-12 MW). 5 %
# leaves headroom for larger outages. Rules 1-3 (no reference, zero reference, all-zero
# sensitivities, which detect a failed analysis) stay exact; the AC modes keep NX_RELATIVE_TOLERANCE.
DC_NX_RELATIVE_TOLERANCE = 5e-2
N0_RELATIVE_TOLERANCE = 1e-3
RELATIVE_FLOOR_AMPERE = 1.0
ZERO_CURRENT_AMPERE = 1.0


def _finite(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _deviation_reason(reference: Optional[float], lf_current: Optional[float], tolerance: float) -> Optional[str]:
    if not _finite(lf_current):
        return None
    if not _finite(reference):
        # a branch the contingency itself disconnects has load-flow current 0 and no reference
        return "no reference current" if abs(lf_current) > ZERO_CURRENT_AMPERE else None
    if reference == 0.0 and abs(lf_current) > ZERO_CURRENT_AMPERE:
        return f"zero reference current while the load flow current is {lf_current:.1f} A"
    deviation = abs(reference - lf_current) / max(abs(lf_current), RELATIVE_FLOOR_AMPERE)
    if deviation > tolerance:
        return f"reference {reference:.1f} A differs from the load flow {lf_current:.1f} A (relative {deviation:.2f})"
    return None


NO_VALID_BRANCH = "no monitored branch has a reference current"


def find_invalid_nx_contingencies(
    reference_nx: dict[str, dict[str, float]],
    all_zero_contingencies: set,
    lf_nx: dict[str, dict[str, dict[int, float]]],
    branch_sides: dict[str, int],
    tolerance: float = NX_RELATIVE_TOLERANCE,
    unreferenced_out: Optional[dict] = None,
) -> dict[str, str]:
    """{contingency_mrid: reason} for every contingency of reference_nx whose sensitivities
    cannot be trusted. all_zero_contingencies: contingencies whose raw sensitivities are all
    exactly zero (taken before any filter).
    A branch without a reference current in a state (an open end, say) is not a failure: its pair is reported in
    unreferenced_out ({contingency: [branch]}, filled when given) to be left out. The contingencies are invalid for
    that cause only when none of them has a branch with a reference current (no valid N-X data at all)."""
    invalid: dict[str, str] = {}
    unreferenced: dict[str, list[str]] = {}
    without_valid_branch: set[str] = set()
    considered = sorted(set(reference_nx) | set(all_zero_contingencies))
    for contingency_mrid in considered:
        per_branch_lf = lf_nx.get(contingency_mrid) or {}
        references = reference_nx.get(contingency_mrid) or {}
        if contingency_mrid in all_zero_contingencies:
            for branch_mrid, per_side in per_branch_lf.items():
                lf_current = (per_side or {}).get(branch_sides.get(branch_mrid, 1))
                if _finite(lf_current) and abs(lf_current) > ZERO_CURRENT_AMPERE:
                    invalid[contingency_mrid] = "all sensitivities are exactly zero while the load flow current is non-zero"
                    break
            if contingency_mrid in invalid:
                continue
        has_valid_branch = False
        for branch_mrid, per_side in per_branch_lf.items():
            lf_current = (per_side or {}).get(branch_sides.get(branch_mrid, 1))
            if branch_mrid not in branch_sides:
                continue
            reference = references.get(branch_mrid)
            if not _finite(reference):
                if _finite(lf_current) and abs(lf_current) > ZERO_CURRENT_AMPERE:
                    unreferenced.setdefault(contingency_mrid, []).append(branch_mrid)
                continue
            has_valid_branch = True
            reason = _deviation_reason(reference, lf_current, tolerance)
            if reason is not None:
                invalid[contingency_mrid] = f"branch {branch_mrid}: {reason}"
                break
        if contingency_mrid not in invalid and contingency_mrid in unreferenced and not has_valid_branch:
            without_valid_branch.add(contingency_mrid)
    if without_valid_branch and len(without_valid_branch) == len(considered):
        return {contingency_mrid: NO_VALID_BRANCH for contingency_mrid in considered}
    if unreferenced_out is not None:
        unreferenced_out.update({c: sorted(b) for c, b in unreferenced.items() if c not in invalid})
    return invalid


def find_invalid_n0_branches(
    reference_n0: dict[str, float],
    lf_initial: dict[str, dict[int, float]],
    branch_sides: dict[str, int],
    tolerance: float = N0_RELATIVE_TOLERANCE,
    unreferenced_out: Optional[list] = None,
) -> dict[str, str]:
    """{branch_mrid: reason} for every branch whose N-0 reference current disagrees with lf_initial. A branch without
    a reference current (NaN) is not a failure: it is appended to unreferenced_out (when given) to be left out."""
    invalid: dict[str, str] = {}
    for branch_mrid, reference in reference_n0.items():
        lf_current = (lf_initial.get(branch_mrid) or {}).get(branch_sides.get(branch_mrid, 1))
        if not _finite(reference):
            if unreferenced_out is not None and _finite(lf_current) and abs(lf_current) > ZERO_CURRENT_AMPERE:
                unreferenced_out.append(branch_mrid)
            continue
        reason = _deviation_reason(reference, lf_current, tolerance)
        if reason is not None:
            invalid[branch_mrid] = reason
    return invalid


def log_invalid(kind: str, invalid: dict[str, str], context: str) -> None:
    """One WARNING: the count and the first 5 invalid contingencies / branches with their reason; all at debug level."""
    if not invalid:
        return
    log_summary.warn_ids(
        log, f"SENSITIVITY GUARD ({kind}, {context}): %d invalid",
        [f"{key} ({reason})" for key, reason in sorted(invalid.items())],
    )


def drop_unreferenced_nx(lf_nx: dict, unreferenced: dict, context: str) -> None:
    """Leave the (contingency, branch) pairs without a reference current out of lf_nx, as a branch missing after a
    network split (LfNx.dropped), so that no constraint is built for them. One summary warning."""
    pairs: list[str] = []
    dropped = getattr(lf_nx, "dropped", None)
    for contingency_mrid, branches in sorted(unreferenced.items()):
        per_branch = lf_nx.get(contingency_mrid)
        if per_branch is None:
            continue
        for branch_mrid in branches:
            if per_branch.pop(branch_mrid, None) is not None:
                pairs.append(f"{contingency_mrid}/{branch_mrid}")
                if dropped is not None:
                    dropped.setdefault(contingency_mrid, []).append(branch_mrid)
    if pairs:
        contingencies = len({pair.split("/")[0] for pair in pairs})
        log_summary.warn_ids(
            log, f"SENSITIVITY GUARD (N-X, {context}): %d (branch, contingency) pair(s) in {contingencies} contingency(ies) "
                 "have no reference current and are left out, like a branch missing after a network split", pairs)


def blank_unreferenced_n0(sensitivity: dict, unreferenced: list, context: str) -> None:
    """The N-0 entries of the branches without a reference current become NaN: the model leaves a branch with a NaN
    sensitivity out. One summary warning."""
    if not unreferenced:
        return
    left_out = set(unreferenced)
    for key in [key for key in sensitivity if key[0] in left_out]:
        sensitivity[key] = float("nan")
    log_summary.warn_ids(
        log, f"SENSITIVITY GUARD (N-0, {context}): %d branch(es) have no reference current and are left out",
        sorted(left_out))


# ---------------------------------------------------------------------------
# Part 2 - reuse of the last valid set
# ---------------------------------------------------------------------------

class SensitivityGuardError(RuntimeError):
    """Invalid sensitivities at the first computation: nothing valid exists to reuse, the run stops."""


FRESH = "fresh"
STALE_REUSED = "stale_reused"


def _key_mismatch(stored: dict, allowed_branches: set, allowed_elements: set) -> Optional[str]:
    for branch_mrid, element_mrid in stored:
        if str(branch_mrid) not in allowed_branches:
            return f"branch {branch_mrid} is no longer monitored"
        if str(element_mrid) not in allowed_elements:
            return f"element {element_mrid} is no longer available"
    return None


def resolve_nx(
    fresh_pst: dict, fresh_rd: dict, invalid: dict, lf_nx: dict, store: dict,
    allowed_branches: set, allowed_psts: set, allowed_gus: set,
    fresh_hvdc: Optional[dict] = None, allowed_hvdc: Optional[set] = None, hvdc_result: Optional[dict] = None,
):
    """Returns (pst, rd, source, failure). pst/rd: {contingency: {(branch, element): A per unit}} to
    hand to the builder; source: {contingency: FRESH | STALE_REUSED}; failure: None or a message.
    store: {contingency: {"pst": {...}, "rd": {...}}}, updated in place with the valid fresh sets.
    HVDC (docs/specs/hvdc_ra_specs.md H8.1): fresh_hvdc ({contingency: {(branch, EI): A per MW}}, None = no HVDC)
    is handled exactly like the RD sets under the store key "hvdc" (allowed_hvdc = the EIs still available);
    the sets to hand to the builder go into hvdc_result, which is cleared and filled in place. Without
    fresh_hvdc nothing about HVDC is stored or checked and the result is as before."""
    pst, rd, source = dict(fresh_pst), dict(fresh_rd), {}
    hvdc = dict(fresh_hvdc) if fresh_hvdc is not None else None

    def done(failure):
        if hvdc_result is not None:
            hvdc_result.clear()
            if hvdc is not None:
                hvdc_result.update(hvdc)
        return pst, rd, source, failure

    for contingency in sorted(set(fresh_pst) | set(fresh_rd) | set(fresh_hvdc or {}) | set(invalid)):
        if contingency not in invalid:
            store[contingency] = {"pst": copy_of(fresh_pst.get(contingency, {})), "rd": copy_of(fresh_rd.get(contingency, {}))}
            if fresh_hvdc is not None:
                store[contingency]["hvdc"] = copy_of(fresh_hvdc.get(contingency, {}))
            source[contingency] = FRESH
            continue
        rule = invalid[contingency]
        stored = store.get(contingency)
        if stored is None:
            return done(f"contingency {contingency}: sensitivities invalid ({rule}) and no earlier valid set exists")
        if contingency not in (lf_nx or {}):
            return done(
                f"contingency {contingency}: sensitivities invalid ({rule}); the stored valid set no longer "
                f"matches the model: the contingency has no load flow at the current state"
            )
        mismatch = (
            _key_mismatch(stored["pst"], allowed_branches, allowed_psts)
            or _key_mismatch(stored["rd"], allowed_branches, allowed_gus)
            or (_key_mismatch(stored.get("hvdc", {}), allowed_branches, allowed_hvdc or set())
                if hvdc is not None else None)
        )
        if mismatch:
            return done(
                f"contingency {contingency}: sensitivities invalid ({rule}); the stored valid set no longer "
                f"matches the model keys: {mismatch}"
            )
        pst[contingency], rd[contingency] = copy_of(stored["pst"]), copy_of(stored["rd"])
        if hvdc is not None:
            hvdc[contingency] = copy_of(stored.get("hvdc", {}))
        source[contingency] = STALE_REUSED
    return done(None)


def resolve_n0(
    fresh_pst: dict, fresh_rd: dict, invalid: dict, monitored_branches: set,
    store_pst: dict, store_rd: dict, allowed_branches: set, allowed_psts: set, allowed_gus: set,
    fresh_hvdc: Optional[dict] = None, store_hvdc: Optional[dict] = None, allowed_hvdc: Optional[set] = None,
    hvdc_result: Optional[dict] = None,
):
    """N-0 counterpart, per branch. Returns (pst, rd, stale_branches, failure); the sets are
    {(branch, element): value}. store_pst / store_rd: {branch: {(branch, element): value}}.
    HVDC: fresh_hvdc / store_hvdc / allowed_hvdc / hvdc_result as for the RD sets (None = no HVDC, then
    the result is as before); the set for the builder is written into hvdc_result in place."""
    def by_branch(values):
        """{branch: {key: value}} of a {(branch, element): value} set, each branch's entries in the order of
        the set (one pass; a rescan of the whole set per branch is quadratic)."""
        grouped = {}
        for key, value in values.items():
            grouped.setdefault(key[0], {})[key] = value
        return grouped

    branches = set(monitored_branches) | {key[0] for key in fresh_pst} | {key[0] for key in fresh_rd}
    if fresh_hvdc is not None:
        branches |= {key[0] for key in fresh_hvdc}
    store_hvdc = store_hvdc if store_hvdc is not None else {}
    pst, rd, stale = dict(fresh_pst), dict(fresh_rd), {}
    hvdc = dict(fresh_hvdc) if fresh_hvdc is not None else None
    fresh_pst_of, fresh_rd_of = by_branch(fresh_pst), by_branch(fresh_rd)
    fresh_hvdc_of = by_branch(fresh_hvdc) if fresh_hvdc is not None else {}

    def done(failure):
        if hvdc_result is not None:
            hvdc_result.clear()
            if hvdc is not None:
                hvdc_result.update(hvdc)
        return pst, rd, stale, failure

    for branch in sorted(branches | set(invalid)):
        if branch not in invalid:
            store_pst[branch], store_rd[branch] = fresh_pst_of.get(branch, {}), fresh_rd_of.get(branch, {})
            if hvdc is not None:
                store_hvdc[branch] = fresh_hvdc_of.get(branch, {})
            continue
        rule = invalid[branch]
        if branch not in store_pst or branch not in store_rd or (hvdc is not None and branch not in store_hvdc):
            return done(f"branch {branch}: N-0 sensitivities invalid ({rule}) and no earlier valid set exists")
        mismatch = (
            _key_mismatch(store_pst[branch], allowed_branches, allowed_psts)
            or _key_mismatch(store_rd[branch], allowed_branches, allowed_gus)
            or (_key_mismatch(store_hvdc[branch], allowed_branches, allowed_hvdc or set()) if hvdc is not None else None)
        )
        if mismatch:
            return done(
                f"branch {branch}: N-0 sensitivities invalid ({rule}); the stored valid set no longer "
                f"matches the model keys: {mismatch}"
            )
        pst = {k: v for k, v in pst.items() if k[0] != branch}
        rd = {k: v for k, v in rd.items() if k[0] != branch}
        pst.update(store_pst[branch])
        rd.update(store_rd[branch])
        if hvdc is not None:
            hvdc = {k: v for k, v in hvdc.items() if k[0] != branch}
            hvdc.update(store_hvdc[branch])
        stale[branch] = STALE_REUSED
    return done(None)
