"""milp_builder.py — the Pyomo MILP of one solve, built at the current operating point.

Blocks, each built only when its inputs exist (RA types compose freely):
- PST: tap_change_upward/downward, tap bounds, group equality, per-TSO tap cap;
- RD: rd_volume_upward/downward (preventive bids only), rd_net_injection_g, Pmax/Pmin,
  balance sum_g rd_net_injection_g[g] = 0, lead times (usable only STRICTLY after
  radp + lead time), RA-count binaries (big-M = the bid's available_mw), RA-group caps;
- HVDC (docs/specs/hvdc_ra_specs.md): hvdc_dp_up/down per unit, bounds Pmin_op <= P + dP <= Pmax_op, the balance
  dP_g1 + dP_g2 = 0 per group (outside the RD balance), the effect on every monitored branch;
- N-0: secured (PATL) and scanned (effective limit) loading constraints with slacks;
- N-X: PATL, TATL (separate constraint, no loading pre-filter) and scanned constraints
  per (branch, contingency), with curative PST and curative RD (gated only on rd_inputs;
  cost on the max-over-contingencies reservation).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from data_model import (
    AEType, AssessedElement, BidDirection, HvdcUnit, PST, PowerBid, RAKind, RDUnit, branch_monitoring_rules,
    effective_linked_contingencies, pair_is_monitored, select_by_side,
)
from pre_milp import ct_sensitivity, log_summary
from pre_milp.cra_preprocessing import TATL_FALLBACK_DURATION_S_DEFAULT, compute_rd_cra_availability

log = logging.getLogger(__name__)

# Logging verbosity only, set once by run_pipeline from logging_options.debug_log_mode.
# False: _log_if_changed suppresses repeats of the same message; True: every call logs.
DEBUG_LOG_MODE = False

_last_logged_values: dict[str, object] = {}


def reset_run_log_state() -> None:
    """Forget every message _log_if_changed has logged: called by run_pipeline.main at the start
    of each run, so a second run in the same process (e.g. in tests or notebooks) writes the log
    lines again (F2 rule 15)."""
    _last_logged_values.clear()


def _log_if_changed(key: str, value: object, message: str, *args, level: int = logging.INFO) -> None:
    """Log `message % args` at `level` unless DEBUG_LOG_MODE is False and `value` equals the
    last value logged under `key` in this run (reset_run_log_state). For the messages that
    would repeat unchanged at every rebuild of the linearisation loop: RD curative-bid split,
    N-X pair count, reservePrice fallback, Pmin := PROD, Pmin > Pmax, bids blocked by the unit
    status or an empty GLSK."""
    if DEBUG_LOG_MODE or _last_logged_values.get(key) != value:
        log.log(level, message, *args)
        _last_logged_values[key] = value


# a TATL counts as above the PATL only beyond this margin (A): the PATL is kA x 1000 and the TATL
# comes from the CGM in A, so equal limits can differ in the last bit (1.005 kA = 1004.9999999999999
# A against a TATL of 1005.0 A)
TATL_ABOVE_PATL_TOLERANCE_AMPERE = 1e-6


def tatl_is_above_patl(tatl_ampere: float, patl_ampere: float) -> bool:
    """R11 rule 4: a branch gets a TATL constraint only for a TATL strictly above its PATL, with
    TATL_ABOVE_PATL_TOLERANCE_AMPERE. Used by the builder and the post-RA congestion check (the
    convergence check follows the builder's overload_monitoring)."""
    return float(tatl_ampere) > float(patl_ampere) + TATL_ABOVE_PATL_TOLERANCE_AMPERE


# ---------------------------------------------------------------------------
# Composable MILP input groupings
# ---------------------------------------------------------------------------

@dataclass
class PstMilpInputs:
    """PST inputs of the MILP. Non-None builds the PST block (tap variables, bounds, group
    equality, model.pst_effect_on_branch); None builds no PST component at all.
    max_taps_per_tso: HARD per-TSO cap on sum(tap_change_upward + tap_change_downward) over
    its available preventive PSTs."""
    pst_list: list[PST]
    pst_tap_bounds: dict[str, tuple[int, int]]
    pst_group_equality_pairs: list[tuple[str, str]]
    sensitivity_pst_n0_pruned: dict[tuple[str, str], float]
    pst_degrees_per_tap: dict[str, float]
    penalty_reference_pst_pra: float
    penalty_reference_pst_pra_per_tso: Optional[dict[str, float]] = None
    pst_taping_discrete: bool = True
    max_taps_per_tso: int = 6
    # Optional per-TSO override of max_taps_per_tso (same pattern as the
    # penalty_reference_*_per_tso maps): {tso: limit}. A TSO not listed uses
    # max_taps_per_tso. Values are validated (non-negative integers).
    max_taps_per_tso_overrides: Optional[dict[str, int]] = None
    # tap of each PST in the ORIGINAL network (pst.r0 is the current tap); None: original ==
    # current. The tap limit and penalty act on the cumulative |current - original + delta|.
    pst_original_taps: Optional[dict[str, int]] = None

    # pst_list holds every PST (both kinds) so the AC loop updates r0 in one pass; the kind
    # split is made here: everything preventive reads preventive_pst_list, everything curative
    # curative_pst_list. A transformer with both kinds appears in both, as two PST objects.
    @property
    def preventive_pst_list(self) -> list[PST]:
        return [pst for pst in self.pst_list if pst.kind == RAKind.PREVENTIVE]

    @property
    def curative_pst_list(self) -> list[PST]:
        return [pst for pst in self.pst_list if pst.kind == RAKind.CURATIVE]

    # What the MILP actually builds on: an unavailable PST (PST.available is
    # False, NC normalAvailable) gets no variables, no tap-limit contribution
    # and no objective term, preventive or curative.
    @property
    def available_preventive_pst_list(self) -> list[PST]:
        # no pst_tap_bounds entry: excluded by build_pst_tap_bounds (empty range, warned):
        # no variables, no tap-limit term, no objective term
        return [
            pst for pst in self.preventive_pst_list
            if pst.available and str(pst.mrid) in self.pst_tap_bounds
        ]

    @property
    def available_curative_pst_list(self) -> list[PST]:
        return [pst for pst in self.curative_pst_list if pst.available]


@dataclass
class RdMilpInputs:
    """RD inputs of the MILP. Non-None builds the RD block (bid volumes, rd_net_injection_g,
    rd_effect_on_branch, Pmax/Pmin); None builds no RD component at all.
    generator_data: {gu_mrid: {target_p/initial_p, max_p, min_p, name}}.
    penalty_reference_rd_pra (default 0.0, no-op) with per-TSO overrides.
    RA count: per-TSO limits per timestamp (hard if ra_limit_constraint_hard, default True,
    else soft); a TSO without a limit is not constrained.
    ra_groups: parse_ra_groups output (None/{}: no group caps). raart: None -> radp is one
    hour before `timestamp` (_compute_radp). The timestamp is build_n0_milp_model's own.
    """
    rd_list: list[RDUnit]
    bid_list: list[PowerBid]
    glsk: dict[str, dict[str, float]]
    sensitivity_rd_n0_pruned: dict[tuple[str, str], float]
    generator_data: dict[str, dict]
    penalty_reference_rd_pra: float = 0.0
    penalty_reference_rd_pra_per_tso: Optional[dict[str, float]] = None
    max_ra_count_per_timestamp_tso: Optional[dict[str, int]] = None
    ra_limit_penalty: float = 0.0
    ra_limit_constraint_hard: bool = True
    ra_groups: Optional[dict[str, dict]] = None
    raart: Optional[str] = None
    # cumulative_signed_volume_mw: {bid: MW used by earlier iterations, UP +, DOWN -}; headroom,
    # cost, RA binary and group caps act on used + this solve's change (None: nothing used).
    # original_target_p_mw: {gu: ORIGINAL target_p}, for original / cumulative / final reports.
    cumulative_signed_volume_mw: Optional[dict[str, float]] = None
    original_target_p_mw: Optional[dict[str, float]] = None
    # {gu_mrid: UnitCost}: start-up / shut-down cost of a unit (F2); a unit without a cost
    # cannot be started / shut down by the MILP (None: no unit has a cost)
    unit_costs: Optional[dict] = None
    # {bid mRID: {unit: factor}} for the bids that carry their own participation factors instead of
    # glsk[bid.ra_mrid] (countertrade: up and down weights of one RA differ); None: every bid uses
    # the glsk of its RA
    bid_participation: Optional[dict[str, dict[str, float]]] = None
    # countertrade references (EUR/MW) for the bids of the RAs in rd_list with countertrade=True
    penalty_reference_ct_pra: float = 0.0
    penalty_reference_ct_cra: float = 0.0


@dataclass
class CtMilpInputs:
    """Countertrade inputs that travel through the loops next to the RD inputs (docs/specs/countertrade_ra_specs.md):
    the fixed per-bid weights (computed once from the ORIGINAL set points, C3.5) and the two penalty references
    (EUR/MW). The CT RAs themselves are RDUnit objects with countertrade=True in rd_list and their bids are in
    bid_list; build_model_at_state copies these inputs into RdMilpInputs."""
    bid_participation: dict[str, dict[str, float]]
    penalty_reference_pra: float = 0.0
    penalty_reference_cra: float = 0.0


@dataclass
class HvdcMilpInputs:
    """HVDC inputs of the MILP (docs/specs/hvdc_ra_specs.md H5-H7). Non-None builds the HVDC block
    (hvdc_dp_up/down, hvdc_bounds, hvdc_group_balance, hvdc_effect_on_branch); None builds none.
    units: the usable units (validated: each in a balanced group of two). limits: {unit RA mRID:
    (Pmin_op, Pmax_op)} MW from resolve_operating_limits on the ORIGINAL set point (generation sign).
    original_p_mw / current_p_mw: {EI mRID: MW} the set point in the ORIGINAL network and at the
    operating point this model is built on (they differ in a later solve of the linearisation loop).
    sensitivity_hvdc_n0: {(branch, EI): A per MW}. penalty_reference_pra / _cra: EUR per MW."""
    units: list[HvdcUnit]
    limits: dict[str, tuple[float, float]]
    original_p_mw: dict[str, float]
    current_p_mw: dict[str, float]
    sensitivity_hvdc_n0: dict[tuple[str, str], float]
    penalty_reference_pra: float = 0.0
    penalty_reference_cra: float = 0.0
    # curative (H6.2): {contingency: EIs that are outage elements of it} and {contingency: EIs whose N-X
    # sensitivities are exactly zero there}; a group with such a unit is unavailable in that contingency
    outage_eis_by_contingency: dict[str, frozenset] = field(default_factory=dict)
    zero_units_nx: dict[str, list] = field(default_factory=dict)

    @property
    def preventive_units(self) -> list[HvdcUnit]:
        return [u for u in self.units if u.kind == RAKind.PREVENTIVE and u.available]

    @property
    def curative_units(self) -> list[HvdcUnit]:
        return [u for u in self.units if u.kind == RAKind.CURATIVE and u.available]


@dataclass
class NxMilpInputs:
    """N-X inputs of the MILP. None: the N-0 model only. N-X terms of an RA type are added
    only when that type's preventive block exists (nx_inputs never creates RA variables).
    - lf_nx / sensitivity_pst_nx / sensitivity_rd_nx: keyed by contingency; only contingencies
      in lf_nx get constraints. The sensitivities are the Filter-2-PRUNED dicts (the caller
      prunes); PST values are A/degree, converted with pst_inputs.pst_degrees_per_tap.
    - min_loading_threshold_pct_nx: the PATL pre-filter, separate from the N-0 threshold.
    - PATL constraint: SECURED branches; scanned branches get the scanned constraint
      (penalty_cost_overload_nx_scanned, effective limit).
    - tatl_data: a separate TATL constraint (preventive state only) for every usable pair of
      a branch whose TATL is above its PATL, with no loading pre-filter; the TATL durations
      also decide the curative RD availability per branch.
    - pst_cra_availability: one curative variable pair per (PST, contingency) with at least
      one available XNE; each available (pst, xne, c) entry adds its own magnitude bound.
    - pst_tap_bounds_curative: effective bounds (the caller applies the preventive fallback);
      a curative PST missing from it has no bound constraint (warning).
    - activate_cra_xne_linkage (default False): True also requires the XNE to be in the
      PST's linked_xne_mrids for its term to enter that XNE's constraint.
    """
    lf_nx: dict[str, dict[str, dict[int, float]]]
    sensitivity_pst_nx: dict[str, dict[tuple[str, str], float]]
    sensitivity_rd_nx: dict[str, dict[tuple[str, str], float]]
    penalty_cost_overload_nx_patl: float = 0.0
    min_loading_threshold_pct_nx: float = 0.0
    tatl_data: Optional[dict[str, dict]] = None
    penalty_cost_overload_nx_tatl: float = 0.0
    pst_cra_availability: Optional[dict[tuple[str, str, str], dict]] = None
    pst_tap_bounds_curative: Optional[dict[str, tuple[int, int]]] = None
    # (pst_1, pst_2, contingency) triples whose curative delta tap (up - down) must be equal,
    # from cra_preprocessing.build_curative_pst_group_constraints (curative PSTs only)
    curative_pst_group_triples: Optional[list[tuple[str, str, str]]] = None
    activate_cra_xne_linkage: bool = False
    penalty_cost_overload_nx_scanned: float = 0.0
    # TATL duration of a branch without a tatl_data entry (config tatl_fallback_duration_s):
    # a curative RD RA enters a branch's constraints only if its reaction time is <= the
    # branch's TATL duration (CORNET-613)
    tatl_fallback_duration_seconds: int = TATL_FALLBACK_DURATION_S_DEFAULT
    # {contingency: {(branch, EI): A per MW}} of the HVDC units (docs/specs/hvdc_ra_specs.md H8); {} without HVDC
    sensitivity_hvdc_nx: dict = field(default_factory=dict)
    # Leave the terms with a zero (or missing) sensitivity out of the N-X rows. Set by the pipeline only when the N-X
    # pruning threshold of that kind is above 0 (then most of the (branch, variable) pairs are missing and a row would
    # carry a 0.0 * variable term for each); with the threshold 0 (default) the rows keep every term, as always.
    drop_zero_pst_terms: bool = False
    drop_zero_rd_terms: bool = False


# ---------------------------------------------------------------------------
# Shared helpers (RA-type-agnostic)
# ---------------------------------------------------------------------------

def _hhmmz_to_minutes(hhmmz: str) -> int:
    """"HHMMZ" (e.g. "1030Z") -> minutes since midnight."""
    hh = int(hhmmz[0:2])
    mm = int(hhmmz[2:4])
    return hh * 60 + mm


def _minutes_to_hhmmz(minutes: int) -> str:
    """Inverse of _hhmmz_to_minutes, mod 1440: one hour before "0030Z" is "2330Z"."""
    minutes = minutes % (24 * 60)
    return f"{minutes // 60:02d}{minutes % 60:02d}Z"


def _compute_radp(raart: Optional[str], timestamp: str, interval_minutes: int = 60) -> str:
    """radp (RD availability reference point): raart when given ("2200Z" -> "2200Z"), else
    the timestamp interval_minutes (60, the hourly timestamp list) before `timestamp`."""
    if raart is not None:
        return raart
    return _minutes_to_hhmmz(_hhmmz_to_minutes(timestamp) - interval_minutes)


def _timestamp_is_usable(radp: str, bid_lead_time_hours: float, timestamp: str) -> bool:
    """True if `timestamp` is usable for a bid: t_start_minutes > radp_minutes +
    lead_time_minutes (STRICT: a lead time ending within or exactly at the start of a
    timestamp's hour blocks it). Example: radp 23:30, lead 3 h -> cutoff 02:30 -> 0230Z
    blocked, 0330Z usable. A timestamp earlier than radp on the clock is the next day (+24 h).
    """
    radp_minutes = _hhmmz_to_minutes(radp)
    timestamp_minutes = _hhmmz_to_minutes(timestamp)
    if timestamp_minutes < radp_minutes:
        timestamp_minutes += 24 * 60

    cutoff_minutes = radp_minutes + bid_lead_time_hours * 60
    return timestamp_minutes > cutoff_minutes


def _compute_earliest_usable_timestamp(radp: str, bid_lead_time_hours: float, timestamp: str) -> Optional[str]:
    """The run timestamp when the bid is usable at it (the same check as the bound tightening), for
    results_lead_times.csv; None (a blank in the file) when its lead time blocks the run timestamp."""
    return timestamp if _timestamp_is_usable(radp, bid_lead_time_hours, timestamp) else None


def _get_monitored_branch_limits_and_ae_types(
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_initial: Optional[dict[str, dict[int, float]]],
) -> tuple[dict[str, float], dict[str, "AEType"]]:
    """({branch_mrid: limit_ampere}, {branch_mrid: ae_type}) for every SECURED and SCANNED AE:
    PATL for secured, effective_limit = max(PATL, initial loading) for scanned (virtual
    margin 0). build_n0_milp_model splits the branches by ae_type into the secured and the
    scanned constraint, both reading the same limits dict."""
    limits: dict[str, float] = {}
    ae_types: dict[str, AEType] = {}
    for ae in ae_flat_list:
        if ae.ae_type not in (AEType.SECURED, AEType.SCANNED) or ae.branch_mrid is None:
            continue
        branch_mrid = str(ae.branch_mrid)
        patl_ka = ae.patl_ka.get(timestamp)
        if patl_ka is None:
            continue

        if ae.ae_type == AEType.SCANNED:
            side = int(getattr(ae, "branch_side", 1))
            per_side_currents = lf_initial.get(branch_mrid) if lf_initial else None
            i_a = select_by_side(per_side_currents, side, branch_mrid) if per_side_currents else None
            limit_ka = max(patl_ka, float(i_a) / 1000.0) if i_a is not None else patl_ka
        else:
            limit_ka = patl_ka

        limits[branch_mrid] = float(limit_ka) * 1000.0
        ae_types[branch_mrid] = ae.ae_type
    return limits, ae_types


def _get_monitored_branch_sides(ae_flat_list: list[AssessedElement]) -> dict[str, int]:
    """{branch_mrid: resolved side (1 or 2)} for every SECURED and SCANNED AE (the side of its
    CurrentLimit; used for the side-aware lf_initial lookups)."""
    sides: dict[str, int] = {}
    for ae in ae_flat_list:
        if ae.ae_type not in (AEType.SECURED, AEType.SCANNED) or not ae.branch_mrid:
            continue
        sides[str(ae.branch_mrid)] = int(getattr(ae, "branch_side", 1))
    return sides


def _log_unusable_branches(excluded: dict[str, list[str]]) -> None:
    """One warning per reason: the count and the first 5 branches (all of them at debug level)."""
    for reason, branches in excluded.items():
        log_summary.warn_ids(log, "%d branch(es) with " + reason.replace("%", "%%") + " excluded from the MILP", branches)


def _branch_is_above_threshold(
    branch_mrid: str,
    lf_initial: dict[str, dict[int, float]],
    branch_patls: dict[str, float],
    branch_sides: dict[str, int],
    min_loading_threshold_pct: float,
    excluded: Optional[dict[str, list[str]]] = None,
) -> bool:
    """excluded: {reason: branches}; a branch left out is added there (the caller logs one summary per reason) instead of
    being logged here."""
    side = branch_sides.get(branch_mrid, 1)
    per_side_currents = lf_initial.get(branch_mrid)
    initial_current = select_by_side(per_side_currents, side, branch_mrid) if per_side_currents else None

    def _leave_out(reason: str) -> bool:
        if excluded is None:
            log.warning("Branch %s: %s; excluding from MILP", branch_mrid, reason)
        else:
            excluded.setdefault(reason, []).append(branch_mrid)
        return False

    if initial_current is None:
        return _leave_out(f"missing initial current (side {side})")
    if math.isnan(float(initial_current)):
        return _leave_out(f"NaN initial current (side {side})")

    patl_ampere = branch_patls.get(branch_mrid)
    if patl_ampere is None or patl_ampere <= 0.0:
        return _leave_out("missing or invalid PATL")

    loading_pct = (initial_current / patl_ampere) * 100.0
    return loading_pct >= float(min_loading_threshold_pct)


# ---------------------------------------------------------------------------
# PST-specific helpers
# ---------------------------------------------------------------------------

def _get_pst_ids(pst_list: list[PST]) -> list[str]:
    return [str(pst.mrid) for pst in pst_list]


def validate_max_taps_per_tso(value, description: str = "max_taps_per_tso") -> int:
    """A tap limit must be a non-negative integer (bool is not accepted)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer, got {value!r}")
    return value


def _get_effective_penalty_costs(
    pst_list: list[PST],
    penalty_reference_pst_pra: float,
    penalty_reference_pst_pra_per_tso: Optional[dict[str, float]] = None,
) -> dict[str, float]:
    override = penalty_reference_pst_pra_per_tso or {}
    # callers pass one kind's list at a time (preventive and curative never share an entry);
    # two RAs of one kind on one transformer with different penalties: the larger (warning)
    costs: dict[str, float] = {}
    for pst in pst_list:
        mrid = str(pst.mrid)
        cost = float(pst.penalty_factor) * float(override.get(pst.tso, penalty_reference_pst_pra))
        if mrid in costs and costs[mrid] != cost:
            log.warning(
                "PST %s has two RAs of the same kind with different effective "
                "penalties (%s vs %s) — using the larger, order-independent value",
                mrid, costs[mrid], cost,
            )
            cost = max(costs[mrid], cost)
        costs[mrid] = cost
    return costs


def _add_pst_block(model, pyo, pst_inputs: PstMilpInputs, branch_ids: list[str]) -> list[str]:
    """Add the PST variables, tap bound / group equality constraints and
    model.pst_effect_on_branch. Returns pst_ids for the objective."""
    if not pst_inputs.pst_list:
        raise ValueError("PST list is empty; cannot build MILP model")
    preventive_pst_list = pst_inputs.available_preventive_pst_list
    if not preventive_pst_list:
        # only curative PSTs: no preventive block (no tap variables, no tap cap); the curative
        # block is built by _add_nx_block
        log.info(
            "_add_pst_block: pst_list contains %d PST(s) but none is an available "
            "preventive PST — no preventive PST variables added (curative PSTs are "
            "handled by _add_curative_pst_block)", len(pst_inputs.pst_list),
        )
        return []
    pst_by_mrid: dict[str, PST] = {str(pst.mrid): pst for pst in preventive_pst_list}
    # One variable pair per transformer: two preventive RAs on the same
    # transformer share their mRID and must count ONCE in the branch effect
    # and in the per-TSO tap limit (pyo.Set dedups, plain lists do not).
    pst_ids = list(dict.fromkeys(_get_pst_ids(preventive_pst_list)))
    if len(pst_ids) != len(preventive_pst_list):
        log.warning(
            "_add_pst_block: %d available preventive PST object(s) share an mRID with another "
            "— each transformer is modelled once (variables, tap limit, branch effect)",
            len(preventive_pst_list) - len(pst_ids),
        )

    missing_deg_per_tap = [p for p in pst_ids if p not in pst_inputs.pst_degrees_per_tap]
    if missing_deg_per_tap:
        log.warning(
            "PST(s) missing degrees-per-tap conversion factor — excluded from "
            "tap_effect (their sensitivity cannot be safely applied): %s",
            missing_deg_per_tap,
        )

    model.pst_ids = pyo.Set(initialize=pst_ids)
    # Only the equality pairs naming a PST without model variables (excluded
    # for an empty tap range, or unavailable) are dropped, since an equality
    # with a PST that is not modelled cannot be built; the group's other
    # pairs stay.
    group_pairs = []
    for pair in pst_inputs.pst_group_equality_pairs:
        if pair[0] in model.pst_ids and pair[1] in model.pst_ids:
            group_pairs.append(pair)
        else:
            log.warning(
                "_add_pst_block: PST group pair %s dropped - a member has no model "
                "variables (excluded for an empty tap range or unavailable)", tuple(pair),
            )
    model.pst_group_pairs = pyo.Set(initialize=group_pairs, dimen=2)

    tap_domain = pyo.NonNegativeIntegers if pst_inputs.pst_taping_discrete else pyo.NonNegativeReals
    model.tap_change_upward = pyo.Var(model.pst_ids, domain=tap_domain)
    model.tap_change_downward = pyo.Var(model.pst_ids, domain=tap_domain)

    def tap_upper_bound_rule(m, pst_mrid):
        pst = pst_by_mrid[str(pst_mrid)]
        current_tap = int(getattr(pst, "r0", 0))
        effective_max = int(pst_inputs.pst_tap_bounds[str(pst_mrid)][1])
        return current_tap + m.tap_change_upward[pst_mrid] - m.tap_change_downward[pst_mrid] <= effective_max

    def tap_lower_bound_rule(m, pst_mrid):
        pst = pst_by_mrid[str(pst_mrid)]
        current_tap = int(getattr(pst, "r0", 0))
        effective_min = int(pst_inputs.pst_tap_bounds[str(pst_mrid)][0])
        return effective_min <= current_tap + m.tap_change_upward[pst_mrid] - m.tap_change_downward[pst_mrid]

    def pst_group_equality_rule(m, pst_mrid_1, pst_mrid_2):
        current_tap_1 = int(getattr(pst_by_mrid[str(pst_mrid_1)], "r0", 0))
        current_tap_2 = int(getattr(pst_by_mrid[str(pst_mrid_2)], "r0", 0))
        return (
            current_tap_1 + m.tap_change_upward[pst_mrid_1] - m.tap_change_downward[pst_mrid_1]
            ==
            current_tap_2 + m.tap_change_upward[pst_mrid_2] - m.tap_change_downward[pst_mrid_2]
        )

    model.tap_upper_bound = pyo.Constraint(model.pst_ids, rule=tap_upper_bound_rule)
    model.tap_lower_bound = pyo.Constraint(model.pst_ids, rule=tap_lower_bound_rule)
    model.pst_group_equality = pyo.Constraint(model.pst_group_pairs, rule=pst_group_equality_rule)

    # CUMULATIVE tap change against the ORIGINAL network: offset = pst.r0 - original tap;
    # cumulative change = offset + up - down, on which the per-TSO tap limit and the penalty
    # act. Offset 0: the cumulative terms are the up/down variables. Otherwise two extra
    # non-negative variables with cum_up - cum_down = offset + up - down carry |.| (no
    # binary: the limit and the positive penalty push their sum down to the absolute value).
    original_taps = pst_inputs.pst_original_taps or {}
    tap_offset_by_pst = {
        pst_mrid: int(getattr(pst_by_mrid[str(pst_mrid)], "r0", 0))
        - int(original_taps.get(str(pst_mrid), getattr(pst_by_mrid[str(pst_mrid)], "r0", 0)))
        for pst_mrid in pst_ids
    }
    model.pst_start_tap_by_mrid = {
        str(p): int(getattr(pst_by_mrid[str(p)], "r0", 0)) for p in pst_ids
    }
    model.pst_original_tap_by_mrid = {
        str(p): int(model.pst_start_tap_by_mrid[str(p)] - tap_offset_by_pst[p]) for p in pst_ids
    }
    pst_ids_with_offset = [p for p in pst_ids if tap_offset_by_pst[p] != 0]
    model.cum_tap_pst_ids = pyo.Set(initialize=pst_ids_with_offset)
    model.cum_tap_change_upward = pyo.Var(model.cum_tap_pst_ids, domain=pyo.NonNegativeReals)
    model.cum_tap_change_downward = pyo.Var(model.cum_tap_pst_ids, domain=pyo.NonNegativeReals)

    def cum_tap_balance_rule(m, pst_mrid):
        return (
            m.cum_tap_change_upward[pst_mrid] - m.cum_tap_change_downward[pst_mrid]
            == tap_offset_by_pst[pst_mrid] + m.tap_change_upward[pst_mrid] - m.tap_change_downward[pst_mrid]
        )

    model.cum_tap_balance = pyo.Constraint(model.cum_tap_pst_ids, rule=cum_tap_balance_rule)
    # plain dicts {pst_mrid: Var}: the term to use for "cumulative up" / "cumulative down"
    model.cum_tap_up_term = {
        p: (model.cum_tap_change_upward[p] if p in model.cum_tap_pst_ids else model.tap_change_upward[p])
        for p in pst_ids
    }
    model.cum_tap_down_term = {
        p: (model.cum_tap_change_downward[p] if p in model.cum_tap_pst_ids else model.tap_change_downward[p])
        for p in pst_ids
    }

    def pst_effect_on_branch_rule(m, branch_mrid):
        # sensitivity_pst_n0_pruned is in A/degree: x pst_degrees_per_tap gives A per tap
        return sum(
            float(pst_inputs.sensitivity_pst_n0_pruned.get((branch_mrid, pst_mrid), 0.0))
            * float(pst_inputs.pst_degrees_per_tap[pst_mrid])
            * (m.tap_change_upward[pst_mrid] - m.tap_change_downward[pst_mrid])
            for pst_mrid in pst_ids
            if pst_mrid in pst_inputs.pst_degrees_per_tap
        )

    model.pst_effect_on_branch = pyo.Expression(branch_ids, rule=pst_effect_on_branch_rule)

    # Hard per-TSO tap-change limit: sum over the TSO's PSTs of (up + down) <= max_taps_per_tso.
    # Why: a large combined multi-PST move makes the linear sensitivities wrong, even in sign,
    # and no TSO would allow such a swing within one timestamp. up + down is |change| at the
    # optimum (both positive only adds cost), so no abs variable is needed. A TSO without
    # PSTs gets a vacuous constraint.
    tso_to_pst_ids: dict[str, list[str]] = {}
    for pst_mrid in pst_ids:
        tso_to_pst_ids.setdefault(pst_by_mrid[pst_mrid].tso, []).append(pst_mrid)

    model.tso_ids_for_tap_change_limit = pyo.Set(initialize=sorted(tso_to_pst_ids.keys()))

    default_limit = validate_max_taps_per_tso(pst_inputs.max_taps_per_tso)
    limit_overrides = {
        str(tso): validate_max_taps_per_tso(limit, f"max_taps_per_tso_overrides[{tso!r}]")
        for tso, limit in (pst_inputs.max_taps_per_tso_overrides or {}).items()
    }
    tso_tap_limits = {tso: limit_overrides.get(tso, default_limit) for tso in tso_to_pst_ids}
    # kept on the model so the results file reports the limit that was applied
    model.tso_tap_limit_values = tso_tap_limits  # plain dict attribute: {tso: limit}

    def max_taps_per_tso_rule(m, tso):
        # CUMULATIVE against the original network: later
        # iterations only get the headroom the earlier ones left.
        return sum(
            m.cum_tap_up_term[pst_mrid] + m.cum_tap_down_term[pst_mrid]
            for pst_mrid in tso_to_pst_ids[tso]
        ) <= tso_tap_limits[tso]

    model.max_taps_per_tso_constraint = pyo.Constraint(
        model.tso_ids_for_tap_change_limit, rule=max_taps_per_tso_rule,
    )

    return pst_ids


# ---------------------------------------------------------------------------
# RD-specific helpers
# ---------------------------------------------------------------------------

def _get_gu_ids_reachable_via_glsk(glsk: dict[str, dict[str, float]]) -> list[str]:
    """Every generating-unit mRID that appears with a non-empty participation
    factor in ANY RA's resolved GLSK — the only units RD can actually move.

    Sorted for deterministic Pyomo Set ordering (matters for reproducible
    solves and test assertions).
    """
    gu_ids: set[str] = set()
    for ra_participation in glsk.values():
        gu_ids.update(ra_participation.keys())
    return sorted(gu_ids)


def _participation_of_bid(rd_inputs: RdMilpInputs, bid: PowerBid) -> dict[str, float]:
    """{unit: participation factor} of one bid: its own entry in rd_inputs.bid_participation, else the
    glsk of its remedial action (empty when neither exists)."""
    if rd_inputs.bid_participation is not None and str(bid.mrid) in rd_inputs.bid_participation:
        return rd_inputs.bid_participation[str(bid.mrid)]
    return rd_inputs.glsk.get(str(bid.ra_mrid), {})


def _ct_ra_mrids(rd_inputs: RdMilpInputs) -> set[str]:
    """mRIDs of the countertrade RAs of rd_inputs.rd_list."""
    return {str(rd.mrid) for rd in rd_inputs.rd_list if rd.countertrade}


def _ct_bids_without_weights(rd_inputs: RdMilpInputs, bid_by_mrid: dict[str, PowerBid], bid_ids) -> set[str]:
    """The countertrade bids of bid_ids whose own participation is empty (no unit can move in their direction)."""
    ct_ra_mrids = _ct_ra_mrids(rd_inputs)
    return {
        b for b in bid_ids
        if str(bid_by_mrid[b].ra_mrid) in ct_ra_mrids and not _participation_of_bid(rd_inputs, bid_by_mrid[b])
    }


def _gu_ids_reachable(rd_inputs: RdMilpInputs) -> list[str]:
    """Every unit with a participation factor in the glsk of any RA or in the participation of any bid."""
    gu_ids = set(_get_gu_ids_reachable_via_glsk(rd_inputs.glsk))
    for participation in (rd_inputs.bid_participation or {}).values():
        gu_ids.update(participation.keys())
    return sorted(gu_ids)


# ---------------------------------------------------------------------------
# Generator on/off (F2): status and class of each generating unit, on the ORIGINAL production
# ---------------------------------------------------------------------------

# unit classes: CONTINUOUS (Pmin <= P <= Pmax, today's constraint), STARTABLE (initially off,
# may be started by preventive up bids: binary), STOPPABLE (initially on, may be shut down by
# preventive down bids: binary), FIXED_OFF (initially off, cannot be started: stays at 0, every
# bid containing it is blocked). Per unit, for single- and multi-unit bids alike (F2 rule 9).
UNIT_CONTINUOUS, UNIT_STARTABLE, UNIT_STOPPABLE, UNIT_FIXED_OFF = "continuous", "startable", "stoppable", "fixed_off"

# why a FIXED_OFF unit cannot be started (the log line of every bid it blocks, F2 rule 9)
NO_STARTUP_COST_REASON = "no start-up cost"
PMIN_UNREACHABLE_REASON = "Pmin unreachable within the bid volume"


@dataclass
class RdUnitStatus:
    initially_on: bool
    unit_class: str
    pmin_mw: float            # effective Pmin (Pmin := PROD for 0 < PROD < Pmin, F2 rule 11)
    pmax_mw: float
    original_p_mw: float
    startup_cost_eur: Optional[float] = None
    shutdown_cost_eur: Optional[float] = None
    off_reason: Optional[str] = None   # FIXED_OFF only: NO_STARTUP_COST_REASON and/or PMIN_UNREACHABLE_REASON


# |PROD| at or below this (MW) is 0: a CGM value of 1e-9 MW is an off unit (F2 rule 1)
PROD_ZERO_TOLERANCE_MW = 1e-6


def _rd_unit_status(rd_inputs: RdMilpInputs, timestamp: str) -> dict[str, RdUnitStatus]:
    """{gu: RdUnitStatus} for every unit of an available RD RA's GLSK (F2).
    - on: connected and (PROD != 0, or PROD = 0 with Pmin <= 0); PROD = the ORIGINAL target_p,
      |PROD| <= PROD_ZERO_TOLERANCE_MW counts as 0.
    - binary on/off for every unit with Pmin > 0, whether its bids are single- or multi-unit
      (F2 rule 9, after Open RAO: one binary per generator); Pmin <= 0 is CONTINUOUS.
    - STARTABLE: off, a start-up cost and Pmin <= PROD + sum over the preventive up bids of
      participation factor x available volume; otherwise FIXED_OFF with off_reason (the static
      check). STOPPABLE: on, a shut-down cost and PROD - the same sum over the down bids <= 0;
      otherwise CONTINUOUS.
    - Pmin := PROD for a unit with Pmin > 0 and 0 < PROD < Pmin (one log line per unit and run).
    - Pmin > Pmax: one warning per unit and run (such a unit can never be on)."""
    available = {str(rd.mrid): rd for rd in rd_inputs.rd_list if rd.available}
    original = rd_inputs.original_target_p_mw or {}
    unit_costs = rd_inputs.unit_costs or {}
    up_mw: dict[str, float] = {}
    down_mw: dict[str, float] = {}
    for bid in rd_inputs.bid_list:
        rd = available.get(str(bid.ra_mrid))
        if rd is None or rd.kind != RAKind.PREVENTIVE or rd.countertrade:
            continue    # a countertrade bid never starts a unit (C3.9)
        for gu, factor in _participation_of_bid(rd_inputs, bid).items():
            target = up_mw if bid.direction == BidDirection.UP else down_mw
            target[gu] = target.get(gu, 0.0) + float(factor) * float(bid.available_mw.get(timestamp, 0.0))

    status_units = {g for ra in available for g in rd_inputs.glsk.get(ra, {})}
    for bid in rd_inputs.bid_list:
        ct_rd = available.get(str(bid.ra_mrid))
        if ct_rd is not None and ct_rd.countertrade:
            status_units.update(_participation_of_bid(rd_inputs, bid))    # CT-only units get a status too (C4.2)

    status: dict[str, RdUnitStatus] = {}
    for gu in sorted(status_units):
        data = rd_inputs.generator_data.get(gu, {})
        prod = float(original.get(gu, data.get("target_p", data.get("initial_p", 0.0))))
        if abs(prod) <= PROD_ZERO_TOLERANCE_MW:
            prod = 0.0
        pmin, pmax = float(data.get("min_p", 0.0)), float(data.get("max_p", 0.0))
        if pmin > pmax:
            _log_if_changed(
                f"pmin_above_pmax_{gu}", (pmin, pmax),
                "generating unit %s: Pmin %.3f MW is above Pmax %.3f MW - the unit can never be on "
                "(check the CGM limits)", gu, pmin, pmax, level=logging.WARNING,
            )
        connected = bool(data.get("connected", True))
        initially_on = connected and (prod != 0.0 or pmin <= 0.0)
        if pmin > 0.0 and 0.0 < prod < pmin:
            _log_if_changed(
                f"pmin_set_to_prod_{gu}", (prod, pmin),
                "generating unit %s: PROD %.3f MW below Pmin %.3f MW - Pmin set to PROD (CORNET-37)",
                gu, prod, pmin,
            )
            pmin = prod
        cost = unit_costs.get(gu)
        startup = cost.startup_cost_eur if cost is not None else None
        shutdown = cost.shutdown_cost_eur if cost is not None else None
        off_reason = None
        if pmin <= 0.0:
            unit_class = UNIT_CONTINUOUS
        elif not initially_on:
            reasons = [reason for reason, applies in (
                (NO_STARTUP_COST_REASON, startup is None),
                (PMIN_UNREACHABLE_REASON, pmin > prod + up_mw.get(gu, 0.0)),
            ) if applies]
            unit_class = UNIT_FIXED_OFF if reasons else UNIT_STARTABLE
            off_reason = "; ".join(reasons) or None
        else:
            unit_class = (UNIT_STOPPABLE if shutdown is not None
                          and prod - down_mw.get(gu, 0.0) <= PROD_ZERO_TOLERANCE_MW
                          else UNIT_CONTINUOUS)
        status[gu] = RdUnitStatus(initially_on, unit_class, pmin, pmax, prod, startup, shutdown, off_reason)
    return status


def _block_bids_of_fixed_off_units(bid_ids, bid_by_mrid, rd_inputs, unit_status, kind: str) -> set[str]:
    """F2 rule 9 static check: every bid (pf > 0) containing a FIXED_OFF unit is blocked; one log
    line per blocked bid and run names the unit and the reason. Returns the blocked bid mRIDs;
    the caller sets their volume bounds to 0."""
    blocked: set[str] = set()
    for bid_mrid in bid_ids:
        participation = _participation_of_bid(rd_inputs, bid_by_mrid[bid_mrid])
        for gu_mrid in sorted(participation):
            status = unit_status.get(gu_mrid)
            if float(participation[gu_mrid]) > 0.0 and status is not None and status.unit_class == UNIT_FIXED_OFF:
                blocked.add(bid_mrid)
                _log_if_changed(
                    f"blocked_{kind}_{bid_mrid}", gu_mrid,
                    "%s bid %s blocked (volume 0): generating unit %s is off (PROD 0, Pmin %.3f MW > 0) "
                    "and cannot be started - %s", kind, bid_mrid, gu_mrid, status.pmin_mw, status.off_reason,
                )
                break
    return blocked


def validate_participation_factors(glsk: dict[str, dict[str, float]], owner: str = "GLSK of RD remedial action") -> None:
    """Every GLSK participation factor must be >= 0: the per-unit curative Pmin/Pmax rows
    (CORNET-4614) and the max-over-contingencies rule assume it. ValueError naming the owner (the RA,
    or the countertrade bid for RdMilpInputs.bid_participation) and the generating unit otherwise."""
    for ra_mrid, participation in sorted(glsk.items()):
        for gu_mrid, factor in sorted(participation.items()):
            if float(factor) < 0.0:
                raise ValueError(
                    f"{owner} {ra_mrid}: generating unit {gu_mrid} has a negative "
                    f"participation factor ({factor}); participation factors must be >= 0"
                )


def _add_rd_block(model, pyo, rd_inputs: RdMilpInputs, timestamp: str, branch_ids: list[str]) -> bool:
    """Add the RD variables, model.rd_net_injection_g / rd_effect_on_branch and the Pmax/Pmin
    constraints, for bids of PREVENTIVE RD units only (curative bids get their
    contingency-gated variables in _add_curative_rd_block).
    Returns False and adds nothing when there are no bids or no generating unit is
    reachable via GLSK (a legitimate empty RD block, not an error).
    """
    ra_kind_by_mrid: dict[str, RAKind] = {str(rd.mrid): rd.kind for rd in rd_inputs.rd_list}
    # An unavailable RD unit (RDUnit.available False, NC normalAvailable) gets
    # no bid variables, no RA-count variable, no group-limit entry and no cost
    # term — same rule as for PSTs.
    available_ra_mrids = {str(rd.mrid) for rd in rd_inputs.rd_list if rd.available}
    preventive_bid_list = [
        bid for bid in rd_inputs.bid_list
        if ra_kind_by_mrid.get(str(bid.ra_mrid)) == RAKind.PREVENTIVE
        and str(bid.ra_mrid) in available_ra_mrids
    ]
    curative_bid_count = sum(1 for bid in rd_inputs.bid_list if ra_kind_by_mrid.get(str(bid.ra_mrid)) == RAKind.CURATIVE)
    if curative_bid_count:
        _log_if_changed(
            "curative_rd_bid_split", (curative_bid_count, len(rd_inputs.bid_list)),
            "_add_rd_block: %d of %d bid(s) belong to a curative RD RA — "
            "excluded from the preventive rd_volume_upward/downward variables "
            "(handled separately by _add_curative_rd_block, if this "
            "workflow's nx_inputs provides curative RD data)",
            curative_bid_count, len(rd_inputs.bid_list),
        )

    gu_ids_reachable = _gu_ids_reachable(rd_inputs)

    if not preventive_bid_list or not gu_ids_reachable:
        log.warning(
            "RD block skipped — preventive bid_list has %d entries (of %d total; "
            "%d curative), %d generating unit(s) reachable via GLSK (need both "
            "non-zero). This is the confirmed empty-GLSK/empty-bid dataset "
            "boundary condition on the current Jotunheim dataset cut (see "
            "power-bid and GLSK findings) if this is unexpected, check "
            "bid_list and glsk directly before assuming a bug.",
            len(preventive_bid_list), len(rd_inputs.bid_list), curative_bid_count, len(gu_ids_reachable),
        )
        return False

    missing_generator_data = [g for g in gu_ids_reachable if g not in rd_inputs.generator_data]
    if missing_generator_data:
        log.warning(
            "Generating unit(s) reachable via GLSK but missing from "
            "generator_data (network.get_generators()) — excluded from "
            "Pmax/Pmin and rd_net_injection_g: %s",
            missing_generator_data,
        )
        gu_ids_reachable = [g for g in gu_ids_reachable if g not in missing_generator_data]
        if not gu_ids_reachable:
            log.warning("RD block skipped — no generating unit has both GLSK participation and generator_data")
            return False

    bid_ids = [str(bid.mrid) for bid in preventive_bid_list]
    bid_by_mrid: dict[str, PowerBid] = {str(bid.mrid): bid for bid in preventive_bid_list}

    model.bid_ids = pyo.Set(initialize=bid_ids)
    model.gu_ids = pyo.Set(initialize=gu_ids_reachable)

    model.rd_volume_upward = pyo.Var(model.bid_ids, domain=pyo.NonNegativeReals)
    model.rd_volume_downward = pyo.Var(model.bid_ids, domain=pyo.NonNegativeReals)

    # Upper bounds per bid direction (a PowerBidSchedule is UP or DOWN): the other direction
    # is bounded to 0.0. CUMULATIVE headroom: with `used` MW already applied in the bid's
    # direction, this solve may add (availability - used) and take back at most `used`.
    used_signed_by_bid = rd_inputs.cumulative_signed_volume_mw or {}
    used_magnitude_by_bid: dict[str, float] = {}
    for bid_mrid in bid_ids:
        bid = bid_by_mrid[bid_mrid]
        signed_used = float(used_signed_by_bid.get(bid_mrid, 0.0))
        used_magnitude_by_bid[bid_mrid] = max(signed_used if bid.direction == BidDirection.UP else -signed_used, 0.0)
    for bid_mrid in bid_ids:
        bid = bid_by_mrid[bid_mrid]
        available_mw = float(bid.available_mw.get(timestamp, 0.0))
        used_mw = used_magnitude_by_bid[bid_mrid]
        if bid.direction == BidDirection.UP:
            model.rd_volume_upward[bid_mrid].setub(max(available_mw - used_mw, 0.0))
            model.rd_volume_downward[bid_mrid].setub(used_mw)
        else:
            model.rd_volume_downward[bid_mrid].setub(max(available_mw - used_mw, 0.0))
            model.rd_volume_upward[bid_mrid].setub(used_mw)

    # cumulative volume per bid and direction: the plain up/down variables when nothing is
    # used yet, otherwise used + this solve's change
    model.rd_used_signed_volume_mw = {b: float(used_signed_by_bid.get(b, 0.0)) for b in bid_ids}
    model.rd_cumulative_up_term = {}
    model.rd_cumulative_down_term = {}
    model.rd_cumulative_volume_term = {}
    for bid_mrid in bid_ids:
        bid = bid_by_mrid[bid_mrid]
        used_mw = used_magnitude_by_bid[bid_mrid]
        up_var, down_var = model.rd_volume_upward[bid_mrid], model.rd_volume_downward[bid_mrid]
        if used_mw == 0.0:
            model.rd_cumulative_volume_term[bid_mrid] = up_var + down_var
            model.rd_cumulative_up_term[bid_mrid] = up_var
            model.rd_cumulative_down_term[bid_mrid] = down_var
        elif bid.direction == BidDirection.UP:
            cumulative = used_mw + up_var - down_var
            model.rd_cumulative_volume_term[bid_mrid] = cumulative
            model.rd_cumulative_up_term[bid_mrid] = cumulative
            model.rd_cumulative_down_term[bid_mrid] = 0.0
        else:
            cumulative = used_mw + down_var - up_var
            model.rd_cumulative_volume_term[bid_mrid] = cumulative
            model.rd_cumulative_up_term[bid_mrid] = 0.0
            model.rd_cumulative_down_term[bid_mrid] = cumulative
    model.rd_cumulative_used_in_any_bid = any(v != 0.0 for v in used_magnitude_by_bid.values())

    # lead time: after the direction bounds (it only tightens): a bid whose lead time is not
    # satisfied at this timestamp gets setub(0.0)
    radp = _compute_radp(rd_inputs.raart, timestamp)
    for bid_mrid in bid_ids:
        bid = bid_by_mrid[bid_mrid]
        if bid.lead_time_h <= 0:
            continue  # no lead time restriction for this bid
        if not _timestamp_is_usable(radp, bid.lead_time_h, timestamp):
            model.rd_volume_upward[bid_mrid].setub(0.0)
            model.rd_volume_downward[bid_mrid].setub(0.0)
            log.info(
                "Bid %s: lead_time_h=%.2f not satisfied by radp=%s at timestamp=%s "
                "— rd_volume_upward/downward bounds fixed to 0.0",
                bid_mrid, bid.lead_time_h, radp, timestamp,
            )

    # a bid whose GLSK has no modelled unit (empty, e.g. all its units unconnected) moves nothing:
    # both bounds 0, so it is never reported as used (F2 rule 13); a countertrade bid with empty weights
    # is already logged once per RA and direction by the weights (C3.7)
    empty_ct_bid_ids = _ct_bids_without_weights(rd_inputs, bid_by_mrid, bid_ids)
    for bid_mrid in bid_ids:
        participation = _participation_of_bid(rd_inputs, bid_by_mrid[bid_mrid])
        if not any(gu_mrid in gu_ids_reachable for gu_mrid in participation):
            model.rd_volume_upward[bid_mrid].setub(0.0)
            model.rd_volume_downward[bid_mrid].setub(0.0)
            if bid_mrid in empty_ct_bid_ids:
                continue
            _log_if_changed(
                f"empty_glsk_{bid_mrid}", True,
                "Bid %s (RA %s): no generating unit of its GLSK is in the model (empty GLSK) - "
                "rd_volume_upward/downward fixed to 0", bid_mrid, bid_by_mrid[bid_mrid].ra_mrid,
            )

    def rd_net_injection_rule(m, gu_mrid):
        terms = [
            float(participation_factor) * (m.rd_volume_upward[bid_mrid] - m.rd_volume_downward[bid_mrid])
            for bid_mrid in bid_ids
            for participation_factor in [_participation_of_bid(rd_inputs, bid_by_mrid[bid_mrid]).get(gu_mrid)]
            if participation_factor is not None
        ]
        return sum(terms) if terms else 0.0

    model.rd_net_injection_g = pyo.Expression(model.gu_ids, rule=rd_net_injection_rule)

    # countertrade (docs/specs/countertrade_ra_specs.md C4.2): the CT bids share rd_net_injection_g and the rows
    # below; rd_only_net_injection_g is the RD part of the unit total. In the flow rows a CT bid acts through the
    # sensitivity of its RA and direction (2 terms per RA, DECISIONS 169), the RD bids through their units
    ct_ra_mrids = _ct_ra_mrids(rd_inputs)
    ct_bid_ids = [b for b in bid_ids if str(bid_by_mrid[b].ra_mrid) in ct_ra_mrids]
    model.ct_bid_ids = frozenset(ct_bid_ids)
    model.ct_bid_variable = {
        b: (ct_sensitivity.ct_variable_id(bid_by_mrid[b].ra_mrid, bid_by_mrid[b].direction),
            _participation_of_bid(rd_inputs, bid_by_mrid[b]))
        for b in ct_bid_ids
    }
    if ct_bid_ids:
        def rd_only_net_injection_rule(m, gu_mrid):
            terms = [
                float(factor) * (m.rd_volume_upward[b] - m.rd_volume_downward[b])
                for b in bid_ids if b not in model.ct_bid_ids
                for factor in [_participation_of_bid(rd_inputs, bid_by_mrid[b]).get(gu_mrid)]
                if factor is not None
            ]
            return sum(terms) if terms else 0.0

        model.rd_only_net_injection_g = pyo.Expression(model.gu_ids, rule=rd_only_net_injection_rule)
        rd_units_in_rows = sorted({
            g for b in bid_ids if b not in model.ct_bid_ids for g in _participation_of_bid(rd_inputs, bid_by_mrid[b])
        } & set(gu_ids_reachable))
    else:
        rd_units_in_rows = gu_ids_reachable
    model.rd_effect_unit_ids = rd_units_in_rows
    flow_injection = model.rd_only_net_injection_g if ct_bid_ids else model.rd_net_injection_g

    def rd_effect_on_branch_rule(m, branch_mrid):
        effect = sum(
            float(rd_inputs.sensitivity_rd_n0_pruned.get((branch_mrid, gu_mrid), 0.0))
            * flow_injection[gu_mrid]
            for gu_mrid in rd_units_in_rows
        )
        for bid_mrid in ct_bid_ids:
            variable_id, weights = m.ct_bid_variable[bid_mrid]
            coefficient = ct_sensitivity.coefficient(rd_inputs.sensitivity_rd_n0_pruned, branch_mrid, variable_id, weights)
            if coefficient != 0.0:
                effect = effect + coefficient * (m.rd_volume_upward[bid_mrid] - m.rd_volume_downward[bid_mrid])
        return effect

    model.rd_effect_on_branch = pyo.Expression(branch_ids, rule=rd_effect_on_branch_rule)

    # generator on/off (F2), classes from the ORIGINAL production (_rd_unit_status)
    unit_status = _rd_unit_status(rd_inputs, timestamp)
    model.rd_unit_status = unit_status

    # a unit that is off and cannot be started: every bid containing it stays at 0 (no free
    # start-up; in a multi-unit bid the unit's share could not reach Pmin either), F2 rule 9
    for bid_mrid in _block_bids_of_fixed_off_units(bid_ids, bid_by_mrid, rd_inputs, unit_status, "Preventive"):
        model.rd_volume_upward[bid_mrid].setub(0.0)
        model.rd_volume_downward[bid_mrid].setub(0.0)

    binary_gu_ids = [
        gu for gu in gu_ids_reachable
        if unit_status.get(gu) is not None and unit_status[gu].unit_class in (UNIT_STARTABLE, UNIT_STOPPABLE)
    ]
    model.rd_binary_gu_ids = pyo.Set(initialize=binary_gu_ids)
    model.rd_unit_on = pyo.Var(model.rd_binary_gu_ids, domain=pyo.Binary)

    def rd_unit_output_rule(m, gu_mrid):
        """PROD0 + cumulative preventive injection (MW): the output the on/off binary judges,
        independent of the refreshed target_p of a later linearisation pass."""
        return unit_status[gu_mrid].original_p_mw + sum(
            float(factor) * (m.rd_cumulative_up_term[bid_mrid] - m.rd_cumulative_down_term[bid_mrid])
            for bid_mrid in bid_ids
            for factor in [_participation_of_bid(rd_inputs, bid_by_mrid[bid_mrid]).get(gu_mrid)]
            if factor is not None
        )

    model.rd_unit_output_after_preventive = pyo.Expression(model.rd_binary_gu_ids, rule=rd_unit_output_rule)

    # semi-continuous output: on -> Pmin <= P <= Pmax, off -> P = 0 (CORNET-37)
    model.rd_unit_on_lower = pyo.Constraint(
        model.rd_binary_gu_ids,
        rule=lambda m, g: m.rd_unit_output_after_preventive[g] >= unit_status[g].pmin_mw * m.rd_unit_on[g],
    )
    model.rd_unit_on_upper = pyo.Constraint(
        model.rd_binary_gu_ids,
        rule=lambda m, g: m.rd_unit_output_after_preventive[g] <= unit_status[g].pmax_mw * m.rd_unit_on[g],
    )

    def pmax_pmin_rule(m, gu_mrid):
        gu_data = rd_inputs.generator_data.get(gu_mrid, {})
        target_p = float(gu_data.get("target_p", gu_data.get("initial_p", 0.0)))
        max_p = float(gu_data.get("max_p", 0.0))
        status = unit_status.get(gu_mrid)
        if status is not None and status.unit_class in (UNIT_STARTABLE, UNIT_STOPPABLE):
            return pyo.Constraint.Skip  # rd_unit_on_lower / rd_unit_on_upper
        if status is not None and status.unit_class == UNIT_FIXED_OFF:
            return (None, target_p + m.rd_net_injection_g[gu_mrid], max_p)  # off: no Pmin bound
        min_p = status.pmin_mw if status is not None else float(gu_data.get("min_p", 0.0))
        return (min_p, target_p + m.rd_net_injection_g[gu_mrid], max_p)

    model.pmax_pmin_constraint = pyo.Constraint(model.gu_ids, rule=pmax_pmin_rule)

    # a unit with an on/off binary u receives CT up and CT down volume only while it is on (one-sided rows, so up
    # and down of different bids cannot cancel on an off unit)
    if ct_bid_ids:
        ct_unit_ids = sorted({
            g for b in ct_bid_ids for g in _participation_of_bid(rd_inputs, bid_by_mrid[b])
            if g in binary_gu_ids and g in gu_ids_reachable
        })

        def ct_side(m, gu_mrid, cumulative_term):
            return sum(
                float(factor) * cumulative_term[b]
                for b in ct_bid_ids
                for factor in [_participation_of_bid(rd_inputs, bid_by_mrid[b]).get(gu_mrid)]
                if factor is not None
            )

        model.ct_on_off_unit_ids = pyo.Set(initialize=ct_unit_ids)
        model.ct_on_off_up = pyo.Constraint(
            model.ct_on_off_unit_ids,
            rule=lambda m, g: ct_side(m, g, m.rd_cumulative_up_term) <= unit_status[g].pmax_mw * m.rd_unit_on[g],
        )
        model.ct_on_off_down = pyo.Constraint(
            model.ct_on_off_unit_ids,
            rule=lambda m, g: ct_side(m, g, m.rd_cumulative_down_term) <= unit_status[g].pmax_mw * m.rd_unit_on[g],
        )

    # bid activation (BUC 14#102): a binary only for a bid with an activation cost > 0; the
    # bid's cumulative volume needs it
    activation_bid_ids = [b for b in bid_ids if float(bid_by_mrid[b].activation_cost_eur or 0.0) > 0.0]
    model.rd_activation_bid_ids = pyo.Set(initialize=activation_bid_ids)
    model.rd_bid_activated = pyo.Var(model.rd_activation_bid_ids, domain=pyo.Binary)
    model.rd_bid_activation_link = pyo.Constraint(
        model.rd_activation_bid_ids,
        rule=lambda m, b: m.rd_cumulative_volume_term[b]
        <= float(bid_by_mrid[b].available_mw.get(timestamp, 0.0)) * m.rd_bid_activated[b],
    )

    # the production baseline (target_p, MW) this model was built on, for the result
    # exporters (not the original CGM value after a rebuild); no model term depends on it
    model.rd_original_target_p_mw = {
        str(gu_mrid): float(
            (rd_inputs.original_target_p_mw or {}).get(
                str(gu_mrid),
                rd_inputs.generator_data.get(gu_mrid, {}).get(
                    "target_p", rd_inputs.generator_data.get(gu_mrid, {}).get("initial_p", 0.0)),
            )
        )
        for gu_mrid in gu_ids_reachable
    }
    model.rd_baseline_target_p_mw = {
        str(gu_mrid): float(
            rd_inputs.generator_data.get(gu_mrid, {}).get(
                "target_p", rd_inputs.generator_data.get(gu_mrid, {}).get("initial_p", 0.0))
        )
        for gu_mrid in gu_ids_reachable
    }

    return True


# ---------------------------------------------------------------------------
# HVDC (docs/specs/hvdc_ra_specs.md H5)
# ---------------------------------------------------------------------------

HVDC_CHANGE_TOLERANCE_MW = 1e-9


def _add_hvdc_block(model, pyo, hvdc_inputs: HvdcMilpInputs, branch_ids: list[str]) -> bool:
    """The preventive HVDC block; returns False and adds nothing without a preventive unit.
    Per unit u (a remedial action on one EI): dP+_u, dP-_u >= 0 (hvdc_dp_up/down, continuous, no
    binaries), the change of THIS solve hvdc_change[u] = dP+ - dP- against the set point P of the
    operating point (hvdc_inputs.current_p_mw). hvdc_bounds: Pmin_op <= P + change <= Pmax_op (the limits
    are anchored on the ORIGINAL set point). hvdc_group_balance: the changes of the two units of a
    group sum to 0 (generation sign: equal change, opposite direction), hard, and NOT part of the RD
    balance. hvdc_effect_on_branch[b] = sum_u S_{b,u} * change[u] (A). hvdc_cost_volume[u]: |cumulative
    change against the ORIGINAL set point| in MW, the volume the penalty is charged on (dP+ + dP- while
    nothing is applied yet, otherwise through hvdc_cum_up/down)."""
    units = hvdc_inputs.preventive_units
    if not units:
        return False
    unit_ids = [u.mrid for u in units]
    by_id = {u.mrid: u for u in units}
    group_ids = sorted({u.group_mrid for u in units})
    model.hvdc_unit_ids = pyo.Set(initialize=unit_ids)
    model.hvdc_group_ids = pyo.Set(initialize=group_ids)
    model.hvdc_dp_up = pyo.Var(model.hvdc_unit_ids, domain=pyo.NonNegativeReals)
    model.hvdc_dp_down = pyo.Var(model.hvdc_unit_ids, domain=pyo.NonNegativeReals)
    model.hvdc_change = pyo.Expression(model.hvdc_unit_ids, rule=lambda m, u: m.hvdc_dp_up[u] - m.hvdc_dp_down[u])

    current = {u: float(hvdc_inputs.current_p_mw[by_id[u].ei_mrid]) for u in unit_ids}
    original = {u: float(hvdc_inputs.original_p_mw[by_id[u].ei_mrid]) for u in unit_ids}

    def bounds_rule(m, u):
        lower, upper = hvdc_inputs.limits[u]
        return (float(lower), current[u] + m.hvdc_change[u], float(upper))

    model.hvdc_bounds = pyo.Constraint(model.hvdc_unit_ids, rule=bounds_rule)
    model.hvdc_group_balance = pyo.Constraint(
        model.hvdc_group_ids,
        rule=lambda m, g: sum(m.hvdc_change[u] for u in unit_ids if by_id[u].group_mrid == g) == 0.0,
    )

    # cumulative change against the original set point (the penalty is charged on it)
    applied_ids = [u for u in unit_ids if abs(current[u] - original[u]) > HVDC_CHANGE_TOLERANCE_MW]
    model.hvdc_applied_unit_ids = pyo.Set(initialize=applied_ids)
    model.hvdc_cum_up = pyo.Var(model.hvdc_applied_unit_ids, domain=pyo.NonNegativeReals)
    model.hvdc_cum_down = pyo.Var(model.hvdc_applied_unit_ids, domain=pyo.NonNegativeReals)
    model.hvdc_cumulative_link = pyo.Constraint(
        model.hvdc_applied_unit_ids,
        rule=lambda m, u: m.hvdc_cum_up[u] - m.hvdc_cum_down[u] == (current[u] - original[u]) + m.hvdc_change[u],
    )
    model.hvdc_cost_volume = {
        u: (model.hvdc_cum_up[u] + model.hvdc_cum_down[u]) if u in applied_ids
        else (model.hvdc_dp_up[u] + model.hvdc_dp_down[u])
        for u in unit_ids
    }

    def effect_rule(m, branch_mrid):
        return sum(
            float(hvdc_inputs.sensitivity_hvdc_n0.get((branch_mrid, by_id[u].ei_mrid), 0.0)) * m.hvdc_change[u]
            for u in unit_ids
        )

    model.hvdc_effect_on_branch = pyo.Expression(branch_ids, rule=effect_rule)
    model.hvdc_info = {
        u: {
            "ei_mrid": by_id[u].ei_mrid, "group_mrid": by_id[u].group_mrid, "tso": by_id[u].tso, "name": by_id[u].name,
            "penalty_factor": float(by_id[u].penalty_factor), "p_original_mw": original[u], "p_current_mw": current[u],
            "pmin_op_mw": float(hvdc_inputs.limits[u][0]), "pmax_op_mw": float(hvdc_inputs.limits[u][1]),
        }
        for u in unit_ids
    }
    return True


def _add_curative_hvdc_block(model, pyo, nx_inputs: NxMilpInputs, hvdc_inputs: HvdcMilpInputs, branch_limits: dict) -> int:
    """The curative HVDC variables and constraints (docs/specs/hvdc_ra_specs.md H6, H7.4); returns the number of
    (unit, contingency) pairs, 0 and nothing added without any.
    A group (two curative units) has variables in a contingency c when
      - R2: both units apply in c (a unit without linked contingencies applies in every contingency of lf_nx),
      - the gate: no EI of the group is an outage element of c (hvdc_inputs.outage_eis_by_contingency),
      - the N-X guard: no EI of the group has exactly zero sensitivities in c (hvdc_inputs.zero_units_nx),
      - R9: the group is available on at least one monitored branch (both units' reaction time <= the
        branch's TATL duration, the helper of the RD units; hvdc_cra_availability).
    The group is atomic: a unit that fails one of these takes its partner with it (no variables for either).
    hvdc_cur_up/down[u, c] >= 0 (continuous, no binaries); hvdc_cur_bounds: Pmin_op <= P + preventive change +
    the curative changes of c <= Pmax_op (exact per contingency; every curative unit of the same EI active in c
    counts); hvdc_cur_balance[group, c]: the curative changes of the two units sum to 0 (outside the R15 balance
    of the curative RD); hvdc_cur_max[u] >= up + down for every c (lower-bounded only: the cost, factor x
    reference x max over c, pushes it down; the exact bounds above stop any inflation). Curative actions are
    re-optimised in every solve and never applied."""
    units = hvdc_inputs.curative_units
    if not units or not nx_inputs.lf_nx:
        return 0
    by_group: dict[str, list[HvdcUnit]] = {}
    for unit in units:
        by_group.setdefault(unit.group_mrid, []).append(unit)
    availability = compute_rd_cra_availability(
        units, nx_inputs.tatl_data or {}, branch_mrids=list(branch_limits),
        fallback_duration_seconds=nx_inputs.tatl_fallback_duration_seconds,
    )
    model.hvdc_cra_availability = availability

    def available_on(group_units, branch_mrid) -> bool:
        return all(availability[(u.mrid, branch_mrid)]["available"] for u in group_units)

    group_contingencies: list[tuple[str, str]] = []
    skipped: dict[str, list[str]] = {}
    # hvdc_cra_availability.csv rows: per (unit, branch) the reaction-time rule, per (unit, contingency) the gate, each
    # with the reason it is unavailable (the one source of the rules above)
    availability_rows: list[dict] = []
    model.hvdc_cra_availability_rows = availability_rows

    def add_row(unit, scope, available, reason, branch_mrid="", contingency_mrid="", duration=""):
        availability_rows.append({
            "ra_mrid": unit.mrid, "tso": unit.tso, "group_mrid": unit.group_mrid, "ei_mrid": unit.ei_mrid, "scope": scope,
            "branch_mrid": branch_mrid, "contingency_mrid": contingency_mrid,
            "reaction_time_minutes": "" if unit.reaction_time_minutes is None else float(unit.reaction_time_minutes),
            "tatl_duration_seconds": duration, "available": bool(available), "reason": reason,
        })

    for group_mrid, members in sorted(by_group.items()):
        for branch_mrid in sorted(branch_limits):
            for unit in members:
                entry = availability[(unit.mrid, branch_mrid)]
                partners = [p for p in members if p is not unit and not availability[(p.mrid, branch_mrid)]["available"]]
                if not entry["available"]:
                    reason = "reaction time above the TATL duration"
                elif partners:
                    reason = f"group atomicity: partner {partners[0].mrid} has a reaction time above the TATL duration"
                else:
                    reason = ""
                add_row(unit, "branch", not reason, reason, branch_mrid=branch_mrid, duration=entry["tatl_duration_seconds"])
        if len(members) != 2:
            skipped.setdefault(f"group {group_mrid} has {len(members)} curative unit(s)", []).append("*")
            continue
        group_reaction_ok = any(available_on(members, branch_mrid) for branch_mrid in branch_limits)
        if not group_reaction_ok:
            skipped.setdefault(f"group {group_mrid}: reaction time above the TATL duration on every branch (R9)", []).append("*")
        for contingency_mrid in sorted(nx_inputs.lf_nx):
            if any((u.linked_contingency_mrids and contingency_mrid not in u.linked_contingency_mrids)
                   or contingency_mrid in u.excluded_contingency_mrids for u in members):
                continue                                              # not linked (R2) or excluded: no row
            outage = hvdc_inputs.outage_eis_by_contingency.get(contingency_mrid, frozenset())
            zero = set(hvdc_inputs.zero_units_nx.get(contingency_mrid, ()))
            reasons = {}
            for unit in members:
                if unit.ei_mrid in outage:
                    reasons[unit.mrid] = "outage element"
                elif unit.ei_mrid in zero:
                    reasons[unit.mrid] = "zero sensitivity"
                elif not group_reaction_ok:
                    reasons[unit.mrid] = "reaction time above the TATL duration on every branch"
            for unit in members:
                partner = next(p for p in members if p is not unit)
                if unit.mrid in reasons:
                    reason = reasons[unit.mrid]
                elif partner.mrid in reasons:
                    reason = f"group atomicity: partner {partner.mrid}: {reasons[partner.mrid]}"
                else:
                    reason = ""
                add_row(unit, "contingency", not reason, reason, contingency_mrid=contingency_mrid)
            if not group_reaction_ok:
                continue
            if any(u.ei_mrid in outage for u in members):
                skipped.setdefault(f"group {group_mrid}: an EI is an outage element", []).append(contingency_mrid)
                continue
            if any(u.ei_mrid in zero for u in members):
                skipped.setdefault(f"group {group_mrid}: exactly zero sensitivities of an EI", []).append(contingency_mrid)
                continue
            group_contingencies.append((group_mrid, contingency_mrid))
    for reason, where in sorted(skipped.items()):
        _log_if_changed(
            f"hvdc_curative_skipped_{reason}", tuple(where),
            "Curative HVDC: %s - unavailable in contingency(ies) %s (both units of the group are fixed to zero there)",
            reason, "all" if where == ["*"] else sorted(where),
        )
    if not group_contingencies:
        return 0

    pairs = sorted((u.mrid, c) for group_mrid, c in group_contingencies for u in by_group[group_mrid])
    unit_by_id = {u.mrid: u for u in units}
    model.hvdc_cur_pairs = pyo.Set(initialize=pairs, dimen=2)
    model.hvdc_cur_group_contingencies = pyo.Set(initialize=sorted(group_contingencies), dimen=2)
    model.hvdc_cur_unit_ids = pyo.Set(initialize=sorted({u for u, _c in pairs}))
    model.hvdc_cur_up = pyo.Var(model.hvdc_cur_pairs, domain=pyo.NonNegativeReals)
    model.hvdc_cur_down = pyo.Var(model.hvdc_cur_pairs, domain=pyo.NonNegativeReals)

    def net(m, unit_mrid, contingency_mrid):
        return m.hvdc_cur_up[unit_mrid, contingency_mrid] - m.hvdc_cur_down[unit_mrid, contingency_mrid]

    pair_set = set(pairs)

    def preventive_change(m, ei_mrid):
        if not hasattr(m, "hvdc_change"):
            return 0.0
        return sum((m.hvdc_change[v] for v in m.hvdc_unit_ids if m.hvdc_info[v]["ei_mrid"] == ei_mrid), 0.0)

    def bounds_rule(m, unit_mrid, contingency_mrid):
        unit = unit_by_id[unit_mrid]
        lower, upper = hvdc_inputs.limits[unit_mrid]
        same_ei = [v for v, c in pairs if c == contingency_mrid and unit_by_id[v].ei_mrid == unit.ei_mrid]
        total = float(hvdc_inputs.current_p_mw[unit.ei_mrid]) + preventive_change(m, unit.ei_mrid) + sum(
            net(m, v, contingency_mrid) for v in same_ei
        )
        return (float(lower), total, float(upper))

    model.hvdc_cur_bounds = pyo.Constraint(model.hvdc_cur_pairs, rule=bounds_rule)
    model.hvdc_cur_balance = pyo.Constraint(
        model.hvdc_cur_group_contingencies,
        rule=lambda m, g, c: sum(net(m, u.mrid, c) for u in by_group[g]) == 0.0,
    )
    model.hvdc_cur_max = pyo.Var(model.hvdc_cur_unit_ids, domain=pyo.NonNegativeReals)
    model.hvdc_cur_max_lower = pyo.Constraint(
        model.hvdc_cur_pairs,
        rule=lambda m, u, c: m.hvdc_cur_max[u] >= m.hvdc_cur_up[u, c] + m.hvdc_cur_down[u, c],
    )
    model.hvdc_cur_info = {
        u: {
            "ei_mrid": unit_by_id[u].ei_mrid, "group_mrid": unit_by_id[u].group_mrid, "tso": unit_by_id[u].tso,
            "name": unit_by_id[u].name, "penalty_factor": float(unit_by_id[u].penalty_factor),
            "p_current_mw": float(hvdc_inputs.current_p_mw[unit_by_id[u].ei_mrid]),
            "pmin_op_mw": float(hvdc_inputs.limits[u][0]), "pmax_op_mw": float(hvdc_inputs.limits[u][1]),
        }
        for u in model.hvdc_cur_unit_ids
    }
    model.hvdc_cur_group_members = {g: [u.mrid for u in members] for g, members in by_group.items()}
    model.hvdc_cur_units_by_contingency = {}
    for unit_mrid, contingency_mrid in pairs:
        model.hvdc_cur_units_by_contingency.setdefault(contingency_mrid, []).append(unit_mrid)
    log.info(
        "Curative HVDC: %d (unit, contingency) pair(s) for %d group(s) in %d contingency(ies)",
        len(pairs), len({g for g, _c in group_contingencies}), len({c for _g, c in group_contingencies}),
    )
    return len(pairs)


# ---------------------------------------------------------------------------
# N-X PATL and TATL loading constraints
# ---------------------------------------------------------------------------

def _add_curative_pst_block(
    model,
    pyo,
    nx_inputs: NxMilpInputs,
    pst_inputs: PstMilpInputs,
) -> int:
    """Curative PST variables, curative tap bounds and magnitude constraints; returns the
    number of (pst, contingency) pairs. Called by _add_nx_block when pst_cra_availability is
    not empty. One variable pair per (PST, contingency); each available (pst, xne, c) entry
    adds its own magnitude bound on it, so the tightest max_curative_taps binds.
    """
    pst_cra_availability = nx_inputs.pst_cra_availability
    available_curative_mrids = {str(pst.mrid) for pst in pst_inputs.available_curative_pst_list}

    curative_pst_contingency_pairs = sorted({
        (pst_mrid, co_mrid)
        for (pst_mrid, xne_mrid, co_mrid), entry in pst_cra_availability.items()
        if entry.get("available") and str(pst_mrid) in available_curative_mrids
    })

    if not curative_pst_contingency_pairs:
        log.info(
            "_add_curative_pst_block: pst_cra_availability has entries but none "
            "with available=True — no curative PST variables added (genuine "
            "outcome, not an error; see nx_sensitivity.compute_pst_cra_availability)"
        )
        return 0

    model.curative_pst_contingency_pairs = pyo.Set(initialize=curative_pst_contingency_pairs, dimen=2)
    # same domain as the preventive taps: whole taps when pst_taping_discrete (the spec makes it one switch for preventive and curative)
    curative_tap_domain = pyo.NonNegativeIntegers if pst_inputs.pst_taping_discrete else pyo.NonNegativeReals
    model.curative_tap_change_upward = pyo.Var(
        model.curative_pst_contingency_pairs, domain=curative_tap_domain,
    )
    model.curative_tap_change_downward = pyo.Var(
        model.curative_pst_contingency_pairs, domain=curative_tap_domain,
    )

    pst_by_mrid: dict[str, PST] = {str(pst.mrid): pst for pst in pst_inputs.available_curative_pst_list}
    pst_tap_bounds_curative = nx_inputs.pst_tap_bounds_curative or {}
    preventive_pst_id_set = set(model.pst_ids) if hasattr(model, "pst_ids") else set()

    def curative_tap_bounds_rule(m, pst_mrid, co_mrid):
        if pst_mrid not in pst_tap_bounds_curative:
            log.warning(
                "_add_curative_pst_block: PST %s has curative variables but no "
                "entry in pst_tap_bounds_curative — constraint skipped for "
                "(pst=%s, contingency=%s) rather than crashing or guessing bounds",
                pst_mrid, pst_mrid, co_mrid,
            )
            return pyo.Constraint.Skip
        curative_tap_min, curative_tap_max = pst_tap_bounds_curative[pst_mrid]
        current_tap = int(getattr(pst_by_mrid[pst_mrid], "r0", 0))
        # preventive delta only when the transformer also has a preventive RA (else 0)
        preventive_delta = (
            m.tap_change_upward[pst_mrid] - m.tap_change_downward[pst_mrid]
            if pst_mrid in preventive_pst_id_set else 0.0
        )
        combined_tap = (
            current_tap
            + preventive_delta
            + m.curative_tap_change_upward[pst_mrid, co_mrid]
            - m.curative_tap_change_downward[pst_mrid, co_mrid]
        )
        return (float(curative_tap_min), combined_tap, float(curative_tap_max))

    model.curative_tap_bounds = pyo.Constraint(
        model.curative_pst_contingency_pairs, rule=curative_tap_bounds_rule,
    )

    magnitude_triples = sorted(
        (pst_mrid, xne_mrid, co_mrid)
        for (pst_mrid, xne_mrid, co_mrid), entry in pst_cra_availability.items()
        if entry.get("available")
    )
    model.curative_tap_magnitude_triples = pyo.Set(initialize=magnitude_triples, dimen=3)

    def curative_tap_magnitude_rule(m, pst_mrid, xne_mrid, co_mrid):
        max_curative_taps = pst_cra_availability[(pst_mrid, xne_mrid, co_mrid)]["max_curative_taps"]
        return (
            m.curative_tap_change_upward[pst_mrid, co_mrid]
            + m.curative_tap_change_downward[pst_mrid, co_mrid]
            <= float(max_curative_taps)
        )

    model.curative_tap_magnitude = pyo.Constraint(
        model.curative_tap_magnitude_triples, rule=curative_tap_magnitude_rule,
    )

    # curative PST group: equal curative delta tap (up - down) per contingency, for triples
    # where both PSTs have a curative variable pair in that contingency
    curative_pair_set = set(curative_pst_contingency_pairs)
    group_triples = sorted(
        (pst_1, pst_2, co_mrid)
        for pst_1, pst_2, co_mrid in (nx_inputs.curative_pst_group_triples or [])
        if (pst_1, co_mrid) in curative_pair_set and (pst_2, co_mrid) in curative_pair_set
    )
    if group_triples:
        model.curative_pst_group_triples = pyo.Set(initialize=group_triples, dimen=3)

        def curative_pst_group_equality_rule(m, pst_1, pst_2, co_mrid):
            return (
                m.curative_tap_change_upward[pst_1, co_mrid] - m.curative_tap_change_downward[pst_1, co_mrid]
                == m.curative_tap_change_upward[pst_2, co_mrid] - m.curative_tap_change_downward[pst_2, co_mrid]
            )

        model.curative_pst_group_equality = pyo.Constraint(
            model.curative_pst_group_triples, rule=curative_pst_group_equality_rule,
        )

    return len(curative_pst_contingency_pairs)


def get_effective_rd_penalty_cost_per_mw(
    rd_list: list[RDUnit],
    penalty_reference_rd_pra: float,
    penalty_reference_rd_pra_per_tso: Optional[dict[str, float]] = None,
) -> dict[str, tuple[float, str]]:
    """{ra_mrid: (RA penalty_factor x penalty reference [EUR/MW], tso)}, with the
    per-TSO reference override applied. The single source of this product for
    the curative RD objective term AND results_curative_rd.csv (spec: the
    penalty is not recomputed a second time in the exporter)."""
    override = penalty_reference_rd_pra_per_tso or {}
    return {
        str(rd.mrid): (
            float(rd.penalty_factor) * float(override.get(rd.tso, penalty_reference_rd_pra)),
            rd.tso,
        )
        for rd in rd_list
    }


def curative_rd_reservation_price_eur_mw(bid: PowerBid) -> float:
    """The second component of the curative RD cost per MW (BUC 14#43): the bid's
    reservePrice when given, else its energy price (logged once per bid). The single source
    for the objective and results_curative_rd.csv."""
    if bid.reserve_price_eur_mw is not None:
        return float(bid.reserve_price_eur_mw)
    _log_if_changed(
        f"reserve_price_fallback_{bid.mrid}", True,
        "Curative bid %s has no reservePrice - its energy price %.3f EUR/MW is used for the "
        "reservation", bid.mrid, float(bid.price_eur_mw),
    )
    return float(bid.price_eur_mw)


def _add_curative_rd_block(
    model,
    pyo,
    nx_inputs: NxMilpInputs,
    rd_inputs: RdMilpInputs,
    timestamp: str,
) -> int:
    """Curative RD variables, operation gating, combined preventive + curative Pmax/Pmin and the
    max-over-contingencies variables for the objective; returns the number of (bid,
    contingency) pairs. A curative bid acts only on a unit in operation after the preventive
    actions (F2 status; a unit with an on/off binary follows it); otherwise it is fixed to 0.
    activate_cra_xne_linkage is checked per generating unit (allowed if ANY curative RD RA of
    the unit is linked to the XNE): curative_rd_net_injection_g has no per-bid identity.
    """
    ra_kind_by_mrid: dict[str, RAKind] = {str(rd.mrid): rd.kind for rd in rd_inputs.rd_list}
    rd_by_ra_mrid: dict[str, RDUnit] = {str(rd.mrid): rd for rd in rd_inputs.rd_list}
    available_ra_mrids = {str(rd.mrid) for rd in rd_inputs.rd_list if rd.available}
    curative_bid_list = [
        bid for bid in rd_inputs.bid_list
        if ra_kind_by_mrid.get(str(bid.ra_mrid)) == RAKind.CURATIVE
        and str(bid.ra_mrid) in available_ra_mrids
    ]
    ct_ra_mrids = _ct_ra_mrids(rd_inputs)

    # an RA linked to contingencies acts only in those; an RA without any contingency link acts
    # in every contingency with a usable N-X result (BUC 14#106)
    curative_bid_contingency_pairs: list[tuple[str, str]] = []
    for bid in curative_bid_list:
        rd_unit = rd_by_ra_mrid.get(str(bid.ra_mrid))
        if rd_unit is None:
            continue
        co_mrids = effective_linked_contingencies(
            rd_unit.linked_contingency_mrids, rd_unit.excluded_contingency_mrids, nx_inputs.lf_nx,
        )
        for co_mrid in co_mrids:
            if co_mrid in nx_inputs.lf_nx:
                curative_bid_contingency_pairs.append((str(bid.mrid), co_mrid))

    if not curative_bid_contingency_pairs:
        log.info(
            "_add_curative_rd_block: %d curative RD bid(s) found but none is linked to a "
            "usable contingency in nx_inputs.lf_nx — no curative RD variables added "
            "(genuine outcome, not an error)",
            len(curative_bid_list),
        )
        return 0

    curative_bid_contingency_pairs = sorted(set(curative_bid_contingency_pairs))
    model.curative_bid_contingency_pairs = pyo.Set(initialize=curative_bid_contingency_pairs, dimen=2)
    model.curative_rd_volume_upward = pyo.Var(model.curative_bid_contingency_pairs, domain=pyo.NonNegativeReals)
    model.curative_rd_volume_downward = pyo.Var(model.curative_bid_contingency_pairs, domain=pyo.NonNegativeReals)

    bid_by_mrid: dict[str, PowerBid] = {str(bid.mrid): bid for bid in curative_bid_list}

    # "in operation" (BUC 14#106, F2, R4 rule 5): a curative bid acts only if at least one unit of
    # its GLSK is on after the preventive actions: volume <= available MW x (sum of the on/off
    # binaries of its units), where a continuous unit that is on counts as a constant 1 (no row
    # needed). A bid containing a unit that cannot be started is blocked (F2 rule 9 static check);
    # the per-unit R7 rows keep an off unit's share at 0. A curative action never starts or
    # stops a unit.
    unit_status = getattr(model, "rd_unit_status", None) or _rd_unit_status(rd_inputs, timestamp)
    binary_gu_ids = set(model.rd_binary_gu_ids) if hasattr(model, "rd_binary_gu_ids") else set()
    curative_bid_ids_all = sorted({bid_mrid for bid_mrid, _co in curative_bid_contingency_pairs})
    blocked_by_unit = _block_bids_of_fixed_off_units(
        curative_bid_ids_all, bid_by_mrid, rd_inputs, unit_status, "Curative",
    )
    gating_units_by_bid: dict[str, list[str]] = {}
    gated_by_binary: list[tuple[str, str]] = []

    def _curative_gate(bid: PowerBid) -> tuple[bool, list[str]]:
        """(allowed, the units whose on/off binaries gate the bid; [] = no gate row)."""
        if str(bid.mrid) in blocked_by_unit:
            return False, []
        units = [g for g, factor in _participation_of_bid(rd_inputs, bid).items() if float(factor) > 0.0]
        if any(g not in binary_gu_ids and unit_status.get(g) is not None and unit_status[g].initially_on
               for g in units):
            return True, []
        binary_units = sorted(g for g in units if g in binary_gu_ids)
        return bool(binary_units), binary_units

    # a countertrade bid with empty weights: fixed to 0 below; it is already logged once per RA and direction by
    # the weights (C3.7)
    empty_ct_bid_ids = _ct_bids_without_weights(rd_inputs, bid_by_mrid, curative_bid_ids_all)
    for bid_mrid, co_mrid in curative_bid_contingency_pairs:
        bid = bid_by_mrid[bid_mrid]
        available_mw = float(bid.available_mw.get(timestamp, 0.0))
        allowed, gating_units = _curative_gate(bid)

        if not allowed:
            model.curative_rd_volume_upward[bid_mrid, co_mrid].fix(0.0)
            model.curative_rd_volume_downward[bid_mrid, co_mrid].fix(0.0)
            if bid_mrid in empty_ct_bid_ids:
                continue
            log.info(
                "_add_curative_rd_block: bid %s (RA %s) — its generating unit is not in "
                "operation at %s and is not started preventively — "
                "curative_rd_volume_upward/downward fixed to 0.0 for contingency %s",
                bid_mrid, bid.ra_mrid, timestamp, co_mrid,
            )
            continue
        if gating_units:
            gating_units_by_bid[bid_mrid] = gating_units
            gated_by_binary.append((bid_mrid, co_mrid))

        if bid.direction == BidDirection.UP:
            model.curative_rd_volume_upward[bid_mrid, co_mrid].setub(available_mw)
            model.curative_rd_volume_downward[bid_mrid, co_mrid].setub(0.0)
        else:
            model.curative_rd_volume_downward[bid_mrid, co_mrid].setub(available_mw)
            model.curative_rd_volume_upward[bid_mrid, co_mrid].setub(0.0)

    # a bid whose units are all on/off units: no curative volume while all of them are off after
    # the preventive actions (started preventively -> allowed; shut down -> 0)
    model.curative_rd_unit_gate_index = pyo.Set(initialize=sorted(gated_by_binary), dimen=2)
    model.curative_rd_gating_units = gating_units_by_bid   # plain dict {bid: [unit]}

    def curative_rd_unit_gate_rule(m, bid_mrid, co_mrid):
        available_mw = float(bid_by_mrid[bid_mrid].available_mw.get(timestamp, 0.0))
        return (
            m.curative_rd_volume_upward[bid_mrid, co_mrid] + m.curative_rd_volume_downward[bid_mrid, co_mrid]
            <= available_mw * sum(m.rd_unit_on[gu_mrid] for gu_mrid in gating_units_by_bid[bid_mrid])
        )

    model.curative_rd_unit_gate = pyo.Constraint(model.curative_rd_unit_gate_index, rule=curative_rd_unit_gate_rule)

    # curative lead time, the same rule and radp as _add_rd_block: an unsatisfied bid gets
    # both upper bounds 0 (bounded, not dropped, so the rows keep their shape)
    radp = _compute_radp(rd_inputs.raart, timestamp)
    blocked_bid_ids: set[str] = set()
    for bid_mrid, co_mrid in curative_bid_contingency_pairs:
        bid = bid_by_mrid[bid_mrid]
        if bid.lead_time_h <= 0:
            continue  # no lead time restriction for this bid
        if not _timestamp_is_usable(radp, bid.lead_time_h, timestamp):
            model.curative_rd_volume_upward[bid_mrid, co_mrid].setub(0.0)
            model.curative_rd_volume_downward[bid_mrid, co_mrid].setub(0.0)
            blocked_bid_ids.add(bid_mrid)
    for bid_mrid in sorted(blocked_bid_ids):
        log.info(
            "_add_curative_rd_block: bid %s lead_time_h=%.2f not satisfied by "
            "radp=%s at timestamp=%s — curative_rd_volume_upward/downward "
            "bounds fixed to 0.0",
            bid_mrid, bid_by_mrid[bid_mrid].lead_time_h, radp, timestamp,
        )

    curative_gu_contingency_pairs = sorted({
        (gu_mrid, co_mrid)
        for bid_mrid, co_mrid in curative_bid_contingency_pairs
        for gu_mrid in _participation_of_bid(rd_inputs, bid_by_mrid[bid_mrid])
    })
    model.curative_gu_contingency_pairs = pyo.Set(initialize=curative_gu_contingency_pairs, dimen=2)

    # the pairs and the unit pairs of each contingency, in the order of the sorted lists (the rules below
    # then touch only their own contingency instead of scanning every pair for each row)
    pairs_of_contingency: dict[str, list[tuple[str, str]]] = {}
    for pair in curative_bid_contingency_pairs:
        pairs_of_contingency.setdefault(pair[1], []).append(pair)
    unit_pairs_of_contingency: dict[str, list[tuple[str, str]]] = {}
    for pair in curative_gu_contingency_pairs:
        unit_pairs_of_contingency.setdefault(pair[1], []).append(pair)

    def curative_rd_net_injection_rule(m, gu_mrid, co_mrid):
        terms = [
            float(participation_factor)
            * (m.curative_rd_volume_upward[bid_mrid, co] - m.curative_rd_volume_downward[bid_mrid, co])
            for bid_mrid, co in pairs_of_contingency.get(co_mrid, ())
            for participation_factor in [_participation_of_bid(rd_inputs, bid_by_mrid[bid_mrid]).get(gu_mrid)]
            if participation_factor is not None
        ]
        return sum(terms) if terms else 0.0

    model.curative_rd_net_injection_g = pyo.Expression(
        model.curative_gu_contingency_pairs, rule=curative_rd_net_injection_rule,
    )

    # curative balance per contingency (CORNET-574): the curative RD injections of one
    # contingency sum to 0, hard like the N-0 balance; a fixed (gated) bid contributes 0
    curative_contingency_ids = sorted({co_mrid for _gu, co_mrid in curative_gu_contingency_pairs})
    model.curative_rd_contingency_ids = pyo.Set(initialize=curative_contingency_ids)

    def curative_rd_balancing_rule(m, co_mrid):
        return sum(
            m.curative_rd_net_injection_g[gu_mrid, co]
            for gu_mrid, co in unit_pairs_of_contingency.get(co_mrid, ())
        ) == 0.0

    model.curative_rd_balancing = pyo.Constraint(
        model.curative_rd_contingency_ids, rule=curative_rd_balancing_rule,
    )

    # max-over-contingencies variables (3.42/3.43, CORNET-4614), EXACT: the objective pays the
    # worst-case reservation (not the sum over contingencies), and the per-unit Pmin/Pmax rows
    # below use these maxima in the net form, where a max bounded only from below could be
    # inflated to relax a bound. Each bid is one-directional: the other direction's max is 0.
    # Bid with one contingency: M = x. More than one: M >= x[c] for every c and, with binaries
    # z[c] (exactly one chosen), M <= x[c] + U (1 - z[c]), U = the bid's available MW.
    curative_bid_ids = sorted({bid_mrid for bid_mrid, _co in curative_bid_contingency_pairs})
    contingencies_by_bid: dict[str, list[str]] = {}
    for bid_mrid, co_mrid in curative_bid_contingency_pairs:
        contingencies_by_bid.setdefault(bid_mrid, []).append(co_mrid)
    model.bid_ids_curative = pyo.Set(initialize=curative_bid_ids)
    model.ct_curative_bid_ids = frozenset(b for b in curative_bid_ids if str(bid_by_mrid[b].ra_mrid) in ct_ra_mrids)
    model.max_curative_rd_volume_upward = pyo.Var(model.bid_ids_curative, domain=pyo.NonNegativeReals)
    model.max_curative_rd_volume_downward = pyo.Var(model.bid_ids_curative, domain=pyo.NonNegativeReals)

    def _active(m, bid_mrid):
        """(max variable, volume variable) of the bid's own direction."""
        if bid_by_mrid[bid_mrid].direction == BidDirection.UP:
            return m.max_curative_rd_volume_upward, m.curative_rd_volume_upward
        return m.max_curative_rd_volume_downward, m.curative_rd_volume_downward

    for bid_mrid in curative_bid_ids:
        inactive = (model.max_curative_rd_volume_downward if bid_by_mrid[bid_mrid].direction == BidDirection.UP
                    else model.max_curative_rd_volume_upward)
        inactive[bid_mrid].fix(0.0)

    model.max_curative_lower = pyo.Constraint(
        model.curative_bid_contingency_pairs,
        rule=lambda m, b, c: _active(m, b)[0][b] >= _active(m, b)[1][b, c],
    )
    single_contingency_bids = [b for b in curative_bid_ids if len(contingencies_by_bid[b]) == 1]
    model.curative_single_contingency_bid_ids = pyo.Set(initialize=single_contingency_bids)
    model.max_curative_single = pyo.Constraint(
        model.curative_single_contingency_bid_ids,
        rule=lambda m, b: _active(m, b)[0][b] <= _active(m, b)[1][b, contingencies_by_bid[b][0]],
    )
    choice_pairs = [(b, c) for b in curative_bid_ids if len(contingencies_by_bid[b]) > 1 for c in contingencies_by_bid[b]]
    model.curative_max_choice_pairs = pyo.Set(initialize=choice_pairs, dimen=2)
    model.curative_max_choice = pyo.Var(model.curative_max_choice_pairs, domain=pyo.Binary)
    multi_contingency_bids = sorted({b for b, _c in choice_pairs})
    model.curative_multi_contingency_bid_ids = pyo.Set(initialize=multi_contingency_bids)
    model.curative_max_choose_one = pyo.Constraint(
        model.curative_multi_contingency_bid_ids,
        rule=lambda m, b: sum(m.curative_max_choice[b, c] for c in contingencies_by_bid[b]) == 1,
    )
    model.curative_max_upper = pyo.Constraint(
        model.curative_max_choice_pairs,
        rule=lambda m, b, c: _active(m, b)[0][b] <= _active(m, b)[1][b, c]
        + max(float(bid_by_mrid[b].available_mw.get(timestamp, 0.0)), 0.0) * (1 - m.curative_max_choice[b, c]),
    )
    log.info(
        "_add_curative_rd_block: exact curative maxima - %d binary(ies) for %d bid(s) with more than "
        "one contingency, %d single-contingency bid(s) with M = x",
        len(choice_pairs), len(multi_contingency_bids), len(single_contingency_bids),
    )

    # CORNET-4614 (R7): one Pmin/Pmax pair per unit with curative bids, net form:
    #   Pmin <= P_after_preventive + sum_b pf*M_up_b - sum_b pf*M_down_b <= Pmax
    # with the exact per-bid maxima; a unit with an on/off binary uses Pmin*u and Pmax*u (F2
    # rule 7: the Pmin bound only while on), a FIXED_OFF unit no Pmin bound
    curative_gu_ids = sorted({gu for gu, _co in curative_gu_contingency_pairs})
    model.curative_unit_ids = pyo.Set(initialize=curative_gu_ids)

    # the bids with a factor on each unit, in the order of curative_bid_ids
    bids_of_unit: dict[str, list[tuple[str, float]]] = {}
    for b in curative_bid_ids:
        for gu_of_bid, factor in _participation_of_bid(rd_inputs, bid_by_mrid[b]).items():
            if factor is not None:
                bids_of_unit.setdefault(gu_of_bid, []).append((b, factor))

    def curative_maxima_net(m, gu_mrid):
        return sum(
            float(factor) * (m.max_curative_rd_volume_upward[b] - m.max_curative_rd_volume_downward[b])
            for b, factor in bids_of_unit.get(gu_mrid, ())
        )

    def output_after_preventive(m, gu_mrid):
        if gu_mrid in binary_gu_ids:
            return m.rd_unit_output_after_preventive[gu_mrid]
        gu_data = rd_inputs.generator_data.get(gu_mrid, {})
        target_p = float(gu_data.get("target_p", gu_data.get("initial_p", 0.0)))
        preventive = m.rd_net_injection_g[gu_mrid] if hasattr(m, "rd_net_injection_g") and gu_mrid in m.gu_ids else 0.0
        return target_p + preventive

    def curative_pmax_pmin_rule(m, gu_mrid):
        if gu_mrid in binary_gu_ids:
            return pyo.Constraint.Skip  # curative_on_off_lower / curative_on_off_upper
        gu_data = rd_inputs.generator_data.get(gu_mrid, {})
        status = unit_status.get(gu_mrid)
        min_p = status.pmin_mw if status is not None else float(gu_data.get("min_p", 0.0))
        if status is not None and status.unit_class == UNIT_FIXED_OFF:
            min_p = None
        return (min_p, output_after_preventive(m, gu_mrid) + curative_maxima_net(m, gu_mrid),
                float(gu_data.get("max_p", 0.0)))

    model.curative_pmax_pmin_constraint = pyo.Constraint(model.curative_unit_ids, rule=curative_pmax_pmin_rule)
    model.curative_on_off_unit_ids = pyo.Set(initialize=[g for g in curative_gu_ids if g in binary_gu_ids])
    model.curative_on_off_lower = pyo.Constraint(
        model.curative_on_off_unit_ids,
        rule=lambda m, g: output_after_preventive(m, g) + curative_maxima_net(m, g)
        >= unit_status[g].pmin_mw * m.rd_unit_on[g],
    )
    model.curative_on_off_upper = pyo.Constraint(
        model.curative_on_off_unit_ids,
        rule=lambda m, g: output_after_preventive(m, g) + curative_maxima_net(m, g)
        <= unit_status[g].pmax_mw * m.rd_unit_on[g],
    )

    # F2 rule 7: with u = 0 the net rows above only force sum up - sum down = 0, so up and down
    # maxima of different bids could cancel and move an off unit; one-sided rows make both 0
    # (with u = 1 they are looser than the net row)
    def curative_maxima_one_side(m, gu_mrid, max_variable):
        return sum(
            float(factor) * max_variable[b]
            for b, factor in bids_of_unit.get(gu_mrid, ())
        )

    model.curative_on_off_up_maxima = pyo.Constraint(
        model.curative_on_off_unit_ids,
        rule=lambda m, g: curative_maxima_one_side(m, g, m.max_curative_rd_volume_upward)
        <= unit_status[g].pmax_mw * m.rd_unit_on[g],
    )
    model.curative_on_off_down_maxima = pyo.Constraint(
        model.curative_on_off_unit_ids,
        rule=lambda m, g: curative_maxima_one_side(m, g, m.max_curative_rd_volume_downward)
        <= unit_status[g].pmax_mw * m.rd_unit_on[g],
    )

    return len(curative_bid_contingency_pairs)


def _add_nx_block(
    model,
    pyo,
    nx_inputs: NxMilpInputs,
    pst_inputs: Optional[PstMilpInputs],
    rd_inputs: Optional[RdMilpInputs],
    timestamp: str,
    branch_limits: dict[str, float],
    branch_ae_types: dict[str, "AEType"],
    branch_sides: dict[str, int],
    hvdc_inputs: Optional[HvdcMilpInputs] = None,
    restricted_pairs: Optional[dict] = None,
) -> int:
    """N-X constraints; returns the number of PATL pairs. restricted_pairs (data_model.branch_monitoring_rules): the
    branches whose assessed elements are restricted to / away from contingencies by the NCP links; no pair is built
    where every element of the branch is restricted away. PATL candidates are ALL secured
    branches filtered by min_loading_threshold_pct_nx against lf_nx (not the N-0-filtered
    set: a branch light at N-0 can be heavy after a contingency). PST/RD terms enter only
    when that type's preventive block exists. Curative PST variables are built first (the
    shared _nx_pst_and_rd_effect needs them); the TATL and scanned constraints reuse it.
    """
    if (
        pst_inputs is not None
        and pst_inputs.available_curative_pst_list
        and nx_inputs.pst_cra_availability
    ):
        _add_curative_pst_block(model, pyo, nx_inputs, pst_inputs)

    if rd_inputs is not None and rd_inputs.bid_list:
        # curative RD is gated only on rd_inputs carrying bids, not on the preventive RD block
        # (rd_inputs exists for curative-only RD: workflow_gates.rd_inputs_required)
        _add_curative_rd_block(model, pyo, nx_inputs, rd_inputs, timestamp)

    if hvdc_inputs is not None and hvdc_inputs.curative_units:
        _add_curative_hvdc_block(model, pyo, nx_inputs, hvdc_inputs, branch_limits)

    # Same rule as the N-0 side (_branch_is_above_threshold): a branch with a limit
    # <= 0 A cannot be evaluated and is excluded (one warning per branch).
    nx_unusable_limit_branches = sorted(b for b, lim in branch_limits.items() if float(lim) <= 0.0)
    if nx_unusable_limit_branches:
        log_summary.warn_ids(
            log, "N-X: %d branch(es) with a limit of 0.0 A (<= 0) excluded from the N-X constraints", nx_unusable_limit_branches)
    nx_unusable_limit_set = set(nx_unusable_limit_branches)

    all_secured_branch_mrids = [
        branch_mrid for branch_mrid in branch_limits
        if branch_ae_types.get(branch_mrid) == AEType.SECURED
        and branch_mrid not in nx_unusable_limit_set
    ]

    # A pair is only built from usable data: a missing or NaN N-X current, or a NaN
    # sensitivity of the branch under the contingency, skips the pair (never 0.0).
    def _nan_sensitivity_branches(contingency_mrid: str) -> set:
        branches = set()
        for table in (nx_inputs.sensitivity_pst_nx, nx_inputs.sensitivity_rd_nx, nx_inputs.sensitivity_hvdc_nx):
            sensitivities = table.get(contingency_mrid) or {}
            if hasattr(sensitivities, "nan_branches"):
                branches |= sensitivities.nan_branches()
                continue
            for (branch, _ra), value in sensitivities.items():
                if value is not None and math.isnan(float(value)):
                    branches.add(branch)
        return branches

    def _pair_current_or_skip(branch_mrid, contingency_mrid, nan_sens_branches, skipped, side=None):
        """The side-selected N-X current of the pair, or None (and the skip is recorded)."""
        side = branch_sides.get(branch_mrid, 1) if side is None else side
        per_side_currents = nx_inputs.lf_nx[contingency_mrid].get(branch_mrid)
        current = select_by_side(per_side_currents, side, branch_mrid) if per_side_currents else None
        if current is None or math.isnan(float(current)):
            # why the branch has no current (LfNx records it; a plain dict does not): not monitored at all (no N-0 result,
            # warned once by compute_lf_nx) and dropped for this contingency (one summary by compute_lf_nx) are not
            # reported again; the contingency's own equipment has no current by definition (debug only)
            gaps = nx_inputs.lf_nx
            if branch_mrid in getattr(gaps, "not_monitored", ()) or branch_mrid in getattr(gaps, "dropped", {}).get(contingency_mrid, ()):
                return None
            if branch_mrid in getattr(gaps, "tripped", {}).get(contingency_mrid, ()):
                skipped.setdefault(contingency_mrid, {}).setdefault("tripped", []).append(branch_mrid)
                return None
            skipped.setdefault(contingency_mrid, {}).setdefault("current", []).append(branch_mrid)
            return None
        if branch_mrid in nan_sens_branches:
            skipped.setdefault(contingency_mrid, {}).setdefault("sensitivity", []).append(branch_mrid)
            return None
        return current

    def _warn_skipped_pairs(kind: str, skipped: dict) -> None:
        """One summary per cause and constraint type (pairs, contingencies, the first 5 pairs); every contingency at debug
        level. The contingency's own equipment (reason "tripped") is debug only. The same summary is not repeated."""
        for reason, label in (("current", "no usable N-X current (missing or NaN)"), ("sensitivity", "NaN sensitivity"),
                              ("tripped", "its own tripped equipment")):
            by_contingency = {c: sorted(skipped[c][reason]) for c in sorted(skipped) if skipped[c].get(reason)}
            if not by_contingency:
                continue
            pairs = [f"{c}/{b}" for c, branches in by_contingency.items() for b in branches]
            for contingency_mrid, branches in by_contingency.items():
                log.debug("N-X %s: contingency %s: %d pair(s) skipped - %s: %s", kind, contingency_mrid, len(branches), label, branches)
            if reason != "tripped" and log_summary.first_time(("nx-pairs", kind, reason, tuple(pairs))):
                log_summary.warn_ids(
                    log, f"N-X {kind}: %d pair(s) in {len(by_contingency)} contingency(ies) skipped - {label}", pairs)

    def _loaded_pairs(kind: str, branch_mrids: list[str]) -> list[tuple[str, str]]:
        """(branch, contingency) pairs with a usable current whose loading against the branch's
        limit (all limits > 0 here) reaches min_loading_threshold_pct_nx."""
        pairs: list[tuple[str, str]] = []
        skipped: dict = {}
        for contingency_mrid in sorted(nx_inputs.lf_nx):
            nan_sens_branches = _nan_sensitivity_branches(contingency_mrid)
            for branch_mrid in branch_mrids:
                if not pair_is_monitored(restricted_pairs or {}, branch_mrid, contingency_mrid):
                    continue
                current = _pair_current_or_skip(branch_mrid, contingency_mrid, nan_sens_branches, skipped)
                if current is None:
                    continue
                loading_pct = (float(current) / float(branch_limits[branch_mrid])) * 100.0
                if loading_pct >= float(nx_inputs.min_loading_threshold_pct_nx):
                    pairs.append((branch_mrid, contingency_mrid))
        _warn_skipped_pairs(kind, skipped)
        return pairs

    nx_pairs = _loaded_pairs("PATL", all_secured_branch_mrids)

    model.nx_branch_contingency_pairs = pyo.Set(initialize=nx_pairs, dimen=2)
    model.nx_patl_overload_slack = pyo.Var(model.nx_branch_contingency_pairs, domain=pyo.NonNegativeReals)

    pst_degrees_per_tap = pst_inputs.pst_degrees_per_tap if pst_inputs is not None else {}
    pst_by_mrid_for_linkage: dict[str, PST] = (
        {str(pst.mrid): pst for pst in pst_inputs.available_curative_pst_list} if pst_inputs is not None else {}
    )

    def _curative_xne_linkage_allows(pst_mrid: str, branch_mrid: str) -> bool:
        """True when activate_cra_xne_linkage is False; else (pst_mrid, branch_mrid) must be
        in that PST's linked_xne_mrids."""
        if not nx_inputs.activate_cra_xne_linkage:
            return True
        pst = pst_by_mrid_for_linkage.get(pst_mrid)
        return pst is not None and branch_mrid in pst.linked_xne_mrids

    rd_by_mrid_for_linkage: dict[str, RDUnit] = (
        {str(rd.mrid): rd for rd in rd_inputs.rd_list} if rd_inputs is not None else {}
    )

    def _curative_rd_xne_linkage_allows(gu_mrid: str, branch_mrid: str) -> bool:
        """As _curative_xne_linkage_allows, per generating unit: allowed if ANY curative RD RA
        contributing to the unit via GLSK is linked to the branch."""
        if not nx_inputs.activate_cra_xne_linkage:
            return True
        if rd_inputs is None:
            return False
        for ra_mrid, participation in rd_inputs.glsk.items():
            if gu_mrid not in participation:
                continue
            rd_unit = rd_by_mrid_for_linkage.get(ra_mrid)
            if rd_unit is not None and branch_mrid in rd_unit.linked_xne_mrids:
                return True
        for bid_mrid, participation in (rd_inputs.bid_participation or {}).items():
            if gu_mrid not in participation:
                continue
            rd_unit = rd_by_mrid_for_linkage.get(str(bid_by_mrid_nx[bid_mrid].ra_mrid)) if bid_mrid in bid_by_mrid_nx else None
            if rd_unit is not None and branch_mrid in rd_unit.linked_xne_mrids:
                return True
        return False

    curative_pst_ids_by_contingency: dict[str, list[str]] = {}
    if hasattr(model, "curative_pst_contingency_pairs"):
        for _pst_mrid, _co_mrid in sorted(model.curative_pst_contingency_pairs):
            curative_pst_ids_by_contingency.setdefault(_co_mrid, []).append(_pst_mrid)

    # curative RD per bid (not per unit): two RAs on one unit can differ in reaction time, so
    # the per-branch availability (CORNET-613) is decided per bid
    curative_bid_ids_by_contingency: dict[str, list[str]] = {}
    if hasattr(model, "curative_bid_contingency_pairs"):
        for _bid_mrid, _co_mrid in sorted(model.curative_bid_contingency_pairs):
            curative_bid_ids_by_contingency.setdefault(_co_mrid, []).append(_bid_mrid)
    ra_mrid_by_bid: dict[str, str] = (
        {str(bid.mrid): str(bid.ra_mrid) for bid in rd_inputs.bid_list} if rd_inputs is not None else {}
    )
    bid_by_mrid_nx: dict[str, PowerBid] = (
        {str(bid.mrid): bid for bid in rd_inputs.bid_list} if rd_inputs is not None else {}
    )
    # availability of each curative RD RA on every branch this block evaluates (secured and
    # scanned), from the same helper that writes rd_cra_availability.csv (R9 rule 10)
    model.rd_cra_availability = (
        compute_rd_cra_availability(
            rd_inputs.rd_list, nx_inputs.tatl_data or {}, branch_mrids=list(branch_limits),
            fallback_duration_seconds=nx_inputs.tatl_fallback_duration_seconds,
        ) if curative_bid_ids_by_contingency else {}
    )

    def _curative_rd_available_on_branch(ra_mrid: str, branch_mrid: str) -> bool:
        entry = model.rd_cra_availability.get((ra_mrid, branch_mrid))
        return entry is not None and entry["available"]

    def _curative_rd_coefficients(branch_mrid: str, contingency_mrid: str) -> list[tuple[float, str]]:
        """[(A per MW of the bid's up - down volume, bid)] of the curative RD bids acting on
        the branch under the contingency: sum over the bid's GLSK units of participation x
        sensitivity, only for bids available on the branch and units the XNE linkage allows."""
        sensitivity_for_contingency = nx_inputs.sensitivity_rd_nx.get(contingency_mrid, {})
        terms = []
        for bid_mrid in curative_bid_ids_by_contingency.get(contingency_mrid, []):
            ra_mrid = ra_mrid_by_bid[bid_mrid]
            if not _curative_rd_available_on_branch(ra_mrid, branch_mrid):
                continue
            participation = _participation_of_bid(rd_inputs, bid_by_mrid_nx[bid_mrid])
            rd_unit = rd_by_mrid_for_linkage.get(ra_mrid)
            if rd_unit is not None and rd_unit.countertrade and (
                not nx_inputs.activate_cra_xne_linkage or branch_mrid in rd_unit.linked_xne_mrids
            ):
                # a countertrade RA that acts on the branch with all its units: the sensitivity of its RA and direction
                coefficient = ct_sensitivity.coefficient(
                    sensitivity_for_contingency, branch_mrid,
                    ct_sensitivity.ct_variable_id(ra_mrid, bid_by_mrid_nx[bid_mrid].direction), participation,
                )
            else:
                coefficient = sum(
                    float(weight) * float(sensitivity_for_contingency.get((branch_mrid, gu_mrid), 0.0))
                    for gu_mrid, weight in participation.items()
                    if _curative_rd_xne_linkage_allows(gu_mrid, branch_mrid)
                )
            if coefficient != 0.0:
                terms.append((coefficient, bid_mrid))
        return terms

    def _nx_pst_and_rd_effect(m, branch_mrid, contingency_mrid, include_curative=True):
        """RA-effect term on a branch under a contingency, shared by the PATL, TATL and scanned
        constraints (it does not depend on the limit). Preventive PST/RD terms when their
        block exists. Curative PST: sensitivity x (curative up - down) for the same PST, only
        where pst_cra_availability marks (pst, branch, c) available (and, with
        activate_cra_xne_linkage, the branch is linked). Curative RD: per bid, only where the
        RA's reaction time fits the branch's TATL duration, independently of the preventive RD
        block. include_curative=False gives the preventive state only (the TATL constraint).
        """
        pst_term = 0.0
        if hasattr(m, "tap_change_upward"):
            sensitivity_for_contingency = nx_inputs.sensitivity_pst_nx.get(contingency_mrid, {})
            pst_term = sum(
                float(sensitivity_for_contingency.get((branch_mrid, pst_mrid), 0.0))
                * float(pst_degrees_per_tap[pst_mrid])
                * (m.tap_change_upward[pst_mrid] - m.tap_change_downward[pst_mrid])
                for pst_mrid in m.pst_ids
                if pst_mrid in pst_degrees_per_tap
                and (not nx_inputs.drop_zero_pst_terms or float(sensitivity_for_contingency.get((branch_mrid, pst_mrid), 0.0)) != 0.0)
            )

        # curative PST term: the curative PSTs with a variable pair for THIS contingency (works
        # without preventive PSTs)
        if include_curative and hasattr(m, "curative_tap_change_upward"):
            sensitivity_for_contingency = nx_inputs.sensitivity_pst_nx.get(contingency_mrid, {})
            pst_term += sum(
                float(sensitivity_for_contingency.get((branch_mrid, pst_mrid), 0.0))
                * float(pst_degrees_per_tap[pst_mrid])
                * (
                    m.curative_tap_change_upward[pst_mrid, contingency_mrid]
                    - m.curative_tap_change_downward[pst_mrid, contingency_mrid]
                )
                for pst_mrid in curative_pst_ids_by_contingency.get(contingency_mrid, [])
                if pst_mrid in pst_degrees_per_tap
                and (pst_mrid, branch_mrid, contingency_mrid) in m.curative_tap_magnitude_triples
                and _curative_xne_linkage_allows(pst_mrid, branch_mrid)
                and (not nx_inputs.drop_zero_pst_terms or float(sensitivity_for_contingency.get((branch_mrid, pst_mrid), 0.0)) != 0.0)
            )

        rd_term = 0.0
        if hasattr(m, "rd_net_injection_g"):
            sensitivity_for_contingency = nx_inputs.sensitivity_rd_nx.get(contingency_mrid, {})
            # with countertrade bids: the RD part of the unit totals through the units, the CT volumes through the
            # sensitivity of their RA and direction
            flow_injection = m.rd_only_net_injection_g if m.ct_bid_ids else m.rd_net_injection_g
            rd_term = sum(
                float(sensitivity_for_contingency.get((branch_mrid, gu_mrid), 0.0))
                * flow_injection[gu_mrid]
                for gu_mrid in (m.rd_effect_unit_ids if m.ct_bid_ids else m.gu_ids)
                if not nx_inputs.drop_zero_rd_terms or float(sensitivity_for_contingency.get((branch_mrid, gu_mrid), 0.0)) != 0.0
            )
            for bid_mrid in sorted(m.ct_bid_ids):
                variable_id, weights = m.ct_bid_variable[bid_mrid]
                coefficient = ct_sensitivity.coefficient(sensitivity_for_contingency, branch_mrid, variable_id, weights)
                if coefficient != 0.0:
                    rd_term = rd_term + coefficient * (m.rd_volume_upward[bid_mrid] - m.rd_volume_downward[bid_mrid])

        if include_curative and hasattr(m, "curative_rd_volume_upward"):
            rd_term += sum(
                coefficient * (
                    m.curative_rd_volume_upward[bid_mrid, contingency_mrid]
                    - m.curative_rd_volume_downward[bid_mrid, contingency_mrid]
                )
                for coefficient, bid_mrid in _curative_rd_coefficients(branch_mrid, contingency_mrid)
            )

        return pst_term, rd_term

    def _curative_hvdc_pairs_on_branch(m, branch_mrid, contingency_mrid):
        """The curative HVDC units acting on a branch under a contingency: those of the groups with variables
        in it whose BOTH units are available on the branch (reaction time against its TATL duration, H4.7)."""
        availability = getattr(m, "hvdc_cra_availability", {})
        return [
            u for u in getattr(m, "hvdc_cur_units_by_contingency", {}).get(contingency_mrid, [])
            if all(
                availability[(partner, branch_mrid)]["available"]
                for partner in m.hvdc_cur_group_members[m.hvdc_cur_info[u]["group_mrid"]]
            )
        ]

    def _nx_hvdc_effect(m, branch_mrid, contingency_mrid, include_curative=True):
        """HVDC term of a branch under a contingency (0.0 without an HVDC block): the sensitivity of the unit's EI
        under the contingency times the preventive change of this solve; with include_curative (not in the TATL
        row, R11) plus the curative changes of the contingency of the units available on the branch."""
        sensitivity_for_contingency = nx_inputs.sensitivity_hvdc_nx.get(contingency_mrid, {})
        term = 0.0
        if hasattr(m, "hvdc_change"):
            term = sum(
                float(sensitivity_for_contingency.get((branch_mrid, m.hvdc_info[u]["ei_mrid"]), 0.0)) * m.hvdc_change[u]
                for u in m.hvdc_unit_ids
            )
        if include_curative and hasattr(m, "hvdc_cur_up"):
            term = term + sum(
                float(sensitivity_for_contingency.get((branch_mrid, m.hvdc_cur_info[u]["ei_mrid"]), 0.0))
                * (m.hvdc_cur_up[u, contingency_mrid] - m.hvdc_cur_down[u, contingency_mrid])
                for u in _curative_hvdc_pairs_on_branch(m, branch_mrid, contingency_mrid)
            )
        return term

    def _nx_curative_effect_ampere(branch_mrid: str, contingency_mrid: str) -> float:
        """The linearised effect of the CURATIVE actions of the SOLVED model on the
        N-X current of (branch, contingency), in A: exactly the curative terms of
        _nx_pst_and_rd_effect, evaluated at the solution. 0.0 when the model has no
        curative action for the pair. Used by the convergence check (the AC N-X
        current plus this effect is the post-curative current the PATL/TATL
        constraints bound). Only valid after the model was solved."""
        effect = 0.0
        if hasattr(model, "hvdc_cur_up"):
            sensitivity_for_contingency = nx_inputs.sensitivity_hvdc_nx.get(contingency_mrid, {})
            effect += sum(
                float(sensitivity_for_contingency.get((branch_mrid, model.hvdc_cur_info[u]["ei_mrid"]), 0.0))
                * (pyo.value(model.hvdc_cur_up[u, contingency_mrid]) - pyo.value(model.hvdc_cur_down[u, contingency_mrid]))
                for u in _curative_hvdc_pairs_on_branch(model, branch_mrid, contingency_mrid)
            )
        if hasattr(model, "curative_tap_change_upward"):
            sensitivity_for_contingency = nx_inputs.sensitivity_pst_nx.get(contingency_mrid, {})
            effect += sum(
                float(sensitivity_for_contingency.get((branch_mrid, pst_mrid), 0.0))
                * float(pst_degrees_per_tap[pst_mrid])
                * (
                    pyo.value(model.curative_tap_change_upward[pst_mrid, contingency_mrid])
                    - pyo.value(model.curative_tap_change_downward[pst_mrid, contingency_mrid])
                )
                for pst_mrid in curative_pst_ids_by_contingency.get(contingency_mrid, [])
                if pst_mrid in pst_degrees_per_tap
                and (pst_mrid, branch_mrid, contingency_mrid) in model.curative_tap_magnitude_triples
                and _curative_xne_linkage_allows(pst_mrid, branch_mrid)
            )
        if hasattr(model, "curative_rd_volume_upward"):
            effect += sum(
                coefficient * (
                    pyo.value(model.curative_rd_volume_upward[bid_mrid, contingency_mrid])
                    - pyo.value(model.curative_rd_volume_downward[bid_mrid, contingency_mrid])
                )
                for coefficient, bid_mrid in _curative_rd_coefficients(branch_mrid, contingency_mrid)
            )
        return float(effect)

    model.nx_curative_effect_ampere = _nx_curative_effect_ampere

    def _nx_hvdc_preventive_effect_ampere(branch_mrid: str, contingency_mrid: str) -> float:
        """The linearised effect of the PREVENTIVE HVDC change of the SOLVED model on the N-X current of
        (branch, contingency), in A (the preventive term of the N-X rows; 0.0 without an HVDC block)."""
        if not hasattr(model, "hvdc_change"):
            return 0.0
        sensitivity_for_contingency = nx_inputs.sensitivity_hvdc_nx.get(contingency_mrid, {})
        return float(sum(
            float(sensitivity_for_contingency.get((branch_mrid, model.hvdc_info[u]["ei_mrid"]), 0.0))
            * pyo.value(model.hvdc_change[u])
            for u in model.hvdc_unit_ids
        ))

    model.nx_hvdc_preventive_effect_ampere = _nx_hvdc_preventive_effect_ampere
    # what the convergence check monitors, with the side and limit the constraints use
    model.overload_monitoring["nx_patl"] = {
        (branch_mrid, contingency_mrid): (branch_sides.get(branch_mrid, 1), float(branch_limits[branch_mrid]))
        for branch_mrid, contingency_mrid in nx_pairs
    }

    def nx_patl_loading_rule(m, branch_mrid, contingency_mrid):
        side = branch_sides.get(branch_mrid, 1)
        nx_current_ampere = select_by_side(
            nx_inputs.lf_nx[contingency_mrid].get(branch_mrid, {}), side, branch_mrid,
        ) or 0.0
        patl_ampere = float(branch_limits[branch_mrid])
        pst_term, rd_term = _nx_pst_and_rd_effect(m, branch_mrid, contingency_mrid)
        body = float(nx_current_ampere) + pst_term + rd_term
        if hasattr(m, "hvdc_change") or hasattr(m, "hvdc_cur_up"):
            body = body + _nx_hvdc_effect(m, branch_mrid, contingency_mrid)

        return body <= patl_ampere + m.nx_patl_overload_slack[branch_mrid, contingency_mrid]

    model.nx_patl_loading = pyo.Constraint(model.nx_branch_contingency_pairs, rule=nx_patl_loading_rule)

    # N-X TATL constraint (CORNET-1790 acceptance criterion): the state after the preventive
    # actions only, against the TATL; the PATL constraint holds the state after preventive and
    # curative actions. Only for a branch with a TATL strictly above its PATL (spec: I_TATL >
    # I_PATL; a missing TATL, a synthetic one or one raised to the PATL gives no constraint).
    # No loading pre-filter: every usable pair of such a branch.
    if nx_inputs.tatl_data:
        # SECURED branches only, also when tatl_data was built directly (scanned branches have
        # no TATL)
        tatl_pairs = []
        skipped_tatl_pairs: dict = {}
        tatl_above_patl_branches = sorted(
            branch_mrid for branch_mrid, entry in nx_inputs.tatl_data.items()
            if branch_mrid in branch_limits
            and tatl_is_above_patl(entry["tatl_limit_ampere"], branch_limits[branch_mrid])
        )
        if len(tatl_above_patl_branches) < len(nx_inputs.tatl_data):
            log.info(
                "N-X TATL: %d of %d branch(es) with TATL data have a TATL above the PATL and get "
                "a TATL constraint (the others: TATL missing, synthetic or at most the PATL)",
                len(tatl_above_patl_branches), len(nx_inputs.tatl_data),
            )
        for contingency_mrid in sorted(nx_inputs.lf_nx):
            nan_sens_branches = _nan_sensitivity_branches(contingency_mrid)
            for branch_mrid in tatl_above_patl_branches:
                if branch_ae_types.get(branch_mrid) != AEType.SECURED:
                    continue
                if not pair_is_monitored(restricted_pairs or {}, branch_mrid, contingency_mrid):
                    continue
                if _pair_current_or_skip(
                    branch_mrid, contingency_mrid, nan_sens_branches, skipped_tatl_pairs,
                    side=int(nx_inputs.tatl_data[branch_mrid]["side"]),
                ) is None:
                    continue
                tatl_pairs.append((branch_mrid, contingency_mrid))
        _warn_skipped_pairs("TATL", skipped_tatl_pairs)
        model.nx_tatl_branch_contingency_pairs = pyo.Set(initialize=tatl_pairs, dimen=2)
        model.nx_tatl_overload_slack = pyo.Var(
            model.nx_tatl_branch_contingency_pairs, domain=pyo.NonNegativeReals,
        )

        def nx_tatl_loading_rule(m, branch_mrid, contingency_mrid):
            tatl_entry = nx_inputs.tatl_data[branch_mrid]
            side = int(tatl_entry["side"])
            nx_current_ampere = select_by_side(
                nx_inputs.lf_nx[contingency_mrid].get(branch_mrid, {}), side, branch_mrid,
            ) or 0.0
            tatl_limit_ampere = float(tatl_entry["tatl_limit_ampere"])
            pst_term, rd_term = _nx_pst_and_rd_effect(m, branch_mrid, contingency_mrid, include_curative=False)
            body = float(nx_current_ampere) + pst_term + rd_term
            if hasattr(m, "hvdc_change"):
                body = body + _nx_hvdc_effect(m, branch_mrid, contingency_mrid, include_curative=False)

            return body <= tatl_limit_ampere + m.nx_tatl_overload_slack[branch_mrid, contingency_mrid]

        model.nx_tatl_loading = pyo.Constraint(
            model.nx_tatl_branch_contingency_pairs, rule=nx_tatl_loading_rule,
        )
        model.overload_monitoring["nx_tatl"] = {
            (branch_mrid, contingency_mrid): (
                int(nx_inputs.tatl_data[branch_mrid]["side"]),
                float(nx_inputs.tatl_data[branch_mrid]["tatl_limit_ampere"]),
            )
            for branch_mrid, contingency_mrid in tatl_pairs
        }

    # N-X scanned-element constraint: as PATL (same _nx_pst_and_rd_effect), for SCANNED
    # branches against the effective limit in branch_limits. Scanned elements have no TATL.
    all_scanned_branch_mrids = [
        branch_mrid for branch_mrid in branch_limits
        if branch_ae_types.get(branch_mrid) == AEType.SCANNED
        and branch_mrid not in nx_unusable_limit_set
    ]

    nx_scanned_pairs = _loaded_pairs("scanned", all_scanned_branch_mrids)

    model.nx_scanned_branch_contingency_pairs = pyo.Set(initialize=nx_scanned_pairs, dimen=2)
    model.nx_scanned_overload_slack = pyo.Var(
        model.nx_scanned_branch_contingency_pairs, domain=pyo.NonNegativeReals,
    )

    def nx_scanned_loading_rule(m, branch_mrid, contingency_mrid):
        side = branch_sides.get(branch_mrid, 1)
        nx_current_ampere = select_by_side(
            nx_inputs.lf_nx[contingency_mrid].get(branch_mrid, {}), side, branch_mrid,
        ) or 0.0
        effective_limit_ampere = float(branch_limits[branch_mrid])
        pst_term, rd_term = _nx_pst_and_rd_effect(m, branch_mrid, contingency_mrid)
        body = float(nx_current_ampere) + pst_term + rd_term
        if hasattr(m, "hvdc_change") or hasattr(m, "hvdc_cur_up"):
            body = body + _nx_hvdc_effect(m, branch_mrid, contingency_mrid)

        return body <= effective_limit_ampere + m.nx_scanned_overload_slack[branch_mrid, contingency_mrid]

    model.nx_scanned_loading = pyo.Constraint(
        model.nx_scanned_branch_contingency_pairs, rule=nx_scanned_loading_rule,
    )

    return len(nx_pairs)


def _add_ra_activation_and_count_constraints(
    model,
    pyo,
    rd_inputs: RdMilpInputs,
    bid_ids: list[str],
    bid_by_mrid: dict[str, PowerBid],
    timestamp: str,
) -> None:
    """Binary ra_activated variables and per-TSO RA count limits; called only when the RD
    block was built. big-M per bid = its own available_mw[timestamp] (its volume upper
    bound): the tightest valid value, avoiding a large shared big-M.
    """
    ra_mrid_to_tso: dict[str, str] = {str(rd.mrid): rd.tso for rd in rd_inputs.rd_list}
    ra_mrids_in_model = sorted({str(bid_by_mrid[b].ra_mrid) for b in bid_ids})

    model.ra_mrids = pyo.Set(initialize=ra_mrids_in_model)
    model.ra_activated = pyo.Var(model.ra_mrids, domain=pyo.Binary)

    def big_m_activation_rule(m, bid_mrid):
        bid = bid_by_mrid[bid_mrid]
        big_m = float(bid.available_mw.get(timestamp, 0.0))
        # linked to the CUMULATIVE volume: an RA used in an
        # earlier iteration stays "activated" as long as its cumulative net volume
        # is non-zero. The binary is NOT fixed to 1.
        return (
            m.rd_cumulative_volume_term[bid_mrid]
            <= big_m * m.ra_activated[str(bid.ra_mrid)]
        )

    model.ra_activation_big_m = pyo.Constraint(model.bid_ids, rule=big_m_activation_rule)

    def _activated_sum_for_tso(m, tso: str):
        return sum(
            m.ra_activated[ra_mrid] for ra_mrid in ra_mrids_in_model
            if ra_mrid_to_tso.get(ra_mrid) == tso
        )

    # per-TSO per-timestamp RA count: only TSOs with a limit (a TSO with a limit and no RA
    # gets a trivially satisfied constraint)
    max_ra_count_per_timestamp_tso = rd_inputs.max_ra_count_per_timestamp_tso or {}
    if max_ra_count_per_timestamp_tso:
        model.tsos_with_timestamp_ra_limit = pyo.Set(initialize=sorted(max_ra_count_per_timestamp_tso.keys()))

        if not rd_inputs.ra_limit_constraint_hard:
            model.ra_count_slack_per_timestamp = pyo.Var(
                model.tsos_with_timestamp_ra_limit, domain=pyo.NonNegativeReals,
            )

        def ra_count_limit_per_timestamp_rule(m, tso):
            slack = m.ra_count_slack_per_timestamp[tso] if hasattr(m, "ra_count_slack_per_timestamp") else 0.0
            return _activated_sum_for_tso(m, tso) - slack <= max_ra_count_per_timestamp_tso[tso]

        model.ra_count_limit_per_timestamp = pyo.Constraint(
            model.tsos_with_timestamp_ra_limit, rule=ra_count_limit_per_timestamp_rule,
        )


def _add_ra_group_volume_constraints(
    model,
    pyo,
    rd_inputs: RdMilpInputs,
    bid_ids: list[str],
    bid_by_mrid: dict[str, PowerBid],
) -> None:
    """Restrictive RA group volume limits, hard (no slack); called only when the RD block was
    built. Per group: sum of its bids' rd_volume_upward <= max_regulating_up (downward
    likewise), for each limit that is not None; a group without limits is logged.
    """
    ra_groups = rd_inputs.ra_groups or {}
    if not ra_groups:
        return

    group_bid_ids: dict[str, list[str]] = {}
    for group_mrid, group_data in ra_groups.items():
        member_ra_mrids = set(group_data.get("ra_mrids", []))
        group_bid_ids[group_mrid] = [
            bid_mrid for bid_mrid in bid_ids
            if str(bid_by_mrid[bid_mrid].ra_mrid) in member_ra_mrids
        ]

    groups_with_upward_limit = {
        g: data for g, data in ra_groups.items()
        if data.get("max_regulating_up") is not None and group_bid_ids.get(g)
    }
    groups_with_downward_limit = {
        g: data for g, data in ra_groups.items()
        if data.get("max_regulating_down") is not None and group_bid_ids.get(g)
    }

    for group_mrid, data in ra_groups.items():
        if data.get("max_regulating_up") is None and data.get("max_regulating_down") is None:
            log.warning(
                "RA group %s (restrictive) has no resolved volume limit in "
                "either direction — no group volume constraint added for it",
                group_mrid,
            )
        elif not group_bid_ids.get(group_mrid):
            log.warning(
                "RA group %s (restrictive) has a resolved volume limit but "
                "no member RA has a bid in the model — no group volume "
                "constraint added for it",
                group_mrid,
            )

    if groups_with_upward_limit:
        model.ra_groups_with_upward_limit = pyo.Set(initialize=sorted(groups_with_upward_limit.keys()))

        def group_volume_upward_rule(m, group_mrid):
            if getattr(m, "rd_cumulative_used_in_any_bid", False):
                # cumulative group volume (earlier iterations + this solve)
                return (
                    sum(m.rd_cumulative_up_term[b] for b in group_bid_ids[group_mrid])
                    <= groups_with_upward_limit[group_mrid]["max_regulating_up"]
                )
            return (
                sum(m.rd_volume_upward[b] for b in group_bid_ids[group_mrid])
                <= groups_with_upward_limit[group_mrid]["max_regulating_up"]
            )

        model.ra_group_volume_upward_limit = pyo.Constraint(
            model.ra_groups_with_upward_limit, rule=group_volume_upward_rule,
        )

    if groups_with_downward_limit:
        model.ra_groups_with_downward_limit = pyo.Set(initialize=sorted(groups_with_downward_limit.keys()))

        def group_volume_downward_rule(m, group_mrid):
            if getattr(m, "rd_cumulative_used_in_any_bid", False):
                return (
                    sum(m.rd_cumulative_down_term[b] for b in group_bid_ids[group_mrid])
                    <= groups_with_downward_limit[group_mrid]["max_regulating_down"]
                )
            return (
                sum(m.rd_volume_downward[b] for b in group_bid_ids[group_mrid])
                <= groups_with_downward_limit[group_mrid]["max_regulating_down"]
            )

        model.ra_group_volume_downward_limit = pyo.Constraint(
            model.ra_groups_with_downward_limit, rule=group_volume_downward_rule,
        )


# ---------------------------------------------------------------------------
# Composable model builder
# ---------------------------------------------------------------------------

def build_n0_milp_model(
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_initial: dict[str, dict[int, float]],
    penalty_cost_overload_n0: float,
    pst_inputs: Optional[PstMilpInputs] = None,
    rd_inputs: Optional[RdMilpInputs] = None,
    nx_inputs: Optional[NxMilpInputs] = None,
    min_loading_threshold_pct: float = 0.0,
    scanned_element_constraint_hard: bool = True,
    scanned_element_penalty_eur_per_ka: float = 0.0,
    lf_original: Optional[dict[str, dict[int, float]]] = None,
    hvdc_inputs: Optional[HvdcMilpInputs] = None,
):
    """Build the MILP: PST-only, RD-only or PST + RD, selected by which of pst_inputs /
    rd_inputs is non-None (both None -> ValueError before any model is created; an RD block
    that adds nothing is logged, not raised). hvdc_inputs adds the HVDC block (preventive units) to any of
    them. nx_inputs adds the N-X constraints on top.
    lf_initial: {branch: {1: A, 2: A}}, side-selected per branch.
    scanned_element_constraint_hard (default True) / scanned_element_penalty_eur_per_ka: the
    N-0 scanned constraint, independent of the RA types.
    """
    if pst_inputs is None and rd_inputs is None and nx_inputs is None and hvdc_inputs is None:
        raise ValueError(
            "At least one of pst_inputs, rd_inputs, hvdc_inputs, or nx_inputs must be "
            "provided — there is nothing to optimise with all four absent. "
            "nx_inputs alone (even without dispatchable capacity) is valid: the "
            "slacks absorb what nothing can resolve."
        )

    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to build the MILP model") from exc

    # lf_original: the scanned-element effective limit is
    # "at least the pre-RA initial loading", so it stays anchored to the
    # ORIGINAL network's N-0 load flow while the flows below use lf_initial (the
    # current operating point). None = lf_initial is the original.
    branch_limits, branch_ae_types = _get_monitored_branch_limits_and_ae_types(
        ae_flat_list, timestamp, lf_original if lf_original is not None else lf_initial,
    )
    branch_sides = _get_monitored_branch_sides(ae_flat_list)
    # NCP switches (inBaseCase, links to contingencies): a branch whose elements are all "not in the base case" has no
    # N-0 row; a pair (branch, contingency) that every element of the branch is restricted away from has no N-X row
    base_case_off, restricted_pairs = branch_monitoring_rules(ae_flat_list)
    if base_case_off & set(branch_limits):
        log.info("N-0: %d branch(es) are not monitored in the base case (inBaseCase=false): %s",
                 len(base_case_off & set(branch_limits)), sorted(base_case_off & set(branch_limits)))
    excluded_branches: dict[str, list[str]] = {}
    branch_ids = [
        branch_mrid
        for branch_mrid in sorted(branch_limits)
        if branch_mrid not in base_case_off
        and _branch_is_above_threshold(
            branch_mrid, lf_initial, branch_limits, branch_sides, min_loading_threshold_pct, excluded_branches
        )
    ]
    _log_unusable_branches(excluded_branches)
    # same rule as the N-X pairs: a NaN N-0 sensitivity (PST or RD) excludes the branch
    nan_sensitivity_branches = {
        branch_mrid
        for table in (
            pst_inputs.sensitivity_pst_n0_pruned if pst_inputs is not None else {},
            rd_inputs.sensitivity_rd_n0_pruned if rd_inputs is not None else {},
            hvdc_inputs.sensitivity_hvdc_n0 if hvdc_inputs is not None else {},
        )
        for (branch_mrid, _ra), value in table.items()
        if value is not None and math.isnan(float(value))
    }
    nan_branches_in_model = sorted(nan_sensitivity_branches & set(branch_ids))
    if nan_branches_in_model:
        log_summary.warn_ids(log, "%d branch(es) with a NaN N-0 sensitivity excluded from the MILP", nan_branches_in_model)
    branch_ids = [b for b in branch_ids if b not in nan_sensitivity_branches]
    # secured branches -> xne_loading_n0 (PATL), scanned -> scanned_element_n0 (effective
    # limit); both read branch_limits, which already holds the type-correct limit
    secured_branch_ids = [b for b in branch_ids if branch_ae_types.get(b) == AEType.SECURED]
    scanned_branch_ids = [b for b in branch_ids if branch_ae_types.get(b) == AEType.SCANNED]

    model = pyo.ConcreteModel()
    # The elements whose overload the linearisation convergence check follows, with the
    # side and the limit (A) the model's own constraints use: N-0 = every branch of
    # model.branch_ids (PATL for secured, effective limit for scanned branches);
    # N-X PATL / TATL = the pairs of the N-X constraints (filled by _add_nx_block).
    model.overload_monitoring = {
        "n0": {b: (branch_sides.get(b, 1), float(branch_limits[b])) for b in branch_ids},
        "nx_patl": {},
        "nx_tatl": {},
    }
    # model.branch_ids is the combined secured + scanned set over which
    # pst_effect_on_branch / rd_effect_on_branch are indexed
    model.branch_ids = pyo.Set(initialize=branch_ids)
    model.secured_branch_ids = pyo.Set(initialize=secured_branch_ids)
    # overload_slack_n0: secured branches only (scanned have scanned_overload_slack)
    model.overload_slack_n0 = pyo.Var(model.secured_branch_ids, domain=pyo.NonNegativeReals)

    pst_ids: list[str] = []
    if pst_inputs is not None:
        pst_ids = _add_pst_block(model, pyo, pst_inputs, branch_ids)

    if rd_inputs is not None:
        validate_participation_factors(rd_inputs.glsk)
        if rd_inputs.bid_participation:
            validate_participation_factors(rd_inputs.bid_participation, owner="Weights of countertrade bid")
        rd_block_added = _add_rd_block(model, pyo, rd_inputs, timestamp, branch_ids)
        if rd_block_added:
            # RA count: only once RD variables exist
            bid_by_mrid: dict[str, PowerBid] = {str(bid.mrid): bid for bid in rd_inputs.bid_list}
            _add_ra_activation_and_count_constraints(
                model, pyo, rd_inputs, list(model.bid_ids), bid_by_mrid, timestamp,
            )
            # restrictive RA group volume limits
            _add_ra_group_volume_constraints(
                model, pyo, rd_inputs, list(model.bid_ids), bid_by_mrid,
            )

    # HVDC (preventive units): its own group balance, outside the RD balance below (H5.4)
    if hvdc_inputs is not None:
        _add_hvdc_block(model, pyo, hvdc_inputs, branch_ids)

    # hard RD balance, sum_g rd_net_injection_g[g] = 0, no slack: added when
    # rd_net_injection_g exists (not merely rd_inputs, whose block can add nothing)
    if hasattr(model, "rd_net_injection_g"):
        def rd_balancing_n0_rule(m):
            return sum(m.rd_net_injection_g[gu_mrid] for gu_mrid in m.gu_ids) == 0.0

        model.rd_balancing_n0 = pyo.Constraint(rule=rd_balancing_n0_rule)

    def xne_loading_n0_rule(m, branch_mrid):
        side = branch_sides.get(branch_mrid, 1)
        initial_current = select_by_side(lf_initial.get(branch_mrid, {}), side, branch_mrid) or 0.0
        patl_ampere = float(branch_limits[branch_mrid])
        pst_term = m.pst_effect_on_branch[branch_mrid] if hasattr(m, "pst_effect_on_branch") else 0.0
        rd_term = m.rd_effect_on_branch[branch_mrid] if hasattr(m, "rd_effect_on_branch") else 0.0
        body = float(initial_current) + pst_term + rd_term
        if hasattr(m, "hvdc_effect_on_branch"):
            body = body + m.hvdc_effect_on_branch[branch_mrid]
        return body <= patl_ampere + m.overload_slack_n0[branch_mrid]

    # secured_branch_ids only (model.branch_ids also holds the scanned branches)
    model.xne_loading_n0 = pyo.Constraint(model.secured_branch_ids, rule=xne_loading_n0_rule)

    # N-0 scanned element constraint on the same effect Expressions; only when a scanned
    # branch survived the min_loading_threshold_pct pre-filter
    if scanned_branch_ids:
        model.scanned_branch_ids = pyo.Set(initialize=scanned_branch_ids)

        if not scanned_element_constraint_hard:
            model.scanned_overload_slack = pyo.Var(model.scanned_branch_ids, domain=pyo.NonNegativeReals)

        def scanned_element_n0_rule(m, branch_mrid):
            side = branch_sides.get(branch_mrid, 1)
            initial_current = select_by_side(lf_initial.get(branch_mrid, {}), side, branch_mrid) or 0.0
            effective_limit_ampere = float(branch_limits[branch_mrid])
            pst_term = m.pst_effect_on_branch[branch_mrid] if hasattr(m, "pst_effect_on_branch") else 0.0
            rd_term = m.rd_effect_on_branch[branch_mrid] if hasattr(m, "rd_effect_on_branch") else 0.0
            slack = m.scanned_overload_slack[branch_mrid] if hasattr(m, "scanned_overload_slack") else 0.0
            body = float(initial_current) + pst_term + rd_term
            if hasattr(m, "hvdc_effect_on_branch"):
                body = body + m.hvdc_effect_on_branch[branch_mrid]
            expression = body <= effective_limit_ampere + slack
            if isinstance(expression, bool):
                # no remedial action touches a hard row (constant left side): Pyomo has no object for it
                if not expression:
                    raise ValueError(
                        f"scanned element {branch_mrid}: N-0 current {float(initial_current):.1f} A is above its limit "
                        f"{effective_limit_ampere:.1f} A and no remedial action acts on it (hard scanned-element constraint)"
                    )
                return pyo.Constraint.Feasible
            return expression

        model.scanned_element_n0 = pyo.Constraint(model.scanned_branch_ids, rule=scanned_element_n0_rule)

    # N-X constraints when nx_inputs is given; RA-type terms gated inside _add_nx_block
    nx_pair_count = 0
    if nx_inputs is not None:
        nx_pair_count = _add_nx_block(
            model, pyo, nx_inputs, pst_inputs, rd_inputs, timestamp,
            branch_limits, branch_ae_types, branch_sides, hvdc_inputs=hvdc_inputs, restricted_pairs=restricted_pairs,
        )
        _log_if_changed(
            "nx_pair_count", nx_pair_count,
            "N-X PATL loading constraint added for %d (branch, contingency) pair(s)",
            nx_pair_count,
        )

    objective_terms = []
    # Results summary: the slack and RA-count penalty terms are recorded by name
    # (the very expressions that are summed into the objective) so the summary
    # reports them without repeating the penalty coefficients.
    model.objective_penalty_term_expressions = {}
    # every term of the objective by name, for the log block after a solve (the summary above stays as it is)
    model.objective_term_expressions = {}

    def _append_named_term(name, expression):
        objective_terms.append(expression)
        model.objective_penalty_term_expressions[name] = expression
        model.objective_term_expressions[name] = expression

    def _append_logged_term(name, expression):
        objective_terms.append(expression)
        model.objective_term_expressions[name] = expression

    if hasattr(model, "tap_change_upward"):
        effective_penalties = _get_effective_penalty_costs(
            pst_inputs.available_preventive_pst_list,
            pst_inputs.penalty_reference_pst_pra,
            pst_inputs.penalty_reference_pst_pra_per_tso,
        )
        # model.pst_ids (a deduplicated Set): a PST with a preventive and a curative RA appears
        # twice in pst_ids and would be costed twice
        _append_logged_term(
            "pst_tap_change_penalty",
            sum(
                effective_penalties[pst_mrid]
                * (model.cum_tap_up_term[pst_mrid] + model.cum_tap_down_term[pst_mrid])
                for pst_mrid in model.pst_ids
            ),
        )
    # curative PST penalty: SUM over (PST, contingency) of the curative RA's effective penalty
    # x (up + down); a sum, not a max (unlike curative RD). The TATL duration is in the tap
    # cap. Penalty from the curative PST object, same helper and reference as preventive.
    # Absent without curative PST pairs.
    if hasattr(model, "curative_pst_contingency_pairs"):
        curative_effective_penalties = _get_effective_penalty_costs(
            pst_inputs.available_curative_pst_list,
            pst_inputs.penalty_reference_pst_pra,
            pst_inputs.penalty_reference_pst_pra_per_tso,
        )
        _append_logged_term(
            "curative_pst_tap_change_penalty",
            sum(
                curative_effective_penalties[pst_mrid]
                * (
                    model.curative_tap_change_upward[pst_mrid, contingency_mrid]
                    + model.curative_tap_change_downward[pst_mrid, contingency_mrid]
                )
                for pst_mrid, contingency_mrid in model.curative_pst_contingency_pairs
            ),
        )
    # RD penalty and price cost, gated on rd_volume_upward (absent for PST-only and for an
    # RD block that added nothing)
    if hasattr(model, "rd_volume_upward"):
        # RA-level penalty_factor and tso per bid (PowerBid.ra_mrid -> RDUnit.mrid); the
        # bid-level PowerBid.penalty_factor is a different field and not used
        ra_penalty_factor_and_tso: dict[str, tuple[float, str]] = {
            str(rd.mrid): (float(rd.penalty_factor), rd.tso) for rd in rd_inputs.rd_list
        }
        penalty_reference_rd_pra_per_tso = rd_inputs.penalty_reference_rd_pra_per_tso or {}

        bid_by_mrid_all_for_cost: dict[str, PowerBid] = {str(b.mrid): b for b in rd_inputs.bid_list}
        rd_cost_terms = []
        ct_bid_ids_in_model = getattr(model, "ct_bid_ids", frozenset())
        for bid_mrid in model.bid_ids:
            if bid_mrid in ct_bid_ids_in_model:
                continue    # countertrade bids have their own named terms below
            bid = bid_by_mrid_all_for_cost[bid_mrid]
            penalty_factor, bid_tso = ra_penalty_factor_and_tso.get(str(bid.ra_mrid), (None, None))
            if penalty_factor is None:
                log.warning(
                    "PowerBid %s references RA %s not found in rd_list — "
                    "penalty component of its RD cost term defaults to 0.0 "
                    "(price component is unaffected)",
                    bid_mrid, bid.ra_mrid,
                )
                penalty_factor, bid_tso = 0.0, ""

            effective_penalty_reference_rd_pra = float(
                penalty_reference_rd_pra_per_tso.get(bid_tso, rd_inputs.penalty_reference_rd_pra)
            )

            # cost = (penalty_factor * penalty_reference_rd_pra + price) * (up + down): each bid is
            # one-directional and the other direction is bounded to 0, so this equals the
            # separate up/down form with price_eur_mw as the bid direction's price
            rd_cost_terms.append(
                (penalty_factor * effective_penalty_reference_rd_pra + float(bid.price_eur_mw))
                * model.rd_cumulative_volume_term[bid_mrid]
            )

        if rd_cost_terms:
            _append_logged_term("rd_penalty_and_price_cost", sum(rd_cost_terms))

        # start-up (3.39, BUC 14#103), shut-down (F2, ahead of CORNET-71) and bid activation
        # (3.38, BUC 14#102) costs, preventive only; each term exists only when it has a binary
        if len(model.rd_binary_gu_ids):
            status = model.rd_unit_status
            startup_units = [g for g in model.rd_binary_gu_ids if status[g].unit_class == UNIT_STARTABLE]
            shutdown_units = [g for g in model.rd_binary_gu_ids if status[g].unit_class == UNIT_STOPPABLE]
            if startup_units:
                _append_named_term("startup_cost", sum(
                    float(status[g].startup_cost_eur) * model.rd_unit_on[g] for g in startup_units
                ))
            if shutdown_units:
                _append_named_term("shutdown_cost", sum(
                    float(status[g].shutdown_cost_eur) * (1 - model.rd_unit_on[g]) for g in shutdown_units
                ))
        rd_activation_bid_ids = [b for b in model.rd_activation_bid_ids if b not in ct_bid_ids_in_model]
        if len(rd_activation_bid_ids):
            _append_named_term("bid_activation_cost", sum(
                float(bid_by_mrid_all_for_cost[b].activation_cost_eur) * model.rd_bid_activated[b]
                for b in rd_activation_bid_ids
            ))

        # countertrade bids (C6.6): penalty factor x reference x volume, bid price x volume, activation cost;
        # each a named term, so the summary accounts for it
        if ct_bid_ids_in_model:
            ct_unit_by_ra = {str(rd.mrid): rd for rd in rd_inputs.rd_list if rd.countertrade}
            ct_penalty_terms, ct_energy_terms, ct_activation_terms = [], [], []
            for bid_mrid in model.bid_ids:
                if bid_mrid not in ct_bid_ids_in_model:
                    continue
                bid = bid_by_mrid_all_for_cost[bid_mrid]
                ct_ra = ct_unit_by_ra[str(bid.ra_mrid)]
                ct_penalty_terms.append(
                    float(ct_ra.penalty_factor) * float(rd_inputs.penalty_reference_ct_pra) * model.rd_cumulative_volume_term[bid_mrid]
                )
                ct_energy_terms.append(float(bid.price_eur_mw) * model.rd_cumulative_volume_term[bid_mrid])
                if bid_mrid in model.rd_activation_bid_ids:
                    ct_activation_terms.append(float(bid.activation_cost_eur) * model.rd_bid_activated[bid_mrid])
            _append_named_term("ct_pra_penalty_cost", sum(ct_penalty_terms))
            _append_named_term("ct_pra_energy_cost", sum(ct_energy_terms))
            if ct_activation_terms:
                _append_named_term("ct_bid_activation_cost", sum(ct_activation_terms))

    # HVDC preventive penalty (H7.3): both units pay their own factor x reference x |cumulative change|;
    # the term is recorded by name for the results summary
    if hasattr(model, "hvdc_cost_volume"):
        _append_named_term("hvdc_pra_penalty_cost", sum(
            model.hvdc_info[u]["penalty_factor"] * float(hvdc_inputs.penalty_reference_pra) * model.hvdc_cost_volume[u]
            for u in model.hvdc_unit_ids
        ))

    # HVDC curative penalty (H7.4): factor x reference x the maximum over the contingencies of the unit's
    # curative change (up + down), lower-bounded max variables, no energy or reserve price
    if hasattr(model, "hvdc_cur_max"):
        _append_named_term("hvdc_cra_penalty_cost", sum(
            model.hvdc_cur_info[u]["penalty_factor"] * float(hvdc_inputs.penalty_reference_cra) * model.hvdc_cur_max[u]
            for u in model.hvdc_cur_unit_ids
        ))

    # Curative RD reservation cost on the max-over-contingencies variables (worst-case
    # reservation, not a sum), gated on max_curative_rd_volume_upward: (penalty factor x the
    # preventive penalty reference + the reservation price) x max volume, the reservation
    # price being the bid's reservePrice, else its energy price (curative_rd_reservation_price_eur_mw);
    # for one-directional bids each bid pays its own direction only. No start-up and no
    # activation cost for curative bids (2547, 6097).
    if hasattr(model, "max_curative_rd_volume_upward"):
        penalty_cost_per_mw_and_tso = get_effective_rd_penalty_cost_per_mw(
            rd_inputs.rd_list, rd_inputs.penalty_reference_rd_pra, rd_inputs.penalty_reference_rd_pra_per_tso,
        )
        bid_by_mrid_all = {str(bid.mrid): bid for bid in rd_inputs.bid_list}

        curative_rd_cost_terms = []
        ct_curative_bid_ids = getattr(model, "ct_curative_bid_ids", frozenset())
        for bid_mrid in model.bid_ids_curative:
            if bid_mrid in ct_curative_bid_ids:
                continue    # countertrade bids: named terms below
            bid = bid_by_mrid_all[bid_mrid]
            penalty_cost_per_mw, _bid_tso = penalty_cost_per_mw_and_tso.get(str(bid.ra_mrid), (0.0, ""))
            curative_rd_cost_terms.append(
                (penalty_cost_per_mw + curative_rd_reservation_price_eur_mw(bid))
                * (model.max_curative_rd_volume_upward[bid_mrid] + model.max_curative_rd_volume_downward[bid_mrid])
            )

        if curative_rd_cost_terms:
            _append_logged_term("curative_rd_reservation_cost", sum(curative_rd_cost_terms))

        # curative countertrade (C7.5): penalty factor x CRA reference and the reservation price (reservePrice, else
        # the energy price), both on the max-over-contingencies volume of the bid
        if ct_curative_bid_ids:
            ct_unit_by_ra = {str(rd.mrid): rd for rd in rd_inputs.rd_list if rd.countertrade}
            ct_cra_penalty_terms, ct_cra_reservation_terms = [], []
            for bid_mrid in model.bid_ids_curative:
                if bid_mrid not in ct_curative_bid_ids:
                    continue
                bid = bid_by_mrid_all[bid_mrid]
                maximum = model.max_curative_rd_volume_upward[bid_mrid] + model.max_curative_rd_volume_downward[bid_mrid]
                ct_cra_penalty_terms.append(
                    float(ct_unit_by_ra[str(bid.ra_mrid)].penalty_factor) * float(rd_inputs.penalty_reference_ct_cra) * maximum
                )
                ct_cra_reservation_terms.append(curative_rd_reservation_price_eur_mw(bid) * maximum)
            _append_named_term("ct_cra_penalty_cost", sum(ct_cra_penalty_terms))
            _append_named_term("ct_cra_reservation_cost", sum(ct_cra_reservation_terms))

    # RA count slack penalty: ra_count_slack_per_timestamp exists only when soft and a TSO has a limit
    if hasattr(model, "ra_count_slack_per_timestamp"):
        _append_named_term(
            "ra_count_timestamp",
            rd_inputs.ra_limit_penalty
            * sum(model.ra_count_slack_per_timestamp[tso] for tso in model.tsos_with_timestamp_ra_limit)
        )

    # over secured_branch_ids (scanned branches are penalised separately below)
    _append_named_term(
        "overload_n0",
        penalty_cost_overload_n0
        * sum(model.overload_slack_n0[branch_mrid] for branch_mrid in secured_branch_ids)
    )

    # N-0 scanned slack penalty (soft mode): scanned_element_penalty_eur_per_ka is EUR/kA and
    # the slack is in A, hence / 1000
    if hasattr(model, "scanned_overload_slack"):
        _append_named_term(
            "overload_n0_scanned",
            scanned_element_penalty_eur_per_ka
            * sum(model.scanned_overload_slack[branch_mrid] / 1000.0 for branch_mrid in scanned_branch_ids)
        )

    # N-X PATL slack penalty, gated on nx_patl_overload_slack (an empty pair set sums to 0)
    if hasattr(model, "nx_patl_overload_slack"):
        _append_named_term(
            "overload_nx_patl",
            nx_inputs.penalty_cost_overload_nx_patl
            * sum(model.nx_patl_overload_slack[pair] for pair in model.nx_branch_contingency_pairs)
        )

    # N-X TATL slack penalty (its own rate, penalty_cost_overload_nx_tatl)
    if hasattr(model, "nx_tatl_overload_slack"):
        _append_named_term(
            "overload_nx_tatl",
            nx_inputs.penalty_cost_overload_nx_tatl
            * sum(model.nx_tatl_overload_slack[pair] for pair in model.nx_tatl_branch_contingency_pairs)
        )

    # N-X scanned slack penalty (its own rate, penalty_cost_overload_nx_scanned)
    if hasattr(model, "nx_scanned_overload_slack"):
        _append_named_term(
            "overload_nx_scanned",
            nx_inputs.penalty_cost_overload_nx_scanned
            * sum(model.nx_scanned_overload_slack[pair] for pair in model.nx_scanned_branch_contingency_pairs)
        )

    model.objective = pyo.Objective(expr=sum(objective_terms), sense=pyo.minimize)

    return model
