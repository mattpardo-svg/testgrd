"""
data_model.py — shared dataclasses and enums of the RAO, plus select_by_side, the
per-side value picker every module that reads branch currents uses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AEType(str, Enum):
    """Assessed-element classification (secured, scanned, excluded)."""
    SECURED  = "secured"   # included in MILP as XNE
    SCANNED  = "scanned"   # monitored only; scanned-element constraint in the MILP
    EXCLUDED = "excluded"  # absent from all MILP constraints; present in SA only


class ContingencyType(str, Enum):
    ORDINARY     = "OrdinaryContingency"
    EXCEPTIONAL  = "ExceptionalContingency"
    OUT_OF_RANGE = "OutOfRangeContingency"


class RAKind(str, Enum):
    PREVENTIVE = "preventive"
    CURATIVE   = "curative"


class BidDirection(str, Enum):
    UP   = "up"
    DOWN = "down"


class AcVerificationStatus(str, Enum):
    """How the AC verification loop ended; written next to the raw termination reason in
    every result file. MAX_ITERATIONS_REACHED: the last solution is applied and its load
    flow converged, the criteria did not hold. NOT_APPLICABLE: a result written without
    any AC verification. Map a termination reason with from_termination_reason."""
    CONVERGED = "CONVERGED"
    REVERTED_NON_CONVERGENT = "REVERTED_NON_CONVERGENT"
    # A later load flow did not converge: the network is back at the last converged state.
    # REVERTED_NON_CONVERGENT: nothing was applied, the network is the original one.
    REVERTED_TO_LAST_CONVERGED_STATE = "REVERTED_TO_LAST_CONVERGED_STATE"
    MAX_ITERATIONS_REACHED = "MAX_ITERATIONS_REACHED"
    # The new applied solution repeats an earlier one (not the previous iteration) within
    # max_change_ra; the last applied state is kept.
    CYCLIC_REPETITION = "CYCLIC_REPETITION"
    SENSITIVITY_INVALID = "SENSITIVITY_INVALID"
    # A MILP solve ended non-optimal (infeasible or other): the loop stopped with the last
    # applied state (the original network when nothing was applied).
    SOLVER_NON_OPTIMAL = "SOLVER_NON_OPTIMAL"
    # calculation_mode ac_single_pass: one build, one solve, one applied solution and one AC
    # check; there is no convergence criterion, so the run is neither converged nor capped.
    SINGLE_PASS = "SINGLE_PASS"
    NOT_APPLICABLE = "NOT_APPLICABLE"

    @classmethod
    def from_termination_reason(cls, termination_reason: str) -> "AcVerificationStatus":
        # The mapping lives at module level: inside the Enum body any non-dunder name
        # would become an enum member.
        try:
            return cls(_AC_VERIFICATION_TERMINATION_REASON_MAP[termination_reason])
        except KeyError:
            raise ValueError(
                f"Unrecognized AC verification termination_reason: {termination_reason!r} "
                f"— ac_convergence_loop.py's own termination_reason strings have changed "
                f"without AcVerificationStatus.from_termination_reason being updated to match."
            )


# Every termination reason of ac_convergence_loop mapped to its status. An unknown reason
# raises ValueError in from_termination_reason (never a silent default).
_AC_VERIFICATION_TERMINATION_REASON_MAP = {
    "remedial action movement and congestion levels stabilized between iterations": "CONVERGED",
    "load flow non-convergent": "REVERTED_NON_CONVERGENT",
    "load flow non-convergent - reverted to last converged state": "REVERTED_TO_LAST_CONVERGED_STATE",
    "max iterations reached": "MAX_ITERATIONS_REACHED",
    "cyclic repetition of results": "CYCLIC_REPETITION",
    "invalid sensitivities - no usable earlier valid set": "SENSITIVITY_INVALID",
    "MILP solve not optimal": "SOLVER_NON_OPTIMAL",
    "single pass completed": "SINGLE_PASS",
}


# ---------------------------------------------------------------------------
# Assessed element (XNE)
# ---------------------------------------------------------------------------

@dataclass
class AssessedElement:
    """One assessed element (cross-border or national network element).

    patl_ka: timestamp -> PATL in kA (SHS CurrentLimitTimePoint, else the static fallback).
    branch_mrid: the CGM branch (ACLineSegment or PowerTransformer); None = logged and skipped.
    """
    mrid:                str
    name:                str
    tso:                 str
    ae_type:             AEType
    branch_mrid:         Optional[str]
    secured_for_region:  Optional[str]      # securedForRegionCode
    scanned_for_region:  Optional[str]      # scannedForRegionCode
    exclusion_reason:    Optional[str]      # exclusionReason (None ⇒ secured)
    patl_ka:             dict[str, float]   # {timestamp → kA}
    # Side (1 or 2) of the CurrentLimit's terminal: PATL, current, sensitivity and static
    # PATL of this branch all use this side. 1 when the limit could not be matched (logged).
    branch_side:         int = 1
    # NCP switches applied at parse time (ASSUMPTIONS, assessed-element switches). A disabled element has ae_type EXCLUDED.
    in_base_case:        bool = True                 # False: not monitored in the N state, only after contingencies
    included_contingencies: Optional[tuple[str, ...]] = None   # not None: monitored only after these contingencies
    excluded_contingencies: tuple[str, ...] = ()               # not monitored after these contingencies
    switch_notes:        tuple[str, ...] = ()        # what was applied, for the log


class LfNx(dict):
    """lf_nx = {contingency: {branch: {side: A}}} with how its gaps came about (plain attributes; a plain dict has none):
    not_monitored: monitored branches that have no N-0 result (left out for every contingency);
    tripped: {contingency: monitored branches that are its own equipment (no current by definition)};
    dropped: {contingency: monitored branches that have a result in N-0 but none after this contingency (islanded):
    dropped for that contingency only, the contingency itself is evaluated); guard_dropped: the pairs the sensitivity
    guard left out, per cause (no reference current, deviation)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.not_monitored: set[str] = set()
        self.tripped: dict[str, list[str]] = {}
        self.dropped: dict[str, list[str]] = {}
        # the "contingency/branch" pairs the sensitivity guard left out (also in dropped), per cause
        self.guard_dropped: dict[str, list[str]] = {}


def monitored_after_contingency(ae: "AssessedElement", contingency_mrid: str) -> bool:
    """True when the assessed element is monitored in the state after the contingency (links included / excluded)."""
    if contingency_mrid in ae.excluded_contingencies:
        return False
    return ae.included_contingencies is None or contingency_mrid in ae.included_contingencies


def branch_monitoring_rules(ae_flat_list: list["AssessedElement"]) -> tuple[set[str], dict[str, list["AssessedElement"]]]:
    """(branches that are not monitored in the N state, {branch: its SECURED / SCANNED elements that carry a contingency
    restriction}). A branch is monitored in the N state when one of its elements is (inBaseCase); after a contingency
    when one of its elements is (links). Branches without any restriction are in neither result."""
    by_branch: dict[str, list[AssessedElement]] = {}
    for ae in ae_flat_list:
        if ae.ae_type in (AEType.SECURED, AEType.SCANNED) and ae.branch_mrid:
            by_branch.setdefault(str(ae.branch_mrid), []).append(ae)
    not_in_base_case = {b for b, aes in by_branch.items() if not any(a.in_base_case for a in aes)}
    restricted = {
        b: aes for b, aes in by_branch.items()
        if any(a.included_contingencies is not None or a.excluded_contingencies for a in aes)
    }
    return not_in_base_case, restricted


def pair_is_monitored(restricted: dict[str, list["AssessedElement"]], branch_mrid: str, contingency_mrid: str) -> bool:
    """False when every element of the branch is restricted away from the contingency (see branch_monitoring_rules)."""
    aes = restricted.get(branch_mrid)
    return True if aes is None else any(monitored_after_contingency(a, contingency_mrid) for a in aes)


# ---------------------------------------------------------------------------
# Contingency
# ---------------------------------------------------------------------------

@dataclass
class Contingency:
    """One contingency (N-1 or N-k).

    must_study: timestamp -> bool. equipment_mrids: CGM equipment tripped together;
    equipment_types: the CoMo type of each (index-aligned), taken from the pypowsybl
    accessor the mRID was found in ("bus" has no equivalent and never appears).
    tso: display name of the TSO whose CO file holds the contingency.
    """
    mrid:              str
    name:              str
    tso:               str
    contingency_type:  ContingencyType
    must_study:        dict[str, bool]      # {timestamp → bool}
    equipment_mrids:   list[str]            # CGM mRIDs of tripped elements
    equipment_types:   list[str] = field(default_factory=list)  # parallel to equipment_mrids


# ---------------------------------------------------------------------------
# PST remedial action
# ---------------------------------------------------------------------------

@dataclass
class PST:
    """One PST remedial action.

    low_step / high_step: TapChanger hardware limits (CGM). operating_tap_min/max: from the
    SSI > RA profile > IGM priority chain; *_relative: optional relative limits (None if absent).
    r0: current tap position. group_id: set when several TapPositionActions share one remedial action.
    """
    mrid:               str
    name:               str
    tso:                str
    available:          bool
    kind:               RAKind
    penalty_factor:     float
    low_step:           int
    high_step:          int
    r0:                 int                  # current tap position
    operating_tap_min:  int                  # operating lower bound (≥ low_step)
    operating_tap_max:  int                  # operating upper bound (≤ high_step)
    group_id:           Optional[str]        # None if not in a group
    low_step_relative:  Optional[int] = None # relative lower limit from current tap, if provided
    high_step_relative: Optional[int] = None # relative upper limit from current tap, if provided
    # Curative fields: None / [] when not in the data, so "absent" differs from "zero".
    reaction_time_minutes:   Optional[float] = None  # from RemedialAction.timeToImplement (GSARA-level)
    time_per_stage_minutes:  Optional[float] = None  # from GridStateAlteration.timePerStage (TapPositionAction-level)
    linked_contingency_mrids: list[str] = field(default_factory=list)  # via ContingencyWithRemedialAction
    # links with combinationConstraintKind excluded: the RA is not used in these contingencies
    excluded_contingency_mrids: list[str] = field(default_factory=list)
    linked_xne_mrids:         list[str] = field(default_factory=list)  # via AssessedElementWithRemedialAction
    ra_mrid:                  str = field(default="", compare=False)  # the remedial action this row comes from (messages only)


def effective_linked_contingencies(linked, excluded, all_contingencies) -> list[str]:
    """The contingencies a curative RA acts in: its included links, or every contingency of
    `all_contingencies` when it has none, minus its excluded ones (order of the first list kept)."""
    base = list(linked) if linked else list(all_contingencies)
    skipped = set(excluded)
    return [c for c in base if c not in skipped]


# ---------------------------------------------------------------------------
# RD remedial action (generating unit)
# ---------------------------------------------------------------------------

@dataclass
class RDUnit:
    """One redispatch remedial action linked to a GeneratingUnit.

    initial_p: scheduled output (MW); ramp rates in MW/min, None if not in the CGM.
    penalty_factor: RA-level RemedialAction.penaltyFactor (default 1.0) — not the bid-level
    PowerBid.penalty_factor of the SIS file.
    """
    mrid:             str
    name:             str
    tso:              str
    gu_mrid:          str
    gu_name:          str
    available:        bool
    kind:             RAKind
    initial_p:        float              # MW — initialP from CGM
    max_p:            float              # MW — maxP
    min_p:            float              # MW — minP
    raise_ramp_rate:  Optional[float]   # MW/min — raiseRampRate
    lower_ramp_rate:  Optional[float]   # MW/min — lowerRampRate
    penalty_factor:   float = 1.0        # RemedialAction.penaltyFactor (RA-level)
    # Curative linkage, same CIM classes as for PSTs, keyed by this unit's own mrid.
    linked_contingency_mrids: list[str] = field(default_factory=list)
    excluded_contingency_mrids: list[str] = field(default_factory=list)
    linked_xne_mrids:         list[str] = field(default_factory=list)
    # RemedialAction.timeToImplement in minutes; None = not given (treated as 0: always available)
    reaction_time_minutes:    Optional[float] = None
    # True for a countertrade RA (docs/specs/countertrade_ra_specs.md): the MILP treats it as an RD unit
    # per unit of its bidding zone (mrid = the CT RA), the result files tell the two apart
    countertrade:             bool = False
    # countertrade only: this unit's weight in the up and in the down direction of the RA (0 = it does not move)
    ct_weight_up:             float = 0.0
    ct_weight_down:           float = 0.0


# ---------------------------------------------------------------------------
# Power bid (per RD unit, per direction, per timestamp)
# ---------------------------------------------------------------------------

@dataclass
class PowerBid:
    """One power bid attached to an RD remedial action.

    available_mw maps timestamp → offered MW (upward or downward capacity).
    Scalar fields are static across timestamps (from the NCP profile).
    Optional fields default to None when absent from the NCP.
    """
    mrid:               str
    ra_mrid:            str             # parent RDUnit.mrid
    direction:          BidDirection
    available_mw:       dict[str, float]   # {timestamp → MW}
    price_eur_mw:       float
    lead_time_h:        float
    activation_cost_eur: Optional[float]  # None if not in NCP (the MILP treats it as 0)
    # PowerBidScheduleTimePoint.reservePrice at the run timestamp (EUR/MW): the curative
    # reservation price; None = not given (the energy price is used, logged)
    reserve_price_eur_mw: Optional[float] = None


# ---------------------------------------------------------------------------
# Unit start-up / shut-down costs (per generating unit)
# ---------------------------------------------------------------------------

@dataclass
class UnitCost:
    """Start-up and shut-down cost (EUR) of one generating unit at the run timestamp; None =
    not given, which forbids that state change (BUC 14#103: no cost, no start-up). Sources in
    priority order: start-up UnitCostTimePoint.warmStartupCost, UnitCostTimePoint.startupCost,
    GeneratingUnit.normalWarmStartupCost (ER); shut-down UnitCostTimePoint.shutdownCost,
    GeneratingUnit.normalShutdownCost (ER)."""
    startup_cost_eur:  Optional[float] = None
    shutdown_cost_eur: Optional[float] = None


# ---------------------------------------------------------------------------
# HVDC remedial action (one converter unit = one EquivalentInjection of the CGM)
# ---------------------------------------------------------------------------

# RangeConstraint.direction / valueKind tokens (the text after the last "." of the enumeration)
HVDC_DIRECTIONS = ("up", "down", "upAndDown", "none")
HVDC_VALUE_KINDS = ("absolute", "incremental")


@dataclass
class HvdcRange:
    """One static RangeConstraint of an HVDC RA. value: absolute = a set point in generation sign
    (MW); incremental = a magnitude (MW) added to (up) or subtracted from (down) the original P0."""
    direction: str      # one of HVDC_DIRECTIONS
    value_kind: str     # one of HVDC_VALUE_KINDS
    value: float


@dataclass
class HvdcUnit:
    """One HVDC remedial action: a GridStateAlterationRemedialAction with one
    EquivalentInjectionAction. mrid is the RA's mRID, ei_mrid the EquivalentInjection (the
    pypowsybl generator id). penalty_factor: RemedialAction.penaltyFactor, 1.0 when missing.
    ranges: the static RangeConstraints of the action; empty = none given.
    group_mrid: the balanced RemedialActionGroup (None until attached by attach_hvdc_groups)."""
    mrid:             str
    name:             str
    tso:              str
    ei_mrid:          str
    available:        bool
    kind:             RAKind
    penalty_factor:   float
    reaction_time_minutes:    Optional[float] = None
    linked_contingency_mrids: list[str] = field(default_factory=list)
    excluded_contingency_mrids: list[str] = field(default_factory=list)
    ranges:           list[HvdcRange] = field(default_factory=list)
    group_mrid:       Optional[str] = None


@dataclass
class HvdcGroup:
    """A RemedialActionGroup with a balanced RemedialActionDependency: ra_mrids are all the
    remedial actions that depend on the group (HVDC or not; a foreign one is a data error)."""
    mrid:     str
    name:     str
    tso:      str
    ra_mrids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Countertrade remedial action (docs/specs/countertrade_ra_specs.md)
# ---------------------------------------------------------------------------

@dataclass
class CtRa:
    """One countertrade remedial action: a nc:CountertradeRemedialAction of one bidding zone with
    the strategy it declares (only proportionalForRemainingCapacity is implemented). Its bids are
    PowerBid objects with ra_mrid = mrid. penalty_factor: RemedialAction.penaltyFactor, 1.0 when
    missing. reaction_time_minutes None = not given (0, always available). zone_name is filled when
    the zone is resolved against the Organisation profile."""
    mrid:             str
    name:             str
    tso:              str
    zone_mrid:        str
    available:        bool
    kind:             RAKind
    penalty_factor:   float
    strategy:         str
    zone_name:        str = ""
    reaction_time_minutes:    Optional[float] = None
    linked_contingency_mrids: list[str] = field(default_factory=list)
    excluded_contingency_mrids: list[str] = field(default_factory=list)
    linked_xne_mrids:         list[str] = field(default_factory=list)


@dataclass
class CtZoneUnit:
    """One generating unit of a bidding zone that can take part in a countertrade: gu_mrid is the
    network generator id (the SynchronousMachine), unit_mrid the CGMES GeneratingUnit mRID. prod,
    pmin, pmax: the ORIGINAL set point and the unit limits (MW, generation sign)."""
    gu_mrid:    str
    unit_mrid:  str
    name:       str
    prod:       float
    pmin:       float
    pmax:       float
    connected:  bool = True


# ---------------------------------------------------------------------------
# Scanned element
# ---------------------------------------------------------------------------

@dataclass
class ScannedElement:
    """Derived from an AssessedElement with ae_type SCANNED, per timestamp:
    patl_scanned_ka = patl_cgm_ka × (1 + virtual_load_margin_pct / 100);
    effective_limit_ka = max(patl_scanned_ka, initial loading), = patl_scanned_ka without lf_initial.
    virtual_load_margin_pct 0-100 (0 when absent). soft: soft constraint with `penalty` (EUR).
    """
    mrid:                   str
    name:                   str
    tso:                    str
    branch_mrid:            str
    virtual_load_margin_pct: int                  # 0–100
    patl_cgm_ka:            dict[str, float]      # {timestamp → kA}
    patl_scanned_ka:        dict[str, float]      # {timestamp → kA}
    effective_limit_ka:     dict[str, float]      # {timestamp → kA}
    soft:                   bool  = False
    penalty:                float = 0.0


# ---------------------------------------------------------------------------
# Per-side values
# ---------------------------------------------------------------------------

def select_by_side(per_side: dict, side: int, branch_mrid: str = "") -> Optional[float]:
    """Value of the resolved side (1 or 2) of a per-side dict; falls back to the other side
    (logged); None (not 0.0) when neither side has data or per_side is empty or None."""
    if not per_side:
        return None
    if side in per_side:
        return per_side[side]
    other_side = 2 if side == 1 else 1
    if other_side in per_side:
        log.warning(
            "_select_by_side: branch %s missing side %d data, using side %d instead",
            branch_mrid, side, other_side,
        )
        return per_side[other_side]
    return None
