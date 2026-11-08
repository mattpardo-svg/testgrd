"""n0_results_export.py — N-0-level result CSVs of the MILP (PST, curative PST, RD, RA
count, RA groups, lead times, summary, generating units, lf_post_n0, completeness iterations).
Curative and preventive tap changes are whole numbers when pst_taping_discrete is true (the
default) and continuous model values, written unrounded, otherwise.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

from data_model import AcVerificationStatus, BidDirection, PST, PowerBid, RAKind, RDUnit, select_by_side
from milp.milp_builder import (
    _compute_radp,
    _compute_earliest_usable_timestamp,
    curative_rd_reservation_price_eur_mw,
    get_effective_rd_penalty_cost_per_mw,
)
from text_encoding import TEXT_ENCODING

log = logging.getLogger(__name__)

# values at or below these count as zero: tap headroom, MW volume, overload slack (A)
_ZERO_TAPS = 1e-6
_ZERO_VOLUME_MW = 1e-9
_ZERO_SLACK_AMPERE = 1e-6


# ---------------------------------------------------------------------------
# From milp_solver.py
# ---------------------------------------------------------------------------

def write_pst_results_csv(
    pst_results: Dict[str, Dict[str, object]],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
    curative_pst_results: Optional[Dict[str, Dict[str, object]]] = None,
) -> None:
    """results_pst.csv. curative_pst_results (from extract_curative_pst_results) adds the
    columns curative_tap_change_total and curative_penalty_cost_eur (0.0 without it); a
    transformer with only a curative RA gets a row with neutral preventive columns.
    ac_verification_status/_termination_reason are required (no default) so a reverted run is
    always marked; the proposed values are still written on a REVERTED_NON_CONVERGENT run
    (diagnostics), the status column carries the correctness signal.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "pst_mrid",
            "pst_name",
            "tso",
            "original_tap_position",
            "tap_change_result",
            "result_tap_position",
            "pst_penalty_factor",
            "penalty_reference_pst_pra",
            "effective_penalty_cost_per_tap_eur",
            "penalty_cost_eur",
            "timestamp",
            "ac_verification_status",
            "ac_verification_termination_reason",
            "curative_tap_change_total",
            "curative_penalty_cost_eur",
            "last_solve_start_tap_position",
            "last_solve_tap_change",
        ])
        curative_pst_results = curative_pst_results or {}
        all_mrids = sorted(set(pst_results) | set(curative_pst_results))
        for pst_mrid in all_mrids:
            curative_row = curative_pst_results.get(pst_mrid)
            curative_taps = curative_row["curative_tap_change_total"] if curative_row else 0.0
            curative_cost = curative_row["curative_penalty_cost_eur"] if curative_row else 0.0
            row = pst_results.get(pst_mrid)
            if row is None:
                # curative-only transformer: no preventive variables exist, so
                # the preventive columns are neutral (nothing moved preventively)
                row = {
                    "pst_name": curative_row["pst_name"],
                    "tso": curative_row["tso"],
                    "current_tap_position": curative_row["current_tap_position"],
                    "tap_change_result": 0.0,
                    "result_tap_position": curative_row["current_tap_position"],
                    "pst_penalty_factor": curative_row["pst_penalty_factor"],
                    "penalty_reference_pst_pra": curative_row["penalty_reference_pst_pra"],
                    "effective_penalty_cost_per_tap_eur": curative_row["effective_penalty_cost_per_tap_eur"],
                    "penalty_cost_eur": 0.0,
                }
            writer.writerow([
                pst_mrid,
                row["pst_name"],
                row["tso"],
                row.get("original_tap_position", row.get("current_tap_position")),
                row["tap_change_result"],
                row["result_tap_position"],
                row["pst_penalty_factor"],
                row["penalty_reference_pst_pra"],
                row["effective_penalty_cost_per_tap_eur"],
                row["penalty_cost_eur"],
                timestamp,
                ac_verification_status.value,
                ac_verification_termination_reason,
                curative_taps,
                curative_cost,
                row.get("last_solve_start_tap_position", row.get("original_tap_position", row.get("current_tap_position"))),
                row.get("last_solve_tap_change", row["tap_change_result"]),
            ])

    log.info("Written PST results CSV: %s (%d rows)", output_path, len(all_mrids))


def extract_curative_pst_results(
    milp_model,
    pst_list: Optional[list[PST]],
    penalty_reference_pst_pra: float,
    penalty_reference_pst_pra_per_tso: Optional[dict[str, float]] = None,
) -> Dict[str, Dict[str, object]]:
    """Per-transformer curative PST totals: curative_tap_change_total = sum over contingencies
    of (up + down); curative_penalty_cost_eur = curative RA penalty_factor x (per-TSO override
    or penalty_reference_pst_pra) x that total, as in the objective (no max). Only curative
    PST objects are read. {} when the model has no curative PST pairs."""
    if not hasattr(milp_model, "curative_pst_contingency_pairs") or not pst_list:
        return {}
    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to extract curative PST results") from exc

    per_tso = penalty_reference_pst_pra_per_tso or {}
    curative_pst_by_mrid = {str(p.mrid): p for p in pst_list if p.kind == RAKind.CURATIVE}
    totals: Dict[str, float] = {}
    for pst_mrid, co_mrid in milp_model.curative_pst_contingency_pairs:
        totals[str(pst_mrid)] = totals.get(str(pst_mrid), 0.0) + float(
            pyo.value(milp_model.curative_tap_change_upward[pst_mrid, co_mrid])
            + pyo.value(milp_model.curative_tap_change_downward[pst_mrid, co_mrid])
        )

    results: Dict[str, Dict[str, object]] = {}
    for pst_mrid, total in sorted(totals.items()):
        pst = curative_pst_by_mrid.get(pst_mrid)
        if pst is None:
            continue
        reference = float(per_tso.get(pst.tso, penalty_reference_pst_pra))
        cost_per_tap = float(pst.penalty_factor) * reference
        results[pst_mrid] = {
            "pst_name": pst.name,
            "tso": pst.tso,
            "current_tap_position": int(getattr(pst, "r0", 0)),
            "pst_penalty_factor": float(pst.penalty_factor),
            "penalty_reference_pst_pra": reference,
            "effective_penalty_cost_per_tap_eur": cost_per_tap,
            "curative_tap_change_total": total,
            "curative_penalty_cost_eur": cost_per_tap * total,
        }
    return results


def extract_curative_pst_delta_results(
    milp_model,
    pst_list: Optional[list[PST]],
    pst_tap_bounds_curative: Optional[dict[str, tuple[int, int]]],
    pst_cra_availability: Optional[dict[tuple[str, str, str], dict]],
    penalty_reference_pst_pra: float,
    penalty_reference_pst_pra_per_tso: Optional[dict[str, float]] = None,
) -> list[dict]:
    """One row per (curative PST, contingency) pair with variables in the model, sorted.
    curative_delta_taps_up/_down: optimised (continuous) values; net = up - down.
    curative_tap_lower/upper_bound: the absolute bounds the builder used
    (pst_tap_bounds_curative), blank when none. available: from pst_cra_availability (blank
    without it). penalty_cost_eur = penalty_factor x reference x (up + down), so the rows add
    up to curative_penalty_cost_eur of results_pst.csv. [] without curative PST pairs.
    """
    if not hasattr(milp_model, "curative_pst_contingency_pairs") or not pst_list:
        return []
    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to extract curative PST delta results") from exc

    per_tso = penalty_reference_pst_pra_per_tso or {}
    bounds = pst_tap_bounds_curative or {}
    curative_pst_by_mrid = {str(p.mrid): p for p in pst_list if p.kind == RAKind.CURATIVE}
    available_pairs = None
    if pst_cra_availability is not None:
        available_pairs = {
            (str(pst_mrid), str(co_mrid))
            for (pst_mrid, _xne_mrid, co_mrid), entry in pst_cra_availability.items()
            if entry.get("available")
        }

    rows: list[dict] = []
    for pst_mrid, co_mrid in sorted(milp_model.curative_pst_contingency_pairs):
        pst = curative_pst_by_mrid.get(str(pst_mrid))
        if pst is None:
            continue
        up = float(pyo.value(milp_model.curative_tap_change_upward[pst_mrid, co_mrid]))
        down = float(pyo.value(milp_model.curative_tap_change_downward[pst_mrid, co_mrid]))
        reference = float(per_tso.get(pst.tso, penalty_reference_pst_pra))
        lower, upper = bounds.get(str(pst_mrid), (None, None))
        rows.append({
            "tso": pst.tso,
            "pst_mrid": str(pst_mrid),
            "contingency_mrid": str(co_mrid),
            "curative_delta_taps_up": up,
            "curative_delta_taps_down": down,
            "curative_delta_taps_net": up - down,
            "curative_tap_lower_bound": lower,
            "curative_tap_upper_bound": upper,
            "available": None if available_pairs is None else (str(pst_mrid), str(co_mrid)) in available_pairs,
            "penalty_cost_eur": float(pst.penalty_factor) * reference * (up + down),
        })
    return rows


def write_curative_pst_delta_results_csv(
    rows: list[dict],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_curative_pst.csv (columns: see extract_curative_pst_delta_results); header-only
    without rows; taps written unrounded."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "tso", "pst_mrid", "contingency_mrid", "timestamp",
            "curative_delta_taps_up", "curative_delta_taps_down", "curative_delta_taps_net",
            "curative_tap_lower_bound", "curative_tap_upper_bound", "available", "penalty_cost_eur",
            "ac_verification_status", "ac_verification_termination_reason",
        ])
        for row in rows:
            writer.writerow([
                row["tso"], row["pst_mrid"], row["contingency_mrid"], timestamp,
                row["curative_delta_taps_up"], row["curative_delta_taps_down"], row["curative_delta_taps_net"],
                "" if row["curative_tap_lower_bound"] is None else row["curative_tap_lower_bound"],
                "" if row["curative_tap_upper_bound"] is None else row["curative_tap_upper_bound"],
                "" if row["available"] is None else row["available"],
                row["penalty_cost_eur"],
                ac_verification_status.value, ac_verification_termination_reason,
            ])
    log.info("Written curative PST results CSV: %s (%d rows)", output_path, len(rows))


def write_curative_pst_group_results_csv(
    group_results: Dict[tuple, Dict[str, object]],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_curative_pst_groups.csv, one row per (group, contingency): group_id,
    contingency_mrid, pst_mrids (';'-joined), the common delta tap (up - down), timestamp and
    the ac_verification columns. Header-only without groups."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "group_id", "contingency_mrid", "pst_mrids", "group_delta_tap",
            "timestamp", "ac_verification_status", "ac_verification_termination_reason",
        ])
        for (group_id, co_mrid), row in sorted(group_results.items()):
            writer.writerow([
                group_id, co_mrid, ";".join(row["pst_mrids"]), row["group_delta_tap"],
                timestamp, ac_verification_status.value, ac_verification_termination_reason,
            ])
    log.info("Written curative PST group results CSV: %s (%d rows)", output_path, len(group_results))


def extract_curative_pst_group_results(
    milp_model,
    group_triples: list[tuple[str, str, str]],
    pst_list: Optional[list[PST]],
) -> Dict[tuple, Dict[str, object]]:
    """Common curative delta tap (up - down) of every curative PST group per contingency with
    a group constraint: {(group_id, co_mrid): {"pst_mrids": [...], "group_delta_tap": float}};
    {} without group constraints."""
    if not group_triples or not hasattr(milp_model, "curative_pst_group_equality"):
        return {}
    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to extract curative PST group results") from exc

    group_id_by_mrid = {
        str(p.mrid): str(p.group_id) for p in (pst_list or []) if p.kind == RAKind.CURATIVE and p.group_id
    }
    members: Dict[tuple, set] = {}
    for pst_1, pst_2, co_mrid in group_triples:
        group_id = group_id_by_mrid.get(pst_1)
        if group_id is None:
            continue
        members.setdefault((group_id, co_mrid), set()).update((pst_1, pst_2))

    results: Dict[tuple, Dict[str, object]] = {}
    for (group_id, co_mrid), pst_mrids in sorted(members.items()):
        first = sorted(pst_mrids)[0]
        delta = (
            pyo.value(milp_model.curative_tap_change_upward[first, co_mrid])
            - pyo.value(milp_model.curative_tap_change_downward[first, co_mrid])
        )
        results[(group_id, co_mrid)] = {"pst_mrids": sorted(pst_mrids), "group_delta_tap": float(delta)}
    return results


def extract_tso_tap_limit_results(milp_model, pst_list: Optional[list[PST]]) -> list[dict]:
    """Per TSO with preventive PSTs: taps used (sum of up + down over its PSTs), the applied
    limit (model.tso_tap_limit_values), the headroom and whether it binds. [] without a
    preventive PST block (header-only file)."""
    if not hasattr(milp_model, "tso_ids_for_tap_change_limit") or not hasattr(milp_model, "tap_change_upward"):
        return []
    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to extract TSO tap limit results") from exc

    model_pst_ids = set(milp_model.pst_ids)
    tso_by_mrid: Dict[str, str] = {}
    for pst in pst_list or []:
        if pst.kind == RAKind.PREVENTIVE and str(pst.mrid) in model_pst_ids:
            tso_by_mrid[str(pst.mrid)] = pst.tso   # one entry per transformer

    rows = []
    for tso in sorted(milp_model.tso_ids_for_tap_change_limit):
        total = sum(
            float(pyo.value(milp_model.cum_tap_up_term[m])) + float(pyo.value(milp_model.cum_tap_down_term[m]))
            if hasattr(milp_model, "cum_tap_up_term")
            else float(pyo.value(milp_model.tap_change_upward[m])) + float(pyo.value(milp_model.tap_change_downward[m]))
            for m, t in tso_by_mrid.items() if t == tso
        )
        limit = float(milp_model.tso_tap_limit_values[tso])
        remaining = limit - total
        rows.append({
            "tso": tso, "tap_total": total, "max_taps_per_tso": limit,
            "remaining": remaining, "binding": remaining <= _ZERO_TAPS,
        })
    return rows


def write_tso_tap_limit_results_csv(
    rows: list[dict],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_tso_tap_limits.csv — one row per TSO with preventive PSTs;
    header-only when there are none (project convention)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "tso", "tap_total", "max_taps_per_tso", "remaining", "binding",
            "timestamp", "ac_verification_status", "ac_verification_termination_reason",
        ])
        for row in rows:
            writer.writerow([
                row["tso"], row["tap_total"], row["max_taps_per_tso"], row["remaining"],
                str(bool(row["binding"])).lower(),
                timestamp, ac_verification_status.value, ac_verification_termination_reason,
            ])
    log.info("Written TSO tap limit results CSV: %s (%d rows)", output_path, len(rows))


def extract_curative_rd_results(
    milp_model,
    rd_list: list[RDUnit],
    bid_list: list[PowerBid],
    penalty_reference_rd_pra: float,
    penalty_reference_rd_pra_per_tso: Optional[dict[str, float]] = None,
) -> list[dict]:
    """One row per (curative bid, contingency) pair (model.curative_bid_contingency_pairs).
    curative_volume_mw: the variable of the bid's own direction; max_curative_volume_mw: its
    max-over-contingencies variable; penalty_cost_eur_per_mw from
    get_effective_rd_penalty_cost_per_mw. curative_cost_eur = (penalty/MW + reservation price) x
    (max_up + max_down), on the bid's first row only (0.0 on the others), so the column sums to
    the curative RD part of the objective; reservation price = the bid's reservePrice, else its
    energy price (curative_rd_reservation_price_eur_mw, as the objective). [] without curative RD."""
    if not hasattr(milp_model, "curative_bid_contingency_pairs"):
        return []
    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to extract curative RD results") from exc

    penalty_cost_per_mw_and_tso = get_effective_rd_penalty_cost_per_mw(
        rd_list, penalty_reference_rd_pra, penalty_reference_rd_pra_per_tso,
    )
    bid_by_mrid = {str(bid.mrid): bid for bid in bid_list}

    rows: list[dict] = []
    costed_bids: set[str] = set()
    ct_curative_bid_ids = getattr(milp_model, "ct_curative_bid_ids", frozenset())    # not RD results
    for bid_mrid, co_mrid in sorted(milp_model.curative_bid_contingency_pairs):
        if str(bid_mrid) in ct_curative_bid_ids:
            continue
        bid = bid_by_mrid.get(str(bid_mrid))
        if bid is None:
            log.warning("MILP model references curative PowerBid %s not found in PowerBidList — skipped", bid_mrid)
            continue
        is_up = bid.direction == BidDirection.UP
        volume_var = milp_model.curative_rd_volume_upward if is_up else milp_model.curative_rd_volume_downward
        max_var = milp_model.max_curative_rd_volume_upward if is_up else milp_model.max_curative_rd_volume_downward
        penalty_cost_per_mw, tso = penalty_cost_per_mw_and_tso.get(str(bid.ra_mrid), (0.0, ""))

        cost = 0.0
        if bid_mrid not in costed_bids:
            costed_bids.add(bid_mrid)
            max_total = float(pyo.value(milp_model.max_curative_rd_volume_upward[bid_mrid])) + float(
                pyo.value(milp_model.max_curative_rd_volume_downward[bid_mrid])
            )
            cost = (penalty_cost_per_mw + curative_rd_reservation_price_eur_mw(bid)) * max_total
        rows.append({
            "bid_mrid": str(bid_mrid),
            "ra_mrid": str(bid.ra_mrid),
            "tso": tso,
            "contingency_mrid": str(co_mrid),
            "direction": bid.direction.value,
            "curative_volume_mw": float(pyo.value(volume_var[bid_mrid, co_mrid])) + 0.0,  # + 0.0: no "-0.0"
            "max_curative_volume_mw": float(pyo.value(max_var[bid_mrid])) + 0.0,
            "price_eur_mw": float(bid.price_eur_mw),
            "reservation_price_eur_mw": curative_rd_reservation_price_eur_mw(bid),
            "penalty_cost_eur_per_mw": penalty_cost_per_mw,
            "curative_cost_eur": cost,
        })
    return rows


def write_curative_rd_results_csv(
    rows: list[dict],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_curative_rd.csv; header-only when there are no curative bids."""
    columns = [
        "bid_mrid", "ra_mrid", "tso", "contingency_mrid", "direction", "curative_volume_mw",
        "max_curative_volume_mw", "price_eur_mw", "reservation_price_eur_mw", "penalty_cost_eur_per_mw",
        "curative_cost_eur",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(columns + ["timestamp", "ac_verification_status", "ac_verification_termination_reason"])
        for row in rows:
            writer.writerow(
                [row[c] for c in columns]
                + [timestamp, ac_verification_status.value, ac_verification_termination_reason]
            )
    log.info("Written curative RD results CSV: %s (%d rows)", output_path, len(rows))


def write_results_rd_csv(
    rd_results: Dict[str, Dict[str, object]],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_preventive_rd.csv: [ra_mrid, bid_mrid, tso, rd_volume_net_mw, price_eur_mw,
    rd_penalty_factor, penalty_cost_eur, rd_cost_eur] plus the ac_verification columns
    (required, see write_pst_results_csv)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "ra_mrid",
            "bid_mrid",
            "tso",
            "rd_volume_net_mw",
            "price_eur_mw",
            "rd_penalty_factor",
            "penalty_cost_eur",
            "rd_cost_eur",
            "last_solve_rd_volume_net_mw",
            "timestamp",
            "ac_verification_status",
            "ac_verification_termination_reason",
        ])
        for bid_mrid, row in sorted(rd_results.items()):
            writer.writerow([
                row["ra_mrid"],
                bid_mrid,
                row["tso"],
                row["rd_volume_net_mw"],
                row["price_eur_mw"],
                row["rd_penalty_factor"],
                row["penalty_cost_eur"],
                row["rd_cost_eur"],
                row.get("last_solve_rd_volume_net_mw", row["rd_volume_net_mw"]),
                timestamp,
                ac_verification_status.value,
                ac_verification_termination_reason,
            ])

    log.info("Written RD results CSV: %s (%d rows)", output_path, len(rd_results))


def _last_solve_signed_rd_volume_mw(milp_model, bid_mrid: str) -> float:
    """Change the LAST solve made to this bid (up - down), in MW."""
    import pyomo.environ as pyo
    return float(pyo.value(milp_model.rd_volume_upward[bid_mrid])) - float(
        pyo.value(milp_model.rd_volume_downward[bid_mrid])
    )


def _cumulative_signed_rd_volume_mw(milp_model, bid_mrid: str) -> float:
    """Net volume of this bid against the ORIGINAL network: what earlier
    iterations already applied (model.rd_used_signed_volume_mw, signed: UP bids
    positive) plus the last solve's change. For a model built on the original
    network the used volume is 0 and this equals the last-solve change."""
    used = float(getattr(milp_model, "rd_used_signed_volume_mw", {}).get(bid_mrid, 0.0))
    return used + _last_solve_signed_rd_volume_mw(milp_model, bid_mrid)


def extract_rd_results(
    milp_model,
    rd_list: list[RDUnit],
    bid_list: list[PowerBid],
    penalty_reference_rd_pra: float,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
    penalty_reference_rd_pra_per_tso: Optional[dict[str, float]] = None,
    timestamp: str = "",
    output_path: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    """Per-bid RD results of an already-solved model (the RA penalty_factor and per-TSO
    override as in the objective). The ac_verification arguments are required (see
    write_pst_results_csv). {} when the model has no rd_volume_upward (no RD block); the CSV
    is still written, header-only (an empty file means "checked, nothing to report")."""
    if not hasattr(milp_model, "rd_volume_upward"):
        # write the header-only file: a missing file and an empty file are different signals
        if output_path is not None:
            write_results_rd_csv({}, output_path, timestamp, ac_verification_status, ac_verification_termination_reason)
        return {}

    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to extract RD results") from exc

    ra_penalty_factor_and_tso: dict[str, tuple[float, str]] = {
        str(rd.mrid): (float(rd.penalty_factor), rd.tso) for rd in rd_list
    }
    penalty_reference_rd_pra_per_tso = penalty_reference_rd_pra_per_tso or {}

    ct_bid_ids = getattr(milp_model, "ct_bid_ids", frozenset())    # countertrade bids are not RD results
    model_bid_ids = [str(b) for b in getattr(milp_model, "bid_ids", []) if str(b) not in ct_bid_ids]
    bid_by_mrid: dict[str, PowerBid] = {str(bid.mrid): bid for bid in bid_list}

    rd_results: Dict[str, Dict[str, object]] = {}
    for bid_mrid in model_bid_ids:
        bid = bid_by_mrid.get(bid_mrid)
        if bid is None:
            log.warning(
                "MILP model references PowerBid %s not found in PowerBidList — skipped",
                bid_mrid,
            )
            continue

        penalty_factor, tso = ra_penalty_factor_and_tso.get(str(bid.ra_mrid), (None, None))
        if penalty_factor is None:
            log.warning(
                "PowerBid %s references RA %s not found in RDList — "
                "penalty_cost_eur defaults to 0.0 for this bid",
                bid_mrid, bid.ra_mrid,
            )
            penalty_factor, tso = 0.0, ""

        # cumulative against the original network (== last solve on a first solve)
        last_solve_volume_net_mw = _last_solve_signed_rd_volume_mw(milp_model, bid_mrid)
        rd_volume_net_mw = _cumulative_signed_rd_volume_mw(milp_model, bid_mrid)

        effective_penalty_reference_rd_pra = float(
            penalty_reference_rd_pra_per_tso.get(tso, penalty_reference_rd_pra)
        )
        # each bid is one-directional, so abs(net) equals the objective's (up + down)
        activated_volume_mw = abs(rd_volume_net_mw)
        penalty_cost_eur = penalty_factor * effective_penalty_reference_rd_pra * activated_volume_mw
        rd_cost_eur = (
            penalty_factor * effective_penalty_reference_rd_pra + float(bid.price_eur_mw)
        ) * activated_volume_mw

        rd_results[bid_mrid] = {
            "ra_mrid": str(bid.ra_mrid),
            "tso": tso,
            "rd_volume_net_mw": rd_volume_net_mw,
            "last_solve_rd_volume_net_mw": last_solve_volume_net_mw,
            "price_eur_mw": float(bid.price_eur_mw),
            "rd_penalty_factor": penalty_factor,
            "penalty_cost_eur": penalty_cost_eur,
            "rd_cost_eur": rd_cost_eur,
        }

    log.info(
        "RD results extracted: %d bid(s), %d with non-zero net volume",
        len(rd_results),
        sum(1 for r in rd_results.values() if abs(r["rd_volume_net_mw"]) > _ZERO_VOLUME_MW),
    )

    if output_path is not None:
        write_results_rd_csv(rd_results, output_path, timestamp, ac_verification_status, ac_verification_termination_reason)

    return rd_results


# An RA counts as "activated" in results_ra_count when the absolute net RD volume of at least
# one of its bids exceeds this tolerance (MW). The model binary is reported separately
# (ra_activated_model_binary): it is bounded by the volume only from above and can be 1 at 0 MW.
RA_ACTIVATION_NET_VOLUME_TOLERANCE_MW = 1e-6


def write_results_ra_count_csv(
    ra_count_results: Dict[str, Dict[str, object]],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_preventive_rd_ra_count.csv: [tso, activated_ra_count, limit_per_timestamp,
    slack_per_timestamp], ra_activated_model_binary
    (raw sum of the binaries per TSO) and the ac_verification columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "tso",
            "activated_ra_count",
            "limit_per_timestamp",
            "slack_per_timestamp",
            "ra_activated_model_binary",
            "timestamp",
            "ac_verification_status",
            "ac_verification_termination_reason",
        ])
        for tso, row in sorted(ra_count_results.items()):
            writer.writerow([
                tso,
                row["activated_ra_count"],
                row["limit_per_timestamp"] if row["limit_per_timestamp"] is not None else "",
                row["slack_per_timestamp"] if row["slack_per_timestamp"] is not None else "",
                row["ra_activated_model_binary"],
                timestamp,
                ac_verification_status.value,
                ac_verification_termination_reason,
            ])

    log.info("Written RA count results CSV: %s (%d rows)", output_path, len(ra_count_results))


def extract_ra_count_results(
    milp_model,
    rd_list: list[RDUnit],
    bid_list: list[PowerBid],
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
    max_ra_count_per_timestamp_tso: Optional[dict[str, int]] = None,
    timestamp: str = "",
    output_path: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    """Per-TSO RA activation counts and count-limit slack of a solved model.
    activated_ra_count: RAs with a bid |up - down| > RA_ACTIVATION_NET_VOLUME_TOLERANCE_MW
    (bid_list is required to map volumes to RAs); ra_activated_model_binary: sum of binaries.
    Every TSO with an RA in the model or a limit is reported. {} without ra_activated (the
    CSV is still written, header-only).
    """
    max_ra_count_per_timestamp_tso = max_ra_count_per_timestamp_tso or {}

    if not hasattr(milp_model, "ra_activated"):
        if output_path is not None:
            write_results_ra_count_csv({}, output_path, timestamp, ac_verification_status, ac_verification_termination_reason)
        return {}

    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to extract RA count results") from exc

    ra_mrid_to_tso: dict[str, str] = {str(rd.mrid): rd.tso for rd in rd_list}
    ra_mrids_in_model = [str(m) for m in getattr(milp_model, "ra_mrids", [])]

    tsos_with_limit_defined = set(max_ra_count_per_timestamp_tso.keys())
    tsos_with_ra_in_model = {ra_mrid_to_tso.get(ra) for ra in ra_mrids_in_model if ra_mrid_to_tso.get(ra) is not None}
    all_tsos = sorted(tsos_with_limit_defined | tsos_with_ra_in_model)

    timestamp_limit_tsos = set(getattr(milp_model, "tsos_with_timestamp_ra_limit", []))

    bid_by_mrid: dict[str, PowerBid] = {str(bid.mrid): bid for bid in bid_list}
    ra_mrids_with_net_volume: set[str] = set()
    for bid_mrid in (str(b) for b in getattr(milp_model, "bid_ids", [])):
        bid = bid_by_mrid.get(bid_mrid)
        if bid is None:
            continue
        net_volume_mw = _cumulative_signed_rd_volume_mw(milp_model, bid_mrid)
        if abs(net_volume_mw) > RA_ACTIVATION_NET_VOLUME_TOLERANCE_MW:
            ra_mrids_with_net_volume.add(str(bid.ra_mrid))

    ra_count_results: Dict[str, Dict[str, object]] = {}
    for tso in all_tsos:
        activated_ra_count = float(sum(
            1 for ra_mrid in ra_mrids_in_model
            if ra_mrid_to_tso.get(ra_mrid) == tso and ra_mrid in ra_mrids_with_net_volume
        ))
        ra_activated_model_binary = sum(
            float(pyo.value(milp_model.ra_activated[ra_mrid]))
            for ra_mrid in ra_mrids_in_model
            if ra_mrid_to_tso.get(ra_mrid) == tso
        )

        limit_per_timestamp = max_ra_count_per_timestamp_tso.get(tso)
        slack_per_timestamp = None
        if tso in timestamp_limit_tsos:
            if hasattr(milp_model, "ra_count_slack_per_timestamp"):
                slack_per_timestamp = float(pyo.value(milp_model.ra_count_slack_per_timestamp[tso]))
            else:
                slack_per_timestamp = 0.0  # hard mode — no slack variable exists, always 0

        ra_count_results[tso] = {
            "activated_ra_count": activated_ra_count,
            "ra_activated_model_binary": ra_activated_model_binary,
            "limit_per_timestamp": limit_per_timestamp,
            "slack_per_timestamp": slack_per_timestamp,
        }

    log.info("RA count results extracted: %d TSO(s)", len(ra_count_results))

    if output_path is not None:
        write_results_ra_count_csv(ra_count_results, output_path, timestamp, ac_verification_status, ac_verification_termination_reason)

    return ra_count_results


def write_results_ra_groups_csv(
    ra_group_results: Dict[str, Dict[str, object]],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_preventive_rd_ra_groups.csv: [group_mrid, total_volume_upward_mw,
    max_volume_upward_mw, total_volume_downward_mw, max_volume_downward_mw] plus the
    ac_verification columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "group_mrid",
            "total_volume_upward_mw",
            "max_volume_upward_mw",
            "total_volume_downward_mw",
            "max_volume_downward_mw",
            "timestamp",
            "ac_verification_status",
            "ac_verification_termination_reason",
        ])
        for group_mrid, row in sorted(ra_group_results.items()):
            writer.writerow([
                group_mrid,
                row["total_volume_upward_mw"],
                row["max_volume_upward_mw"] if row["max_volume_upward_mw"] is not None else "",
                row["total_volume_downward_mw"],
                row["max_volume_downward_mw"] if row["max_volume_downward_mw"] is not None else "",
                timestamp,
                ac_verification_status.value,
                ac_verification_termination_reason,
            ])

    log.info("Written RA group results CSV: %s (%d rows)", output_path, len(ra_group_results))


def extract_ra_group_results(
    milp_model,
    ra_groups: Optional[dict[str, dict]],
    bid_list: list[PowerBid],
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
    timestamp: str = "",
    output_path: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    """Per-group total dispatched volume against its limit. Every group of ra_groups gets a
    row (a group without a limit: totals from its bids, max_* blank). {} when ra_groups is
    empty or there is no RD block (header-only CSV still written)."""
    ra_groups = ra_groups or {}
    if not ra_groups or not hasattr(milp_model, "rd_volume_upward"):
        if output_path is not None:
            write_results_ra_groups_csv({}, output_path, timestamp, ac_verification_status, ac_verification_termination_reason)
        return {}

    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to extract RA group results") from exc

    bid_by_mrid: dict[str, PowerBid] = {str(bid.mrid): bid for bid in bid_list}
    model_bid_ids = [str(b) for b in getattr(milp_model, "bid_ids", [])]

    ra_group_results: Dict[str, Dict[str, object]] = {}
    for group_mrid, group_data in ra_groups.items():
        member_ra_mrids = set(group_data.get("ra_mrids", []))
        group_bid_ids = [
            b for b in model_bid_ids
            if b in bid_by_mrid and str(bid_by_mrid[b].ra_mrid) in member_ra_mrids
        ]

        total_upward = sum(
            float(pyo.value(milp_model.rd_cumulative_up_term[b])) if hasattr(milp_model, "rd_cumulative_up_term")
            else float(pyo.value(milp_model.rd_volume_upward[b])) for b in group_bid_ids
        )
        total_downward = sum(
            float(pyo.value(milp_model.rd_cumulative_down_term[b])) if hasattr(milp_model, "rd_cumulative_down_term")
            else float(pyo.value(milp_model.rd_volume_downward[b])) for b in group_bid_ids
        )

        ra_group_results[group_mrid] = {
            "total_volume_upward_mw": total_upward,
            "max_volume_upward_mw": group_data.get("max_regulating_up"),
            "total_volume_downward_mw": total_downward,
            "max_volume_downward_mw": group_data.get("max_regulating_down"),
        }

    log.info("RA group results extracted: %d group(s)", len(ra_group_results))

    if output_path is not None:
        write_results_ra_groups_csv(ra_group_results, output_path, timestamp, ac_verification_status, ac_verification_termination_reason)

    return ra_group_results


def write_results_lead_times_csv(
    lead_time_results: Dict[str, Dict[str, object]],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_lead_times.csv: [bid_mrid, ra_mrid, bid_lead_time_hours, radp,
    earliest_usable_timestamp], phase (preventive or curative) and the ac_verification columns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "bid_mrid",
            "ra_mrid",
            "bid_lead_time_hours",
            "radp",
            "earliest_usable_timestamp",
            "phase",
            "timestamp",
            "ac_verification_status",
            "ac_verification_termination_reason",
        ])
        for bid_mrid, row in sorted(lead_time_results.items()):
            writer.writerow([
                bid_mrid,
                row["ra_mrid"],
                row["bid_lead_time_hours"],
                row["radp"],
                row["earliest_usable_timestamp"] if row["earliest_usable_timestamp"] is not None else "",
                row.get("phase", ""),
                timestamp,
                ac_verification_status.value,
                ac_verification_termination_reason,
            ])

    log.info("Written lead time results CSV: %s (%d rows)", output_path, len(lead_time_results))


def extract_lead_time_results(
    bid_list: list[PowerBid],
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
    raart: Optional[str] = None,
    timestamp: str = "",
    output_path: Optional[Path] = None,
    rd_list: Optional[list[RDUnit]] = None,
) -> Dict[str, Dict[str, object]]:
    """Lead-time usability per bid at the run timestamp, from PowerBidList and raart only (no solved model).
    Every bid is reported; earliest_usable_timestamp is the run timestamp when the bid is usable at it, blank when its
    lead time blocks it.
    phase = the kind of the bid's RA in rd_list (blank when unknown).
    {} when bid_list is empty (header-only CSV still written)."""
    if not bid_list:
        if output_path is not None:
            write_results_lead_times_csv({}, output_path, timestamp, ac_verification_status, ac_verification_termination_reason)
        return {}

    radp = _compute_radp(raart, timestamp)

    phase_by_ra_mrid = {str(rd.mrid): rd.kind.value for rd in (rd_list or [])}
    lead_time_results: Dict[str, Dict[str, object]] = {}
    for bid in bid_list:
        bid_mrid = str(bid.mrid)
        earliest_usable = _compute_earliest_usable_timestamp(radp, float(bid.lead_time_h), timestamp)
        lead_time_results[bid_mrid] = {
            "ra_mrid": str(bid.ra_mrid),
            "bid_lead_time_hours": float(bid.lead_time_h),
            "radp": radp,
            "earliest_usable_timestamp": earliest_usable,
            "phase": phase_by_ra_mrid.get(str(bid.ra_mrid), ""),
        }

    log.info(
        "Lead time results extracted: %d bid(s), %d with bid_lead_time_hours > 0",
        len(lead_time_results),
        sum(1 for b in bid_list if float(b.lead_time_h) > 0),
    )

    if output_path is not None:
        write_results_lead_times_csv(lead_time_results, output_path, timestamp, ac_verification_status, ac_verification_termination_reason)

    return lead_time_results


def write_results_summary_csv(
    summary: Dict[str, object],
    output_path: Path,
) -> None:
    """results_summary.csv, columns [metric, value], one row per metric."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in summary.items():
            writer.writerow([metric, value])

    log.info("Written results summary CSV: %s (%d metrics)", output_path, len(summary))


def extract_hvdc_results(model, penalty_reference_pra: float) -> dict[str, dict[str, object]]:
    """Preventive HVDC rows of a solved (or described) model, {unit RA mRID: row}, against the ORIGINAL
    network: the set point of the operating point the model was built on plus this solve's change is the
    result; the change and the penalty (factor x reference x |change|) are cumulative, as the objective
    charges them. Floats throughout (docs/specs/hvdc_ra_specs.md H10.2). {} for a model without an HVDC block."""
    if not hasattr(model, "hvdc_change"):
        return {}
    import pyomo.environ as pyo

    rows: dict[str, dict[str, object]] = {}
    for unit_mrid in sorted(model.hvdc_unit_ids):
        info = model.hvdc_info[unit_mrid]
        last_change = float(pyo.value(model.hvdc_change[unit_mrid]))
        result = info["p_current_mw"] + last_change
        change = result - info["p_original_mw"]
        rows[unit_mrid] = {
            "unit_mrid": unit_mrid,
            "unit_name": info["name"],
            "ei_mrid": info["ei_mrid"],
            "group_mrid": info["group_mrid"],
            "tso": info["tso"],
            "original_p_mw": float(info["p_original_mw"]),
            "result_p_mw": float(result),
            "change_mw": float(change),
            "last_solve_start_p_mw": float(info["p_current_mw"]),
            "last_solve_change_mw": last_change,
            "pmin_op_mw": info["pmin_op_mw"],
            "pmax_op_mw": info["pmax_op_mw"],
            "penalty_factor": info["penalty_factor"],
            "penalty_reference_eur_per_mw": float(penalty_reference_pra),
            "penalty_cost_eur": info["penalty_factor"] * float(penalty_reference_pra) * abs(change),
        }
    return rows


def extract_curative_hvdc_results(model, penalty_reference_cra: float) -> list[dict[str, object]]:
    """Curative HVDC rows of a solved model, one per (unit, contingency) pair, sorted: the curative change
    (up - down, MW) of the pair, the set point after the preventive actions of the model (the curative change is
    never applied), the unit's maximum over its contingencies and its penalty (factor x reference x maximum, the
    term of the objective). Floats (docs/specs/hvdc_ra_specs.md H10.2). [] for a model without curative HVDC pairs."""
    if not hasattr(model, "hvdc_cur_up"):
        return []
    import pyomo.environ as pyo

    preventive_change: dict[str, float] = {}
    if hasattr(model, "hvdc_change"):
        for unit_mrid in model.hvdc_unit_ids:
            ei = model.hvdc_info[unit_mrid]["ei_mrid"]
            preventive_change[ei] = preventive_change.get(ei, 0.0) + float(pyo.value(model.hvdc_change[unit_mrid]))
    rows = []
    for unit_mrid, contingency_mrid in sorted(model.hvdc_cur_pairs):
        info = model.hvdc_cur_info[unit_mrid]
        up = float(pyo.value(model.hvdc_cur_up[unit_mrid, contingency_mrid]))
        down = float(pyo.value(model.hvdc_cur_down[unit_mrid, contingency_mrid]))
        # the maximum of the unit's actual changes (the model variable is only lower-bounded: free at factor 0)
        maximum = max(
            0.0, *(float(pyo.value(model.hvdc_cur_up[unit_mrid, c])) + float(pyo.value(model.hvdc_cur_down[unit_mrid, c]))
            for u, c in model.hvdc_cur_pairs if u == unit_mrid)
        )
        before = info["p_current_mw"] + preventive_change.get(info["ei_mrid"], 0.0)
        rows.append({
            "unit_mrid": unit_mrid, "unit_name": info["name"], "ei_mrid": info["ei_mrid"], "group_mrid": info["group_mrid"],
            "tso": info["tso"], "contingency_mrid": contingency_mrid,
            "p_after_preventive_mw": before, "curative_change_mw": up - down + 0.0, "p_after_curative_mw": before + up - down,
            "pmin_op_mw": info["pmin_op_mw"], "pmax_op_mw": info["pmax_op_mw"],
            "max_curative_change_mw": maximum, "penalty_factor": info["penalty_factor"],
            "penalty_reference_eur_per_mw": float(penalty_reference_cra),
            "penalty_cost_eur": info["penalty_factor"] * float(penalty_reference_cra) * maximum,
        })
    return rows


HVDC_RESULT_COLUMNS = [
    "unit_mrid", "unit_name", "ei_mrid", "group_mrid", "tso", "original_p_mw", "result_p_mw", "change_mw",
    "last_solve_start_p_mw", "last_solve_change_mw", "pmin_op_mw", "pmax_op_mw", "penalty_factor",
    "penalty_reference_eur_per_mw", "penalty_cost_eur",
]
HVDC_CURATIVE_RESULT_COLUMNS = [
    "unit_mrid", "unit_name", "ei_mrid", "group_mrid", "tso", "contingency_mrid", "p_after_preventive_mw",
    "curative_change_mw", "p_after_curative_mw", "pmin_op_mw", "pmax_op_mw", "max_curative_change_mw",
    "penalty_factor", "penalty_reference_eur_per_mw", "penalty_cost_eur",
]
HVDC_AVAILABILITY_COLUMNS = [
    "ra_mrid", "tso", "group_mrid", "ei_mrid", "scope", "branch_mrid", "contingency_mrid", "reaction_time_minutes",
    "tatl_duration_seconds", "available", "reason",
]


def _write_hvdc_rows(rows: list[dict], columns: list[str], output_path: Path, extra: list, label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(columns + (["timestamp", "ac_verification_status", "ac_verification_termination_reason"] if extra else []))
        for row in rows:
            writer.writerow([row[c] for c in columns] + extra)
    log.info("Written %s CSV: %s (%d rows)", label, output_path, len(rows))


def write_hvdc_results_csv(
    rows: dict, output_path: Path, timestamp: str, ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_hvdc.csv: the preventive HVDC set points (floats) and changes per unit (docs/specs/hvdc_ra_specs.md H10)."""
    _write_hvdc_rows(
        list(rows.values()), HVDC_RESULT_COLUMNS, output_path,
        [timestamp, ac_verification_status.value, ac_verification_termination_reason], "HVDC results",
    )


def write_curative_hvdc_results_csv(
    rows: list[dict], output_path: Path, timestamp: str, ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
) -> None:
    """results_curative_hvdc.csv: one row per (unit, contingency); header-only without curative HVDC pairs."""
    _write_hvdc_rows(
        rows, HVDC_CURATIVE_RESULT_COLUMNS, output_path,
        [timestamp, ac_verification_status.value, ac_verification_termination_reason], "curative HVDC results",
    )


def write_hvdc_cra_availability_csv(rows: list[dict], output_path: Path) -> None:
    """hvdc_cra_availability.csv (like rd_cra_availability.csv): per (curative unit, branch) the reaction-time rule
    (scope "branch") and per (unit, contingency) the gate (scope "contingency"), with the reason when unavailable:
    reaction time, outage element, zero sensitivity or group atomicity (the partner's reason)."""
    _write_hvdc_rows(rows, HVDC_AVAILABILITY_COLUMNS, output_path, [], "HVDC CRA availability")


def extract_results_summary(
    milp_model,
    objective_value: Optional[float],
    pst_list: Optional[list[PST]] = None,
    penalty_reference_pst_pra: float = 0.0,
    penalty_reference_pst_pra_per_tso: Optional[dict[str, float]] = None,
    rd_results: Optional[Dict[str, Dict[str, object]]] = None,
    output_path: Optional[Path] = None,
    curative_rd_results: Optional[list[dict]] = None,
    ac_verification_status: Optional[AcVerificationStatus] = None,
    ac_verification_termination_reason: Optional[str] = None,
    ac_iteration_count: Optional[int] = None,
    max_stale_reused_sensitivity_contingencies: Optional[int] = None,
    completeness_limit_reached: Optional[bool] = None,
    calculation_mode: Optional[str] = None,
    ra_types: Optional[list[str]] = None,
    completeness_iteration_count: Optional[int] = None,
    final_check: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Combined PST + RD summary metrics. Every metric is always present (0.0 for an inactive
    RA type). Cost metrics: preventive PST (penalty_factor x reference x (up + down), from the
    model), curative PST, preventive RD (from rd_results), curative RD (curative_rd_results),
    RA-count penalty, the five slack penalties and the start-up / shut-down / bid activation
    costs (model.objective_penalty_term_expressions);
    objective_residual_eur = objective - all of them (~0 when every term is accounted for).
    Optional rows appended at the end: ac_iteration_count, completeness_iteration_limit_reached,
    ac_verification status/reason, calculation_mode, ra_types ("pst+rd"), completeness_iterations and,
    in ac_single_pass, the final-check overload metrics (final_check, see single_pass.py).
    objective_value None: objective and residual are blank.
    """
    rd_results = rd_results or {}

    sum_rd_volume_upward_mw = sum(max(r["rd_volume_net_mw"], 0.0) for r in rd_results.values())
    sum_rd_volume_downward_mw = sum(max(-r["rd_volume_net_mw"], 0.0) for r in rd_results.values())
    sum_rd_cost_eur = sum(r["rd_cost_eur"] for r in rd_results.values())
    sum_rd_penalty_cost_eur = sum(r["penalty_cost_eur"] for r in rd_results.values())

    sum_pst_penalty_cost_eur = 0.0
    if hasattr(milp_model, "tap_change_upward") and pst_list:
        try:
            import pyomo.environ as pyo
        except ImportError as exc:
            raise ImportError("pyomo is required to extract PST results") from exc

        penalty_reference_pst_pra_per_tso = penalty_reference_pst_pra_per_tso or {}
        counted_pst_mrids: set[str] = set()
        for pst in pst_list:
            if pst.kind != RAKind.PREVENTIVE:
                # a transformer with a preventive and a curative RA is two PST objects with
                # one mrid: only the preventive object is costed
                continue
            pst_mrid = str(pst.mrid)
            if pst_mrid not in getattr(milp_model, "pst_ids", []):
                continue
            if pst_mrid in counted_pst_mrids:
                continue  # two preventive RAs on one transformer: one variable pair, costed once
            counted_pst_mrids.add(pst_mrid)
            # cumulative against the ORIGINAL network, as the objective prices it
            tap_up = float(pyo.value(milp_model.cum_tap_up_term[pst_mrid]))
            tap_down = float(pyo.value(milp_model.cum_tap_down_term[pst_mrid]))
            effective_penalty_reference_pst_pra = float(
                penalty_reference_pst_pra_per_tso.get(pst.tso, penalty_reference_pst_pra)
            )
            sum_pst_penalty_cost_eur += (
                float(pst.penalty_factor) * effective_penalty_reference_pst_pra * (tap_up + tap_down)
            )

    remaining_overload_count = 0
    if hasattr(milp_model, "overload_slack_n0"):
        try:
            import pyomo.environ as pyo
        except ImportError as exc:
            raise ImportError("pyomo is required to extract overload results") from exc
        # overload_slack_n0 exists for the secured branches only (scanned ones have their own slack)
        remaining_overload_count = sum(
            1 for branch_mrid in milp_model.secured_branch_ids
            if float(pyo.value(milp_model.overload_slack_n0[branch_mrid])) > _ZERO_SLACK_AMPERE
        )

    # curative PST penalty (0.0 when none), appended after the existing metrics
    sum_curative_pst_penalty_cost_eur = sum(
        float(r["curative_penalty_cost_eur"])
        for r in extract_curative_pst_results(
            milp_model, pst_list, penalty_reference_pst_pra, penalty_reference_pst_pra_per_tso,
        ).values()
    )

    sum_curative_rd_cost_eur = sum(float(r["curative_cost_eur"]) for r in (curative_rd_results or []))

    penalty_term_values: Dict[str, float] = {}
    if getattr(milp_model, "objective_penalty_term_expressions", None):
        try:
            import pyomo.environ as pyo
        except ImportError as exc:
            raise ImportError("pyomo is required to extract penalty terms") from exc
        penalty_term_values = {
            name: float(pyo.value(expression))
            for name, expression in milp_model.objective_penalty_term_expressions.items()
        }
    sum_ra_count_penalty_eur = (
        penalty_term_values.get("ra_count_timestamp", 0.0)
    )
    slack_penalties = {
        "sum_overload_slack_penalty_n0_eur": penalty_term_values.get("overload_n0", 0.0),
        "sum_overload_slack_penalty_n0_scanned_eur": penalty_term_values.get("overload_n0_scanned", 0.0),
        "sum_overload_slack_penalty_nx_patl_eur": penalty_term_values.get("overload_nx_patl", 0.0),
        "sum_overload_slack_penalty_nx_tatl_eur": penalty_term_values.get("overload_nx_tatl", 0.0),
        "sum_overload_slack_penalty_nx_scanned_eur": penalty_term_values.get("overload_nx_scanned", 0.0),
    }
    # start-up, shut-down and bid activation costs (preventive, from their binaries)
    on_off_costs = {
        "sum_startup_cost_eur": penalty_term_values.get("startup_cost", 0.0),
        "sum_shutdown_cost_eur": penalty_term_values.get("shutdown_cost", 0.0),
        "sum_bid_activation_cost_eur": penalty_term_values.get("bid_activation_cost", 0.0),
    }
    # HVDC penalty costs (docs/specs/hvdc_ra_specs.md H7.5): rows only for a run with hvdc in ra_types
    hvdc_costs: Dict[str, float] = {}
    if ra_types is not None and "hvdc" in ra_types:
        pra = penalty_term_values.get("hvdc_pra_penalty_cost", 0.0) + 0.0      # + 0.0: no "-0.0" in the file
        cra = penalty_term_values.get("hvdc_cra_penalty_cost", 0.0) + 0.0
        hvdc_costs = {
            "sum_hvdc_penalty_cost_eur": pra + cra,
            "sum_pra_penalty_cost_hvdc_eur": pra,
            "sum_cra_penalty_cost_hvdc_eur": cra,
        }
    ct_costs: Dict[str, float] = {}
    ct_cost_names = (
        "ct_pra_penalty_cost", "ct_pra_energy_cost", "ct_bid_activation_cost", "ct_cra_penalty_cost", "ct_cra_reservation_cost",
    )
    ct_cost_eur = sum(penalty_term_values.get(name, 0.0) for name in ct_cost_names)
    # countertrade rows (docs/specs/countertrade_ra_specs.md C11.2): only for a run with countertrade in ra_types
    if ra_types is not None and "countertrade" in ra_types:
        ct_bid_ids = getattr(milp_model, "ct_bid_ids", frozenset())
        ct_volume_up = ct_volume_down = 0.0
        if ct_bid_ids and hasattr(milp_model, "rd_cumulative_up_term"):
            import pyomo.environ as pyo
            ct_volume_up = sum(float(pyo.value(milp_model.rd_cumulative_up_term[b])) for b in ct_bid_ids)
            ct_volume_down = sum(float(pyo.value(milp_model.rd_cumulative_down_term[b])) for b in ct_bid_ids)
        ct_costs = {
            "sum_ct_volume_upward_mw": ct_volume_up + 0.0,           # + 0.0: no "-0.0" in the file
            "sum_ct_volume_downward_mw": ct_volume_down + 0.0,
            "sum_pra_penalty_cost_ct_eur": penalty_term_values.get("ct_pra_penalty_cost", 0.0) + 0.0,
            "sum_cra_penalty_cost_ct_eur": penalty_term_values.get("ct_cra_penalty_cost", 0.0) + 0.0,
            "sum_cra_reservation_cost_ct_eur": penalty_term_values.get("ct_cra_reservation_cost", 0.0) + 0.0,
            "sum_ct_cost_eur": ct_cost_eur + 0.0,
        }
    # sum_rd_penalty_cost_eur is a PART of sum_rd_cost_eur (penalty + price) and
    # is deliberately not added a second time; sum_hvdc_penalty_cost_eur is the total of the two HVDC rows.
    accounted_cost_eur = (
        sum_pst_penalty_cost_eur + sum_curative_pst_penalty_cost_eur + sum_rd_cost_eur
        + sum_curative_rd_cost_eur + sum_ra_count_penalty_eur + sum(slack_penalties.values())
        + sum(on_off_costs.values()) + hvdc_costs.get("sum_hvdc_penalty_cost_eur", 0.0) + ct_cost_eur
    )

    summary = {
        "sum_rd_volume_upward_mw": sum_rd_volume_upward_mw,
        "sum_rd_volume_downward_mw": sum_rd_volume_downward_mw,
        "sum_rd_cost_eur": sum_rd_cost_eur,
        "sum_rd_penalty_cost_eur": sum_rd_penalty_cost_eur,
        "sum_pst_penalty_cost_eur": sum_pst_penalty_cost_eur,
        "objective_value": "" if objective_value is None else float(objective_value),
        "remaining_overload_count": remaining_overload_count,
        "sum_curative_pst_penalty_cost_eur": sum_curative_pst_penalty_cost_eur,
        "sum_curative_rd_cost_eur": sum_curative_rd_cost_eur,
        "sum_ra_count_penalty_eur": sum_ra_count_penalty_eur,
        **slack_penalties,
        **on_off_costs,
        **hvdc_costs,
        **ct_costs,
        "objective_residual_eur": "" if objective_value is None else float(objective_value) - accounted_cost_eur,
    }
    if ac_verification_status is not None:
        summary["ac_verification_status"] = ac_verification_status.value
        summary["ac_verification_termination_reason"] = ac_verification_termination_reason or ""
    if ac_iteration_count is not None:
        summary["ac_iteration_count"] = int(ac_iteration_count)
    if max_stale_reused_sensitivity_contingencies is not None:
        # the largest number of contingencies, in any one solve, whose MILP sensitivities were the
        # last valid ones instead of the (invalid) fresh ones
        summary["max_stale_reused_sensitivity_contingencies"] = int(max_stale_reused_sensitivity_contingencies)
    if completeness_limit_reached is not None:
        summary["completeness_iteration_limit_reached"] = bool(completeness_limit_reached)
    if completeness_iteration_count is not None:
        summary["completeness_iterations"] = int(completeness_iteration_count)
    if calculation_mode is not None:
        summary["calculation_mode"] = calculation_mode
    if ra_types is not None:
        summary["ra_types"] = "+".join(ra_types)
    if final_check:
        summary.update(final_check)

    log.info("Results summary extracted: %s", summary)

    if output_path is not None:
        write_results_summary_csv(summary, output_path)

    return summary


def write_results_generating_units_csv(
    gu_results: Dict[str, Dict[str, object]],
    output_path: Path,
    timestamp: str,
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
    ct_column: bool = False,
) -> None:
    """results_generating_units.csv: [gu_mrid, gu_name, tso, initial_production_mw,
    rd_volume_net_mw, result_production_mw, rd_cost_eur, rd_penalty_cost_eur], timestamp and
    the ac_verification columns. ct_column (a run with a usable countertrade RA) adds ct_volume_net_mw
    after the RD columns and writes the countertrade-only units (row key "ct_only") after the RD units."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ct_header = ["ct_volume_net_mw"] if ct_column else []
    rows = sorted(gu_results.items(), key=lambda item: (bool(item[1].get("ct_only")), item[0])) if ct_column \
        else sorted(gu_results.items())
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow([
            "gu_mrid", "gu_name", "tso", "initial_production_mw",
            "rd_volume_net_mw", "result_production_mw",
            "rd_cost_eur", "rd_penalty_cost_eur", *ct_header,
            "timestamp", "ac_verification_status", "ac_verification_termination_reason",
        ])
        for gu_mrid, row in rows:
            writer.writerow([
                gu_mrid, row["gu_name"], row["tso"], row["initial_production_mw"],
                row["rd_volume_net_mw"], row["result_production_mw"],
                row["rd_cost_eur"], row["rd_penalty_cost_eur"], *([row["ct_volume_net_mw"]] if ct_column else []),
                timestamp, ac_verification_status.value, ac_verification_termination_reason,
            ])

    log.info("Written generating unit results CSV: %s (%d rows)", output_path, len(gu_results))


def extract_results_generating_units(
    milp_model,
    rd_list: list[RDUnit],
    ac_verification_status: AcVerificationStatus,
    ac_verification_termination_reason: str,
    generator_data: Optional[dict[str, dict]] = None,
    glsk: Optional[dict[str, dict[str, float]]] = None,
    rd_results: Optional[Dict[str, Dict[str, object]]] = None,
    timestamp: str = "",
    output_path: Optional[Path] = None,
    ct_bid_participation: Optional[dict[str, dict[str, float]]] = None,
    ct_units: Optional[dict[str, tuple[str, float]]] = None,
) -> Dict[str, Dict[str, object]]:
    """Per generating unit of the RD block (model.gu_ids): initial_production_mw (the baseline
    the model was built on, model.rd_baseline_target_p_mw or generator_data), rd_volume_net_mw
    = sum over bids of glsk factor x (up - down), result = initial + net. Bid costs are split
    over the units by GLSK factor. For an RA with its own units the net volumes add up to its
    bids' net volume. {} without rd_net_injection_g (header-only CSV).
    ct_bid_participation ({countertrade bid: {unit: weight}}, a run with a usable countertrade RA): the units
    of an RD glsk; the RD columns hold the RD part only, ct_volume_net_mw the countertrade part (both
    cumulative against the original network) and result_production_mw = initial + RD + countertrade
    (DECISIONS 76, 79). ct_units ({unit: (TSO, original MW)} of every countertrade unit): the units that are in no
    RD glsk get a row after the RD units, RD columns 0, production = initial + countertrade (DECISIONS 80).
    """
    ct_column = ct_bid_participation is not None
    if not hasattr(milp_model, "rd_net_injection_g"):
        if output_path is not None:
            write_results_generating_units_csv(
                {}, output_path, timestamp, ac_verification_status, ac_verification_termination_reason,
                ct_column=ct_column,
            )
        return {}

    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to extract generating unit results") from exc

    glsk = glsk or {}
    rd_results = rd_results or {}
    gu_ids_in_model = [str(g) for g in getattr(milp_model, "gu_ids", [])]
    # a model with countertrade bids: this file is the RD part, over the units of an RD glsk. The baseline of a
    # later build also holds the countertrade moves of earlier iterations, so the RD part is summed from the
    # cumulative RD bid volumes (glsk factor x net volume against the original network), not baseline + last solve
    rd_part_mw: Optional[dict[str, float]] = None
    ct_part_mw: dict[str, float] = {}
    if ct_column or hasattr(milp_model, "rd_only_net_injection_g"):
        gu_ids_in_model = [g for g in gu_ids_in_model if any(g in participation for participation in glsk.values())]
        rd_part_mw = {gu: 0.0 for gu in gu_ids_in_model}
        for bid_result in rd_results.values():
            for gu_mrid, factor in glsk.get(bid_result["ra_mrid"], {}).items():
                if gu_mrid in rd_part_mw:
                    rd_part_mw[gu_mrid] += float(factor) * float(bid_result["rd_volume_net_mw"])
        # the countertrade part: weight x net volume of every preventive countertrade bid of the model
        for bid_mrid in sorted(str(b) for b in getattr(milp_model, "ct_bid_ids", frozenset())):
            signed = _cumulative_signed_rd_volume_mw(milp_model, bid_mrid)
            for gu_mrid, factor in (ct_bid_participation or {}).get(bid_mrid, {}).items():
                ct_part_mw[gu_mrid] = ct_part_mw.get(gu_mrid, 0.0) + float(factor) * signed
    ct_only_units = sorted(gu for gu in (ct_units or {}) if rd_part_mw is not None and gu not in rd_part_mw) if ct_column else []
    baseline_target_p = getattr(milp_model, "rd_baseline_target_p_mw", {})
    generator_data = generator_data or {}

    gu_to_tso: dict[str, str] = {}
    for rd in rd_list:
        gu_to_tso.setdefault(str(rd.gu_mrid), rd.tso)

    gu_cost_accum: dict[str, float] = {gu: 0.0 for gu in gu_ids_in_model}
    gu_penalty_cost_accum: dict[str, float] = {gu: 0.0 for gu in gu_ids_in_model}
    for bid_mrid, bid_result in rd_results.items():
        ra_mrid = bid_result["ra_mrid"]
        participation = glsk.get(ra_mrid, {})
        for gu_mrid, factor in participation.items():
            if gu_mrid not in gu_cost_accum:
                continue
            gu_cost_accum[gu_mrid] += bid_result["rd_cost_eur"] * factor
            gu_penalty_cost_accum[gu_mrid] += bid_result["penalty_cost_eur"] * factor

    gu_results: Dict[str, Dict[str, object]] = {}
    for gu_mrid in gu_ids_in_model:
        gu_data = generator_data.get(gu_mrid, {})
        original_production_mw = getattr(milp_model, "rd_original_target_p_mw", {}).get(gu_mrid)
        if original_production_mw is None:
            if generator_data:
                original_production_mw = float(gu_data.get("target_p", gu_data.get("initial_p", 0.0)))
            else:
                original_production_mw = float(baseline_target_p.get(gu_mrid, 0.0))
        initial_production_mw = float(original_production_mw)
        if rd_part_mw is not None:
            rd_volume_net_mw = rd_part_mw[gu_mrid]
            result_production_mw = initial_production_mw + rd_volume_net_mw + ct_part_mw.get(gu_mrid, 0.0)
        else:
            # the model was built on the CURRENT baseline (== original on a first solve)
            build_baseline_mw = float(baseline_target_p.get(gu_mrid, initial_production_mw))
            last_solve_net_mw = float(pyo.value(milp_model.rd_net_injection_g[gu_mrid]))
            result_production_mw = build_baseline_mw + last_solve_net_mw
            rd_volume_net_mw = result_production_mw - initial_production_mw  # cumulative

        gu_results[gu_mrid] = {
            "gu_name": gu_data.get("name", gu_mrid),
            "tso": gu_to_tso.get(gu_mrid, ""),
            "initial_production_mw": initial_production_mw,
            "rd_volume_net_mw": rd_volume_net_mw,
            "result_production_mw": result_production_mw,
            "rd_cost_eur": gu_cost_accum.get(gu_mrid, 0.0),
            "rd_penalty_cost_eur": gu_penalty_cost_accum.get(gu_mrid, 0.0),
        }
        if ct_column:
            gu_results[gu_mrid]["ct_volume_net_mw"] = ct_part_mw.get(gu_mrid, 0.0) + 0.0    # + 0.0: no "-0.0"

    # countertrade units in no RD glsk: RD columns 0, production = initial + countertrade
    for gu_mrid in ct_only_units:
        tso, original_mw = ct_units[gu_mrid]
        initial_production_mw = float(getattr(milp_model, "rd_original_target_p_mw", {}).get(gu_mrid, original_mw))
        ct_mw = ct_part_mw.get(gu_mrid, 0.0)
        gu_results[gu_mrid] = {
            "gu_name": generator_data.get(gu_mrid, {}).get("name", gu_mrid),
            "tso": tso,
            "initial_production_mw": initial_production_mw,
            "rd_volume_net_mw": 0.0,
            "result_production_mw": initial_production_mw + ct_mw,
            "rd_cost_eur": 0.0,
            "rd_penalty_cost_eur": 0.0,
            "ct_volume_net_mw": ct_mw + 0.0,
            "ct_only": True,
        }

    log.info("Generating unit results extracted: %d unit(s)", len(gu_results))

    if output_path is not None:
        write_results_generating_units_csv(
            gu_results, output_path, timestamp, ac_verification_status, ac_verification_termination_reason,
            ct_column=ct_column,
        )

    return gu_results


# ---------------------------------------------------------------------------
# lf_post_n0.csv (one block per AC iteration)
# ---------------------------------------------------------------------------

def _lf_post_branch_row_values(details: dict, branch_mrid: str, lf_before, lf_after) -> list:
    """One branch's [branch_mrid .. patl_ka] values for lf_post_n0.csv.
    Used by write_lf_post_iterations_csv; currents and loadings are
    side-aware."""
    patl_ka = float(details["patl_ka"])
    side = int(details.get("branch_side", 1))
    i_before_a = select_by_side(lf_before.get(branch_mrid, {}), side, branch_mrid) or 0.0
    i_after_a = select_by_side(lf_after.get(branch_mrid, {}), side, branch_mrid) or 0.0
    i_before = float(i_before_a) / 1000.0
    i_after = float(i_after_a) / 1000.0
    loading_before = (i_before / patl_ka * 100.0) if patl_ka > 0 else 0.0
    loading_after = (i_after / patl_ka * 100.0) if patl_ka > 0 else 0.0
    return [
        branch_mrid,
        details["branch_name"],
        details["tso"],
        i_before,
        i_after,
        loading_before,
        loading_after,
        patl_ka,
    ]


_LF_POST_COLUMNS = [
    "branch_mrid",
    "branch_name",
    "tso",
    "i_ka_before",
    "i_ka_after",
    "loading_pct_before",
    "loading_pct_after",
    "patl_ka",
]


def write_lf_post_iterations_csv(
    output_path: Path,
    branch_details: dict[str, dict[str, object]],
    iteration_records: list[dict],
    timestamp: str,
) -> None:
    """lf_post_n0.csv with every AC verification iteration. iteration_records: one dict per
    iteration with ac_iteration (1-based), lf_before, lf_after (A per side) and
    solution_applied_to_network. An iteration that ended without applying its solution has
    solution_applied_to_network = false and lf_after = the state that exists afterwards.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(["ac_iteration"] + _LF_POST_COLUMNS + ["solution_applied_to_network", "timestamp"])
        for record in iteration_records:
            for branch_mrid, details in sorted(branch_details.items()):
                writer.writerow(
                    [record["ac_iteration"]]
                    + _lf_post_branch_row_values(details, branch_mrid, record["lf_before"], record["lf_after"])
                    + ["true" if record["solution_applied_to_network"] else "false", timestamp]
                )

    log.info(
        "Written post-RA LF CSV: %s (%d iteration(s), %d rows)",
        output_path, len(iteration_records), len(iteration_records) * len(branch_details),
    )


# ---------------------------------------------------------------------------
# Completeness loop
# ---------------------------------------------------------------------------

def write_completeness_iterations_csv(iteration_log: list[dict], output_path: Path) -> None:
    """results_completeness_iterations.csv — columns:
    [completeness_iteration,
    new_constraints_added, total_nx_constraints]."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(["completeness_iteration", "new_constraints_added", "total_nx_constraints"])
        for row in iteration_log:
            writer.writerow([
                row["completeness_iteration"], row["new_constraints_added"], row["total_nx_constraints"],
            ])

    log.info("Written completeness iterations CSV: %s", output_path)


ITERATION_HISTORY_COLUMNS = [
    "solve_index", "completeness_iteration", "linearisation_iteration", "objective_value",
    "sum_abs_tap_change", "sum_abs_rd_change_mw", "max_abs_tap_change", "max_abs_rd_change_mw",
    "total_overload", "overload_change_ampere", "load_flow_converged",
    "solution_applied", "criterion_1_met", "criterion_3_met", "criterion_4_met", "converged_by",
    "cycle_match_iteration", "cycle_selected_iteration", "converged", "new_constraints_added",
    "invalid_sensitivity_contingencies", "invalid_sensitivity_branches_n0",
    "stale_reused_sensitivity_contingencies", "stale_reused_sensitivity_branches_n0",
    "nx_contingencies_lost_at_refresh",
    "seconds_build_and_solve", "seconds_load_flow", "seconds_n0_sensitivity",
    "seconds_nx_load_flow", "seconds_nx_sensitivity",
]


def write_iteration_history_csv(solve_history: list[dict], output_path: Path) -> None:
    """results_iteration_history.csv — one row per MILP solve of the whole nested
    run (completeness iteration x linearisation iteration), with the value of every
    convergence criterion (total_overload = the model's own slack total, diagnostic
    only; overload_change_ampere = criterion 4, from the AC state), which criterion
    declared convergence, the iteration a cyclic repetition matched, and the
    per-solve runtimes of the load flow, the sensitivities and the solve."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(ITERATION_HISTORY_COLUMNS)
        for record in solve_history:
            writer.writerow([record.get(column, "") for column in ITERATION_HISTORY_COLUMNS])
    log.info("Written iteration history CSV: %s (%d rows)", output_path, len(solve_history))
