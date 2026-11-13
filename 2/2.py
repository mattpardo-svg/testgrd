"""
overload_diagnosis.py - after the final check, one INFO block per overloaded N-X (branch, contingency) pair: the limit,
the AC current, the overload and what the preventive PSTs could do about it. Log only; no result file changes.
"""

import logging

log = logging.getLogger(__name__)

MAX_PAIRS_AT_INFO = 20
_KIND_LABEL = {"nx_patl": "N-X PATL", "nx_tatl": "N-X TATL", "nx_scanned": "N-X scanned"}


def _applied_tap_change(model, pst_mrid: str) -> float:
    """Up - down of the solved model (0 where the variables have no value)."""
    import pyomo.environ as pyo
    up = pyo.value(model.tap_change_upward[pst_mrid], exception=False)
    down = pyo.value(model.tap_change_downward[pst_mrid], exception=False)
    return float(up or 0.0) - float(down or 0.0)


def _pst_effects(model, branch_mrid: str, contingency_mrid: str) -> tuple[float, float]:
    """(largest |A per tap| of one PST, best possible relief in A over all preventive PSTs within their tap ranges
    from the applied taps) on the pair, from the sensitivities the model was built on."""
    baseline = getattr(model, "nx_build_baseline", None) or {}
    sensitivities = (baseline.get("sensitivity_pst_nx") or {}).get(contingency_mrid, {})
    degrees = getattr(model, "pst_degrees_per_tap_by_mrid", {})
    largest, relief = 0.0, 0.0
    for pst_mrid, (tap_low, tap_high) in getattr(model, "pst_tap_range_by_mrid", {}).items():
        if pst_mrid not in degrees:
            continue
        per_tap = float(sensitivities.get((branch_mrid, pst_mrid), 0.0)) * degrees[pst_mrid]
        current_tap = model.pst_start_tap_by_mrid[pst_mrid] + _applied_tap_change(model, pst_mrid)
        largest = max(largest, abs(per_tap))
        # the flow changes by per_tap x (move); the relief is the largest decrease the range allows
        relief += max(-per_tap * (tap_low - current_tap), -per_tap * (tap_high - current_tap), 0.0)
    return largest, relief


def _value(variable) -> float:
    import pyomo.environ as pyo
    return float(pyo.value(variable, exception=False) or 0.0)


def _bid_room(model, bid_mrid: str, participation: dict) -> tuple[float, float]:
    """(MW the bid can still move up, MW it can still move down) from the applied volumes: its volume bounds and the
    Pmax / Pmin of the units of its participation (a unit that can be shut down may fall to 0)."""
    net = _value(model.rd_volume_upward[bid_mrid]) - _value(model.rd_volume_downward[bid_mrid])
    up = (model.rd_volume_upward[bid_mrid].ub or 0.0) - net
    down = (model.rd_volume_downward[bid_mrid].ub or 0.0) + net
    status = getattr(model, "rd_unit_status", {})
    for gu_mrid, factor in participation.items():
        if gu_mrid not in status or float(factor) <= 0.0:
            continue
        unit = status[gu_mrid]
        current = unit.original_p_mw + _value(model.rd_net_injection_g[gu_mrid]) if gu_mrid in model.rd_net_injection_g else unit.original_p_mw
        lowest = 0.0 if unit.unit_class == "stoppable" else unit.pmin_mw
        up = min(up, max(unit.pmax_mw - current, 0.0) / float(factor))
        down = min(down, max(current - lowest, 0.0) / float(factor))
    return max(up, 0.0), max(down, 0.0)


def _bid_effects(model, branch_mrid: str, contingency_mrid: str, countertrade: bool) -> tuple[float, float]:
    """(largest |A per 100 MW| of one preventive bid, best possible relief in A over all of them within their volumes and
    unit limits) on the pair, redispatch bids or countertrade bids; an upper bound (the balance between bids is not
    modelled). A per MW of a bid: the weighted sum of the unit sensitivities (the variable of the RA for countertrade)."""
    from pre_milp import ct_sensitivity
    baseline = getattr(model, "nx_build_baseline", None) or {}
    sensitivities = (baseline.get("sensitivity_rd_nx") or {}).get(contingency_mrid, {})
    ct_variable = getattr(model, "ct_bid_variable", {})
    largest = relief = 0.0
    for bid_mrid, (direction, participation) in getattr(model, "rd_bid_diagnosis", {}).items():
        if (bid_mrid in getattr(model, "ct_bid_ids", ())) != countertrade:
            continue
        if bid_mrid in ct_variable:
            per_mw = ct_sensitivity.coefficient(sensitivities, branch_mrid, ct_variable[bid_mrid][0], ct_variable[bid_mrid][1])
        else:
            per_mw = sum(float(factor) * float(sensitivities.get((branch_mrid, gu), 0.0)) for gu, factor in participation.items())
        largest = max(largest, abs(per_mw) * 100.0)
        up, down = _bid_room(model, bid_mrid, participation)
        relief += max(-per_mw * up, per_mw * down, 0.0)
    return largest, relief


def _block(model, key: tuple, detail: dict) -> str:
    kind, branch_mrid, contingency_mrid = key
    largest, relief = _pst_effects(model, branch_mrid, contingency_mrid)
    pruned = bool(getattr(model, "nx_pst_sensitivity_pruned", False))
    before = lambda value: "pruned" if pruned else f"{value:.1f} A"      # noqa: E731 (the raw table is gone once pruned)
    effect = float(detail.get("curative_effect", 0.0))
    lines = [
        f"overload diagnosis {_KIND_LABEL[kind]}: branch {branch_mrid}, contingency {contingency_mrid}: "
        f"limit {detail['limit']:.1f} A, AC current {detail['current']:.1f} A, overload {detail['overload']:.1f} A"
        + (f" (with the curative effect {effect:.1f} A)" if effect else ""),
        f"  preventive PSTs: largest effect per tap {largest:.1f} A (before the threshold {before(largest)}), "
        f"best possible relief {relief:.1f} A (before the threshold {before(relief)})",
    ]
    if relief < detail["overload"]:
        lines.append(f"  no PST can relieve this overload by more than {relief:.1f} A")
    reliefs = {"PST": relief}
    rd_pruned = bool(getattr(model, "nx_rd_sensitivity_pruned", False))
    rd_before = lambda value: "pruned" if rd_pruned else f"{value:.1f} A"        # noqa: E731
    for label, countertrade in (("redispatch", False), ("countertrade", True)):
        if not hasattr(model, "rd_bid_diagnosis"):
            break
        largest_per_100, bid_relief = _bid_effects(model, branch_mrid, contingency_mrid, countertrade)
        reliefs[label] = bid_relief
        lines.append(
            f"  preventive {label}: largest effect per 100 MW {largest_per_100:.1f} A (before the threshold "
            f"{rd_before(largest_per_100)}), best possible relief {bid_relief:.1f} A (before the threshold {rd_before(bid_relief)})")
    best = max(reliefs, key=lambda name: reliefs[name])                         # the first of equal reliefs (PST first)
    lines.append(
        f"  best RA type: {best}, relief {reliefs[best]:.1f} A" if reliefs[best] > 0.0
        else "  best RA type: none, no preventive RA type can relieve this pair")
    return "\n".join(lines)


def log_overload_diagnosis(model, details: dict, max_pairs_at_info: int = MAX_PAIRS_AT_INFO) -> None:
    """details: {(kind, branch, contingency): {"limit", "current", "curative_effect", "overload"}} of the overloaded N-X
    pairs. The largest overloads first; the first max_pairs_at_info at INFO, the rest at DEBUG. Never raises."""
    try:
        ordered = sorted(details.items(), key=lambda item: (-item[1]["overload"], item[0]))
        for index, (key, detail) in enumerate(ordered):
            log.log(logging.INFO if index < max_pairs_at_info else logging.DEBUG, "%s", _block(model, key, detail))
    except Exception:
        log.debug("overload diagnosis could not be written", exc_info=True)
