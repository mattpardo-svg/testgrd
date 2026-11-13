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
