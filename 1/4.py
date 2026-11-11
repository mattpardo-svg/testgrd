"""
objective_report.py - the INFO block after a solve: the value of each objective term, the PST moves and the PST groups whose
members started on different taps. Log only.
"""

import logging

log = logging.getLogger(__name__)

LARGEST_MOVES_SHOWN = 5
MOVE_TOLERANCE_TAPS = 1e-6


def _tap(value: float) -> str:
    return str(int(round(value))) if abs(value - round(value)) < 1e-6 else f"{value:.2f}"


def _term_values(milp_model) -> dict[str, float]:
    import pyomo.environ as pyo
    return {name: float(pyo.value(expression)) for name, expression in getattr(milp_model, "objective_term_expressions", {}).items()}


def _groups(milp_model, pst_results: dict, group_id_by_mrid: dict[str, str]) -> list[tuple[str, list[str]]]:
    """[(group label, sorted member mRIDs)] from the equality pairs of the model, the members present in pst_results."""
    parent: dict[str, str] = {}

    def find(item):
        while parent.setdefault(item, item) != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for first, second in getattr(milp_model, "pst_group_pairs", []):
        parent[find(str(first))] = find(str(second))
    members: dict[str, list[str]] = {}
    for mrid in list(parent):
        if mrid in pst_results:
            members.setdefault(find(mrid), []).append(mrid)
    groups = []
    for member_list in members.values():
        member_list.sort()
        label = next((group_id_by_mrid[m] for m in member_list if m in group_id_by_mrid), ", ".join(member_list))
        groups.append((label, member_list))
    return sorted(groups)


def log_objective_report(milp_model, pst_results: dict, pst_list) -> None:
    """One INFO block; never raises (a log must not stop a run)."""
    try:
        terms = _term_values(milp_model)
        lines = [f"Objective breakdown (total {sum(terms.values()):,.2f}):"]
        lines += [f"  {name} {value:,.2f}" for name, value in terms.items()]
        if pst_results:
            moves = {
                mrid: row for mrid, row in pst_results.items() if abs(float(row["tap_change_result"])) > MOVE_TOLERANCE_TAPS
            }
            steps = sum(abs(float(row["tap_change_result"])) for row in moves.values())
            lines.append(f"PST moves: {len(moves)} of {len(pst_results)} PST(s) moved, {_tap(steps)} tap step(s) in total")
            largest = sorted(moves.items(), key=lambda item: (-abs(float(item[1]["tap_change_result"])), item[0]))
            if largest:
                lines[-1] += "; largest: " + ", ".join(
                    f"{mrid} {_tap(float(row['current_tap_position']))} -> {_tap(float(row['result_tap_position']))}"
                    for mrid, row in largest[:LARGEST_MOVES_SHOWN]
                )
            group_id_by_mrid = {str(p.mrid): str(p.group_id) for p in pst_list if p.group_id}
            for label, members in _groups(milp_model, pst_results, group_id_by_mrid):
                before = [float(pst_results[m]["current_tap_position"]) for m in members]
                if len(set(before)) > 1:
                    after = [float(pst_results[m]["result_tap_position"]) for m in members]
                    lines.append(
                        f"PST group {label}: taps before [{', '.join(_tap(v) for v in before)}], "
                        f"after [{', '.join(_tap(v) for v in after)}]"
                    )
        log.info("\n".join(lines))
    except Exception:
        log.debug("objective breakdown could not be written", exc_info=True)
