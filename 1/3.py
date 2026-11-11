"""milp_solver.py

Solves the model built by milp_builder.py (N-0, and N-X when enabled) with HiGHS and
returns the PST rows and the N-0 overload slack. A non-optimal termination raises
MilpNotOptimalError; the result files are written by post_milp, not here.
"""

import logging
from typing import Dict, Optional

import run_profile
from data_model import PST, RAKind
from milp.objective_report import log_objective_report

log = logging.getLogger(__name__)

# HiGHS options of every solve, set once per run by run_pipeline from the performance config
# section; None passes nothing (no time limit, the solver's default gap).
SOLVER_TIME_LIMIT_SECONDS: Optional[float] = None
SOLVER_MIP_REL_GAP: Optional[float] = None


class MilpNotOptimalError(RuntimeError):
    """The solver ended with a termination condition other than optimal (infeasible or other)."""


def _get_pst_dict(pst_list: list[PST]) -> Dict[str, PST]:
    return {str(pst.mrid): pst for pst in pst_list}


def _get_effective_penalty_reference(
    pst: PST,
    penalty_reference_pst_pra: float,
    penalty_reference_pst_pra_per_tso: Optional[dict[str, float]] = None,
) -> float:
    override = penalty_reference_pst_pra_per_tso or {}
    return float(override.get(pst.tso, penalty_reference_pst_pra))


def _is_optimal_termination(termination, pyo_module) -> bool:
    """True when the termination condition is optimal: enum comparison, or the exact string
    "optimal" for test doubles. Never a substring check ("suboptimal" is not optimal)."""
    try:
        if termination == pyo_module.TerminationCondition.optimal:
            return True
    except Exception:
        pass
    return str(termination).strip().lower() == "optimal"


def solve_n0_milp_model(
    milp_model,
    pst_list: list[PST],
    timestamp: str,
    penalty_reference_pst_pra: float,
    penalty_reference_pst_pra_per_tso: Optional[dict[str, float]] = None,
    solver_name: str = "appsi_highs",
    solver_io: Optional[str] = None,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, float]]:
    """Solve the MILP and return (PST rows, N-0 overload slack per secured branch).

    The default solver is appsi_highs (HiGHS through the APPSI interface, which rejects
    solver_io); solver_io is passed to SolverFactory only when set.
    """
    run_profile.mark("MILP solve")
    try:
        import pyomo.environ as pyo
    except ImportError as exc:
        raise ImportError("pyomo is required to solve the MILP model") from exc

    try:
        if solver_io is not None:
            solver = pyo.SolverFactory(solver_name, solver_io=solver_io)
        else:
            solver = pyo.SolverFactory(solver_name)
    except Exception as exc:
        raise RuntimeError(f"Failed to create solver factory for {solver_name}") from exc

    # load_solutions=False: an infeasible model must reach the termination check below
    # instead of failing inside the solver interface while loading a missing solution.
    # A time limit or gap is passed only when set; a stop at the time limit is not optimal.
    highs_options = {
        name: float(value) for name, value in (
            ("time_limit", SOLVER_TIME_LIMIT_SECONDS), ("mip_rel_gap", SOLVER_MIP_REL_GAP),
        ) if value is not None
    }
    if highs_options:
        results = solver.solve(milp_model, tee=False, load_solutions=False, options=highs_options)
    else:
        results = solver.solve(milp_model, tee=False, load_solutions=False)
    solver_info = getattr(results, "solver", None)
    termination = getattr(solver_info, "termination_condition", None)

    if not _is_optimal_termination(termination, pyo):
        log.error("MILP solve did not terminate optimally: %s", termination)
        raise MilpNotOptimalError(f"MILP solve not optimal (termination={termination})")
    milp_model.solutions.load_from(results)

    # rows of PREVENTIVE PSTs only: a transformer with a preventive and a curative remedial
    # action appears twice in pst_list, and only the preventive one has tap variables
    pst_by_mrid = _get_pst_dict([pst for pst in pst_list if pst.kind == RAKind.PREVENTIVE])

    # only PSTs in the model's pst_ids (the builder leaves out PSTs without valid bounds);
    # a model without pst_ids (test double) falls back to every preventive PST
    if hasattr(milp_model, "pst_ids"):
        model_pst_ids = [str(m) for m in milp_model.pst_ids]
    elif hasattr(milp_model, "tap_change_upward"):
        model_pst_ids = list(pst_by_mrid.keys())
    else:
        # curative-only PSTs: no preventive PST block on the model
        model_pst_ids = []

    pst_results: Dict[str, Dict[str, object]] = {}
    for pst_mrid in model_pst_ids:
        pst = pst_by_mrid.get(pst_mrid)
        if pst is None:
            log.warning(
                "MILP model references PST %s not found in PSTList — skipped",
                pst_mrid,
            )
            continue
        current_tap_position = int(getattr(pst, "r0", 0))
        tap_up = float(pyo.value(milp_model.tap_change_upward[pst_mrid]))
        tap_down = float(pyo.value(milp_model.tap_change_downward[pst_mrid]))
        tap_change_result = tap_up - tap_down
        result_tap_position = current_tap_position + tap_change_result

        penalty_reference = _get_effective_penalty_reference(
            pst,
            penalty_reference_pst_pra,
            penalty_reference_pst_pra_per_tso,
        )
        effective_penalty_cost_per_tap = float(pst.penalty_factor) * penalty_reference
        penalty_cost_eur = abs(tap_change_result) * effective_penalty_cost_per_tap

        pst_results[pst_mrid] = {
            "tap_change_result": tap_change_result,
            "current_tap_position": current_tap_position,
            "result_tap_position": result_tap_position,
            "penalty_cost_eur": penalty_cost_eur,
            "pst_name": pst.name,
            "tso": pst.tso,
            "pst_penalty_factor": float(pst.penalty_factor),
            "penalty_reference_pst_pra": penalty_reference,
            "effective_penalty_cost_per_tap_eur": effective_penalty_cost_per_tap,
        }

    overload_slack_n0_result: Dict[str, float] = {}
    # overload_slack_n0 is indexed over the SECURED branches only (scanned branches have their
    # own slack variable), so iterate secured_branch_ids, not the combined branch_ids.
    for branch_mrid in getattr(milp_model, "secured_branch_ids", []):
        overload_slack_n0_result[str(branch_mrid)] = float(pyo.value(milp_model.overload_slack_n0[branch_mrid]))

    objective_value = float(pyo.value(milp_model.objective)) if hasattr(milp_model, "objective") else None
    log.info(
        "MILP solved optimally: objective=%s, pst_count=%d, branches_with_overload=%d",
        objective_value,
        len(pst_results),
        sum(1 for v in overload_slack_n0_result.values() if v > 0),
    )
    log_objective_report(milp_model, pst_results, pst_list)
    run_profile.mark("Apply and AC verification")

    return pst_results, overload_slack_n0_result


