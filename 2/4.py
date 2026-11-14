"""single_pass.py — calculation_mode ac_single_pass.

One pass: the AC load flow, the N-X analysis and the sensitivities are computed once at the
initial state (run_pipeline), then ONE MILP build and solve, ONE applied solution with its AC load
flow, and ONE final check at the applied state (N-X security analysis and the remaining overloads
against PATL and TATL). No linearisation loop, no completeness loop, no convergence criteria, no
cyclic check, no new-pair check and no sensitivity refresh. The pairs in the model are the ones
the builder selects at the initial state; the final check only reports.

The three calculation calls (load flow, N-X security analysis, N-X sensitivities) sit behind
CalculationBackend, so a later backend (dc) can be added without touching the flow below.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from milp.ac_convergence_loop import (
    APPLY_APPLIED,
    APPLY_LOAD_FLOW_FAILED,
    APPLY_READBACK_EMPTY,
    NON_CONVERGENT_REASON,
    SOLVER_NON_OPTIMAL_REASON,
    NxLoopHooks,
    _count_nx_pairs,
    _cumulative_pst_result_rows,
    _get_monitored_branch_details,
    _log_iteration_summary,
    _runtime_columns,
    _zero_change_pst_rows,
    apply_solution_and_refresh,
    build_model_at_state,
    describe_original_network,
    element_overloads_ampere,
)
from data_model import AEType, select_by_side
from milp import nx_constraint_filter
from milp.milp_builder import (
    _get_monitored_branch_limits_and_ae_types, _get_monitored_branch_sides, _get_scanned_patl_ampere, tatl_is_above_patl,
)
from milp.milp_solver import MilpNotOptimalError, solve_n0_milp_model
from milp.operating_point_state import OperatingPointState
from milp.overload_diagnosis import log_overload_diagnosis
from milp.post_ra_congestion_check import _run_ac_loadflow, extract_hvdc_target_p_solution, extract_rd_target_p_solution
from post_milp.n0_results_export import write_lf_post_iterations_csv

log = logging.getLogger(__name__)

# termination reason of a finished single pass (mapped in data_model.AcVerificationStatus)
SINGLE_PASS_REASON = "single pass completed"
# the value of the loop-only columns of results_iteration_history.csv
NOT_APPLICABLE = "n/a"
# an overload smaller than this counts as none
_OVERLOAD_TOLERANCE_AMPERE = 1e-6


@dataclass
class CalculationBackend:
    """The calculation calls of a pass behind one interface.
    run_load_flow(network)                    -> (converged, status_name, result)
    compute_lf_nx(network)                    -> lf_nx {contingency: {branch: {1: A, 2: A}}}
                                                 ({} if nothing could be evaluated)
    compute_sensitivities_nx(network, lf_nx)  -> (sensitivity_pst_nx, sensitivity_rd_nx), keyed by
                                                 contingency, then (branch, RA)
    The load-flow result shape lf_n0 {branch: {1: A, 2: A}} is built by cgmes_loader.build_lf_initial."""
    name: str
    run_load_flow: Callable
    compute_lf_nx: Optional[Callable] = None
    compute_sensitivities_nx: Optional[Callable] = None


def ac_backend(nx_hooks: Optional[NxLoopHooks], use_reactive_limits: bool, extra_lf_params: Optional[dict]) -> CalculationBackend:
    """The AC backend: the pipeline's AC load flow and, when N-X is on, its N-X hooks."""
    return CalculationBackend(
        name="ac",
        run_load_flow=lambda network: _run_ac_loadflow(
            network, use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params,
        ),
        compute_lf_nx=nx_hooks.compute_lf_nx if nx_hooks is not None else None,
        compute_sensitivities_nx=nx_hooks.compute_sensitivities_nx if nx_hooks is not None else None,
    )


def nx_element_limits(
    ae_flat_list: list, timestamp: str, lf_original: dict, tatl_data: Optional[dict], lf_nx_original: Optional[dict] = None,
) -> dict:
    """The limit of every monitored N-X element, with the rules of the builder and of find_new_congested_pairs:
    "nx_patl" = secured branches (PATL), "nx_tatl" = secured branches whose TATL is above the PATL (TATL, side of
    the TATL), all by branch (the same for every contingency). "nx_scanned" = scanned branches with their PATL
    (the limit of a pair without an original N-X current) and "nx_scanned_pair" = the limit of each scanned
    (branch, contingency) pair of lf_nx_original: max(PATL, its original post-contingency current, same side).
    Returns {kind: {branch or pair: (side, limit A)}}."""
    branch_limits, ae_types = _get_monitored_branch_limits_and_ae_types(ae_flat_list, timestamp, lf_original)
    sides = _get_monitored_branch_sides(ae_flat_list)
    scanned_patl = _get_scanned_patl_ampere(ae_flat_list, timestamp)
    limits: dict = {"nx_patl": {}, "nx_tatl": {}, "nx_scanned": {}, "nx_scanned_pair": {}}
    for branch_mrid, limit_ampere in branch_limits.items():
        if limit_ampere <= 0.0:
            continue
        if ae_types[branch_mrid] == AEType.SCANNED:
            side = sides.get(branch_mrid, 1)
            limits["nx_scanned"][branch_mrid] = (side, scanned_patl[branch_mrid])
            for contingency_mrid, per_branch in (lf_nx_original or {}).items():
                per_side = per_branch.get(branch_mrid)
                initial = select_by_side(per_side, side, branch_mrid) if per_side else None
                if initial is not None:
                    limits["nx_scanned_pair"][(branch_mrid, contingency_mrid)] = (
                        side, max(scanned_patl[branch_mrid], float(initial)),
                    )
            continue
        limits["nx_patl"][branch_mrid] = (sides.get(branch_mrid, 1), limit_ampere)
        tatl_entry = (tatl_data or {}).get(branch_mrid)
        if tatl_entry is not None and tatl_is_above_patl(tatl_entry["tatl_limit_ampere"], limit_ampere):
            limits["nx_tatl"][branch_mrid] = (int(tatl_entry["side"]), float(tatl_entry["tatl_limit_ampere"]))
    return limits


def nx_overloads_ampere(
    nx_limits: dict, lf_nx: Optional[dict], monitoring: dict, model_scanned_pairs, curative_effect: dict,
    details: Optional[dict] = None,
) -> dict:
    """Overload (A) of every N-X element at one AC state: each (branch, contingency) of lf_nx whose
    branch has a limit in nx_limits, plus the pairs the model constrains. A pair of the model keeps
    the model's side and limit (monitoring) and, for PATL and scanned, adds the linearised effect
    of the curative actions of the solve (the quantity those rows bound; the TATL rows bound the
    state before the curative actions). A pair outside the model has no curative action of the
    solve on it: its AC current alone. model_scanned_pairs: a set of pairs, or a dict {pair: (side,
    limit)} with the limits of the model. A scanned pair is checked against its own limit
    (nx_limits["nx_scanned_pair"]), else the PATL of its branch. Keys: (kind, branch, contingency), kind
    nx_patl, nx_tatl, nx_scanned; a pair without AC data is left out. details (filled when given):
    {key: {"limit", "current" (AC), "curative_effect", "overload"}}."""
    overloads: dict = {}
    for kind in ("nx_patl", "nx_tatl", "nx_scanned"):
        limits_by_branch = nx_limits.get(kind, {})
        pair_limits = nx_limits.get("nx_scanned_pair", {}) if kind == "nx_scanned" else {}
        if kind == "nx_scanned":
            model_limits = model_scanned_pairs if isinstance(model_scanned_pairs, dict) else {}
            in_model = {
                pair: model_limits.get(pair) or pair_limits.get(pair) or limits_by_branch[pair[0]]
                for pair in model_scanned_pairs if pair[0] in limits_by_branch
            }
        else:
            in_model = monitoring.get(kind, {})
        elements = {
            (branch_mrid, contingency_mrid): pair_limits.get((branch_mrid, contingency_mrid)) or limits_by_branch[branch_mrid]
            for contingency_mrid, per_branch in (lf_nx or {}).items()
            for branch_mrid in per_branch
            if branch_mrid in limits_by_branch
        }
        elements.update(in_model)
        for (branch_mrid, contingency_mrid), (side, limit_ampere) in elements.items():
            per_side = (lf_nx or {}).get(contingency_mrid, {}).get(branch_mrid)
            current = select_by_side(per_side, side, branch_mrid) if per_side else None
            if current is None:
                continue
            effect = (
                curative_effect.get((branch_mrid, contingency_mrid), 0.0)
                if kind != "nx_tatl" and (branch_mrid, contingency_mrid) in in_model else 0.0
            )
            overload = max(0.0, float(current) + effect - limit_ampere)
            overloads[(kind, branch_mrid, contingency_mrid)] = overload
            if details is not None and overload > _OVERLOAD_TOLERANCE_AMPERE:
                details[(kind, branch_mrid, contingency_mrid)] = {
                    "limit": float(limit_ampere), "current": float(current), "curative_effect": float(effect), "overload": overload,
                }
    return overloads


def summarize_overloads(overloads: dict) -> dict:
    """Overload metrics of a final check, one group per kind: n0 (N-0, secured and scanned
    branches of the model), nx_patl, nx_tatl and nx_scanned ((branch, contingency) pairs).
    *_overloaded_elements = elements (N-0: branches, N-X: pairs) over their limit,
    *_overloaded_branches = distinct branches among them, *_overload_a = the sum of the
    overloads in ampere."""
    summary: dict = {}
    for kind in ("n0", "nx_patl", "nx_tatl", "nx_scanned"):
        over = {key: value for key, value in overloads.items() if key[0] == kind and value > _OVERLOAD_TOLERANCE_AMPERE}
        summary[f"final_check_{kind}_overloaded_elements"] = len(over)
        summary[f"final_check_{kind}_overloaded_branches"] = len({key[1] for key in over})
        summary[f"final_check_{kind}_overload_a"] = sum(over.values())
    return summary


def run_final_check(
    network, state: OperatingPointState, model, backend: CalculationBackend, refresh_nx: bool = True,
    nx_limits: Optional[dict] = None,
) -> dict:
    """The final check at the state the network is in: refresh the N-X load flow through the
    backend (when N-X is on), then the remaining overloads. N-0: the branches of the model,
    against the limits the model used. N-X (nx_overloads_ampere): every pair of the N-X load flow
    whose branch has a limit in nx_limits (nx_element_limits), PATL, TATL and scanned, not only
    the pairs of the model, with the linearised effect of the curative actions on the PATL and
    scanned pairs of the model (curative actions are never applied to the network). It reports
    only: no pair is added and no further pass starts. Returns the summary metrics
    (summarize_overloads) plus final_check_nx_evaluated (False when N-X is off or its load flow
    could not be evaluated at the applied state). refresh_nx False: the network is the original
    one (nothing applied), so the N-X load flow of the state is used as it is. nx_limits None:
    the pairs of the model only."""
    nx_evaluated = backend.compute_lf_nx is not None and state.current_lf_nx is not None
    if refresh_nx and nx_evaluated:
        refreshed_lf_nx = backend.compute_lf_nx(network)
        if refreshed_lf_nx:
            state.current_lf_nx = refreshed_lf_nx
        else:
            nx_evaluated = False
            log.warning(
                "Final check: the N-X load flow could not be evaluated at the applied state - the "
                "N-X overloads below describe the initial N-X state."
            )
    monitoring = getattr(model, "overload_monitoring", {})
    # every scanned pair the builder evaluated (rows built and rows skipped as never binding) with its limit
    scanned_sides = (nx_limits or {}).get("nx_scanned", {})
    scanned_pair_limits = getattr(model, "nx_scanned_pair_limit_ampere", {})
    model_scanned_pairs = {
        pair: (scanned_sides.get(pair[0], (1, 0.0))[0], scanned_pair_limits[pair]) if pair in scanned_pair_limits else None
        for pair in nx_constraint_filter.pairs_with_skipped(model, "nx_scanned")
    }
    curative_effect: dict = {}
    if hasattr(model, "nx_curative_effect_ampere"):
        for branch_mrid, contingency_mrid in [*monitoring.get("nx_patl", {}), *sorted(model_scanned_pairs)]:
            curative_effect[(branch_mrid, contingency_mrid)] = model.nx_curative_effect_ampere(branch_mrid, contingency_mrid)
    state.last_curative_effect_ampere = {pair: curative_effect[pair] for pair in monitoring.get("nx_patl", {}) if pair in curative_effect}
    overloads = element_overloads_ampere({"n0": monitoring.get("n0", {})}, state.current_lf_n0, None, {})
    overload_details: dict = {}
    overloads.update(nx_overloads_ampere(
        nx_limits or {}, state.current_lf_nx, monitoring, model_scanned_pairs, curative_effect, details=overload_details,
    ))
    result = summarize_overloads(overloads)
    result["final_check_nx_evaluated"] = nx_evaluated
    log.info(
        "[SINGLE PASS] final check | N-0 overloaded branches=%d (%.1f A) | N-X PATL pairs=%d (%.1f A) | "
        "N-X TATL pairs=%d (%.1f A) | N-X scanned pairs=%d (%.1f A)",
        result["final_check_n0_overloaded_elements"], result["final_check_n0_overload_a"],
        result["final_check_nx_patl_overloaded_elements"], result["final_check_nx_patl_overload_a"],
        result["final_check_nx_tatl_overloaded_elements"], result["final_check_nx_tatl_overload_a"],
        result["final_check_nx_scanned_overloaded_elements"], result["final_check_nx_scanned_overload_a"],
    )
    log_overload_diagnosis(model, overload_details)
    return result


def run_single_pass(
    *,
    state: OperatingPointState,
    nx_hooks: Optional[NxLoopHooks],
    network,
    pst_list: list,
    rd_list: list,
    ae_flat_list: list,
    timestamp: str,
    lf_initial: dict,
    pst_tap_bounds: dict,
    pst_group_equality_pairs: list,
    penalty_reference_pst_pra: float,
    penalty_reference_rd_pra: float,
    penalty_cost_overload_n0: float,
    pst_degrees_per_tap: dict,
    generator_data: dict,
    bid_list: Optional[list] = None,
    glsk: Optional[dict] = None,
    ra_groups: Optional[dict] = None,
    raart: Optional[str] = None,
    decision_point_hours_before_run: float = 1.0,
    penalty_reference_pst_pra_per_tso: Optional[dict] = None,
    penalty_reference_rd_pra_per_tso: Optional[dict] = None,
    max_ra_count_per_timestamp_tso: Optional[dict] = None,
    ra_limit_penalty: float = 0.0,
    ra_limit_constraint_hard: bool = True,
    min_loading_threshold_pct: float = 0.0,
    pst_taping_discrete: bool = True,
    max_taps_per_tso: int = 6,
    max_taps_per_tso_overrides: Optional[dict] = None,
    incremental_ac_fallback_max_steps: int = 5,
    output_path: Optional[Path] = None,
    nx_inputs=None,
    use_reactive_limits: bool = False,
    extra_lf_params: Optional[dict] = None,
    scanned_element_constraint_hard: bool = True,
    scanned_element_penalty_eur_per_ka: float = 0.0,
    iteration_records: Optional[list] = None,
    unit_costs: Optional[dict] = None,
    hvdc_inputs=None,
    ct_inputs=None,
    backend: Optional[CalculationBackend] = None,
    **ignored_loop_settings,
) -> tuple[dict, object, dict, str, list[dict], dict]:
    """The single pass on `state` (the original network). Accepts the keyword arguments of the
    linearisation loop; the iteration caps, convergence thresholds and the sensitivity thresholds
    of the refresh (everything in ignored_loop_settings, e.g. max_change_ra) have no effect.
    Returns (N-0 load flow, final model, PST rows, termination_reason, completeness log (row 0
    only), final check metrics)."""
    log.info(
        "calculation_mode ac_single_pass: iteration caps and convergence thresholds are ignored "
        "(ignored here: %s)", ", ".join(sorted(ignored_loop_settings)) or "none",
    )
    branch_details = _get_monitored_branch_details(ae_flat_list, timestamp)
    if not branch_details:
        raise ValueError("No monitored branches found for timestamp %s" % timestamp)
    bid_list = bid_list or []
    glsk = glsk or {}
    backend = backend or ac_backend(nx_hooks, use_reactive_limits, extra_lf_params)

    timings = {"build_and_solve": 0.0, "load_flow": 0.0, "n0_sensitivity": 0.0, "nx_lf": 0.0, "nx_sensitivity": 0.0}
    lf_before = state.current_lf_n0
    lf_post_iteration_records: list[dict] = []

    _t = time.perf_counter()
    milp_model, current_pst_list, current_generator_data = build_model_at_state(
        state, nx_inputs,
        pst_list=pst_list, rd_list=rd_list, ae_flat_list=ae_flat_list, timestamp=timestamp,
        lf_before=lf_before, pst_tap_bounds=pst_tap_bounds,
        pst_group_equality_pairs=pst_group_equality_pairs, pst_degrees_per_tap=pst_degrees_per_tap,
        penalty_reference_pst_pra=penalty_reference_pst_pra,
        penalty_reference_pst_pra_per_tso=penalty_reference_pst_pra_per_tso,
        penalty_reference_rd_pra=penalty_reference_rd_pra,
        penalty_reference_rd_pra_per_tso=penalty_reference_rd_pra_per_tso,
        penalty_cost_overload_n0=penalty_cost_overload_n0, generator_data=generator_data,
        bid_list=bid_list, glsk=glsk, ra_groups=ra_groups, raart=raart,
        decision_point_hours_before_run=decision_point_hours_before_run,
        max_ra_count_per_timestamp_tso=max_ra_count_per_timestamp_tso,
        ra_limit_penalty=ra_limit_penalty, ra_limit_constraint_hard=ra_limit_constraint_hard,
        min_loading_threshold_pct=min_loading_threshold_pct, pst_taping_discrete=pst_taping_discrete,
        max_taps_per_tso=max_taps_per_tso, max_taps_per_tso_overrides=max_taps_per_tso_overrides,
        scanned_element_constraint_hard=scanned_element_constraint_hard,
        scanned_element_penalty_eur_per_ka=scanned_element_penalty_eur_per_ka, unit_costs=unit_costs,
        **({"hvdc_inputs": hvdc_inputs} if hvdc_inputs is not None else {}),
        **({"ct_inputs": ct_inputs} if ct_inputs is not None else {}),
    )
    total_pairs = _count_nx_pairs(milp_model)
    # the final check covers every monitored N-X element, not only the pairs of the model
    nx_limits = (
        nx_element_limits(
            ae_flat_list, timestamp, state.original_lf_n0, nx_inputs.tatl_data, lf_nx_original=state.original_lf_nx,
        )
        if nx_inputs is not None else None
    )
    completeness_log = [{"completeness_iteration": 0, "new_constraints_added": 0, "total_nx_constraints": total_pairs}]

    solve_record = {
        "solve_index": 1, "completeness_iteration": 0, "linearisation_iteration": 1,
        "load_flow_converged": False, "solution_applied": False,
        "overload_change_ampere": NOT_APPLICABLE, "criterion_1_met": NOT_APPLICABLE,
        "criterion_3_met": NOT_APPLICABLE, "criterion_4_met": NOT_APPLICABLE,
        "cycle_match_iteration": NOT_APPLICABLE, "cycle_selected_iteration": NOT_APPLICABLE,
        "converged_by": NOT_APPLICABLE, "converged": NOT_APPLICABLE, "new_constraints_added": 0,
        "nx_contingencies_lost_at_refresh": 0,
        "invalid_sensitivity_contingencies": len(state.invalid_sensitivity_nx),
        "invalid_sensitivity_branches_n0": len(state.invalid_sensitivity_n0),
        "stale_reused_sensitivity_contingencies": 0, "stale_reused_sensitivity_branches_n0": 0,
    }

    def _original_network_report(reason: str):
        """Nothing applied: the files describe the original network."""
        describe_original_network(milp_model)
        rows = _zero_change_pst_rows(
            milp_model, pst_list, state, penalty_reference_pst_pra, penalty_reference_pst_pra_per_tso,
        )
        check = run_final_check(network, state, milp_model, backend, refresh_nx=False, nx_limits=nx_limits)
        return state.current_lf_n0, milp_model, rows, reason, check

    try:
        pst_results, overload_slack = solve_n0_milp_model(
            milp_model=milp_model, pst_list=current_pst_list, timestamp=timestamp,
            penalty_reference_pst_pra=penalty_reference_pst_pra,
            penalty_reference_pst_pra_per_tso=penalty_reference_pst_pra_per_tso,
        )
    except MilpNotOptimalError as exc:
        timings["build_and_solve"] = time.perf_counter() - _t
        log.error("[SINGLE PASS] %s - nothing is applied", exc)
        state.solve_history.append({**solve_record, **_runtime_columns(timings)})
        lf_n0, model, rows, reason, check = _original_network_report(SOLVER_NON_OPTIMAL_REASON)
        _finish(output_path, iteration_records, branch_details, lf_post_iteration_records, timestamp)
        return lf_n0, model, rows, reason, completeness_log, check
    timings["build_and_solve"] = time.perf_counter() - _t
    import pyomo.environ as pyo
    solve_record["objective_value"] = float(pyo.value(milp_model.objective))
    rd_target_p_solution = extract_rd_target_p_solution(milp_model, current_generator_data)
    # the HVDC set points travel in the same dict (apply, write-back, fallback ramp)
    rd_target_p_solution.update(extract_hvdc_target_p_solution(milp_model))

    applied = apply_solution_and_refresh(
        network=network, state=state, milp_model=milp_model, pst_results=pst_results,
        rd_target_p_solution=rd_target_p_solution, current_generator_data=current_generator_data,
        cost=solve_record["objective_value"], nx_hooks=None,
        use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params,
        incremental_ac_fallback_max_steps=incremental_ac_fallback_max_steps, iteration=1,
        run_load_flow=backend.run_load_flow,
    )
    timings["load_flow"] = applied.seconds_load_flow
    if applied.kind != APPLY_APPLIED:
        log.warning(
            "[SINGLE PASS] the solution could not be applied (%s) - the original network is reported",
            "empty generator read-back" if applied.kind == APPLY_READBACK_EMPTY else "load flow non-convergent",
        )
        if applied.kind == APPLY_LOAD_FLOW_FAILED:
            lf_post_iteration_records.append({
                "ac_iteration": 1, "lf_before": lf_before, "lf_after": lf_before,
                "solution_applied_to_network": False,
            })
        state.solve_history.append({**solve_record, **_runtime_columns(timings)})
        lf_n0, model, rows, reason, check = _original_network_report(NON_CONVERGENT_REASON)
        _finish(output_path, iteration_records, branch_details, lf_post_iteration_records, timestamp)
        return lf_n0, model, rows, reason, completeness_log, check

    solve_record["load_flow_converged"] = True
    solve_record["solution_applied"] = True
    _log_iteration_summary(
        "SINGLE PASS", 1, 1, milp_model, applied.lf_after, branch_details, pst_results,
        rd_target_p_solution, state.original_tap_by_pst, state.original_target_p_by_gu,
        applied.convergence_path,
        hvdc_ei_mrids=frozenset(u.ei_mrid for u in hvdc_inputs.units) if hvdc_inputs is not None else frozenset(),
    )
    lf_post_iteration_records.append({
        "ac_iteration": 1, "lf_before": lf_before, "lf_after": applied.lf_after,
        "solution_applied_to_network": True,
    })
    _t = time.perf_counter()
    check = run_final_check(network, state, milp_model, backend, nx_limits=nx_limits)
    timings["nx_lf"] = time.perf_counter() - _t
    state.solve_history.append({**solve_record, **_runtime_columns(timings)})
    _finish(output_path, iteration_records, branch_details, lf_post_iteration_records, timestamp)
    return (
        state.current_lf_n0, milp_model, _cumulative_pst_result_rows(pst_results, state),
        SINGLE_PASS_REASON, completeness_log, check,
    )


def _finish(output_path, iteration_records, branch_details, lf_post_iteration_records, timestamp) -> None:
    """lf_post_n0.csv and the iteration records, as the completeness loop writes them."""
    if output_path is not None:
        write_lf_post_iterations_csv(output_path, branch_details, lf_post_iteration_records, timestamp)
    if iteration_records is not None:
        iteration_records.extend(
            {"ac_iteration": r["ac_iteration"], "solution_applied_to_network": r["solution_applied_to_network"]}
            for r in lf_post_iteration_records
        )
