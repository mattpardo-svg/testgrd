"""dc_pass.py — calculation_mode dc (docs/specs/dc_mode_specs.md).

The DC load flow, the DC N-X security analysis and the DC sensitivities (run_pipeline, through
dc_converter) give the currents and sensitivities in ampere at the initial state.
Here: ONE MILP build and solve, ONE applied solution with its AC load flow and ONE final AC check
(N-X security analysis and the remaining overloads against PATL and TATL). No linearisation loop,
no completeness loop, no convergence criteria, no new-pair check, no refresh, no feedback.

It reuses the shared pieces of the single pass (build_model_at_state, apply_solution_and_refresh,
run_final_check); run_single_pass and the loops are not touched. Differences to run_single_pass:
- the apply step gets a hook that obtains bus voltages for the incremental fallback (a DC load
  flow leaves them NaN): one extra AC load flow at the unmodified state, only on the failure path;
- an AC load flow that does not converge after the fallback keeps the DC solution (D5): the model
  and PST/RD rows are the solved ones, final_check_evaluated is False, the run goes on; the N-X
  state of the solution is unknown (state.current_lf_nx None), so the N-X result files show the
  model's prediction, not the initial DC state;
- a solve that is not optimal reports the original network without any AC calculation.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from milp.ac_convergence_loop import (
    APPLY_APPLIED,
    APPLY_READBACK_EMPTY,
    SOLVER_NON_OPTIMAL_REASON,
    NxLoopHooks,
    _count_nx_pairs,
    _cumulative_pst_result_rows,
    _get_monitored_branch_details,
    _log_iteration_summary,
    _runtime_columns,
    _signed_bid_volume_delta,
    _zero_change_pst_rows,
    apply_solution_and_refresh,
    build_model_at_state,
    describe_original_network,
)
from milp.milp_solver import MilpNotOptimalError, solve_n0_milp_model
from milp.operating_point_state import OperatingPointState
from milp.post_ra_congestion_check import (
    _run_ac_loadflow,
    _snapshot_bus_voltage_state,
    extract_hvdc_target_p_solution,
    extract_rd_target_p_solution,
    restore_network_state,
)
from milp.single_pass import (
    NOT_APPLICABLE,
    SINGLE_PASS_REASON,
    CalculationBackend,
    _finish,
    ac_backend,
    nx_element_limits,
    run_final_check,
)

log = logging.getLogger(__name__)

# the final check of the DC mode is an AC check
FINAL_CHECK_BASIS = "ac"


def ac_check_parameters(extra_lf_params: Optional[dict]) -> dict:
    """Load-flow parameters of the final AC check. A DC load flow leaves no bus voltages, so a
    configured PREVIOUS_VALUES start cannot be used for the first AC load flow: UNIFORM_VALUES then."""
    import pypowsybl as pp

    parameters = dict(extra_lf_params or {})
    if parameters.get("voltage_init_mode") == pp.loadflow.VoltageInitMode.PREVIOUS_VALUES:
        log.info("calculation_mode dc: voltage_init_mode PREVIOUS_VALUES has no voltages to start from after the "
                 "DC load flow - the final AC load flow starts from UNIFORM_VALUES")
        parameters["voltage_init_mode"] = pp.loadflow.VoltageInitMode.UNIFORM_VALUES
    return parameters


def fallback_voltages_at_unmodified_state(use_reactive_limits: bool, extra_lf_params: Optional[dict]):
    """The prepare_fallback hook of apply_solution_and_refresh: the network goes back to the start
    taps and set points and one AC load flow there gives the bus voltages the incremental fallback
    warm-starts from. Returns the voltage snapshot, or None when that load flow does not converge."""
    def prepare(network, start_tap_positions, start_target_p):
        log.info("calculation_mode dc: the direct AC jump did not converge - one AC load flow at the unmodified "
                 "state gives the start voltages of the incremental fallback")
        restore_network_state(network, previous_taps=start_tap_positions, previous_target_p=start_target_p)
        converged, status_name, _ = _run_ac_loadflow(
            network, use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params,
        )
        if not converged:
            log.warning("calculation_mode dc: the AC load flow at the unmodified state did not converge (%s)", status_name)
            return None
        return _snapshot_bus_voltage_state(network)
    return prepare


def not_evaluated_check(reason: str) -> dict:
    """The final-check rows of a check that did not run: no overload rows."""
    return {"final_check_basis": FINAL_CHECK_BASIS, "final_check_evaluated": False, "final_check_reason": reason}


def run_dc_pass(
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
    """The DC pass on `state` (the original network, DC currents and sensitivities). Accepts the
    keyword arguments of the loop; the iteration caps, convergence thresholds and sensitivity
    thresholds (ignored_loop_settings) have no effect. `backend` is the AC backend of the final
    check (default: ac_backend of nx_hooks). Returns (N-0 load flow, final model, PST rows,
    termination_reason, completeness log (row 0 only), final-check rows)."""
    log.info(
        "calculation_mode dc: iteration caps and convergence thresholds are ignored "
        "(ignored here: %s)", ", ".join(sorted(ignored_loop_settings)) or "none",
    )
    branch_details = _get_monitored_branch_details(ae_flat_list, timestamp)
    if not branch_details:
        raise ValueError("No monitored branches found for timestamp %s" % timestamp)
    bid_list = bid_list or []
    glsk = glsk or {}
    ac_parameters = ac_check_parameters(extra_lf_params)
    backend = backend or ac_backend(nx_hooks, use_reactive_limits, ac_parameters)

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
    # the final check covers every monitored N-X element, not only the pairs of the model; the scanned
    # effective limit is anchored to the original (DC-converted) N-0 currents, as in the model
    nx_limits = (
        nx_element_limits(ae_flat_list, timestamp, state.original_lf_n0, nx_inputs.tatl_data)
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

    try:
        pst_results, overload_slack = solve_n0_milp_model(
            milp_model=milp_model, pst_list=current_pst_list, timestamp=timestamp,
            penalty_reference_pst_pra=penalty_reference_pst_pra,
            penalty_reference_pst_pra_per_tso=penalty_reference_pst_pra_per_tso,
        )
    except MilpNotOptimalError as exc:
        timings["build_and_solve"] = time.perf_counter() - _t
        log.error("[DC PASS] %s - nothing is applied", exc)
        state.solve_history.append({**solve_record, **_runtime_columns(timings)})
        describe_original_network(milp_model)
        rows = _zero_change_pst_rows(
            milp_model, pst_list, state, penalty_reference_pst_pra, penalty_reference_pst_pra_per_tso,
        )
        _finish(output_path, iteration_records, branch_details, lf_post_iteration_records, timestamp)
        return (
            state.current_lf_n0, milp_model, rows, SOLVER_NON_OPTIMAL_REASON, completeness_log,
            not_evaluated_check("MILP solve not optimal - nothing applied"),
        )
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
        prepare_fallback=fallback_voltages_at_unmodified_state(use_reactive_limits, ac_parameters),
    )
    timings["load_flow"] = applied.seconds_load_flow
    if applied.kind != APPLY_APPLIED:
        reason = (
            "the generator read-back of the network was empty - nothing could be applied"
            if applied.kind == APPLY_READBACK_EMPTY
            else f"the AC load flow did not converge after the incremental fallback ({applied.status_name})"
        )
        log.warning("[DC PASS] %s - the DC solution is kept, the final AC check is not evaluated", reason)
        # the DC solution has no AC state that could be reverted: record it without a load flow
        state.record_applied_solution(
            applied_tap_by_pst={
                mrid: int(round(float(row["result_tap_position"]))) for mrid, row in pst_results.items()
            },
            refreshed_target_p_by_gu={
                gu: float(rd_target_p_solution.get(gu, data.get("target_p", 0.0)))
                for gu, data in current_generator_data.items()
            },
            signed_bid_volume_delta_mw=_signed_bid_volume_delta(milp_model),
            lf_n0_after=state.current_lf_n0,
        )
        state.annotate_latest_member(model=milp_model, pst_results=pst_results, cost=solve_record["objective_value"])
        # no N-X load flow describes the DC solution: the exports (lf_nx_final) fall back to the model's
        # prediction instead of showing the initial DC state as the post-PRA state (D5.3a)
        state.current_lf_nx = None
        lf_post_iteration_records.append({
            "ac_iteration": 1, "lf_before": lf_before, "lf_after": lf_before, "solution_applied_to_network": False,
        })
        solve_record["solution_applied"] = False
        state.solve_history.append({**solve_record, **_runtime_columns(timings)})
        _finish(output_path, iteration_records, branch_details, lf_post_iteration_records, timestamp)
        return (
            state.current_lf_n0, milp_model, _cumulative_pst_result_rows(pst_results, state), SINGLE_PASS_REASON,
            completeness_log, not_evaluated_check(reason),
        )

    solve_record["load_flow_converged"] = True
    solve_record["solution_applied"] = True
    _log_iteration_summary(
        "DC PASS", 1, 1, milp_model, applied.lf_after, branch_details, pst_results,
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
        SINGLE_PASS_REASON, completeness_log,
        {"final_check_basis": FINAL_CHECK_BASIS, "final_check_evaluated": True, **check},
    )
