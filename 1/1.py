"""ac_convergence_loop.py — the linearisation loop (build at the current operating point,
solve, apply, AC load flow, convergence check; run_post_rao_ac_verification_loop) and the
completeness loop around it (run_nested_rao_loop), both on one OperatingPointState.
A PST-only run is the same loop with rd_list=[].
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional

import run_profile
from data_model import AEType, AssessedElement, PST, RAKind, RDUnit
from pre_milp import cgmes_loader
from data_model import select_by_side
from milp.milp_builder import build_n0_milp_model, CtMilpInputs, HvdcMilpInputs, NxMilpInputs, PstMilpInputs, RdMilpInputs
from milp.milp_solver import MilpNotOptimalError, solve_n0_milp_model
from milp.operating_point_state import OperatingPointState, find_cyclic_repetition
from pre_milp.n0_sensitivity import compute_sensitivity_hvdc_n0, compute_sensitivity_pst_n0, compute_sensitivity_rd_n0
from pre_milp.sensitivity_checks import STALE_REUSED, log_invalid, resolve_n0, resolve_nx
from post_milp.n0_results_export import write_lf_post_iterations_csv
from milp.post_ra_congestion_check import (
    ZERO_MOVEMENT,
    apply_preventive_solution_to_network,
    extract_hvdc_target_p_solution,
    extract_rd_target_p_solution,
    restore_network_state,
    _run_ac_loadflow,
    _snapshot_bus_voltage_state,
    _restore_bus_voltage_state,
    _apply_changes_incrementally,
)

log = logging.getLogger(__name__)

# termination reason of the cyclic-repetition exit (mapped in data_model.AcVerificationStatus)
CYCLIC_REPETITION_REASON = "cyclic repetition of results"
# termination reason when the guard marked a sensitivity set invalid and the last valid set of the
# same contingency/branch is missing or no longer matches the model (mapped in data_model)
SENSITIVITY_INVALID_REASON = "invalid sensitivities - no usable earlier valid set"
# load flow failures: back at the original network / back at the last converged state
NON_CONVERGENT_REASON = "load flow non-convergent"
REVERTED_TO_LAST_REASON = "load flow non-convergent - reverted to last converged state"
# a MILP solve ended non-optimal (infeasible or other)
SOLVER_NON_OPTIMAL_REASON = "MILP solve not optimal"
# reasons that end the loop without a new applied solution in that iteration
# cycle exit ties: total AC overload compared to 1e-9 A, cumulative cost to 1e-6 EUR
_TIE_DIGITS_OVERLOAD_AMPERE = 9
_TIE_DIGITS_COST_EUR = 6
_STOP_REASONS = (NON_CONVERGENT_REASON, REVERTED_TO_LAST_REASON, SENSITIVITY_INVALID_REASON, SOLVER_NON_OPTIMAL_REASON)

# (slack variable, constraint that carries it) of every overload slack of the model
_SLACK_CONSTRAINTS = (
    ("overload_slack_n0", "xne_loading_n0"),
    ("scanned_overload_slack", "scanned_element_n0"),
    ("nx_patl_overload_slack", "nx_patl_loading"),
    ("nx_tatl_overload_slack", "nx_tatl_loading"),
    ("nx_scanned_overload_slack", "nx_scanned_loading"),
)


def _get_monitored_branch_details(ae_flat_list: list[AssessedElement], timestamp: str) -> dict[str, dict[str, object]]:
    branch_details: dict[str, dict[str, object]] = {}
    for ae in ae_flat_list:
        if ae.ae_type != AEType.SECURED or ae.branch_mrid is None:
            continue
        patl_ka = ae.patl_ka.get(timestamp)
        if patl_ka is None:
            continue
        branch_details[str(ae.branch_mrid)] = {
            "branch_name": ae.name,
            "tso": ae.tso,
            "patl_ka": float(patl_ka),
            "branch_side": int(getattr(ae, "branch_side", 1)),
        }
    return branch_details


def _extract_phase_tap_positions(network) -> dict[str, float]:
    tap_positions: dict[str, float] = {}
    try:
        ptcs = network.get_phase_tap_changers()
        if hasattr(ptcs, "index") and not callable(getattr(ptcs, "index", None)):
            for mrid in ptcs.index:
                row = ptcs.loc[mrid] if hasattr(ptcs, "loc") else None
                tap_positions[str(mrid)] = float(row.get("tap", 0.0)) if row is not None else 0.0
        else:
            for item in ptcs:
                mrid = getattr(item, "mrid", getattr(item, "id", str(item)))
                tap_positions[str(mrid)] = float(getattr(item, "tap", 0.0))
    except Exception:
        log.exception("Failed to extract phase tap changer positions")
    return tap_positions


def _extract_generator_target_p(network) -> dict[str, float]:
    """{gu_mrid: target_p} read from network.get_generators() after a load flow (the RD
    baseline the MILP's RD bounds add to, as the tap position is for a PST)."""
    target_p_by_gu: dict[str, float] = {}
    try:
        gens = network.get_generators()
        if hasattr(gens, "index") and not callable(getattr(gens, "index", None)):
            for mrid in gens.index:
                row = gens.loc[mrid] if hasattr(gens, "loc") else None
                target_p_by_gu[str(mrid)] = float(row.get("target_p", 0.0)) if row is not None else 0.0
        else:
            for item in gens:
                mrid = getattr(item, "mrid", getattr(item, "id", str(item)))
                target_p_by_gu[str(mrid)] = float(getattr(item, "target_p", 0.0))
    except Exception:
        log.exception("Failed to extract generator target_p values")
    return target_p_by_gu


def _apply_pst_results_to_network(network, pst_results: dict[str, dict[str, object]]) -> None:
    for pst_mrid, result in pst_results.items():
        tap_position = result["result_tap_position"]
        network.update_phase_tap_changers(id=pst_mrid, tap=int(round(float(tap_position))))


def _log_iteration_summary(
    loop_label: str,
    iteration: int,
    max_iterations: int,
    milp_model,
    lf_state: dict,
    branch_details: dict,
    pst_results: dict,
    rd_target_p_solution: dict,
    original_tap_positions: dict,
    original_target_p: dict,
    convergence_path: str,
    hvdc_ei_mrids: frozenset = frozenset(),
) -> None:
    """One INFO line per iteration. XNECs violated = branches whose AC current in lf_state
    (this iteration's load flow, else the last verified one) exceeds the PATL, N-0 only: an
    N-X count would need an extra security analysis per iteration (the completeness check
    does it once). Activated PSTs/units, taps and MW are measured against the original
    network, not the previous iteration. RD MW is one figure: the balance constraint and
    GLSK weights summing to 1 make total up equal total down.
    """
    import pyomo.environ as pyo

    objective = float(pyo.value(milp_model.objective)) if hasattr(milp_model, "objective") else None

    xnec_base = sum(
        1 for branch_mrid, details in branch_details.items()
        if lf_state.get(branch_mrid, {}).get(details["branch_side"], 0.0) > details["patl_ka"] * 1000.0
    )

    psts_activated = [
        (mrid, row) for mrid, row in pst_results.items()
        if abs(float(row["result_tap_position"]) - float(original_tap_positions.get(mrid, row["current_tap_position"]))) > ZERO_MOVEMENT
    ]
    total_taps = sum(
        abs(float(row["result_tap_position"]) - float(original_tap_positions.get(mrid, row["current_tap_position"])))
        for mrid, row in psts_activated
    )

    # the HVDC units (EI generators) are reported on their own, not as redispatch
    all_gu_ids = (set(original_target_p) | set(rd_target_p_solution)) - set(hvdc_ei_mrids)
    rd_deviations = {
        gu: abs(float(rd_target_p_solution.get(gu, original_target_p.get(gu, 0.0)))
                - float(original_target_p.get(gu, 0.0)))
        for gu in all_gu_ids
    }
    hvdc_deviations = {
        ei: abs(float(rd_target_p_solution.get(ei, original_target_p.get(ei, 0.0))) - float(original_target_p.get(ei, 0.0)))
        for ei in hvdc_ei_mrids
    }
    gus_activated = sum(1 for dev in rd_deviations.values() if dev > ZERO_MOVEMENT)
    # / 2: the balance constraint makes total up equal total down, so the absolute sum over
    # units counts each MW twice
    total_rd_mw = sum(rd_deviations.values()) / 2.0
    # countertrade bids share the unit movements counted above (they are part of the redispatch MW of the
    # criteria); their own volume is shown next to it (cumulative against the original network)
    ct_text = ""
    ct_bid_ids = getattr(milp_model, "ct_bid_ids", frozenset())
    if ct_bid_ids:
        ct_up = sum(float(pyo.value(milp_model.rd_cumulative_up_term[b])) for b in ct_bid_ids)
        ct_down = sum(float(pyo.value(milp_model.rd_cumulative_down_term[b])) for b in ct_bid_ids)
        ct_text = " | countertrade: %.1f MW up, %.1f MW down" % (ct_up, ct_down)

    log.info(
        "[%s] Iteration %d/%d | objective=%s | XNECs violated=%d | "
        "PSTs activated=%d (%s taps total) | RD activated=%d generating unit(s) "
        "(%.1f MW redispatched) | %s",
        loop_label, iteration, max_iterations,
        f"{objective:,.0f}" if objective is not None else "N/A",
        xnec_base,
        len(psts_activated), f"{total_taps:g}",
        gus_activated, total_rd_mw,
        convergence_path + (
            " | HVDC activated=%d unit(s) (%.1f MW moved per side)"
            % (sum(1 for dev in hvdc_deviations.values() if dev > ZERO_MOVEMENT), sum(hvdc_deviations.values()) / 2.0)
            if hvdc_ei_mrids else ""
        ) + ct_text,
    )



@dataclass
class NxLoopHooks:
    """The N-X pieces the loop needs at the CURRENT operating point, supplied by
    run_pipeline (which owns the N-X parameters), so this module stays free of
    N-X plumbing.

    compute_lf_nx(network)                  -> lf_nx at the network's current state
                                                ({} if nothing could be evaluated)
    compute_sensitivities_nx(network, lf_nx) -> (sensitivity_pst_nx, sensitivity_rd_nx),
                                                pruned like the original ones
    find_new_pairs(model, lf_nx)            -> number of overloaded (branch, contingency)
                                                pairs at that state that the model
                                                does not constrain yet (outage
                                                approximation / completeness check)
    """
    compute_lf_nx: Callable
    compute_sensitivities_nx: Callable
    find_new_pairs: Callable


def _signed_bid_volume_delta(milp_model) -> dict[str, float]:
    """This solve's signed volume per preventive bid: up - down (UP bids come out
    positive, DOWN bids negative, because each bid only has its own direction
    available in the first place)."""
    if not hasattr(milp_model, "rd_volume_upward"):
        return {}
    import pyomo.environ as pyo
    return {
        str(bid_mrid): float(pyo.value(milp_model.rd_volume_upward[bid_mrid]))
        - float(pyo.value(milp_model.rd_volume_downward[bid_mrid]))
        for bid_mrid in milp_model.bid_ids
    }


def _cumulative_pst_result_rows(pst_results: dict, state: OperatingPointState) -> dict:
    """Report PST results against the ORIGINAL network: original tap, cumulative
    change and final (applied) tap, plus the last solve's own start tap and change
    as diagnostics. The penalty is priced on the cumulative change, matching the
    objective."""
    rows: dict[str, dict] = {}
    for pst_mrid, row in pst_results.items():
        original_tap = int(state.original_tap_by_pst.get(pst_mrid, row["current_tap_position"]))
        final_tap = int(state.current_tap_by_pst.get(pst_mrid, round(float(row["result_tap_position"]))))
        cumulative_change = final_tap - original_tap
        per_tap = float(row.get("effective_penalty_cost_per_tap_eur", 0.0))
        rows[pst_mrid] = {
            **row,
            "original_tap_position": original_tap,
            "tap_change_result": cumulative_change,
            "result_tap_position": final_tap,
            "penalty_cost_eur": abs(cumulative_change) * per_tap,
            "last_solve_start_tap_position": row["current_tap_position"],
            "last_solve_tap_change": row["tap_change_result"],
        }
    return rows


def _last_applied_of_run(state: OperatingPointState, model, pst_results: dict):
    """The model and PST rows of the last applied solution: this call's, else the newest one of
    the whole run kept in the state; (None, {}) when nothing was applied yet."""
    if model is not None:
        return model, pst_results
    for record in reversed(state.member_records):
        if record.get("model") is not None:
            return record["model"], record["pst_results"]
    return None, {}


def describe_original_network(model) -> None:
    """Sets every variable of the model to 0 (no remedial action) and each overload slack to the
    overload of its constraint at that point, so the result exporters report the network the
    model was built on, unchanged. The on/off binary of a unit that is initially on is 1 (on as
    in the network, so no shut-down cost is reported; F2 rule 14). Marks the model with
    no_applied_solution = True."""
    import pyomo.environ as pyo
    for var in model.component_objects(pyo.Var):
        for index in var:
            var[index].set_value(0, skip_validation=True)
    unit_status = getattr(model, "rd_unit_status", None) or {}
    if hasattr(model, "rd_unit_on"):
        for gu_mrid in model.rd_unit_on:
            if unit_status.get(gu_mrid) is not None and unit_status[gu_mrid].initially_on:
                model.rd_unit_on[gu_mrid].set_value(1, skip_validation=True)
    for slack_name, constraint_name in _SLACK_CONSTRAINTS:
        if not (hasattr(model, slack_name) and hasattr(model, constraint_name)):
            continue
        slack, constraint = getattr(model, slack_name), getattr(model, constraint_name)
        for index in constraint:
            # constraint "current + effects <= limit + slack": body - upper is the overload at slack 0
            body, upper, lower = (pyo.value(constraint[index].body), constraint[index].upper, constraint[index].lower)
            # a constant left-hand side (no action touches the row) is stored by Pyomo as "lower <= body"
            overload = float(body) - float(pyo.value(upper)) if upper is not None else float(pyo.value(lower)) - float(body)
            slack[index].set_value(max(0.0, overload), skip_validation=True)
    model.no_applied_solution = True


def _zero_change_pst_rows(model, pst_list, state: OperatingPointState, penalty_reference_pst_pra: float,
                          penalty_reference_pst_pra_per_tso: Optional[dict]) -> dict:
    """PST rows of the original network: original tap, no change, no cost (same columns as
    _cumulative_pst_result_rows)."""
    preventive = {str(p.mrid): p for p in pst_list if p.kind == RAKind.PREVENTIVE}
    rows: dict[str, dict] = {}
    for pst_mrid in getattr(model, "pst_ids", []):
        pst = preventive.get(str(pst_mrid))
        if pst is None:
            continue
        original_tap = int(state.original_tap_by_pst.get(str(pst_mrid), pst.r0))
        reference = float((penalty_reference_pst_pra_per_tso or {}).get(pst.tso, penalty_reference_pst_pra))
        rows[str(pst_mrid)] = {
            "tap_change_result": 0, "current_tap_position": original_tap, "result_tap_position": original_tap,
            "penalty_cost_eur": 0.0, "pst_name": pst.name, "tso": pst.tso,
            "pst_penalty_factor": float(pst.penalty_factor), "penalty_reference_pst_pra": reference,
            "effective_penalty_cost_per_tap_eur": float(pst.penalty_factor) * reference,
            "original_tap_position": original_tap, "last_solve_start_tap_position": original_tap,
            "last_solve_tap_change": 0,
        }
    return rows


def element_overloads_ampere(
    monitoring: dict,
    lf_n0: dict,
    lf_nx: Optional[dict],
    curative_effect_ampere: dict,
) -> dict[tuple, float]:
    """Overload (A) of every monitored element at one AC state:
    overload_e = max(0, I_e - limit_e).

    monitoring = model.overload_monitoring: the elements of the model with the side and
    the limit its own constraints use ("n0": branch -> (side, limit A); "nx_patl" /
    "nx_tatl": (branch, contingency) -> (side, limit A)). I_e is the AC current at the
    state; for "nx_patl" it is the AC N-X current PLUS the linearised effect of the
    curative actions of the solve (curative_effect_ampere[(branch, contingency)], the
    quantity the PATL constraint bounds; curative actions are never applied to the
    network). "nx_tatl" bounds the state before the curative actions, so its I_e is the AC
    N-X current alone. An element without AC data at the state is left out.

    Keys: ("n0", branch), ("nx_patl", branch, contingency), ("nx_tatl", branch, contingency)."""
    overloads: dict[tuple, float] = {}
    for branch_mrid, (side, limit_ampere) in monitoring.get("n0", {}).items():
        per_side = lf_n0.get(branch_mrid)
        current = select_by_side(per_side, side, branch_mrid) if per_side else None
        if current is None:
            continue
        overloads[("n0", branch_mrid)] = max(0.0, float(current) - limit_ampere)
    for set_name in ("nx_patl", "nx_tatl"):
        for (branch_mrid, contingency_mrid), (side, limit_ampere) in monitoring.get(set_name, {}).items():
            per_side = (lf_nx or {}).get(contingency_mrid, {}).get(branch_mrid)
            current = select_by_side(per_side, side, branch_mrid) if per_side else None
            if current is None:
                continue
            curative_effect = (
                curative_effect_ampere.get((branch_mrid, contingency_mrid), 0.0) if set_name == "nx_patl" else 0.0
            )
            overloads[(set_name, branch_mrid, contingency_mrid)] = max(
                0.0, float(current) + curative_effect - limit_ampere,
            )
    return overloads


def overload_change_ampere(overloads_now: dict, overloads_before: dict) -> float:
    """Criterion 4 (change of element overloads): the SUM over the elements of
    |overload_e(now) - overload_e(before)|, in ampere. Per element, never netted: an
    element that gets worse and another that gets better both count. An element
    missing on one side counts as overload 0 there."""
    return sum(
        abs(overloads_now.get(key, 0.0) - overloads_before.get(key, 0.0))
        for key in set(overloads_now) | set(overloads_before)
    )


# outcomes of apply_solution_and_refresh
APPLY_APPLIED = "applied"
APPLY_READBACK_EMPTY = "readback_empty"
APPLY_LOAD_FLOW_FAILED = "load_flow_failed"


@dataclass
class AppliedSolution:
    """Result of apply_solution_and_refresh. kind APPLY_APPLIED: the solution is on the network,
    the load flow converged and the state was updated (lf_after, status_name, convergence_path,
    nx_refresh_ok and nx_contingencies_lost are set). APPLY_READBACK_EMPTY: nothing was applied
    (the generator read-back was empty). APPLY_LOAD_FLOW_FAILED: the network is reverted to the
    state before the solution and the state is untouched."""
    kind: str
    lf_after: Optional[dict] = None
    status_name: str = ""
    convergence_path: str = ""
    nx_refresh_ok: bool = True
    nx_contingencies_lost: Optional[int] = None
    seconds_load_flow: float = 0.0
    seconds_nx_lf: float = 0.0


def apply_solution_and_refresh(
    network, state: OperatingPointState, milp_model, pst_results: dict, rd_target_p_solution: dict,
    current_generator_data: dict, cost: float, nx_hooks: Optional[NxLoopHooks],
    use_reactive_limits: bool, extra_lf_params: Optional[dict], incremental_ac_fallback_max_steps: int,
    iteration: int, run_load_flow: Optional[Callable] = None, prepare_fallback: Optional[Callable] = None,
) -> AppliedSolution:
    """Apply the solved PST taps and RD set points to the network, run the AC load flow (direct
    jump, then the incremental fallback) and, once it converged, record the solution in the state
    and refresh the N-X load flow at the applied state. On a non-converging load flow the network
    is reverted. Used by the linearisation loop, by the single pass and by the DC pass. `cost` is
    the objective of this solve; `iteration` only labels log lines. run_load_flow(network) ->
    (converged, status_name, result): the calculation backend's load flow (default: the AC load flow).
    prepare_fallback(network, start_taps, start_target_p) -> bus voltage state or None
    (calculation_mode dc): called only when the direct jump failed. A DC load flow leaves every bus
    voltage NaN, which the warm-started fallback cannot start from; the hook returns the voltages
    of an AC load flow at the unmodified state (None: it did not converge, no fallback). Default
    None: the snapshot taken before the apply is used, as always."""
    import time

    # snapshot of the last converged state, for a possible revert
    iteration_start_tap_positions = _extract_phase_tap_positions(network)
    iteration_start_target_p = _extract_generator_target_p(network)
    iteration_start_voltage_state = _snapshot_bus_voltage_state(network)
    if rd_target_p_solution and not iteration_start_target_p:
        log.error(
            "[AC LOOP] iteration %d: the generator target_p read-back of the network is empty, so a "
            "failed load flow could not be reverted - stopping before the solution is applied", iteration,
        )
        return AppliedSolution(kind=APPLY_READBACK_EMPTY)

    _t = time.perf_counter()
    _apply_pst_results_to_network(network, pst_results)
    apply_preventive_solution_to_network(
        network, pst_tap_solution={}, rd_target_p_solution=rd_target_p_solution,
    )

    if run_load_flow is None:
        converged, status_name, lf_result = _run_ac_loadflow(
            network, use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params,
        )
    else:
        converged, status_name, lf_result = run_load_flow(network)
    convergence_path = "direct jump converged"
    if not converged:
        log.warning(
            ">>> Post-RA load flow did not converge (%s) via the direct, single-jump apply — "
            "attempting the incremental fallback before reverting.",
            status_name,
        )
        if prepare_fallback is not None:
            fallback_voltage_state = prepare_fallback(
                network, iteration_start_tap_positions, iteration_start_target_p,
            )
            if fallback_voltage_state is None:
                log.warning(
                    ">>> No start voltages for the incremental fallback (the AC load flow at the unmodified "
                    "state did not converge) - reverting to the last converged state."
                )
                restore_network_state(
                    network, previous_taps=iteration_start_tap_positions, previous_target_p=iteration_start_target_p,
                )
                return AppliedSolution(
                    kind=APPLY_LOAD_FLOW_FAILED, status_name="no_start_voltages",
                    seconds_load_flow=time.perf_counter() - _t,
                )
            iteration_start_voltage_state = fallback_voltage_state
        fallback_succeeded, fallback_status_name, steps_used = _apply_changes_incrementally(
            network,
            start_tap_positions=iteration_start_tap_positions,
            start_target_p=iteration_start_target_p,
            start_voltage_state=iteration_start_voltage_state,
            # a non-converged solution cannot evaluate the grid: revert to the last
            # converged state (what the network held before this iteration)
            revert_tap_positions=iteration_start_tap_positions,
            revert_target_p=iteration_start_target_p,
            pst_results=pst_results,
            rd_target_p_solution=rd_target_p_solution,
            max_steps=incremental_ac_fallback_max_steps,
            use_reactive_limits=use_reactive_limits,
            extra_lf_params=extra_lf_params,
        )
        if fallback_succeeded:
            log.info(
                "Incremental fallback converged in %d step(s) where the direct jump did not "
                "— treating this iteration as converged and proceeding.",
                steps_used,
            )
            converged, status_name = True, fallback_status_name
            convergence_path = f"incremental fallback ({steps_used} steps)"
        else:
            log.warning(
                ">>> Incremental fallback ALSO did not converge (%s, failed at/after step %d of "
                "%d) — reverting to the last converged state.",
                fallback_status_name, steps_used, incremental_ac_fallback_max_steps,
            )
            _restore_bus_voltage_state(network, iteration_start_voltage_state)
            return AppliedSolution(
                kind=APPLY_LOAD_FLOW_FAILED, status_name=fallback_status_name,
                seconds_load_flow=time.perf_counter() - _t,
            )
    seconds_load_flow = time.perf_counter() - _t

    # the load flow converged: the solution is part of the state now
    lf_after = cgmes_loader.build_lf_initial(network)
    readback_taps = _extract_phase_tap_positions(network)
    applied_taps = {
        mrid: int(round(float(readback_taps.get(mrid, row["result_tap_position"]))))
        for mrid, row in pst_results.items()
    }
    refreshed_target_p = _extract_generator_target_p(network)
    refreshed_target_p = {
        gu: refreshed_target_p.get(gu, current_generator_data.get(gu, {}).get("target_p", 0.0))
        for gu in current_generator_data
    }
    state.record_applied_solution(
        applied_tap_by_pst=applied_taps,
        refreshed_target_p_by_gu=refreshed_target_p,
        signed_bid_volume_delta_mw=_signed_bid_volume_delta(milp_model),
        lf_n0_after=lf_after,
    )
    state.annotate_latest_member(model=milp_model, pst_results=pst_results, cost=cost)

    nx_refresh_ok = True
    nx_contingencies_lost = None
    seconds_nx_lf = 0.0
    if nx_hooks is not None and state.current_lf_nx is not None:
        _t = time.perf_counter()
        refreshed_lf_nx = nx_hooks.compute_lf_nx(network)
        seconds_nx_lf = time.perf_counter() - _t
        if refreshed_lf_nx:
            nx_contingencies_lost = len(set(state.current_lf_nx) - set(refreshed_lf_nx))
            state.current_lf_nx = refreshed_lf_nx
        else:
            nx_refresh_ok = False
            log.warning(
                "N-X load flow could not be evaluated at the applied state — keeping the "
                "previous N-X baseline rather than replacing it with an empty result."
            )
    return AppliedSolution(
        kind=APPLY_APPLIED, lf_after=lf_after, status_name=status_name, convergence_path=convergence_path,
        nx_refresh_ok=nx_refresh_ok, nx_contingencies_lost=nx_contingencies_lost,
        seconds_load_flow=seconds_load_flow, seconds_nx_lf=seconds_nx_lf,
    )


def build_model_at_state(
    state: OperatingPointState, nx_inputs: Optional[NxMilpInputs], *, pst_list, rd_list, ae_flat_list,
    timestamp, lf_before, pst_tap_bounds, pst_group_equality_pairs, pst_degrees_per_tap,
    penalty_reference_pst_pra, penalty_reference_pst_pra_per_tso, penalty_reference_rd_pra,
    penalty_reference_rd_pra_per_tso, penalty_cost_overload_n0, generator_data, bid_list, glsk,
    ra_groups, raart, max_ra_count_per_timestamp_tso, ra_limit_penalty,
    ra_limit_constraint_hard, min_loading_threshold_pct, pst_taping_discrete, max_taps_per_tso,
    max_taps_per_tso_overrides, scanned_element_constraint_hard, scanned_element_penalty_eur_per_ka,
    unit_costs, hvdc_inputs: Optional[HvdcMilpInputs] = None, ct_inputs: Optional[CtMilpInputs] = None,
):
    """Build the MILP at the operating point held by `state` (taps, target_p, load flows and
    sensitivities of the state). Returns (model, PSTs at the current taps, generator data at the
    current point). Used by the linearisation loop (every iteration) and by the single pass."""
    import dataclasses

    run_profile.mark("MILP build")
    current_pst_list = state.psts_at_current_taps(pst_list)
    current_generator_data = state.generator_data_at_current(generator_data)
    pst_inputs_this_iteration = PstMilpInputs(
        pst_list=current_pst_list,
        pst_tap_bounds=pst_tap_bounds,
        pst_group_equality_pairs=pst_group_equality_pairs,
        sensitivity_pst_n0_pruned=state.sensitivity_pst_n0,
        pst_degrees_per_tap=pst_degrees_per_tap,
        penalty_reference_pst_pra=penalty_reference_pst_pra,
        penalty_reference_pst_pra_per_tso=penalty_reference_pst_pra_per_tso,
        pst_taping_discrete=pst_taping_discrete,
        max_taps_per_tso=max_taps_per_tso,
        max_taps_per_tso_overrides=max_taps_per_tso_overrides,
        pst_original_taps=dict(state.original_tap_by_pst),
    ) if pst_list else None

    rd_inputs_this_iteration = RdMilpInputs(
        rd_list=rd_list,
        bid_list=bid_list,
        glsk=glsk,
        sensitivity_rd_n0_pruned=state.sensitivity_rd_n0,
        generator_data=current_generator_data,
        penalty_reference_rd_pra=penalty_reference_rd_pra,
        penalty_reference_rd_pra_per_tso=penalty_reference_rd_pra_per_tso,
        max_ra_count_per_timestamp_tso=max_ra_count_per_timestamp_tso,
        ra_limit_penalty=ra_limit_penalty,
        ra_limit_constraint_hard=ra_limit_constraint_hard,
        ra_groups=ra_groups,
        raart=raart,
        cumulative_signed_volume_mw=dict(state.cumulative_signed_volume_mw),
        original_target_p_mw=dict(state.original_target_p_by_gu),
        unit_costs=unit_costs,
        # countertrade: the weights are fixed (computed once from the original set points) and identical in every build
        **({
            "bid_participation": ct_inputs.bid_participation,
            "penalty_reference_ct_pra": ct_inputs.penalty_reference_pra,
            "penalty_reference_ct_cra": ct_inputs.penalty_reference_cra,
        } if ct_inputs is not None else {}),
    ) if rd_list else None

    nx_inputs_this_iteration = (
        dataclasses.replace(
            nx_inputs,
            lf_nx=state.current_lf_nx,
            sensitivity_pst_nx=state.sensitivity_pst_nx,
            sensitivity_rd_nx=state.sensitivity_rd_nx,
            sensitivity_hvdc_nx=state.sensitivity_hvdc_nx,
        ) if nx_inputs is not None else None
    )

    # HVDC: the set points of the ORIGINAL network and of the operating point, and the sensitivities of the state
    hvdc_inputs_this_iteration = None
    if hvdc_inputs is not None:
        unit_eis = sorted({u.ei_mrid for u in hvdc_inputs.units})
        hvdc_inputs_this_iteration = dataclasses.replace(
            hvdc_inputs,
            original_p_mw={ei: float(state.original_target_p_by_gu[ei]) for ei in unit_eis},
            current_p_mw={ei: float(state.current_target_p_by_gu[ei]) for ei in unit_eis},
            sensitivity_hvdc_n0=state.sensitivity_hvdc_n0,
            zero_units_nx=state.hvdc_zero_units_nx,
        )

    milp_model = build_n0_milp_model(
        ae_flat_list=ae_flat_list,
        timestamp=timestamp,
        lf_initial=lf_before,
        penalty_cost_overload_n0=penalty_cost_overload_n0,
        pst_inputs=pst_inputs_this_iteration,
        rd_inputs=rd_inputs_this_iteration,
        nx_inputs=nx_inputs_this_iteration,
        min_loading_threshold_pct=min_loading_threshold_pct,
        scanned_element_constraint_hard=scanned_element_constraint_hard,
        scanned_element_penalty_eur_per_ka=scanned_element_penalty_eur_per_ka,
        lf_original=state.original_lf_n0,
        **({"hvdc_inputs": hvdc_inputs_this_iteration} if hvdc_inputs_this_iteration is not None else {}),
    )

    # what this model was built on, kept so the N-X loadings export can show the
    # linearised prediction next to the actual AC result of the applied solution
    milp_model.nx_build_baseline = (
        {
            "lf_nx": state.current_lf_nx,
            "sensitivity_pst_nx": state.sensitivity_pst_nx,
            "sensitivity_rd_nx": state.sensitivity_rd_nx,
        } if nx_inputs is not None else None
    )
    return milp_model, current_pst_list, current_generator_data


def run_post_rao_ac_verification_loop(
    network,
    pst_list: list[PST],
    rd_list: list[RDUnit],
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_initial: dict[str, dict[int, float]],
    pst_tap_bounds: dict[str, tuple[int, int]],
    sensitivity_pst_n0: dict[tuple[str, str], float],
    sensitivity_rd_n0: dict[tuple[str, str], float],
    pst_group_equality_pairs: list[tuple[str, str]],
    penalty_reference_pst_pra: float,
    penalty_reference_rd_pra: float,
    penalty_cost_overload_n0: float,
    pst_degrees_per_tap: dict[str, float],
    generator_data: dict[str, dict],
    bid_list: list = None,
    glsk: dict[str, dict[str, float]] = None,
    ra_groups: Optional[dict[str, dict]] = None,
    raart: Optional[str] = None,
    penalty_reference_pst_pra_per_tso: Optional[dict[str, float]] = None,
    penalty_reference_rd_pra_per_tso: Optional[dict[str, float]] = None,
    max_ra_count_per_timestamp_tso: Optional[dict[str, int]] = None,
    ra_limit_penalty: float = 0.0,
    ra_limit_constraint_hard: bool = True,
    min_loading_threshold_pct: float = 0.0,
    pst_taping_discrete: bool = True,
    max_taps_per_tso: int = 6,
    max_taps_per_tso_overrides: Optional[dict[str, int]] = None,
    max_change_ra_sum_threshold: float = 1.0,
    max_rd_mw_change_threshold: float = 1.0,
    max_change_ra: float = 1.0,
    max_change_overload: float = 1.0,
    max_linearisation_iterations: int = 10,
    output_path: Optional[Path] = None,
    sensitivity_threshold_pct_pst: float = 0.0,
    sensitivity_threshold_pct_rd: float = 0.0,
    scanned_element_constraint_hard: bool = True,
    scanned_element_penalty_eur_per_ka: float = 0.0,
    nx_inputs: Optional[NxMilpInputs] = None,
    use_reactive_limits: bool = False,
    extra_lf_params: dict = None,
    incremental_ac_fallback_max_steps: int = 5,
    iteration_records: Optional[list[dict]] = None,
    state: Optional[OperatingPointState] = None,
    nx_hooks: Optional[NxLoopHooks] = None,
    completeness_iteration: int = 0,
    lf_post_iteration_records: Optional[list[dict]] = None,
    unit_costs: Optional[dict] = None,
    hvdc_inputs: Optional[HvdcMilpInputs] = None,
    ct_inputs: Optional[CtMilpInputs] = None,
) -> tuple[dict[str, dict[int, float]], object, dict[str, dict[str, object]], str]:
    """The linearisation loop on one operating-point state (operating_point_state.py).

    Each iteration: 1. build the MILP at the CURRENT point (taps, target_p, N-0/N-X load
    flows, sensitivities; tap limits, RD headroom, RA count and costs act on the CUMULATIVE
    change against the original network); 2. solve; 3. apply and run the load flow (direct
    jump, then the incremental fallback); 4. if it converged, update the state and only
    then run the convergence check, so the last solution is always applied.

    Stops without a new applied solution on a non-converging load flow (network reverted:
    REVERTED_NON_CONVERGENT, or REVERTED_TO_LAST_CONVERGED_STATE when something was applied),
    SENSITIVITY_INVALID or SOLVER_NON_OPTIMAL; with nothing applied the latest model
    describes the original network (describe_original_network).

    Converged when ANY criterion holds (recorded in results_iteration_history.csv):
      1+2. sum|tap change| < max_change_ra_sum_threshold AND sum|delta MW per unit| <
           max_rd_mw_change_threshold (one value, recorded as criterion_1_met);
      3. max|tap change| < max_change_ra AND max|delta MW| < max_change_ra;
      4. sum_e |overload_e(k) - overload_e(k-1)| < max_change_overload (A), overload_e =
         max(0, I_e - limit_e) on the AC state (N-X: plus the curative effect);
      max_linearisation_iterations: hard cap;
      cyclic repetition: the cumulative solution after k >= 2 is within max_change_ra of
         the one after some iteration 0..k-2 (CYCLIC_REPETITION; checked last).

    Returns (current N-0 load flow, final model, PST rows against the ORIGINAL network,
    termination_reason). state: shared with the completeness loop (created when None).
    nx_hooks: None when N-X is not enabled.
    """
    import time

    import pyomo.environ as pyo

    branch_details = _get_monitored_branch_details(ae_flat_list, timestamp)
    if not branch_details:
        raise ValueError("No monitored branches found for timestamp %s" % timestamp)

    bid_list = bid_list or []
    glsk = glsk or {}

    if state is None:
        state = OperatingPointState.at_original_network(
            pst_list=pst_list, generator_data=generator_data, lf_n0=lf_initial,
            lf_nx=nx_inputs.lf_nx if nx_inputs is not None else None,
            sensitivity_pst_n0=sensitivity_pst_n0, sensitivity_rd_n0=sensitivity_rd_n0,
            sensitivity_pst_nx=nx_inputs.sensitivity_pst_nx if nx_inputs is not None else None,
            sensitivity_rd_nx=nx_inputs.sensitivity_rd_nx if nx_inputs is not None else None,
            monitored_branches_n0=set(branch_details),
            sensitivity_hvdc_n0=({} if hvdc_inputs is not None else None),
            sensitivity_hvdc_nx=({} if hvdc_inputs is not None else None),
        )
    own_lf_post_records = lf_post_iteration_records is None
    lf_post_iteration_records = [] if own_lf_post_records else lf_post_iteration_records

    final_model = None
    final_pst_results: dict[str, dict[str, object]] = {}
    last_applied_model = None
    last_applied_pst_results: dict[str, dict[str, object]] = {}
    termination_reason = ""
    # the model built in the latest iteration (also when its solution was not applied)
    latest_model = None

    for iteration in range(1, max_linearisation_iterations + 1):
        log.info("AC verification iteration %d starting (completeness iteration %d)", iteration, completeness_iteration)
        timings = {"build_and_solve": 0.0, "load_flow": 0.0, "n0_sensitivity": 0.0, "nx_lf": 0.0, "nx_sensitivity": 0.0}
        lf_before = state.current_lf_n0
        # the AC state before this iteration (criterion 4 compares against it)
        lf_nx_before = state.current_lf_nx
        curative_effect_before = dict(state.last_curative_effect_ampere)

        # 1. refresh the sensitivities that belong to the current operating point
        if state.sensitivities_stale:
            allowed_branches = set(branch_details)
            allowed_psts = {str(pst.mrid) for pst in pst_list if pst.available}
            allowed_gus = {str(rd.gu_mrid) for rd in rd_list if rd.available}
            allowed_hvdc = {u.ei_mrid for u in hvdc_inputs.units} if hvdc_inputs is not None else None
            _t = time.perf_counter()
            failure = _refresh_n0_sensitivities(
                state, network, pst_list, rd_list, ae_flat_list, timestamp,
                sensitivity_threshold_pct_pst, sensitivity_threshold_pct_rd,
                use_reactive_limits, extra_lf_params,
                allowed=(allowed_branches, allowed_psts, allowed_gus),
                hvdc_inputs=hvdc_inputs,
            )
            timings["n0_sensitivity"] = time.perf_counter() - _t
            if failure is None and nx_hooks is not None and (
                state.sensitivity_pst_nx or state.sensitivity_rd_nx or (hvdc_inputs is not None and state.sensitivity_hvdc_nx)
            ):
                _t = time.perf_counter()
                fresh_pst, fresh_rd = nx_hooks.compute_sensitivities_nx(network, state.current_lf_nx)
                timings["nx_sensitivity"] = time.perf_counter() - _t
                # an invalid set never reaches the builder: the last valid one of the contingency replaces it
                hvdc_nx_result: dict = {}
                hvdc_kwargs = (
                    dict(fresh_hvdc=state.fresh_sensitivity_hvdc_nx or {}, allowed_hvdc=allowed_hvdc, hvdc_result=hvdc_nx_result)
                    if hvdc_inputs is not None else {}
                )
                pst_nx, rd_nx, source_nx, failure = resolve_nx(
                    fresh_pst, fresh_rd, state.invalid_sensitivity_nx, state.current_lf_nx,
                    state.last_valid_sensitivity_nx, allowed_branches, allowed_psts, allowed_gus, **hvdc_kwargs,
                )
                if failure is None:
                    state.sensitivity_pst_nx, state.sensitivity_rd_nx = pst_nx, rd_nx
                    if hvdc_inputs is not None:
                        state.sensitivity_hvdc_nx = dict(hvdc_nx_result)
                        state.hvdc_zero_units_nx = dict(state.fresh_hvdc_zero_units_nx)
                    state.sensitivity_source_nx = source_nx
                    for co, kind in source_nx.items():
                        if kind == STALE_REUSED:
                            state.sensitivity_source_nx_ever[co] = STALE_REUSED
                        else:
                            state.sensitivity_source_nx_ever.setdefault(co, kind)
            if failure is not None:
                state.sensitivity_failure = failure
                log.error("SENSITIVITY GUARD: stopping the loop - %s", failure)
                termination_reason = SENSITIVITY_INVALID_REASON
                break
            state.sensitivities_stale = False

        # 2. build + solve at the current operating point
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

        latest_model = milp_model
        try:
            pst_results, overload_slack = solve_n0_milp_model(
                milp_model=milp_model,
                pst_list=current_pst_list,
                timestamp=timestamp,
                penalty_reference_pst_pra=penalty_reference_pst_pra,
                penalty_reference_pst_pra_per_tso=penalty_reference_pst_pra_per_tso,
            )
        except MilpNotOptimalError as exc:
            timings["build_and_solve"] = time.perf_counter() - _t
            log.error("[AC LOOP] iteration %d: %s - stopping with the last applied state", iteration, exc)
            state.solve_history.append({
                "solve_index": len(state.solve_history) + 1, "completeness_iteration": completeness_iteration,
                "linearisation_iteration": iteration, "load_flow_converged": False,
                "solution_applied": False, "converged": False, **_runtime_columns(timings),
            })
            _log_runtime(timings, iteration, completeness_iteration)
            termination_reason = SOLVER_NON_OPTIMAL_REASON
            break
        timings["build_and_solve"] = time.perf_counter() - _t
        rd_target_p_solution = extract_rd_target_p_solution(milp_model, current_generator_data)
        # the HVDC set points travel in the same dict (apply, restore, fallback ramp, criteria 1 and 3)
        rd_target_p_solution.update(extract_hvdc_target_p_solution(milp_model))
        hvdc_eis = frozenset(u.ei_mrid for u in hvdc_inputs.units) if hvdc_inputs is not None else frozenset()

        total_tap_change_sum = sum(abs(float(row["tap_change_result"])) for row in pst_results.values())
        total_rd_mw_change = sum(
            abs(new_target_p - float(current_generator_data.get(gu_mrid, {}).get("target_p", 0.0)))
            for gu_mrid, new_target_p in rd_target_p_solution.items()
        )

        final_model = milp_model
        final_pst_results = pst_results
        pst_converged = total_tap_change_sum < float(max_change_ra_sum_threshold)
        rd_converged = total_rd_mw_change < float(max_rd_mw_change_threshold)

        # criterion 3: per-RA movement threshold
        max_individual_pst_change = max(
            (abs(float(row["tap_change_result"])) for row in pst_results.values()), default=0.0,
        )
        max_individual_rd_change = max(
            (
                abs(new_target_p - float(current_generator_data.get(gu_mrid, {}).get("target_p", 0.0)))
                for gu_mrid, new_target_p in rd_target_p_solution.items()
            ),
            default=0.0,
        )
        criterion_3_converged = (
            max_individual_pst_change < float(max_change_ra)
            and max_individual_rd_change < float(max_change_ra)
        )

        # criteria 1 and 3 are fixed by the solution itself; criterion 4 needs the AC state
        # AFTER the solution was applied and is evaluated in step 5
        criterion_1_met = pst_converged and rd_converged
        total_overload_this_iteration = (
            sum(pyo.value(milp_model.overload_slack_n0[b]) for b in milp_model.secured_branch_ids)
            + (sum(pyo.value(milp_model.nx_patl_overload_slack[p]) for p in milp_model.nx_branch_contingency_pairs)
               if hasattr(milp_model, "nx_patl_overload_slack") else 0.0)
            + (sum(pyo.value(milp_model.nx_tatl_overload_slack[p]) for p in milp_model.nx_tatl_branch_contingency_pairs)
               if hasattr(milp_model, "nx_tatl_overload_slack") else 0.0)
        )  # diagnostic only (the model's own slack); NOT a convergence criterion any more

        solve_record = {
            "solve_index": len(state.solve_history) + 1,
            "completeness_iteration": completeness_iteration,
            "linearisation_iteration": iteration,
            "objective_value": float(pyo.value(milp_model.objective)),
            "sum_abs_tap_change": total_tap_change_sum,
            "sum_abs_rd_change_mw": total_rd_mw_change,
            "total_overload": total_overload_this_iteration,
            "max_abs_tap_change": max_individual_pst_change,
            "max_abs_rd_change_mw": max_individual_rd_change,
            "load_flow_converged": False,
            "solution_applied": False,
            "converged": False,
            "criterion_1_met": criterion_1_met,
            "criterion_3_met": criterion_3_converged,
            "overload_change_ampere": None,
            "criterion_4_met": False,
            "cycle_match_iteration": None,
            "cycle_selected_iteration": None,
            "converged_by": "",
            "new_constraints_added": 0,
            # contingencies the N-X load flow refresh after this solve lost (kept: see below)
            "nx_contingencies_lost_at_refresh": 0,
            # sensitivity guard: what the MILP of this solve was built from
            "invalid_sensitivity_contingencies": len(state.invalid_sensitivity_nx),
            "invalid_sensitivity_branches_n0": len(state.invalid_sensitivity_n0),
            # ... and how many of those were replaced by the last valid set
            "stale_reused_sensitivity_contingencies": sum(
                1 for kind in state.sensitivity_source_nx.values() if kind == STALE_REUSED
            ),
            "stale_reused_sensitivity_branches_n0": len(state.stale_reused_n0),
        }

        # 3. APPLY and 4. record the converged solution (shared with the single pass)
        applied = apply_solution_and_refresh(
            network=network, state=state, milp_model=milp_model, pst_results=pst_results,
            rd_target_p_solution=rd_target_p_solution, current_generator_data=current_generator_data,
            cost=solve_record["objective_value"], nx_hooks=nx_hooks,
            use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params,
            incremental_ac_fallback_max_steps=incremental_ac_fallback_max_steps, iteration=iteration,
        )
        timings["load_flow"] = applied.seconds_load_flow
        timings["nx_lf"] = applied.seconds_nx_lf
        if applied.kind == APPLY_READBACK_EMPTY:
            earlier_solution_applied = any(r["solution_applied"] for r in state.solve_history)
            state.solve_history.append({**solve_record, **_runtime_columns(timings)})
            _log_runtime(timings, iteration, completeness_iteration)
            termination_reason = REVERTED_TO_LAST_REASON if earlier_solution_applied else NON_CONVERGENT_REASON
            break
        if applied.kind == APPLY_LOAD_FLOW_FAILED:
            earlier_solution_applied = any(r["solution_applied"] for r in state.solve_history)
            _log_iteration_summary(
                "AC LOOP", iteration, max_linearisation_iterations, milp_model,
                lf_before, branch_details,
                pst_results, rd_target_p_solution,
                state.original_tap_by_pst, state.original_target_p_by_gu,
                "FAILED — reverted to " + ("last converged state" if earlier_solution_applied else "the original network"),
                hvdc_ei_mrids=hvdc_eis,
            )
            # the state is untouched: it still describes the last converged network
            lf_post_iteration_records.append({
                "ac_iteration": len(state.solve_history) + 1, "lf_before": lf_before, "lf_after": lf_before,
                "solution_applied_to_network": False,
            })
            state.solve_history.append({**solve_record, **_runtime_columns(timings)})
            _log_runtime(timings, iteration, completeness_iteration)
            termination_reason = REVERTED_TO_LAST_REASON if earlier_solution_applied else NON_CONVERGENT_REASON
            break
        lf_after = applied.lf_after
        status_name = applied.status_name
        convergence_path = applied.convergence_path
        nx_refresh_ok = applied.nx_refresh_ok
        last_applied_model, last_applied_pst_results = milp_model, pst_results
        solve_record["load_flow_converged"] = True
        solve_record["solution_applied"] = True
        if applied.nx_contingencies_lost is not None:
            solve_record["nx_contingencies_lost_at_refresh"] = applied.nx_contingencies_lost

        _log_iteration_summary(
            "AC LOOP", iteration, max_linearisation_iterations, milp_model,
            lf_after, branch_details,
            pst_results, rd_target_p_solution,
            state.original_tap_by_pst, state.original_target_p_by_gu,
            convergence_path, hvdc_ei_mrids=hvdc_eis,
        )
        log.info(
            "Iteration %d post-RA LF converged (%s)", iteration, status_name,
        )
        lf_post_iteration_records.append({
            "ac_iteration": len(state.solve_history) + 1, "lf_before": lf_before, "lf_after": lf_after,
            "solution_applied_to_network": True,
        })

        # 5. convergence check — only now, on a state that exists. Each criterion is an
        # independent acceptance criterion: the loop is converged when ANY of them holds.
        monitoring = getattr(milp_model, "overload_monitoring", {})
        curative_effect_now: dict[tuple, float] = {}
        if hasattr(milp_model, "nx_curative_effect_ampere"):
            # only the PATL rows hold the post-curative state (the TATL rows bound the
            # state before the curative actions)
            for branch_mrid, contingency_mrid in monitoring.get("nx_patl", {}):
                curative_effect_now[(branch_mrid, contingency_mrid)] = (
                    milp_model.nx_curative_effect_ampere(branch_mrid, contingency_mrid)
                )
        # criterion 4 (change of element overloads): AC state k against AC state k-1
        # (state 0 = the initial AC state), per element, in ampere
        overload_change = overload_change_ampere(
            element_overloads_ampere(monitoring, lf_after, state.current_lf_nx, curative_effect_now),
            element_overloads_ampere(monitoring, lf_before, lf_nx_before, curative_effect_before),
        )
        state.last_curative_effect_ampere = curative_effect_now
        state.annotate_latest_member(lf_nx=state.current_lf_nx, curative_effect=dict(curative_effect_now))
        if not nx_refresh_ok:
            log.warning(
                "Criterion 4 not evaluated: the N-X load flow of the applied state is missing, so the N-X "
                "overloads of this state are unknown (an unchanged stale baseline is not a stabilised overload)."
            )
        criterion_4_met = nx_refresh_ok and overload_change < float(max_change_overload)
        solve_record["overload_change_ampere"] = overload_change
        solve_record["criterion_4_met"] = criterion_4_met
        met = [
            name for name, ok in (
                ("criterion_1", criterion_1_met), ("criterion_3", criterion_3_converged),
                ("criterion_4", criterion_4_met),
            ) if ok
        ]
        log.info(
            "[CONVERGENCE] iteration %d | criterion 1: sum|tap|=%.3f sum|RD MW|=%.3f (< %.3f / %.3f) %s | "
            "criterion 3: max|tap|=%.3f max|RD MW|=%.3f (< %.3f) %s | criterion 4: overload change=%.3f A (< %.3f A) %s",
            iteration, total_tap_change_sum, total_rd_mw_change, float(max_change_ra_sum_threshold),
            float(max_rd_mw_change_threshold), "MET" if criterion_1_met else "no",
            max_individual_pst_change, max_individual_rd_change, float(max_change_ra),
            "MET" if criterion_3_converged else "no",
            overload_change, float(max_change_overload), "MET" if criterion_4_met else "no",
        )
        if met:
            solve_record["converged"] = True
            solve_record["converged_by"] = "+".join(met)
            state.solve_history.append({**solve_record, **_runtime_columns(timings)})
            _log_runtime(timings, iteration, completeness_iteration)
            termination_reason = "remedial action movement and congestion levels stabilized between iterations"
            break

        # cyclic repetition: the new cumulative solution equals an earlier one (not the
        # previous iteration, which the change criteria cover)
        cycle = find_cyclic_repetition(state.solution_snapshots, float(max_change_ra))
        if cycle is not None:
            matched_iteration, distance = cycle
            solve_record["cycle_match_iteration"] = matched_iteration
            log.info(
                "Cyclic repetition: the solution after iteration %d equals the solution after iteration %d "
                "(maximum difference %.4f, below max_change_ra=%.4f) — stopping, keeping the best member of "
                "the cycle.",
                len(state.solution_snapshots) - 1, matched_iteration, distance, float(max_change_ra),
            )
            cycle_exit = resolve_cyclic_repetition(
                state, network, matched_iteration, monitoring, nx_hooks, use_reactive_limits, extra_lf_params,
            )
            solve_record["cycle_selected_iteration"] = cycle_exit.selected_iteration
            final_model, final_pst_results = cycle_exit.model, cycle_exit.pst_results
            last_applied_model, last_applied_pst_results = cycle_exit.model, cycle_exit.pst_results
            state.solve_history.append({**solve_record, **_runtime_columns(timings)})
            _log_runtime(timings, iteration, completeness_iteration)
            # both re-applications failed: the network is back at the newest member's set-points
            termination_reason = REVERTED_TO_LAST_REASON if cycle_exit.reapply_failed else CYCLIC_REPETITION_REASON
            break
        state.solve_history.append({**solve_record, **_runtime_columns(timings)})
        _log_runtime(timings, iteration, completeness_iteration)
    else:
        termination_reason = "max iterations reached"

    if own_lf_post_records and output_path is not None:
        write_lf_post_iterations_csv(output_path, branch_details, lf_post_iteration_records, timestamp)
    if own_lf_post_records and iteration_records is not None:
        iteration_records.extend(
            {"ac_iteration": r["ac_iteration"], "solution_applied_to_network": r["solution_applied_to_network"]}
            for r in lf_post_iteration_records
        )

    log.info(
        "\n".join([
            "=" * 72,
            "AC verification loop terminated after %d iterations: %s" % (iteration, termination_reason),
            "=" * 72,
        ])
    )

    if termination_reason in _STOP_REASONS:
        # the result files describe only what is applied: the last applied solution of the
        # run, or the original network when nothing was applied at all
        final_model, final_pst_results = _last_applied_of_run(state, last_applied_model, last_applied_pst_results)
        if final_model is None and latest_model is not None:
            log.warning(
                "[AC LOOP] no solution was applied in this run - the result files describe the original network"
            )
            describe_original_network(latest_model)
            final_model = latest_model
            final_pst_results = _zero_change_pst_rows(
                latest_model, pst_list, state, penalty_reference_pst_pra, penalty_reference_pst_pra_per_tso,
            )

    if final_model is None:
        reported_pst_results = {}
    elif getattr(final_model, "no_applied_solution", False):
        reported_pst_results = final_pst_results
    else:
        reported_pst_results = _cumulative_pst_result_rows(final_pst_results, state)
    return state.current_lf_n0, final_model, reported_pst_results, termination_reason


def _runtime_columns(timings: dict) -> dict:
    return {
        "seconds_build_and_solve": round(timings["build_and_solve"], 3),
        "seconds_load_flow": round(timings["load_flow"], 3),
        "seconds_n0_sensitivity": round(timings["n0_sensitivity"], 3),
        "seconds_nx_load_flow": round(timings["nx_lf"], 3),
        "seconds_nx_sensitivity": round(timings["nx_sensitivity"], 3),
    }


def _log_runtime(timings: dict, iteration: int, completeness_iteration: int) -> None:
    log.info(
        "[RUNTIME] completeness iteration %d, linearisation iteration %d: build+solve %.2fs | "
        "AC load flow %.2fs | N-0 sensitivity %.2fs | N-X load flow %.2fs | N-X sensitivity %.2fs",
        completeness_iteration, iteration, timings["build_and_solve"], timings["load_flow"],
        timings["n0_sensitivity"], timings["nx_lf"], timings["nx_sensitivity"],
    )


def _refresh_n0_sensitivities(
    state: OperatingPointState, network, pst_list, rd_list, ae_flat_list, timestamp,
    sensitivity_threshold_pct_pst, sensitivity_threshold_pct_rd, use_reactive_limits,
    extra_lf_params=None, allowed=None, hvdc_inputs: Optional[HvdcMilpInputs] = None,
) -> Optional[str]:
    """Refreshes the N-0 sensitivities at the applied state; an invalid branch gets its last valid
    set (PST and RD). Returns None, or the failure message that must stop the loop."""
    import pypowsybl as pp
    # the pipeline's load-flow parameters, warm-started from the applied state's converged
    # load flow (a flat start can fail at the extreme point the incremental fallback reached)
    refresh_extra_lf_params = {
        **(extra_lf_params or {}), "voltage_init_mode": pp.loadflow.VoltageInitMode.PREVIOUS_VALUES,
    }
    guard_pst: dict = {}
    guard_rd: dict = {}
    # only PREVENTIVE PSTs have an N-0 sensitivity matrix
    if any(pst.kind == RAKind.PREVENTIVE for pst in pst_list):
        state.sensitivity_pst_n0 = compute_sensitivity_pst_n0(
            network=network, pst_list=pst_list, ae_flat_list=ae_flat_list,
            timestamp=timestamp, lf_initial=state.current_lf_n0,
            sensitivity_threshold_pct=sensitivity_threshold_pct_pst,
            use_reactive_limits=use_reactive_limits, extra_lf_params=refresh_extra_lf_params,
            guard_out=guard_pst,
        )
    # the N-0 RD sensitivity is preventive-only
    if any(rd.available and rd.kind == RAKind.PREVENTIVE for rd in rd_list):
        state.sensitivity_rd_n0 = compute_sensitivity_rd_n0(
            network=network, rd_list=rd_list, ae_flat_list=ae_flat_list,
            timestamp=timestamp, lf_initial=state.current_lf_n0,
            sensitivity_threshold_pct=sensitivity_threshold_pct_rd,
            use_reactive_limits=use_reactive_limits, extra_lf_params=refresh_extra_lf_params,
            guard_out=guard_rd,
        )
    guard_hvdc: dict = {}
    hvdc_eis = sorted({u.ei_mrid for u in hvdc_inputs.preventive_units}) if hvdc_inputs is not None else []
    if hvdc_eis:
        state.sensitivity_hvdc_n0 = compute_sensitivity_hvdc_n0(
            network, hvdc_eis, ae_flat_list, timestamp, lf_initial=state.current_lf_n0,
            sensitivity_threshold_pct=sensitivity_threshold_pct_rd,
            use_reactive_limits=use_reactive_limits, extra_lf_params=refresh_extra_lf_params, guard_out=guard_hvdc,
        )
    state.invalid_sensitivity_n0 = {**guard_pst, **guard_rd, **guard_hvdc}
    log_invalid("N-0", state.invalid_sensitivity_n0, "after the refresh at the applied state")
    if allowed is None:
        return None
    hvdc_n0_result: dict = {}
    hvdc_kwargs = (
        dict(fresh_hvdc=state.sensitivity_hvdc_n0, store_hvdc=state.last_valid_sensitivity_n0_hvdc,
             allowed_hvdc={u.ei_mrid for u in hvdc_inputs.units}, hvdc_result=hvdc_n0_result)
        if hvdc_inputs is not None else {}
    )
    pst, rd, stale, failure = resolve_n0(
        state.sensitivity_pst_n0, state.sensitivity_rd_n0, state.invalid_sensitivity_n0,
        allowed[0], state.last_valid_sensitivity_n0_pst, state.last_valid_sensitivity_n0_rd, *allowed, **hvdc_kwargs,
    )
    if failure is None:
        state.sensitivity_pst_n0, state.sensitivity_rd_n0, state.stale_reused_n0 = pst, rd, stale
        if hvdc_inputs is not None:
            state.sensitivity_hvdc_n0 = dict(hvdc_n0_result)
    return failure


@dataclass
class CycleExit:
    selected_iteration: int      # snapshot index kept (= number of applied solves of that state)
    restored: bool               # True iff the network was moved back to an earlier member
    model: object                # solved model of the kept member (result files)
    pst_results: dict            # PST rows of the kept member
    reapply_failed: bool = False  # True iff re-applying the newest member also failed to converge


def select_best_cycle_member(candidates: list[tuple[int, float, float]]) -> int:
    """candidates: (snapshot index, total AC overload in A, cumulative cost). Lowest overload,
    ties (to 1e-9 A) by lowest cumulative cost (to 1e-6), remaining ties by the latest member."""
    return min(
        candidates, key=lambda c: (round(c[1], _TIE_DIGITS_OVERLOAD_AMPERE), round(c[2], _TIE_DIGITS_COST_EUR), -c[0]),
    )[0]


def _reapply_cumulative_solution(network, state: OperatingPointState, index: int, use_reactive_limits, extra_lf_params):
    snapshot = state.solution_snapshots[index]
    _apply_pst_results_to_network(
        network, {mrid: {"result_tap_position": tap} for mrid, tap in snapshot["tap"].items()},
    )
    apply_preventive_solution_to_network(
        network, pst_tap_solution={},
        rd_target_p_solution={
            gu: state.original_target_p_by_gu[gu] + float(net) for gu, net in snapshot["net_rd_mw"].items()
        },
    )
    return _run_ac_loadflow(network, use_reactive_limits=use_reactive_limits, extra_lf_params=extra_lf_params)


def resolve_cyclic_repetition(
    state: OperatingPointState, network, matched_iteration: int, monitoring: dict,
    nx_hooks: Optional[NxLoopHooks], use_reactive_limits: bool = False, extra_lf_params: Optional[dict] = None,
) -> CycleExit:
    """Cyclic-repetition exit: among the cycle members (the matched earlier solution j through the
    newest k, inclusive) keep the one with the lowest total AC overload (criterion-4 overload summed
    over the monitored elements, evaluated at each member's own AC state), ties by the lowest
    cumulative cost. If it is not the newest member, its cumulative state is re-applied to the
    network, the load flow is run, and the state object describes it.

    The original network (snapshot 0) is not a candidate when j = 0: the newest member repeats it
    within max_change_ra and is the member that has a solved model to report."""
    newest = len(state.solution_snapshots) - 1
    candidates = []
    for index in range(max(int(matched_iteration), 1), newest + 1):
        record = state.member_records[index]
        overload = sum(element_overloads_ampere(
            monitoring, record["lf_n0"], record["lf_nx"], record["curative_effect"],
        ).values())
        candidates.append((index, float(overload), float(record["cost"])))
    selected = select_best_cycle_member(candidates)
    log.info(
        "Cyclic repetition: members (iteration, total AC overload A, cumulative cost) %s — keeping iteration %d",
        [(i, round(o, 3), round(c, 2)) for i, o, c in candidates], selected,
    )
    newest_record = state.member_records[newest]
    if selected == newest:
        return CycleExit(newest, False, newest_record["model"], newest_record["pst_results"])

    # voltages of the newest member (its load flow converged), restored if a re-application fails
    newest_voltage_state = _snapshot_bus_voltage_state(network)
    converged, status_name, _ = _reapply_cumulative_solution(network, state, selected, use_reactive_limits, extra_lf_params)
    if not converged:
        log.warning(
            "Cyclic repetition: the load flow of the selected iteration %d did not converge (%s) — "
            "keeping the newest iteration %d.", selected, status_name, newest,
        )
        _restore_bus_voltage_state(network, newest_voltage_state)
        converged_again, status_again, _ = _reapply_cumulative_solution(
            network, state, newest, use_reactive_limits, extra_lf_params,
        )
        if not converged_again:
            log.error(
                "Cyclic repetition: re-applying the newest iteration %d did not converge either (%s) — "
                "set-points of iteration %d restored with its voltage snapshot; stopping.", newest, status_again, newest,
            )
            _restore_bus_voltage_state(network, newest_voltage_state)
            return CycleExit(newest, False, newest_record["model"], newest_record["pst_results"], reapply_failed=True)
        return CycleExit(newest, False, newest_record["model"], newest_record["pst_results"])

    record = state.member_records[selected]
    lf_nx = None
    if nx_hooks is not None and state.current_lf_nx is not None:
        lf_nx = nx_hooks.compute_lf_nx(network)
        if not lf_nx:
            log.warning(
                "N-X load flow could not be evaluated at the selected iteration %d — using the one "
                "recorded when that state was first reached.", selected,
            )
            lf_nx = record["lf_nx"]
    state.rewind_to_member(selected, cgmes_loader.build_lf_initial(network), lf_nx)
    return CycleExit(selected, True, record["model"], record["pst_results"])



def run_nested_rao_loop(
    max_completeness_iterations: int,
    state: OperatingPointState,
    nx_hooks: Optional[NxLoopHooks],
    **loop_kwargs,
) -> tuple[dict, object, dict, str, list[dict]]:
    """Completeness (outage approximation) as the outer loop around the linearisation loop,
    on the same state: run the linearisation loop to convergence, then look for overloaded
    (branch, contingency) pairs the model does not constrain yet; none (or MaxCountCompIter)
    -> stop. Returns (N-0 LF, final model, PST rows, termination_reason, completeness log:
    row 0 plus one row per check, results_completeness_iterations.csv shape)."""
    completeness_log: list[dict] = []
    state.completeness_limit_reached = False
    lf_post_iteration_records: list[dict] = []
    output_path = loop_kwargs.pop("output_path", None)
    iteration_records = loop_kwargs.pop("iteration_records", None)
    lf_current = state.current_lf_n0
    final_model, pst_rows, reason = None, {}, ""
    for completeness_iteration in range(0, max_completeness_iterations + 1):
        lf_current, final_model, pst_rows, reason = run_post_rao_ac_verification_loop(
            state=state, nx_hooks=nx_hooks, completeness_iteration=completeness_iteration,
            lf_post_iteration_records=lf_post_iteration_records, **loop_kwargs,
        )
        total_pairs = _count_nx_pairs(final_model)
        if completeness_iteration == 0:
            # row 0 = the model as first built (the existing results_completeness_iterations.csv shape);
            # every later row is the outage-approximation check that followed a converged loop
            completeness_log.append({
                "completeness_iteration": 0, "new_constraints_added": 0, "total_nx_constraints": total_pairs,
            })
        if nx_hooks is None or reason in _STOP_REASONS:
            if completeness_iteration > 0:
                completeness_log.append({
                    "completeness_iteration": completeness_iteration,
                    "new_constraints_added": 0, "total_nx_constraints": total_pairs,
                })
            break
        new_pairs = nx_hooks.find_new_pairs(final_model, state.current_lf_nx)
        completeness_log.append({
            "completeness_iteration": completeness_iteration + 1,
            "new_constraints_added": new_pairs, "total_nx_constraints": total_pairs + new_pairs,
        })
        if state.solve_history:
            state.solve_history[-1]["new_constraints_added"] = new_pairs
        if new_pairs == 0:
            log.info(
                "\n".join(["=" * 72,
                           "Completeness converged after %d completeness iteration(s): no new congestions" % completeness_iteration,
                           "=" * 72])
            )
            break
        log.info(
            "[COMPLETENESS LOOP] iteration %d/%d | new XNECs found=%d | continuing at the applied state",
            completeness_iteration, max_completeness_iterations, new_pairs,
        )
    else:
        state.completeness_limit_reached = True
        log.warning(
            "max_completeness_iterations=%d reached without full convergence — new congestions "
            "may still exist that were never added", max_completeness_iterations,
        )
    if output_path is not None:
        write_lf_post_iterations_csv(
            output_path,
            _get_monitored_branch_details(loop_kwargs["ae_flat_list"], loop_kwargs["timestamp"]),
            lf_post_iteration_records, loop_kwargs["timestamp"],
        )
    if iteration_records is not None:
        iteration_records.extend(
            {"ac_iteration": r["ac_iteration"], "solution_applied_to_network": r["solution_applied_to_network"]}
            for r in lf_post_iteration_records
        )
    return lf_current, final_model, pst_rows, reason, completeness_log


def _count_nx_pairs(model) -> int:
    return sum(
        len(getattr(model, name))
        for name in ("nx_branch_contingency_pairs", "nx_tatl_branch_contingency_pairs", "nx_scanned_branch_contingency_pairs")
        if hasattr(model, name)
    )
