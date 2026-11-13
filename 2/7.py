"""post_ra_congestion_check.py

- find_new_congested_pairs: on the N-X load flow of the applied state, the (branch,
  contingency) pairs over their PATL / TATL / scanned limit that the model does not
  constrain yet (outage approximation, called by run_nested_rao_loop).
- the apply / restore / extract helpers and the AC load-flow helpers of the AC verification
  loop, including _apply_changes_incrementally (warm-started step-by-step fallback).
PST taps are applied with network.update_phase_tap_changers(id=..., tap=...).
"""

import logging
from typing import Optional

from data_model import AEType, AssessedElement, branch_monitoring_rules, pair_is_monitored, select_by_side
from pre_milp.lf_parameters import build_loadflow_parameters

log = logging.getLogger(__name__)

# a tap or set-point change smaller than this counts as no change
ZERO_MOVEMENT = 1e-9


def extract_rd_target_p_solution(model, generator_data: dict) -> dict[str, float]:
    """Returns dict[gu_mrid, new_target_p_mw] for every generating unit
    with a preventive rd_net_injection_g Expression on the model —
    target_p (from the CGM) plus the solved net preventive injection."""
    if not hasattr(model, "rd_net_injection_g"):
        return {}

    import pyomo.environ as pyo

    target_p_solution: dict[str, float] = {}
    for gu_mrid in model.gu_ids:
        target_p = float(generator_data.get(str(gu_mrid), {}).get("target_p", 0.0))
        net_injection = pyo.value(model.rd_net_injection_g[gu_mrid])
        target_p_solution[str(gu_mrid)] = target_p + net_injection
    return target_p_solution


def extract_hvdc_target_p_solution(model) -> dict[str, float]:
    """{EI mRID: new target_p (MW, generation sign)} of every preventive HVDC unit of the model: the set
    point of the operating point the model was built on plus the solved change. Merged into the dict
    extract_rd_target_p_solution returns, so apply, restore and the fallback ramp treat the EI like any
    generator. {} for a model without an HVDC block."""
    if not hasattr(model, "hvdc_change"):
        return {}

    import pyomo.environ as pyo

    return {
        info["ei_mrid"]: info["p_current_mw"] + float(pyo.value(model.hvdc_change[unit]))
        for unit, info in model.hvdc_info.items()
    }


# ---------------------------------------------------------------------------
# Apply / restore the solved preventive RA solution on the real network
# ---------------------------------------------------------------------------


def apply_preventive_solution_to_network(
    network,
    pst_tap_solution: dict[str, int],
    rd_target_p_solution: dict[str, float],
) -> tuple[dict[str, int], dict[str, float]]:
    """Apply the solved preventive PST taps and RD target_p values to the network (kwargs
    'tap' and 'target_p'); returns the previous values for restore_network_state."""
    previous_taps: dict[str, int] = {}
    if pst_tap_solution:
        current_ptc = network.get_phase_tap_changers()
        for pst_mrid, new_tap in pst_tap_solution.items():
            if pst_mrid not in current_ptc.index:
                log.warning(
                    "apply_preventive_solution_to_network: PST %s not found in "
                    "network.get_phase_tap_changers() — skipped",
                    pst_mrid,
                )
                continue
            previous_taps[pst_mrid] = int(current_ptc.loc[pst_mrid, "tap"])
            network.update_phase_tap_changers(id=pst_mrid, tap=int(new_tap))

    previous_target_p: dict[str, float] = {}
    if rd_target_p_solution:
        current_gens = network.get_generators()
        for gu_mrid, new_target_p in rd_target_p_solution.items():
            if gu_mrid not in current_gens.index:
                log.warning(
                    "apply_preventive_solution_to_network: generating unit %s not "
                    "found in network.get_generators() — skipped",
                    gu_mrid,
                )
                continue
            previous_target_p[gu_mrid] = float(current_gens.loc[gu_mrid, "target_p"])
            network.update_generators(id=gu_mrid, target_p=float(new_target_p))

    return previous_taps, previous_target_p


def restore_network_state(
    network,
    previous_taps: dict[str, int],
    previous_target_p: dict[str, float],
) -> None:
    """Restore PST taps and generator target_p to the given values."""
    for pst_mrid, tap in previous_taps.items():
        network.update_phase_tap_changers(id=pst_mrid, tap=int(tap))
    for gu_mrid, target_p in previous_target_p.items():
        network.update_generators(id=gu_mrid, target_p=float(target_p))


# ---------------------------------------------------------------------------
# AC load flow and incremental (warm-started) application, used by ac_convergence_loop
# ---------------------------------------------------------------------------


def _run_ac_loadflow(network, use_reactive_limits: bool = False, extra_lf_params: dict = None):
    """Run the AC load flow with the pipeline's parameters (lf_parameters); returns
    (converged, status name, result). Convergence is read from component 0."""
    try:
        import pypowsybl as pp
    except Exception as exc:
        raise ImportError("pypowsybl is required to run post-RA load flow") from exc

    lf_result = pp.loadflow.run_ac(
        network,
        parameters=build_loadflow_parameters(use_reactive_limits, extra_lf_params),
    )
    status = None
    try:
        status = getattr(lf_result[0], "status", None)
    except Exception:
        status = getattr(lf_result, "status", None)

    status_name = getattr(status, "name", None) or str(status)
    converged = "CONVERGED" in str(status_name).upper()
    return converged, status_name, lf_result


def _snapshot_bus_voltage_state(network) -> dict[str, tuple[float, float]]:
    """Every bus's (v_mag, v_angle) as a plain dict. After a failed AC solve the network's
    voltages are NaN, so a warm start (PREVIOUS_VALUES) needs this explicit snapshot of the
    last good state (pypowsybl's update_buses restores it exactly)."""
    buses = network.get_buses()
    return {
        str(bus_id): (float(row["v_mag"]), float(row["v_angle"]))
        for bus_id, row in buses.iterrows()
    }


def _restore_bus_voltage_state(network, voltage_state: dict[str, tuple[float, float]]) -> None:
    """Write a _snapshot_bus_voltage_state snapshot back to the network; NaN entries (buses
    of small islands without a solved voltage) are skipped."""
    ids, v_mags, v_angles = [], [], []
    for bus_id, (v_mag, v_angle) in voltage_state.items():
        if v_mag != v_mag or v_angle != v_angle:  # NaN check without importing math/numpy here
            continue
        ids.append(bus_id)
        v_mags.append(v_mag)
        v_angles.append(v_angle)
    if ids:
        network.update_buses(id=ids, v_mag=v_mags, v_angle=v_angles)


def _apply_changes_incrementally(
    network,
    start_tap_positions: dict,
    start_target_p: dict,
    start_voltage_state: dict,
    revert_tap_positions: dict,
    revert_target_p: dict,
    pst_results: dict,
    rd_target_p_solution: dict,
    max_steps: int,
    use_reactive_limits: bool,
    extra_lf_params: dict,
):
    """Fallback after a direct single-jump AC solve failed: ramp every changed PST tap and RD
    target_p together from the start state to the solution in max_steps equal steps, each
    load flow warm-started from the previous step (PREVIOUS_VALUES). Steps alone or a warm
    start alone do not converge on hard cases; together they do. Needs component_mode
    MAIN_SYNCHRONOUS (small islands have no voltage to warm-start from).

    The start state is restored first from start_tap_positions/start_target_p and the explicit
    voltage snapshot start_voltage_state (the live voltages are NaN after the failed jump).
    Returns (success, status_name, steps_used); on failure the network is set to
    revert_tap_positions/revert_target_p before returning.
    """
    import pypowsybl as pp

    # undo the failed direct jump: start taps/target_p and the voltage snapshot of that state
    restore_network_state(network, previous_taps=start_tap_positions, previous_target_p=start_target_p)
    _restore_bus_voltage_state(network, start_voltage_state)

    pst_targets = {
        pst_mrid: (
            float(start_tap_positions.get(pst_mrid, row["current_tap_position"])),
            float(row["result_tap_position"]),
        )
        for pst_mrid, row in pst_results.items()
        if abs(float(row["tap_change_result"])) > ZERO_MOVEMENT
    }
    rd_targets = {
        gu_mrid: (float(start_target_p.get(gu_mrid, 0.0)), float(new_target_p))
        for gu_mrid, new_target_p in rd_target_p_solution.items()
        if abs(float(new_target_p) - float(start_target_p.get(gu_mrid, 0.0))) > ZERO_MOVEMENT
    }

    if not pst_targets and not rd_targets:
        # nothing changed, so the failure is not caused by remedial-action movement
        log.warning(
            "Incremental fallback: no nonzero PST tap or RD target_p change found to ramp — "
            "the direct jump's non-convergence is not explained by any RA movement this "
            "fallback can address."
        )
        restore_network_state(network, previous_taps=revert_tap_positions, previous_target_p=revert_target_p)
        return False, "no_change_to_ramp", 0

    warm_start_params = dict(extra_lf_params or {})
    warm_start_params["voltage_init_mode"] = pp.loadflow.VoltageInitMode.PREVIOUS_VALUES

    for step in range(1, max_steps + 1):
        frac = step / max_steps
        intermediate_taps = {
            pst_mrid: int(round(start + frac * (end - start)))
            for pst_mrid, (start, end) in pst_targets.items()
        }
        intermediate_target_p = {
            gu_mrid: start + frac * (end - start)
            for gu_mrid, (start, end) in rd_targets.items()
        }
        apply_preventive_solution_to_network(
            network, pst_tap_solution=intermediate_taps, rd_target_p_solution=intermediate_target_p,
        )

        converged, status_name, _ = _run_ac_loadflow(
            network, use_reactive_limits=use_reactive_limits, extra_lf_params=warm_start_params,
        )
        if not converged:
            log.warning(
                "Incremental fallback: step %d/%d (fraction=%.2f) did not converge (%s) — "
                "aborting fallback, reverting to the caller's own revert baseline.",
                step, max_steps, frac, status_name,
            )
            restore_network_state(network, previous_taps=revert_tap_positions, previous_target_p=revert_target_p)
            return False, status_name, step

    log.info(
        "Incremental fallback: SUCCEEDED — all %d step(s) converged, reaching the full "
        "proposed PST/RD change via warm-started intermediate steps instead of one direct jump.",
        max_steps,
    )
    return True, status_name, max_steps


# ---------------------------------------------------------------------------
# New congested pairs at the applied state
# ---------------------------------------------------------------------------


def find_new_congested_pairs(
    post_ra_lf_nx: dict,
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    current_patl_pairs: set,
    current_tatl_pairs: set,
    current_scanned_pairs: set = frozenset(),
    tatl_data: Optional[dict] = None,
    lf_initial: Optional[dict] = None,
    skipped=None,
    lf_nx_original: Optional[dict] = None,
) -> tuple[set, set, set]:
    """Pairs (branch, contingency) whose current in post_ra_lf_nx exceeds the PATL, TATL or
    scanned limit and that are not yet constrained (current_*_pairs; the pairs the model evaluated,
    rows skipped as never binding included). The scanned limit of a pair is max(PATL, its ORIGINAL
    post-contingency current, lf_nx_original, same side); lf_initial is only used to find the scanned
    branches. A TATL counts only where the MILP builds a TATL constraint: a TATL strictly above the
    branch's PATL."""
    from milp.milp_builder import (
        _get_monitored_branch_limits_and_ae_types, _get_monitored_branch_sides, _get_scanned_patl_ampere,
        tatl_is_above_patl,
    )
    skipped = skipped or []
    branch_sides = _get_monitored_branch_sides(ae_flat_list)
    # PATL limits of SECURED branches only: a scanned branch is checked against its
    # effective limit below, never against its raw PATL.
    branch_limits = {
        str(ae.branch_mrid): float(ae.patl_ka.get(timestamp, 0.0)) * 1000.0
        for ae in ae_flat_list
        if ae.ae_type == AEType.SECURED and ae.branch_mrid and ae.patl_ka.get(timestamp) is not None
    }

    # scanned effective limits from the same resolver the MILP's scanned constraint uses
    scanned_branch_ae_types: dict[str, AEType] = {}
    scanned_patl_ampere: dict[str, float] = {}
    if lf_initial is not None:
        _, scanned_branch_ae_types = _get_monitored_branch_limits_and_ae_types(ae_flat_list, timestamp, lf_initial)
        scanned_patl_ampere = _get_scanned_patl_ampere(ae_flat_list, timestamp)

    new_patl_pairs: set[tuple[str, str]] = set()
    new_tatl_pairs: set[tuple[str, str]] = set()
    new_scanned_pairs: set[tuple[str, str]] = set()
    _, restricted_pairs = branch_monitoring_rules(ae_flat_list)         # NCP links to contingencies

    for contingency_mrid, per_branch_currents in post_ra_lf_nx.items():
        for branch_mrid, per_side_currents in per_branch_currents.items():
            if not pair_is_monitored(restricted_pairs, branch_mrid, contingency_mrid):
                continue
            side = branch_sides.get(branch_mrid, 1)
            post_ra_current = select_by_side(per_side_currents, side, branch_mrid)
            if post_ra_current is None:
                continue

            patl_ampere = branch_limits.get(branch_mrid)
            pair = (branch_mrid, contingency_mrid)
            if patl_ampere and patl_ampere > 0.0 and post_ra_current > patl_ampere and pair not in current_patl_pairs:
                new_patl_pairs.add(pair)

            if tatl_data and branch_mrid in tatl_data:
                tatl_ampere = float(tatl_data[branch_mrid]["tatl_limit_ampere"])
                tatl_is_constrained = patl_ampere is not None and tatl_is_above_patl(tatl_ampere, patl_ampere)
                if tatl_is_constrained and post_ra_current > tatl_ampere and pair not in current_tatl_pairs:
                    new_tatl_pairs.add(pair)

            if scanned_branch_ae_types.get(branch_mrid) == AEType.SCANNED:
                effective_limit_ampere = scanned_patl_ampere.get(branch_mrid)
                original_per_side = ((lf_nx_original or {}).get(contingency_mrid) or {}).get(branch_mrid)
                original_current = select_by_side(original_per_side, side, branch_mrid) if original_per_side else None
                if effective_limit_ampere is not None and original_current is not None:
                    effective_limit_ampere = max(effective_limit_ampere, float(original_current))
                if (
                    effective_limit_ampere and effective_limit_ampere > 0.0
                    and post_ra_current > effective_limit_ampere and pair not in current_scanned_pairs
                ):
                    new_scanned_pairs.add(pair)

    log.info(
        "find_new_congested_pairs: %d new PATL congestion(s), %d new TATL congestion(s), "
        "%d new scanned congestion(s) found post-RA (%d contingencies skipped during LF re-run: %s)",
        len(new_patl_pairs), len(new_tatl_pairs), len(new_scanned_pairs), len(skipped), skipped or "none",
    )

    return new_patl_pairs, new_tatl_pairs, new_scanned_pairs
