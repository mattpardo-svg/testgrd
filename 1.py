"""nx_reference_gap_probe.py - why does the sensitivity guard flag (branch, contingency) pairs?

Diagnostic for the owner, not part of the pipeline (nothing in src/ imports it). It runs the real pipeline (CGM, NCP, config, load
flow, N-X security analysis, N-X sensitivity analysis) up to the first N-X sensitivity computation and stops there. For every pair
the guard's deviation rule flags (reference current of the sensitivity analysis against the current of the security analysis, same
side as the model, same tolerance) it prints ONLY: a running number, the type of the contingency equipment, the security-analysis
status of the contingency, the branch type, both currents, the gaps, the loading, and whether a voltage-controlling transformer or a
generator at a reactive limit is in a substation of the branch. It then reruns only the flagged contingencies with the sensitivity
analysis given exactly the security analysis's load-flow and provider parameters, and prints the gap again, then a summary.
No id, name, EIC code or file name is ever printed (pipeline log output and exceptions are suppressed; only exception class names
are shown).

Run from the repository root, with the usual input/ folder (one cgm*.zip and one ncp*.zip) and the usual config:
    .venv\\Scripts\\python.exe mock_data\\tools\\nx_reference_gap_probe.py [--input DIR]
AC modes only (ac_single_pass, ac_iterative); in calculation_mode dc the script says so and stops.
"""
import argparse
import contextlib
import io
import logging
import math
import statistics
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src" / "rao"), str(REPO)]


class _StopProbe(Exception):
    pass


def _contingency_kind(network, equipment_mrids) -> str:
    kinds = set()
    for mrid in equipment_mrids:
        for accessor, label in (("get_lines", "line"), ("get_2_windings_transformers", "transformer"), ("get_tie_lines", "line"),
                                ("get_generators", "generator"), ("get_loads", "load")):
            try:
                if mrid in getattr(network, accessor)().index:
                    kinds.add(label)
                    break
            except Exception:
                continue
        else:
            kinds.add("other")
    return kinds.pop() if len(kinds) == 1 else ("multiple" if kinds else "other")


def _branch_kind(network, branch_mrid) -> str:
    for accessor, label in (("get_lines", "line"), ("get_2_windings_transformers", "transformer"), ("get_tie_lines", "tie line"),
                            ("get_dangling_lines", "dangling line")):
        try:
            if branch_mrid in getattr(network, accessor)().index:
                return label
        except Exception:
            continue
    return "other"


def _substations_of(network, branch_mrid) -> set:
    voltage_levels = network.get_voltage_levels()
    found = set()
    for accessor, columns in (("get_lines", ("voltage_level1_id", "voltage_level2_id")),
                              ("get_2_windings_transformers", ("voltage_level1_id", "voltage_level2_id"))):
        table = getattr(network, accessor)()
        if branch_mrid in table.index:
            for column in columns:
                found.add(voltage_levels.loc[table.loc[branch_mrid, column], "substation_id"])
    return found


def _substation_flags(network, substations) -> tuple:
    """(a voltage-controlling transformer, a generator at a reactive limit) in any of the substations."""
    voltage_levels = network.get_voltage_levels()
    controlling = at_limit = False
    try:
        ratio = network.get_ratio_tap_changers()
        transformers = network.get_2_windings_transformers()
        for mrid, row in ratio.iterrows():
            if bool(row.get("regulating", False)) and mrid in transformers.index:
                for column in ("voltage_level1_id", "voltage_level2_id"):
                    if voltage_levels.loc[transformers.loc[mrid, column], "substation_id"] in substations:
                        controlling = True
    except Exception:
        pass
    try:
        generators = network.get_generators()
        limits = network.get_minmax_reactive_limits()
        for mrid, row in generators.iterrows():
            if mrid not in limits.index or voltage_levels.loc[row["voltage_level_id"], "substation_id"] not in substations:
                continue
            generated = -float(row["q"])                                            # load convention -> generation
            low, high = float(limits.loc[mrid, "min_q"]), float(limits.loc[mrid, "max_q"])
            span = max(high - low, 1.0)
            if not math.isnan(generated) and (generated >= high - 0.01 * span or generated <= low + 0.01 * span):
                at_limit = True
    except Exception:
        pass
    return controlling, at_limit


def _sensitivity_references(network, contingencies, side_branches, parameters, variable_id):
    """{(contingency, branch, side): reference current in A} of one AC sensitivity analysis (BRANCH_CURRENT_1/2)."""
    import pypowsybl as pp
    from pre_milp.nx_security_analysis import _register_contingencies

    analysis = pp.sensitivity.create_ac_analysis()
    _register_contingencies(analysis, contingencies)
    functions = pp.sensitivity.SensitivityFunctionType
    for side, function_type in ((1, functions.BRANCH_CURRENT_1), (2, functions.BRANCH_CURRENT_2)):
        if side_branches.get(side):
            analysis.add_factor_matrix(
                functions_ids=side_branches[side], variables_ids=[variable_id], contingencies_ids=[],
                contingency_context_type=pp.sensitivity.ContingencyContextType.ALL, sensitivity_function_type=function_type,
                sensitivity_variable_type=pp.sensitivity.SensitivityVariableType.INJECTION_ACTIVE_POWER, matrix_id=f"side{side}")
    result = analysis.run(network, parameters=parameters)
    references = {}
    for contingency in contingencies:
        for side in (1, 2):
            if not side_branches.get(side):
                continue
            try:
                matrix = result.get_reference_matrix(f"side{side}", contingency.mrid)
            except Exception:
                continue
            if matrix is not None:
                for branch in matrix.columns:
                    references[(contingency.mrid, str(branch), side)] = float(matrix.iloc[0][branch])
    return references


def _percent(values, function):
    return function(values) if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, default=None, help="folder with the cgm*.zip and ncp*.zip (default: input/)")
    parser.add_argument("--tolerance", type=float, default=None, help="relative tolerance (default: the guard's own, 0.01)")
    arguments = parser.parse_args()

    import run_pipeline
    from pre_milp import lf_parameters, nx_security_analysis, nx_sensitivity, sensitivity_checks
    from data_model import AEType

    if run_pipeline.CALCULATION_MODE == "dc":
        print("calculation_mode dc: this probe handles the AC modes only.")
        return 1
    logging.disable(logging.CRITICAL)                                               # the pipeline log carries ids
    captured: dict = {}
    real_lf_nx, real_sensitivities = run_pipeline.compute_lf_nx, run_pipeline.compute_sensitivities_nx

    def lf_nx_wrapper(network, co_flat_list, ae_flat_list, timestamp, *args, **kwargs):
        result = real_lf_nx(network, co_flat_list, ae_flat_list, timestamp, *args, **kwargs)
        captured.update(network=network, contingencies=co_flat_list, elements=ae_flat_list, timestamp=timestamp,
                        lf_initial=kwargs.get("lf_initial"))
        return result

    def sensitivities_wrapper(*args, **kwargs):
        captured["references"] = kwargs.setdefault("pst_reference_out", {})
        captured["rd_references"] = kwargs.setdefault("rd_reference_out", {})
        result = real_sensitivities(*args, **kwargs)
        captured["lf_nx"] = args[6] if len(args) > 6 else kwargs.get("lf_nx")
        raise _StopProbe

    run_pipeline.compute_lf_nx, run_pipeline.compute_sensitivities_nx = lf_nx_wrapper, sensitivities_wrapper
    if arguments.input is not None:
        run_pipeline.INPUT_DIR = arguments.input.resolve()
    run_pipeline.ENABLE_NX_ANALYSIS = True                                          # the probe is about N-X
    output_directory = tempfile.TemporaryDirectory(prefix="probe_", ignore_cleanup_errors=True)
    run_pipeline.OUTPUT_DIR = Path(output_directory.name)
    real_stdout = sys.stdout
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                run_pipeline.main()
            except _StopProbe:
                pass
    except BaseException as exc:                                                    # never the message: it may hold ids
        print(f"the pipeline stopped before the N-X sensitivity analysis ({type(exc).__name__})")
        return 1
    if "references" not in captured:
        print("the run did not reach the N-X sensitivity analysis (is N-X analysis enabled?)")
        return 1

    network, timestamp = captured["network"], captured["timestamp"]
    elements, contingencies = captured["elements"], captured["contingencies"]
    lf_nx = captured["lf_nx"]
    references = captured["references"] or captured["rd_references"]
    branch_sides = nx_security_analysis._get_monitored_branch_sides(elements)
    limits, _types = nx_sensitivity._get_limit_and_ae_type_by_branch_nx(elements, timestamp, captured["lf_initial"])
    tolerance = arguments.tolerance if arguments.tolerance is not None else sensitivity_checks.NX_RELATIVE_TOLERANCE
    by_mrid = {c.mrid: c for c in contingencies}

    flagged = []                                                                    # (contingency, branch, security A, reference A)
    for contingency_mrid, per_branch in lf_nx.items():
        for branch_mrid, per_side in per_branch.items():
            side = branch_sides.get(branch_mrid, 1)
            reference = (references.get(contingency_mrid) or {}).get(branch_mrid)
            current = (per_side or {}).get(side)
            if reference is None or current is None or math.isnan(reference) or math.isnan(current):
                continue
            if sensitivity_checks._deviation_reason(reference, current, tolerance) is not None:
                flagged.append((contingency_mrid, branch_mrid, current, reference))
    print(f"guard rule: relative gap |reference - security current| / max(|security current|, "
          f"{sensitivity_checks.RELATIVE_FLOOR_AMPERE:g} A) above {tolerance:g}; pairs flagged: {len(flagged)}")
    if not flagged:
        print("nothing flagged: no pair differs by more than the tolerance.")
        return 0

    # the security analysis of the flagged contingencies only: status, with the pipeline's parameters
    import pypowsybl as pp
    flagged_contingencies = [by_mrid[m] for m in sorted({f[0] for f in flagged})]
    security = pp.security.create_analysis()
    nx_security_analysis._register_contingencies(security, flagged_contingencies)
    security.add_monitored_elements(branch_ids=sorted(branch_sides))
    security_parameters = lf_parameters.build_security_analysis_parameters(
        run_pipeline.USE_REACTIVE_LIMITS, run_pipeline.EXTRA_LF_PARAMS)
    security_result = security.run_ac(network, parameters=security_parameters)
    status_of = {m: str(getattr(r.status, "name", r.status)) for m, r in security_result.post_contingency_results.items()}
    # the rerun of the flagged contingencies: the sensitivity analysis with exactly the security analysis's parameters
    side_branches = {side: sorted(b for b, s in branch_sides.items() if s == side) for side in (1, 2)}
    variable_id = str(network.get_generators().index[0])
    pipeline_parameters = lf_parameters.build_sensitivity_parameters(run_pipeline.USE_REACTIVE_LIMITS, run_pipeline.EXTRA_LF_PARAMS)
    equal_parameters = pp.sensitivity.Parameters(
        load_flow_parameters=security_parameters.load_flow_parameters, provider_parameters=dict(security_parameters.provider_parameters))
    again = _sensitivity_references(network, flagged_contingencies, side_branches, pipeline_parameters, variable_id)
    equal = _sensitivity_references(network, flagged_contingencies, side_branches, equal_parameters, variable_id)

    print(f"{'#':>4} {'contingency':<12} {'status':<20} {'branch':<14} {'security A':>11} {'reference A':>12} {'abs gap A':>10} "
          f"{'rel gap':>8} {'loading %':>9} {'ref. same params A':>18} {'gap same params A':>17} {'volt.ctrl trafo':>15} {'gen at Q limit':>14}")
    rows = []
    for number, (contingency_mrid, branch_mrid, current, reference) in enumerate(flagged, 1):
        side = branch_sides.get(branch_mrid, 1)
        absolute = abs(reference - current)
        relative = absolute / max(abs(current), sensitivity_checks.RELATIVE_FLOOR_AMPERE)
        limit = limits.get(branch_mrid)
        loading = 100.0 * abs(current) / limit if limit else float("nan")
        same = equal.get((contingency_mrid, branch_mrid, side), float("nan"))
        same_gap = abs(same - current)
        same_relative = same_gap / max(abs(current), sensitivity_checks.RELATIVE_FLOOR_AMPERE)
        controlling, at_limit = _substation_flags(network, _substations_of(network, branch_mrid))
        rows.append((absolute, relative, same_gap, same_relative))
        print(f"{number:>4} {_contingency_kind(network, by_mrid[contingency_mrid].equipment_mrids):<12} "
              f"{status_of.get(contingency_mrid, 'n/a'):<20} {_branch_kind(network, branch_mrid):<14} {current:>11.2f} {reference:>12.2f} "
              f"{absolute:>10.3f} {relative:>8.4f} {loading:>9.1f} {same:>18.2f} {same_gap:>17.3f} "
              f"{'yes' if controlling else 'no':>15} {'yes' if at_limit else 'no':>14}")

    absolutes, relatives = [r[0] for r in rows], [r[1] for r in rows]
    disappear = sum(1 for r in rows if not (r[3] > tolerance))
    print("summary")
    print(f"  pairs flagged: {len(rows)}")
    print(f"  absolute gap A: max {max(absolutes):.3f}, median {statistics.median(absolutes):.3f}")
    print(f"  relative gap:   max {max(relatives):.4f}, median {statistics.median(relatives):.4f}")
    print(f"  gap with the pipeline's sensitivity parameters on rerun: max "
          f"{max(abs(again.get((c, b, branch_sides.get(b, 1)), float('nan')) - i) for c, b, i, _r in flagged):.3f} A")
    print(f"  pairs that disappear (relative gap within the tolerance) with identical parameters: {disappear} of {len(rows)}")
    print(f"  after identical parameters: absolute gap max {max(r[2] for r in rows):.3f} A, relative gap max {max(r[3] for r in rows):.4f}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:
        print(f"probe failed ({type(exc).__name__})")
        sys.exit(2)
