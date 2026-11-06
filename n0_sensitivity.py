"""n0_sensitivity.py — N-0 PST (A per degree) and RD (A per MW) sensitivities from one
pypowsybl AC sensitivity analysis each, and the threshold pruning of both matrices.
"""

import csv
import logging
from pathlib import Path
from typing import Optional

from data_model import AEType, AssessedElement, PST, RAKind, RDUnit, select_by_side
from pre_milp import ct_sensitivity, log_summary
from pre_milp.lf_parameters import build_loadflow_parameters, build_sensitivity_parameters
from pre_milp.sensitivity_checks import find_invalid_n0_branches
from text_encoding import TEXT_ENCODING

log = logging.getLogger(__name__)


def _matrix_entry(matrix, variable_id: str, branch_mrid: str) -> float:
    """One sensitivity of a pypowsybl matrix (index = variables, columns = branches); NaN when
    the entry cannot be read, never 0.0 (the model then excludes the branch)."""
    try:
        value = matrix.loc[variable_id, branch_mrid]
    except (KeyError, IndexError):
        return float("nan")
    return float(value)


def _new_analysis(pp, dc_converter):
    """(analysis, function type side 1, function type side 2): the AC analysis with the branch
    current functions, or (dc_converter given) the DC analysis with the active-power functions, the
    only ones it supports."""
    functions = pp.sensitivity.SensitivityFunctionType
    if dc_converter is None:
        return pp.sensitivity.create_ac_analysis(), functions.BRANCH_CURRENT_1, functions.BRANCH_CURRENT_2
    return pp.sensitivity.create_dc_analysis(), functions.BRANCH_ACTIVE_POWER_1, functions.BRANCH_ACTIVE_POWER_2


def _dc_guard_references(p_ref_mw: dict, branch_sides: dict, dc_converter) -> dict[str, float]:
    """The DC reference flows (MW, signed) as reference currents (A) for the sensitivity guard."""
    return {b: dc_converter.current(b, branch_sides.get(b, 1), p) for b, p in p_ref_mw.items()}


def _sensitivity_entry(matrix, variable_id, branch_mrid, dc_converter, branch_sides, p_ref_mw) -> float:
    """One matrix entry in A per unit. DC: the MW entry converted with the sign of the reference
    flow of the branch side (dc_conversion); a branch without a reference is NaN."""
    value = _matrix_entry(matrix, variable_id, branch_mrid)
    if dc_converter is None:
        return value
    return dc_converter.sensitivity(
        branch_mrid, branch_sides.get(branch_mrid, 1), value, p_ref_mw.get(branch_mrid, float("nan")),
    )


def _warn_unreadable_entries(kind: str, sensitivities: dict[tuple[str, str], float], describe=None) -> None:
    """One warning per branch with an entry that could not be read (NaN). describe(branch) is the reason shown after
    the branch (what it is, its state, the side, what the analysis returned)."""
    unreadable: dict[str, list[str]] = {}
    for (branch_mrid, variable_id), value in sensitivities.items():
        if value != value:
            unreadable.setdefault(branch_mrid, []).append(variable_id)
    for branch_mrid in sorted(unreadable):
        ids = sorted(unreadable[branch_mrid])
        log.warning(
            "%s N-0 sensitivity of branch %s unreadable for %d variable(s) %s - NaN (the branch is excluded from the MILP)%s",
            kind, branch_mrid, len(ids), log_summary.first_ids(ids), f" [{describe(branch_mrid)}]" if describe else "",
        )
        if len(ids) > log_summary.SHOWN_IDS:
            log.debug("%s N-0 sensitivity of branch %s unreadable for all variables: %s", kind, branch_mrid, ", ".join(ids))


def _why_unreadable(network, branch_sides: dict, matrix_branch_ids: set, variable_ids) -> "callable":
    """The reason text for _warn_unreadable_entries: element type and connected state (cgmes_loader.describe_branch), the
    side used, whether the branch is itself a variable (a PST), and whether the analysis returned a column for it (NaN
    value) or none (missing column)."""
    from pre_milp import cgmes_loader

    def describe(branch_mrid: str) -> str:
        returned = "the analysis returned NaN" if branch_mrid in matrix_branch_ids else "the analysis returned no column for it"
        own = ", it is itself a variable (PST)" if branch_mrid in set(variable_ids) else ""
        return f"{cgmes_loader.describe_branch(network, branch_mrid)}, side {branch_sides.get(branch_mrid, 1)}{own}, {returned}"

    return describe


def _get_loading_filtered_branch_sides(
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_initial: Optional[dict[str, dict[int, float]]] = None,
    sensitivity_threshold_pct: float = 0.0,
) -> dict[str, int]:
    """{monitored branch mRID: resolved side (1 or 2)} for SECURED and SCANNED elements,
    optionally pruned by loading: loading_pct = I_initial / limit * 100 with limit = PATL
    (secured) or effective_limit = max(PATL, initial loading) (scanned; virtual margin 0).
    The side travels with the branch so BRANCH_CURRENT_1 or _2 is requested per branch.
    lf_initial None: no filtering. A branch without a current or PATL is kept (loading 0).
    """
    branch_sides: dict[str, int] = {}
    for ae in ae_flat_list:
        if ae.ae_type not in (AEType.SECURED, AEType.SCANNED) or not ae.branch_mrid:
            continue
        branch_mrid = str(ae.branch_mrid)
        side = int(getattr(ae, "branch_side", 1))

        if lf_initial is not None:
            patl_ka = ae.patl_ka.get(timestamp)
            per_side_currents = lf_initial.get(branch_mrid)
            i_a = select_by_side(per_side_currents, side, branch_mrid) if per_side_currents else None

            if ae.ae_type == AEType.SCANNED and patl_ka is not None and i_a is not None:
                # virtual_load_margin_pct defaults to 0 (see docstring) ->
                # patl_scanned_ka == patl_ka.
                limit_ka = max(patl_ka, float(i_a) / 1000.0)
            else:
                limit_ka = patl_ka

            if limit_ka is not None and limit_ka > 0.0 and i_a is not None:
                loading_pct = (float(i_a) / 1000.0) / limit_ka * 100.0
                if loading_pct < float(sensitivity_threshold_pct):
                    log.debug(
                        "Branch %s (side %d, ae_type=%s) loading_pct=%.2f below "
                        "sensitivity_threshold_pct=%.2f — excluded",
                        branch_mrid, side, ae.ae_type, loading_pct, sensitivity_threshold_pct,
                    )
                    continue

        branch_sides[branch_mrid] = side

    return branch_sides


def _get_preventive_pst_ids(pst_list: list[PST], network=None) -> list[str]:
    """Available preventive PST mRIDs. With `network`, PSTs on a 3-winding transformer are
    left out (logged): pypowsybl's TRANSFORMER_PHASE variable supports 2-winding
    transformers only ("Two windings transformer '<mrid>' not found").
    """
    candidates = [str(pst.mrid) for pst in pst_list if pst.available and pst.kind == RAKind.PREVENTIVE]

    if network is None:
        return candidates

    try:
        two_w_ids = set(network.get_2_windings_transformers().index)
    except Exception:
        log.exception("Failed to fetch 2-winding transformer IDs — skipping 3W-transformer PST filter")
        return candidates

    filtered = [pst_mrid for pst_mrid in candidates if pst_mrid in two_w_ids]
    excluded = [pst_mrid for pst_mrid in candidates if pst_mrid not in two_w_ids]
    if excluded:
        log.warning(
            "Excluding %d PST(s) backed by a 3-winding transformer from AC "
            "sensitivity (unsupported by pypowsybl's TRANSFORMER_PHASE "
            "variable type for 3W transformers): %s",
            len(excluded), excluded,
        )
    return filtered


def _get_preventive_rd_gu_ids(rd_list: list[RDUnit], with_countertrade: bool = False) -> list[str]:
    """Sorted, de-duplicated generating-unit mRIDs of the available preventive RD units
    (the sensitivity variable is the unit's injection, shared by its remedial actions). The units of the
    countertrade RAs have a variable per RA and direction instead (ct_sensitivity), unless with_countertrade."""
    return ct_sensitivity.individual_unit_ids(rd_list, preventive_only=True, with_countertrade=with_countertrade)


def _reference_currents_n0(result, matrix_ids) -> dict[str, float]:
    """{branch_mrid: reference (base case) current in A} as seen by the sensitivity analysis
    itself, for the guard in sensitivity_checks (compared with the N-0 load flow)."""
    references: dict[str, float] = {}
    for matrix_id in matrix_ids:
        try:
            reference_matrix = result.get_reference_matrix(matrix_id)
        except Exception:
            continue
        if reference_matrix is None:
            continue
        for branch_mrid in reference_matrix.columns:
            references[str(branch_mrid)] = float(reference_matrix.iloc[0][branch_mrid])
    return references


def compute_sensitivity_pst_n0(
    network,
    pst_list: list[PST],
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_initial: Optional[dict[str, dict[int, float]]] = None,
    sensitivity_threshold_pct: float = 0.0,
    output_path: Optional[Path] = None,
    use_reactive_limits: bool = False,
    extra_lf_params: Optional[dict] = None,
    reference_out: Optional[dict] = None,
    guard_out: Optional[dict] = None,
    dc_converter=None,
) -> dict[tuple[str, str], float]:
    """N-0 PST sensitivity matrix dI/ds, {(branch, pst): A per degree}, per branch on its
    resolved side (the two sides of a transformer differ roughly by the turns ratio).
    lf_initial: {branch: {1: A, 2: A}} for the loading filter (None: no filter).
    reference_out / guard_out: filled with the reference currents and the guard failures.
    dc_converter (calculation_mode dc): DC sensitivity analysis (MW per degree), converted to A per
    degree by dc_converter (dc_conversion); the reference flows become reference currents. None = AC.
    Runs with the pipeline's load-flow parameters (lf_parameters): pypowsybl's own defaults
    are a different load flow and can return all-zero sensitivities."""
    branch_sides = _get_loading_filtered_branch_sides(
        ae_flat_list, timestamp,
        lf_initial=lf_initial,
        sensitivity_threshold_pct=sensitivity_threshold_pct,
    )
    pst_ids = _get_preventive_pst_ids(pst_list, network=network)

    if not branch_sides:
        raise ValueError("No monitored branch IDs found in AEFlatList")
    if not pst_ids:
        raise ValueError("No available preventive PSTs found in PSTList")

    try:
        import pypowsybl as pp
    except Exception as exc:
        raise ImportError("pypowsybl is required to compute sensitivities") from exc

    side1_branch_ids = sorted(b for b, s in branch_sides.items() if s == 1)
    side2_branch_ids = sorted(b for b, s in branch_sides.items() if s == 2)

    sa, function_type_side1, function_type_side2 = _new_analysis(pp, dc_converter)
    # explicit TRANSFORMER_PHASE (AUTO_DETECT does not recognise PowerTransformer mRIDs); one
    # factor matrix per side (one function type per call), both solved by one sa.run
    if side1_branch_ids:
        sa.add_factor_matrix(
            functions_ids=side1_branch_ids,
            variables_ids=pst_ids,
            contingencies_ids=[],
            contingency_context_type=pp.sensitivity.ContingencyContextType.NONE,
            sensitivity_function_type=function_type_side1,
            sensitivity_variable_type=pp.sensitivity.SensitivityVariableType.TRANSFORMER_PHASE,
            matrix_id="psdf_n0_side1",
        )
    if side2_branch_ids:
        sa.add_factor_matrix(
            functions_ids=side2_branch_ids,
            variables_ids=pst_ids,
            contingencies_ids=[],
            contingency_context_type=pp.sensitivity.ContingencyContextType.NONE,
            sensitivity_function_type=function_type_side2,
            sensitivity_variable_type=pp.sensitivity.SensitivityVariableType.TRANSFORMER_PHASE,
            matrix_id="psdf_n0_side2",
        )

    result = sa.run(network, parameters=build_sensitivity_parameters(use_reactive_limits, extra_lf_params))
    p_ref_mw: dict[str, float] = {}
    if dc_converter is not None:
        p_ref_mw = _reference_currents_n0(result, ["psdf_n0_side1", "psdf_n0_side2"])
    if reference_out is not None or guard_out is not None:
        _references = _reference_currents_n0(result, ["psdf_n0_side1", "psdf_n0_side2"])
        if dc_converter is not None:
            _references = _dc_guard_references(_references, branch_sides, dc_converter)
        if reference_out is not None:
            reference_out.clear()
            reference_out.update(_references)
        if guard_out is not None:
            guard_out.clear()
            guard_out.update(find_invalid_n0_branches(_references, lf_initial or {}, branch_sides))

    # get_sensitivity_matrix(matrix_id): index = variables_ids (PSTs), columns = branches
    available_pst_ids: set[str] = set()
    available_branch_ids: set[str] = set()
    sensitivity_pst_n0: dict[tuple[str, str], float] = {}

    for matrix_id, expected_branch_ids in (
        ("psdf_n0_side1", side1_branch_ids),
        ("psdf_n0_side2", side2_branch_ids),
    ):
        if not expected_branch_ids:
            continue
        psdf_matrix = result.get_sensitivity_matrix(matrix_id)
        if psdf_matrix is None:
            log.warning("Sensitivity analysis returned no matrix for %s", matrix_id)
            continue

        matrix_pst_ids = [str(p) for p in getattr(psdf_matrix, "index", pst_ids)]
        matrix_branch_ids = [str(b) for b in getattr(psdf_matrix, "columns", expected_branch_ids)]
        available_pst_ids.update(matrix_pst_ids)
        available_branch_ids.update(matrix_branch_ids)

        for branch_mrid in matrix_branch_ids:
            for pst_mrid in matrix_pst_ids:
                sensitivity_pst_n0[(branch_mrid, pst_mrid)] = _sensitivity_entry(
                    psdf_matrix, pst_mrid, branch_mrid, dc_converter, branch_sides, p_ref_mw,
                )
    _warn_unreadable_entries("PST", sensitivity_pst_n0, _why_unreadable(network, branch_sides, available_branch_ids, pst_ids))

    missing_branch_ids = set(branch_sides.keys()) - available_branch_ids
    missing_pst_ids = set(pst_ids) - available_pst_ids
    if missing_branch_ids:
        log.warning("Sensitivity matrix missing monitored branches: %s", sorted(missing_branch_ids))
    if missing_pst_ids:
        log.warning("Sensitivity matrix missing preventive PSTs: %s", sorted(missing_pst_ids))

    if output_path is not None:
        write_sensitivity_csv(sensitivity_pst_n0, output_path, timestamp)

    return sensitivity_pst_n0


def _injection_sensitivity_n0(
    network,
    gu_ids: list[str],
    branch_sides: dict[str, int],
    lf_initial: Optional[dict[str, dict[int, float]]],
    use_reactive_limits: bool,
    extra_lf_params: Optional[dict],
    reference_out: Optional[dict],
    guard_out: Optional[dict],
    dc_converter,
    matrix_prefix: str,
    label: str,
    zones: Optional[dict[str, dict[str, float]]] = None,
) -> dict[tuple[str, str], float]:
    """The N-0 sensitivity of the monitored branches (branch_sides) to the active-power injection
    of the generators gu_ids, {(branch, generator): A per MW}: one factor matrix per branch side
    (variable INJECTION_ACTIVE_POWER, BRANCH_CURRENT_1/2, or the DC functions with dc_converter),
    one analysis. Shared by the RD units (matrix_prefix "ptdf_rd_n0", label "RD") and the HVDC units
    (matrix_prefix "ptdf_hvdc_n0", label "HVDC"); the label only appears in the log lines.
    zones ({variable id: {unit: weight}}, the countertrade RAs): each is one more variable, the weighted
    sum of its units' sensitivities, computed by pypowsybl as a zone; the units get no entry of their own.
    reference_out / guard_out as in compute_sensitivity_rd_n0."""
    try:
        import pypowsybl as pp
    except Exception as exc:
        raise ImportError("pypowsybl is required to compute sensitivities") from exc

    side1_branch_ids = sorted(b for b, s in branch_sides.items() if s == 1)
    side2_branch_ids = sorted(b for b, s in branch_sides.items() if s == 2)

    sa, function_type_side1, function_type_side2 = _new_analysis(pp, dc_converter)
    zone_ids = sorted(zones) if zones else []
    if zone_ids:
        sa.set_zones([
            pp.sensitivity.create_zone_from_injections_and_shift_keys(zone_id, list(zones[zone_id]), list(zones[zone_id].values()))
            for zone_id in zone_ids
        ])
    # {matrix id: (branches, variables, variable type, scale per variable)}; pypowsybl normalises the shift keys of a
    # zone to sum 1, the scale gives back their sum
    variable_types = pp.sensitivity.SensitivityVariableType
    matrices: dict[str, tuple[list, list, object, dict]] = {}
    for side, side_branch_ids, function_type in ((1, side1_branch_ids, function_type_side1), (2, side2_branch_ids, function_type_side2)):
        if not side_branch_ids:
            continue
        if gu_ids:
            matrices[f"{matrix_prefix}_side{side}"] = (side_branch_ids, gu_ids, variable_types.INJECTION_ACTIVE_POWER, {})
        if zone_ids:
            matrices[f"{matrix_prefix}_ctz_side{side}"] = (
                side_branch_ids, zone_ids, variable_types.AUTO_DETECT, {z: sum(zones[z].values()) for z in zone_ids},
            )
    # one factor matrix per side, as in compute_sensitivity_pst_n0
    for matrix_id, (side_branch_ids, variable_ids, variable_type, _scale) in matrices.items():
        sa.add_factor_matrix(
            functions_ids=side_branch_ids,
            variables_ids=variable_ids,
            contingencies_ids=[],
            contingency_context_type=pp.sensitivity.ContingencyContextType.NONE,
            sensitivity_function_type=function_type_side1 if matrix_id.endswith("_side1") else function_type_side2,
            sensitivity_variable_type=variable_type,
            matrix_id=matrix_id,
        )

    result = sa.run(network, parameters=build_sensitivity_parameters(use_reactive_limits, extra_lf_params))
    matrix_ids = list(matrices)
    p_ref_mw: dict[str, float] = {}
    if dc_converter is not None:
        p_ref_mw = _reference_currents_n0(result, matrix_ids)
    if reference_out is not None or guard_out is not None:
        _references = _reference_currents_n0(result, matrix_ids)
        if dc_converter is not None:
            _references = _dc_guard_references(_references, branch_sides, dc_converter)
        if reference_out is not None:
            reference_out.clear()
            reference_out.update(_references)
        if guard_out is not None:
            guard_out.clear()
            guard_out.update(find_invalid_n0_branches(_references, lf_initial or {}, branch_sides))

    # same orientation as PST: index = gu_mrids, columns = branch mrids
    available_gu_ids: set[str] = set()
    available_branch_ids: set[str] = set()
    sensitivity_n0: dict[tuple[str, str], float] = {}

    for matrix_id, (expected_branch_ids, expected_variable_ids, _type, scale) in matrices.items():
        ptdf_matrix = result.get_sensitivity_matrix(matrix_id)
        if ptdf_matrix is None:
            log.warning("Sensitivity analysis returned no matrix for %s", matrix_id)
            continue

        matrix_gu_ids = [str(g) for g in getattr(ptdf_matrix, "index", expected_variable_ids)]
        matrix_branch_ids = [str(b) for b in getattr(ptdf_matrix, "columns", expected_branch_ids)]
        available_gu_ids.update(matrix_gu_ids)
        available_branch_ids.update(matrix_branch_ids)

        for branch_mrid in matrix_branch_ids:
            for gu_mrid in matrix_gu_ids:
                sensitivity_n0[(branch_mrid, gu_mrid)] = scale.get(gu_mrid, 1.0) * _sensitivity_entry(
                    ptdf_matrix, gu_mrid, branch_mrid, dc_converter, branch_sides, p_ref_mw,
                )
    _warn_unreadable_entries(label, sensitivity_n0, _why_unreadable(network, branch_sides, available_branch_ids, gu_ids))

    missing_branch_ids = set(branch_sides.keys()) - available_branch_ids
    missing_gu_ids = set(gu_ids) - available_gu_ids
    if missing_branch_ids:
        log.warning("%s sensitivity matrix missing monitored branches: %s", label, sorted(missing_branch_ids))
    if missing_gu_ids:
        log.warning(
            "%s sensitivity matrix missing preventive %s generating units: %s", label, label, sorted(missing_gu_ids),
        )
    return sensitivity_n0


def compute_sensitivity_rd_n0(
    network,
    rd_list: list[RDUnit],
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_initial: Optional[dict[str, dict[int, float]]] = None,
    sensitivity_threshold_pct: float = 0.0,
    output_path: Optional[Path] = None,
    use_reactive_limits: bool = False,
    extra_lf_params: Optional[dict] = None,
    reference_out: Optional[dict] = None,
    guard_out: Optional[dict] = None,
    dc_converter=None,
) -> dict[tuple[str, str], float]:
    """N-0 RD sensitivity matrix dI/dP, {(branch, gu_mrid): A per MW}; same structure as
    compute_sensitivity_pst_n0. Variable INJECTION_ACTIVE_POWER (increase in MW) with
    BRANCH_CURRENT_1/2 (A), so the value is already A/MW and needs no conversion.
    Variables are de-duplicated generating units (RDUnit.gu_mrid).
    dc_converter (calculation_mode dc): DC analysis with BRANCH_ACTIVE_POWER_1/2 (MW per MW),
    converted to A per MW by dc_converter. None = AC."""
    branch_sides = _get_loading_filtered_branch_sides(
        ae_flat_list, timestamp,
        lf_initial=lf_initial,
        sensitivity_threshold_pct=sensitivity_threshold_pct,
    )
    # the magnitude pruning (sensitivity_threshold_pct) is defined per entry: with it the countertrade units keep
    # their own sensitivities, so the pruned values are those of the units (DECISIONS 169)
    aggregate_countertrade = float(sensitivity_threshold_pct) == 0.0
    gu_ids = _get_preventive_rd_gu_ids(rd_list, with_countertrade=not aggregate_countertrade)
    zones = ct_sensitivity.ct_zones(rd_list, preventive_only=True) if aggregate_countertrade else {}

    if not branch_sides:
        raise ValueError("No monitored branch IDs found in AEFlatList")
    if not gu_ids and not zones:
        raise ValueError("No available preventive RD units found in RDList")

    sensitivity_rd_n0 = _injection_sensitivity_n0(
        network, gu_ids, branch_sides, lf_initial, use_reactive_limits, extra_lf_params, reference_out, guard_out,
        dc_converter, matrix_prefix="ptdf_rd_n0", label="RD", zones=zones,
    )

    if output_path is not None:
        write_sensitivity_rd_csv(sensitivity_rd_n0, branch_sides, output_path, timestamp)

    return sensitivity_rd_n0


def compute_sensitivity_hvdc_n0(
    network,
    ei_mrids: list[str],
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_initial: Optional[dict[str, dict[int, float]]] = None,
    sensitivity_threshold_pct: float = 0.0,
    use_reactive_limits: bool = False,
    extra_lf_params: Optional[dict] = None,
    reference_out: Optional[dict] = None,
    guard_out: Optional[dict] = None,
    dc_converter=None,
) -> dict[tuple[str, str], float]:
    """N-0 sensitivity of the monitored branches to the HVDC units (docs/specs/hvdc_ra_specs.md H8.1):
    {(branch, EI mRID): A per MW of injection}, the same variable type as RD (the EI is a
    generator of the network). ei_mrids: the EIs of the available preventive HVDC units.
    Silent-zero guard (H8.2, HVDC units only): pypowsybl returns exactly 0 for a unit outside the
    solved network, without an error; a unit whose sensitivities to every monitored branch are
    exactly 0 is a ValueError naming the unit (checked before any pruning).
    dc_converter, reference_out, guard_out as in compute_sensitivity_rd_n0."""
    branch_sides = _get_loading_filtered_branch_sides(
        ae_flat_list, timestamp, lf_initial=lf_initial, sensitivity_threshold_pct=sensitivity_threshold_pct,
    )
    ids = sorted({str(e) for e in ei_mrids})
    if not branch_sides:
        raise ValueError("No monitored branch IDs found in AEFlatList")
    if not ids:
        raise ValueError("No available preventive HVDC units found")

    sensitivity = _injection_sensitivity_n0(
        network, ids, branch_sides, lf_initial, use_reactive_limits, extra_lf_params, reference_out, guard_out,
        dc_converter, matrix_prefix="ptdf_hvdc_n0", label="HVDC",
    )
    for ei_mrid in ids:
        values = [v for (_branch, unit), v in sensitivity.items() if unit == ei_mrid and v == v]
        if values and all(v == 0.0 for v in values):
            raise ValueError(
                f"HVDC unit with EquivalentInjection {ei_mrid}: its N-0 sensitivities to all {len(values)} monitored "
                f"branches are exactly zero - the unit is outside the solved network or electrically isolated"
            )
    return sensitivity



def _resolve_branch_patl_ampere_and_initial_current_ampere(
    branch_mrid: str,
    ae_by_branch_mrid: dict[str, AssessedElement],
    timestamp: str,
    lf_initial: Optional[dict[str, dict[int, float]]],
) -> tuple[Optional[float], Optional[float]]:
    """(limit_ampere, initial_current_ampere) of one branch on its resolved side; the limit is
    PATL (secured) or effective_limit = max(PATL, initial loading) (scanned), as in
    _get_loading_filtered_branch_sides. A value that cannot be resolved is None; callers keep
    such a branch (never dropped for missing data)."""
    assessed_element = ae_by_branch_mrid.get(branch_mrid)
    if assessed_element is None:
        return None, None

    resolved_side = int(getattr(assessed_element, "branch_side", 1))

    patl_kiloampere = assessed_element.patl_ka.get(timestamp)

    initial_current_ampere = None
    if lf_initial is not None:
        per_side_currents = lf_initial.get(branch_mrid)
        if per_side_currents:
            initial_current_ampere = select_by_side(per_side_currents, resolved_side, branch_mrid)

    if (
        assessed_element.ae_type == AEType.SCANNED
        and patl_kiloampere is not None
        and initial_current_ampere is not None
    ):
        limit_kiloampere = max(patl_kiloampere, float(initial_current_ampere) / 1000.0)
    else:
        limit_kiloampere = patl_kiloampere

    limit_ampere = float(limit_kiloampere) * 1000.0 if limit_kiloampere is not None else None

    return limit_ampere, initial_current_ampere


def prune_sensitivity_by_threshold(
    sensitivity_matrix: dict[tuple[str, str], float],
    ae_flat_list: list[AssessedElement],
    timestamp: str,
    lf_initial: Optional[dict[str, dict[int, float]]] = None,
    sensitivity_threshold_pct: float = 0.0,
    min_loading_threshold_pct: float = 0.0,
) -> dict[tuple[str, str], float]:
    """Prune one sensitivity matrix (called separately for PST and RD, so a branch can survive
    in one and not the other). A branch is dropped if EITHER:
    1. initial_current / PATL * 100 < min_loading_threshold_pct, or
    2. max |sensitivity| over this matrix's RAs < sensitivity_threshold_pct * PATL / 100.
    Both default 0.0 (no-op). A branch that cannot be evaluated is kept and logged.
    """
    ae_by_branch_mrid: dict[str, AssessedElement] = {
        str(ae.branch_mrid): ae for ae in ae_flat_list if ae.branch_mrid
    }

    branch_mrids_in_matrix = sorted({branch_mrid for (branch_mrid, _ra_mrid) in sensitivity_matrix.keys()})

    branches_excluded_by_loading: set[str] = set()
    branches_excluded_by_sensitivity: set[str] = set()
    branches_not_evaluable_for_loading: set[str] = set()
    branches_not_evaluable_for_sensitivity: set[str] = set()

    for branch_mrid in branch_mrids_in_matrix:
        limit_ampere, initial_current_ampere = _resolve_branch_patl_ampere_and_initial_current_ampere(
            branch_mrid, ae_by_branch_mrid, timestamp, lf_initial,
        )
        # limit_ampere is PATL for AEType.SECURED branches, effective_limit
        # for AEType.SCANNED branches — see the resolver's own docstring.

        # --- Criterion 1: min_loading_threshold_pct ---
        if limit_ampere is not None and limit_ampere > 0.0 and initial_current_ampere is not None:
            loading_pct = (initial_current_ampere / limit_ampere) * 100.0
            if loading_pct < float(min_loading_threshold_pct):
                branches_excluded_by_loading.add(branch_mrid)
        else:
            branches_not_evaluable_for_loading.add(branch_mrid)

        # --- Criterion 2: sensitivity_threshold_pct ---
        sensitivities_for_branch = [
            abs(value) for (b, _ra_mrid), value in sensitivity_matrix.items() if b == branch_mrid
        ]
        max_absolute_sensitivity = max(sensitivities_for_branch) if sensitivities_for_branch else 0.0

        if limit_ampere is not None and limit_ampere > 0.0:
            sensitivity_threshold_ampere = float(sensitivity_threshold_pct) * limit_ampere / 100.0
            if max_absolute_sensitivity < sensitivity_threshold_ampere:
                branches_excluded_by_sensitivity.add(branch_mrid)
        else:
            branches_not_evaluable_for_sensitivity.add(branch_mrid)

    if branches_not_evaluable_for_loading:
        log.warning(
            "prune_sensitivity_by_threshold: %d branch(es) could not be evaluated "
            "for min_loading_threshold_pct (missing PATL or initial current) — "
            "conservatively kept: %s",
            len(branches_not_evaluable_for_loading), sorted(branches_not_evaluable_for_loading),
        )
    if branches_not_evaluable_for_sensitivity:
        log.warning(
            "prune_sensitivity_by_threshold: %d branch(es) could not be evaluated "
            "for sensitivity_threshold_pct (missing PATL) — conservatively kept: %s",
            len(branches_not_evaluable_for_sensitivity), sorted(branches_not_evaluable_for_sensitivity),
        )

    branches_excluded = branches_excluded_by_loading | branches_excluded_by_sensitivity
    branches_remaining = set(branch_mrids_in_matrix) - branches_excluded

    pruned_sensitivity_matrix = {
        (branch_mrid, ra_mrid): value
        for (branch_mrid, ra_mrid), value in sensitivity_matrix.items()
        if branch_mrid in branches_remaining
    }

    log.info(
        "prune_sensitivity_by_threshold: %d branches in -> %d branches out "
        "(excluded_by_min_loading_threshold_pct=%d, excluded_by_sensitivity_threshold_pct=%d, "
        "overlap=%d) | %d (branch, ra) pairs in -> %d pairs out",
        len(branch_mrids_in_matrix), len(branches_remaining),
        len(branches_excluded_by_loading), len(branches_excluded_by_sensitivity),
        len(branches_excluded_by_loading & branches_excluded_by_sensitivity),
        len(sensitivity_matrix), len(pruned_sensitivity_matrix),
    )

    return pruned_sensitivity_matrix


def write_sensitivity_csv(
    sensitivity_pst_n0: dict[tuple[str, str], float],
    output_path: Path,
    timestamp: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(["branch_mrid", "pst_mrid", "di_ds_ampere_per_tap", "timestamp"])
        for (branch_mrid, pst_mrid), sensitivity in sorted(sensitivity_pst_n0.items()):
            writer.writerow([branch_mrid, pst_mrid, sensitivity, timestamp])

    log.info("Written sensitivity CSV: %s (%d rows)", output_path, len(sensitivity_pst_n0))


def write_sensitivity_rd_csv(
    sensitivity_rd_n0: dict[tuple[str, str], float],
    branch_sides: dict[str, int],
    output_path: Path,
    timestamp: str,
) -> None:
    """sensitivity_rd_n0.csv with a side column (the side the sensitivity was computed on,
    from branch_sides)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding=TEXT_ENCODING) as f:
        writer = csv.writer(f)
        writer.writerow(["branch_mrid", "gu_mrid", "di_da_ampere_per_mw", "side", "timestamp"])
        for (branch_mrid, gu_mrid), sensitivity in sorted(sensitivity_rd_n0.items()):
            side = branch_sides.get(branch_mrid, 1)
            writer.writerow([branch_mrid, gu_mrid, sensitivity, side, timestamp])

    log.info("Written RD sensitivity CSV: %s (%d rows)", output_path, len(sensitivity_rd_n0))
